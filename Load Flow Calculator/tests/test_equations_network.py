"""Network stamps, independent finite differences, and safe input validation."""

import numpy as np
import pytest

from loadflow_app.equations import calculated_injections, equation_indices, jacobian, mismatch
from loadflow_app.models import BranchData, BusData, NetworkCase, SolverConfig
from loadflow_app.network import build_ybus, initial_voltages, parse_ybus, unknown_labels, validate_case


def test_example_stamps_and_bus_order(three_bus, expected_ybus):
    np.testing.assert_allclose(build_ybus(three_bus.buses, three_bus.branches), expected_ybus)
    order = [2, 0, 1]
    three_bus.buses = [three_bus.buses[i] for i in order]
    np.testing.assert_allclose(validate_case(three_bus), expected_ybus[np.ix_(order, order)])


def test_parallel_lines_charging_and_bus_shunts():
    buses = [BusData(id="B", shunt_g=0.07, shunt_b=0.11), BusData(id="A", shunt_b=-0.04)]
    branches = [BranchData("x", "A", "B", 0.1, 0.2, 0.06),
                BranchData("y", "B", "A", 0.0, 0.25, 0.10)]
    series = 1 / complex(0.1, 0.2) + 1 / complex(0, 0.25)
    expected = np.array([[series + 0.07 + 0.19j, -series], [-series, series + 0.04j]])
    np.testing.assert_allclose(build_ybus(buses, branches), expected, atol=1e-14)


def test_mixed_unknown_and_residual_order_with_middle_slack():
    buses = [BusData("gen", type="PV", p=0.7, v_set=1.03),
             BusData("ref", type="Slack", v_set=1.02, angle_set=7),
             BusData("load-z", type="PQ", p=-0.4, q=-0.2),
             BusData("load-a", type="PQ", p=-0.3, q=-0.1)]
    types = [b.type for b in buses]
    non_slack, pq = equation_indices(types)
    assert list(non_slack) == [0, 2, 3]
    assert list(pq) == [2, 3]
    labels = unknown_labels(buses)
    assert len(labels) == 5
    assert all(bus_id in str(label) for bus_id, label in zip(
        ["gen", "load-z", "load-a", "load-z", "load-a"], labels))
    r = mismatch(buses, types, [None, None, -0.2, -0.1],
                 np.array([0.1, 0.5, -0.2, -0.1]), np.array([0.2, 0.9, -0.07, -0.08]))
    np.testing.assert_allclose(r, [0.6, -0.2, -0.2, -0.13, -0.02])


@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_jacobian_matches_independent_central_differences(nonsymmetric):
    # Conductance, nonzero angles, off-nominal magnitudes, and a middle slack
    # exercise every diagonal and off-diagonal block of the polar Jacobian.
    y = np.array([[3.4-10j, -1+3j, -0.9+2j, -1.5+5j],
                  [-1+3j, 3.5-9j, -1.1+4j, -1.4+2j],
                  [-0.9+2j, -1.1+4j, 2.8-10j, -0.8+4j],
                  [-1.5+5j, -1.4+2j, -0.8+4j, 3.7-11j]])
    if nonsymmetric:
        y[0, 2] *= np.exp(0.13j)
        y[2, 0] *= np.exp(-0.13j)
        assert not np.allclose(y, y.T)
    types = ["PV", "Slack", "PQ", "PQ"]
    magnitudes = np.array([1.03, 1.01, 0.94, 0.98])
    angles = np.array([0.13, 0.04, -0.15, -0.06])
    angle_indices = [0, 2, 3]
    magnitude_indices = [2, 3]
    x = np.r_[angles[angle_indices], magnitudes[magnitude_indices]]

    def power_equations(x_values):
        vm, va = magnitudes.copy(), angles.copy()
        va[angle_indices] = x_values[:3]
        vm[magnitude_indices] = x_values[3:]
        v = vm * np.exp(1j * va)
        # Independent complex-power evaluation, not the implementation helper.
        power = v * np.conj(y @ v)
        return np.r_[power.real[angle_indices], power.imag[magnitude_indices]]

    h = 1e-6
    numeric = np.column_stack([
        (power_equations(x + h * np.eye(len(x))[k]) -
         power_equations(x - h * np.eye(len(x))[k])) / (2 * h)
        for k in range(len(x))
    ])
    analytic = jacobian(y, magnitudes * np.exp(1j * angles), types)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-8, atol=2e-8)


def test_complex_power_uses_conjugate_current():
    y = np.array([[2-4j, -1+2j], [-0.8+1.6j, 2.1-4.2j]])
    v = np.array([1.02+0.1j, 0.93-0.12j])
    expected = np.array([v[i] * sum(y[i, j] * v[j] for j in range(2)).conjugate()
                         for i in range(2)])
    p, q = calculated_injections(y, v)
    np.testing.assert_allclose(p + 1j * q, expected)


@pytest.mark.parametrize("text", ["0-12.5j, 0+10j\n0+10j, -15j", "-12.5j 10j\n10j -15j"])
def test_safe_matrix_parser(text):
    np.testing.assert_allclose(parse_ybus(text), [[-12.5j, 10j], [10j, -15j]])


@pytest.mark.parametrize("text", ["", "1 2 3\n4 5 6", "1 2\n3", "nan 1\n1 1",
                                   "inf 1\n1 1", "__import__('os').getcwd() 1\n1 1"])
def test_matrix_parser_rejects_invalid_or_executable_input(text):
    with pytest.raises(ValueError):
        parse_ybus(text)


def test_direct_nonsymmetric_nonzero_row_sum_singular_full_ybus_allowed(three_bus, expected_ybus):
    three_bus.mode = "direct"
    three_bus.ybus = expected_ybus.copy()
    # Ordinary ungrounded Ybus is singular; this is valid for load flow.
    assert np.linalg.matrix_rank(three_bus.ybus) == 2
    validate_case(three_bus)
    three_bus.ybus[0, 1] *= 0.8
    three_bus.ybus[2, 2] += 0.1j
    validate_case(three_bus)


@pytest.mark.parametrize("field,value", [("p", float("nan")), ("q", float("inf")),
                                           ("v_guess", 0), ("v_guess", -1),
                                           ("angle_guess", float("nan")),
                                           ("p", 10**500), ("v_guess", 10**500),
                                           ("angle_guess", 10**500)])
def test_invalid_pq_inputs_identify_bus(three_bus, field, value):
    setattr(three_bus.buses[1], field, value)
    with pytest.raises(ValueError, match="2"):
        validate_case(three_bus)


@pytest.mark.parametrize("field,value", [("tolerance", 0), ("tolerance", float("nan")),
                                           ("max_iterations", 0), ("max_iterations", 2.5),
                                           ("max_iterations", True), ("method", "Unknown")])
def test_invalid_solver_settings(three_bus, field, value):
    setattr(three_bus.config, field, value)
    with pytest.raises(ValueError):
        validate_case(three_bus)


@pytest.mark.parametrize("fault", ["duplicate", "no_slack", "two_slacks", "bad_type", "one_bus",
                                    "disconnected", "dimensions", "nonfinite", "base"])
def test_invalid_networks(three_bus, expected_ybus, fault):
    if fault == "duplicate":
        three_bus.buses[1].id = "1"
    elif fault == "no_slack":
        three_bus.buses[0] = BusData("1", type="PQ", p=1, q=0)
    elif fault == "two_slacks":
        three_bus.buses[1] = BusData("2", type="Slack", v_set=1, angle_set=0)
    elif fault == "bad_type":
        three_bus.buses[1].type = "load"
    elif fault == "one_bus":
        three_bus.buses = three_bus.buses[:1]
        three_bus.branches = []
    elif fault == "disconnected":
        three_bus.branches = three_bus.branches[:1]
    elif fault in {"dimensions", "nonfinite"}:
        three_bus.mode = "direct"
        three_bus.ybus = expected_ybus.copy()
        if fault == "dimensions":
            three_bus.ybus = three_bus.ybus[:2, :2]
        else:
            three_bus.ybus[1, 1] = np.inf
    elif fault == "base":
        three_bus.base_mva = -100
    with pytest.raises(ValueError):
        validate_case(three_bus)


def test_direct_graph_uses_either_direction_and_rejects_disconnection(three_bus, expected_ybus):
    three_bus.mode = "direct"
    three_bus.ybus = np.triu(expected_ybus)
    validate_case(three_bus)
    three_bus.ybus[0, 2] = three_bus.ybus[1, 2] = 0
    with pytest.raises(ValueError, match="[Cc]onnect"):
        validate_case(three_bus)


@pytest.mark.parametrize("fault", ["endpoint", "self", "zero", "nonfinite", "duplicate"])
def test_invalid_branch_inputs(three_bus, fault):
    branch = three_bus.branches[0]
    if fault == "endpoint":
        branch.from_bus = "missing"
    elif fault == "self":
        branch.to_bus = branch.from_bus
    elif fault == "zero":
        branch.r = branch.x = 0
    elif fault == "nonfinite":
        branch.b = np.nan
    elif fault == "duplicate":
        three_bus.branches[1].id = branch.id
    with pytest.raises(ValueError):
        validate_case(three_bus)


@pytest.mark.parametrize("field,value", [("q_min", None), ("q_min", 6),
                                           ("q_max", np.inf), ("q_load", np.nan)])
def test_invalid_enabled_q_limits(three_bus, field, value):
    three_bus.config.enforce_q_limits = True
    setattr(three_bus.buses[2], field, value)
    with pytest.raises(ValueError, match="3"):
        validate_case(three_bus)


def test_specified_voltages_ignore_irrelevant_guesses(three_bus):
    three_bus.buses[0].v_guess = 0.4
    three_bus.buses[0].angle_guess = 20
    three_bus.buses[0].angle_set = 5
    three_bus.buses[2].v_guess = 0.6
    v = initial_voltages(three_bus.buses)
    assert abs(v[0]) == pytest.approx(1)
    assert np.angle(v[0], deg=True) == pytest.approx(5)
    assert abs(v[2]) == pytest.approx(1)


def test_large_representable_integer_angles_and_limits_are_numeric(three_bus):
    # Python's arbitrary-size int must be converted to floating-point before
    # NumPy trigonometry; otherwise an object array can leak a TypeError.
    from loadflow_app.run_controller import RunController

    three_bus.buses[0].angle_set = 10**100
    three_bus.buses[1].angle_guess = 10**100
    three_bus.buses[2].q_min = -(10**100)
    three_bus.buses[2].q_max = 10**100
    three_bus.config.enforce_q_limits = True
    validate_case(three_bus)
    run = RunController(three_bus)
    assert np.all(np.isfinite(run.current.voltages))
    assert np.all(np.isfinite(run.current.angles))
