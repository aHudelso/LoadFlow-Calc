"""Numerical acceptance, reactive limits, iteration accounting, and failure state."""

import copy

import numpy as np
import pytest

from loadflow_app.equations import calculated_injections
from loadflow_app.models import BranchData, BusData, NetworkCase, SolverConfig
from loadflow_app.network import initial_voltages, validate_case
from loadflow_app.run_controller import RunController


METHODS = ["Newton-Raphson", "Gauss-Seidel"]


def test_iteration_zero_and_first_newton_update(three_bus):
    run = RunController(three_bus)
    initial = run.current
    assert initial.iteration == 0
    assert initial.phase == "initial"
    np.testing.assert_allclose(initial.voltages, np.ones(3))
    np.testing.assert_allclose(initial.p_calculated, 0, atol=1e-14)
    np.testing.assert_allclose(initial.q_calculated, 0, atol=1e-14)
    np.testing.assert_allclose(initial.residual, [-2, 1, -0.5])
    assert initial.max_mismatch == pytest.approx(2)
    assert initial.delta_p[0] is None
    assert initial.delta_q[0] is None and initial.delta_q[2] is None
    run.step()
    current = run.current
    assert current.iteration == 1
    np.testing.assert_allclose(abs(current.voltages), [1, 0.9666666666667, 1], atol=1e-12)
    np.testing.assert_allclose(current.angles, [0, -0.1142857142857, 0.05714285714286], atol=1e-12)
    np.testing.assert_allclose(current.delta_magnitude, [0, -1/30, 0], atol=1e-12)
    np.testing.assert_allclose(current.delta_angle, current.angles, atol=1e-12)
    p, q = calculated_injections(run.ybus, current.voltages)
    np.testing.assert_allclose(current.p_calculated, p)
    np.testing.assert_allclose(current.q_calculated, q)


def test_first_gauss_seidel_sweep(three_bus):
    three_bus.config.method = "Gauss-Seidel"
    run = RunController(three_bus)
    run.step()
    current = run.current
    np.testing.assert_allclose(current.voltages,
        [1+0j, 0.966666666666667-0.133333333333333j, 0.999013806466+0.044400613621j], atol=1e-12)
    assert current.max_mismatch == pytest.approx(0.317700344494, abs=2e-12)
    assert current.q_calculated[2] == pytest.approx(0.203499161662, abs=2e-12)
    # The detail container is tested in CSV integration as well: it must retain
    # temporary Q=1/6 separately from the final calculated Q above.
    assert current.details


@pytest.mark.parametrize("method", METHODS)
def test_both_methods_reproduce_reference_and_preserve_constraints(three_bus, method):
    three_bus.config.method = method
    run = RunController(three_bus)
    run.run_until_converged()
    assert run.status == "Converged"
    current = run.current
    np.testing.assert_allclose(abs(current.voltages), [1, 0.954905106202, 1], atol=1e-9)
    np.testing.assert_allclose(np.rad2deg(current.angles), [0, -6.902024491409, 3.383052860574], atol=1e-7)
    np.testing.assert_allclose(current.p_calculated, [1, -2, 1], atol=1e-9)
    np.testing.assert_allclose(current.q_calculated, [0.524506675355, -0.5, 0.306550266404], atol=1e-9)
    assert abs(sum(current.p_calculated)) < 1e-12  # This network alone is lossless.
    for snapshot in run.history.snapshots:
        assert snapshot.voltages[0] == 1+0j
        assert abs(snapshot.voltages[2]) == pytest.approx(1, abs=5e-15)
        p, q = calculated_injections(run.ybus, snapshot.voltages)
        np.testing.assert_allclose(snapshot.p_calculated, p, atol=1e-14)
        np.testing.assert_allclose(snapshot.q_calculated, q, atol=1e-14)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("q_load,q_max", [(0, 0.2), (0.1, 0.3)])
def test_binding_limit_reference_and_generator_net_distinction(three_bus, method, q_load, q_max):
    three_bus.config.method = method
    three_bus.config.enforce_q_limits = True
    three_bus.buses[2].q_load = q_load
    three_bus.buses[2].q_max = q_max
    run = RunController(three_bus)
    assert run.state.effective_types[2] == "PV"  # No switch on a poor initial guess.
    run.run_until_converged()
    assert run.status == "Converged"
    assert run.case.buses[2].type == "PV"
    assert run.case.buses[2].v_set == 1.0
    assert run.state.effective_types[2] == "PQ"
    assert run.current.q_targets[2] == pytest.approx(0.2)
    assert run.current.q_calculated[2] + q_load == pytest.approx(q_max, abs=2e-10)
    np.testing.assert_allclose(abs(run.current.voltages)[1:], [0.948234781851, 0.981422768612], atol=1e-8)
    np.testing.assert_allclose(np.rad2deg(run.current.angles)[1:], [-6.973788902110, 3.535463037165], atol=1e-7)
    snapshots = run.history.snapshots
    changes = [(i, s) for i, s in enumerate(snapshots) if s.phase == "constraint_change"]
    assert len(changes) == 1
    i, change = changes[0]
    before = snapshots[i-1]
    assert before.iteration == change.iteration
    assert before.snapshot_id < change.snapshot_id
    assert before.effective_types[2] == "PV" and change.effective_types[2] == "PQ"
    assert before.max_mismatch <= run.case.config.tolerance
    assert before.delta_q[2] is None and change.delta_q[2] is not None
    assert len(before.residual) == 3 and len(change.residual) == 4
    assert change.max_mismatch > run.case.config.tolerance
    np.testing.assert_array_equal(before.voltages, change.voltages)
    assert run.history.events
    assert len(set(s.snapshot_id for s in snapshots)) == len(snapshots)


@pytest.mark.parametrize("method", METHODS)
def test_disabled_limits_and_generous_limits_do_not_change_types(three_bus, method):
    three_bus.config.method = method
    three_bus.buses[2].q_max = 0.2
    run = RunController(three_bus)
    run.run_until_converged()
    assert run.current.effective_types[2] == "PV"
    assert run.current.q_calculated[2] > 0.2
    three_bus.config.enforce_q_limits = True
    three_bus.buses[2].q_max = 5
    generous = RunController(three_bus)
    generous.run_until_converged()
    assert generous.status == "Converged"
    assert generous.current.effective_types[2] == "PV"
    np.testing.assert_allclose(run.current.voltages, generous.current.voltages, atol=1e-12)


def test_q_limit_checked_at_iteration_zero_and_reset_restores_pv(three_bus):
    # Start from a constructed exact operating point with insufficient Q.
    three_bus.buses[1].v_guess = 0.95
    three_bus.buses[1].angle_guess = -6
    three_bus.buses[2].angle_guess = 3
    p, q = calculated_injections(validate_case(three_bus), initial_voltages(three_bus.buses))
    three_bus.buses[1].p = p[1]
    three_bus.buses[1].q = q[1]
    three_bus.buses[2].p = p[2]
    three_bus.buses[2].q_max = float(q[2] - 0.1)
    three_bus.config.enforce_q_limits = True
    run = RunController(three_bus)
    assert run.state.iteration == 0
    assert run.current.phase == "constraint_change"
    assert run.current.effective_types[2] == "PQ"
    assert run.history.snapshots[0].phase == "initial"
    assert run.history.snapshots[0].effective_types[2] == "PV"
    assert run.history.snapshots[0].iteration == run.current.iteration == 0
    assert run.current.max_mismatch == pytest.approx(0.1)


def test_q_minimum_is_supported(three_bus):
    three_bus.config.enforce_q_limits = True
    three_bus.buses[2].q_min = 0.4
    run = RunController(three_bus)
    run.run_until_converged()
    assert run.status == "Converged"
    assert run.current.effective_types[2] == "PQ"
    assert run.current.q_calculated[2] == pytest.approx(0.4, abs=2e-10)
    assert abs(run.current.voltages[2]) > 1.0


def test_snapshots_and_original_inputs_are_copies(three_bus):
    run = RunController(three_bus)
    initial = run.history.snapshots[0]
    saved = copy.deepcopy(initial)
    three_bus.buses[1].p = -99
    run.run_n(2)
    assert run.case.buses[1].p == -2
    np.testing.assert_array_equal(initial.voltages, saved.voltages)
    np.testing.assert_array_equal(initial.residual, saved.residual)
    np.testing.assert_array_equal(initial.p_calculated, saved.p_calculated)
    assert initial.effective_types == saved.effective_types
    assert initial.iteration == 0
    assert not np.shares_memory(initial.voltages, run.state.voltages)


@pytest.mark.parametrize("method", METHODS)
def test_manual_run_n_and_until_share_engine(three_bus, method):
    three_bus.config.method = method
    manual, batch = RunController(three_bus), RunController(three_bus)
    for _ in range(3):
        manual.step()
    batch.run_n(2)
    batch.run_n(1)  # N means additional steps.
    assert manual.state.iteration == batch.state.iteration == 3
    np.testing.assert_array_equal(manual.current.voltages, batch.current.voltages)
    manual.run_until_converged()
    batch.run_until_converged()
    assert manual.state.iteration == batch.state.iteration
    np.testing.assert_array_equal(manual.current.voltages, batch.current.voltages)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_batch_counts(three_bus, n):
    run = RunController(three_bus)
    with pytest.raises(ValueError):
        run.run_n(n)
    assert run.state.iteration == 0


@pytest.mark.parametrize("method", METHODS)
def test_converged_initial_state_takes_no_updates(three_bus, method):
    three_bus.config.method = method
    for bus in three_bus.buses[1:]:
        bus.p = 0
    three_bus.buses[1].q = 0
    run = RunController(three_bus)
    assert run.status == "Converged"
    run.step()
    run.run_n(4)
    run.run_until_converged()
    assert run.state.iteration == 0
    assert len(run.history.snapshots) == 1


@pytest.mark.parametrize("method", METHODS)
def test_final_allowed_update_converges_and_one_less_hits_cap(three_bus, method):
    three_bus.config.method = method
    unlimited = RunController(three_bus)
    unlimited.run_until_converged()
    count = unlimited.state.iteration
    assert count > 1
    three_bus.config.max_iterations = count
    final_allowed = RunController(three_bus)
    final_allowed.run_until_converged()
    assert final_allowed.status == "Converged"
    assert final_allowed.state.iteration == count
    three_bus.config.max_iterations = count - 1
    capped = RunController(three_bus)
    capped.run_n(count + 5)
    assert capped.status == "Maximum iterations reached"
    assert capped.state.iteration == count - 1
    before = capped.current.voltages.copy()
    capped.step()
    np.testing.assert_array_equal(capped.current.voltages, before)


def test_stop_and_reset_create_fresh_history_and_original_types(three_bus):
    three_bus.config.enforce_q_limits = True
    three_bus.buses[2].q_max = 0.2
    run = RunController(three_bus)
    run.run_n(1)
    run.stop()
    assert run.status == "Paused"
    assert run.state.iteration == 1
    assert any("stop" in str(e).lower() for e in run.history.events)
    run.run_until_converged()
    assert run.current.effective_types[2] == "PQ"
    old_id = run.history.run_id
    run.reset()
    assert run.history.run_id != old_id
    assert run.state.iteration == 0
    assert run.current.effective_types[2] == "PV"
    assert len(run.history.snapshots) == 1
    np.testing.assert_allclose(run.current.voltages, 1)
    three_bus.config.method = "Gauss-Seidel"
    new_run = RunController(three_bus)
    assert new_run.history.run_id != old_id
    assert len(new_run.history.snapshots) == 1
    assert new_run.current.effective_types[2] == "PV"


def test_singular_newton_step_preserves_last_valid_snapshot():
    # At zero angle in a purely resistive line, this PV's active-power
    # angle derivative is exactly zero, but its scheduled mismatch is not.
    case = NetworkCase(
        buses=[BusData("ref", type="Slack", v_set=1, angle_set=0),
               BusData("gen", type="PV", p=1, v_set=1)],
        mode="direct", ybus=np.array([[1, -1], [-1, 1]], complex))
    run = RunController(case)
    initial = run.current
    run.step()
    assert run.status == "Numerical failure"
    assert run.state.iteration == 0
    assert run.current is initial
    np.testing.assert_array_equal(run.state.voltages, np.ones(2))
    assert "singular" in str(run.history.events).lower()
    assert "1" in str(run.history.events)


@pytest.mark.parametrize("correction", [float("nan"), float("inf"), -5.0])
def test_invalid_attempted_newton_states_preserve_history(three_bus, monkeypatch, correction):
    run = RunController(three_bus)
    initial = run.current

    def invalid_solve(matrix, residual):
        output = np.zeros(len(residual))
        output[-1] = correction
        return output

    monkeypatch.setattr(np.linalg, "solve", invalid_solve)
    run.step()
    assert run.status == "Numerical failure"
    assert run.current is initial
    assert run.state.iteration == 0
    assert len(run.history.snapshots) == 1
    assert run.history.events
    np.testing.assert_array_equal(run.state.voltages, initial.voltages)


def test_gauss_seidel_rejects_zero_diagonal_before_iteration():
    case = NetworkCase(
        buses=[BusData("ref", type="Slack", v_set=1, angle_set=0),
               BusData("pq", type="PQ", p=-0.1, q=-0.1)],
        mode="direct", ybus=np.array([[1, -1], [-1, 0]], complex),
        config=SolverConfig(method="Gauss-Seidel"))
    with pytest.raises(ValueError, match="[Dd]iagonal"):
        RunController(case)


def test_simultaneous_limit_switches_follow_bus_order_at_iteration_zero():
    # This zero-angle exact inner solution needs Q=0 from both PV buses.
    # Both lower bounds bind together, without consuming any solver update.
    y = 1j * np.array([[-10, 5, 5], [5, -10, 5], [5, 5, -10]])
    buses = [BusData("z-generator", type="PV", p=0, v_set=1, q_min=0.1, q_max=1),
             BusData("reference", type="Slack", v_set=1, angle_set=0),
             BusData("a-generator", type="PV", p=0, v_set=1, q_min=0.2, q_max=1)]
    case = NetworkCase(buses=buses, mode="direct", ybus=y,
                       config=SolverConfig(enforce_q_limits=True, tolerance=1e-10))
    run = RunController(case)
    assert run.current.iteration == 0
    assert len(run.history.snapshots) == 2
    assert run.history.snapshots[0].effective_types == ["PV", "Slack", "PV"]
    assert run.current.effective_types == ["PQ", "Slack", "PQ"]
    assert [event["bus_id"] for event in run.history.events] == ["z-generator", "a-generator"]
    assert len(run.current.residual) == 4
    run.run_until_converged()
    assert run.status == "Converged"
    np.testing.assert_allclose(run.current.q_calculated[[0, 2]], [0.1, 0.2], atol=1e-10)


def test_constraint_switch_at_final_allowed_update_hits_cap(three_bus):
    ordinary = RunController(three_bus)
    ordinary.run_until_converged()
    three_bus.config.max_iterations = ordinary.state.iteration
    three_bus.config.enforce_q_limits = True
    three_bus.buses[2].q_max = 0.2
    run = RunController(three_bus)
    run.run_until_converged()
    assert run.status == "Maximum iterations reached"
    assert run.current.phase == "constraint_change"
    assert run.current.iteration == ordinary.state.iteration
    assert run.current.effective_types[2] == "PQ"
    assert run.current.max_mismatch > three_bus.config.tolerance


def test_zero_gauss_seidel_provisional_voltage_discards_entire_sweep():
    case = NetworkCase(
        buses=[BusData("ref", type="Slack", v_set=1, angle_set=0),
               BusData("pq", type="PQ", p=0, q=-10)],
        mode="direct", ybus=1j * np.array([[-10, 10], [10, -10]]),
        config=SolverConfig(method="Gauss-Seidel"))
    run = RunController(case)
    initial = run.current
    run.step()
    assert run.status == "Numerical failure"
    assert run.current is initial
    assert run.state.iteration == 0
    assert "provisional" in str(run.history.events)
    np.testing.assert_array_equal(run.state.voltages, np.ones(2))


def test_newton_retains_unwrapped_angles_and_corrections(three_bus):
    three_bus.buses[1].angle_guess = 360
    three_bus.buses[2].angle_guess = -360
    run = RunController(three_bus)
    run.step()
    np.testing.assert_allclose(run.current.angles,
        [0, 2*np.pi - 0.1142857142857, -2*np.pi + 0.05714285714286], atol=1e-12)
    np.testing.assert_allclose(run.current.delta_angle,
        [0, -0.1142857142857, 0.05714285714286], atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", [2, 7, 50])
def test_dynamic_lossy_networks_recover_independent_constructed_solution(method, size):
    # Synthesize schedules from independently selected feasible voltages. This
    # checks dynamic dimensions and a middle slack without an external solver.
    slack = size // 2
    vm = np.linspace(0.98, 1.01, size)
    va = np.linspace(-0.012, 0.008, size)
    vm[slack], va[slack] = 1.02, 0.005
    expected_v = vm * np.exp(1j * va)
    y = np.zeros((size, size), complex)
    branches = []
    for i in range(size):
        if i == slack:
            continue
        impedance = complex(0.01 + 0.0001*i, 0.1 + 0.001*i)
        admittance = 1/impedance
        y[slack, slack] += admittance
        y[i, i] += admittance
        y[i, slack] -= admittance
        y[slack, i] -= admittance
        branches.append(BranchData(f"line-{i}", f"bus-{slack}", f"bus-{i}",
                                   impedance.real, impedance.imag))
    expected_s = expected_v * np.conj(y @ expected_v)
    buses = []
    for i in range(size):
        if i == slack:
            buses.append(BusData(f"bus-{i}", type="Slack", v_set=vm[i], angle_set=np.rad2deg(va[i])))
        elif i % 2 == 0:
            buses.append(BusData(f"bus-{i}", type="PQ", p=expected_s[i].real, q=expected_s[i].imag))
        else:
            buses.append(BusData(f"bus-{i}", type="PV", p=expected_s[i].real, v_set=vm[i]))
    case = NetworkCase(buses=buses, branches=branches,
                       config=SolverConfig(method=method, tolerance=1e-10, max_iterations=200))
    run = RunController(case)
    run.run_until_converged()
    assert run.status == "Converged"
    np.testing.assert_allclose(run.current.voltages, expected_v, rtol=0, atol=2e-10)
    assert sum(run.current.p_calculated) > 0  # Positive real losses in resistive lines.
