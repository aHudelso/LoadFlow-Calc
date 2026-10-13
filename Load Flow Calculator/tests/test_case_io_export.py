"""Versioned JSON and the complete rectangular CSV are part of the public UI."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from loadflow_app.case_io import case_from_dict, case_to_dict, example_case, load_case, save_case
from loadflow_app.export import CSV_COLUMNS, export_csv
from loadflow_app.models import BusData, NetworkCase
from loadflow_app.network import validate_case
from loadflow_app.run_controller import RunController


def read_export(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(CSV_COLUMNS)
        rows = list(reader)
    assert rows
    assert all(set(row) == set(CSV_COLUMNS) for row in rows)
    assert all(value is not None for row in rows for value in row.values())
    return rows


@pytest.mark.parametrize("direct", [False, True])
def test_json_roundtrip_preserves_inputs_order_guesses_settings(three_bus, tmp_path, direct):
    three_bus.buses[1].v_guess = 0.987654321012345
    three_bus.buses[1].angle_guess = -4.3210123456789
    three_bus.buses[0].name = "Reference, café"
    three_bus.buses[0].shunt_g = 0.005
    three_bus.base_mva = 100
    three_bus.config.method = "Gauss-Seidel"
    three_bus.config.enforce_q_limits = True
    if direct:
        y = validate_case(three_bus)
        y[0, 2] += 0.002 + 0.007j  # Nonsymmetric direct entry survives exactly.
        three_bus.mode = "direct"
        three_bus.ybus = y
        three_bus.branches = []
    three_bus.buses = [three_bus.buses[i] for i in [2, 0, 1]]
    if direct:
        three_bus.ybus = three_bus.ybus[np.ix_([2, 0, 1], [2, 0, 1])]
    path = tmp_path / "case.json"
    save_case(three_bus, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["buses"][2]["v_guess"] == three_bus.buses[2].v_guess
    if direct:
        assert raw["ybus"][0][1] == [three_bus.ybus[0, 1].real, three_bus.ybus[0, 1].imag]
    loaded = load_case(path)
    assert case_to_dict(loaded) == case_to_dict(three_bus)
    np.testing.assert_array_equal(validate_case(loaded), validate_case(three_bus))
    assert [b.id for b in loaded.buses] == ["3", "1", "2"]
    assert RunController(loaded).state.iteration == 0


def test_shipped_example_matches_builtin(expected_ybus):
    path = Path(__file__).resolve().parents[1] / "examples" / "three_bus.json"
    disk = load_case(path)
    builtin = example_case()
    assert case_to_dict(disk) == case_to_dict(builtin)
    np.testing.assert_array_equal(validate_case(disk), expected_ybus)


@pytest.mark.parametrize("mutation", ["version", "unknown", "matrix_string", "bool_tolerance",
                                       "bad_bus", "one_bound", "nan", "both_networks"])
def test_malformed_json_does_not_mutate_existing_case(three_bus, mutation):
    original = case_to_dict(three_bus)
    invalid = copy.deepcopy(original)
    if mutation == "version":
        invalid["schema_version"] = 100
    elif mutation == "unknown":
        invalid["surprise"] = 1
    elif mutation == "matrix_string":
        invalid["network_mode"] = "direct"
        invalid["ybus"] = "1j"
    elif mutation == "bool_tolerance":
        invalid["solver"]["tolerance"] = True
    elif mutation == "bad_bus":
        invalid["buses"][1]["p"] = "bad"
    elif mutation == "one_bound":
        invalid["solver"]["enforce_q_limits"] = True
        invalid["buses"][2]["q_max"] = None
    elif mutation == "nan":
        invalid["buses"][1]["p"] = np.nan
    elif mutation == "both_networks":
        invalid["ybus"] = [[[0, 0]] * 3 for _ in range(3)]
    with pytest.raises(ValueError):
        case_from_dict(invalid)
    assert case_to_dict(three_bus) == original


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":2}',
                                   '{"schema_version":NaN}', '{"schema_version":1', '[]'])
def test_invalid_json_file_has_useful_error(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_case(path)
    assert str(error.value)


@pytest.mark.parametrize("method", ["Newton-Raphson", "Gauss-Seidel"])
def test_csv_full_precision_inputs_history_and_na_fields(three_bus, tmp_path, method):
    three_bus.config.method = method
    three_bus.buses[1].angle_guess = -0.1234567890123
    three_bus.buses[1].v_guess = 0.99876543210123
    run = RunController(three_bus)
    run.run_n(2)
    snapshots_before = list(run.history.snapshots)
    voltage_before = run.state.voltages.copy()
    path = tmp_path / "all.csv"
    export_csv(run, path)
    rows = read_export(path)
    assert len(run.history.snapshots) == len(snapshots_before)
    np.testing.assert_array_equal(run.state.voltages, voltage_before)
    history = [r for r in rows if r["record_type"] == "iteration_bus"]
    assert len(history) == len(run.history.snapshots) * 3
    assert {int(r["snapshot_id"]) for r in history} == {s.snapshot_id for s in run.history.snapshots}
    assert {int(r["iteration"]) for r in history} == {0, 1, 2}
    for row in history:
        if row["bus_id"] == "1":
            assert row["delta_p"] == row["delta_q"] == ""
            assert row["p_target"] == row["q_target"] == ""
        if row["bus_id"] == "3":
            assert row["delta_q"] == row["q_target"] == ""
        if row["iteration"] == "0":
            assert row["delta_angle_deg"] == row["delta_magnitude"] == ""
        else:
            assert row["delta_angle_deg"] != ""
    bus_inputs = [r for r in rows if r["record_type"] == "bus_input"]
    assert [r["bus_id"] for r in bus_inputs] == ["1", "2", "3"]
    assert float(bus_inputs[1]["v_guess"]) == three_bus.buses[1].v_guess
    assert float(bus_inputs[1]["angle_guess_deg"]) == three_bus.buses[1].angle_guess
    assert len([r for r in rows if r["record_type"] == "branch_input"]) == 3
    yrows = [r for r in rows if r["record_type"] == "ybus_entry"]
    assert len(yrows) == 9
    for row in yrows:
        i, j = int(row["row_index"])-1, int(row["column_index"])-1
        assert complex(float(row["y_real"]), float(row["y_imag"])) == run.ybus[i, j]
        assert row["row_bus_id"] == run.case.buses[i].id
        assert row["column_bus_id"] == run.case.buses[j].id
    metadata = {r["key"]: r["value"] for r in rows if r["record_type"] == "metadata"}
    assert metadata["final_status"] == run.status
    assert metadata["method"] == method
    assert metadata["run_id"] == run.history.run_id
    assert metadata["nr_magnitude_correction"] == "absolute per unit"


def test_csv_newton_details_record_source_residual_jacobian_and_correction(three_bus, tmp_path):
    run = RunController(three_bus)
    run.step()
    path = tmp_path / "newton.csv"
    export_csv(run, path)
    details = [r for r in read_export(path) if r["record_type"] == "solver_detail"]
    jrows = [r for r in details if r["detail_kind"] == "jacobian"]
    expected = np.array([[15, -5, 0], [-5, 7.5, 0], [0, 0, 15]])
    assert len(jrows) == 9
    for row in jrows:
        assert float(row["value"]) == expected[int(row["row_index"])-1, int(row["column_index"])-1]
        assert row["row_label"] and row["column_label"]
    residual = [r for r in details if r["detail_kind"] == "residual"]
    np.testing.assert_allclose([float(r["value"]) for r in residual], [-2, 1, -0.5])
    assert all(r["source_iteration"] == "0" and r["iteration"] == "1" for r in details)
    correction = [r for r in details if r["detail_kind"] == "correction"]
    np.testing.assert_allclose([float(r["value"]) for r in correction], [-0.114285714285714, 0.057142857142857, -1/30])
    assert [r["unit"] for r in correction] == ["rad", "rad", "pu"]


def test_csv_gauss_seidel_temporary_q_is_distinct(three_bus, tmp_path):
    three_bus.config.method = "Gauss-Seidel"
    run = RunController(three_bus)
    run.step()
    path = tmp_path / "gs.csv"
    export_csv(run, path)
    rows = read_export(path)
    temporary = [r for r in rows if r["record_type"] == "solver_detail" and r["detail_kind"] == "gs_pv_q_used"]
    assert len(temporary) == 1
    assert temporary[0]["bus_id"] == "3"
    assert float(temporary[0]["value"]) == pytest.approx(1/6)
    final = [r for r in rows if r["record_type"] == "iteration_bus" and r["iteration"] == "1" and r["bus_id"] == "3"][0]
    assert float(final["q_calculated"]) == pytest.approx(0.203499161662)
    assert float(final["q_calculated"]) != pytest.approx(float(temporary[0]["value"]))


def test_csv_iteration_zero_and_q_switch_snapshots(three_bus, tmp_path):
    three_bus.config.enforce_q_limits = True
    three_bus.buses[2].q_max = 0.3
    three_bus.buses[2].q_load = 0.1
    run = RunController(three_bus)
    path = tmp_path / "history.csv"
    export_csv(run, path)
    initial_rows = read_export(path)
    assert len([r for r in initial_rows if r["record_type"] == "iteration_bus"]) == 3
    assert run.state.iteration == 0
    run.run_until_converged()
    export_csv(run, path)
    rows = read_export(path)
    changed = [r for r in rows if r["record_type"] == "iteration_bus" and r["phase"] == "constraint_change" and r["bus_id"] == "3"]
    assert len(changed) == 1
    assert changed[0]["original_type"] == "PV" and changed[0]["effective_type"] == "PQ"
    assert float(changed[0]["q_target"]) == pytest.approx(0.2)
    assert changed[0]["delta_q"] != "" and changed[0]["binding_limit"] != ""
    assert changed[0]["v_set"] == "1.0" and changed[0]["fixed_v_set"] == ""
    assert [r for r in rows if r["record_type"] == "event"]


def test_csv_failure_retains_final_status_and_last_valid_values(tmp_path):
    case = NetworkCase(
        buses=[BusData("S", type="Slack", v_set=1, angle_set=0),
               BusData("G", type="PV", p=1, v_set=1)],
        mode="direct", ybus=np.array([[1, -1], [-1, 1]], complex))
    run = RunController(case)
    run.step()
    path = tmp_path / "failed.csv"
    export_csv(run, path)
    rows = read_export(path)
    metadata = {r["key"]: r["value"] for r in rows if r["record_type"] == "metadata"}
    assert metadata["final_status"] == "Numerical failure"
    history = [r for r in rows if r["record_type"] == "iteration_bus"]
    assert len(history) == 2
    assert {r["iteration"] for r in history} == {"0"}
    assert all(float(r["voltage_magnitude"]) == 1 for r in history)
    assert any("singular" in r["message"].lower() for r in rows if r["record_type"] == "event")
    failure = [r for r in rows if r["record_type"] == "event"][0]
    assert failure["iteration"] == "0" and failure["attempted_iteration"] == "1"
    assert metadata["branch_data"] == "Branch data not supplied"


def test_export_write_error_leaves_run_usable(three_bus, tmp_path):
    run = RunController(three_bus)
    with pytest.raises(OSError):
        export_csv(run, tmp_path / "missing-directory" / "run.csv")
    assert run.state.iteration == 0
    run.step()
    assert run.state.iteration == 1


def test_json_null_irrelevant_guesses_roundtrip(three_bus):
    three_bus.buses[0].v_guess = None
    three_bus.buses[0].angle_guess = None
    three_bus.buses[0].q_load = None
    three_bus.buses[2].v_guess = None
    data = case_to_dict(three_bus)
    loaded = case_from_dict(data)
    assert case_to_dict(loaded) == data
    run = RunController(loaded)
    np.testing.assert_allclose(run.current.voltages, 1)
    for field in ("v_guess", "angle_guess"):
        invalid = copy.deepcopy(data)
        invalid["buses"][1][field] = None
        with pytest.raises(ValueError):
            case_from_dict(invalid)
