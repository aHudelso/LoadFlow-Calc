"""Real Tk widgets/callbacks and event-loop tests; native dialogs are stubbed.

These tests require a usable graphical Tcl/Tk environment and skip otherwise.
They test application behavior, not the OS-native dialog's visual appearance.
"""

import csv
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter", reason="Tkinter is not installed in this Python environment")

from loadflow_app import gui
from loadflow_app.case_io import example_case, load_case, save_case
from loadflow_app.network import validate_case


@pytest.fixture
def app(monkeypatch):
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, message, **kw: errors.append((title, message)))
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args, **kw: None)
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *args, **kw: True)
    try:
        instance = gui.LoadFlowApp()
    except tk.TclError as exc:
        pytest.skip(f"A usable graphical Tcl/Tk environment is unavailable: {exc}")
    instance.withdraw()
    callbacks = []
    instance.report_callback_exception = lambda *args: callbacks.append(args)
    instance.test_errors = errors
    instance.update()
    yield instance
    instance.close()
    assert not callbacks, f"Tk callback exception(s): {callbacks}"


def initial(app, method="Newton-Raphson"):
    app.apply_case(example_case())
    app.method.set(method)
    app.continue_to_guesses()
    app.calculate_initial()
    assert app.run is not None, app.test_errors
    assert app.run.state.iteration == 0


def finish_batch(app):
    deadline = time.monotonic() + 5
    while app.busy and time.monotonic() < deadline:
        app.update()
        time.sleep(0.002)
    assert not app.busy, "Batch did not finish through Tk's event loop."
    app.update()


@pytest.mark.parametrize("method", ["Newton-Raphson", "Gauss-Seidel"])
def test_gui_example_unknowns_initial_and_one_manual_step(app, method):
    app.method.set(method)
    app.continue_to_guesses()
    assert list(app.guess_vars["1"]) == []
    assert set(app.guess_vars["2"]) == {"v_guess", "angle_guess"}
    assert set(app.guess_vars["3"]) == {"angle_guess"}
    assert app.unknowns_text.get().startswith("3 unknowns")
    app.calculate_initial()
    assert app.tabs.select() == str(app.results_tab)
    assert app.run.state.iteration == 0
    np.testing.assert_allclose(app.run.current.residual, [-2, 1, -0.5])
    app.step_once()
    assert app.run.state.iteration == 1
    assert app.tabs.select() == str(app.results_tab)
    expected = 0.966666666666667 if method == "Newton-Raphson" else 0.975818744554
    assert abs(app.run.current.voltages[1]) == pytest.approx(expected, abs=1e-12)
    assert not app.test_errors


def test_gui_run_n_until_and_history_do_not_rewind(app):
    initial(app)
    app.batch_n.set("2")
    app.start_batch(False)
    assert app.busy
    assert str(app.step_button.cget("state")) == "disabled"
    assert str(app.tabs.tab(app.configure_tab, "state")) == "disabled"
    finish_batch(app)
    assert app.run.state.iteration == 2
    assert app.run.status == "Paused"
    live = app.run.state.voltages.copy()
    app.history_combo.current(0)
    app.show_snapshot()
    assert "Viewing iteration 0" in app.viewing.get()
    np.testing.assert_array_equal(app.run.state.voltages, live)
    first_values = app.result_tree.item(app.result_tree.get_children()[1], "values")
    assert float(first_values[2]) == 1
    app.start_batch(True)
    finish_batch(app)
    assert app.run.status == "Converged"
    assert str(app.step_button.cget("state")) == "disabled"
    assert str(app.export_button.cget("state")) == "normal"


def test_gui_stop_between_updates_and_reset(app):
    initial(app, "Gauss-Seidel")
    app.start_batch(True)
    app.after(3, app.stop_batch)
    finish_batch(app)
    assert app.run.status == "Paused"
    assert app.run.state.iteration < 10
    assert app.run.history.events[-1]["type"] == "stop"
    old_run_id = app.run.history.run_id
    app.reset_run()
    assert app.run.history.run_id != old_run_id
    assert app.run.state.iteration == 0
    assert len(app.run.history.snapshots) == 1


def test_gui_configuration_changes_invalidate_existing_run(app):
    initial(app)
    app.step_once()
    old_id = app.run.history.run_id
    app.method.set("Gauss-Seidel")
    assert app.run is None
    assert str(app.tabs.tab(app.results_tab, "state")) == "disabled"
    app.continue_to_guesses()
    app.calculate_initial()
    assert app.run.history.run_id != old_id
    assert app.run.case.config.method == "Gauss-Seidel"
    assert app.run.state.iteration == 0
    app.guess_vars["2"]["v_guess"].set("0.97")
    assert app.run is None
    app.calculate_initial()
    assert abs(app.run.current.voltages[1]) == pytest.approx(0.97)


def test_gui_flat_start_preserves_specified_values(app):
    case = example_case()
    case.buses[0].v_set = 1.04
    case.buses[0].angle_set = 5
    case.buses[2].v_set = 1.02
    app.apply_case(case)
    app.continue_to_guesses()
    app.guess_vars["2"]["v_guess"].set("0.9")
    app.guess_vars["2"]["angle_guess"].set("-7")
    app.guess_vars["3"]["angle_guess"].set("2")
    app.flat_start()
    app.calculate_initial()
    np.testing.assert_allclose(abs(app.run.current.voltages), [1.04, 1, 1.02])
    np.testing.assert_allclose(np.rad2deg(app.run.current.angles), [5, 0, 0])


def test_gui_direct_matrix_dimensions_and_invalid_edits(app):
    app.mode.set("direct")
    app.change_mode()
    app.set_matrix(np.array([[-10j, 10j], [10j, -10j]]))
    app.apply_matrix()
    assert len(app.buses) == 2
    assert "Branch data not supplied" in app.branch_count.get()
    assert str(app.add_bus_button.cget("state")) == "disabled"
    app.buses[1].shunt_g = 0.2
    dialog = gui.FieldDialog(app, "bus", app.buses[1])
    assert str(dialog.entries["shunt_g"].cget("state")) == "disabled"
    assert str(dialog.entries["shunt_b"].cget("state")) == "disabled"
    dialog.variables["shunt_g"].set("999")
    dialog.apply()
    assert app.buses[1].shunt_g == 0.2
    app.continue_to_guesses()
    app.calculate_initial()
    assert app.run.case.mode == "direct"
    assert app.run.ybus.shape == (2, 2)
    app.direct_text.insert("end", "\ninvalid input")
    app.update()
    assert app.run is None
    app.continue_to_guesses()
    assert app.test_errors
    assert "Ybus" in app.test_errors[-1][1]


def test_gui_integer_ids_survive_load_and_bus_branch_edits(app):
    case = example_case()
    for bus in case.buses:
        bus.id = int(bus.id)
    for i, branch in enumerate(case.branches):
        branch.id = i + 10
        branch.from_bus, branch.to_bus = int(branch.from_bus), int(branch.to_bus)
    app.apply_case(case)
    assert app.next_bus_id() == "4"
    dialog = gui.FieldDialog(app, "bus", app.buses[1])
    dialog.variables["p"].set("-1.9")
    dialog.apply()
    assert app.buses[1].id == 2
    assert app.buses[1].p == -1.9
    dialog = gui.FieldDialog(app, "branch", app.branches[0])
    dialog.variables["x"].set("0.12")
    dialog.apply()
    assert app.branches[0].id == 10
    assert app.branches[0].from_bus == 1 and app.branches[0].to_bus == 2
    assert app.branches[0].x == 0.12
    app.continue_to_guesses()
    app.calculate_initial()
    assert app.run is not None
    assert not app.test_errors


def test_gui_remove_bus_removes_incident_branches(app):
    app.bus_tree.selection_set("1")
    app.remove_bus()
    assert [b.id for b in app.buses] == ["1", "3"]
    assert [b.id for b in app.branches] == ["L13"]
    app.continue_to_guesses()
    assert app.case is not None
    assert not app.test_errors


def test_gui_q_limit_event_and_reset(app):
    case = example_case()
    case.config.enforce_q_limits = True
    case.buses[2].q_max = 0.2
    app.apply_case(case)
    app.continue_to_guesses()
    app.calculate_initial()
    app.start_batch(True)
    finish_batch(app)
    assert app.run.status == "Converged"
    assert app.run.current.effective_types[2] == "PQ"
    assert "PV to PQ" in app.event_log.get("1.0", "end")
    assert "PV / PQ" in str(app.result_tree.item(app.result_tree.get_children()[2], "values"))
    app.reset_run()
    assert app.run.current.effective_types[2] == "PV"
    assert app.run.state.iteration == 0


def test_gui_export_path_cancel_and_error(app, monkeypatch, tmp_path):
    initial(app)
    path = tmp_path / "initial.csv"
    calls = []

    def choose_path(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", choose_path)
    app.export_results()
    assert path.exists()
    assert calls[0]["defaultextension"] == ".csv"
    with path.open(encoding="utf-8", newline="") as stream:
        assert len([r for r in csv.DictReader(stream) if r["record_type"] == "iteration_bus"]) == 3
    assert app.run.state.iteration == 0
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kw: "")
    before = app.footer.get()
    app.export_results()
    assert app.footer.get() == before
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kw: str(tmp_path / "missing" / "failed.csv"))
    app.export_results()
    assert app.test_errors[-1][0] == "Could not export results"
    assert app.run.state.iteration == 0
    app.step_once()
    assert app.run.state.iteration == 1


def test_gui_save_open_guesses_and_invalid_open_is_transactional(app, monkeypatch, tmp_path):
    initial(app)
    app.guess_vars["2"]["v_guess"].set("0.96")
    app.guess_vars["2"]["angle_guess"].set("-3.5")
    path = tmp_path / "guesses.json"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kw: str(path))
    app.save_case()
    stored = load_case(path)
    assert stored.buses[1].v_guess == 0.96
    assert stored.buses[1].angle_guess == -3.5
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: str(path))
    app.open_case()
    assert app.run is None
    app.continue_to_guesses()
    app.calculate_initial()
    assert abs(app.run.current.voltages[1]) == pytest.approx(0.96)
    old_run = app.run
    bad_path = tmp_path / "broken.json"
    bad_path.write_text('{"schema_version": 1, "broken": true}', encoding="utf-8")
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: str(bad_path))
    app.open_case()
    assert app.run is old_run
    assert app.test_errors[-1][0] == "Could not open case"
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: "")
    app.open_case()
    assert app.run is old_run


def test_gui_invalid_configuration_is_reported(app):
    app.tolerance.set("0")
    app.continue_to_guesses()
    assert app.run is None
    assert app.test_errors[-1][0] == "Check configuration"
    assert "tolerance" in app.test_errors[-1][1].lower()
