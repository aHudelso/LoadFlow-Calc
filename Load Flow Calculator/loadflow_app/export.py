"""A rectangular CSV containing original inputs and every recorded snapshot."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .case_io import APP_VERSION

CSV_SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "record_type", "run_id", "key", "value", "unit", "snapshot_id", "iteration",
    "phase", "status", "bus_order", "bus_id", "bus_name", "original_type",
    "effective_type", "p_schedule", "q_schedule", "v_set", "angle_set_deg",
    "v_guess", "angle_guess_deg", "q_min", "q_max", "q_load", "shunt_g", "shunt_b",
    "branch_id", "from_bus", "to_bus", "r", "x", "line_b", "row_bus_id",
    "column_bus_id", "y_real", "y_imag", "voltage_real", "voltage_imag",
    "voltage_magnitude", "angle_deg", "p_target", "q_target", "p_calculated",
    "q_calculated", "q_generator", "delta_p", "delta_q", "delta_magnitude",
    "delta_angle_deg", "fixed_v_set", "max_mismatch", "max_mismatch_label",
    "tolerance", "binding_limit", "event_type", "message", "source_iteration",
    "detail_kind", "row_index", "column_index", "row_label", "column_label",
    "attempted_iteration", "generator_q_limit", "net_q_target",
)


def _item(values: Any, index: int) -> Any:
    return None if values is None else values[index]


def _rows(run: Any, exported_at: str) -> Iterator[dict]:
    case = run.case
    config = case.config
    metadata = {
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "run_id": run.history.run_id,
        "export_time_utc": exported_at,
        "method": config.method,
        "tolerance": config.tolerance,
        "q_limit_tolerance": config.tolerance,
        "max_iterations": config.max_iterations,
        "enforce_q_limits": config.enforce_q_limits,
        "q_limit_policy": "outer loop after inner convergence; simultaneous violations in bus order; no restoration",
        "base_mva": case.base_mva,
        "network_mode": case.mode,
        "bus_count": len(case.buses),
        "branch_count": len(case.branches) if case.mode == "branches" else None,
        "branch_data": "Supplied" if case.mode == "branches" else "Branch data not supplied",
        "final_status": run.status,
        "completed_iterations": run.state.iteration,
        "snapshot_count": len(run.history.snapshots),
        "power_convention": "net injection = generation minus load; per unit",
        "angle_convention": "degrees in bus inputs and snapshot columns; unwrapped angles; Newton detail corrections in radians",
        "nr_magnitude_correction": "absolute per unit",
        "equation_order": "P at non-slack buses, then Q at effective PQ buses; each in bus order",
        "unknown_order": "angles at non-slack buses, then magnitudes at effective PQ buses; each in bus order",
    }
    for key, value in metadata.items():
        yield {"record_type": "metadata", "key": key, "value": value}
    for order, bus in enumerate(case.buses, start=1):
        yield {
            "record_type": "bus_input", "bus_order": order, "bus_id": bus.id,
            "bus_name": bus.name, "original_type": bus.type,
            "p_schedule": bus.p, "q_schedule": bus.q,
            "v_set": bus.v_set, "angle_set_deg": bus.angle_set,
            "v_guess": bus.v_guess, "angle_guess_deg": bus.angle_guess,
            "q_min": bus.q_min, "q_max": bus.q_max, "q_load": bus.q_load,
            "shunt_g": bus.shunt_g, "shunt_b": bus.shunt_b,
        }
    if case.mode == "branches":
        for branch in case.branches:
            yield {"record_type": "branch_input", "branch_id": branch.id,
                   "from_bus": branch.from_bus, "to_bus": branch.to_bus,
                   "r": branch.r, "x": branch.x, "line_b": branch.b}
    for i, row in enumerate(run.ybus):
        for j, entry in enumerate(row):
            yield {"record_type": "ybus_entry", "row_index": i + 1, "column_index": j + 1,
                   "row_bus_id": case.buses[i].id, "column_bus_id": case.buses[j].id,
                   "y_real": float(entry.real), "y_imag": float(entry.imag)}
    angle_count = sum(bus.type != "Slack" for bus in case.buses)
    for snapshot in run.history.snapshots:
        common = {"snapshot_id": snapshot.snapshot_id, "iteration": snapshot.iteration,
                  "phase": snapshot.phase, "status": snapshot.status}
        for i, bus in enumerate(case.buses):
            voltage = snapshot.voltages[i]
            angle_change = _item(snapshot.delta_angle, i)
            yield {
                **common, "record_type": "iteration_bus", "bus_order": i + 1,
                "bus_id": bus.id, "bus_name": bus.name, "original_type": bus.type,
                "effective_type": snapshot.effective_types[i],
                "voltage_real": float(voltage.real), "voltage_imag": float(voltage.imag),
                "voltage_magnitude": float(abs(voltage)),
                "angle_deg": float(np.degrees(snapshot.angles[i])),
                "p_schedule": bus.p, "q_schedule": bus.q,
                "p_target": snapshot.p_targets[i], "q_target": snapshot.q_targets[i],
                "p_calculated": snapshot.p_calculated[i], "q_calculated": snapshot.q_calculated[i],
                "q_generator": snapshot.q_calculated[i] + bus.q_load if bus.type == "PV" else None,
                "delta_p": snapshot.delta_p[i], "delta_q": snapshot.delta_q[i],
                "delta_magnitude": _item(snapshot.delta_magnitude, i),
                "delta_angle_deg": None if angle_change is None else float(np.degrees(angle_change)),
                "v_set": bus.v_set,
                "fixed_v_set": bus.v_set if snapshot.effective_types[i] in {"Slack", "PV"} else None,
                "max_mismatch": snapshot.max_mismatch,
                "max_mismatch_label": snapshot.max_mismatch_label,
                "tolerance": config.tolerance, "binding_limit": snapshot.binding_limits[i],
                "message": snapshot.event,
            }
        details = snapshot.details or {}
        detail_common = {**common, "record_type": "solver_detail",
                         "source_iteration": details.get("source_iteration")}
        equation_labels = details.get("equation_labels", [])
        unknown_labels = details.get("unknown_labels", [])
        if "residual" in details:
            for i, value in enumerate(details["residual"]):
                yield {**detail_common, "detail_kind": "residual", "row_index": i + 1,
                       "row_label": equation_labels[i], "value": value, "unit": "pu"}
        if "jacobian" in details:
            for i, row in enumerate(details["jacobian"]):
                for j, value in enumerate(row):
                    yield {**detail_common, "detail_kind": "jacobian", "row_index": i + 1,
                           "column_index": j + 1, "row_label": equation_labels[i],
                           "column_label": unknown_labels[j], "value": value,
                           "unit": "pu/rad" if j < angle_count else "pu/pu"}
        if "correction" in details:
            for i, value in enumerate(details["correction"]):
                yield {**detail_common, "detail_kind": "correction", "column_index": i + 1,
                       "column_label": unknown_labels[i], "value": value,
                       "unit": "rad" if i < angle_count else "pu"}
        for bus_id, value in details.get("pv_q_used", {}).items():
            yield {**detail_common, "detail_kind": "gs_pv_q_used", "bus_id": bus_id,
                   "value": value, "unit": "pu"}
    for event in run.history.events:
        yield {"record_type": "event", "iteration": event.get("iteration"),
               "snapshot_id": event.get("snapshot_id"), "event_type": event.get("type"),
               "message": event.get("message"), "status": event.get("status"),
               "attempted_iteration": event.get("attempted_iteration"),
               "bus_id": event.get("bus_id"), "binding_limit": event.get("bound"),
               "generator_q_limit": event.get("generator_q_limit"),
               "net_q_target": event.get("net_q_target")}


def export_csv(run: Any, path: str | Path) -> None:
    """Export full history without advancing the live solver or selected snapshot.

    A failure can leave the last valid snapshot with an earlier status. The authoritative
    final run status is the metadata row with key ``final_status``.
    """
    exported_at = datetime.now(timezone.utc).isoformat()
    rows = list(_rows(run, exported_at))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"run_id": run.history.run_id, **row})
