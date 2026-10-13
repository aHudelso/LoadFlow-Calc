"""Network parsing, admittance stamps, input validation, and initial state."""

from __future__ import annotations

import math
import numbers
import re

import numpy as np

from .models import BranchData, BusData, NetworkCase, ValidationError


METHODS = ("Newton-Raphson", "Gauss-Seidel")
BUS_TYPES = ("Slack", "PV", "PQ")


def _finite(value, field: str, *, positive: bool = False) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{field} must be a finite real number.")
    try:
        finite = math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite or (positive and value <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValidationError(f"{field} must be a {qualifier} real number.")


def positive_integer(value, field: str = "Iteration count") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValidationError(f"{field} must be a positive integer.")
    return int(value)


def parse_ybus(text: str) -> np.ndarray:
    """Parse only complex numeric literals, never expressions or Python code.

    Columns may be separated by commas or whitespace. Both i and j suffixes
    are accepted. Empty lines are ignored; empty comma entries are rejected.
    """
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if re.search(r"(^\s*,|,\s*,|,\s*$)", line):
            raise ValidationError(f"Ybus row {line_number} contains an empty entry.")
        values = []
        for column, token in enumerate(re.split(r"[\s,]+", line.strip()), 1):
            try:
                value = complex(token.lower().replace("i", "j"))
            except (ValueError, OverflowError) as exc:
                raise ValidationError(
                    f"Ybus row {line_number}, column {column}: {token!r} is not a complex number. "
                    "Use a literal such as 0-12.5j."
                ) from exc
            if not np.isfinite(value):
                raise ValidationError(f"Ybus row {line_number}, column {column} must be finite.")
            values.append(value)
        rows.append(values)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValidationError("Ybus must be a nonempty square matrix with one row per line.")
    return np.asarray(rows, dtype=complex)


def _validate_ids(buses: list[BusData]) -> dict:
    index = {}
    for i, bus in enumerate(buses):
        if not isinstance(bus.id, (str, int)) or isinstance(bus.id, bool) or not str(bus.id).strip():
            raise ValidationError(f"Bus row {i + 1} needs a nonempty string or integer ID.")
        if bus.id in index or any(str(old) == str(bus.id) for old in index):
            raise ValidationError(f"Bus ID {bus.id!r} is duplicated.")
        index[bus.id] = i
    return index


def _branch_graph(buses: list[BusData], branches: list[BranchData]):
    index = _validate_ids(buses)
    adjacency = [set() for _ in buses]
    branch_ids = set()
    for branch in branches:
        label = f"Branch {branch.id}"
        if not isinstance(branch.id, (str, int)) or isinstance(branch.id, bool) or not str(branch.id).strip():
            raise ValidationError("Every branch needs a nonempty string or integer ID.")
        if str(branch.id) in branch_ids:
            raise ValidationError(f"{label}: duplicate branch ID.")
        branch_ids.add(str(branch.id))
        if branch.from_bus not in index or branch.to_bus not in index:
            raise ValidationError(f"{label}: both endpoints must name existing bus IDs.")
        if branch.from_bus == branch.to_bus:
            raise ValidationError(f"{label}: from and to buses must be distinct.")
        for name in ("r", "x", "b"):
            _finite(getattr(branch, name), f"{label} {name}")
        if branch.r == 0 and branch.x == 0:
            raise ValidationError(f"{label}: series impedance r + jx cannot be zero.")
        i, j = index[branch.from_bus], index[branch.to_bus]
        adjacency[i].add(j)
        adjacency[j].add(i)
    return index, adjacency


def build_ybus(buses: list[BusData], branches: list[BranchData]) -> np.ndarray:
    """Stamp ordinary series branches, half line charging, and bus shunts."""
    index, _ = _branch_graph(buses, branches)
    ybus = np.zeros((len(buses), len(buses)), dtype=complex)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for branch in branches:
            i, j = index[branch.from_bus], index[branch.to_bus]
            admittance = 1.0 / complex(branch.r, branch.x)
            charging = 0.5j * branch.b
            ybus[i, i] += admittance + charging
            ybus[j, j] += admittance + charging
            ybus[i, j] -= admittance
            ybus[j, i] -= admittance
        for i, bus in enumerate(buses):
            _finite(bus.shunt_g, f"Bus {bus.id} shunt conductance")
            _finite(bus.shunt_b, f"Bus {bus.id} shunt susceptance")
            ybus[i, i] += complex(bus.shunt_g, bus.shunt_b)
    if not np.all(np.isfinite(ybus)):
        raise ValidationError("Branch impedances or shunts produce nonfinite Ybus entries.")
    return ybus


def _check_connected(adjacency: list[set], buses: list[BusData]) -> None:
    reached = {0}
    pending = [0]
    while pending:
        i = pending.pop()
        for j in adjacency[i] - reached:
            reached.add(j)
            pending.append(j)
    if len(reached) != len(buses):
        missing = ", ".join(str(buses[i].id) for i in range(len(buses)) if i not in reached)
        raise ValidationError(
            f"The network is disconnected (separate buses include {missing}). "
            "Version 1 requires one connected network with exactly one Slack bus."
        )


def validate_case(case: NetworkCase) -> np.ndarray:
    """Validate without modifying user inputs, returning an independent Ybus."""
    if len(case.buses) < 2:
        raise ValidationError("At least two buses are required.")
    _validate_ids(case.buses)
    if case.mode not in ("branches", "direct"):
        raise ValidationError("Network mode must be 'branches' or 'direct'.")
    config = case.config
    if config.method not in METHODS:
        raise ValidationError(f"Solver method must be one of {', '.join(METHODS)}.")
    _finite(config.tolerance, "Mismatch tolerance", positive=True)
    positive_integer(config.max_iterations, "Maximum total iterations")
    if not isinstance(config.enforce_q_limits, (bool, np.bool_)):
        raise ValidationError("Enforce Q limits must be a boolean.")
    if case.base_mva is not None:
        _finite(case.base_mva, "System base MVA", positive=True)
    slack_count = 0
    for bus in case.buses:
        label = f"Bus {bus.id}"
        if bus.type not in BUS_TYPES:
            raise ValidationError(f"{label}: type must be Slack, PV, or PQ.")
        slack_count += bus.type == "Slack"
        if bus.type != "Slack":
            _finite(bus.p, f"{label} net P schedule")
            _finite(bus.angle_guess, f"{label} angle guess")
        if bus.type == "PQ":
            _finite(bus.q, f"{label} net Q schedule")
            _finite(bus.v_guess, f"{label} voltage magnitude guess", positive=True)
        else:
            _finite(bus.v_set, f"{label} specified voltage magnitude", positive=True)
        if bus.type == "Slack":
            _finite(bus.angle_set, f"{label} specified angle")
        for name in ("shunt_g", "shunt_b"):
            _finite(getattr(bus, name), f"{label} {name}")
        if bus.type == "PV":
            _finite(bus.q_load, f"{label} local reactive load")
            for name in ("q_min", "q_max"):
                if getattr(bus, name) is not None:
                    _finite(getattr(bus, name), f"{label} generator {name}")
            if config.enforce_q_limits and (bus.q_min is not None or bus.q_max is not None):
                if bus.q_min is None or bus.q_max is None:
                    raise ValidationError(f"{label}: enabled generator Q limits need both minimum and maximum.")
                if bus.q_min > bus.q_max:
                    raise ValidationError(f"{label}: generator Q minimum must not exceed maximum.")
                with np.errstate(over="ignore", invalid="ignore"):
                    if not math.isfinite(float(bus.q_min) - float(bus.q_load)) or not math.isfinite(float(bus.q_max) - float(bus.q_load)):
                        raise ValidationError(f"{label}: generator limits minus local Q load must remain finite.")
    if slack_count != 1:
        raise ValidationError("Exactly one Slack bus is required for the connected network.")
    if case.mode == "branches":
        ybus = build_ybus(case.buses, case.branches)
        _, adjacency = _branch_graph(case.buses, case.branches)
    else:
        try:
            ybus = np.array(case.ybus, dtype=complex, copy=True)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValidationError("Direct Ybus entries must be complex numeric values.") from exc
        n = len(case.buses)
        if ybus.shape != (n, n):
            raise ValidationError(f"Ybus must be square with {n} rows and columns to match the bus table.")
        if not np.all(np.isfinite(ybus)):
            raise ValidationError("Every Ybus entry must be finite.")
        adjacency = [
            {j for j in range(n) if j != i and (ybus[i, j] != 0 or ybus[j, i] != 0)}
            for i in range(n)
        ]
    _check_connected(adjacency, case.buses)
    if config.method == "Gauss-Seidel":
        for i, bus in enumerate(case.buses):
            if bus.type != "Slack" and ybus[i, i] == 0:
                raise ValidationError(f"Bus {bus.id}: Gauss-Seidel requires a nonzero diagonal Ybus admittance.")
    return ybus


def initial_voltages(buses: list[BusData]) -> np.ndarray:
    magnitudes = [bus.v_guess if bus.type == "PQ" else bus.v_set for bus in buses]
    angles = [bus.angle_set if bus.type == "Slack" else bus.angle_guess for bus in buses]
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(magnitudes, dtype=float) * np.exp(1j * np.deg2rad(np.asarray(angles, dtype=float)))


def unknown_labels(buses: list[BusData], effective_types: list[str] | None = None) -> list[str]:
    types = effective_types if effective_types is not None else [bus.type for bus in buses]
    return ([f"angle at Bus {bus.id}" for bus, typ in zip(buses, types) if typ != "Slack"]
            + [f"magnitude at Bus {bus.id}" for bus, typ in zip(buses, types) if typ == "PQ"])
