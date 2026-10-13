"""Versioned, validated case files and the built-in teaching example.

Case files hold configuration and original guesses, never a partially completed run.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .models import BranchData, BusData, NetworkCase, SolverConfig
from .network import validate_case

SCHEMA_VERSION = 1
APP_VERSION = "1.0.0"


def _object(value: Any, label: str, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}.")
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(sorted(missing))}.")
    return value


def _number(value: Any, label: str, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number{' or null' if optional else ''}.")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError):
        finite = False
    if not finite:
        raise ValueError(f"{label} must be finite.")
    return float(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    return value


def _identifier(value: Any, label: str) -> str | int:
    if type(value) not in (str, int):
        raise ValueError(f"{label} must be a string or integer identifier.")
    return value


def _list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array.")
    return value


def case_from_dict(data: Any) -> NetworkCase:
    """Parse and fully validate schema v1 without modifying any existing case."""
    data = _object(data, "Case", {
        "schema_version", "app_version", "network_mode", "base_mva", "buses",
        "branches", "ybus", "solver",
    }, {"schema_version", "network_mode", "buses", "solver"})
    version = data["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported case schema version {version!r}; expected {SCHEMA_VERSION}.")
    if "app_version" in data:
        _text(data["app_version"], "app_version")
    mode = _text(data["network_mode"], "network_mode")
    if mode not in {"branches", "direct"}:
        raise ValueError("network_mode must be 'branches' or 'direct'.")
    buses = []
    nullable = {"p", "q", "v_set", "angle_set", "q_min", "q_max"}
    numeric = nullable | {"v_guess", "angle_guess", "q_load", "shunt_g", "shunt_b"}
    bus_fields = numeric | {"id", "name", "type"}
    for i, item in enumerate(_list(data["buses"], "buses")):
        label = f"Bus entry {i + 1}"
        item = _object(item, label, bus_fields, {"id", "type"})
        parsed = dict(item)
        for key, value in item.items():
            if key in numeric:
                not_applicable = (
                    (key == "v_guess" and item["type"] != "PQ")
                    or (key == "angle_guess" and item["type"] == "Slack")
                    or (key == "q_load" and item["type"] != "PV")
                )
                parsed[key] = _number(value, f"{label} {key}", key in nullable or not_applicable)
            elif key == "id":
                parsed[key] = _identifier(value, f"{label} {key}")
            else:
                parsed[key] = _text(value, f"{label} {key}")
        buses.append(BusData(**parsed))
    branches = []
    for i, item in enumerate(_list(data.get("branches", []), "branches")):
        label = f"Branch entry {i + 1}"
        item = _object(item, label, {"id", "from_bus", "to_bus", "r", "x", "b"},
                       {"id", "from_bus", "to_bus", "r", "x"})
        parsed = {key: (_identifier(value, f"{label} {key}") if key in {"id", "from_bus", "to_bus"}
                        else _number(value, f"{label} {key}")) for key, value in item.items()}
        branches.append(BranchData(**parsed))
    config_data = _object(data["solver"], "solver", {
        "method", "tolerance", "max_iterations", "enforce_q_limits",
    }, set())
    config_values = dict(config_data)
    if "method" in config_values:
        _text(config_values["method"], "solver method")
    if "tolerance" in config_values:
        config_values["tolerance"] = _number(config_values["tolerance"], "solver tolerance")
    if "max_iterations" in config_values and type(config_values["max_iterations"]) is not int:
        raise ValueError("solver max_iterations must be an integer.")
    if "enforce_q_limits" in config_values and type(config_values["enforce_q_limits"]) is not bool:
        raise ValueError("solver enforce_q_limits must be true or false.")
    matrix_data = data.get("ybus")
    matrix = None
    if matrix_data is not None:
        rows = _list(matrix_data, "ybus")
        parsed_rows = []
        for i, row in enumerate(rows):
            values = []
            for j, pair in enumerate(_list(row, f"ybus row {i + 1}")):
                pair = _list(pair, f"ybus entry ({i + 1}, {j + 1})")
                if len(pair) != 2:
                    raise ValueError(f"ybus entry ({i + 1}, {j + 1}) needs [real, imaginary].")
                values.append(complex(_number(pair[0], "Ybus real component"),
                                      _number(pair[1], "Ybus imaginary component")))
            if len(values) != len(rows):
                raise ValueError("ybus must be a square matrix of [real, imaginary] pairs.")
            parsed_rows.append(values)
        matrix = np.array(parsed_rows, dtype=complex)
    if mode == "direct" and matrix is None:
        raise ValueError("Direct mode requires ybus.")
    if mode == "direct" and branches:
        raise ValueError("Direct mode cannot include branch inputs; use an empty branches array.")
    if mode == "branches" and matrix is not None:
        raise ValueError("Branch mode builds Ybus from branches; set the ybus field to null.")
    case = NetworkCase(buses=buses, mode=mode, branches=branches, ybus=matrix,
                       base_mva=_number(data.get("base_mva"), "base_mva", optional=True),
                       config=SolverConfig(**config_values))
    validate_case(case)
    return case


def case_to_dict(case: NetworkCase) -> dict:
    """Return JSON-compatible original inputs, after validating the case."""
    validate_case(case)
    matrix = None
    if case.mode == "direct":
        matrix = [[[float(value.real), float(value.imag)] for value in row]
                  for row in np.asarray(case.ybus, dtype=complex)]
    return {
        "schema_version": SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "network_mode": case.mode,
        "base_mva": case.base_mva,
        "buses": [asdict(bus) for bus in case.buses],
        "branches": [asdict(branch) for branch in case.branches] if case.mode == "branches" else [],
        "ybus": matrix,
        "solver": asdict(case.config),
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains an invalid nonfinite number: {value}.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains a duplicate field: {key}.")
        result[key] = value
    return result


def load_case(path: str | Path) -> NetworkCase:
    """Load a UTF-8 JSON case; malformed cases raise ValueError or OSError."""
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        data = json.load(stream, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    return case_from_dict(data)


def save_case(case: NetworkCase, path: str | Path) -> None:
    """Validate and serialize before opening the destination for writing."""
    payload = json.dumps(case_to_dict(case), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    Path(path).write_text(payload, encoding="utf-8")


def example_case() -> NetworkCase:
    """Return a fresh lossless three-bus acceptance case with a flat start."""
    return NetworkCase(
        buses=[
            BusData(id="1", name="Reference", type="Slack", v_set=1.0, angle_set=0.0),
            BusData(id="2", name="Load", type="PQ", p=-2.0, q=-0.5,
                    v_guess=1.0, angle_guess=0.0),
            BusData(id="3", name="Generator", type="PV", p=1.0, v_set=1.0,
                    angle_guess=0.0, q_min=-5.0, q_max=5.0, q_load=0.0),
        ],
        mode="branches",
        branches=[
            BranchData(id="L12", from_bus="1", to_bus="2", r=0.0, x=0.1),
            BranchData(id="L13", from_bus="1", to_bus="3", r=0.0, x=0.4),
            BranchData(id="L23", from_bus="2", to_bus="3", r=0.0, x=0.2),
        ],
        config=SolverConfig(),
    )
