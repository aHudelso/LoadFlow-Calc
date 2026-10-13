"""Data models shared by the numerical engine, desktop UI, and file formats.

All powers and voltage magnitudes are per unit. Bus input angles are degrees;
solver state, snapshot angles, and angle corrections are unwrapped radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BusData:
    id: str
    name: str = ""
    type: str = "PQ"
    p: float | None = None
    q: float | None = None
    v_set: float | None = None
    angle_set: float | None = None
    v_guess: float = 1.0
    angle_guess: float = 0.0
    q_min: float | None = None
    q_max: float | None = None
    q_load: float = 0.0
    shunt_g: float = 0.0
    shunt_b: float = 0.0


@dataclass
class BranchData:
    id: str
    from_bus: str
    to_bus: str
    r: float
    x: float
    b: float = 0.0


@dataclass
class SolverConfig:
    method: str = "Newton-Raphson"
    tolerance: float = 1e-6
    max_iterations: int = 100
    enforce_q_limits: bool = False


@dataclass
class NetworkCase:
    buses: list[BusData]
    mode: str = "branches"
    branches: list[BranchData] = field(default_factory=list)
    ybus: np.ndarray | None = None
    base_mva: float | None = None
    config: SolverConfig = field(default_factory=SolverConfig)


@dataclass
class SolverState:
    voltages: np.ndarray
    angles: np.ndarray
    effective_types: list[str]
    q_targets: list[float | None]
    iteration: int = 0
    binding_limits: list[str | None] = field(default_factory=list)


@dataclass
class IterationSnapshot:
    snapshot_id: int
    iteration: int
    phase: str
    status: str
    voltages: np.ndarray
    angles: np.ndarray
    effective_types: list[str]
    p_calculated: np.ndarray
    q_calculated: np.ndarray
    p_targets: list[float | None]
    q_targets: list[float | None]
    delta_p: list[float | None]
    delta_q: list[float | None]
    residual: np.ndarray
    max_mismatch: float
    max_mismatch_label: str
    delta_magnitude: np.ndarray | None = None
    delta_angle: np.ndarray | None = None
    binding_limits: list[str | None] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    event: str | None = None


@dataclass
class RunHistory:
    run_id: str
    case: NetworkCase
    snapshots: list[IterationSnapshot] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class ValidationError(ValueError):
    """A case input does not satisfy the supported model's requirements."""


class NumericalError(RuntimeError):
    """An attempted update could not produce a complete valid numerical state."""
