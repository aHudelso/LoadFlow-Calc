"""One complete, transactional numerical update for each implemented method.

The input state is never mutated. The controller commits a returned candidate
only after checking the complete state and its recalculated power equations.
"""

from __future__ import annotations

import numpy as np

from .equations import calculated_injections, equation_indices, equation_labels, jacobian, mismatch
from .models import NetworkCase, NumericalError, SolverState
from .network import unknown_labels


def validate_state(case: NetworkCase, state: SolverState) -> None:
    voltages = state.voltages
    if not np.all(np.isfinite(voltages)) or not np.all(np.isfinite(state.angles)):
        raise NumericalError("The attempted update produced nonfinite voltage or angle values.")
    magnitudes = np.abs(voltages)
    if not np.all(np.isfinite(magnitudes)) or np.any(magnitudes <= 0):
        raise NumericalError("Every voltage magnitude must remain positive and finite.")
    for i, (bus, typ) in enumerate(zip(case.buses, state.effective_types)):
        if typ == "Slack":
            expected = bus.v_set * np.exp(1j * np.deg2rad(float(bus.angle_set)))
            if not np.isclose(voltages[i], expected, rtol=1e-12, atol=1e-12):
                raise NumericalError(f"Bus {bus.id}: the fixed Slack voltage was not preserved.")
        elif typ == "PV":
            if not np.isclose(magnitudes[i], bus.v_set, rtol=1e-12, atol=1e-12):
                raise NumericalError(f"Bus {bus.id}: the effective PV voltage magnitude was not preserved.")


def newton_step(case: NetworkCase, ybus: np.ndarray, state: SolverState):
    p, q = calculated_injections(ybus, state.voltages)
    residual = mismatch(case.buses, state.effective_types, state.q_targets, p, q)
    matrix = jacobian(ybus, state.voltages, state.effective_types)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(residual)):
        raise NumericalError("The Newton Jacobian or residual contains nonfinite values.")
    try:
        correction = np.linalg.solve(matrix, residual)
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            "The Newton Jacobian is singular: the present case or voltage guesses do not "
            "permit this Newton step. Check the network, schedules, and initial guesses."
        ) from exc
    if not np.all(np.isfinite(correction)):
        raise NumericalError("The Newton solve produced a nonfinite correction; the last valid state was retained.")
    p_indices, q_indices = equation_indices(state.effective_types)
    angles = state.angles.copy()
    magnitudes = np.abs(state.voltages)
    with np.errstate(over="ignore", invalid="ignore"):
        angles[p_indices] += correction[:len(p_indices)]
        magnitudes[q_indices] += correction[len(p_indices):]
    for i in q_indices:
        if not np.isfinite(magnitudes[i]) or magnitudes[i] <= 0:
            raise NumericalError(
                f"Bus {case.buses[i].id}: Newton proposed a nonpositive or nonfinite PQ magnitude "
                f"({magnitudes[i]} pu). No damping was applied; the last valid state was retained."
            )
    for i, bus in enumerate(case.buses):
        if state.effective_types[i] == "PV":
            magnitudes[i] = bus.v_set
    with np.errstate(over="ignore", invalid="ignore"):
        voltages = magnitudes * np.exp(1j * angles)
    for i, typ in enumerate(state.effective_types):
        if typ == "Slack":
            voltages[i] = state.voltages[i]
            angles[i] = state.angles[i]
    details = {
        "source_iteration": state.iteration,
        "residual": residual.copy(),
        "jacobian": matrix.copy(),
        "correction": correction.copy(),
        "equation_labels": equation_labels(case.buses, state.effective_types),
        "unknown_labels": unknown_labels(case.buses, state.effective_types),
    }
    return voltages, angles, details


def gauss_seidel_step(case: NetworkCase, ybus: np.ndarray, state: SolverState):
    voltages = state.voltages.copy()
    angles = state.angles.copy()
    pv_q_used = {}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i, (bus, typ) in enumerate(zip(case.buses, state.effective_types)):
            if typ == "Slack":
                continue
            old = voltages[i]
            if old == 0 or not np.isfinite(old):
                raise NumericalError(f"Bus {bus.id}: Gauss-Seidel cannot divide by a zero or nonfinite voltage.")
            if ybus[i, i] == 0 or not np.isfinite(ybus[i, i]):
                raise NumericalError(f"Bus {bus.id}: Gauss-Seidel requires a usable nonzero diagonal admittance.")
            if typ == "PV":
                # Includes the diagonal and uses the latest available values.
                q_used = float((old * np.conjugate(ybus[i, :] @ voltages)).imag)
                pv_q_used[bus.id] = q_used
            else:
                q_used = state.q_targets[i]
            off_diagonal = ybus[i, :i] @ voltages[:i] + ybus[i, i + 1:] @ voltages[i + 1:]
            provisional = ((bus.p - 1j * q_used) / np.conjugate(old) - off_diagonal) / ybus[i, i]
            magnitude = abs(provisional)
            if not np.isfinite(magnitude) or magnitude == 0:
                raise NumericalError(
                    f"Bus {bus.id}: Gauss-Seidel produced a zero or nonfinite provisional voltage "
                    "magnitude. The incomplete sweep was discarded."
                )
            if typ == "PV":
                provisional = bus.v_set * (provisional / magnitude)
            voltages[i] = provisional
            # Preserve continuous angles rather than snapping to [-pi, pi].
            angles[i] += np.angle(provisional) - np.angle(old)
            change = angles[i] - state.angles[i]
            if change > np.pi:
                angles[i] -= 2 * np.pi
            elif change < -np.pi:
                angles[i] += 2 * np.pi
    p, q = calculated_injections(ybus, state.voltages)
    details = {
        "source_iteration": state.iteration,
        "residual": mismatch(case.buses, state.effective_types, state.q_targets, p, q),
        "equation_labels": equation_labels(case.buses, state.effective_types),
        "unknown_labels": unknown_labels(case.buses, state.effective_types),
        "pv_q_used": pv_q_used,
    }
    return voltages, angles, details
