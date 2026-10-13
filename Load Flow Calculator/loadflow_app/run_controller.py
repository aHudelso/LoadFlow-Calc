"""Run lifecycle, one-step accounting, immutable numerical history, Q limits."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import numpy as np

from .equations import calculated_injections, equation_labels, mismatch
from .models import IterationSnapshot, NetworkCase, NumericalError, RunHistory, SolverState, ValidationError
from .network import initial_voltages, positive_integer, validate_case
from .solvers import gauss_seidel_step, newton_step, validate_state


TERMINAL_STATUSES = {"Converged", "Maximum iterations reached", "Numerical failure"}


class RunController:
    """Own a validated, copied case and advance it only through :meth:`step`.

    A constraint switch adds a snapshot but consumes no iteration. Failed steps
    consume no iterations and leave the previous state and snapshot untouched.
    GUI batches should call step through Tk.after, allowing stop between steps.
    """

    def __init__(self, case: NetworkCase):
        self.case = deepcopy(case)
        self.ybus = validate_case(self.case)
        self.history: RunHistory
        self.state: SolverState
        self.status = "Ready"
        self._stop_requested = False
        self.reset()

    @property
    def current(self) -> IterationSnapshot:
        return self.history.snapshots[-1]

    @property
    def can_step(self) -> bool:
        return self.status not in TERMINAL_STATUSES

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.case.config.max_iterations - self.state.iteration)

    def reset(self) -> IterationSnapshot:
        """Restore input guesses/types and start a newly identified history."""
        buses = self.case.buses
        self.state = SolverState(
            voltages=initial_voltages(buses),
            angles=np.deg2rad(np.asarray([bus.angle_set if bus.type == "Slack" else bus.angle_guess for bus in buses], dtype=float)),
            effective_types=[bus.type for bus in buses],
            q_targets=[bus.q if bus.type == "PQ" else None for bus in buses],
            binding_limits=[None for _ in buses],
        )
        self.history = RunHistory(run_id=str(uuid4()), case=deepcopy(self.case))
        self.status = "Ready"
        self._stop_requested = False
        try:
            self._check_candidate(self.state)
        except NumericalError as exc:
            raise ValidationError(f"Initial guesses do not produce a valid numerical state: {exc}") from exc
        self._record_complete_state("initial")
        return self.current

    def _check_candidate(self, candidate: SolverState):
        validate_state(self.case, candidate)
        p, q = calculated_injections(self.ybus, candidate.voltages)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = mismatch(self.case.buses, candidate.effective_types, candidate.q_targets, p, q)
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(residual)):
            raise NumericalError("The attempted state produced nonfinite calculated injections or mismatches.")
        with np.errstate(over="ignore", invalid="ignore"):
            for i, bus in enumerate(self.case.buses):
                if bus.type == "PV" and not np.isfinite(q[i] + bus.q_load):
                    raise NumericalError(f"Bus {bus.id}: generator Q (calculated net Q plus local reactive load) is nonfinite.")

    def _snapshot(self, phase, *, delta_magnitude=None, delta_angle=None, details=None, event=None):
        state = self.state
        p, q = calculated_injections(self.ybus, state.voltages)
        residual = mismatch(self.case.buses, state.effective_types, state.q_targets, p, q)
        labels = equation_labels(self.case.buses, state.effective_types)
        largest = int(np.argmax(np.abs(residual)))
        p_targets = [bus.p if typ != "Slack" else None
                     for bus, typ in zip(self.case.buses, state.effective_types)]
        q_targets = [target if typ == "PQ" else None
                     for target, typ in zip(state.q_targets, state.effective_types)]
        return IterationSnapshot(
            snapshot_id=len(self.history.snapshots), iteration=state.iteration,
            phase=phase, status=self.status, voltages=state.voltages.copy(), angles=state.angles.copy(),
            effective_types=state.effective_types.copy(), p_calculated=p, q_calculated=q,
            p_targets=p_targets, q_targets=q_targets,
            delta_p=[None if target is None else float(target - value) for target, value in zip(p_targets, p)],
            delta_q=[None if target is None else float(target - value) for target, value in zip(q_targets, q)],
            residual=residual.copy(), max_mismatch=float(abs(residual[largest])), max_mismatch_label=labels[largest],
            delta_magnitude=None if delta_magnitude is None else delta_magnitude.copy(),
            delta_angle=None if delta_angle is None else delta_angle.copy(),
            binding_limits=state.binding_limits.copy(), details=deepcopy(details or {}), event=event,
        )

    def _violations(self, snapshot):
        if not self.case.config.enforce_q_limits:
            return []
        violations = []
        tolerance = self.case.config.tolerance
        for i, (bus, typ) in enumerate(zip(self.case.buses, self.state.effective_types)):
            if typ != "PV" or bus.q_min is None or bus.q_max is None:
                continue
            generator_q = snapshot.q_calculated[i] + bus.q_load
            if generator_q < bus.q_min - tolerance:
                violations.append((i, "minimum", bus.q_min, generator_q))
            elif generator_q > bus.q_max + tolerance:
                violations.append((i, "maximum", bus.q_max, generator_q))
        return violations

    def _record_complete_state(self, phase, *, delta_magnitude=None, delta_angle=None, details=None):
        snapshot = self._snapshot(phase, delta_magnitude=delta_magnitude,
                                  delta_angle=delta_angle, details=details)
        inner_converged = snapshot.max_mismatch <= self.case.config.tolerance
        violations = self._violations(snapshot) if inner_converged else []
        if violations:
            # Save the converged old equation set before activating new ones.
            snapshot.status = "Iterating"
            self.history.snapshots.append(snapshot)
            messages = []
            for i, bound, value, required in violations:
                bus = self.case.buses[i]
                self.state.effective_types[i] = "PQ"
                self.state.q_targets[i] = float(value - bus.q_load)
                self.state.binding_limits[i] = bound
                message = (
                    f"Bus {bus.id}: PV to PQ at generator Q {bound} {value:.12g} pu; "
                    f"required QG was {required:.12g} pu. New net Q target is "
                    f"{self.state.q_targets[i]:.12g} pu (QG limit minus local Q load)."
                )
                messages.append(message)
                self.history.events.append({"iteration": self.state.iteration,
                                            "type": "constraint_change", "message": message,
                                            "bus_id": bus.id, "bound": bound,
                                            "generator_q_limit": value, "net_q_target": self.state.q_targets[i]})
            changed = self._snapshot("constraint_change", event="\n".join(messages))
            if changed.max_mismatch <= self.case.config.tolerance:
                self.status = "Converged"
            elif self.remaining_iterations == 0:
                self.status = "Maximum iterations reached"
            else:
                self.status = "Ready" if self.state.iteration == 0 else "Iterating"
            changed.status = self.status
            self.history.snapshots.append(changed)
            return
        if inner_converged:
            self.status = "Converged"
        elif self.remaining_iterations == 0:
            self.status = "Maximum iterations reached"
        else:
            self.status = "Ready" if self.state.iteration == 0 else "Iterating"
        snapshot.status = self.status
        self.history.snapshots.append(snapshot)

    def step(self) -> IterationSnapshot:
        """Perform exactly one complete update unless already terminal/failing."""
        if not self.can_step:
            return self.current
        if self.remaining_iterations == 0:
            self.status = "Maximum iterations reached"
            return self.current
        self._stop_requested = False
        previous = self.state
        attempted_iteration = previous.iteration + 1
        try:
            solver = newton_step if self.case.config.method == "Newton-Raphson" else gauss_seidel_step
            voltages, angles, details = solver(self.case, self.ybus, previous)
            candidate = SolverState(
                voltages=voltages, angles=angles,
                effective_types=previous.effective_types.copy(), q_targets=previous.q_targets.copy(),
                iteration=attempted_iteration, binding_limits=previous.binding_limits.copy(),
            )
            self._check_candidate(candidate)
        except (NumericalError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            self.status = "Numerical failure"
            self.history.events.append({"iteration": previous.iteration,
                                        "attempted_iteration": attempted_iteration,
                                        "type": "numerical_failure", "message": str(exc)})
            return self.current
        delta_magnitude = np.abs(candidate.voltages) - np.abs(previous.voltages)
        delta_angle = candidate.angles - previous.angles
        self.state = candidate
        self._record_complete_state("post_update", delta_magnitude=delta_magnitude,
                                    delta_angle=delta_angle, details=details)
        return self.current

    def run_n(self, count: int) -> IterationSnapshot:
        """Run up to count additional successful updates, stopping early."""
        positive_integer(count, "Batch iteration count")
        self._stop_requested = False
        for _ in range(count):
            if not self.can_step or self._stop_requested:
                break
            self.step()
        if self.can_step:
            self.status = "Paused"
        return self.current

    def run_until_converged(self) -> IterationSnapshot:
        self._stop_requested = False
        while self.can_step and not self._stop_requested:
            self.step()
        return self.current

    def stop(self) -> None:
        """Request that a batch stop between complete updates."""
        self._stop_requested = True
        if self.can_step:
            self.status = "Paused"
            self.history.events.append({"iteration": self.state.iteration,
                                        "type": "stop", "message": "Stopped between complete solver updates."})
