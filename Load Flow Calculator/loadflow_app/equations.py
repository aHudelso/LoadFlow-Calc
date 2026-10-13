"""Power equations and the reduced polar Newton Jacobian (absolute Δ|V|)."""

from __future__ import annotations

import numpy as np


def calculated_injections(ybus: np.ndarray, voltages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        power = voltages * np.conjugate(ybus @ voltages)
    return power.real.copy(), power.imag.copy()


def equation_indices(effective_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([i for i, typ in enumerate(effective_types) if typ != "Slack"], dtype=int),
            np.asarray([i for i, typ in enumerate(effective_types) if typ == "PQ"], dtype=int))


def equation_labels(buses, effective_types) -> list[str]:
    p_indices, q_indices = equation_indices(effective_types)
    return ([f"P at Bus {buses[i].id}" for i in p_indices]
            + [f"Q at Bus {buses[i].id}" for i in q_indices])


def mismatch(buses, effective_types, q_targets, p, q) -> np.ndarray:
    p_indices, q_indices = equation_indices(effective_types)
    return np.asarray([buses[i].p - p[i] for i in p_indices]
                      + [q_targets[i] - q[i] for i in q_indices], dtype=float)


def jacobian(ybus: np.ndarray, voltages: np.ndarray, effective_types: list[str]) -> np.ndarray:
    """Derivative of calculated [P(non-slack), Q(PQ)] in unknown order.

    Rows and columns use bus-table order. This implementation does not assume
    reciprocity/symmetry of the supplied admittance matrix.
    """
    p_indices, q_indices = equation_indices(effective_types)
    p, q = calculated_injections(ybus, voltages)
    v = np.abs(voltages)
    angles = np.angle(voltages)
    difference = angles[:, None] - angles[None, :]
    g, b = ybus.real, ybus.imag
    sine, cosine = np.sin(difference), np.cos(difference)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = v[:, None] * v[None, :] * (g * sine - b * cosine)
        n = v[:, None] * (g * cosine + b * sine)
        m = -v[:, None] * v[None, :] * (g * cosine + b * sine)
        l = v[:, None] * (g * sine - b * cosine)
        np.fill_diagonal(h, -q - b.diagonal() * v * v)
        np.fill_diagonal(n, p / v + g.diagonal() * v)
        np.fill_diagonal(m, p - g.diagonal() * v * v)
        np.fill_diagonal(l, q / v - b.diagonal() * v)
    return np.block([[h[np.ix_(p_indices, p_indices)], n[np.ix_(p_indices, q_indices)]],
                     [m[np.ix_(q_indices, p_indices)], l[np.ix_(q_indices, q_indices)]]])
