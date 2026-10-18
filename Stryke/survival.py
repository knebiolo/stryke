# -*- coding: utf-8 -*-
"""Turbine blade-strike and impingement survival as pure, seedable functions.

What this file is for
---------------------
The same survival equations that ``Stryke.stryke.simulation.Kaplan``, ``Propeller`` and ``Francis`` use (Franke et al. 1997 blade-strike
model, with the USFWS strike constant of 0.2), but with no simulation state, no global random generator and no pandas, so that other
software (AloDyn, the downstream migration model) can call them and get the same numbers as Stryke. Units follow Stryke: feet, cubic feet
per second, revolutions per minute, and a fish length in the same length unit as the runner diameter ``D``.

Why this exists: Stryke's methods draw the strike position on the blade (``rR``, uniform from 0.3 to 1.0 for a Kaplan turbine) from NumPy's
global generator. A model that must be reproducible needs either a seeded generator (``rng``) or the expected survival with the strike
position integrated out (``kaplan_survival_expected``). The default behaviour of ``Stryke.stryke`` is unchanged by this file; the tests pin
these functions to the original methods.

Hostile-reviewer notes: the Franke et al. strike probability is a model of a fish meeting a blade, not a measurement; it is not bounded
by one, so it is clipped to [0, 1]; the strike constant (0.2) and the radial-position range (0.3 to 1.0, Deng et al.) are inherited choices;
the Francis and Propeller forms are carried over as written in ``stryke.py`` (including its corrections dated 2/5/20).
"""

from __future__ import annotations

import numpy as np

G_FT_S2 = 32.2          # gravitational acceleration used by Stryke (ft/s^2)
STRIKE_CONSTANT = 0.2   # USFWS lambda
RR_LOW, RR_HIGH = 0.3, 1.0   # range of the radial strike position on a Kaplan blade


def _positive(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return array


def _non_negative_length(length) -> np.ndarray:
    array = np.asarray(length, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"length must be finite and not negative; got {length!r}")
    return array


def _coefficients(H, RPM, D, Q):
    omega = _positive("RPM", RPM) * (2.0 * np.pi / 60.0)
    D = _positive("D", D)
    Ewd = (G_FT_S2 * _positive("H", H)) / (omega * D) ** 2
    Qwd = _positive("Q", Q) / (omega * D ** 3)
    return omega, D, Ewd, Qwd


def kaplan_survival(length, H, RPM, D, Q, ada, N, lam=STRIKE_CONSTANT, rR=None, rng=None):
    """Probability that a fish of ``length`` survives blade strike in a Kaplan turbine (Franke et al. 1997).

    ``rR`` is the radial strike position (0 to 1). If it is None it is drawn U(0.3, 1.0) once per fish from ``rng``, which must then be a
    ``numpy.random.Generator``; with neither, a ValueError is raised (no silent global state). Returns an array shaped like ``length``.
    """
    length = _non_negative_length(length)
    if rR is None:
        if rng is None:
            raise ValueError("give rR, or a numpy Generator as rng to draw it")
        rR = rng.uniform(RR_LOW, RR_HIGH, size=length.shape)
    rR = np.asarray(rR, dtype=float)
    if np.any(rR <= 0) or np.any(rR > 1) or not np.all(np.isfinite(rR)):
        raise ValueError(f"rR must be in (0, 1]; got {rR!r}")
    omega, D, Ewd, Qwd = _coefficients(H, RPM, D, Q)
    alpha = np.arctan((np.pi * _positive("ada", ada) * Ewd) / (2.0 * Qwd * rR))
    p_strike = lam * (_positive("N", N) * length / D) * (np.cos(alpha) / (8.0 * Qwd) + np.sin(alpha) / (np.pi * rR))
    return np.clip(1.0 - p_strike, 0.0, 1.0)


def kaplan_survival_expected(length, H, RPM, D, Q, ada, N, lam=STRIKE_CONSTANT, nodes=32):
    """Mean Kaplan survival with the strike position integrated out: the average of ``kaplan_survival`` over rR uniform on [0.3, 1.0].

    Gauss-Legendre quadrature with ``nodes`` points (the clip at 0 and 1 is inside the average, as it is when rR is drawn), so this is the
    expected value of Stryke's stochastic result without any random draw.
    """
    if int(nodes) != nodes or nodes < 2:
        raise ValueError(f"nodes must be a whole number of at least 2; got {nodes!r}")
    points, weights = np.polynomial.legendre.leggauss(int(nodes))
    rR = 0.5 * (RR_HIGH - RR_LOW) * points + 0.5 * (RR_HIGH + RR_LOW)
    weights = weights / 2.0                       # mean over the interval
    length = _non_negative_length(length)
    out = np.zeros(length.shape)
    for r, w in zip(rR, weights):
        out = out + w * kaplan_survival(length, H, RPM, D, Q, ada, N, lam, rR=r)
    return out


def propeller_survival(length, H, RPM, D, Q, ada, N, Qopt, lam=STRIKE_CONSTANT, rR=0.75):
    """Blade-strike survival in a propeller turbine, as ``Stryke.stryke.simulation.Propeller`` (fixed rR of 0.75)."""
    length = _non_negative_length(length)
    omega, D, Ewd, Qwd = _coefficients(H, RPM, D, Q)
    Qwd_opt = _positive("Qopt", Qopt) / (omega * D ** 3)
    beta = np.arctan((np.pi / 8.0 * rR) / Qwd_opt)
    alpha = np.arctan((np.pi * Ewd * _positive("ada", ada)) / (2.0 * Qwd * rR) + (np.pi / 8.0 * rR) / Qwd - np.tan(beta))
    p_strike = lam * (_positive("N", N) * length / D) * (np.cos(alpha) / (8.0 * Qwd) + np.sin(alpha) / (np.pi * rR))
    return np.clip(1.0 - p_strike, 0.0, 1.0)


def francis_survival(length, H, RPM, D, Q, ada, N, Qper, iota, D1, D2, B, lam=STRIKE_CONSTANT):
    """Blade-strike survival in a Francis turbine, as ``Stryke.stryke.simulation.Francis``."""
    length = _non_negative_length(length)
    omega, D, Ewd, Qwd = _coefficients(H, RPM, D, Q)
    D1, D2, B = _positive("D1", D1), _positive("D2", D2), _positive("B", B)
    beta = np.arctan((0.707 * (np.pi / 8.0)) / (_positive("iota", iota) * _positive("Qper", Qper) * (D1 / D2) ** 3))
    alpha = np.arctan((2.0 * np.pi * Ewd * _positive("ada", ada)) / Qwd * (B / D1)
                      + (np.pi * 0.707 ** 2) / (2.0 * Qwd) * (B / D1) * (D2 / D1) ** 2
                      - 4.0 * 0.707 * np.tan(beta) * (B / D1) * (D1 / D2))
    p_strike = lam * (_positive("N", N) * length / D) * ((np.sin(alpha) * (B / D1)) / (2.0 * Qwd) + np.cos(alpha) / np.pi)
    return np.clip(1.0 - p_strike, 0.0, 1.0)


def impingement(length, width_ratio, rack_spacing, u_crit, intake_vel):
    """Impingement and screening at a rack, as in ``simulation.node_surv_components``.

    A fish whose width (``length * width_ratio``) exceeds ``rack_spacing`` is blocked by the rack. If it cannot out-swim the intake velocity
    (``u_crit < intake_vel``) it is impinged and counted dead; if it can, it is screened (not entrained). Returns ``(survival, blocked,
    screened)`` as arrays: survival is 0 for an impinged fish and 1 otherwise; ``screened`` is 1 for a blocked fish that escapes.
    """
    length = _non_negative_length(length)
    width_ratio = _positive("width_ratio", width_ratio)
    rack_spacing, intake_vel = _positive("rack_spacing", rack_spacing), _positive("intake_vel", intake_vel)
    u_crit = np.asarray(u_crit, dtype=float)
    if not np.all(np.isfinite(u_crit)) or np.any(u_crit < 0):
        raise ValueError(f"u_crit must be finite and not negative; got {u_crit!r}")
    blocked = length * width_ratio > rack_spacing
    impinged = blocked & (u_crit < intake_vel)
    survival = np.where(impinged, 0.0, 1.0)
    screened = np.where(blocked & ~impinged, 1.0, 0.0)
    return survival, blocked, screened
