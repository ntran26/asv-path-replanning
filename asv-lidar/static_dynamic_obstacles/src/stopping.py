"""Where a stopping own ship actually goes: the A23 stop test.

A18 let the supervisor stop only when the target would clear *an own ship
stationary where it is now*.  A stopping vessel does not stop where it is.  The
supervisor's latch runs full astern (`S2 = -100`) until the speed estimate falls
below `ESTOP_STOP_SPEED`, and the hull keeps moving forward while it brakes.
With accurate close-range geometry (C15) the stationary test fired in draws the
braking vessel then slid into: 29 stops and 15 stop-then-hit episodes, against
13 and 6 with the biased centroid (F66).

A23 (decided, option 1) asks the physical question instead: the closest the
target's constant-velocity track comes to the own ship **along its braking
path**, then while it lies stopped.

Kept out of `emergency_stop.py`, which the bridge imports and which must stay
free of the simulator; this module needs the identified hull.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

import constants as cfg
import emergency_stop as estop
from ship import ShipModel


# Two ways of losing way.  "latch": the supervisor's full astern (A23).
# "coast": the policy's own slowdown -- at propulsion stage 4 its floor is RPM 0,
# with no reverse -- which is what R-2 and the Rule 8 credit ask about (F68).
_MODES = ("latch", "coast")


@lru_cache(maxsize=1024)
def _profile(u0_cmps: int, mode: str = "latch") -> Tuple[np.ndarray, np.ndarray]:
    if mode not in _MODES:
        raise ValueError(f"unknown stopping mode {mode!r}")
    dt = float(cfg.STOP_TEST_DT_S)
    model = ShipModel()
    model._s[0, 0] = u0_cmps / 100.0
    if mode == "latch":
        rpm, cap = estop.s2_to_rpm(estop.S2_FULL_ASTERN), float(cfg.ESTOP_MAX_BRAKE_S)
    else:
        rpm, cap = float(cfg.POLICY_SLOWDOWN_RPM), float(cfg.SLOWDOWN_TEST_MAX_S)
    t, s = [0.0], [0.0]
    travelled = 0.0
    while model.u > float(cfg.ESTOP_STOP_SPEED) and t[-1] < cap:
        dx, dy, _, _ = model.update(rpm, 0.0, dt)
        travelled += math.hypot(dx, dy)
        t.append(t[-1] + dt)
        s.append(travelled)
    return np.asarray(t), np.asarray(s)


def braking_profile(u0: float, mode: str = "latch") -> Tuple[np.ndarray, np.ndarray]:
    """(time s, distance along the heading m) from surge `u0` until the stop
    speed, on the nominal hull, cached per cm/s -- under the latch's full
    astern, or coasting at the policy's RPM floor.  A vessel already at or below
    the stop speed returns a single point: it is where it is."""
    return _profile(int(round(max(float(u0), 0.0) * 100.0)), mode)


def _cpa_from(r0: np.ndarray, v: np.ndarray) -> float:
    speed_sq = float(v @ v)
    if speed_sq <= 1e-12:
        return float(np.linalg.norm(r0))
    t = max(0.0, -float(r0 @ v) / speed_sq)
    return float(np.linalg.norm(r0 + v * t))


def dcpa_over_stop(rng: float, alpha_deg: float, ct_deg: float, speed_ts: float, u_own: float,
                   mode: str = "latch") -> float:
    """Closest approach of the target to an own ship that stops now (A23).

    Own-ship frame, heading +y.  The target starts at range `rng`, relative
    bearing `alpha_deg`, and runs at `speed_ts` on relative course `ct_deg`.  The
    own ship follows `braking_profile(u_own)` along +y, then lies stopped; the
    result is the minimum centre distance over both phases.  With `u_own` at or
    below the stop speed this is exactly A18's stationary DCPA.
    """
    rng = float(rng)
    if not np.isfinite(rng):
        return rng
    a, c = math.radians(float(alpha_deg)), math.radians(float(ct_deg))
    r0 = rng * np.array([math.sin(a), math.cos(a)])
    v = float(speed_ts) * np.array([math.sin(c), math.cos(c)])
    t, s = braking_profile(u_own, mode)
    if len(t) == 1:
        return _cpa_from(r0, v)
    rel = r0[None, :] + t[:, None] * v[None, :] - np.column_stack([np.zeros_like(s), s])
    during = float(np.min(np.linalg.norm(rel, axis=1)))
    after = _cpa_from(rel[-1], v)
    return min(during, after)
