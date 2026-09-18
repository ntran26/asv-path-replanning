"""Bluefin vessel plant: the model identified from the July 2026 field logs.

**Replaces the Paper 2 v2 hull**, which is kept as `ship_v2.py` for comparison.
The identification lives in `../bluefin/` (05 part 1): fitted on the 2026-07-02
session, held out on 2026-07-03, with its own acceptance suite and a
reproducible pipeline.  This module **imports** that model rather than copying
it, so when basin session 1 refits the parameters or the structure, the
simulator follows without a second copy drifting out of step.

Same public interface as v2, so no call site changes:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rudder_percent, dt)

`rudder_percent` is in the simulator convention, exactly as v2 took it.

What this wrapper adds to the identified model, and nothing else
---------------------------------------------------------------
1. **Substepping.**  `bluefin/REPORT.md` §9 requires `sub_dt <= 0.05`; a single
   0.1 s RK4 step drifts about 0.5 m over 30 s of manoeuvring.  `update` splits
   any `dt` into equal substeps no longer than `sub_dt`.
2. **Reverse braking.**  The identified model cannot represent reverse thrust:
   `dynamics.thrust` clamps the command at zero and the integrator clips surge
   at zero, because the July logs contain no reverse command at all.  The
   emergency stop needs one, so a negative rpm command is applied by operator
   splitting -- the identified dynamics advance with thrust at zero (which is
   what `dynamics` already does for a negative command), then a braking force
   `REVERSE_THRUST_EFFICIENCY * T(|rpm|)` removes surge momentum, floored at
   zero.  Reverse thrust enters only the surge equation, so every forward-thrust
   trajectory is **bit-identical** to `bluefin/ship_model_v3.ShipModel`, which
   `tests/test_vessel_model.py` asserts.
3. **Cached parameter arrays.**  The reference model rebuilds a dict of numpy
   arrays on every call; at 10 Hz with two substeps that dominated the plant's
   cost.

What it deliberately does **not** do: astern motion.  Surge stays clipped at
zero, so a stop brings the vessel to rest and holds it there.  The emergency
stop latches back to zero thrust at rest, so astern motion should not arise --
but on the real vessel the latch sees speed at the telemetry rate, and full
astern continues for up to one control interval past zero.  That overshoot is
not modelled, and it is one of the things the crash-stop basin test must
measure.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_BLUEFIN_DIR = Path(__file__).resolve().parent.parent / "bluefin"
if str(_BLUEFIN_DIR) not in sys.path:
    # Appended, not prepended: `src/` and the standard library keep precedence,
    # and the assertion below catches any other module named `dynamics` that
    # would otherwise win silently.
    sys.path.append(str(_BLUEFIN_DIR))

import dynamics as dyn  # noqa: E402
import ship_model_v3 as _v3  # noqa: E402

if Path(dyn.__file__).resolve().parent != _BLUEFIN_DIR:
    raise ImportError(
        f"`dynamics` resolved to {dyn.__file__}, not the identified model in "
        f"{_BLUEFIN_DIR}. Another module named `dynamics` is shadowing it.")

# --- Geometry, from the identified model -----------------------------------
VESSEL_LENGTH = dyn.VESSEL_LENGTH
VESSEL_WIDTH = dyn.VESSEL_WIDTH
HULL_MARGIN = dyn.HULL_MARGIN
LIDAR_OFFSET_M = dyn.LIDAR_OFFSET_M
MASS = dyn.MASS
M11 = dyn.M11                           # surge mass including added mass

# --- Actuation --------------------------------------------------------------
MAX_RUD_ANGLE = dyn.MAX_RUD_ANGLE_DEG   # 40 deg

# The rudder rate limit that matters is the **bridge's command limiter**, not a
# servo property: `udp_live_rl.py` ramps the transmitted rudder at 50 %/s, and
# the identified servo rate (2985 deg/s) is effectively unconstrained because
# nothing slower than the bridge limit exists to identify (REPORT §8).
# 50 %/s of a 40 deg rudder is 20 deg/s, the same number v2 carried as a servo
# rate -- which is why `constants.KAPPA_DELTA` does not move.
COMMAND_RATE_PCT_S = 50.0
MAX_RUD_RATE_DPS = COMMAND_RATE_PCT_S / 100.0 * MAX_RUD_ANGLE     # 20 deg/s

# Reverse thrust as a fraction of forward thrust at the same command magnitude.
#
# TODO(05): **unmeasured, and it decides whether the emergency stop meets 03a's
# T9 on the water.**  No July log contains a reverse command.  Fixed-pitch
# propellers typically deliver 50-70 % astern; 0.5 is the conservative end of
# that.  In simulation (0.1 s control step) the stop meets T9 for any value
# >= 0.19.  At the deployment's 2 Hz it needs >= 0.26, and if thrust shares
# the rudder's 0.73 s effective delay, >= 0.56 -- so 0.5 would fail there.
# A crash-stop run in basin session 1 settles both numbers.
REVERSE_THRUST_EFFICIENCY = 0.5

SUB_DT = 0.05                           # REPORT §9: required <= 0.05

# --- Identified parameters, re-exported ------------------------------------
IDENTIFIED: Dict[str, float] = dict(_v3.IDENTIFIED)
BOOTSTRAP = _v3.BOOTSTRAP
sample_params = _v3.sample_params


def forward_thrust(rpm: float, params: Optional[Dict[str, float]] = None,
                   u: float = 0.0) -> float:
    """Identified forward thrust at a command, newtons (zero for rpm <= 0)."""
    p = IDENTIFIED if params is None else params
    pa = {k: np.array([float(v)]) for k, v in p.items()}
    return float(dyn.thrust(np.array([float(rpm)]), np.array([float(u)]), pa)[0])


def braking_thrust(rpm: float, params: Optional[Dict[str, float]] = None,
                   efficiency: float = REVERSE_THRUST_EFFICIENCY) -> float:
    """Magnitude of reverse thrust at a negative command, newtons.

    Mirrors the forward law at the same command magnitude, scaled by the
    reverse efficiency.  The square-law RPM exponent is *assumed* in the forward
    law too (REPORT §8, `T12` anchored at one operating point), so full astern
    at -24 rpm-units inherits that extrapolation.
    """
    if rpm >= 0.0:
        return 0.0
    return float(efficiency) * forward_thrust(abs(rpm), params, u=0.0)


def steady_speed(rpm: float, params: Optional[Dict[str, float]] = None) -> float:
    """Straight-line steady surge at a command, from the identified balance.

    Solved from thrust = drag with sway and yaw at zero.  With the identified
    `t_boost = 0` this reduces to `(rpm/12) * sqrt(T12 / X_uu)` = 1.116 m/s at
    12 rpm-units, but the bisection keeps it correct if a refit brings the
    low-speed boost back.
    """
    p = IDENTIFIED if params is None else params
    if rpm <= 0.0:
        return 0.0
    lo, hi = 0.0, float(dyn.MAX_SURGE_SPEED)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        net = forward_thrust(rpm, p, u=mid) - float(p["X_uu"]) * mid * mid
        lo, hi = (mid, hi) if net > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


class ShipModel:
    """The identified 3-DOF model with substepping and reverse braking."""

    def __init__(self, params: Optional[Dict[str, float]] = None, *,
                 sub_dt: float = SUB_DT,
                 reverse_efficiency: float = REVERSE_THRUST_EFFICIENCY) -> None:
        self.sub_dt = float(sub_dt)
        self.reverse_efficiency = float(reverse_efficiency)
        self.set_params(IDENTIFIED if params is None else params)
        self.reset()

    # -- parameters ---------------------------------------------------------
    def set_params(self, params: Dict[str, float]) -> None:
        """Adopt a parameter set -- a domain-randomisation draw, typically.

        Clears the actuator delay line, because its length is set by
        `rud_delay` and a stale buffer would carry the previous vessel's delay
        into the next episode.
        """
        self.p = {k: float(v) for k, v in params.items()}
        self._pa = {k: np.array([v]) for k, v in self.p.items()}
        self._cmd_buf = deque()
        self._buf_dt = None

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self._s = np.zeros((7, 1))
        self._cmd_buf = deque()
        self._buf_dt = None
        self.last_braking_force = 0.0
        self.last_astern_impulse = 0.0

    @property
    def u(self) -> float:
        """Surge velocity [m/s]."""
        return float(self._s[0, 0])

    @property
    def v(self) -> float:
        """Sway velocity [m/s]."""
        return float(self._s[1, 0])

    @property
    def yaw_rate(self) -> float:
        """Yaw rate [rad/s]."""
        return float(self._s[2, 0])

    @property
    def heading_deg(self) -> float:
        return math.degrees(float(self._s[3, 0])) % 360.0

    @property
    def rudder_deg(self) -> float:
        return math.degrees(float(self._s[4, 0]))

    def state_dict(self) -> Dict[str, float]:
        return {
            "u_body_mps": self.u,
            "v_body_mps": self.v,
            "yaw_rate_radps": self.yaw_rate,
            "yaw_rate_degps": math.degrees(self.yaw_rate),
            "heading_deg": self.heading_deg,
            "rudder_deg": self.rudder_deg,
            "x_m": float(self._s[5, 0]),
            "y_m": float(self._s[6, 0]),
            "speed_mps": math.hypot(self.u, self.v),
        }

    # -- actuator delay -----------------------------------------------------
    def _delayed_command(self, delta_cmd: float, dt: float) -> float:
        """Pure transport delay as a FIFO sized by dt.

        Copied from `ship_model_v3.ShipModel._delayed_command` line for line,
        including its initial-fill behaviour, so that trajectories match the
        reference model exactly.
        """
        n = int(round(self.p["rud_delay"] / max(dt, 1e-6)))
        if self._buf_dt != dt:
            self._cmd_buf = deque([delta_cmd] * max(n, 0), maxlen=max(n, 1))
            self._buf_dt = dt
        if n <= 0:
            return delta_cmd
        out = self._cmd_buf[0] if len(self._cmd_buf) == self._cmd_buf.maxlen else delta_cmd
        self._cmd_buf.append(delta_cmd)
        return out

    # -- integration --------------------------------------------------------
    def update(self, rpm: float, rud: float, dt: float) -> Tuple[float, float, float, float]:
        """Advance `dt` seconds; return (dx, dy, heading_deg, yaw_rate_degps)."""
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        n = max(1, int(math.ceil(float(dt) / self.sub_dt - 1e-9)))
        h = float(dt) / n
        x_prev, y_prev = float(self._s[5, 0]), float(self._s[6, 0])

        # simulator percent -> transmitted convention -> angle (see dynamics.py)
        delta_cmd = float(dyn.cmd_percent_to_angle(np.array([-float(rud)]))[0])
        rpm_arr = np.array([float(rpm)])
        self.last_braking_force = braking_thrust(float(rpm), self.p,
                                                 self.reverse_efficiency)
        # Impulse of the reverse thrust applied while surge was already zero:
        # what the clip at zero throws away, and what would drive the real
        # vessel astern.  `impulse / M11` is the astern speed it implies.
        self.last_astern_impulse = 0.0

        for _ in range(n):
            delayed = self._delayed_command(delta_cmd, h)
            self._s = dyn.rk4_step(self._s, rpm_arr, np.array([delayed]), self._pa, h)
            force = self.last_braking_force
            if force > 0.0:
                u = float(self._s[0, 0])
                decel = force / M11
                if u > 0.0:
                    self._s[0, 0] = max(0.0, u - decel * h)
                    if u < decel * h:
                        self.last_astern_impulse += force * (h - u / decel)
                else:
                    self.last_astern_impulse += force * h

        return (float(self._s[5, 0]) - x_prev,
                float(self._s[6, 0]) - y_prev,
                self.heading_deg,
                math.degrees(float(self._s[2, 0])))
