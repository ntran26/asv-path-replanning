"""Experimental conditional enclosures of the simulator's numerical plant map.

This is an OFFLINE arithmetic prototype, not an online safety filter or a mission
safety certificate.  Its subject is the real-arithmetic interpretation of the
0.05 s numerical update in ``ship.py`` / ``bluefin/dynamics.py``: delayed command,
analytic/clipped servo, midpoint-servo RK4, endpoint velocity clips, then the
immediate reverse-surge split.  It does not validate the continuous vessel ODE,
inter-sample swept motion, binary64/libm implementation error, obstacle occupancy,
or a terminal continuation.  ``mpmath.iv`` is an experimental interval backend;
tests of sampled containment are not a proof of its arithmetic implementation.

Every parameter/state bound is an explicit caller assumption.  In particular,
the simulator's Gaussian parameter jitter and sensor errors have no declared
finite deterministic support.  Bootstrap extrema, covariance and observed RMSE
must not silently become that support.  Parameter intervals are reused at every
step: natural interval arithmetic loses correlations but does not exclude a
fixed admissible realization merely because it is reused.

The caller declares one integer FIFO-delay branch.  ``rud_delay`` is required
for provenance but is NOT used to choose that branch.  The caller must cover all
admitted ``round(rud_delay / 0.05)`` branches separately.  Only decision durations
that are integer multiples of 0.05 s are supported; a different integration step
would resize/reset the simulator FIFO and requires a separate implementation.

Model-expression provenance: ``bluefin/dynamics.py`` derivatives/advance_rudder/
rk4_step and ``src/ship.py`` _delayed_command/update (audited 2026-10-03).
Algebraic simplifications below preserve the declared REAL-arithmetic map:
combine the two u*v yaw moments before interval evaluation, use the interval
square image for v**2, and use the monotone image of z*abs(z) for damping.  These
identities reduce repeated-variable wrapping; they are not binary64 bit-parity
claims and do not tighten any supplied uncertainty bounds.
Reachable-set verification is motivated by Kochdumper et al. (2023),
https://arxiv.org/abs/2210.10691 ; this small interval implementation does not
implement their polynomial-zonotope shield or inherit its guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from mpmath.ctx_iv import MPIntervalContext

from ship import dyn as _dyn


SUBSTEP_S = 0.05
HISTORY_COMMAND_S = 0.5
PARAMETER_NAMES = tuple(_dyn.PARAM_NAMES)
STATE_NAMES = ("x", "y", "heading", "u", "v", "r")
DEFAULT_MAX_INTERNAL_MAGNITUDE = 1e12  # resource guard, NOT a physical bound
MAX_DELAY_STEPS = 4096               # resource guard; rejection is inconclusive
MAX_SUBSTEPS_PER_CALL = 10000
MAX_HISTORY_COMMANDS = 2000


class EnclosureFailure(RuntimeError):
    """No enclosure returned: invalid arithmetic or a resource limit was reached."""


@dataclass(frozen=True)
class IntervalBounds:
    """Finite ordered bounds; all angles and angular rates use radians."""

    lo: float
    hi: float

    def __post_init__(self):
        lo, hi = float(self.lo), float(self.hi)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("interval bounds must be finite and ordered")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)


class ConditionalPlantTube:
    """Propagate a supplied state box under exact, held scalar commands.

    ``command_history`` is chronological ``(rudder_percent, signed_rpm)`` pairs,
    each held for 0.5 s.  It reconstructs ONLY the servo and FIFO before the
    supplied body state; it never advances that body's x/y/heading/u/v/r.  The
    ``servo`` argument denotes the servo *before* this history, or the current
    servo when history is empty.  With no history the FIFO is uninitialized and
    its first fill contains the first future command, exactly as in ShipModel.

    Percentage commands use the simulator convention; positive percentages map
    to negative rudder angles.  RPM is a signed simulator command unit, not shaft
    RPM.  Reverse efficiency is the simulator's fixed 0.5, not a physical bound.
    Any exception from ``advance`` leaves the prior state and FIFO unchanged.
    """

    def __init__(self, parameters: Mapping[str, IntervalBounds],
                 state: Mapping[str, IntervalBounds], delay_steps: int,
                 servo: IntervalBounds = IntervalBounds(0.0, 0.0),
                 command_history: Sequence[tuple[float, float]] | None = None,
                 *, max_internal_magnitude: float = DEFAULT_MAX_INTERNAL_MAGNITUDE):
        self._ctx = MPIntervalContext()
        self._ctx.dps = 50  # private context; never change process-global iv/mp
        self._limit = float(max_internal_magnitude)
        if not math.isfinite(self._limit) or self._limit <= 0.0:
            raise ValueError("max_internal_magnitude must be finite and positive")
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, (int, np.integer)):
            raise ValueError("delay_steps must be an explicit nonnegative integer branch")
        if delay_steps < 0:
            raise ValueError("delay_steps must be nonnegative")
        if delay_steps > MAX_DELAY_STEPS:
            raise EnclosureFailure("delay branch exceeds FIFO resource limit")
        self.delay_steps = int(delay_steps)
        self._validate_keys(parameters, PARAMETER_NAMES, "parameters")
        self._validate_keys(state, STATE_NAMES, "state")
        self._p = {key: self._interval(parameters[key]) for key in PARAMETER_NAMES}
        if parameters["rud_rate"].lo < 0.0:
            raise ValueError("rud_rate must be nonnegative for this servo branch")
        if parameters["rud_delay"].lo < 0.0:
            raise ValueError("rud_delay must be nonnegative")
        self._state = [self._interval(state[key]) for key in ("u", "v", "r", "heading")]
        self._state += [self._interval(servo), self._interval(state["x"]), self._interval(state["y"])]
        self._pending = None
        self._h = self._point(SUBSTEP_S)
        self._rud_limit = self._point(float(np.deg2rad(_dyn.MAX_RUD_ANGLE_DEG)))
        self._deg_to_rad = self._point(float(np.deg2rad(1.0)))
        self.diagnostics = {
            "scope": "conditional_real_arithmetic_numerical_map",
            "backend": "experimental_mpmath.iv_private_context",
            "mission_safety_certificate": False,
            "binary64_roundoff_validated": False,
            "continuous_ode_validated": False,
            "swept_geometry_validated": False,
            "declared_delay_steps": self.delay_steps,
            "rud_delay_used_to_select_branch": False,
            "parameter_dependence": "fixed_realizations_enclosed_without_correlation_tracking",
            "max_internal_magnitude": self._limit,
            "history_commands": 0,
            "advanced_substeps": 0,
            "last_failure": None,
        }
        history = [] if command_history is None else list(command_history)
        if len(history) > MAX_HISTORY_COMMANDS:
            raise EnclosureFailure("actuator history exceeds resource limit")
        for rudder, rpm in history:
            command = self._command(rudder)
            self._finite_command(rpm, "signed_rpm")  # unused by the rudder servo
            for _ in range(10):
                delayed, self._pending = self._delay(command, self._pending)
                self._state[4] = self._servo(self._state[4], delayed)
        self.diagnostics["history_commands"] = len(history)
        self._guard(self._state)

    @staticmethod
    def _validate_keys(values, expected, label):
        missing, extra = set(expected) - set(values), set(values) - set(expected)
        if missing or extra:
            raise ValueError(f"{label} keys: missing={sorted(missing)}, unexpected={sorted(extra)}")

    def _point(self, value):
        return self._ctx.mpf(float(value))

    def _interval(self, bound):
        if not isinstance(bound, IntervalBounds):
            raise TypeError("state, servo and parameter values must be IntervalBounds")
        result = self._ctx.mpf([bound.lo, bound.hi])
        self._guard([result])
        return result

    def _guard(self, values):
        for value in values:
            try:
                lo, hi = float(value.a), float(value.b)
            except (ValueError, OverflowError, TypeError) as error:
                raise EnclosureFailure("nonfinite interval arithmetic") from error
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise EnclosureFailure("nonfinite or unordered interval arithmetic")
            if max(abs(lo), abs(hi)) > self._limit:
                raise EnclosureFailure("internal magnitude resource limit exceeded")

    def _maximum(self, left, right):
        return self._ctx.mpf([max(left.a, right.a), max(left.b, right.b)])

    def _minimum(self, left, right):
        return self._ctx.mpf([min(left.a, right.a), min(left.b, right.b)])

    def _clip(self, value, low, high):
        return self._minimum(self._maximum(value, low), high)

    def _divide(self, numerator, denominator):
        if denominator.a <= 0 <= denominator.b:
            raise EnclosureFailure("division interval includes zero")
        result = numerator / denominator
        self._guard([result])
        return result

    @staticmethod
    def _finite_command(value, name):
        value = float(value)
        if not math.isfinite(value):
            raise EnclosureFailure(f"{name} must be finite")
        return value

    def _command(self, rudder_percent):
        rudder = self._finite_command(rudder_percent, "rudder_percent")
        rudder = min(100.0, max(-100.0, rudder))
        return -self._point(rudder) / self._point(100.0) * self._rud_limit

    def _delay(self, command, pending):
        if self.delay_steps == 0:
            return command, []
        if pending is None:
            pending = [command] * self.delay_steps
        return pending[0], pending[1:] + [command]

    def _servo(self, servo, command):
        """Enclose the servo image by its monotone box extrema.

        For a in [0,1] and L >= 0, the unclipped map is
        g(s,c,a,L) = median(s-L, (1-a)*s+a*c, s+L), hence is nondecreasing
        in s and c.  For fixed s,c it is also
        s + sign(c-s)*min(a*abs(c-s), L), monotone in a and L with the
        direction determined by sign(c-s).  Thus all box extrema occur among
        endpoint combinations.  The final angle clip preserves these extrema.

        Evaluate the 16 corners with interval endpoint points, without a float
        conversion.  This is an analytic image enclosure, not sampled tightening.
        It removes the artificial repeated-s dependence of ``s + (c-s)*a``;
        fixed parameter dependence across different times is still discarded.
        """
        zero, one = self._point(0.0), self._point(1.0)
        tau = self._maximum(self._p["rud_tau"], self._point(1e-4))
        rate = self._p["rud_rate"] * self._deg_to_rad
        # The analytic response lies in [0,1], independently of roundoff width.
        fraction = self._clip(1 - self._ctx.exp(-self._divide(self._h, tau)), zero, one)
        command = self._clip(command, -self._rud_limit, self._rud_limit)
        limit = rate * self._h
        corners = []
        for s in (servo.a, servo.b):
            for c in (command.a, command.b):
                for a in (fraction.a, fraction.b):
                    for cap in (limit.a, limit.b):
                        step = self._clip((c - s) * a, -cap, cap)
                        corners.append(self._clip(s + step, -self._rud_limit, self._rud_limit))
        result = self._ctx.mpf([min(v.a for v in corners), max(v.b for v in corners)])
        self._guard([result])
        return result

    def _thrust(self, rpm, surge):
        zero = self._point(0.0)
        effective_u = self._maximum(surge, zero)
        boost_scale = self._maximum(self._p["u_boost"], self._point(1e-6))
        boost = 1 + self._p["t_boost"] * self._ctx.exp(-self._divide(effective_u, boost_scale))
        ratio = self._maximum(rpm, zero) / 12
        result = self._p["T12"] * ratio * ratio * boost
        self._guard([result])
        return result

    def _signed_square(self, value):
        """Exact real interval image of z*abs(z), up to backend enclosure.

        The function is continuous and nondecreasing: its derivative is
        2*abs(z) away from zero and it crosses zero without a jump.  Therefore
        its minimum and maximum are the lower and upper endpoint images, even
        for an asymmetric interval straddling zero.  Keep endpoint points in
        interval arithmetic; a nearest-float conversion could exclude an end.
        """
        self._guard([value])
        low = value.a * abs(value.a)
        high = value.b * abs(value.b)
        result = self._ctx.mpf([low.a, high.b])
        self._guard([result])
        return result

    def _rhs(self, state, rpm):
        self._guard(state)
        u, v, r, psi, delta, _, _ = state
        p, c = self._p, self._ctx
        zero = self._point(0.0)
        u_eff = self._maximum(u, zero)
        race = p["k_race"] * c.sqrt(self._maximum(rpm, zero) / 12)
        u_r = self._maximum(self._point(_dyn.MIN_FLOW_SPEED), (1 - _dyn.W_R) * u_eff + race)
        v_r = v + _dyn.L_R * r
        alpha = delta - c.atan2(v_r, u_r)
        f_n = (0.5 * _dyn.RHO * _dyn.A_R * _dyn.F_ALPHA) * u_r * u_r * c.sin(alpha)
        cos_d = c.cos(delta)
        x_damp = -p["X_uu"] * u_eff * abs(u_eff) - p["X_vv"] * v ** 2 - p["X_rr"] * (_dyn.L * r) ** 2
        x_rud = -p["X_delta"] * abs(f_n) * abs(c.sin(delta))
        x_total = self._thrust(rpm, u) + x_damp + x_rud + _dyn.M22 * v * r
        y_total = (-p["Y_v"] * u_eff * v - p["Y_vv"] * self._signed_square(v)
                   - (1 + _dyn.A_H) * p["k_R"] * f_n * cos_d - _dyn.M11 * u_eff * r)
        # -N_uv*u*v + (M11-M22)*u*v = (M11-M22-N_uv)*u*v exactly over
        # the reals.  Combining first retains that cancellation for intervals.
        n_total = (-p["N_r"] * u_eff * r - p["N_rr"] * self._signed_square(r)
                   + (_dyn.M11 - _dyn.M22 - p["N_uv"]) * u_eff * v
                   - _dyn.RUD_ARM * p["k_R"] * f_n * cos_d)
        result = [self._divide(x_total, self._point(_dyn.M11)),
                  self._divide(y_total, self._point(_dyn.M22)),
                  self._divide(n_total, self._point(_dyn.M33)), r, zero,
                  u_eff * c.sin(psi) + v * c.cos(psi),
                  u_eff * c.cos(psi) - v * c.sin(psi)]
        self._guard(result)
        return result

    def _substep(self, state, command, rpm, braking_force):
        servo_new = self._servo(state[4], command)
        middle = list(state)
        middle[4] = (state[4] + servo_new) / 2
        k1 = self._rhs(middle, rpm)
        k2 = self._rhs([s + self._h * k / 2 for s, k in zip(middle, k1)], rpm)
        k3 = self._rhs([s + self._h * k / 2 for s, k in zip(middle, k2)], rpm)
        k4 = self._rhs([s + self._h * k for s, k in zip(middle, k3)], rpm)
        result = [s + self._h / 6 * (a + 2*b + 2*d + e)
                  for s, a, b, d, e in zip(middle, k1, k2, k3, k4)]
        self._guard(result)
        zero = self._point(0.0)
        result[0] = self._clip(result[0], zero, self._point(_dyn.MAX_SURGE_SPEED))
        result[1] = self._clip(result[1], self._point(-_dyn.MAX_SWAY_SPEED), self._point(_dyn.MAX_SWAY_SPEED))
        result[2] = self._clip(result[2], self._point(-_dyn.MAX_YAW_RATE_RAD), self._point(_dyn.MAX_YAW_RATE_RAD))
        result[4] = servo_new
        if braking_force is not None:
            # Plant skips a nonpositive force.  Enclose both branches if necessary.
            braking = self._maximum(braking_force, zero)
            result[0] = self._maximum(zero, result[0] - self._divide(braking, self._point(_dyn.M11)) * self._h)
        self._guard(result)
        return result

    def advance(self, rudder_percent: float, signed_rpm: float,
                dt: float = HISTORY_COMMAND_S) -> dict[str, IntervalBounds]:
        """Advance the declared discrete map, or fail without updating the tube."""
        try:
            dt = self._finite_command(dt, "dt")
            count = int(round(dt / SUBSTEP_S))
            if dt <= 0.0 or count < 1 or not math.isclose(dt, count * SUBSTEP_S, rel_tol=0.0, abs_tol=1e-12):
                raise EnclosureFailure("dt must be a positive integer multiple of 0.05 s")
            if count > MAX_SUBSTEPS_PER_CALL:
                raise EnclosureFailure("advance duration exceeds substep resource limit")
            command = self._command(rudder_percent)
            rpm_value = self._finite_command(signed_rpm, "signed_rpm")
            rpm = self._point(rpm_value)
            self._guard([rpm])
            braking = (self._point(0.5) * self._thrust(-rpm, self._point(0.0))
                       if rpm_value < 0.0 else None)
            state = list(self._state)
            pending = None if self._pending is None else list(self._pending)
            for _ in range(count):
                delayed, pending = self._delay(command, pending)
                state = self._substep(state, delayed, rpm, braking)
            result = self._bounds_for(state)
        except EnclosureFailure as error:
            self.diagnostics["last_failure"] = str(error)
            raise
        except (ArithmeticError, ValueError, TypeError) as error:
            self.diagnostics["last_failure"] = str(error)
            raise EnclosureFailure(f"interval propagation failed: {error}") from error
        self._state, self._pending = state, pending
        self.diagnostics["advanced_substeps"] += count
        self.diagnostics["last_failure"] = None
        return result

    def _bounds_for(self, state):
        self._guard(state)
        result = {}
        for key, value in zip(("u", "v", "r", "heading", "servo", "x", "y"), state):
            lo = float(np.nextafter(float(value.a), -np.inf))
            hi = float(np.nextafter(float(value.b), np.inf))
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise EnclosureFailure("outward conversion exceeded finite float bounds")
            result[key] = IntervalBounds(lo, hi)
        return result

    def bounds(self) -> dict[str, IntervalBounds]:
        """Return outward-converted state/servo bounds; no certification status."""
        return self._bounds_for(self._state)
