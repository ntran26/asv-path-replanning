"""Nonlinear 3-DOF manoeuvring model of the Bluefin-class ASV.

Usage:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rudder_percent, dt)

`rudder_percent` is the commanded rudder in [-100, 100].  Note the sign
inversion in `_derivatives`: a positive command produces a *negative* rudder
angle.  That is the convention the trained policies were built against.

Hull forces follow the MATLAB Bluefin model with three empirical adjustments
that were needed to match the measured response:

* a speed-dependent propeller law (more thrust near zero speed, less at high
  speed) instead of a constant one;
* separate scales for rudder sway force, yaw moment and axial drag, so turning
  authority can be tuned without bleeding an unrealistic amount of speed;
* linear damping in all three axes.
"""

from __future__ import annotations

import math

import numpy as np

# --- Geometry --------------------------------------------------------------
VESSEL_LENGTH = 1.725
VESSEL_WIDTH = 0.50
HULL_MARGIN = 0.15                      # inflation applied to the collision hull
LIDAR_OFFSET_M = VESSEL_LENGTH / 2.0    # sensor sits at the bow

# --- Hull and hydrodynamics ------------------------------------------------
RHO = 1000.0
MASS = 64.55
MX = 3.662                              # added mass, surge
MY = 62.7366                            # added mass, sway
MOMINERTIA = 9.6038 + 0.6309            # Iz + Jz

DRAFT = 0.193
SW = 0.7614                             # wetted surface area

MAX_RUD_ANGLE = 40.0
MAX_RUD_RATE_DPS = 20.0

TP = 0.193                              # thrust deduction
AH = 0.443853                           # rudder-hull interaction
X_RUDDER = -1.05309
X_HULL = -0.733125
KX = 0.6177                             # propeller race factor
WR = 0.22                               # wake fraction at the rudder
AR = 0.0091                             # rudder area
FALP = 2.69279                          # rudder lift slope
L_R = -0.77735                          # rudder longitudinal position

# Manoeuvring derivatives.
XVV, XVR, XRR = 0.0623, 1.1415, 0.0027
YV, YR = 2.47781051381700e-003, 94.5956792789195e-009
YVV, YRR, YVR = 1.08140832998334e-003, 22.7583008858493e-012, 262.214901533461e-009
NV, NR = 1.10546039494704e-003, 42.2032985948020e-009
NVV, NRR, NVR = 482.463882083071e-006, 10.1534803187344e-012, 116.985615725573e-009

# --- Calibrated gains ------------------------------------------------------
# THRUST_CAL scales the whole thrust map so that steady surge at CRUISE_RPM
# matches the measured field cruise speed (constants.U_REF).  Paper 2's map was
# never validated against the trial logs -- 05 §2 lists "Paper 2 used thrust
# proportional to RPM^2; verify" -- and mining those logs (02b T1) put the real
# cruise at 1.14 m/s against the simulator's 1.77.
#
# The discrepancy lives in this one number by design (02b C2): when 05's
# identification lands, this is the single value that changes, rather than every
# speed normaliser downstream.
#
# Solved by bisection for steady u = 1.14 m/s at 12 RPM.
# TODO(05): replace with the identified thrust map; this is a calibration, not
# an identification.
THRUST_CAL = 0.3751

THRUST_COEF = 0.06 * THRUST_CAL
DRAG_COEF = 1.5
TURN_COEF = 3.0                         # hull sway/yaw damping scale

THRUST_LOW_SPEED_BOOST = 1.6            # extra thrust near zero speed
THRUST_BOOST_U0 = 0.7                   # e-folding speed of that boost [m/s]
THRUST_HIGH_SPEED_DECAY = 0.26          # thrust roll-off with speed^2
LINEAR_SURGE_DAMP = 2.0

RUDDER_FORCE_SCALE = 0.32               # rudder normal force scale
RUDDER_YAW_SCALE = 2.60                 # extra yaw authority
RUDDER_X_DRAG_SCALE = 0.02              # axial speed loss from rudder side force
LINEAR_SWAY_DAMP = 18.0
LINEAR_YAW_DAMP = 1.5

# --- Numerical limits ------------------------------------------------------
MIN_FLOW_SPEED = 0.05
MAX_SURGE_SPEED = 5.0
MAX_SWAY_SPEED = 3.0
MAX_YAW_RATE_RAD = math.radians(160.0)

MAX_RUD_RAD = math.radians(MAX_RUD_ANGLE)
MAX_RUD_RATE_RADPS = math.radians(MAX_RUD_RATE_DPS)

# Skin friction, evaluated once at a fixed reference Reynolds number.
CF = 0.4631 / (math.log(4.0e7) ** 2.6)

# State vector layout.
_U, _V, _R, _PSI, _DELTA, _X, _Y = range(7)


class ShipModel:
    """RK4-integrated 3-DOF hull.  Angles are radians internally."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._s = np.zeros(7, dtype=float)

    # Read-only views of the integrated state.
    @property
    def u(self) -> float:
        """Surge velocity [m/s]."""
        return float(self._s[_U])

    @property
    def v(self) -> float:
        """Sway velocity [m/s]."""
        return float(self._s[_V])

    @property
    def yaw_rate(self) -> float:
        """Yaw rate [rad/s]."""
        return float(self._s[_R])

    @property
    def heading_deg(self) -> float:
        return math.degrees(float(self._s[_PSI])) % 360.0

    @property
    def rudder_deg(self) -> float:
        return math.degrees(float(self._s[_DELTA]))

    def update(self, rpm: float, rud: float, dt: float):
        """Advance one step; return (dx, dy, heading_deg, yaw_rate_degps)."""
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        s0 = self._s
        x_prev, y_prev = s0[_X], s0[_Y]

        k1 = self._derivatives(s0, rpm, rud)
        k2 = self._derivatives(s0 + 0.5 * dt * k1, rpm, rud)
        k3 = self._derivatives(s0 + 0.5 * dt * k2, rpm, rud)
        k4 = self._derivatives(s0 + dt * k3, rpm, rud)
        s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        s1[_U] = float(np.clip(s1[_U], 0.0, MAX_SURGE_SPEED))
        s1[_V] = float(np.clip(s1[_V], -MAX_SWAY_SPEED, MAX_SWAY_SPEED))
        s1[_R] = float(np.clip(s1[_R], -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD))
        s1[_DELTA] = float(np.clip(s1[_DELTA], -MAX_RUD_RAD, MAX_RUD_RAD))
        self._s = s1

        return (
            float(s1[_X] - x_prev),
            float(s1[_Y] - y_prev),
            self.heading_deg,
            math.degrees(float(s1[_R])),
        )

    @staticmethod
    def _propeller_thrust(rpm: float, u_eff: float) -> float:
        """Empirical thrust law: boosted at low speed, rolled off at high speed."""
        n = max(rpm, 0.0)
        static = THRUST_COEF * n * abs(n)
        boost = 1.0 + THRUST_LOW_SPEED_BOOST * math.exp(-u_eff / THRUST_BOOST_U0)
        decay = 1.0 / (1.0 + THRUST_HIGH_SPEED_DECAY * u_eff * u_eff)
        return (1.0 - TP) * static * boost * decay

    def _derivatives(self, s: np.ndarray, rpm: float, rud: float) -> np.ndarray:
        u = float(s[_U])
        v = float(s[_V])
        r = float(s[_R])
        psi = float(s[_PSI])
        delta = float(s[_DELTA])

        # Rudder servo: rate-limited tracking of the commanded angle.
        delta_cmd = -float(np.clip(rud, -100.0, 100.0)) / 100.0 * MAX_RUD_RAD
        delta_dot = float(np.clip(delta_cmd - delta, -MAX_RUD_RATE_RADPS, MAX_RUD_RATE_RADPS))

        u_eff = max(u, 0.0)
        flow = max(math.hypot(u_eff, v), MIN_FLOW_SPEED)
        beta = math.atan2(v, max(abs(u_eff), MIN_FLOW_SPEED))
        r_nd = r * VESSEL_LENGTH / flow

        # Hull surge: skin friction, cross-flow drag, linear damping.
        x_hull = (
            -DRAG_COEF * 0.5 * RHO * SW * CF * u_eff * abs(u_eff)
            - DRAG_COEF * 0.5 * RHO * VESSEL_LENGTH * DRAFT * flow * flow * (
                XVV * (math.sin(beta) ** 2)
                + XVR * abs(math.sin(beta)) * abs(r_nd)
                + XRR * (r_nd ** 2)
            )
            - LINEAR_SURGE_DAMP * u_eff
        )

        # Hull sway force and yaw moment.
        y_hull = -TURN_COEF * (
            0.5 * RHO * VESSEL_LENGTH * DRAFT * flow * flow * (
                YV * beta + YVV * abs(beta) * beta
                + YR * r_nd + YRR * abs(r_nd) * r_nd
                + YVR * beta * abs(r_nd)
            )
            + LINEAR_SWAY_DAMP * v
        )
        n_hull = -TURN_COEF * (
            0.5 * RHO * (VESSEL_LENGTH ** 2) * DRAFT * flow * flow * (
                NV * beta + NVV * abs(beta) * beta
                + NR * r_nd + NRR * abs(r_nd) * r_nd
                + NVR * abs(beta) * r_nd
            )
            + LINEAR_YAW_DAMP * r
        )

        # Rudder: inflow accelerated by the propeller race, then normal force.
        n_prop = max(rpm, 0.0) / 60.0
        u_r = max(MIN_FLOW_SPEED, (1.0 - WR) * u_eff + 0.6 * KX * n_prop)
        v_r = v + L_R * r
        alpha_r = delta - math.atan2(v_r, u_r)
        f_n = RUDDER_FORCE_SCALE * 0.5 * RHO * AR * FALP * (u_r * u_r + v_r * v_r) * math.sin(alpha_r)

        x_rud = -RUDDER_X_DRAG_SCALE * abs(f_n) * abs(math.sin(delta))
        y_rud = -(1.0 + AH) * f_n * math.cos(delta)
        n_rud = -RUDDER_YAW_SCALE * abs(X_RUDDER + AH * X_HULL) * f_n * math.cos(delta)

        m11 = MASS + MX
        m22 = MASS + MY

        du = (x_hull + self._propeller_thrust(rpm, u_eff) + x_rud + m22 * v * r) / m11
        dv = (y_hull + y_rud - m11 * u_eff * r) / m22
        dr = (n_hull + n_rud) / MOMINERTIA

        return np.array([
            du, dv, dr, r, delta_dot,
            u_eff * math.sin(psi) + v * math.cos(psi),
            u_eff * math.cos(psi) - v * math.sin(psi),
        ], dtype=float)


if __name__ == "__main__":
    for label, rpm, rud, steps in [("Straight demo", 14.0, 0.0, 60), ("Turning demo", 20.0, 62.5, 120)]:
        model = ShipModel()
        print(label)
        for k in range(steps):
            _, _, hdg, yaw = model.update(rpm, rud, 0.1)
            if k % 10 == 0:
                print(f"  t={0.1 * (k + 1):4.1f}s u={model.u:5.2f} yaw={yaw:6.2f} hdg={hdg:6.2f}")
