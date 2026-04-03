"""Bluefin-inspired nonlinear 3-DOF ship model, v2.

This version targets the two biggest remaining calibration errors seen in the
output-only sweeps:

1) Surge transient shape is wrong: the v1 model is too slow early but too fast
   eventually. v2 replaces the constant propeller law with an empirical
   *speed-dependent thrust law* that gives more thrust at low speed and less at
   high speed.

2) Turning authority vs speed loss is too tightly coupled: v1 uses one rudder
   scale for sway force, yaw moment, and axial drag. v2 separates these so the
   model can gain yaw-rate authority without unrealistically bleeding speed.

The public interface is unchanged:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)

where `rud` is percent of max rudder in [-100, 100].
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Geometry / compatibility constants
# ---------------------------------------------------------------------------
VESSEL_LENGTH = 1.725
VESSEL_WIDTH = 0.50
HULL_MARGIN = 0.15
HULL_FORWARD_SHIFT = 0.0
LIDAR_OFFSET_M = VESSEL_LENGTH / 2.0

# ---------------------------------------------------------------------------
# Physical parameters (from / inspired by the MATLAB Bluefin model)
# ---------------------------------------------------------------------------
rho = 1000.0
MASS = 64.55
MX = 3.662
MY = 62.7366
IZ = 9.6038
JZ = 0.6309
MOMINERTIA = IZ + JZ

L = VESSEL_LENGTH
DRAFT = 0.193
SW = 0.7614

MAX_RUD_ANGLE = 40.0
MAX_RUD_RATE_DPS = 20.0

TP = 0.193
TR = 0.256311
AH = 0.443853
X_RUDDER = -1.05309
X_HULL = -0.733125
KX = 0.6177
WR = 0.22
AR = 0.0091
FALP = 2.69279
L_R = -0.77735

XVV = 0.0623
XVR = 1.1415
XRR = 0.0027
YV = 2.47781051381700e-003
YR = 94.5956792789195e-009
YVV = 1.08140832998334e-003
YRR = 22.7583008858493e-012
YVR = 262.214901533461e-009
NV = 1.10546039494704e-003
NR = 42.2032985948020e-009
NVV = 482.463882083071e-006
NRR = 10.1534803187344e-012
NVR = 116.985615725573e-009

# ---------------------------------------------------------------------------
# Tunable gains (defaults taken near the best shared region)
# ---------------------------------------------------------------------------
THRUST_COEF = 0.06
DRAG_COEF = 1.5
TURN_COEF = 3.0              # hull sway/yaw damping scale ONLY in v2

# New surge-shape parameters
THRUST_LOW_SPEED_BOOST = 1.6     # more thrust near zero speed
THRUST_BOOST_U0 = 0.7            # e-folding speed [m/s] for low-speed boost
THRUST_HIGH_SPEED_DECAY = 0.26   # thrust roll-off with speed^2
LINEAR_SURGE_DAMP = 2.0

# New rudder-force split
RUDDER_FORCE_SCALE = 0.32        # sway force / normal force scale
RUDDER_YAW_SCALE = 2.60          # extra yaw authority
RUDDER_X_DRAG_SCALE = 0.02       # axial speed loss due to rudder side force
LINEAR_SWAY_DAMP = 18.0
LINEAR_YAW_DAMP = 1.5
BOW_THRUSTER_YAW_GAIN = 0.0

# Optional effective inertia scales
SURGE_INERTIA_SCALE = 1.0
YAW_INERTIA_SCALE = 1.0

# Numerical safety limits
MIN_FLOW_SPEED = 0.05
MAX_SURGE_SPEED = 5.0
MAX_SWAY_SPEED = 3.0
MAX_YAW_RATE_RAD = math.radians(160.0)


class ShipModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._v = 0.0
        self._v_sway = 0.0
        self._w = 0.0
        self._h = 0.0
        self._delta = 0.0
        self._x = 0.0
        self._y = 0.0

    def state_dict(self) -> Dict[str, float]:
        return {
            "u_body_mps": float(self._v),
            "v_body_mps": float(self._v_sway),
            "yaw_rate_radps": float(self._w),
            "yaw_rate_degps": float(math.degrees(self._w)),
            "heading_rad": float(self._h),
            "heading_deg": float(math.degrees(self._h) % 360.0),
            "rudder_deg": float(math.degrees(self._delta)),
            "x_m": float(self._x),
            "y_m": float(self._y),
            "speed_mps": float(math.hypot(self._v, self._v_sway)),
        }

    def update(self, rpm: float, rud: float, dt: float, *, thruster_rpm: float = 0.0) -> Tuple[float, float, float, float]:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        s0 = self._state_vector()
        x_prev, y_prev = self._x, self._y

        k1 = self._derivatives(s0, rpm, rud, thruster_rpm)
        k2 = self._derivatives(s0 + 0.5 * dt * k1, rpm, rud, thruster_rpm)
        k3 = self._derivatives(s0 + 0.5 * dt * k2, rpm, rud, thruster_rpm)
        k4 = self._derivatives(s0 + dt * k3, rpm, rud, thruster_rpm)
        s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        s1[0] = float(np.clip(s1[0], 0.0, MAX_SURGE_SPEED))
        s1[1] = float(np.clip(s1[1], -MAX_SWAY_SPEED, MAX_SWAY_SPEED))
        s1[2] = float(np.clip(s1[2], -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD))
        s1[4] = float(np.clip(s1[4], -self._max_rudder_rad(), self._max_rudder_rad()))

        self._set_state_vector(s1)
        dx = self._x - x_prev
        dy = self._y - y_prev
        heading_deg = math.degrees(self._h) % 360.0
        yaw_rate_degps = math.degrees(self._w)
        return dx, dy, heading_deg, yaw_rate_degps

    def _state_vector(self) -> np.ndarray:
        return np.array([
            self._v, self._v_sway, self._w, self._h, self._delta, self._x, self._y
        ], dtype=float)

    def _set_state_vector(self, s: np.ndarray) -> None:
        self._v = float(s[0])
        self._v_sway = float(s[1])
        self._w = float(s[2])
        self._h = float(s[3])
        self._delta = float(s[4])
        self._x = float(s[5])
        self._y = float(s[6])

    @staticmethod
    def _safe_log_cf() -> float:
        return 0.4631 / (math.log(4.0e7) ** 2.6)

    @staticmethod
    def _clip_rudder_percent(rud: float) -> float:
        return float(np.clip(rud, -100.0, 100.0))

    @staticmethod
    def _max_rudder_rad() -> float:
        return math.radians(MAX_RUD_ANGLE)

    @staticmethod
    def _max_rudder_rate_radps() -> float:
        return math.radians(MAX_RUD_RATE_DPS)

    def _propeller_force(self, rpm: float, u_eff: float) -> float:
        """Empirical speed-dependent thrust law.

        Goal: more thrust near zero speed, less thrust at high speed.
        This directly targets the current mismatch pattern:
        - too slow early
        - too fast eventually
        """
        n = max(rpm, 0.0)
        static_term = THRUST_COEF * n * abs(n)
        low_speed_boost = 1.0 + THRUST_LOW_SPEED_BOOST * math.exp(-u_eff / max(THRUST_BOOST_U0, 1e-6))
        high_speed_decay = 1.0 / (1.0 + THRUST_HIGH_SPEED_DECAY * u_eff * u_eff)
        return (1.0 - TP) * static_term * low_speed_boost * high_speed_decay

    def _derivatives(self, s: np.ndarray, rpm: float, rud: float, thruster_rpm: float) -> np.ndarray:
        u, v, r, psi, delta, x, y = [float(z) for z in s]

        delta_cmd = self._clip_rudder_percent(rud) / 100.0 * self._max_rudder_rad()
        delta_dot = float(np.clip(delta_cmd - delta, -self._max_rudder_rate_radps(), self._max_rudder_rate_radps()))

        u_eff = max(u, 0.0)
        U = max(math.hypot(u_eff, v), MIN_FLOW_SPEED)
        beta = math.atan2(v, max(abs(u_eff), MIN_FLOW_SPEED))
        r_nd = r * L / U

        # Hull surge force
        cf = self._safe_log_cf()
        x_visc = -DRAG_COEF * 0.5 * rho * SW * cf * u_eff * abs(u_eff)
        x_cross = -DRAG_COEF * 0.5 * rho * L * DRAFT * U * U * (
            XVV * (math.sin(beta) ** 2) + XVR * abs(math.sin(beta)) * abs(r_nd) + XRR * (r_nd ** 2)
        )
        x_lin = -LINEAR_SURGE_DAMP * u_eff
        x_hull = x_visc + x_cross + x_lin

        # Hull sway / yaw damping only
        y_hull = -TURN_COEF * (
            0.5 * rho * L * DRAFT * U * U * (
                YV * beta + YVV * abs(beta) * beta + YR * r_nd + YRR * abs(r_nd) * r_nd + YVR * beta * abs(r_nd)
            ) + LINEAR_SWAY_DAMP * v
        )
        n_hull = -TURN_COEF * (
            0.5 * rho * (L ** 2) * DRAFT * U * U * (
                NV * beta + NVV * abs(beta) * beta + NR * r_nd + NRR * abs(r_nd) * r_nd + NVR * abs(beta) * r_nd
            ) + LINEAR_YAW_DAMP * r
        )

        # Propeller thrust with speed-dependent shaping
        x_prop = self._propeller_force(rpm, u_eff)

        # Optional bow-thruster contribution to yaw moment
        n_thr = thruster_rpm / 60.0
        n_thr_moment = BOW_THRUSTER_YAW_GAIN * n_thr * abs(n_thr)

        # Rudder inflow and normal force
        n_prop = max(rpm, 0.0) / 60.0
        u_r = max(MIN_FLOW_SPEED, (1.0 - WR) * u_eff + 0.6 * KX * n_prop)
        v_r = v + L_R * r
        alpha_r = delta - math.atan2(v_r, u_r)

        f_n = RUDDER_FORCE_SCALE * 0.5 * rho * AR * FALP * (u_r * u_r + v_r * v_r) * math.sin(alpha_r)

        # Split rudder effect into axial loss, sway, and yaw separately.
        x_rud = -RUDDER_X_DRAG_SCALE * abs(f_n) * abs(math.sin(delta))
        y_rud = -(1.0 + AH) * f_n * math.cos(delta)
        rudder_arm = abs(X_RUDDER + AH * X_HULL)
        n_rud = -RUDDER_YAW_SCALE * rudder_arm * f_n * math.cos(delta)

        x_total = x_hull + x_prop + x_rud
        y_total = y_hull + y_rud
        n_total = n_hull + n_rud + n_thr_moment

        m11 = (MASS + MX) * SURGE_INERTIA_SCALE
        m22 = MASS + MY
        m33 = MOMINERTIA * YAW_INERTIA_SCALE

        du = (x_total + m22 * v * r) / m11
        dv = (y_total - m11 * u_eff * r) / m22
        dr = n_total / m33

        dx = u_eff * math.sin(psi) + v * math.cos(psi)
        dy = u_eff * math.cos(psi) - v * math.sin(psi)
        dpsi = r

        return np.array([du, dv, dr, dpsi, delta_dot, dx, dy], dtype=float)


if __name__ == "__main__":
    model = ShipModel()
    print("Straight demo")
    for k in range(60):
        _, _, hdg, yaw = model.update(14.0, 0.0, 0.1)
        if k % 10 == 0:
            print(f"t={0.1*(k+1):4.1f}s u={model._v:5.2f} yaw={yaw:6.2f} hdg={hdg:6.2f}")

    model.reset()
    print("Turning demo")
    for k in range(120):
        _, _, hdg, yaw = model.update(20.0, 62.5, 0.1)  # ~25 deg if max rud=40 deg
        if k % 10 == 0:
            print(f"t={0.1*(k+1):4.1f}s u={model._v:5.2f} yaw={yaw:6.2f} hdg={hdg:6.2f}")
