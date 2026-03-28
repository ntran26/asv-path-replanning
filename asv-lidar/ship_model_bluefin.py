"""Bluefin-inspired nonlinear 3-DOF ship model for Python.

This is **not** a byte-for-byte transliteration of the MATLAB code. The MATLAB
snippets shared in chat mix state indexing and non-dimensionalization in a way
that does not map cleanly onto the old Python interface. Instead, this module
keeps the *structure* of the MATLAB model:

- surge / sway / yaw dynamics (3 DOF)
- added-mass terms
- nonlinear hull damping
- propeller thrust
- rudder actuator saturation + rate limiting
- rudder side force and yaw moment
- body-to-world kinematics

while exposing the same *practical* interface as the old Python model:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)

where:
- rpm: propeller command (same style as old Python model)
- rud: rudder command in percent [-100, 100]
- dt:  timestep [s]

Optional bow-thruster input is supported with a keyword argument:

    model.update(rpm, rud, dt, thruster_rpm=0.0)

Compatibility notes
-------------------
- self._v stores surge (forward) speed for compatibility with existing code.
- self._v_sway stores sway (lateral) speed.
- self._h stores heading [rad]. Heading returned by `update()` is wrapped to
  [0, 360) deg for easier downstream use.
- self._w stores yaw rate [rad/s]. Yaw rate returned by `update()` is in deg/s.

This model is meant to be a better starting point than the old lumped
thrust-drag-turn model, while still remaining simple enough to run inside the
current RL/testing codebase.
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

# Main rigid-body and added-mass terms
MASS = 64.55          # kg
MX = 3.662            # added mass in surge [kg]
MY = 62.7366          # added mass in sway  [kg]
IZ = 9.6038           # yaw inertia [kg m^2]
JZ = 0.6309           # added yaw inertia [kg m^2]
MOMINERTIA = IZ + JZ  # compatibility / convenience

L = VESSEL_LENGTH     # characteristic length [m]
DRAFT = 0.193         # draft [m]
SW = 0.7614           # wetted surface area [m^2]

# Rudder limits / actuator dynamics
MAX_RUD_ANGLE = 40.0        # deg (same style as MATLAB)
MAX_RUD_RATE_DPS = 20.0     # deg/s

# Propeller / rudder coefficients (lifted from the MATLAB file when usable)
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

# Hydrodynamic derivative seeds (used in a stable MMG/Abkowitz-like form)
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
# User-tunable gains
# ---------------------------------------------------------------------------
# Kept in the familiar style of the old Python model.
THRUST_COEF = 0.04        # scales propeller thrust from the old rpm interface
DRAG_COEF = 1.0           # scales surge hull drag terms
TURN_COEF = 1.0           # scales sway / yaw damping + rudder-generated turning

# Extra stabilizing / shaping gains.
# These are not in the old model but help keep the MATLAB-inspired version
# numerically stable with the old simplified input conventions.
RUDDER_FORCE_SCALE = 0.10
LINEAR_SURGE_DAMP = 2.0
LINEAR_SWAY_DAMP = 20.0
LINEAR_YAW_DAMP = 5.0
BOW_THRUSTER_YAW_GAIN = 0.0  # set > 0 if you later want to use thruster_rpm

# Numerical safety limits
MIN_FLOW_SPEED = 0.05
MAX_SURGE_SPEED = 5.0               # m/s
MAX_SWAY_SPEED = 3.0                # m/s
MAX_YAW_RATE_RAD = math.radians(120.0)


class ShipModel:
    """Nonlinear 3-DOF Bluefin-inspired ship model.

    Input interface matches the old ship model:
        update(rpm, rud, dt) -> (dx, dy, heading_deg, yaw_rate_degps)

    where `rud` is a percentage in [-100, 100].
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all dynamic states to zero."""
        self._v = 0.0          # surge speed [m/s] (kept for compatibility)
        self._v_sway = 0.0     # sway speed  [m/s]
        self._w = 0.0          # yaw rate    [rad/s]
        self._h = 0.0          # heading     [rad], 0 points along +Y
        self._delta = 0.0      # actual rudder angle [rad]
        self._x = 0.0          # world x [m]
        self._y = 0.0          # world y [m]

    def state_dict(self) -> Dict[str, float]:
        """Return a dictionary view of the current internal state."""
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

    def update(
        self,
        rpm: float,
        rud: float,
        dt: float,
        *,
        thruster_rpm: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Advance the model by one step.

        Parameters
        ----------
        rpm:
            Propeller command in the same style as the old Python model.
        rud:
            Rudder command percentage in [-100, 100].
        dt:
            Timestep [s].
        thruster_rpm:
            Optional bow-thruster rpm command. Default is 0.0 so old calling
            code can keep using update(rpm, rud, dt).

        Returns
        -------
        dx, dy, heading_deg, yaw_rate_degps
            Same output style as the old Python ship model.
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        s0 = self._state_vector()
        x_prev, y_prev = self._x, self._y

        # RK4 integration for better stability than explicit Euler.
        k1 = self._derivatives(s0, rpm, rud, thruster_rpm)
        k2 = self._derivatives(s0 + 0.5 * dt * k1, rpm, rud, thruster_rpm)
        k3 = self._derivatives(s0 + 0.5 * dt * k2, rpm, rud, thruster_rpm)
        k4 = self._derivatives(s0 + dt * k3, rpm, rud, thruster_rpm)
        s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Safety clipping. This keeps the model numerically robust when driven
        # by the old simplified input conventions.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _state_vector(self) -> np.ndarray:
        return np.array(
            [
                self._v,
                self._v_sway,
                self._w,
                self._h,
                self._delta,
                self._x,
                self._y,
            ],
            dtype=float,
        )

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
        # Matches the viscosity-style friction term used in the MATLAB code.
        return 0.4631 / (math.log(4.0e7) ** 2.6)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _clip_rudder_percent(rud: float) -> float:
        return float(np.clip(rud, -100.0, 100.0))

    @staticmethod
    def _max_rudder_rad() -> float:
        return math.radians(MAX_RUD_ANGLE)

    @staticmethod
    def _max_rudder_rate_radps() -> float:
        return math.radians(MAX_RUD_RATE_DPS)

    def _derivatives(
        self,
        s: np.ndarray,
        rpm: float,
        rud: float,
        thruster_rpm: float,
    ) -> np.ndarray:
        # State unpacking
        u, v, r, psi, delta, x, y = [float(z) for z in s]

        # ------------------------------------------------------------------
        # 1) Rudder actuator dynamics: saturate command and rate-limit motion
        # ------------------------------------------------------------------
        delta_cmd = self._clip_rudder_percent(rud) / 100.0 * self._max_rudder_rad()
        delta_dot = float(np.clip(
            delta_cmd - delta,
            -self._max_rudder_rate_radps(),
            self._max_rudder_rate_radps(),
        ))

        # For force calculations we do not allow the hydrodynamics to see a
        # negative forward inflow. This keeps the simplified input interface
        # from causing unrealistic immediate reverse flow in hard turns.
        u_eff = max(u, 0.0)
        U = max(math.hypot(u_eff, v), MIN_FLOW_SPEED)
        beta = math.atan2(v, max(abs(u_eff), MIN_FLOW_SPEED))
        r_nd = r * L / U

        # ------------------------------------------------------------------
        # 2) Hull surge force: friction + cross-flow + small linear damping
        # ------------------------------------------------------------------
        cf = self._safe_log_cf()
        x_visc = -DRAG_COEF * 0.5 * rho * SW * cf * u_eff * abs(u_eff)
        x_cross = -DRAG_COEF * 0.5 * rho * L * DRAFT * U * U * (
            XVV * (math.sin(beta) ** 2)
            + XVR * abs(math.sin(beta)) * abs(r_nd)
            + XRR * (r_nd ** 2)
        )
        x_lin = -DRAG_COEF * LINEAR_SURGE_DAMP * u_eff
        x_hull = x_visc + x_cross + x_lin

        # ------------------------------------------------------------------
        # 3) Hull sway and yaw damping (MMG/Abkowitz-like form)
        # ------------------------------------------------------------------
        y_hull = -TURN_COEF * (
            0.5 * rho * L * DRAFT * U * U * (
                YV * beta
                + YVV * abs(beta) * beta
                + YR * r_nd
                + YRR * abs(r_nd) * r_nd
                + YVR * beta * abs(r_nd)
            )
            + LINEAR_SWAY_DAMP * v
        )

        n_hull = -TURN_COEF * (
            0.5 * rho * (L ** 2) * DRAFT * U * U * (
                NV * beta
                + NVV * abs(beta) * beta
                + NR * r_nd
                + NRR * abs(r_nd) * r_nd
                + NVR * abs(beta) * r_nd
            )
            + LINEAR_YAW_DAMP * r
        )

        # ------------------------------------------------------------------
        # 4) Propeller thrust from the old rpm-style input
        # ------------------------------------------------------------------
        x_prop = (1.0 - TP) * THRUST_COEF * rpm * abs(rpm)

        # Optional bow-thruster contribution to yaw moment.
        n_thr = thruster_rpm / 60.0
        n_thr_moment = BOW_THRUSTER_YAW_GAIN * n_thr * abs(n_thr)

        # ------------------------------------------------------------------
        # 5) Rudder normal force (MATLAB-inspired, but stabilized)
        # ------------------------------------------------------------------
        n_prop = max(rpm, 0.0) / 60.0
        u_r = max(MIN_FLOW_SPEED, (1.0 - WR) * u_eff + 0.6 * KX * n_prop)
        v_r = v + L_R * r
        alpha_r = delta - math.atan2(v_r, u_r)

        f_n = TURN_COEF * RUDDER_FORCE_SCALE * 0.5 * rho * AR * FALP * (
            u_r * u_r + v_r * v_r
        ) * math.sin(alpha_r)

        x_rud = -(1.0 - TR) * f_n * math.sin(delta)
        y_rud = -(1.0 + AH) * f_n * math.cos(delta)
        n_rud = -(X_RUDDER + AH * X_HULL) * f_n * math.cos(delta)

        # ------------------------------------------------------------------
        # 6) Combine forces and moments
        # ------------------------------------------------------------------
        x_total = x_hull + x_prop + x_rud
        y_total = y_hull + y_rud
        n_total = n_hull + n_rud + n_thr_moment

        # ------------------------------------------------------------------
        # 7) Body dynamics with added mass (3 DOF)
        # ------------------------------------------------------------------
        m11 = MASS + MX
        m22 = MASS + MY
        m33 = MOMINERTIA

        du = (x_total + m22 * v * r) / m11
        dv = (y_total - m11 * u_eff * r) / m22
        dr = n_total / m33

        # ------------------------------------------------------------------
        # 8) Kinematics
        # ------------------------------------------------------------------
        # Heading convention matches the old Python model:
        #   psi = 0 -> vessel points along +Y
        dx = u_eff * math.sin(psi) + v * math.cos(psi)
        dy = u_eff * math.cos(psi) - v * math.sin(psi)
        dpsi = r

        return np.array([du, dv, dr, dpsi, delta_dot, dx, dy], dtype=float)


if __name__ == "__main__":
    model = ShipModel()

    print("Straight run demo")
    for k in range(50):
        dx, dy, hdg, yaw = model.update(30.0, 0.0, 0.1)
        if k % 10 == 0:
            print(
                f"t={0.1*(k+1):4.1f}s  u={model._v:5.2f}  v={model._v_sway:6.2f}  "
                f"yaw={yaw:6.2f} deg/s  hdg={hdg:6.2f} deg"
            )

    print("\nTurning run demo")
    for k in range(100):
        dx, dy, hdg, yaw = model.update(30.0, 100.0, 0.1)
        if k % 10 == 0:
            print(
                f"t={5.0 + 0.1*(k+1):4.1f}s  u={model._v:5.2f}  v={model._v_sway:6.2f}  "
                f"yaw={yaw:6.2f} deg/s  hdg={hdg:6.2f} deg"
            )
