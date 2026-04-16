"""Faithful Python adapter of ``Bluefin4DOFModel02.m`` with repo-compatible I/O.

Public interface matches ``ship_model_bluefin_v2.py``:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)

where
    - ``rpm`` is the repo-facing commanded shaft speed (not solver-native MATLAB rpm)
    - ``rud`` is rudder percentage in ``[-100, 100]``
    - positive ``rud`` follows the same public convention as ``ship_model_bluefin_v2.py``

The internal dynamics are taken directly from ``Bluefin4DOFModel02.m``:
    - 4 DOF: surge, sway, roll, yaw
    - actuator states: rudder, main propeller, bow thruster
    - hull / propeller / rudder / bow-thruster force decomposition

Only a few *numerical guards* are added so the model can run safely from rest
inside Python validation and sweep scripts:
    - floor on total speed U
    - protected division when propeller command is near zero
    - clipping of asin arguments and extreme states

A small command bridge is retained so the model can be swapped into the same
Python environment used by ``ship_model_bluefin_v2.py``. Set
``RPM_INPUT_TO_SOLVER_RPM`` to change the mapping from repo-facing rpm command
into the MATLAB model's command rpm.
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
# Public command interface -> MATLAB command bridge
# ---------------------------------------------------------------------------
# The original MATLAB solver uses ui(2)=1000 rpm as a representative command.
# Keeping the repo-facing interface practical, a command of 15.0 maps to 1000 rpm.
RPM_INPUT_TO_SOLVER_RPM = 1000.0 / 15.0
THRUSTER_INPUT_TO_SOLVER_RPM = 1000.0 / 15.0

# ---------------------------------------------------------------------------
# Optional calibration multipliers (all 1.0 = faithful port of the MATLAB file)
# ---------------------------------------------------------------------------
PROPELLER_THRUST_SCALE = 1.0
RUDDER_FORCE_SCALE = 1.0
BOW_THRUSTER_SCALE = 1.0
ROLL_DAMP_SCALE = 1.0
ROLL_RESTORE_SCALE = 1.0

# ---------------------------------------------------------------------------
# Physical / actuator constants copied from Bluefin4DOFModel02.m
# ---------------------------------------------------------------------------
MASS = 64.55
MAX_RUD_ANGLE = 40.0
MAX_RUD_RATE_DPS = 20.0
MAX_SHAFT_RATE_RPSPS = 1000.0 / 60.0  # MATLAB Nc_max

# Numerical safety limits (not present in MATLAB; needed for robust Python runs)
MIN_FLOW_SPEED = 0.05
MIN_PROP_COMMAND_RPS = 1e-6
MAX_SURGE_SPEED = 6.0
MAX_SWAY_SPEED = 4.0
MAX_ROLL_RATE_RAD = math.radians(180.0)
MAX_YAW_RATE_RAD = math.radians(180.0)
MAX_ROLL_ANGLE_RAD = math.radians(45.0)


class ShipModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._u = 0.0
        self._v = 0.0
        self._p = 0.0
        self._r = 0.0
        self._x = 0.0
        self._y = 0.0
        self._phi = 0.0
        self._psi = 0.0
        self._delta = 0.0
        self._n1 = 0.0  # actual propeller state [rps]
        self._n2 = 0.0  # actual bow-thruster state [rps]

        # Compatibility aliases used elsewhere in the repo
        self._v_sway = 0.0
        self._w = 0.0
        self._h = 0.0

    # ------------------------------------------------------------------
    # Public repo-facing API
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, float]:
        return {
            "u_body_mps": float(self._u),
            "v_body_mps": float(self._v),
            "yaw_rate_radps": float(self._r),
            "yaw_rate_degps": float(math.degrees(self._r)),
            "heading_rad": float(self._psi),
            "heading_deg": float(math.degrees(self._psi) % 360.0),
            "rudder_deg": float(math.degrees(self._delta)),
            "x_m": float(self._x),
            "y_m": float(self._y),
            "speed_mps": float(math.hypot(self._u, self._v)),
            "roll_deg": float(math.degrees(self._phi)),
            "roll_rate_degps": float(math.degrees(self._p)),
            "prop_rps": float(self._n1),
            "thruster_rps": float(self._n2),
        }

    def update(
        self,
        rpm: float,
        rud: float,
        dt: float,
        *,
        thruster_rpm: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Advance the 4DOF model by ``dt`` seconds.

        Parameters
        ----------
        rpm : float
            Repo-facing shaft-speed command. Mapped to MATLAB command rpm by
            ``RPM_INPUT_TO_SOLVER_RPM``.
        rud : float
            Rudder percentage in [-100, 100], same public convention as v2.
        dt : float
            Integration timestep [s].
        thruster_rpm : float, optional
            Repo-facing bow-thruster command, mapped by
            ``THRUSTER_INPUT_TO_SOLVER_RPM``.
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        s0 = self._state_vector()
        x_prev, y_prev = self._x, self._y

        delta_cmd = float(np.clip(rud, -100.0, 100.0)) / 100.0 * math.radians(MAX_RUD_ANGLE)
        n1_cmd_rpm = max(float(rpm), 0.0) * RPM_INPUT_TO_SOLVER_RPM
        n2_cmd_rpm = float(thruster_rpm) * THRUSTER_INPUT_TO_SOLVER_RPM

        # Fixed-step RK4 for robustness.
        k1 = self._derivatives(s0, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        k2 = self._derivatives(s0 + 0.5 * dt * k1, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        k3 = self._derivatives(s0 + 0.5 * dt * k2, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        k4 = self._derivatives(s0 + dt * k3, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Safety clipping only; these are not intended as tuning terms.
        s1[0] = float(np.clip(s1[0], -1.0, MAX_SURGE_SPEED))
        s1[1] = float(np.clip(s1[1], -MAX_SWAY_SPEED, MAX_SWAY_SPEED))
        s1[2] = float(np.clip(s1[2], -MAX_ROLL_RATE_RAD, MAX_ROLL_RATE_RAD))
        s1[3] = float(np.clip(s1[3], -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD))
        s1[6] = float(np.clip(s1[6], -MAX_ROLL_ANGLE_RAD, MAX_ROLL_ANGLE_RAD))
        s1[8] = float(np.clip(s1[8], -math.radians(MAX_RUD_ANGLE), math.radians(MAX_RUD_ANGLE)))
        s1[7] = float((s1[7] + math.pi) % (2.0 * math.pi) - math.pi)

        self._set_state_vector(s1)
        dx = self._x - x_prev
        dy = self._y - y_prev
        heading_deg = math.degrees(self._psi) % 360.0
        yaw_rate_degps = math.degrees(self._r)
        return dx, dy, heading_deg, yaw_rate_degps

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _state_vector(self) -> np.ndarray:
        return np.array(
            [
                self._u,
                self._v,
                self._p,
                self._r,
                self._x,
                self._y,
                self._phi,
                self._psi,
                self._delta,
                self._n1,
                self._n2,
            ],
            dtype=float,
        )

    def _set_state_vector(self, s: np.ndarray) -> None:
        self._u = float(s[0])
        self._v = float(s[1])
        self._p = float(s[2])
        self._r = float(s[3])
        self._x = float(s[4])
        self._y = float(s[5])
        self._phi = float(s[6])
        self._psi = float(s[7])
        self._delta = float(s[8])
        self._n1 = float(s[9])
        self._n2 = float(s[10])

        self._v_sway = self._v
        self._w = self._r
        self._h = self._psi

    # ------------------------------------------------------------------
    # Faithful derivatives from Bluefin4DOFModel02.m
    # ------------------------------------------------------------------
    def _derivatives(
        self,
        s: np.ndarray,
        delta_cmd: float,
        n1_cmd_rpm: float,
        n2_cmd_rpm: float,
    ) -> np.ndarray:
        u, v, p, r, xpos, ypos, phi, psi, delta, n1, n2 = [float(z) for z in s]

        # Guard intermediate RK states against numerical blow-up.
        u = float(np.clip(u, -1.0, MAX_SURGE_SPEED))
        v = float(np.clip(v, -MAX_SWAY_SPEED, MAX_SWAY_SPEED))
        p = float(np.clip(p, -MAX_ROLL_RATE_RAD, MAX_ROLL_RATE_RAD))
        r = float(np.clip(r, -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD))
        phi = float(np.clip(phi, -MAX_ROLL_ANGLE_RAD, MAX_ROLL_ANGLE_RAD))
        delta = float(np.clip(delta, -math.radians(MAX_RUD_ANGLE), math.radians(MAX_RUD_ANGLE)))
        psi = float((psi + math.pi) % (2.0 * math.pi) - math.pi)

        # Normalization variables
        Lpp = 1.725
        L = Lpp
        U_raw = math.hypot(u, v)
        U = max(U_raw, MIN_FLOW_SPEED)
        b = -math.asin(float(np.clip(v / U, -1.0, 1.0)))

        delta_max = math.radians(40.0)
        ddelta_max = math.radians(20.0)
        nc_max = 1000.0 / 60.0

        # Inputs and non-dimensional state variables
        # ui(2), ui(3) are MATLAB command rpm; convert to rps as in the .m file.
        delta_c = delta_cmd
        n1_c = n1_cmd_rpm / 60.0
        n2_c = n2_cmd_rpm / 60.0

        # Dimensional states and non-dimensional rates
        ud = u / U
        vd = v / U
        pd = p * L / U
        rd = r * L / U

        # Parameters (from Bluefin4DOFModel02.m)
        B = 0.5
        dm = 0.193
        d = dm
        disp = 0.06455
        xG = -0.1
        Dp = 0.1
        Dpph = 0.8791
        lambda_r = 1.4697
        eta_unused = 0.879
        AR = 0.0091
        ARpLd = AR / (L * d)
        xR = -1.05309
        GM = 1.87
        zG = 0.005
        zR = -0.01
        zH = 0.02

        cf = 1.0
        onet = 0.859 * cf
        onew = 0.806 * cf
        onetR = 0.857 * cf
        oneaH = 1.403 * cf
        aH = oneaH - 1.0
        xH = -0.646 * cf
        gR0 = 0.394 * cf
        cg = -0.53 * cf
        gR = gR0 * (1.0 + cg * abs(phi)) * cf
        ldR = -0.795 * cf
        epsi = 0.740 * cf
        kappa = 0.810 * cf
        eta = 0.140 * cf

        # Hydrodynamic coefficients
        Xd0 = -0.0212 * cf
        cx0 = -0.02 * cf
        Xdrph = 0.0092 * cf
        Xdbb = -0.0348 * cf
        cxbb = 2.10 * cf
        Xdbrmdy = -0.0957 * cf
        Xdrr = -0.0070 * cf
        cxrr = 3.74 * cf
        Xdbbbb = -0.0018 * cf

        Ydph = 0.0053 * cf
        Ydb = 0.2501 * cf
        cyb = -0.14 * cf
        Ydrmdx = 0.0346 * cf
        cyr = -0.61 * cf
        Ydbbph = -0.2979 * cf
        Ydbrph = 0.6308 * cf
        Ydrrph = -0.0854 * cf
        Ydbbb = 2.6087 * cf
        Ydbbr = -1.7091 * cf
        Ydbrr = 1.1682 * cf
        Ydrrr = -0.0461 * cf

        Kdph = -0.0185 * cf
        Kdb = -0.2586 * cf
        Kdr = 0.0532 * cf
        Kdbbph = 0.2229 * cf
        Kdbrph = 0.5374 * cf
        Kdrrph = -0.0928 * cf
        Kdbbb = -0.7293 * cf
        Kdbbr = 1.1474 * cf
        Kdbrr = -0.3351 * cf
        Kdrrr = -0.0132 * cf

        Ndph = -0.0086 * cf
        Ndb = 0.0966 * cf
        cnb = 0.22 * cf
        Ndr = -0.0513 * cf
        cnr = -0.62 * cf
        Ndbbph = -0.2510 * cf
        Ndbrph = 0.0722 * cf
        Ndrrph = -0.0172 * cf
        Ndbbb = 0.4218 * cf
        Ndbbr = -0.8629 * cf
        Ndbrr = 0.1459 * cf
        Ndrrr = -0.0439 * cf

        g = 9.81
        rho = 1000.0
        m = 64.55
        mx = 3.662
        my = 62.7366
        Ix = 0.567
        Iz = 9.6038
        Jx = 0.6309
        Jz = 10.2347

        m11 = m + mx
        m22 = m + my
        m33 = Ix + Jx
        m44 = Iz + Jz

        # Rudder saturation and dynamics
        if abs(delta_c) >= delta_max:
            delta_c = math.copysign(delta_max, delta_c)
        delta_dot = delta_c - delta
        if abs(delta_dot) >= ddelta_max:
            delta_dot = math.copysign(ddelta_max, delta_dot)

        # Shaft dynamics (faithful to MATLAB; note forces still use command speeds below)
        n1_dot = n1_c - n1
        n2_dot = n2_c - n2
        if abs(n1_dot) >= nc_max:
            n1_dot = math.copysign(nc_max, n1_dot)
        if abs(n2_dot) >= nc_max:
            n2_dot = math.copysign(nc_max, n2_dot)

        # MATLAB uses commanded shaft speeds directly for force generation.
        n1s = n1_c
        n2s = n2_c
        DPs = Dp

        # Propeller force
        if abs(n1s) < MIN_PROP_COMMAND_RPS:
            J = 0.0
            KT = 0.0
            XdP = 0.0
        else:
            J = onew * u / (n1s * DPs)
            a0, a1, a2 = 0.3267, -0.2297, -0.1607
            KT = a0 + a1 * J + a2 * J * J
            XdP = (
                PROPELLER_THRUST_SCALE
                * abs(n1s)
                * n1s
                * onet
                * KT
                * (DPs ** 4)
                / (0.5 * L * d * U * U)
            )

        # Rudder forces and moments
        if abs(n1s) < MIN_PROP_COMMAND_RPS or abs(J) < MIN_PROP_COMMAND_RPS:
            udR = epsi * onew
        else:
            Jsq = max(J * J, MIN_PROP_COMMAND_RPS ** 2)
            prop_term = max(1.0 + 8.0 * KT / (math.pi * Jsq), 0.0)
            udR = epsi * onew * math.sqrt(eta * ((1.0 + kappa * math.sqrt(prop_term) - 1.0) ** 2) + (1.0 - eta))
        vdR = -gR * (b - ldR * rd + (p * (zR - zG) / U))
        UdR = math.hypot(udR, vdR)
        alphaR = delta - math.atan2(-vdR, udR)
        FdN = -(ARpLd) * (6.13 * lambda_r / (2.25 + lambda_r)) * (UdR ** 2) * math.sin(alphaR)

        XdR = RUDDER_FORCE_SCALE * onetR * FdN * math.sin(delta) * math.cos(phi)
        YdR = RUDDER_FORCE_SCALE * (1.0 + aH) * FdN * math.cos(delta) * math.cos(phi)
        KdR = RUDDER_FORCE_SCALE * zR * YdR / L
        NdR = RUDDER_FORCE_SCALE * (xR + aH * xH) * FdN * math.cos(delta) * math.cos(phi)

        # Hull forces and moments
        XdH = (
            Xd0 * (1.0 + cx0 * abs(phi))
            + Xdrph * rd * phi
            + Xdbb * (1.0 + cxbb * abs(phi)) * b * b
            + Xdbrmdy * b * rd
            + Xdrr * (1.0 + cxrr * abs(phi)) * rd * rd
            + Xdbbbb * (b ** 4)
        )
        YdH = (
            Ydph * phi
            + Ydb * (1.0 + cyb * abs(phi)) * b
            + Ydrmdx * (1.0 + cyr * abs(phi)) * rd
            + Ydbbph * b * b * phi
            + Ydbrph * b * rd * phi
            + Ydrrph * rd * rd * phi
            + Ydbbb * (b ** 3)
            + Ydbbr * b * b * rd
            + Ydbrr * b * rd * rd
            + Ydrrr * (rd ** 3)
        )
        KdH = (
            Kdph * phi
            + Kdb * b
            + Kdr * rd
            + Kdbbph * b * b * phi
            + Kdbrph * b * rd * phi
            + Kdrrph * rd * rd * phi
            + Kdbbb * (b ** 3)
            + Kdbbr * b * b * rd
            + Kdbrr * b * rd * rd
            + Kdrrr * (rd ** 3)
        )
        NdH = (
            Ndph * phi
            + Ndb * (1.0 + cnb * abs(phi)) * b
            + Ndr * (1.0 + cnr * abs(phi)) * rd
            + Ndbbph * b * b * phi
            + Ndbrph * b * rd * phi
            + Ndrrph * rd * rd * phi
            + Ndbbb * (b ** 3)
            + Ndbbr * b * b * rd
            + Ndbrr * b * rd * rd
            + Ndrrr * (rd ** 3)
        )

        C44 = g * m * GM
        a = 0.5
        B44 = 2.0 * a / math.pi * math.sqrt(max(g * m * GM * (Ix + Jx), 0.0))
        KdH2 = zG * YdH - ROLL_DAMP_SCALE * B44 * p - ROLL_RESTORE_SCALE * C44 * phi - (zR - zG) * YdR

        # Bow thruster forces and moments
        xB = 0.45
        zB = -0.05
        KBT = 0.026
        FBT = 0.0 if abs(n2s) < MIN_PROP_COMMAND_RPS else (
            BOW_THRUSTER_SCALE * abs(n2s) * n2s * KBT / (0.5 * rho * L * d * U * U)
        )
        YdB = FBT
        KdB = zB * FBT / L
        NbB = xB * FBT / L

        # Overall forces and moments
        Xd = XdH + XdP + XdR
        Yd = YdH + YdR + YdB
        Kd = KdH + KdH2 + KdR + KdB
        Nd = NdH + NdR - xG * Yd + NbB

        # vdot term is computed separately in the MATLAB file for roll equation
        vdot = (Yd * (0.5 * rho * L * d * U * U) - m11 * u * r) / m22

        xdot = np.array(
            [
                (Xd * (0.5 * rho * L * d * U * U) + m22 * v * r) / m11,
                (Yd * (0.5 * rho * L * d * U * U) - m11 * u * r) / m22,
                (Kd * (0.5 * rho * L * d * d * U * U) + (zH - zG) * (my * vdot + mx * u * r)) / m33,
                Nd * (0.5 * rho * L * L * d * U * U) / m44,
                math.cos(psi) * u - math.sin(psi) * v * math.cos(phi),
                math.sin(psi) * u + math.cos(psi) * v * math.cos(phi),
                p,
                r * math.cos(phi),
                delta_dot,
                n1_dot,
                n2_dot,
            ],
            dtype=float,
        )
        return xdot


if __name__ == "__main__":
    # Minimal smoke test: nominal solver-like run.
    m = ShipModel()
    for k in range(200):
        dx, dy, hdg, yaw = m.update(15.0, 87.5, 0.01)
    print(m.state_dict())
