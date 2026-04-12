"""Direct-ish Python port of the MATLAB candidate Bluefin4DOFModel02.m.

Purpose
-------
This model is intended as a closer implementation of the most promising
MATLAB Bluefin model found in ``bluefin_matlab.zip`` than the previous
simplified Python models.

What it keeps from MATLAB
-------------------------
- 4-DOF dynamics: surge, sway, roll, yaw
- earth-fixed x/y/psi kinematics
- actual rudder state with rate limiting
- propeller and bow-thruster actuator states
- Japanese-model-inspired hull, propeller, and rudder force structure
- Bluefin geometry / mass / inertia values embedded in the MATLAB file

Interface
---------
To stay compatible with the existing Python environment, the public API is:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)

where:
- ``rpm`` is the commanded main-propeller speed in rpm
- ``rud`` is rudder command in percent of max rudder angle, consistent with
  the old Python model conventions
- ``dt`` is the integration step in seconds

The optional keyword ``thruster_rpm`` can be used if a bow thruster is needed.

Notes
-----
This is intentionally a *robust* port rather than a byte-for-byte translation.
The MATLAB model assumes non-zero speed and non-zero shaft speed in several
places; small epsilons are used here to avoid division-by-zero and keep the
model usable from rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Principal physical constants taken from Bluefin4DOFModel02.m
# ---------------------------------------------------------------------------
RHO = 1000.0
G = 9.81

LPP = 1.725
BREADTH = 0.5
DRAFT = 0.193
LIDAR_OFFSET = 0.0

MASS = 64.55
MX = 3.662
MY = 62.7366
IX = 0.567
IZ = 9.6038
JX = 0.6309
JZ = 10.2347

# Actuator limits from MATLAB candidate
MAX_RUDDER_DEG = 40.0
MAX_RUDDER_RATE_DPS = 20.0
MAX_SHAFT_RPS = 1000.0 / 60.0

# Bluefin geometry / interaction parameters from Bluefin4DOFModel02.m
DP = 0.1
AR = 0.0091
XR_POS = -1.05309
GM = 1.87
ZG = 0.005
ZR = -0.01
ZH = 0.02

# ---------------------------------------------------------------------------
# Calibration scalers (default = direct MATLAB candidate behaviour)
# These make later sweeps easier without editing the model equations.
# ---------------------------------------------------------------------------
PROP_FORCE_SCALE = 1.0
HULL_FORCE_SCALE = 1.0
RUDDER_FORCE_SCALE = 1.0
RUDDER_YAW_SCALE = 1.0
BOW_THRUSTER_SCALE = 1.0
ROLL_MOMENT_SCALE = 1.0
YAW_MOMENT_SCALE = 1.0

# Rudder command sign. Set to -1.0 if the experiment/controller convention is opposite.
RUDDER_COMMAND_SIGN = 1.0

# Numerical guards
U_EPS = 0.05
J_EPS = 1e-3
SQRT_EPS = 1e-9
MAX_U = 5.0
MAX_V = 3.0
MAX_P = math.radians(60.0)
MAX_R = math.radians(120.0)
MAX_PHI = math.radians(30.0)


@dataclass
class State:
    u: float = 0.05      # surge velocity [m/s]
    v: float = 0.0       # sway velocity [m/s]
    p: float = 0.0       # roll rate [rad/s]
    r: float = 0.0       # yaw rate [rad/s]
    x: float = 0.0       # x position [m]
    y: float = 0.0       # y position [m]
    phi: float = 0.0     # roll angle [rad]
    psi: float = 0.0     # heading [rad]
    delta: float = 0.0   # actual rudder angle [rad]
    n1: float = 0.0      # main shaft speed [rps]
    n2: float = 0.0      # bow thruster speed [rps]

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.u, self.v, self.p, self.r, self.x, self.y, self.phi, self.psi, self.delta, self.n1, self.n2],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "State":
        return cls(*map(float, vec))


class ShipModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = State()
        # Compatibility fields used by the older environment code.
        self._v = self.state.u
        self._v_sway = self.state.v
        self._p = self.state.p
        self._w = self.state.r
        self._h = self.state.psi
        self._delta = self.state.delta

    def state_dict(self) -> Dict[str, float]:
        s = self.state
        return {
            "u": s.u,
            "v": s.v,
            "p": s.p,
            "r": s.r,
            "x": s.x,
            "y": s.y,
            "phi": s.phi,
            "psi": s.psi,
            "delta": s.delta,
            "n1": s.n1,
            "n2": s.n2,
        }

    def update(self, rpm: float, rud: float, dt: float, thruster_rpm: float = 0.0) -> Tuple[float, float, float, float]:
        if dt <= 0:
            raise ValueError("dt must be positive")

        x0 = self.state.as_vector()

        # Convert the environment-style command to the MATLAB-style inputs.
        delta_cmd = math.radians(MAX_RUDDER_DEG * (RUDDER_COMMAND_SIGN * float(rud) / 100.0))
        ui = np.array([delta_cmd, float(rpm), float(thruster_rpm)], dtype=float)

        # 4th-order Runge-Kutta integration.
        k1 = self._derivatives(x0, ui)
        k2 = self._derivatives(x0 + 0.5 * dt * k1, ui)
        k3 = self._derivatives(x0 + 0.5 * dt * k2, ui)
        k4 = self._derivatives(x0 + dt * k3, ui)
        x1 = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Physical clipping / angle wrapping.
        x1[0] = float(np.clip(x1[0], -MAX_U, MAX_U))
        x1[1] = float(np.clip(x1[1], -MAX_V, MAX_V))
        x1[2] = float(np.clip(x1[2], -MAX_P, MAX_P))
        x1[3] = float(np.clip(x1[3], -MAX_R, MAX_R))
        x1[6] = float(np.clip(x1[6], -MAX_PHI, MAX_PHI))
        x1[7] = math.atan2(math.sin(x1[7]), math.cos(x1[7]))
        x1[8] = float(np.clip(x1[8], -math.radians(MAX_RUDDER_DEG), math.radians(MAX_RUDDER_DEG)))

        s_prev = self.state
        s_new = State.from_vector(x1)
        self.state = s_new

        self._v = s_new.u
        self._v_sway = s_new.v
        self._p = s_new.p
        self._w = s_new.r
        self._h = s_new.psi
        self._delta = s_new.delta

        dx = s_new.x - s_prev.x
        dy = s_new.y - s_prev.y
        heading_deg = math.degrees(s_new.psi)
        yaw_rate_degps = math.degrees(s_new.r)
        return dx, dy, heading_deg, yaw_rate_degps

    # ------------------------------------------------------------------
    # Internal dynamics: closely follows Bluefin4DOFModel02.m
    # ------------------------------------------------------------------
    def _derivatives(self, x: np.ndarray, ui: np.ndarray) -> np.ndarray:
        # Unpack state
        u, v, p, r, xpos, ypos, phi, psi, delta, n1, n2 = map(float, x)
        delta_c, n1_cmd_rpm, n2_cmd_rpm = map(float, ui)

        # Effective speed for non-dimensionalization.
        U = max(math.hypot(u, v), U_EPS)
        ud = u / U
        vd = v / U
        pd = p * LPP / U
        rd = r * LPP / U
        beta = -math.asin(np.clip(v / U, -1.0, 1.0))

        # Ship / interaction parameters from the MATLAB candidate.
        onet = 0.859
        onew = 0.806
        onetR = 0.857
        oneaH = 1.403
        aH = oneaH - 1.0
        xH = -0.646
        gR0 = 0.394
        cg = -0.53
        gR = gR0 * (1.0 + cg * abs(phi))
        ldR = -0.795
        epsi = 0.740
        kappa = 0.810
        eta_h = 0.140

        # Hydrodynamic coefficients (already Bluefin-adjusted in MATLAB file).
        Xd0 = -0.0212
        cx0 = -0.02
        Xdrph = 0.0092
        Xdbb = -0.0348
        cxbb = 2.10
        Xdbrmdy = -0.0957
        Xdrr = -0.0070
        cxrr = 3.74
        Xdbbbb = -0.0018

        Ydph = 0.0053
        Ydb = 0.2501
        cyb = -0.14
        Ydrmdx = 0.0346
        cyr = -0.61
        Ydbbph = -0.2979
        Ydbrph = 0.6308
        Ydrrph = -0.0854
        Ydbbb = 2.6087
        Ydbbr = -1.7091
        Ydbrr = 1.1682
        Ydrrr = -0.0461

        Kdph = -0.0185
        Kdb = -0.2586
        Kdr = 0.0532
        Kdbbph = 0.2229
        Kdbrph = 0.5374
        Kdrrph = -0.0928
        Kdbbb = -0.7293
        Kdbbr = 1.1474
        Kdbrr = -0.3351
        Kdrrr = -0.0132

        Ndph = -0.0086
        Ndb = 0.0966
        cnb = 0.22
        Ndr = -0.0513
        cnr = -0.62
        Ndbbph = -0.2510
        Ndbrph = 0.0722
        Ndrrph = -0.0172
        Ndbbb = 0.4218
        Ndbbr = -0.8629
        Ndbrr = 0.1459
        Ndrrr = -0.0439

        # Effective inertias
        m11 = MASS + MX
        m22 = MASS + MY
        m33 = IX + JX
        m44 = IZ + JZ

        # Rudder saturation and dynamics
        delta_lim = math.radians(MAX_RUDDER_DEG)
        delta_rate_lim = math.radians(MAX_RUDDER_RATE_DPS)
        delta_c = float(np.clip(delta_c, -delta_lim, delta_lim))
        delta_dot = np.clip(delta_c - delta, -delta_rate_lim, delta_rate_lim)

        # Shaft-speed dynamics (rpm command -> rps state)
        n1_cmd = n1_cmd_rpm / 60.0
        n2_cmd = n2_cmd_rpm / 60.0
        n1_dot = float(np.clip(n1_cmd - n1, -MAX_SHAFT_RPS, MAX_SHAFT_RPS))
        n2_dot = float(np.clip(n2_cmd - n2, -MAX_SHAFT_RPS, MAX_SHAFT_RPS))

        # Propeller model
        if abs(n1_cmd) < J_EPS or abs(u) < U_EPS:
            J = 0.0
            KT = 0.3267
            XdP = 0.0
        else:
            J = onew * u / (n1_cmd * DP)
            KT = 0.3267 - 0.2297 * J - 0.1607 * J * J
            XdP = PROP_FORCE_SCALE * abs(n1_cmd) * n1_cmd * onet * KT * (DP ** 4) / (0.5 * LPP * DRAFT * U * U)

        # Rudder inflow / rudder force model
        if abs(n1_cmd) < J_EPS:
            udR = max(epsi * onew * abs(ud), J_EPS)
        else:
            J_eff = max(abs(J), J_EPS)
            root_term = max(1.0 + 8.0 * KT / (math.pi * J_eff * J_eff), SQRT_EPS)
            udR = epsi * onew * math.sqrt(eta_h * ((1.0 + kappa * (math.sqrt(root_term) - 1.0)) ** 2) + (1.0 - eta_h))
        vdR = -gR * (beta - ldR * rd + (p * (ZR - ZG) / max(U, U_EPS)))
        UdR = math.sqrt(max(udR * udR + vdR * vdR, SQRT_EPS))
        alphaR = delta - math.atan2(-vdR, udR)
        ARpLd = AR / (LPP * DRAFT)
        FdN = -RUDDER_FORCE_SCALE * (ARpLd) * (6.13 * 1.4697 / (2.25 + 1.4697)) * UdR * UdR * math.sin(alphaR)

        XdR = (onetR) * FdN * math.sin(delta) * math.cos(phi)
        YdR = (1.0 + aH) * FdN * math.cos(delta) * math.cos(phi)
        KdR = ZR * YdR / LPP
        NdR = RUDDER_YAW_SCALE * (XR_POS + aH * xH) * FdN * math.cos(delta) * math.cos(phi)

        # Hull forces / moments
        XdH = (
            Xd0 * (1.0 + cx0 * abs(phi))
            + Xdrph * rd * phi
            + Xdbb * (1.0 + cxbb * abs(phi)) * beta * beta
            + Xdbrmdy * beta * rd
            + Xdrr * (1.0 + cxrr * abs(phi)) * rd * rd
            + Xdbbbb * beta ** 4
        )
        YdH = (
            Ydph * phi
            + Ydb * (1.0 + cyb * abs(phi)) * beta
            + Ydrmdx * (1.0 + cyr * abs(phi)) * rd
            + Ydbbph * beta * beta * phi
            + Ydbrph * beta * rd * phi
            + Ydrrph * rd * rd * phi
            + Ydbbb * beta ** 3
            + Ydbbr * beta * beta * rd
            + Ydbrr * beta * rd * rd
            + Ydrrr * rd ** 3
        )
        KdH = (
            Kdph * phi
            + Kdb * beta
            + Kdr * rd
            + Kdbbph * beta * beta * phi
            + Kdbrph * beta * rd * phi
            + Kdrrph * rd * rd * phi
            + Kdbbb * beta ** 3
            + Kdbbr * beta * beta * rd
            + Kdbrr * beta * rd * rd
            + Kdrrr * rd ** 3
        )
        NdH = (
            Ndph * phi
            + Ndb * (1.0 + cnb * abs(phi)) * beta
            + Ndr * (1.0 + cnr * abs(phi)) * rd
            + Ndbbph * beta * beta * phi
            + Ndbrph * beta * rd * phi
            + Ndrrph * rd * rd * phi
            + Ndbbb * beta ** 3
            + Ndbbr * beta * beta * rd
            + Ndbrr * beta * rd * rd
            + Ndrrr * rd ** 3
        )

        XdH *= HULL_FORCE_SCALE
        YdH *= HULL_FORCE_SCALE
        KdH *= HULL_FORCE_SCALE
        NdH *= HULL_FORCE_SCALE

        # Roll restoring / damping
        C44 = G * MASS * GM
        a_roll = 0.5
        B44 = 2.0 * a_roll / math.pi * math.sqrt(max(G * MASS * GM * (IX + JX), SQRT_EPS))
        vdot_tmp = (YdH * (0.5 * RHO * LPP * DRAFT * U * U) - m11 * u * r) / max(m22, SQRT_EPS)
        KdH2 = ZG * YdH - B44 * p - C44 * phi - (ZR - ZG) * YdR
        Kd_total = ROLL_MOMENT_SCALE * (KdH + KdH2 + KdR)

        # Bow thruster
        xB = 0.45
        zB = -0.05
        KBT = 0.026
        FBT = BOW_THRUSTER_SCALE * abs(n2_cmd) * n2_cmd * KBT / max(0.5 * RHO * LPP * DRAFT * U * U, SQRT_EPS)
        YdB = FBT
        KdB = zB * FBT / LPP
        NbB = xB * FBT / LPP

        # Overall non-dimensional forces/moments
        Xd = XdH + XdP + XdR
        Yd = YdH + YdR + YdB
        Kd = Kd_total + KdB
        Nd = YAW_MOMENT_SCALE * (NdH + NdR - (-0.1) * Yd + NbB)

        # Dimensional accelerations
        Xu = Xd * (0.5 * RHO * LPP * DRAFT * U * U)
        Yu = Yd * (0.5 * RHO * LPP * DRAFT * U * U)
        Ku = Kd * (0.5 * RHO * LPP * (DRAFT ** 2) * U * U)
        Nu = Nd * (0.5 * RHO * (LPP ** 2) * DRAFT * U * U)

        u_dot = (Xu + m22 * v * r) / max(m11, SQRT_EPS)
        v_dot = (Yu - m11 * u * r) / max(m22, SQRT_EPS)
        p_dot = (Ku + (ZH - ZG) * (MY * v_dot + MX * u * r)) / max(m33, SQRT_EPS)
        r_dot = Nu / max(m44, SQRT_EPS)

        # Kinematics
        x_dot = math.cos(psi) * u - math.sin(psi) * v * math.cos(phi)
        y_dot = math.sin(psi) * u + math.cos(psi) * v * math.cos(phi)
        phi_dot = p
        psi_dot = r * math.cos(phi)

        return np.array(
            [u_dot, v_dot, p_dot, r_dot, x_dot, y_dot, phi_dot, psi_dot, delta_dot, n1_dot, n2_dot],
            dtype=float,
        )


if __name__ == "__main__":
    model = ShipModel()
    dt = 0.1
    for _ in range(200):
        model.update(600.0, 25.0, dt)
    print(model.state_dict())
