"""Numerically guarded Python port of ``Bluefin4DOFModel02.m``.

This adapter keeps the current repo interface:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)

The underlying equations follow the MATLAB file closely, but with a few
practical adjustments so the model can run from rest inside the Python
validation code:

- floors on flow speed and advance ratio to avoid divide-by-zero
- RK4 integration instead of Euler
- a command-scale bridge from the repo's simplified ``rpm`` input to the
  MATLAB model's propeller-rpm input

It is therefore best read as a *direct runnable adapter* of the MATLAB model,
not as a claim that the original file was already production-ready.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

VESSEL_LENGTH = 1.725
VESSEL_WIDTH = 0.50
HULL_MARGIN = 0.15
HULL_FORWARD_SHIFT = 0.0
LIDAR_OFFSET_M = VESSEL_LENGTH / 2.0

MASS = 64.55
MAX_RUD_ANGLE = 40.0
MAX_RUD_RATE_DPS = 20.0

# Bridge between the simplified repo command and the MATLAB input units.
RPM_COMMAND_SCALE = 60.0
THRUSTER_COMMAND_SCALE = 60.0

MIN_FLOW_SPEED = 0.05
MIN_ADVANCE_RATIO = 1e-4
MAX_SURGE_SPEED = 5.0
MAX_SWAY_SPEED = 3.0
MAX_ROLL_RATE_RAD = math.radians(180.0)
MAX_YAW_RATE_RAD = math.radians(180.0)
MAX_ROLL_ANGLE_RAD = math.radians(45.0)
MAX_SHAFT_RPS = 1000.0 / 60.0


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
        self._n1 = 0.0
        self._n2 = 0.0

        # Compatibility fields expected elsewhere in the repo.
        self._v_sway = 0.0
        self._w = 0.0
        self._h = 0.0

    def state_dict(self) -> Dict[str, float]:
        return {
            "u_body_mps": float(self._u),
            "v_body_mps": float(self._v),
            "roll_rate_radps": float(self._p),
            "yaw_rate_radps": float(self._r),
            "yaw_rate_degps": float(math.degrees(self._r)),
            "roll_deg": float(math.degrees(self._phi)),
            "heading_rad": float(self._psi),
            "heading_deg": float(math.degrees(self._psi) % 360.0),
            "rudder_deg": float(math.degrees(self._delta)),
            "prop_rps": float(self._n1),
            "thruster_rps": float(self._n2),
            "x_m": float(self._x),
            "y_m": float(self._y),
        }

    def update(
        self,
        rpm: float,
        rud: float,
        dt: float,
        *,
        thruster_rpm: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        s0 = self._state_vector()
        x_prev = self._x
        y_prev = self._y

        delta_cmd = float(np.clip(rud, -100.0, 100.0)) / 100.0 * math.radians(MAX_RUD_ANGLE)
        n1_cmd_rpm = max(float(rpm), 0.0) * RPM_COMMAND_SCALE
        n2_cmd_rpm = float(thruster_rpm) * THRUSTER_COMMAND_SCALE

        k1 = self._derivatives(s0, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        k2 = self._derivatives(s0 + 0.5 * dt * k1, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        k3 = self._derivatives(s0 + 0.5 * dt * k2, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        k4 = self._derivatives(s0 + dt * k3, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
        s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        s1[0] = float(np.clip(s1[0], -1.0, MAX_SURGE_SPEED))
        s1[1] = float(np.clip(s1[1], -MAX_SWAY_SPEED, MAX_SWAY_SPEED))
        s1[2] = float(np.clip(s1[2], -MAX_ROLL_RATE_RAD, MAX_ROLL_RATE_RAD))
        s1[3] = float(np.clip(s1[3], -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD))
        s1[6] = float(np.clip(s1[6], -MAX_ROLL_ANGLE_RAD, MAX_ROLL_ANGLE_RAD))
        s1[8] = float(np.clip(s1[8], -math.radians(MAX_RUD_ANGLE), math.radians(MAX_RUD_ANGLE)))
        s1[9] = float(np.clip(s1[9], -MAX_SHAFT_RPS, MAX_SHAFT_RPS))
        s1[10] = float(np.clip(s1[10], -MAX_SHAFT_RPS, MAX_SHAFT_RPS))

        self._set_state_vector(s1)
        dx = self._x - x_prev
        dy = self._y - y_prev
        heading_deg = math.degrees(self._psi) % 360.0
        yaw_rate_degps = math.degrees(self._r)
        return dx, dy, heading_deg, yaw_rate_degps

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

    def _derivatives(
        self,
        s: np.ndarray,
        delta_cmd: float,
        n1_cmd_rpm: float,
        n2_cmd_rpm: float,
    ) -> np.ndarray:
        u, v, p, r, xpos, ypos, phi, psi, delta, n1, n2 = [float(z) for z in s]

        l_ship = 1.725
        u_mag = max(math.hypot(u, v), MIN_FLOW_SPEED)
        drift = -math.asin(float(np.clip(v / u_mag, -1.0, 1.0)))

        delta_dot = float(
            np.clip(
                delta_cmd - delta,
                -math.radians(MAX_RUD_RATE_DPS),
                math.radians(MAX_RUD_RATE_DPS),
            )
        )

        n1_target = n1_cmd_rpm / 60.0
        n2_target = n2_cmd_rpm / 60.0
        n1_dot = float(np.clip(n1_target - n1, -MAX_SHAFT_RPS, MAX_SHAFT_RPS))
        n2_dot = float(np.clip(n2_target - n2, -MAX_SHAFT_RPS, MAX_SHAFT_RPS))

        # Parameters from the MATLAB file.
        beam = 0.5
        draft = 0.193
        disp = 0.06455
        x_g = -0.1
        d_prop = 0.1
        lambda_r = 1.4697
        eta = 0.879
        area_r = 0.0091
        area_r_over_ld = area_r / (l_ship * draft)
        x_r = -1.05309
        gm = 1.87
        z_g = 0.005
        z_r = -0.01
        z_h = 0.02
        cf = 1.0

        onet = 0.859 * cf
        onew = 0.806 * cf
        onet_r = 0.857 * cf
        one_a_h = 1.403 * cf
        a_h = one_a_h - 1.0
        x_h = -0.646 * cf
        g_r0 = 0.394 * cf
        c_g = -0.53 * cf
        g_r = g_r0 * (1.0 + c_g * abs(phi)) * cf
        ld_r = -0.795 * cf
        epsi = 0.740 * cf
        kappa = 0.810 * cf
        eta_r = 0.140 * cf

        xd0 = -0.0212 * cf
        cx0 = -0.02 * cf
        xdrph = 0.0092 * cf
        xdbb = -0.0348 * cf
        cxbb = 2.10 * cf
        xdbrmdy = -0.0957 * cf
        xdrr = -0.0070 * cf
        cxrr = 3.74 * cf
        xdbbbb = -0.0018 * cf

        ydph = 0.0053 * cf
        ydb = 0.2501 * cf
        cyb = -0.14 * cf
        ydrmdx = 0.0346 * cf
        cyr = -0.61 * cf
        ydbbph = -0.2979 * cf
        ydbrph = 0.6308 * cf
        ydrrph = -0.0854 * cf
        ydbbb = 2.6087 * cf
        ydbbr = -1.7091 * cf
        ydbrr = 1.1682 * cf
        ydrrr = -0.0461 * cf

        kdph = -0.0185 * cf
        kdb = -0.2586 * cf
        kdr = 0.0532 * cf
        kdbbph = 0.2229 * cf
        kdbrph = 0.5374 * cf
        kdrrph = -0.0928 * cf
        kdbbb = -0.7293 * cf
        kdbbr = 1.1474 * cf
        kdbrr = -0.3351 * cf
        kdrrr = -0.0132 * cf

        ndph = -0.0086 * cf
        ndb = 0.0966 * cf
        cnb = 0.22 * cf
        ndr = -0.0513 * cf
        cnr = -0.62 * cf
        ndbbph = -0.2510 * cf
        ndbrph = 0.0722 * cf
        ndrrph = -0.0172 * cf
        ndbbb = 0.4218 * cf
        ndbbr = -0.8629 * cf
        ndbrr = 0.1459 * cf
        ndrrr = -0.0439 * cf

        rho = 1000.0
        g = 9.81
        m = 64.55
        mx = 3.662
        my = 62.7366
        i_x = 0.567
        i_z = 9.6038
        j_x = 0.6309
        j_z = 10.2347

        m11 = m + mx
        m22 = m + my
        m33 = i_x + j_x
        m44 = i_z + j_z

        ud = u / u_mag
        vd = v / u_mag
        rd = r * l_ship / u_mag

        if abs(n1_target) < MIN_ADVANCE_RATIO:
            j_adv = 0.0
            kt = 0.0
            xd_p = 0.0
            ud_r = epsi * onew
        else:
            j_adv = onew * u / max(abs(n1_target) * d_prop, MIN_ADVANCE_RATIO)
            a0, a1, a2 = 0.3267, -0.2297, -0.1607
            kt = a0 + a1 * j_adv + a2 * j_adv * j_adv
            xd_p = abs(n1_target) * n1_target * onet * kt * (d_prop**4) / (0.5 * l_ship * draft * u_mag * u_mag)
            j_sq = max(j_adv * j_adv, MIN_ADVANCE_RATIO * MIN_ADVANCE_RATIO)
            prop_term = max(1.0 + 8.0 * kt / (math.pi * j_sq), 0.0)
            ud_r = epsi * onew * math.sqrt(eta_r * ((1.0 + kappa * math.sqrt(prop_term) - 1.0) ** 2) + (1.0 - eta_r))

        vd_r = -g_r * (drift - ld_r * rd + (p * (z_r - z_g) / u_mag))
        ud_total_r = math.hypot(ud_r, vd_r)
        alpha_r = delta - math.atan2(-vd_r, ud_r)
        fd_n = -(area_r_over_ld) * (6.13 * lambda_r / (2.25 + lambda_r)) * (ud_total_r**2) * math.sin(alpha_r)

        xd_r = onet_r * fd_n * math.sin(delta) * math.cos(phi)
        yd_r = (1.0 + a_h) * fd_n * math.cos(delta) * math.cos(phi)
        kd_r = z_r * yd_r / l_ship
        nd_r = (x_r + a_h * x_h) * fd_n * math.cos(delta) * math.cos(phi)

        xd_h = (
            xd0 * (1.0 + cx0 * abs(phi))
            + xdrph * rd * phi
            + xdbb * (1.0 + cxbb * abs(phi)) * drift * drift
            + xdbrmdy * drift * rd
            + xdrr * (1.0 + cxrr * abs(phi)) * rd * rd
            + xdbbbb * drift**4
        )
        yd_h = (
            ydph * phi
            + ydb * (1.0 + cyb * abs(phi)) * drift
            + ydrmdx * (1.0 + cyr * abs(phi)) * rd
            + ydbbph * drift * drift * phi
            + ydbrph * drift * rd * phi
            + ydrrph * rd * rd * phi
            + ydbbb * drift**3
            + ydbbr * drift * drift * rd
            + ydbrr * drift * rd * rd
            + ydrrr * rd**3
        )
        kd_h = (
            kdph * phi
            + kdb * drift
            + kdr * rd
            + kdbbph * drift * drift * phi
            + kdbrph * drift * rd * phi
            + kdrrph * rd * rd * phi
            + kdbbb * drift**3
            + kdbbr * drift * drift * rd
            + kdbrr * drift * rd * rd
            + kdrrr * rd**3
        )
        nd_h = (
            ndph * phi
            + ndb * (1.0 + cnb * abs(phi)) * drift
            + ndr * (1.0 + cnr * abs(phi)) * rd
            + ndbbph * drift * drift * phi
            + ndbrph * drift * rd * phi
            + ndrrph * rd * rd * phi
            + ndbbb * drift**3
            + ndbbr * drift * drift * rd
            + ndbrr * drift * rd * rd
            + ndrrr * rd**3
        )

        c44 = g * m * gm
        damping_a = 0.5
        b44 = 2.0 * damping_a / math.pi * math.sqrt(max(g * m * gm * (i_x + j_x), 0.0))
        kd_h2 = z_g * yd_h - b44 * p - c44 * phi - (z_r - z_g) * yd_r

        x_b = 0.45
        z_b = -0.05
        k_bt = 0.026
        f_bt = abs(n2_target) * n2_target * k_bt / (0.5 * rho * l_ship * draft * u_mag * u_mag)
        yd_b = f_bt
        kd_b = z_b * f_bt / l_ship
        nb_b = x_b * f_bt / l_ship

        xd = xd_h + xd_p + xd_r
        yd = yd_h + yd_r + yd_b
        kd = kd_h + kd_h2 + kd_r + kd_b
        nd = nd_h + nd_r - x_g * yd + nb_b

        force_scale_x = 0.5 * rho * l_ship * draft * u_mag * u_mag
        force_scale_n = 0.5 * rho * l_ship * l_ship * draft * u_mag * u_mag
        force_scale_k = 0.5 * rho * l_ship * draft * draft * u_mag * u_mag

        vdot = (yd * force_scale_x - m11 * u * r) / m22
        udot = (xd * force_scale_x + m22 * v * r) / m11
        pdot = (kd * force_scale_k + (z_h - z_g) * (my * vdot + mx * u * r)) / m33
        rdot = nd * force_scale_n / m44
        xdot = math.cos(psi) * u - math.sin(psi) * v * math.cos(phi)
        ydot = math.sin(psi) * u + math.cos(psi) * v * math.cos(phi)
        phidot = p
        psidot = r * math.cos(phi)

        return np.array(
            [
                udot,
                vdot,
                pdot,
                rdot,
                xdot,
                ydot,
                phidot,
                psidot,
                delta_dot,
                n1_dot,
                n2_dot,
            ],
            dtype=float,
        )

