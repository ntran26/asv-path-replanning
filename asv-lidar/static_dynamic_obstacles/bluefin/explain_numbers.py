"""Derived numbers quoted in `MODEL_EXPLAINED.md`, computed from the frozen v3 model.

Reads `dynamics.py`, `ship_model_v3.py` and `../src/ship.py` without changing them, and prints:
the mass matrix and Coriolis coefficients, the rudder geometry and its effective lever arm,
the equivalent nondimensional (MMG) hull derivatives, steady speeds, the force balance in a
steady full-helm turn (including the drift angle at midship and at the bow), the linear yaw
response at cruise (first-order Nomoto fit), the yaw decay with the rudder centred at speed and
at rest, and the bootstrap spread of the randomisation draws.

    python bluefin/explain_numbers.py > bluefin/explain_numbers.txt
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "src"), HERE]

import dynamics as dyn  # noqa: E402
import ship_model_v3 as v3  # noqa: E402
import ship  # noqa: E402

P = dict(v3.IDENTIFIED)
PA = {k: np.array([v]) for k, v in P.items()}
RHO, L, D = dyn.RHO, dyn.L, dyn.DRAFT


def section(title):
    print()
    print(title)
    print("-" * len(title))


def forces(s, rpm, p=PA):
    """Force and moment components at state s (7, 1), as in dynamics.derivatives."""
    u, v, r, psi, delta, x, y = s[:, 0]
    ue = max(u, 0.0)
    f_n = float(dyn.rudder_normal_force(np.array([u]), np.array([v]), np.array([r]),
                                        np.array([delta]), np.array([rpm]), p)[0])
    cd = math.cos(delta)
    out = {
        "f_n": f_n,
        "X_prop": float(dyn.thrust(np.array([rpm]), np.array([u]), p)[0]),
        "X_uu": -P["X_uu"] * ue * abs(ue),
        "X_vv": -P["X_vv"] * v * v,
        "X_rr": -P["X_rr"] * (L * r) ** 2,
        "X_rud": -P["X_delta"] * abs(f_n) * abs(math.sin(delta)),
        "X_cor": dyn.M22 * v * r,
        "Y_v": -P["Y_v"] * ue * v,
        "Y_vv": -P["Y_vv"] * v * abs(v),
        "Y_rud": -(1.0 + dyn.A_H) * P["k_R"] * f_n * cd,
        "Y_cor": -dyn.M11 * ue * r,
        "N_r": -P["N_r"] * ue * r,
        "N_rr": -P["N_rr"] * r * abs(r),
        "N_uv": -P["N_uv"] * ue * v,
        "N_munk": (dyn.M11 - dyn.M22) * ue * v,
        "N_rud": -dyn.RUD_ARM * P["k_R"] * f_n * cd,
    }
    return out


def run_model(rpm, rud_pct, seconds, dt=0.05, m=None):
    m = ship.ShipModel() if m is None else m
    for _ in range(int(round(seconds / dt))):
        m.update(rpm, rud_pct, dt)
    return m


def main():
    section("Mass and inertia (Blue02.m values kept in dynamics.py)")
    print(f"m = {dyn.MASS} kg, X_udot-equivalent m_x = {dyn.M_X} kg, Y_vdot-equivalent m_y = {dyn.M_Y} kg, "
          f"I_z + J_z = {dyn.IZ_JZ} kg m^2")
    print(f"M = diag(m11, m22, m33) = diag({dyn.M11:.3f}, {dyn.M22:.3f}, {dyn.M33:.4f})")
    print(f"Munk moment coefficient (m11 - m22) = {dyn.M11 - dyn.M22:.3f} kg")
    print(f"L = {L} m, B = {dyn.VESSEL_WIDTH} m, draft d = {D} m; rudder area A_R = {dyn.A_R} m^2 "
          f"(A_R/(L d) = {dyn.A_R / (L * D) * 100:.2f} %)")

    section("Rudder geometry and the sign of the rudder sway force")
    x_eff = -(dyn.X_RUD + dyn.A_H * dyn.X_H) / (1.0 + dyn.A_H)
    print(f"x_R = {dyn.X_RUD} m, x_H = {dyn.X_H} m, a_H = {dyn.A_H}; |x_R + a_H x_H| = {dyn.RUD_ARM:.4f} m")
    print(f"MMG (Yasukawa and Yoshimura 2015): Y_R = -(1 + a_H) F_N cos(delta), "
          f"N_R = -(x_R + a_H x_H) F_N cos(delta)")
    print(f"  -> rudder force acts at x = (x_R + a_H x_H)/(1 + a_H) = {-x_eff:+.3f} m (aft of midship): "
          f"Y_R and N_R of opposite sign")
    print(f"dynamics.py: N_R = -|x_R + a_H x_H| k_R F_N cos(delta)")
    print(f"  -> equivalent point of action x = +{dyn.RUD_ARM / (1 + dyn.A_H):.3f} m (forward of midship): "
          f"Y_R and N_R of the same sign")
    print("Blue02.m line 117: NR = -(0.646875 + 0.443853*0.7569)*FN*cos(del), same sign as YR (line 116)")

    section("Rudder normal force at full helm (40 deg), straight running, no drift")
    for rpm in (6.0, 12.0):
        u = ship.steady_speed(rpm)
        s = np.zeros((7, 1)); s[0, 0] = u; s[4, 0] = math.radians(40.0)
        f = forces(s, rpm)
        u_r = (1 - dyn.W_R) * u + P["k_race"] * math.sqrt(rpm / 12.0)
        print(f"RPM {rpm:4.1f}: u = {u:.3f} m/s, rudder inflow u_R = {u_r:.3f} m/s, F_N = {f['f_n']:.2f} N, "
              f"k_R F_N = {P['k_R'] * f['f_n']:.2f} N, Y_R = {f['Y_rud']:.2f} N, N_R = {f['N_rud']:.2f} N m")

    section("Surge: thrust and steady speed")
    print(f"T(rpm) = T12 (rpm/12)^2 = {P['T12']:.4f} (rpm/12)^2 N (t_boost = {P['t_boost']})")
    print(f"steady straight speed = (rpm/12) sqrt(T12/X_uu) = (rpm/12) x {math.sqrt(P['T12'] / P['X_uu']):.4f} m/s"
          f" = {math.sqrt(P['T12'] / P['X_uu']) / 12:.4f} m/s per rpm-unit")
    for rpm in (4.5, 6.0, 7.5, 12.0, 24.0):
        print(f"  RPM {rpm:4.1f}: thrust {ship.forward_thrust(rpm):6.2f} N, steady speed {ship.steady_speed(rpm):.3f} m/s")
    print(f"coast-down time constant at speed u: M11/(X_uu u) = {dyn.M11 / P['X_uu']:.2f}/u s")
    print(f"braking force of the simulator's full astern (S2 = -100 -> -24 rpm-units): "
          f"{ship.braking_thrust(-24.0):.2f} N; at -12 rpm-units: {ship.braking_thrust(-12.0):.2f} N "
          f"(REVERSE_THRUST_EFFICIENCY = {ship.REVERSE_THRUST_EFFICIENCY})")

    section("Equivalent nondimensional hull coefficients (MMG scaling 0.5 rho L d U^2 and 0.5 rho L^2 d U^2)")
    qY = 0.5 * RHO * L * D
    qN = 0.5 * RHO * L ** 2 * D
    print(f"0.5 rho L d = {qY:.2f} kg/m, 0.5 rho L^2 d = {qN:.2f} kg, 0.5 rho L^3 d = {qN * L:.2f} kg m")
    print(f"-Y_v u v      -> Y'_v  = -Y_v/(0.5 rho L d)     = {-P['Y_v'] / qY:+.4f}")
    print(f"-Y_vv v|v|    -> cross-flow drag coefficient C_D = Y_vv/(0.5 rho L d) = {P['Y_vv'] / qY:.3f}")
    print(f"-N_r u r      -> N'_r  = -N_r/(0.5 rho L^3 d)   = {-P['N_r'] / (qN * L):+.5f}")
    net = -P["N_uv"] + dyn.M11 - dyn.M22
    print(f"drift moment, total coefficient on u v: -N_uv + (m11 - m22) = {-P['N_uv']:+.2f} {dyn.M11 - dyn.M22:+.2f} "
          f"= {net:+.2f} kg -> N'_v equivalent = {net / qN:+.4f}")
    print(f"-X_uu u|u|    -> C_T = X_uu/(0.5 rho S) with S = 0.7614 m^2 (Blue02.m Sw) = {P['X_uu'] / (0.5 * RHO * 0.7614):.4f}")

    section("Steady full-helm turn (60 s from straight running), force balance")
    for rpm in (6.0, 12.0):
        for helm in (100.0, -100.0):
            m = ship.ShipModel()
            run_model(rpm, 0.0, 60.0, m=m)
            u0 = m.u
            run_model(rpm, helm, 60.0, m=m)
            s = m._s.copy()
            u, v, r = s[0, 0], s[1, 0], s[2, 0]
            f = forces(s, rpm)
            beta_cg = math.degrees(math.atan2(-v, u))
            v_bow = v + dyn.LIDAR_OFFSET_M * r
            beta_bow = math.degrees(math.atan2(-v_bow, u))
            R = math.hypot(u, v) / abs(r)
            print(f"RPM {rpm:4.1f} helm {helm:+6.1f} %: u {u:.3f} m/s (ratio {u / u0:.3f}), v {v:+.4f} m/s, "
                  f"r {math.degrees(r):+.2f} deg/s, R {R:.2f} m, rudder {m.rudder_deg:+.1f} deg")
            print(f"   drift angle beta = atan(-v/u): midship {beta_cg:+.2f} deg, bow (x = +{dyn.LIDAR_OFFSET_M} m) "
                  f"{beta_bow:+.2f} deg  (positive: velocity to port of the heading)")
            u_r = max(dyn.MIN_FLOW_SPEED, (1 - dyn.W_R) * u + P["k_race"] * math.sqrt(rpm / 12.0))
            alpha = s[4, 0] - math.atan2(v + dyn.L_R * r, u_r)
            print(f"   rudder angle {math.degrees(s[4, 0]):+.1f} deg, inflow angle {math.degrees(math.atan2(v + dyn.L_R * r, u_r)):+.1f} deg, "
                  f"angle of attack {math.degrees(alpha):+.1f} deg; a stern rudder (MMG sign) would leave the hull "
                  f"{-f['Y_cor'] + f['Y_rud']:.2f} N of sway force to supply instead of {-f['Y_cor'] - f['Y_rud']:.2f} N")
            print(f"   sway: Y_R {f['Y_rud']:+.2f}, Y_v {f['Y_v']:+.2f}, Y_vv {f['Y_vv']:+.2f}, "
                  f"Coriolis -m11 u r {f['Y_cor']:+.2f} N")
            print(f"   yaw:  N_R {f['N_rud']:+.2f}, N_r {f['N_r']:+.3f}, N_rr {f['N_rr']:+.4f}, N_uv {f['N_uv']:+.2f}, "
                  f"Munk {f['N_munk']:+.2f} N m")
            print(f"   surge: thrust {f['X_prop']:+.2f}, X_uu {f['X_uu']:+.2f}, X_vv {f['X_vv']:+.3f}, "
                  f"X_rr {f['X_rr']:+.3f}, X_rud {f['X_rud']:+.3f}, Coriolis m22 v r {f['X_cor']:+.3f} N")

    section("Linear yaw response at cruise: 10 % helm step at RPM 6, first-order (Nomoto) fit")
    for rpm in (6.0, 12.0):
        m = ship.ShipModel()
        run_model(rpm, 0.0, 60.0, m=m)
        t, rr = [], []
        for k in range(int(40 / 0.05)):
            m.update(rpm, 10.0, 0.05)
            t.append((k + 1) * 0.05); rr.append(m.yaw_rate)
        t, rr = np.array(t), np.array(rr)
        r_ss = rr[-1]
        delta_ss = math.radians(4.0)
        K = r_ss / delta_ss
        # time constant: from the end of the transport delay to 63 % of the final value
        t63 = t[np.argmax(np.abs(rr) >= 0.632 * abs(r_ss))]
        delay = float(np.round(P["rud_delay"] / 0.05) * 0.05)
        print(f"RPM {rpm:4.1f}: K = r_ss/delta = {abs(K):.3f} 1/s (K L/U = {abs(K) * L / ship.steady_speed(rpm):.3f}), "
              f"63 % at t = {t63:.2f} s "
              f"(transport delay {delay:.2f} s, servo lag {P['rud_tau']:.2f} s), "
              f"apparent T = {t63 - delay:.2f} s including the servo lag")

    section("Yaw decay with the rudder centred (from the RPM-12 full-helm turn)")
    for rpm_after, label in ((12.0, "thrust held at RPM 12"), (0.0, "thrust cut to 0"), (-24.0, "full astern (efficiency 0.5)")):
        m = ship.ShipModel()
        run_model(12.0, 0.0, 60.0, m=m)
        run_model(12.0, 100.0, 60.0, m=m)
        r0, h0 = m.yaw_rate, m.heading_deg
        t, dpsi, prev = 0.0, 0.0, h0
        t_half = None
        while t < 120.0:
            m.update(rpm_after, 0.0, 0.05)
            t += 0.05
            dpsi += ((m.heading_deg - prev + 540.0) % 360.0) - 180.0
            prev = m.heading_deg
            if t_half is None and abs(m.yaw_rate) <= 0.5 * abs(r0):
                t_half = t
            if abs(m.yaw_rate) < math.radians(0.5) and t > 2.0:
                break
        print(f"{label:30s}: r from {math.degrees(r0):.1f} deg/s, half after {t_half:.1f} s, "
              f"below 0.5 deg/s after {t:.1f} s, heading change {abs(dpsi):.0f} deg, final u {m.u:.2f} m/s")
    print("yaw damping left at zero surge: N_rr r|r| only; at r = 10 deg/s it is "
          f"{P['N_rr'] * math.radians(10) ** 2 * 1000:.3f} mN m against M33 = {dyn.M33} kg m^2")

    section("Randomisation: bootstrap vectors and sample_params(scale = 1.0)")
    B = v3.BOOTSTRAP
    print(f"{B.shape[0]} bootstrap vectors of {B.shape[1]} parameters")
    rng = np.random.default_rng(0)
    draws = [v3.sample_params(rng, scale=1.0) for _ in range(2000)]
    for k in ("T12", "X_uu", "k_R", "N_r", "N_uv", "Y_vv", "rud_delay", "rud_tau"):
        a = np.array([d[k] for d in draws])
        print(f"  {k:9s}: identified {P[k]:9.4f}; draws 5-95 % {np.percentile(a, 5):9.4f} .. {np.percentile(a, 95):9.4f}")
    sp = np.array([math.sqrt(d["T12"] / d["X_uu"]) / 2.0 for d in draws])
    print(f"  steady speed at RPM 6 over the draws: 5-95 % {np.percentile(sp, 5):.3f} .. {np.percentile(sp, 95):.3f} m/s")
    rr, RR, rat = [], [], []
    for d in draws[:200]:
        m = ship.ShipModel(params=d)
        run_model(6.0, 0.0, 40.0, m=m)
        u0 = m.u
        run_model(6.0, 100.0, 40.0, m=m)
        rr.append(abs(math.degrees(m.yaw_rate)))
        RR.append(math.hypot(m.u, m.v) / max(abs(m.yaw_rate), 1e-9))
        rat.append(m.u / u0)
    rr, RR, rat = np.array(rr), np.array(RR), np.array(rat)
    print(f"  full-helm steady turn at RPM 6 over 200 draws: yaw rate {rr.min():.2f} .. {rr.max():.2f} deg/s "
          f"(5-95 % {np.percentile(rr, 5):.2f} .. {np.percentile(rr, 95):.2f}), radius {RR.min():.2f} .. {RR.max():.2f} m "
          f"(5-95 % {np.percentile(RR, 5):.2f} .. {np.percentile(RR, 95):.2f}), speed ratio "
          f"{np.percentile(rat, 5):.2f} .. {np.percentile(rat, 95):.2f}")


if __name__ == "__main__":
    main()
