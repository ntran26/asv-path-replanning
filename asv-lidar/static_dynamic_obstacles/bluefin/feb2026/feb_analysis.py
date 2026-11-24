"""The February 2026 radio-control logs against the v3 plant.

    python bluefin/feb2026/feb_analysis.py > bluefin/feb2026/feb_analysis.txt

Sections
  1  inventory: line kinds, rates, basin extents
  2  the radio-control channels: which is throttle and which is rudder
  3  throttle to straight-line speed
  4  settled full-rudder turns, measured and predicted by v3 at the same speed
  5  crash stops under reverse throttle: deceleration, reverse force, head reach, heading change;
     the same manoeuvre on v3 with four reverse-force assumptions
  6  yaw decay after the rudder is centred at speed, measured and v3
  7  coast-downs at neutral throttle, measured and v3
  8  windowed replay of the forward-motion segments through v3 (the July V1 test on February data)

Every measurement comes from the 10 Hz pose (0.1 m and 0.1 deg resolution) and the 1 Hz receiver
samples; a stick movement between two receiver samples is unknown, so event times carry up to 1 s
of uncertainty. v3 is the training plant: `src/ship.py` with the identified parameters. The
functions are reused by `refit_feb.py` with other parameter sets.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUEFIN = os.path.dirname(HERE)
sys.path[:0] = [HERE, BLUEFIN, os.path.join(BLUEFIN, "..", "src")]

import feb_logs as fl  # noqa: E402
import dynamics as dyn  # noqa: E402
import ship  # noqa: E402

LOGS = (1, 2, 3, 4)
P = dict(ship.IDENTIFIED)
U_PER_RPM = math.sqrt(P["T12"] / P["X_uu"]) / 12.0          # 0.0930 m/s per rpm-unit
DT = 0.05
T24 = ship.forward_thrust(24.0)
REPLAY_SEGMENTS = ((4, 16.0, 67.5), (4, 137.0, 186.0), (3, 39.5, 55.5), (2, 10.0, 35.0),
                   (1, 59.5, 65.5), (1, 75.5, 79.0))


def kasa(x, y):
    """Algebraic circle fit: radius and rms residual."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    R = math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    return R, float(np.sqrt(np.mean((np.hypot(x - c[0], y - c[1]) - R) ** 2)))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def side_name(side):
    return "port" if side < 0 else "stbd"


# ---------------------------------------------------------------------------
# the model side
# ---------------------------------------------------------------------------
def settled_turn(rpm: float, helm: float, params=None, eff: float = 0.5, seconds: float = 60.0):
    """A plant settled in a straight run, then in a turn at `helm` (simulator percent)."""
    m = ship.ShipModel(params=params, reverse_efficiency=eff)
    for _ in range(int(seconds / DT)):
        m.update(rpm, 0.0, DT)
    u0 = m.u
    for _ in range(int(seconds / DT)):
        m.update(rpm, helm, DT)
    return m, u0


def turn_summary(m, u0) -> Dict:
    U = math.hypot(m.u, m.v)
    v_bow = m.v + dyn.LIDAR_OFFSET_M * m.yaw_rate
    return {"u0": u0, "U": U, "u": m.u, "v": m.v, "r": math.degrees(m.yaw_rate),
            "R": U / max(abs(m.yaw_rate), 1e-9),
            "beta_mid": math.degrees(math.atan2(-m.v, m.u)),
            "beta_bow": math.degrees(math.atan2(-v_bow, m.u))}


def matched_turn(U_target: float, helm: float, params=None, eff: float = 0.5):
    """The plant settled at `helm` with the rpm that gives `U_target` in the turn."""
    rpm = 12.0
    for _ in range(3):
        m, u0 = settled_turn(rpm, helm, params, eff)
        rpm *= U_target / max(math.hypot(m.u, m.v), 1e-3)
    m, u0 = settled_turn(rpm, helm, params, eff)
    return rpm, m, u0


def run_until(m, rpm, rud, t_max, stop, onset=0.0, rpm_before=0.0):
    """Advance; return (time, heading change, track length to zero surge, time of zero surge)."""
    t, psi, prev, s_x, s_y, tz = 0.0, 0.0, m.heading_deg, 0.0, 0.0, None
    while t < t_max:
        dx, dy, hd, _ = m.update(rpm_before if t < onset else rpm, rud, DT)
        t += DT
        psi += wrap180(hd - prev)
        prev = hd
        if tz is None:
            s_x += dx
            s_y += dy
            if m.u <= 1e-3:
                tz = t
        if stop(m, t, tz):
            break
    return t, psi, math.hypot(s_x, s_y), tz


# ---------------------------------------------------------------------------
# the measurement side
# ---------------------------------------------------------------------------
def load_all():
    logs = {k: fl.load(k) for k in LOGS}
    kin = {k: fl.kinematics(logs[k]) for k in LOGS}
    return logs, kin


def throttle_points(logs, kin):
    pts = []
    for k in LOGS:
        d, g = logs[k], kin[k]
        acc = np.gradient(g["U"], 0.1)
        for i in range(3, len(d["trc"])):
            w = d["s1"][i - 3:i + 1]
            if np.ptp(w) > 16 or w.mean() < 1530:
                continue
            m = (g["t"] >= d["trc"][i] - 0.5) & (g["t"] <= d["trc"][i] + 0.5)
            if m.sum() < 5 or abs(acc[m].mean()) > 0.03 or np.abs(g["r"][m]).mean() > 5 or g["U"][m].mean() < 0.3:
                continue
            pts.append((w.mean(), g["U"][m].mean(), k, d["trc"][i], np.mean(g["drift"][m])))
    pts.sort()
    g = kin[3]
    m = (g["t"] >= 49.0) & (g["t"] <= 55.5)
    top = (float(g["U"][m].mean()), float(np.polyfit(g["t"][m], g["U"][m], 1)[0]))
    bias = float(np.mean([p[4] for p in pts if p[1] > 0.5]))
    groups = {}
    for s1, U, *_ in pts:
        groups.setdefault(int(round(s1 / 10.0) * 10), []).append(U)
    tm = [(fl.S1_NEUTRAL, 0.0)] + [(float(s), float(np.mean(u))) for s, u in sorted(groups.items())]
    if tm[-1][0] < 1900:
        tm.append((1937.0, top[0]))
    return pts, top, bias, fl.ThrottleMap(tm, U_PER_RPM), tm


def measure_turns(logs, kin, bias) -> List[Dict]:
    """Segments with S2 held at an end point; first 2 s dropped."""
    turns = []
    for k in LOGS:
        d, g = logs[k], kin[k]
        s2 = d["s2"]
        i = 0
        while i < len(s2):
            side = -1 if s2[i] <= 1010 else (1 if s2[i] >= 1950 else 0)
            if side == 0:
                i += 1
                continue
            j = i
            while j + 1 < len(s2) and ((s2[j + 1] <= 1010) if side < 0 else (s2[j + 1] >= 1950)):
                j += 1
            a, b = d["trc"][i] + 2.0, d["trc"][j] + 0.5
            m = (g["t"] >= a) & (g["t"] <= b)
            rc = (d["trc"] >= d["trc"][i]) & (d["trc"] <= d["trc"][j])
            if m.sum() >= 40 and d["s1"][rc].mean() > 1540 and g["U"][m].mean() > 0.5:
                U, r = float(g["U"][m].mean()), float(g["r"][m].mean())
                dpsi = float(g["yaw"][m][-1] - g["yaw"][m][0])
                Rf = kasa(g["x"][m], g["y"][m]) if abs(dpsi) >= 180 else (float("nan"), float("nan"))
                m3 = (g["t"] >= d["trc"][i]) & (g["t"] <= d["trc"][i] + 3.0)
                turns.append(dict(k=k, side=side, a=a, b=b, s1=float(d["s1"][rc].mean()),
                                  r_peak=float(g["r"][m3][np.argmax(np.abs(g["r"][m3]))]),
                                  U_step=float(g["U"][m3][0]),
                                  s1_sd=float(d["s1"][rc].std()), U=U, r=r,
                                  R=U / math.radians(abs(r)), Rf=Rf[0], dpsi=dpsi,
                                  drift=float(np.mean(g["drift"][m])) - bias))
            i = j + 1
    return turns


def measure_stops(logs, kin) -> List[Dict]:
    """Receiver sample S1 < 1300 after one >= 1490, at U > 0.6 m/s.

    The reverse command arrived somewhere between the two receiver samples. The reverse force is
    taken only where reverse is certain: from the first reverse sample to whichever comes first of
    u = 0.15 m/s and 1 s after the last consecutive reverse sample. Time to stop, head reach and
    heading change are counted from the mid-point of the uncertain second (half a second before
    the first reverse sample) to u = 0.15 m/s (heading change to 3 s after that).
    """
    stops = []
    for k in LOGS:
        d, g = logs[k], kin[k]
        for i in range(1, len(d["trc"])):
            if not (d["s1"][i] < 1300 and d["s1"][i - 1] >= 1490):
                continue
            ti = d["trc"][i]
            if np.interp(ti - 1.0, g["t"], g["U"]) < 0.6:
                continue
            j = i
            while j + 1 < len(d["trc"]) and d["s1"][j + 1] < 1450:
                j += 1
            t_last = d["trc"][j] + 1.0
            ia = np.searchsorted(g["t"], ti)
            ib = ia
            while ib < len(g["t"]) - 1 and g["u"][ib] > 0.15 and g["t"][ib] < t_last:
                ib += 1
            seg = slice(ia, ib + 1)
            dt_ = g["t"][ib] - g["t"][ia]
            F = dyn.M11 * (g["u"][ia] - g["u"][ib]) / max(dt_, 0.1) - float(np.mean(P["X_uu"] * g["u"][seg] ** 2))
            i0 = np.searchsorted(g["t"], ti - 0.5)
            i1 = ia
            while i1 < len(g["t"]) - 1 and g["u"][i1] > 0.15:
                i1 += 1
            h0 = math.radians(g["yaw"][i0])
            reach = (g["x"][i1] - g["x"][i0]) * math.sin(h0) + (g["y"][i1] - g["y"][i0]) * math.cos(h0)
            i2 = min(len(g["t"]) - 1, i0 + 50)
            stops.append(dict(k=k, t_sample=ti, s1=d["s1"][i], s1_min=float(d["s1"][i:j + 1].min()),
                              n_rev=j - i + 1, u0=float(g["u"][i0]), r0=float(g["r"][i0]),
                              helm_prev=float(-fl.rudder_cmd_from_s2(d["s2"][i - 1])),
                              F=F, F_window=dt_, t_stop=float(g["t"][i1] - g["t"][i0]), reach=float(reach),
                              dpsi=float(g["yaw"][i2] - g["yaw"][i0]), r_stop=float(g["r"][i1])))
    return stops


def model_stop(stop: Dict, F: float, onset: float = 0.0, params=None):
    """The same manoeuvre on the model: settled at the measured speed (in a full-helm turn if the
    measured stop began in one), then rudder centred and a reverse force F (after `onset` s)."""
    if abs(stop["r0"]) > 8.0:
        helm = 100.0 if stop["r0"] > 0 else -100.0
        rpm, m, _ = matched_turn(stop["u0"], helm, params, eff=F / T24)
    else:
        rpm = stop["u0"] / U_PER_RPM
        m, _ = settled_turn(rpm, 0.0, params, eff=F / T24, seconds=60.0)
    rec = {"psi": 0.0, "prev": m.heading_deg}

    def stop(mm, t, tz):
        rec["psi"] += wrap180(mm.heading_deg - rec["prev"])
        rec["prev"] = mm.heading_deg
        if "dpsi5" not in rec and t >= 5.0 - 1e-9:
            rec["dpsi5"] = abs(rec["psi"])
        if tz is not None and "r_zero" not in rec:
            rec["r_zero"] = abs(math.degrees(mm.yaw_rate))
        return t >= 5.0 and tz is not None and abs(mm.yaw_rate) < math.radians(0.5) and abs(mm.v) < 0.02

    t, psi, reach, tz = run_until(m, -24.0, 0.0, 60.0, stop, onset=onset, rpm_before=0.0)
    return {"t_stop": tz, "reach": reach, "dpsi5": rec.get("dpsi5", abs(psi)), "dpsi60": abs(psi),
            "r_zero": rec.get("r_zero", float("nan"))}


def measure_centring(logs, kin, throttle) -> List[Dict]:
    """S2 from an end point to neutral with S1 unchanged, at U > 0.6 m/s. Heading change from the
    last end-point sample + 0.5 s to |r| < 1 deg/s or the next receiver sample with helm."""
    out = []
    for k in LOGS:
        d, g = logs[k], kin[k]
        for i in range(1, len(d["trc"])):
            if not (abs(d["s2"][i] - fl.S2_NEUTRAL) < 15 and abs(d["s2"][i - 1] - fl.S2_NEUTRAL) > 450):
                continue
            if abs(d["s1"][i] - d["s1"][i - 1]) > 30 or d["s1"][i] < 1540:
                continue
            t0 = d["trc"][i - 1] + 0.5
            i0 = np.searchsorted(g["t"], t0)
            if i0 >= len(g["t"]) - 30 or g["U"][i0] < 0.6:
                continue
            nxt = i + 1
            while nxt < len(d["trc"]) and abs(d["s2"][nxt] - fl.S2_NEUTRAL) < 15:
                nxt += 1
            t_cut = d["trc"][nxt] - 1.0 if nxt < len(d["trc"]) else g["t"][-1]
            j = i0
            while j < len(g["t"]) - 1 and abs(g["r"][j]) >= 1.0 and g["t"][j] < t_cut:
                j += 1
            out.append(dict(k=k, t0=t0, u0=float(g["u"][i0]), r0=float(g["r"][i0]),
                            helm_prev=float(-fl.rudder_cmd_from_s2(d["s2"][i - 1])),
                            rpm=float(throttle.rpm(d["s1"][i])), dpsi=float(g["yaw"][j] - g["yaw"][i0]),
                            t_settle=float(g["t"][j] - t0), cut=bool(g["t"][j] >= t_cut)))
    return out


def model_centring(ev: Dict, params=None):
    """The model settled in a full-helm turn at the measured speed, then the rudder centred."""
    helm = 100.0 if ev["r0"] > 0 else -100.0
    rpm, m, _ = matched_turn(abs(ev["u0"]), helm, params)
    r_turn = math.degrees(m.yaw_rate)
    t, psi, _, _ = run_until(m, rpm, 0.0, 30.0, lambda mm, t, tz: abs(mm.yaw_rate) < math.radians(1.0) and t > 0.5)
    return {"r_turn": r_turn, "dpsi": psi, "t_settle": t}


def naive_rms(runs, H, mode):
    """As `validate.naive_heading`: freeze the heading, or extrapolate the yaw rate, per window."""
    from simulate import _smooth_derivative
    errs = []
    for run in runs:
        psi, t = run.yaw, run.t
        r_meas = _smooth_derivative(t, psi)
        out = np.empty_like(psi)
        for k in range(len(t)):
            j = int(np.clip(np.searchsorted(t, np.floor(t[k] / H) * H, side="left"), 0, len(t) - 1))
            out[k] = psi[j] if mode == "freeze" else psi[j] + r_meas[j] * (t[k] - t[j])
        errs.append((out - psi)[1:])
    a = np.concatenate(errs)
    return float(np.sqrt(np.mean(a ** 2)))


def replay_pop(batch, theta, reset_every=None):
    """`simulate.simulate_pop`, except that every run starts from its measured state (position,
    heading, surge and yaw rate at t = 0; sway zero). The July runs start from rest at the origin,
    which `simulate_pop` assumes; the February segments start under way."""
    from simulate import _tile_params
    theta = np.atleast_2d(theta)
    Pn, R, F, dt = theta.shape[0], batch.R, batch.F, batch.dt
    p = _tile_params(theta, R)
    shift = np.clip(np.round(np.maximum(theta[:, dyn.PARAM_NAMES.index("rud_delay")], 0.0) / dt),
                    0, F - 1).astype(int)
    cols = np.clip(np.arange(F)[None, :] - shift[:, None], 0, F - 1)
    cmd = np.take_along_axis(np.repeat(batch.cmd[None], Pn, axis=0),
                             np.repeat(cols[:, None, :], R, axis=1), axis=2)
    delta_cmd = dyn.cmd_percent_to_angle(cmd).reshape(Pn * R, F)
    rpm = np.tile(batch.rpm, (Pn, 1))
    mpsi, mx, my = np.tile(batch.mpsi_f, (Pn, 1)), np.tile(batch.mx_f, (Pn, 1)), np.tile(batch.my_f, (Pn, 1))
    mr, mu = np.tile(batch.mr_f, (Pn, 1)), np.tile(batch.mu_f, (Pn, 1))
    s = np.zeros((7, Pn * R))
    s[0], s[2], s[3], s[5], s[6] = mu[:, 0], mr[:, 0], mpsi[:, 0], mx[:, 0], my[:, 0]
    s[4] = delta_cmd[:, 0]
    stride = int(round(reset_every / dt)) if reset_every else 0
    psi_h, x_h, y_h = np.empty((Pn * R, F)), np.empty((Pn * R, F)), np.empty((Pn * R, F))
    psi_h[:, 0], x_h[:, 0], y_h[:, 0] = s[3], s[5], s[6]
    for k in range(1, F):
        s = dyn.rk4_step(s, rpm[:, k - 1], delta_cmd[:, k - 1], p, dt)
        if stride and k % stride == 0:
            s[0], s[2], s[3], s[5], s[6] = mu[:, k], mr[:, k], mpsi[:, k], mx[:, k], my[:, k]
        psi_h[:, k], x_h[:, k], y_h[:, k] = s[3], s[5], s[6]
    return psi_h.reshape(Pn, R, F), x_h.reshape(Pn, R, F), y_h.reshape(Pn, R, F)


def replay_errors(batch, params, H):
    """Heading (deg) and position (m) rms over the pose samples, the one at each reset dropped."""
    psi, x, y = replay_pop(batch, dyn.as_vector(params)[None, :], H)
    ep, ex = [], []
    stride = int(round(H / batch.dt)) if H else 0
    for i in range(batch.R):
        ii = batch.idx[i]
        keep = (ii % stride) != 0 if stride else ii > 0
        ep.append(np.rad2deg(psi[0, i, ii] - batch.meas_psi[i])[keep])
        ex.append(np.hypot(x[0, i, ii] - batch.meas_x[i], y[0, i, ii] - batch.meas_y[i])[keep])
    ep, ex = np.concatenate(ep), np.concatenate(ex)
    return float(np.sqrt(np.mean(ep ** 2))), float(np.sqrt(np.mean(ex ** 2)))


def replay_table(runs, params, horizons=(1.0, 2.0, 3.0, 5.0, 10.0, None)):
    from simulate import make_batch
    batch = make_batch(runs)
    return [(H,) + replay_errors(batch, params, H) for H in horizons]


# ---------------------------------------------------------------------------
def main():
    logs, kin = load_all()

    print("1. INVENTORY")
    print("log     duration  pose Hz  RC Hz  HDG lines  LiDAR lines  WD lines  x range (m)      y range (m)")
    for k in LOGS:
        d = logs[k]
        print(f"test_{k}  {d['t'][-1]:7.1f} s  {1 / np.median(np.diff(d['t'])):6.1f}  {1 / np.median(np.diff(d['trc'])):5.2f}  "
              f"{len(d['hdg']):9d}  {d['n_lidar']:11d}  {d['n_wd']:8d}  {d['x'].min():6.1f} .. {d['x'].max():5.1f}   "
              f"{d['y'].min():6.1f} .. {d['y'].max():5.1f}")
    print(f"widest lateral excursion in one log: test_1, {np.ptp(logs[1]['x']):.1f} m of pose (the simulator's map is 10 m wide); "
          f"longest: test_4, {np.ptp(logs[4]['y']):.1f} m")

    print("\n2. RADIO-CONTROL CHANNELS (samples held at 10 Hz between receiver lines; moving = U > 0.4 m/s)")
    print("log    S2 sample      moving s   mean r (deg/s)  mean |r|")
    for k in LOGS:
        g = kin[k]
        mv = g["U"] > 0.4
        for lab, sel in (("S2 <= 1010", g["s2"] <= 1010), ("S2 >= 1950", g["s2"] >= 1950),
                         ("S2 1455-1480", (g["s2"] >= 1455) & (g["s2"] <= 1480))):
            m = sel & mv
            if m.sum():
                print(f"test_{k}  {lab:13s} {m.sum() / 10.0:8.1f}  {np.mean(g['r'][m]):+12.1f}  {np.mean(np.abs(g['r'][m])):8.1f}")
    for k in LOGS:
        g = kin[k]
        acc = np.gradient(g["u"], 0.1)
        rev = g["s1"] < 1300
        if rev.any():
            ahead = rev & (g["u"] > 0.2)
            print(f"test_{k}: S1 < 1300 for {rev.sum() / 10:.0f} s; mean du/dt {np.mean(acc[ahead]):+.2f} m/s^2 while moving ahead; "
                  f"lowest surge {g['u'][rev].min():+.2f} m/s (astern)")
    print("reading: S2 is the rudder (1000 -> r < 0, 1967 -> r > 0, 1467 -> straight); S1 is the throttle "
          "(neutral 1511, forward above, reverse below, with astern motion).")

    pts, top, bias, throttle, tm = throttle_points(logs, kin)
    print("\n3. THROTTLE TO STRAIGHT-LINE SPEED (S1 steady over 4 samples, |r| < 5 deg/s, |dU/dt| < 0.03 m/s^2, U > 0.3 m/s)")
    for s1, U, k, t, dr in pts:
        print(f"  S1 {s1:6.0f}: {U:.2f} m/s  (test_{k} t = {t:5.1f} s)  -> {U / U_PER_RPM:4.1f} rpm-units on the v3 surge balance")
    print(f"  S1 1926-1942 (test_3 t 49-55 s): {top[0]:.2f} m/s, still accelerating ({top[1]:+.3f} m/s^2) when the basin "
          f"ran out: a lower bound on the full-throttle speed")
    print(f"drift angle (course - heading) on the straight samples above 0.5 m/s: mean {bias:+.1f} deg "
          f"(a fixed offset between the pose heading and the direction of travel)")
    print("throttle map used below (S1, m/s): " + ", ".join(f"({s:.0f}, {u:.2f})" for s, u in tm))

    print("\n4. SETTLED FULL-RUDDER TURNS (S2 held at an end point; the first 2 s after the first end-point sample dropped)")
    print("   R_kin = U/|r| at the pose point; R_fit = circle fit where the segment covers >= 180 deg; drift relative to the straight-line offset")
    print("log    side  t (s)        S1 mean (sd)  U     r(deg/s)  R_kin  R_fit  dpsi(deg) drift | v3, same U: rpm  r(deg/s)  R     ratio  drift mid/bow")
    turns = measure_turns(logs, kin, bias)
    for t_ in turns:
        rpm_eq, m, u0 = matched_turn(t_["U"], 100.0 if t_["side"] > 0 else -100.0)
        st = turn_summary(m, u0)
        t_["v3"], t_["rpm"] = st, rpm_eq
        print(f"test_{t_['k']}  {side_name(t_['side'])}  {t_['a']:5.1f}-{t_['b']:5.1f}  {t_['s1']:5.0f} ({t_['s1_sd']:3.0f})  {t_['U']:5.2f}  "
              f"{t_['r']:+7.1f}  {t_['R']:5.2f}  {t_['Rf']:5.2f}  {t_['dpsi']:+7.0f}  {-np.sign(t_['r']) * t_['drift']:+5.1f} | {rpm_eq:5.1f}  "
              f"{st['r']:+7.1f}  {st['R']:5.2f}  {st['U'] / st['u0']:.2f}  {np.sign(st['r']) * st['beta_mid']:+.1f}/{np.sign(st['r']) * st['beta_bow']:+.1f}")
    print("   drift sign: positive = velocity outward of the heading (bow inside the turn)")
    for t_ in turns:
        print(f"  test_{t_['k']} {side_name(t_['side'])} from {t_['a'] - 2.0:.1f} s: speed {t_['U_step']:.2f} m/s at the first "
              f"end-point sample, largest yaw rate in the next 3 s {t_['r_peak']:+.1f} deg/s")
    for t_ in turns:
        if abs(t_["dpsi"]) >= 360:
            print(f"  settled turn test_{t_['k']} {side_name(t_['side'])}: measured R {t_['R']:.2f} m (circle fit {t_['Rf']:.2f}), "
                  f"r {abs(t_['r']):.1f} deg/s at {t_['U']:.2f} m/s; v3 at the same speed R {t_['v3']['R']:.2f} m, "
                  f"r {abs(t_['v3']['r']):.1f} deg/s, a factor {abs(t_['r']) / abs(t_['v3']['r']):.2f} in yaw rate")
    for t_ in turns:
        if t_["s1"] <= 1660 and abs(t_["dpsi"]) >= 90:
            Us = float(throttle.speed(t_["s1"]))
            print(f"  speed in turn / straight speed at the same throttle, test_{t_['k']} {side_name(t_['side'])} "
                  f"(S1 {t_['s1']:.0f}): {t_['U']:.2f} / {Us:.2f} = {t_['U'] / Us:.2f}  (v3: {t_['v3']['U'] / t_['v3']['u0']:.2f})")

    print("\n5. CRASH STOPS (receiver sample S1 < 1300 after one >= 1490, at U > 0.6 m/s 1 s before)")
    print(f"   F = M11 du/dt - mean(X_uu u^2) where reverse is certain (M11 = {dyn.M11:.1f} kg, X_uu = {P['X_uu']:.2f}); "
          f"stop at u = 0.15 m/s; times from half a second before the first reverse sample")
    print("   v3: the same manoeuvre (settled at the measured speed, in a full-helm turn where the stop began in one), "
          "rudder centred, reverse force F;")
    print("   strong and weak are the safety layer's bracket at full astern (S2 = -100, -24 rpm-units), as "
          "tools/diagnostics/manoeuvre_predictions.py")
    print("   heading change dpsi5 over the first 5 s; for v3 also the yaw rate left when surge reaches zero and the heading change in 60 s")
    print("log    sample  S1 (min)    from      u0    r0      t_stop  reach   dpsi5  F (N) | v3 time to zero surge, reach, dpsi5, r at zero surge (deg/s), dpsi60:")
    print("                                                                                F = 17 N (measured)          strong: 32.1 N (0.5 T(24))   weak: 16.1 N (0.25 T(24)) after 0.75 s")
    stops = measure_stops(logs, kin)
    for s in stops:
        frm = "turn" if abs(s["r0"]) > 8 else "straight"
        sims = [model_stop(s, F, on) for F, on in ((17.0, 0.0), (0.5 * T24, 0.0), (0.25 * T24, 0.75))]
        s["sims"] = sims
        print(f"test_{s['k']}  {s['t_sample']:6.1f}  {s['s1']:4.0f} ({s['s1_min']:4.0f})  {frm:8s}  {s['u0']:4.2f}  {s['r0']:+5.1f}  "
              f"{s['t_stop']:5.1f} s  {s['reach']:4.2f} m  {s['dpsi']:+5.0f}  {s['F']:5.1f} r_stop {s['r_stop']:+4.1f} | "
              + "  ".join(f"{q['t_stop']:4.1f} s {q['reach']:4.2f} m {q['dpsi5']:3.0f} {q['r_zero']:4.1f} {q['dpsi60']:3.0f}" for q in sims))
    full = [s for s in stops if s["s1_min"] < 1100 and s["u0"] > 1.0]
    print(f"  full reverse (S1 about 1011) from above 1.0 m/s, {len(full)} stops: F = {np.mean([s['F'] for s in full]):.1f} N "
          f"(range {min(s['F'] for s in full):.1f} .. {max(s['F'] for s in full):.1f}); to 0.15 m/s in "
          f"{np.mean([s['t_stop'] for s in full]):.1f} s and {np.mean([s['reach'] for s in full]):.2f} m; "
          f"heading change {np.mean([abs(s['dpsi']) for s in full if abs(s['r0']) > 8]):.0f} deg from the turns")
    print(f"  as a fraction of v3's T(24) = {T24:.1f} N: {np.mean([s['F'] for s in full]) / T24:.2f} "
          f"(range {min(s['F'] for s in full) / T24:.2f} .. {max(s['F'] for s in full) / T24:.2f}); bracket 0.25 .. 0.5")
    print(f"  forward thrust for comparison: v3 at RPM 12 {ship.forward_thrust(12.0):.1f} N; full throttle, from the straight "
          f"speed >= {top[0]:.2f} m/s: >= {P['X_uu'] * top[0] ** 2:.1f} N")

    print("\n6. YAW DECAY AFTER THE RUDDER IS CENTRED AT SPEED (S2 end point -> neutral, S1 unchanged within 30, U > 0.6 m/s)")
    print("   measured from the last end-point sample + 0.5 s (+-0.5 s unknown, i.e. +-|r|/2 deg) to |r| < 1 deg/s "
          "('cut' = the next helm sample came first)")
    print("   v3: settled in a full-helm turn at the same speed, rudder centred, thrust held")
    print("log    t      u     r0      measured dpsi  t       | v3: r in turn  dpsi    t")
    for ev in measure_centring(logs, kin, throttle):
        mc = model_centring(ev)
        print(f"test_{ev['k']}  {ev['t0']:5.1f}  {ev['u0']:4.2f}  {ev['r0']:+6.1f}  {ev['dpsi']:+11.1f}  {ev['t_settle']:4.1f} s"
              f"{' cut' if ev['cut'] else '    '} | {mc['r_turn']:+12.1f}  {mc['dpsi']:+6.1f}  {mc['t_settle']:4.1f} s")

    print("\n7. COAST-DOWNS AT NEUTRAL THROTTLE (S1 1505-1515, S2 neutral)")
    print("   measured surge 5 and 10 s after the start against v3 with zero thrust from the same speed")
    print("   (v3 has no idle thrust; S1 = 1511 holds about 0.1 m/s ahead in test_1 t 157-198 s)")
    for k, a, b in ((1, 80.0, 120.0), (1, 23.5, 32.5), (2, 50.5, 61.5), (4, 194.8, 206.0)):
        g = kin[k]
        m = (g["t"] >= a) & (g["t"] <= b)
        tt, uu = g["t"][m] - a, g["u"][m]
        u0 = uu[:5].mean()
        out = []
        for T in (5.0, 10.0):
            if tt[-1] >= T:
                um = uu[np.abs(tt - T) <= 0.5].mean()
                out.append(f"{T:.0f} s: measured {um:.2f}, v3 {u0 / (1.0 + u0 * P['X_uu'] / dyn.M11 * T):.2f} m/s")
        print(f"  test_{k} t {a:5.1f}-{b:5.1f} s from {u0:.2f} m/s: " + "; ".join(out))

    print("\n8. WINDOWED REPLAY OF THE FORWARD-MOTION SEGMENTS THROUGH v3 (pose at 2 Hz as in July; S2 -> +-100 % $CMD; "
          "S1 -> rpm through the throttle map)")
    print("   segments: " + ", ".join(f"test_{k} {a:.0f}-{b:.0f} s" for k, a, b in REPLAY_SEGMENTS))
    runs = [fl.as_run(logs[k], a, b, throttle) for k, a, b in REPLAY_SEGMENTS]
    print("horizon   freeze   const.rate   v3 heading   v3 position")
    for H, ep, ex in replay_table(runs, P):
        Hn = 1e6 if H is None else H
        print(f"  {'free' if H is None else f'{H:.0f} s':6s}  {naive_rms(runs, Hn, 'freeze'):6.2f}   "
              f"{naive_rms(runs, Hn, 'constr'):8.2f}   {ep:10.2f}   {ex:9.3f} m")
    print("  (July holdout, REPORT.md: v3 1.31 / 5.19 / 7.90 / 10.17 / 13.82 / 23.53 deg at 1/2/3/5/10 s/free)")


if __name__ == "__main__":
    main()
