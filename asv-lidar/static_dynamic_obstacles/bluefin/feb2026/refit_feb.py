"""Candidate analysis model "v3f": the v3 structure with its turn-regime parameters refitted on the
July training session and the February radio-control turns together.

    python bluefin/feb2026/refit_feb.py fit      # differential evolution + polish, writes out/candidate.json
    python bluefin/feb2026/refit_feb.py report   # v3 against the candidate, writes nothing

An ANALYSIS model for the sim-to-field comparison, not a training plant: `src/` and the baseline-v3
configuration are untouched, and the candidate is loaded only by passing its parameters to
`ship.ShipModel(params=...)`. It carries no bootstrap and is not meant for `sample_params`.

Set-up, fixed before the fit ran:
* Structure: `dynamics.py` unchanged.
* Free: X_vv, X_rr, X_delta, Y_v, Y_vv, k_R, k_race, N_r, N_uv (bounds as `fit_final.py`, N_uv
  widened to [-250, 250] as in refit v4).
* Fixed at v3: T12, X_uu (the July surge balance, which the February throttle cannot recalibrate
  in rpm-units), and the actuator rud_rate, rud_tau, rud_delay (the 1 Hz receiver samples cannot
  time a rudder movement to better than 1 s).
* Tied: N_rr = Y_vv L^3 / 32, the yaw moment of the same cross-flow drag that Y_vv represents in
  sway (strip theory over a hull of constant draft). It gives the hull a yaw damping that does not
  vanish with surge, the term whose absence leaves v3 spinning at zero speed.
* Data: the July training session (2026-07-02) with the three hauled retrievals trimmed by the
  pre-registered rule of refit v4; the February forward-motion segments of `feb_analysis.py`
  (S2 -> +-100 % `$CMD`, S1 -> rpm through the measured throttle map).
* Objective: v3's pooled July objective (3 s and 10 s windows, free run, turn-rate p90, drift
  IQR) without its steady-turn bands, which the February turns replace; plus the February 3 s and
  10 s windowed heading and position errors, weighted so that the two data sets count equally at
  the v3 solution; plus v3's two structural priors (every helm turns the same way; turn rate grows
  with helm) as a constraint.
* Optimiser: `fit_model.differential_evolution` (population 32, 80 generations) from v3 and from a
  random population, each followed by `fit_model.polish`; the better of the two is kept.
* Check: the July holdout session (2026-07-03, untrimmed, as in `REPORT.md`), never used in the fit.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUEFIN = os.path.dirname(HERE)
sys.path[:0] = [HERE, BLUEFIN, os.path.join(BLUEFIN, "refit_v4"), os.path.join(BLUEFIN, "..", "src")]

import dynamics as dyn  # noqa: E402
import feb_analysis as fa  # noqa: E402
import feb_logs as fl  # noqa: E402
from fit_final import BOUNDS as V3_BOUNDS  # noqa: E402
from fit_model import differential_evolution, polish  # noqa: E402
from simulate import make_batch, objective_multi, simulate, residuals, rms, steady_turn  # noqa: E402

OUT = os.path.join(HERE, "out")
FREE = ("X_vv", "X_rr", "X_delta", "Y_v", "Y_vv", "k_R", "k_race", "N_r", "N_uv")
V3 = dict(fa.P)
NRR_TIE = dyn.L ** 3 / 32.0
# The two data sets weigh the same at the v3 solution: July objective 1.09, February mean of the
# 3 s and 10 s terms 2.85 (computed before the fit), hence 1.09 / 2.85.
W_FEB = 0.38
W_STRUCT = 10.0

BOUNDS = {}
for k in dyn.PARAM_NAMES:
    if k in FREE:
        BOUNDS[k] = (-250.0, 250.0) if k == "N_uv" else V3_BOUNDS[k]
    else:
        BOUNDS[k] = (V3[k], V3[k])
LO = np.array([BOUNDS[k][0] for k in dyn.PARAM_NAMES])
HI = np.array([BOUNDS[k][1] for k in dyn.PARAM_NAMES])
I_NRR, I_YVV = dyn.PARAM_NAMES.index("N_rr"), dyn.PARAM_NAMES.index("Y_vv")


def tie(theta):
    theta = np.array(theta, dtype=float, ndmin=2)
    theta[:, I_NRR] = theta[:, I_YVV] * NRR_TIE
    return theta


def datasets():
    from refit_v4 import datasets as july
    train_raw, train, test_raw, _, trims = july()
    logs, kin = fa.load_all()
    _, _, _, throttle, _ = fa.throttle_points(logs, kin)
    feb = [fl.as_run(logs[k], a, b, throttle) for k, a, b in fa.REPLAY_SEGMENTS]
    return train, test_raw, feb, trims


def feb_objective(batch, theta, H):
    psi, x, y = fa.replay_pop(batch, theta, H)
    stride = int(round(H / batch.dt))
    j = np.zeros(theta.shape[0])
    for i in range(batch.R):
        ii = batch.idx[i]
        ii = ii[(ii % stride) != 0]
        k = np.searchsorted(batch.idx[i], ii)
        dp = np.rad2deg(psi[:, i, ii] - batch.meas_psi[i][k][None, :])
        dx = x[:, i, ii] - batch.meas_x[i][k][None, :]
        dy = y[:, i, ii] - batch.meas_y[i][k][None, :]
        j += np.sqrt(np.mean(dp ** 2, axis=1)) / 10.0 + np.sqrt(np.mean(dx ** 2 + dy ** 2, axis=1))
    j[~np.isfinite(j)] = 1e6
    return j / batch.R


def structural(theta):
    u, r = steady_turn(theta)
    ar = np.abs(r)
    pen = np.sum(np.maximum(0.0, -np.sign(r[:, 1:]) * np.sign(r[:, -1:])) * ar[:, 1:], axis=1) / 5.0
    pen += np.sum(np.maximum(0.0, ar[:, :-1] - ar[:, 1:]), axis=1) / 10.0
    pen[~np.isfinite(pen)] = 1e3
    return pen


def make_objective(b_tr, b_feb):
    def fun(theta):
        theta = tie(theta)
        j = objective_multi(b_tr, theta, w_pos=1.0, w_prior=0.0)
        j = j + W_FEB * 0.5 * (feb_objective(b_feb, theta, 3.0) + feb_objective(b_feb, theta, 10.0))
        j = j + W_STRUCT * structural(theta)
        j[~np.isfinite(j)] = 1e6
        return j
    return fun


def cmd_fit(iters=60, popsize=32):
    os.makedirs(OUT, exist_ok=True)
    train, test, feb, trims = datasets()
    b_tr, b_feb = make_batch(train), make_batch(feb)
    fun = make_objective(b_tr, b_feb)
    x_v3 = dyn.as_vector(V3)
    t0 = time.time()
    j_v3 = float(fun(x_v3[None, :])[0])
    print(f"J(v3, N_rr tied) = {j_v3:.5f}   one evaluation {time.time() - t0:.1f} s")
    best_x, best_f = None, np.inf
    for k, x0 in enumerate((x_v3, None)):
        x, f = differential_evolution(fun, LO, HI, popsize=popsize, iters=iters, seed=100 + k, x0=x0, verbose=True)
        x, f = polish(fun, x, LO, HI, rounds=5, popsize=popsize, seed=200 + k, verbose=True)
        print(f"start {k} ({'v3' if x0 is not None else 'random'}): J = {f:.5f}")
        if f < best_f:
            best_x, best_f = x, f
    best_x = tie(best_x)[0]
    json.dump({"params": dyn.as_dict(best_x), "J": best_f, "J_v3": j_v3,
               "free": list(FREE), "N_rr_tie": f"Y_vv * L^3/32 = Y_vv * {NRR_TIE:.4f}",
               "bounds": {k: list(v) for k, v in BOUNDS.items()}, "trims": trims,
               "feb_segments": [list(s) for s in fa.REPLAY_SEGMENTS],
               "elapsed_s": time.time() - t0},
              open(os.path.join(OUT, "candidate.json"), "w"), indent=2)
    print(f"wrote {os.path.join(OUT, 'candidate.json')}  ({time.time() - t0:.0f} s)")


def holdout_table(params_list, names):
    """July holdout (2026-07-03, untrimmed): heading and position rms against horizon, as validate.py."""
    _, test, _, _ = datasets()
    b = make_batch(test)
    print("July holdout (2026-07-03, 6 runs), rms over pose samples, the one at each reset dropped")
    print("horizon  " + "  ".join(f"{n:>18s}" for n in names))
    for H in (1.0, 2.0, 3.0, 5.0, 10.0, None):
        cells = []
        for p in params_list:
            pr = simulate(b, p, reset_every=H)
            ep, ex = residuals(b, pr, skip_after_reset=1)
            cells.append(f"{rms(ep):7.2f} deg {rms(ex):5.2f} m")
        print(f"  {'free' if H is None else f'{H:.0f} s':6s} " + "  ".join(cells))
    for p, n in zip(params_list, names):
        pf = simulate(b, p, reset_every=None)
        path = [abs(np.sum(np.hypot(np.diff(pf['x'][i]), np.diff(pf['y'][i]))) - r.path_length()) for i, r in enumerate(test)]
        print(f"  {n}: free-run path length, mean absolute error {np.mean(path):.2f} m")


def cmd_report():
    cand = json.load(open(os.path.join(OUT, "candidate.json")))["params"]
    v4 = json.load(open(os.path.join(BLUEFIN, "refit_v4", "out", "params_v4.json")))["params"]
    v4b = json.load(open(os.path.join(BLUEFIN, "refit_v4", "out_b", "params_v4.json")))["params"]
    names = ("v3", "v4 (July only)", "v4b (July only)", "v3f candidate")
    plist = (V3, v4, v4b, cand)
    print("Models: v3 (training plant); refit v4 and v4b (July training session trimmed, not adopted, REVIEW.md 8.2-8.3);")
    print("v3f, this refit (July trimmed + February turns)\n")
    print("PARAMETERS (v3 -> candidate; * = free in the refit, = tied, others fixed at v3)")
    for k in dyn.PARAM_NAMES:
        flag = "*" if k in FREE else ("=" if k == "N_rr" else " ")
        lo, hi = BOUNDS[k]
        at = "  at bound" if k in FREE and (abs(cand[k] - lo) < 0.01 * (hi - lo) or abs(cand[k] - hi) < 0.01 * (hi - lo)) else ""
        print(f"  {flag} {k:10s} {V3[k]:10.4f} -> {cand[k]:10.4f}{at}")
    print()
    holdout_table(plist, names)

    logs, kin = fa.load_all()
    _, _, bias, throttle, _ = fa.throttle_points(logs, kin)
    runs = [fl.as_run(logs[k], a, b, throttle) for k, a, b in fa.REPLAY_SEGMENTS]
    print("\nFebruary forward-motion segments, windowed replay (fitted on these; not a holdout)")
    rows = [fa.replay_table(runs, p) for p in plist]
    for i, H in enumerate((1.0, 2.0, 3.0, 5.0, 10.0, None)):
        print(f"  {'free' if H is None else f'{H:.0f} s':6s} " + "  ".join(f"{r[i][1]:7.2f} deg {r[i][2]:5.2f} m" for r in rows))

    print("\nFebruary settled full-rudder turns at the measured speed (measured | v3 | candidate): r deg/s, R m")
    for t in fa.measure_turns(logs, kin, bias):
        if abs(t["dpsi"]) < 360:
            continue
        helm = 100.0 if t["side"] > 0 else -100.0
        out = []
        for p in plist:
            _, m, u0 = fa.matched_turn(t["U"], helm, p)
            st = fa.turn_summary(m, u0)
            out.append(f"{abs(st['r']):5.1f} {st['R']:5.2f}  (ratio {st['U'] / st['u0']:.2f})")
        print(f"  test_{t['k']} {fa.side_name(t['side'])} at {t['U']:.2f} m/s: {abs(t['r']):5.1f} {t['R']:5.2f} | " + " | ".join(out))

    print("\nCrash stops from the February turns, reverse force 17 N, rudder centred:"
          " heading change in 5 s and yaw rate left at zero surge")
    for s in fa.measure_stops(logs, kin):
        if abs(s["r0"]) <= 8 or s["s1_min"] > 1100:
            continue
        out = [fa.model_stop(s, 17.0, params=p) for p in plist]
        print(f"  test_{s['k']} t {s['t_sample']:.1f} from {s['u0']:.2f} m/s, r {s['r0']:+.1f}: measured dpsi5 {abs(s['dpsi']):.0f} deg | "
              + " | ".join(f"{n}: dpsi5 {q['dpsi5']:.0f} deg, r at zero surge {q['r_zero']:.1f} deg/s, dpsi60 {q['dpsi60']:.0f} deg"
                           for n, q in zip(names, out)))

    print("\nSteady turns of the candidate (A2 form): yaw rate deg/s, radius m and speed ratio at full helm")
    for rpm in (6.0, 12.0):
        rr = []
        for helm in (25.0, 50.0, 75.0, 100.0):
            m, u0 = fa.settled_turn(rpm, helm, cand)
            st = fa.turn_summary(m, u0)
            rr.append(f"{abs(st['r']):5.1f}")
        print(f"  RPM {rpm:4.1f}: helm 25/50/75/100 %: " + " / ".join(rr) + f"; R {st['R']:.2f} m, ratio {st['U'] / st['u0']:.2f}")
    m, u0 = fa.settled_turn(24.0, 100.0, cand)
    print(f"  bounded at RPM 24 full helm over 60 s: u {m.u:.2f} m/s, r {math.degrees(m.yaw_rate):.1f} deg/s")
    print("\nYaw decay at zero surge: from r = 10 deg/s with u = v = 0, rudder centred, no thrust")
    for p, n in zip(plist, names):
        m = fa.ship.ShipModel(params=p)
        m._s[2, 0] = math.radians(10.0)
        t, psi, _, _ = fa.run_until(m, 0.0, 0.0, 60.0, lambda mm, t, tz: abs(mm.yaw_rate) < math.radians(0.5))
        print(f"  {n}: N_rr = {p['N_rr']:.3f}; after {t:.1f} s r = {math.degrees(m.yaw_rate):.2f} deg/s, "
              f"heading change {abs(psi):.0f} deg")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        cmd_fit(*(int(a) for a in sys.argv[2:4]))
    else:
        cmd_report()
