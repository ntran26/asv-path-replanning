"""The field test suite with both ships starting from rest, and the release offset that keeps each
case's designed encounter.

On the water the own ship and the target both start from rest at their marks. In simulation the
suite was verified with both already moving at t = 0 (the scenario generator's backward solve).
This runs each case with:
  - the own ship from rest at its t = 0 point (the environment's `initial_speed` option), the
    deployed policy acting from its release;
  - the target from rest at its start mark, accelerating as u(t) = v tanh(t / tau), tau = K / v,
    K = 5.29 s m/s: the Bluefin plant's own acceleration (90 % of speed after 14 s at cruise),
    standing in until the target platform is measured (FIELD_SESSIONS_PLAN.md, S2-D);
  - a release offset D: the target is released D s after the own ship (D < 0: the own ship is held
    at rest, propulsion off, for -D s after the target's release).

For each case, D is chosen on 3 seeds by how well it reproduces the designed encounter of the
verified run at cruise: |closest range - designed| + |own along-track position at the closest
approach - designed| (metres). Not by success. The chosen D is then run on 10 seeds. The offset is
also compared with a formula, D_f = L_own - L_target, where L_target = K ln 2 / v is the target's
lag behind a ship already at speed and L_own is the own ship's extra lag from rest under the
policy, measured against its own run at cruise over the first 12 s.

Usage (from static_dynamic_obstacles/):
    python tools/diagnostics/field_suite_both_from_rest.py --processes 4
Output: results/field_suite/both_from_rest/ (sweep.csv, final.csv, summary.json, trajectories.json).
No source file changes; the environment is driven through its reset options and the target's
`speed` attribute (and its wall stop, held off for the first metre from rest).
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "results" / "field_suite" / "both_from_rest"
K_TAU = 5.29                    # s m/s: tau = K / v for the Bluefin plant (m_eff / X_uu)
OFFSETS = [float(d) for d in range(-20, 11)]


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(SRC))
    import torch
    torch.set_num_threads(1)


_MODEL = None


def episode(job):
    """(case, seed, mode, offset) -> metrics, and the trajectory if asked.

    mode "cruise": as verified (both moving at t = 0). mode "rest": both from rest, offset D."""
    global _MODEL
    name, seed, mode, offset, keep_traj = job
    import numpy as np
    import constants as cfg
    import curriculum
    import field_test_cases as ftc
    import train_formulation as tf
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    if _MODEL is None:
        algo = json.loads((ftc.DEFAULT_MODEL.parent / "config.json").read_text()).get("algo", "sac")
        _MODEL = tf.ALGORITHMS[algo].load(str(ftc.DEFAULT_MODEL), device="cpu")
    case = ftc.TestCase()
    built, _ = case.scenario(name)
    sheet = case.field_sheet(name)
    (lx0, ly0), (lx1, ly1) = sheet["leg_start"], sheet["leg_goal"]
    ll = math.hypot(lx1 - lx0, ly1 - ly0)
    ux, uy = (lx1 - lx0) / ll, (ly1 - ly0) / ll
    env = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0, vessel_randomisation=None)
    actor = tf.EpisodeActor(_MODEL) if hasattr(tf, "EpisodeActor") else None
    options = {"generated": copy.deepcopy(built)}
    if mode == "rest":
        options["initial_speed"] = 0.0
    obs, _ = env.reset(seed=int(seed), options=options)
    tgt = env.targets[0] if env.targets else None
    v_t = float(tgt.speed) if tgt is not None else 0.0
    own_hold = max(0.0, -offset) if mode == "rest" else 0.0
    t_release_target = max(0.0, offset) if mode == "rest" else 0.0
    box, start_xy = None, None
    if mode == "rest" and tgt is not None:
        tgt.speed = 0.0
        # The wall stop (`stop_box`) halts a target whose hull is outside it after a step. A start
        # mark near the wall passes it at speed but not when creeping from rest, so the check is
        # held off until the target has left its mark by 1 m.
        box, start_xy = tgt.stop_box, (tgt.x, tgt.y)
        tgt.stop_box = None
    dt = cfg.UPDATE_RATE
    steps, outcome = 0, "timeout"
    best = (math.inf, None, None)            # (range, own along-track, time)
    traj = []
    speeds = []
    while True:
        t = steps * dt
        if box is not None and math.hypot(tgt.x - start_xy[0], tgt.y - start_xy[1]) >= 1.0:
            tgt.stop_box, box = box, None
        if mode == "rest" and tgt is not None and t >= t_release_target - 1e-9:
            tau = K_TAU / max(v_t, 1e-6)
            tgt.speed = v_t * math.tanh((t - t_release_target + 0.5 * dt) / tau)
        if t < own_hold - 1e-9:
            action = np.array([0.0, -1.0], dtype=np.float32)          # held: rudder centred, propulsion off
        else:
            action = actor(obs) if actor is not None else _MODEL.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(np.asarray(action, dtype=np.float32).reshape(2))
        steps += 1
        speeds.append(float(env.u_body))
        s_own = (env.asv_x - lx0) * ux + (env.asv_y - ly0) * uy
        if tgt is not None:
            r = math.hypot(tgt.x - env.asv_x, tgt.y - env.asv_y)
            if r < best[0]:
                best = (r, s_own, steps * dt)
        if keep_traj:
            traj.append([round(steps * dt, 2), round(env.asv_x, 3), round(env.asv_y, 3), round(env.asv_h, 1),
                         None if tgt is None else round(tgt.x, 3), None if tgt is None else round(tgt.y, 3)])
        if term or trunc:
            if info.get("collided"):
                outcome = f"collision:{info.get('collision_kind')}"
            elif info.get("reached_goal"):
                outcome = "goal"
            break
    env.close()
    # own-ship lag over the first 12 s: time lost against a ship holding U_NOM
    n = min(len(speeds), int(12 / dt))
    lag = sum(1.0 - u / cfg.U_NOM for u in speeds[:n]) * dt
    row = {"case": name, "seed": int(seed), "mode": mode, "offset_s": offset, "outcome": outcome,
           "min_range_m": round(best[0], 3) if math.isfinite(best[0]) else None,
           "s_at_cpa_m": None if best[1] is None else round(best[1], 3), "t_cpa_s": best[2],
           "time_s": round(steps * dt, 1), "own_lag_12s_s": round(lag, 2), "target_speed": v_t}
    if keep_traj:
        row["trajectory"] = traj
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--sweep-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=10)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--merge", action="store_true",
                    help="replace only these cases' rows in the existing outputs (after a case is re-picked)")
    args = ap.parse_args()
    sys.path.insert(0, str(SRC))
    import field_test_cases as ftc
    case = ftc.TestCase()
    names = case.names() if args.cases == "all" else args.cases.split(",")
    allnames = case.names()

    def merged_csv(fname, new_rows, key):
        old = []
        if args.merge and (OUT / fname).exists():
            with open(OUT / fname, encoding="utf-8") as fh:
                old = [r for r in csv.DictReader(fh) if r["case"] not in names]
        return sorted(old + [dict(r) for r in new_rows], key=key)
    seed0 = {n: case.scenario(n)[1] for n in names}
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool = get_context("spawn").Pool(args.processes, initializer=_init)

    # 1. Designed encounters (cruise) and the sweep (rest).
    jobs = []
    for n in names:
        for j in range(args.sweep_seeds):
            s = seed0[n] + 10_000 * j
            jobs.append((n, s, "cruise", 0.0, False))
            for d in OFFSETS:
                jobs.append((n, s, "rest", d, False))
    rows = list(pool.imap_unordered(episode, jobs))
    print(f"sweep: {len(rows)} episodes, {time.time() - t0:.0f} s", flush=True)
    with open(OUT / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(merged_csv("sweep.csv", rows, lambda r: (allnames.index(r["case"]), r["mode"],
                                                             float(r["offset_s"]), int(r["seed"]))))

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    chosen, lags = {}, {}
    for n in names:
        ref = [r for r in rows if r["case"] == n and r["mode"] == "cruise"]
        d_star = median([r["min_range_m"] for r in ref])
        s_star = median([r["s_at_cpa_m"] for r in ref])
        scores = []
        for d in OFFSETS:
            rs = [r for r in rows if r["case"] == n and r["mode"] == "rest" and r["offset_s"] == d]
            sc = median([abs(r["min_range_m"] - d_star) + abs(r["s_at_cpa_m"] - s_star) for r in rs])
            scores.append((sc, abs(d), d))
        scores.sort()
        chosen[n] = {"offset_s": scores[0][2], "match_m": round(scores[0][0], 3), "designed_range_m": d_star,
                     "designed_s_m": s_star,
                     "scores": {str(d): round(sc, 3) for sc, _, d in sorted(scores, key=lambda x: x[2])}}
        # the own ship's extra lag from rest, against its own verified run at cruise (both under the policy)
        lags[n] = round(median([r["own_lag_12s_s"] for r in rows if r["case"] == n and r["mode"] == "rest"
                                and r["offset_s"] == 0.0])
                        - median([r["own_lag_12s_s"] for r in ref]), 2)

    # 2. The chosen offset on the final seeds, plus one kept trajectory per case and mode.
    jobs = []
    for n in names:
        for j in range(args.final_seeds):
            s = seed0[n] + 10_000 * j
            jobs.append((n, s, "rest", chosen[n]["offset_s"], j == 0))
            jobs.append((n, s, "cruise", 0.0, j == 0))
    final = list(pool.imap_unordered(episode, jobs))
    pool.close()
    trajs = {}
    for r in final:
        if "trajectory" in r:
            trajs.setdefault(r["case"], {})[r["mode"]] = {"offset_s": r["offset_s"], "outcome": r["outcome"],
                                                          "trajectory": r.pop("trajectory")}
    with open(OUT / "final.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in final[0] if k != "trajectory"])
        w.writeheader()
        w.writerows(merged_csv("final.csv", final, lambda r: (allnames.index(r["case"]), r["mode"], int(r["seed"]))))
    if args.merge and (OUT / "trajectories.json").exists():
        trajs = {**json.loads((OUT / "trajectories.json").read_text(encoding="utf-8")), **trajs}
    json.dump(trajs, open(OUT / "trajectories.json", "w", encoding="utf-8"))

    summary = []
    for n in names:
        v = case.field_sheet(n)["target_speed_mps"]
        l_t = K_TAU * math.log(2) / v
        rs = [r for r in final if r["case"] == n and r["mode"] == "rest"]
        rc = [r for r in final if r["case"] == n and r["mode"] == "cruise"]
        summary.append({
            "case": n, "target_speed": v, "offset_s": chosen[n]["offset_s"],
            "offset_formula_s": round(lags[n] - l_t, 1), "own_lag_s": lags[n], "target_lag_s": round(l_t, 1),
            "match_m": chosen[n]["match_m"],
            "goal_cruise": f"{sum(r['outcome'] == 'goal' for r in rc)}/{len(rc)}",
            "goal_both_rest": f"{sum(r['outcome'] == 'goal' for r in rs)}/{len(rs)}",
            "range_cruise_median": median([r["min_range_m"] for r in rc]),
            "range_rest_median": median([r["min_range_m"] for r in rs]),
            "failures": ";".join(sorted({r["outcome"] for r in rs if r["outcome"] != "goal"}))})
    if args.merge and (OUT / "summary.json").exists():
        old = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        chosen = {**old["chosen"], **chosen}
        summary = sorted([x for x in old["summary"] if x["case"] not in names] + summary,
                         key=lambda x: allnames.index(x["case"]))
    json.dump({"k_tau": K_TAU, "offsets": OFFSETS, "sweep_seeds": args.sweep_seeds, "final_seeds": args.final_seeds,
               "elapsed_s": round(time.time() - t0), "chosen": chosen, "summary": summary},
              open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s, flush=True)


if __name__ == "__main__":
    main()
