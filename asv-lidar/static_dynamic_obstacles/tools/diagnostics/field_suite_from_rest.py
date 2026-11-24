"""The field test suite with the own ship starting from rest.

`src/field_test_cases.py` runs each case as the suite was verified: the own ship passes its t = 0
point at cruise (U_NOM, the scenario generator's assumption), the target at its start at t = 0.
On the water the own ship starts from rest at its mark. This runs every case both ways on the
identified plant, deterministic actions, the environment's emergency stop off, with the same
sensor-noise seeds, through the environment's own `initial_speed` reset option (no source change):

  cruise  as verified (initial speed U_NOM)
  rest    initial speed 0, target timing unchanged
  delayed initial speed 0, the target held at its start mark (speed 0) and released after
          --launch-delay-s, the time the plant loses accelerating from rest at the cruise RPM
          (6.5 s): the field start procedure of FIELD_SESSIONS_PLAN.md item B-7

Usage (from static_dynamic_obstacles/):
    python tools/diagnostics/field_suite_from_rest.py --seeds 10 --processes 4 --launch-delay-s 6.5
Output: results/field_suite/from_rest/ (episodes.csv, summary.csv, summary.json).
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
OUT = ROOT / "results" / "field_suite" / "from_rest"


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(SRC))
    import torch
    torch.set_num_threads(1)


_MODEL = None


def run_one(job):
    """One episode: (case, seed, condition) -> row."""
    global _MODEL
    name, seed, condition, delay = job
    import numpy as np
    import constants as cfg
    import curriculum
    import field_test_cases as ftc
    import train_formulation as tf
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    if _MODEL is None:
        cfg_json = json.loads((ftc.DEFAULT_MODEL.parent / "config.json").read_text())
        _MODEL = tf.ALGORITHMS[cfg_json.get("algo", "sac")].load(str(ftc.DEFAULT_MODEL), device="cpu")
    built, _ = ftc.TestCase().scenario(name)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0, vessel_randomisation=None)
    actor = tf.EpisodeActor(_MODEL) if hasattr(tf, "EpisodeActor") else None
    options = {"generated": copy.deepcopy(built)}
    if condition in ("rest", "delayed"):
        options["initial_speed"] = 0.0
    obs, _ = env.reset(seed=int(seed), options=options)
    start_speed = float(env.start_speed)
    held = []
    if condition == "delayed":
        held = [(t, t.speed) for t in env.targets]
        for t, _ in held:
            t.speed = 0.0
    outcome, steps, min_range, t_min = "timeout", 0, math.inf, None
    speeds = []
    while True:
        if held and steps * cfg.UPDATE_RATE >= delay - 1e-9:
            for t, v in held:
                t.speed = v
            held = []
        action = actor(obs) if actor is not None else _MODEL.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(np.asarray(action, dtype=np.float32).reshape(2))
        steps += 1
        speeds.append(float(env.u_body))
        if env.targets:
            t = env.targets[0]
            r = math.hypot(t.x - env.asv_x, t.y - env.asv_y)
            if r < min_range:
                min_range, t_min = r, steps * cfg.UPDATE_RATE
        if term or trunc:
            if info.get("collided"):
                outcome = f"collision:{info.get('collision_kind')}"
            elif info.get("reached_goal"):
                outcome = "goal"
            break
    env.close()
    t_cruise = next((i + 1 for i, u in enumerate(speeds) if u >= 0.9 * cfg.U_NOM), None)
    return {"case": name, "seed": int(seed), "condition": condition,
            "launch_delay_s": delay if condition == "delayed" else 0.0, "start_speed": round(start_speed, 3),
            "outcome": outcome, "time_s": round(steps * cfg.UPDATE_RATE, 1),
            "min_target_range_m": round(min_range, 2) if math.isfinite(min_range) else None,
            "t_closest_s": t_min,
            "t_to_90pct_cruise_s": None if t_cruise is None else round(t_cruise * cfg.UPDATE_RATE, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--launch-delay-s", type=float, default=6.5)
    ap.add_argument("--conditions", default="cruise,rest,delayed")
    args = ap.parse_args()
    sys.path.insert(0, str(SRC))
    import field_test_cases as ftc
    case = ftc.TestCase()
    names = case.names() if args.cases == "all" else args.cases.split(",")
    conditions = args.conditions.split(",")
    jobs = []
    for n in names:
        _, seed0 = case.scenario(n)
        for j in range(args.seeds):
            for cond in conditions:
                jobs.append((n, seed0 + 10_000 * j, cond, args.launch_delay_s))   # the seeds `field_test_cases.py run` uses
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.processes, initializer=_init) as pool:
        rows = []
        for k, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(row)
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (names.index(r["case"]), conditions.index(r["condition"]), r["seed"]))
    with open(OUT / "episodes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for n in names:
        for cond in conditions:
            rs = [r for r in rows if r["case"] == n and r["condition"] == cond]
            goals = sum(r["outcome"] == "goal" for r in rs)
            ranges = [r["min_target_range_m"] for r in rs if r["min_target_range_m"] is not None]
            fails = sorted({r["outcome"] for r in rs if r["outcome"] != "goal"})
            summary.append({"case": n, "condition": cond, "goal": f"{goals}/{len(rs)}",
                            "min_target_range_median_m": round(sorted(ranges)[len(ranges) // 2], 2) if ranges else None,
                            "min_target_range_min_m": min(ranges) if ranges else None,
                            "time_median_s": sorted(r["time_s"] for r in rs)[len(rs) // 2],
                            "failures": ";".join(fails)})
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    json.dump({"seeds": args.seeds, "launch_delay_s": args.launch_delay_s, "elapsed_s": round(time.time() - t0), "summary": summary},
              open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
