"""The reward scale audit (02a §8.2, 02b T6; `OPEN_PROBLEMS.md` C7).

    python tools/scale_audit.py --episodes 240 --workers 10

Runs three fixed policies over the scenario generator at curriculum stage 5 and
reports, per term, the realised episode integral against 02a §8.1's predictions,
the outcome rates, and the orderings §8.1 requires:

* **random**      -- uniform actions: the floor, and every term's worst case;
* **follower**    -- a line-of-sight path follower at cruise that ignores
  targets: the nominal-success design point when nothing is in the way, and a
  maximally non-compliant policy when something is;
* **slowdown**    -- the follower, but it slackens to 0.2 m/s whenever a tracked
  target is engaged and closing: a crude 8(e) behaviour, to check that slowing
  for a target is priced as 02a says it should be.

Writes `results/scale_audit.json` next to `src/`.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

TERMS = ("pf", "prog", "exist", "smooth", "obs", "bnd", "dom", "col")


def follower_action(env, slow_for_targets: bool) -> np.ndarray:
    import constants as cfg
    # Lookahead course error is compass-positive to starboard, and a positive
    # rudder turns to starboard in this convention.
    rudder = float(np.clip(env.lookahead_course_error / 25.0
                           + 0.4 * env.cross_track_error, -1.0, 1.0))
    throttle = 0.0
    if slow_for_targets:
        for ctx in env.encounter_contexts.values():
            if ctx.engaged and ctx.tcpa > 0.0:
                target_rpm = 0.2 / max(cfg.U_REF, 1e-6) * cfg.CRUISE_RPM
                throttle = float(np.clip((target_rpm - cfg.CRUISE_RPM) / max(cfg.RPM_DELTA, 1e-6),
                                         -1.0, 1.0))
                break
    return np.array([rudder, throttle], dtype=np.float32)


def run_chunk(args):
    policy, seeds = args
    import constants as cfg
    import curriculum
    from env import ASVLidarEnv

    curriculum.apply_stage(4)           # full propulsion authority, as training uses
    # Nominal hull: 02a §8.1's design point is defined on the identified plant.
    env = ASVLidarEnv(render_mode=None, scenario_stage=5, scenario_namespace="development",
                      vessel_randomisation=None)
    rng = np.random.default_rng(seeds[0])
    out = []
    for seed in seeds:
        env.reset(seed=int(seed))
        cls = env.scenario.encounter_class
        sums = defaultdict(float)
        total = intervention = 0.0
        steps = 0
        colregs_frames = 0
        while True:
            if policy == "random":
                action = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
            else:
                action = follower_action(env, slow_for_targets=(policy == "slowdown"))
            _, reward, term, trunc, info = env.step(action)
            steps += 1
            total += float(reward)
            intervention += float(info.get("reward/intervention", 0.0))
            for name in TERMS:
                sums[name] += float(info[f"reward/weighted/{name}"])
            colregs_frames += int(info["reward/weighted/col"] < 0.0)
            if term or trunc:
                break
        outcome = ("goal" if info["reached_goal"] else
                   f"collision:{info['collision_kind']}" if info["collided"] else "timeout")
        out.append({"policy": policy, "seed": int(seed), "class": cls, "outcome": outcome,
                    "steps": steps, "return": total, "terminal": float(info["reward/terminal"]),
                    "intervention": intervention, "estops": int(info["estop/events"]),
                    "colregs_frames": colregs_frames, "width": float(env.corridor_width),
                    **{f"sum_{k}": v for k, v in sums.items()}})
    return out


def summarise(rows):
    from reward.audit import PREDICTED_NOMINAL, check_against_prediction
    report = {}
    for policy in sorted({r["policy"] for r in rows}):
        mine = [r for r in rows if r["policy"] == policy]
        outcomes = defaultdict(int)
        for r in mine:
            outcomes[r["outcome"]] += 1
        n = len(mine)
        block = {"episodes": n,
                 "outcome_rates": {k: v / n for k, v in sorted(outcomes.items())},
                 "mean_return": float(np.mean([r["return"] for r in mine])),
                 "mean_steps": float(np.mean([r["steps"] for r in mine])),
                 "estops_per_episode": float(np.mean([r["estops"] for r in mine])),
                 "term_integrals_mean": {k: float(np.mean([r[f"sum_{k}"] for r in mine]))
                                         for k in TERMS}}
        by_outcome = {}
        for outcome in sorted(outcomes):
            sel = [r for r in mine if r["outcome"] == outcome]
            by_outcome[outcome] = {
                "n": len(sel),
                "mean_return": float(np.mean([r["return"] for r in sel])),
                "term_integrals_mean": {k: float(np.mean([r[f"sum_{k}"] for r in sel])) for k in TERMS}}
        block["by_outcome"] = by_outcome
        by_class = {}
        for cls in sorted({r["class"] for r in mine}):
            sel = [r for r in mine if r["class"] == cls]
            goal = [r for r in sel if r["outcome"] == "goal"]
            by_class[cls] = {"n": len(sel),
                             "goal_rate": len(goal) / len(sel),
                             "collision_rate": sum(r["outcome"].startswith("collision") for r in sel) / len(sel),
                             "mean_return": float(np.mean([r["return"] for r in sel])),
                             "mean_col_integral": float(np.mean([r["sum_col"] for r in sel]))}
        block["by_class"] = by_class
        report[policy] = block

    # 02a §8.1 against the nominal design point: follower successes with no target.
    nominal = [r for r in rows if r["policy"] == "follower" and r["outcome"] == "goal"
               and r["class"] == "no_target"]
    if nominal:
        sums = {k: float(np.mean([r[f"sum_{k}"] for r in nominal])) for k in TERMS}
        report["design_point"] = {
            "episodes": len(nominal),
            "realised": sums,
            "predicted": dict(PREDICTED_NOMINAL),
            "problems": check_against_prediction(sums),
            "mean_return": float(np.mean([r["return"] for r in nominal])),
        }

    # The orderings: nominal success > non-compliant success > any collision.
    def mean_return(pred):
        sel = [r["return"] for r in rows if pred(r)]
        return (float(np.mean(sel)), len(sel)) if sel else (None, 0)
    report["orderings"] = {
        "nominal_success_no_target": mean_return(lambda r: r["policy"] == "follower" and r["class"] == "no_target" and r["outcome"] == "goal"),
        "success_with_colregs_penalty": mean_return(lambda r: r["outcome"] == "goal" and r["sum_col"] < -1.0),
        "collision_any_policy": mean_return(lambda r: r["outcome"].startswith("collision")),
        "timeout_any_policy": mean_return(lambda r: r["outcome"] == "timeout"),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "scale_audit.json"))
    args = ap.parse_args()

    jobs = []
    per = max(1, args.episodes // args.workers)
    for p_i, policy in enumerate(("random", "follower", "slowdown")):
        for w in range(args.workers):
            base = 1000 * p_i + w * per
            jobs.append((policy, list(range(base, base + per))))
    with mp.Pool(args.workers) as pool:
        rows = [row for chunk in pool.map(run_chunk, jobs) for row in chunk]

    report = summarise(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"report": report, "episodes": rows}, fh, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
