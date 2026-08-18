"""Evaluate one trained agent on the fixed holdout suite.

Edit the USER SETTINGS block below, then run:

    python src/evaluate_suite.py

Writes per-case details and a per-obstacle-count summary to OUT_DIR, as both
CSV and JSON.  The defaults reproduce the SAC baseline numbers quoted in the
README; for another algorithm set ALGO, MODEL_PATH, OUT_DIR and FILE_PREFIX
together, e.g. ALGO="ppo", OUT_DIR="eval_results/eval_suite_ppo",
FILE_PREFIX="ppo_eval_suite".
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Sequence

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3

import rollout
from env import ASVLidarEnv
from rollout import run_episode

# -----------------------------
# USER SETTINGS
# -----------------------------
ALGO = "sac"                                    # sac | ppo | td3 | ddpg
MODEL_PATH = "models/sac_model_1M.zip"
SUITE_JSON = "eval_suite/asv_eval_suite.json"
OUT_DIR = "eval_results/eval_suite"
FILE_PREFIX = "eval_suite"

MAP_WIDTH = 10.0
MAP_HEIGHT = 25.0
PATH_MODE = "straight"
MAX_STEPS = 2000
DETERMINISTIC = True

# Set to an integer to smoke-test on the first N scenarios only.
LIMIT_SCENARIOS = None

ALGOS = {"sac": SAC, "ppo": PPO, "td3": TD3, "ddpg": DDPG}


def path_length(points: Sequence[Sequence[float]]) -> float:
    if points is None or len(points) < 2:
        return 0.0
    p = np.asarray(points, dtype=np.float32)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def evaluate_one(model, env: ASVLidarEnv, scenario: Dict[str, Any]) -> Dict[str, Any]:
    episode = run_episode(
        model, env, deterministic=DETERMINISTIC, max_steps=MAX_STEPS,
        reset_kwargs={"seed": int(scenario.get("seed", 0)), "options": {"scenario": scenario}},
    )
    track = episode.track
    cte = track("cross_track_error")

    reference_length = path_length(scenario.get("path", []))
    actual_length = path_length(env.asv_path)

    return {
        "case_id": int(scenario["case_id"]),
        "group": str(scenario.get("group", f"obs_{scenario.get('obstacle_count', 0)}")),
        "obstacle_count": int(scenario.get("obstacle_count", len(scenario.get("obstacles", [])))),
        "success": episode.success,
        "term_reason": episode.reason,
        "ep_reward": episode.reward,
        "ep_len": episode.steps,
        "elapsed_time_s": float(env.elapsed_time) if episode.steps else 0.0,
        "mean_abs_cte": rollout.abs_mean(cte),
        "std_cte": rollout.std(cte),
        "max_abs_cte": rollout.abs_max(cte),
        # Moments kept so group std_cte can be pooled exactly; stripped before writing.
        "_cte_count": len(cte),
        "_cte_sum": float(np.sum(cte)) if cte else 0.0,
        "_cte_sumsq": float(np.sum(np.square(cte))) if cte else 0.0,
        "mean_abs_course_error": rollout.abs_mean(track("course_error")),
        "mean_abs_lookahead_error": rollout.abs_mean(track("lookahead_course_error")),
        "mean_speed": rollout.mean(track("speed_mps")),
        "min_speed": rollout.smallest(track("speed_mps")),
        "max_speed": rollout.largest(track("speed_mps")),
        "mean_rpm": rollout.mean(track("rpm")),
        "min_rpm": rollout.smallest(track("rpm")),
        "max_rpm": rollout.largest(track("rpm")),
        "mean_abs_rudder_deg": rollout.abs_mean(track("rudder_deg")),
        "min_front_clearance": rollout.smallest(track("front_clearance")),
        "min_border_clearance": rollout.smallest(track("true_border_clearance")),
        "min_lidar": rollout.smallest(track("min_lidar")),
        "mean_abs_local_target_cte": rollout.abs_mean(track("local_target_cte")),
        "reference_path_length": reference_length,
        "actual_path_length": actual_length,
        "path_efficiency": actual_length / reference_length if reference_length > 1e-6 else float("nan"),
        "d_end": episode.d_end,
    }


def summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for group in sorted({r["obstacle_count"] for r in rows}) + ["all"]:
        subset = rows if group == "all" else [r for r in rows if r["obstacle_count"] == group]
        if not subset:
            continue

        def rate(reason: str) -> float:
            return float(np.mean([r["term_reason"] == reason for r in subset]))

        def avg(key: str) -> float:
            values = [float(r[key]) for r in subset if np.isfinite(float(r[key]))]
            return float(np.mean(values)) if values else float("nan")

        # Pooled over every step in the group, not a mean of per-episode stds.
        n = sum(r["_cte_count"] for r in subset)
        if n > 0:
            mean_cte = sum(r["_cte_sum"] for r in subset) / n
            variance = max(0.0, sum(r["_cte_sumsq"] for r in subset) / n - mean_cte ** 2)
            std_cte = math.sqrt(variance)
        else:
            std_cte = float("nan")

        summary.append({
            "group": "all" if group == "all" else f"obs_{group}",
            "obstacle_count": -1 if group == "all" else int(group),
            "episodes": len(subset),
            "success_rate": float(np.mean([r["success"] for r in subset])),
            "obstacle_rate": rate("obstacle"),
            "border_rate": rate("border"),
            "timeout_rate": rate("timeout"),
            "mean_ep_len": avg("ep_len"),
            "mean_elapsed_time_s": avg("elapsed_time_s"),
            "mean_reward": avg("ep_reward"),
            "mean_abs_cte": avg("mean_abs_cte"),
            "std_cte": std_cte,
            "mean_abs_course_error": avg("mean_abs_course_error"),
            "mean_speed": avg("mean_speed"),
            "mean_rpm": avg("mean_rpm"),
            "mean_path_efficiency": avg("path_efficiency"),
            "mean_min_front_clearance": avg("min_front_clearance"),
            "mean_min_border_clearance": avg("min_border_clearance"),
        })
    return summary


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUITE_JSON, "r") as f:
        scenarios = json.load(f)["scenarios"]
    if LIMIT_SCENARIOS is not None:
        scenarios = scenarios[:int(LIMIT_SCENARIOS)]

    print(f"Loading {ALGO.upper()} model: {MODEL_PATH}")
    model = ALGOS[ALGO].load(MODEL_PATH, device="auto")
    env = ASVLidarEnv(map_width=MAP_WIDTH, map_height=MAP_HEIGHT, max_obs=5, path_mode=PATH_MODE)

    rows = []
    for i, scenario in enumerate(scenarios, 1):
        row = evaluate_one(model, env, scenario)
        rows.append(row)
        if i % 25 == 0 or i == len(scenarios):
            print(f"{i:4d}/{len(scenarios)}  latest case={row['case_id']} "
                  f"obs={row['obstacle_count']} succ={row['success']} reason={row['term_reason']}")

    summary = summarise(rows)
    details = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]

    detail_csv = os.path.join(OUT_DIR, f"{FILE_PREFIX}_details.csv")
    summary_csv = os.path.join(OUT_DIR, f"{FILE_PREFIX}_summary.csv")
    with open(os.path.join(OUT_DIR, f"{FILE_PREFIX}_details.json"), "w") as f:
        json.dump(details, f, indent=2)
    with open(os.path.join(OUT_DIR, f"{FILE_PREFIX}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(detail_csv, details)
    write_csv(summary_csv, summary)

    print("\nSummary:")
    for s in summary:
        print(f"{s['group']:>6s}: eps={s['episodes']:3d} "
              f"success={s['success_rate']:.3f} obst={s['obstacle_rate']:.3f} "
              f"border={s['border_rate']:.3f} timeout={s['timeout_rate']:.3f} "
              f"mean|cte|={s['mean_abs_cte']:.3f} std_cte={s['std_cte']:.3f} "
              f"time={s['mean_elapsed_time_s']:.1f}s eff={s['mean_path_efficiency']:.3f}")
    print(f"\nSaved: {detail_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
