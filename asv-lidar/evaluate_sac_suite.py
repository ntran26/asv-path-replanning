"""
Evaluate one SAC agent on the fixed 600-episode ASV evaluation suite.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Sequence

import numpy as np
from stable_baselines3 import SAC

from rl_env import ASVLidarEnv, DEFAULT_EVAL_LAMBDA

# -----------------------------
# User settings
# -----------------------------
MODEL_PATH = "best_model_900000.zip"
SUITE_JSON = "data/env_setup/eval_suite/asv_eval_suite.json"
SUITE_JSON = "data/env_setup/eval_suite_success_filtered/asv_success_suite.json"
OUT_DIR = "eval_results/eval_suite"
DETAIL_CSV = os.path.join(OUT_DIR, "eval_suite_details.csv")
DETAIL_JSON = os.path.join(OUT_DIR, "eval_suite_details.json")
SUMMARY_JSON = os.path.join(OUT_DIR, "eval_suite_summary.json")
SUMMARY_CSV = os.path.join(OUT_DIR, "eval_suite_summary.csv")

MAP_WIDTH = 10.0
MAP_HEIGHT = 25.0
PATH_MODE = "straight"
EVAL_LAMBDA = DEFAULT_EVAL_LAMBDA
MAX_STEPS = 2000
DETERMINISTIC = True

# Set to 100 to quick-test only first 100 scenarios.
LIMIT_SCENARIOS = None

def path_length(points: Sequence[Sequence[float]]) -> float:
    if points is None or len(points) < 2:
        return 0.0
    p = np.asarray(points, dtype=np.float32)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def recompute_path_s(env: ASVLidarEnv) -> None:
    diffs = np.diff(env.path, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1) if len(diffs) else np.array([], dtype=np.float32)
    env.path_s = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)
    total_length = float(env.path_s[-1]) if len(env.path_s) else 1.0
    env.lookahead_distance = max(2.0, env.lookahead_fraction * total_length)


def reset_to_scenario(env: ASVLidarEnv, scenario: Dict[str, Any]) -> Dict[str, np.ndarray]:
    # Reset internal dynamics/sensors first, then overwrite the random layout.
    env.reset(seed=int(scenario.get("seed", 0)))

    sx, sy = scenario["start"]
    gx, gy = scenario["goal"]
    env.start_x, env.start_y = float(sx), float(sy)
    env.goal_x, env.goal_y = float(gx), float(gy)
    env.asv_x, env.asv_y = env.start_x, env.start_y
    env.asv_h = 0.0
    env.asv_w = 0.0
    env.speed_mps = 0.0
    env.u_body = 0.0
    env.v_body = 0.0

    # Reuse the saved path if present, otherwise regenerate from start/goal.
    if "path" in scenario and len(scenario["path"]) >= 2:
        env.path = np.asarray(scenario["path"], dtype=np.float32)
        recompute_path_s(env)
    else:
        env.path = env._generate_path(env.start_x, env.start_y, env.goal_x, env.goal_y)

    env.obstacles = [
        [(float(x), float(y)) for x, y in obs]
        for obs in scenario.get("obstacles", [])
    ]

    env.asv_path = [(env.asv_x, env.asv_y)]
    env.distance_to_goal = float(np.linalg.norm([env.asv_x - env.goal_x, env.asv_y - env.goal_y]))
    env.step_count = 0
    env.elapsed_time = 0.0

    env.lidar_obs.reset()
    env.lidar_reward.reset()
    env.lidar = env.lidar_obs
    env._sample_lambda()
    env._sample_obs_border_mode()
    env.lidar_obs.scan((env.asv_x, env.asv_y), env.asv_h, obstacles=env.obstacles, map_border=env._get_obs_lidar_border())
    env.lidar_reward.scan((env.asv_x, env.asv_y), env.asv_h, obstacles=env.obstacles, map_border=None)
    env.true_border_clearance = env._border_clearance_true()
    env._update_path_relative_states(course_deg=env.asv_h)
    env._update_local_planner_features()
    return env._get_obs()


def term_reason(env: ASVLidarEnv, info: Dict[str, Any], truncated: bool, hit_max_steps: bool) -> str:
    if bool(info.get("reached_goal", False)):
        return "goal"
    if bool(info.get("timeout", False)) or bool(truncated) or bool(hit_max_steps):
        return "timeout"
    if bool(info.get("collided", False)):
        try:
            if env._check_border_collision_only():
                return "border"
        except Exception:
            pass
        return "obstacle"
    return "terminated"


def evaluate_one(model: SAC, env: ASVLidarEnv, scenario: Dict[str, Any]) -> Dict[str, Any]:
    obs = reset_to_scenario(env, scenario)
    done = False
    last_info: Dict[str, Any] = {}
    last_truncated = False
    ep_reward = 0.0
    step_count = 0

    cte = []
    course_error = []
    lookahead_error = []
    speed = []
    rpm = []
    rudder_abs = []
    front_clearance = []
    border_clearance = []
    local_target = []
    min_lidar = []

    while step_count < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        last_info = info if isinstance(info, dict) else {}
        last_truncated = bool(truncated)
        ep_reward += float(reward)
        step_count += 1

        cte.append(abs(float(getattr(env, "cross_track_error", 0.0))))
        course_error.append(abs(float(getattr(env, "course_error", 0.0))))
        lookahead_error.append(abs(float(getattr(env, "lookahead_course_error", 0.0))))
        speed.append(float(getattr(env, "speed_mps", 0.0)))
        rpm.append(float(last_info.get("rpm", 0.0)))
        rudder_abs.append(abs(float(action[0]) * 40.0))
        front_clearance.append(float(getattr(env, "front_clearance", np.nan)))
        border_clearance.append(float(getattr(env, "true_border_clearance", np.nan)))
        local_target.append(float(getattr(env, "local_target_cte", 0.0)))
        try:
            min_lidar.append(float(np.min(env.lidar.ranges)))
        except Exception:
            pass

        if done:
            break

    hit_max_steps = step_count >= MAX_STEPS and not done
    reason = term_reason(env, last_info, last_truncated, hit_max_steps)
    success = 1 if reason == "goal" else 0

    ref_len = path_length(scenario.get("path", []))
    actual_len = path_length(env.asv_path)
    efficiency = float(actual_len / ref_len) if ref_len > 1e-6 else float("nan")

    def mean(x): return float(np.mean(x)) if len(x) else 0.0
    def smin(x): return float(np.nanmin(x)) if len(x) else float("nan")
    def smax(x): return float(np.nanmax(x)) if len(x) else 0.0

    return {
        "case_id": int(scenario["case_id"]),
        "group": str(scenario.get("group", f"obs_{scenario.get('obstacle_count', 0)}")),
        "obstacle_count": int(scenario.get("obstacle_count", len(scenario.get("obstacles", [])))),
        "success": int(success),
        "term_reason": reason,
        "ep_reward": float(ep_reward),
        "ep_len": int(step_count),
        "elapsed_time_s": float(step_count * getattr(env, "elapsed_time", 0.0) / max(step_count, 1)) if step_count else 0.0,
        "mean_abs_cte": mean(cte),
        "max_abs_cte": smax(cte),
        "mean_abs_course_error": mean(course_error),
        "mean_abs_lookahead_error": mean(lookahead_error),
        "mean_speed": mean(speed),
        "min_speed": smin(speed),
        "max_speed": smax(speed),
        "mean_rpm": mean(rpm),
        "min_rpm": smin(rpm),
        "max_rpm": smax(rpm),
        "mean_abs_rudder_deg": mean(rudder_abs),
        "min_front_clearance": smin(front_clearance),
        "min_border_clearance": smin(border_clearance),
        "min_lidar": smin(min_lidar),
        "mean_abs_local_target_cte": mean([abs(x) for x in local_target]),
        "reference_path_length": ref_len,
        "actual_path_length": actual_len,
        "path_efficiency": efficiency,
        "d_end": float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y)),
    }


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = sorted(set(r["obstacle_count"] for r in rows))
    summary = []
    for g in groups + ["all"]:
        subset = rows if g == "all" else [r for r in rows if r["obstacle_count"] == g]
        if not subset:
            continue

        def rate(reason: str) -> float:
            return float(np.mean([1 if r["term_reason"] == reason else 0 for r in subset]))

        def avg(key: str) -> float:
            vals = [float(r[key]) for r in subset if key in r and np.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        summary.append({
            "group": "all" if g == "all" else f"obs_{g}",
            "obstacle_count": -1 if g == "all" else int(g),
            "episodes": len(subset),
            "success_rate": float(np.mean([r["success"] for r in subset])),
            "obstacle_rate": rate("obstacle"),
            "border_rate": rate("border"),
            "timeout_rate": rate("timeout"),
            "mean_ep_len": avg("ep_len"),
            "mean_reward": avg("ep_reward"),
            "mean_abs_cte": avg("mean_abs_cte"),
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUITE_JSON, "r") as f:
        suite = json.load(f)
    scenarios = suite["scenarios"]
    if LIMIT_SCENARIOS is not None:
        scenarios = scenarios[: int(LIMIT_SCENARIOS)]

    print(f"Loading SAC model: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)
    env = ASVLidarEnv(
        render_mode=None,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        max_obs=5,
        path_mode=PATH_MODE,
        lambda_override=EVAL_LAMBDA,
        test_case=None,
        record_video=False,
    )

    rows: List[Dict[str, Any]] = []
    n = len(scenarios)
    for i, sc in enumerate(scenarios, 1):
        row = evaluate_one(model, env, sc)
        rows.append(row)
        if i % 25 == 0 or i == n:
            print(f"{i:4d}/{n}  latest case={row['case_id']} obs={row['obstacle_count']} succ={row['success']} reason={row['term_reason']}")

    summary = summarize(rows)

    with open(DETAIL_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(DETAIL_CSV, rows)
    write_csv(SUMMARY_CSV, summary)

    print("\nSummary:")
    for s in summary:
        print(
            f"{s['group']:>6s}: eps={s['episodes']:3d} "
            f"success={s['success_rate']:.3f} obst={s['obstacle_rate']:.3f} "
            f"border={s['border_rate']:.3f} timeout={s['timeout_rate']:.3f} "
            f"cte={s['mean_abs_cte']:.3f} eff={s['mean_path_efficiency']:.3f}"
        )
    print(f"\nSaved: {DETAIL_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
