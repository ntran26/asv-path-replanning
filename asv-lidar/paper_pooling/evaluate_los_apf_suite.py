"""
Evaluate a traditional LOS + APF baseline on the fixed ASV evaluation suite.

This is intended as a quick baseline comparison against the SAC policy table.
It uses the same rl_env.py dynamics, collision checks, goal condition, LiDAR
configuration, and saved evaluation-suite scenarios, but replaces the SAC policy
with a hand-written controller:

    desired vector = LOS/look-ahead attraction + APF LiDAR repulsion
    rudder action  = proportional heading control toward desired vector
    throttle action = 0.0 by default, giving cruise RPM in the current env

Example:
    python evaluate_los_apf_suite.py \
      --suite-json data/env_setup/eval_suite/asv_eval_suite.json \
      --out-dir eval_results/los_apf_baseline \
      --method los_apf

If the controller turns the wrong way, rerun with:
    --rudder-sign -1

For pure LOS path-following baseline, use:
    --method los

Outputs:
    los_apf_details.json/csv
    los_apf_summary.json/csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from rl_env import ASVLidarEnv, DEFAULT_EVAL_LAMBDA
from asv_lidar import LIDAR_RANGE


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def bearing_deg(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    dx = float(to_xy[0] - from_xy[0])
    dy = float(to_xy[1] - from_xy[1])
    return float(math.degrees(math.atan2(dx, dy)))


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
    """Reset env to a saved suite scenario, including latest border-guard LiDAR."""
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
    if hasattr(env, "lidar_border_guard"):
        env.lidar_border_guard.reset()

    env.lidar = env.lidar_obs
    env._sample_lambda()
    env._sample_obs_border_mode()

    env.lidar_obs.scan(
        (env.asv_x, env.asv_y),
        env.asv_h,
        obstacles=env.obstacles,
        map_border=env._get_obs_lidar_border(),
    )
    env.lidar_reward.scan(
        (env.asv_x, env.asv_y),
        env.asv_h,
        obstacles=env.obstacles,
        map_border=None,
    )
    if hasattr(env, "lidar_border_guard"):
        env.lidar_border_guard.scan(
            (env.asv_x, env.asv_y),
            env.asv_h,
            obstacles=None,
            map_border=env.map_border,
        )

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


# ---------------------------------------------------------------------------
# LOS + APF baseline controller
# ---------------------------------------------------------------------------

@dataclass
class LOSAPFConfig:
    method: str = "los_apf"
    lidar_source: str = "obs"       # obs or reward
    k_att: float = 1.0
    k_cte: float = 0.15             # extra pull toward closest path point
    k_rep: float = 1.15
    repulse_range: float = 5.0
    repulse_power: float = 2.0
    front_gain: float = 1.5
    kp_heading: float = 1.0
    kd_yaw: float = 0.015
    max_error_for_full_rudder_deg: float = 55.0
    rudder_sign: float = 1.0
    throttle_action: float = 0.0    # 0 means cruise RPM for residual speed envs


def _chosen_lidar(env: ASVLidarEnv, source: str):
    if source == "reward" and hasattr(env, "lidar_reward"):
        return env.lidar_reward
    return env.lidar_obs if hasattr(env, "lidar_obs") else env.lidar


def los_apf_action(env: ASVLidarEnv, cfg: LOSAPFConfig) -> np.ndarray:
    """Return continuous action [rudder_action, throttle_action] in [-1, 1]."""
    pos = np.array([float(env.asv_x), float(env.asv_y)], dtype=np.float64)
    lookahead = np.array([float(env.lookahead_x), float(env.lookahead_y)], dtype=np.float64)
    closest = np.array([float(env.tgt_x), float(env.tgt_y)], dtype=np.float64)

    # LOS attraction toward the look-ahead point.
    v_att = lookahead - pos
    n_att = float(np.linalg.norm(v_att))
    if n_att > 1e-9:
        v_att = cfg.k_att * v_att / n_att
    else:
        v_att = np.array([math.sin(math.radians(env.asv_h)), math.cos(math.radians(env.asv_h))], dtype=np.float64)

    # Small extra pull toward the closest point to reduce steady CTE.
    v_cte = closest - pos
    n_cte = float(np.linalg.norm(v_cte))
    if n_cte > 1e-9:
        v_cte = cfg.k_cte * v_cte / max(n_cte, 1.0)
    else:
        v_cte = np.zeros(2, dtype=np.float64)

    v_rep = np.zeros(2, dtype=np.float64)
    if cfg.method.lower() in {"los_apf", "apf"} and cfg.k_rep > 0.0:
        lidar = _chosen_lidar(env, cfg.lidar_source)
        ranges = np.asarray(lidar.ranges, dtype=np.float64).reshape(-1)
        angles = np.asarray(lidar.angles, dtype=np.float64).reshape(-1)
        n = min(ranges.size, angles.size)
        ranges = ranges[:n]
        angles = angles[:n]

        d0 = max(float(cfg.repulse_range), 1e-6)
        for d, rel_ang in zip(ranges, angles):
            d = float(d)
            if not np.isfinite(d):
                continue
            if d <= 1e-6 or d >= d0 or d >= float(LIDAR_RANGE) - 1e-6:
                continue

            # Direction from vessel/LiDAR toward the return in world frame.
            abs_ang = math.radians(float(env.asv_h) + float(rel_ang))
            beam_dir = np.array([math.sin(abs_ang), math.cos(abs_ang)], dtype=np.float64)

            # Classical APF-style repulsive magnitude.
            mag = cfg.k_rep * (1.0 / d - 1.0 / d0) / (d ** max(float(cfg.repulse_power), 1.0))

            # Emphasize forward returns; side/rear returns still contribute weakly.
            front = max(0.0, math.cos(math.radians(float(rel_ang))))
            mag *= (1.0 + float(cfg.front_gain) * front * front)

            # Repulsion is away from the measured return.
            v_rep -= mag * beam_dir

    desired = v_att + v_cte + v_rep
    if float(np.linalg.norm(desired)) < 1e-9:
        desired = v_att

    desired_bearing = math.degrees(math.atan2(float(desired[0]), float(desired[1])))
    heading_error = wrap180(desired_bearing - float(env.asv_h))

    rudder_action = cfg.rudder_sign * (
        cfg.kp_heading * heading_error / max(cfg.max_error_for_full_rudder_deg, 1e-6)
        - cfg.kd_yaw * float(getattr(env, "asv_w", 0.0))
    )
    rudder_action = float(np.clip(rudder_action, -1.0, 1.0))
    throttle_action = float(np.clip(cfg.throttle_action, -1.0, 1.0))
    return np.array([rudder_action, throttle_action], dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation and summary
# ---------------------------------------------------------------------------

def evaluate_one(env: ASVLidarEnv, scenario: Dict[str, Any], cfg: LOSAPFConfig, *, max_steps: int) -> Dict[str, Any]:
    reset_to_scenario(env, scenario)
    last_info: Dict[str, Any] = {}
    last_truncated = False
    ep_reward = 0.0
    step_count = 0

    cte_signed: List[float] = []
    cte_abs: List[float] = []
    course_error: List[float] = []
    lookahead_error: List[float] = []
    speed: List[float] = []
    rpm: List[float] = []
    rudder_abs: List[float] = []
    front_clearance: List[float] = []
    border_clearance: List[float] = []
    min_lidar: List[float] = []

    while step_count < max_steps:
        action = los_apf_action(env, cfg)
        _, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        last_info = info if isinstance(info, dict) else {}
        last_truncated = bool(truncated)
        ep_reward += float(reward)
        step_count += 1

        cte_val = float(getattr(env, "cross_track_error", 0.0))
        cte_signed.append(cte_val)
        cte_abs.append(abs(cte_val))
        course_error.append(abs(float(getattr(env, "course_error", 0.0))))
        lookahead_error.append(abs(float(getattr(env, "lookahead_course_error", 0.0))))
        speed.append(float(getattr(env, "speed_mps", 0.0)))
        rpm.append(float(last_info.get("rpm", 0.0)))
        rudder_abs.append(abs(float(action[0]) * 40.0))
        front_clearance.append(float(getattr(env, "front_clearance", np.nan)))
        border_clearance.append(float(getattr(env, "true_border_clearance", np.nan)))
        try:
            min_lidar.append(float(np.min(env.lidar.ranges)))
        except Exception:
            pass
        if done:
            break

    hit_max_steps = step_count >= max_steps and not bool(last_info.get("reached_goal", False))
    reason = term_reason(env, last_info, last_truncated, hit_max_steps)
    success = 1 if reason == "goal" else 0

    ref_len = path_length(scenario.get("path", []))
    actual_len = path_length(env.asv_path)
    efficiency = float(actual_len / ref_len) if ref_len > 1e-6 else float("nan")

    def mean(x): return float(np.mean(x)) if len(x) else 0.0
    def std(x): return float(np.std(x)) if len(x) else 0.0
    def smin(x): return float(np.nanmin(x)) if len(x) else float("nan")
    def smax(x): return float(np.nanmax(x)) if len(x) else 0.0

    cte_count = int(len(cte_signed))
    cte_sum = float(np.sum(cte_signed)) if cte_signed else 0.0
    cte_sumsq = float(np.sum(np.square(cte_signed))) if cte_signed else 0.0

    return {
        "case_id": int(scenario.get("case_id", -1)),
        "group": str(scenario.get("group", f"obs_{scenario.get('obstacle_count', 0)}")),
        "obstacle_count": int(scenario.get("obstacle_count", len(scenario.get("obstacles", [])))),
        "controller": cfg.method,
        "success": int(success),
        "term_reason": str(reason),
        "ep_reward": float(ep_reward),
        "ep_len": int(step_count),
        "elapsed_time_s": float(getattr(env, "elapsed_time", step_count * 0.1)) if step_count else 0.0,
        "mean_abs_cte": mean(cte_abs),
        "std_cte": std(cte_signed),
        "max_abs_cte": smax(cte_abs),
        "_cte_count": cte_count,
        "_cte_sum": cte_sum,
        "_cte_sumsq": cte_sumsq,
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
        "reference_path_length": ref_len,
        "actual_path_length": actual_len,
        "path_efficiency": efficiency,
        "d_end": float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y)),
    }


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = sorted(set(int(r["obstacle_count"]) for r in rows))
    summary: List[Dict[str, Any]] = []

    for g in groups + ["all"]:
        subset = rows if g == "all" else [r for r in rows if int(r["obstacle_count"]) == int(g)]
        if not subset:
            continue

        def rate(reason: str) -> float:
            return float(np.mean([1 if r["term_reason"] == reason else 0 for r in subset]))

        def avg(key: str) -> float:
            vals = [float(r[key]) for r in subset if key in r and np.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        def pooled_std_cte() -> float:
            n = float(sum(int(r.get("_cte_count", 0)) for r in subset))
            if n <= 0.0:
                return float("nan")
            cte_sum = float(sum(float(r.get("_cte_sum", 0.0)) for r in subset))
            cte_sumsq = float(sum(float(r.get("_cte_sumsq", 0.0)) for r in subset))
            mean_cte = cte_sum / n
            variance = max(0.0, cte_sumsq / n - mean_cte * mean_cte)
            return float(math.sqrt(variance))

        summary.append({
            "group": "all" if g == "all" else f"obs_{g}",
            "obstacle_count": -1 if g == "all" else int(g),
            "episodes": len(subset),
            "success_rate": float(np.mean([int(r["success"]) for r in subset])),
            "obstacle_rate": rate("obstacle"),
            "border_rate": rate("border"),
            "timeout_rate": rate("timeout"),
            "mean_ep_len": avg("ep_len"),
            "mean_elapsed_time_s": avg("elapsed_time_s"),
            "mean_reward": avg("ep_reward"),
            "mean_abs_cte": avg("mean_abs_cte"),
            "std_cte": pooled_std_cte(),
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-json", default="eval_suite/asv_eval_suite.json")
    ap.add_argument("--out-dir", default="eval_results/los_apf_baseline")
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--map-width", type=float, default=10.0)
    ap.add_argument("--map-height", type=float, default=25.0)
    ap.add_argument("--path-mode", default="straight")
    ap.add_argument("--method", choices=["los", "los_apf", "apf"], default="los_apf")
    ap.add_argument("--lidar-source", choices=["obs", "reward"], default="obs")
    ap.add_argument("--k-att", type=float, default=1.0)
    ap.add_argument("--k-cte", type=float, default=0.15)
    ap.add_argument("--k-rep", type=float, default=1.15)
    ap.add_argument("--repulse-range", type=float, default=5.0)
    ap.add_argument("--repulse-power", type=float, default=2.0)
    ap.add_argument("--front-gain", type=float, default=1.5)
    ap.add_argument("--kp-heading", type=float, default=1.0)
    ap.add_argument("--kd-yaw", type=float, default=0.015)
    ap.add_argument("--max-error-for-full-rudder-deg", type=float, default=55.0)
    ap.add_argument("--rudder-sign", type=float, default=1.0)
    ap.add_argument("--throttle-action", type=float, default=0.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.suite_json, "r") as f:
        suite = json.load(f)
    scenarios = suite.get("scenarios", suite if isinstance(suite, list) else [])
    if args.limit_scenarios is not None:
        scenarios = scenarios[: int(args.limit_scenarios)]
    if not scenarios:
        raise RuntimeError(f"No scenarios found in {args.suite_json}")

    cfg = LOSAPFConfig(
        method=args.method,
        lidar_source=args.lidar_source,
        k_att=args.k_att,
        k_cte=args.k_cte,
        k_rep=args.k_rep,
        repulse_range=args.repulse_range,
        repulse_power=args.repulse_power,
        front_gain=args.front_gain,
        kp_heading=args.kp_heading,
        kd_yaw=args.kd_yaw,
        max_error_for_full_rudder_deg=args.max_error_for_full_rudder_deg,
        rudder_sign=args.rudder_sign,
        throttle_action=args.throttle_action,
    )

    env = ASVLidarEnv(
        render_mode=None,
        map_width=args.map_width,
        map_height=args.map_height,
        max_obs=5,
        path_mode=args.path_mode,
        lambda_override=DEFAULT_EVAL_LAMBDA,
        test_case=None,
        record_video=False,
    )

    rows: List[Dict[str, Any]] = []
    n = len(scenarios)
    for i, sc in enumerate(scenarios, 1):
        row = evaluate_one(env, sc, cfg, max_steps=int(args.max_steps))
        rows.append(row)
        if i % 25 == 0 or i == n:
            print(
                f"{i:4d}/{n}  case={row['case_id']} obs={row['obstacle_count']} "
                f"succ={row['success']} reason={row['term_reason']} "
                f"cte={row['mean_abs_cte']:.2f} R={row['ep_reward']:.1f}"
            )

    summary = summarize(rows)
    detail_rows = [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]

    detail_json = os.path.join(args.out_dir, f"{args.method}_details.json")
    summary_json = os.path.join(args.out_dir, f"{args.method}_summary.json")
    detail_csv = os.path.join(args.out_dir, f"{args.method}_details.csv")
    summary_csv = os.path.join(args.out_dir, f"{args.method}_summary.csv")
    config_json = os.path.join(args.out_dir, f"{args.method}_config.json")

    with open(detail_json, "w") as f:
        json.dump(detail_rows, f, indent=2)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(config_json, "w") as f:
        json.dump(vars(args), f, indent=2)
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary)

    print("\nSummary:")
    for s in summary:
        print(
            f"{s['group']:>6s}: eps={s['episodes']:3d} "
            f"success={s['success_rate']:.3f} obst={s['obstacle_rate']:.3f} "
            f"border={s['border_rate']:.3f} timeout={s['timeout_rate']:.3f} "
            f"mean|cte|={s['mean_abs_cte']:.3f} std_cte={s['std_cte']:.3f} "
            f"time={s['mean_elapsed_time_s']:.1f}s eff={s['mean_path_efficiency']:.3f}"
        )

    print(f"\nSaved: {detail_json}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
