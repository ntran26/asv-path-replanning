"""
Tune/evaluate a traditional LOS + APF baseline on a fixed ASV evaluation suite.

This is a stronger and fairer version of the quick LOS+APF baseline. It supports:

  1) obstacle-only LiDAR for APF via --lidar-source reward
  2) observation LiDAR via --lidar-source obs, if you want realistic border returns
  3) several hand-tuned presets
  4) a small built-in grid/preset search using --grid-search
  5) optional side-memory / tangential APF to reduce oscillation
  6) summary metrics both over all episodes and successful episodes only

Typical workflow:

  # 1. Tune on a smaller validation subset, obstacle-only LiDAR
  python evaluate_los_apf_tuning.py \
    --suite-json data/env_setup/eval_suite_500/asv_eval_suite_500.json \
    --out-dir eval_results/los_apf_tune_reward \
    --lidar-source reward \
    --grid-search \
    --per-group-limit 20

  # 2. Evaluate the best preset/config on the full suite
  python evaluate_los_apf_tuning.py \
    --suite-json data/env_setup/eval_suite_500/asv_eval_suite_500.json \
    --out-dir eval_results/los_apf_best_reward \
    --lidar-source reward \
    --preset conservative_side

  # 3. Stress-test realistic LiDAR with borders/walls visible
  python evaluate_los_apf_tuning.py \
    --suite-json data/env_setup/eval_suite_500/asv_eval_suite_500.json \
    --out-dir eval_results/los_apf_best_obs \
    --lidar-source obs \
    --preset conservative_side

Notes:
  - --lidar-source reward uses env.lidar_reward, which should be obstacle-only in your current env.
  - --lidar-source obs uses env.lidar_obs, which may include border/wall returns depending on OBS_BORDER_MODE.
  - Pure LOS can be tested with --method los.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rl_env import ASVLidarEnv, DEFAULT_EVAL_LAMBDA
from asv_lidar import LIDAR_RANGE


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


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


def scenario_list_from_json(suite_obj: Any) -> List[Dict[str, Any]]:
    """Support several suite formats used during this project."""
    if isinstance(suite_obj, list):
        return suite_obj
    if isinstance(suite_obj, dict):
        if "scenarios" in suite_obj:
            return list(suite_obj["scenarios"])
        if "cases" in suite_obj:
            return list(suite_obj["cases"])
    raise ValueError("Suite JSON must be a list or contain 'scenarios'/'cases'.")


def subsample_per_group(scenarios: List[Dict[str, Any]], per_group_limit: Optional[int]) -> List[Dict[str, Any]]:
    if per_group_limit is None or per_group_limit <= 0:
        return scenarios
    out: List[Dict[str, Any]] = []
    counts: Dict[int, int] = {}
    for sc in scenarios:
        nobs = int(sc.get("obstacle_count", len(sc.get("obstacles", []))))
        if counts.get(nobs, 0) < per_group_limit:
            out.append(sc)
            counts[nobs] = counts.get(nobs, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Environment scenario injection
# ---------------------------------------------------------------------------


def reset_to_scenario(env: ASVLidarEnv, scenario: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Reset env to a saved suite scenario."""
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
        recompute_path_s(env)

    env.obstacles = [[(float(x), float(y)) for x, y in obs] for obs in scenario.get("obstacles", [])]
    env.num_obs = int(len(env.obstacles))

    env.asv_path = [(env.asv_x, env.asv_y)]
    env.distance_to_goal = float(np.linalg.norm([env.asv_x - env.goal_x, env.asv_y - env.goal_y]))
    env.step_count = 0
    env.elapsed_time = 0.0

    # Current env variants use lidar_obs/lidar_reward. Keep this robust.
    if hasattr(env, "lidar_obs"):
        env.lidar_obs.reset()
    if hasattr(env, "lidar_reward"):
        env.lidar_reward.reset()
    if hasattr(env, "lidar_border_guard"):
        env.lidar_border_guard.reset()
    if hasattr(env, "lidar_obs"):
        env.lidar = env.lidar_obs

    if hasattr(env, "_sample_lambda"):
        env._sample_lambda()
    if hasattr(env, "_sample_obs_border_mode"):
        env._sample_obs_border_mode()

    # Scan observation lidar.
    obs_border = env._get_obs_lidar_border() if hasattr(env, "_get_obs_lidar_border") else None
    if hasattr(env, "lidar_obs"):
        env.lidar_obs.scan((env.asv_x, env.asv_y), env.asv_h, obstacles=env.obstacles, map_border=obs_border)
    else:
        env.lidar.scan((env.asv_x, env.asv_y), env.asv_h, obstacles=env.obstacles, map_border=obs_border)

    # Scan reward lidar as obstacle-only if available.
    if hasattr(env, "lidar_reward"):
        env.lidar_reward.scan((env.asv_x, env.asv_y), env.asv_h, obstacles=env.obstacles, map_border=None)

    if hasattr(env, "lidar_border_guard"):
        env.lidar_border_guard.scan((env.asv_x, env.asv_y), env.asv_h, obstacles=None, map_border=env.map_border)

    if hasattr(env, "_border_clearance_true"):
        env.true_border_clearance = env._border_clearance_true()

    if hasattr(env, "_update_path_relative_states"):
        env._update_path_relative_states(course_deg=env.asv_h)
    if hasattr(env, "_update_local_planner_features"):
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
# LOS + APF controller
# ---------------------------------------------------------------------------


@dataclass
class LOSAPFConfig:
    name: str = "custom"
    method: str = "los_apf_side"  # los, los_apf, los_apf_side, apf
    lidar_source: str = "reward"  # reward or obs

    # LOS/path attraction
    k_att: float = 0.8
    k_cte: float = 0.10

    # Repulsion
    k_rep: float = 2.0
    repulse_range: float = 6.0
    repulse_power: float = 1.5
    front_gain: float = 2.0
    max_rep_norm: float = 5.0

    # Tangential/side bias for underactuated ASV
    k_tangent: float = 1.2
    block_threshold: float = 4.5
    clear_threshold: float = 6.5
    side_arc_min_deg: float = 15.0
    side_arc_max_deg: float = 100.0
    side_tie: float = 0.25
    side_memory: bool = True
    default_side: str = "right"  # right/starboard or left

    # Heading controller
    kp_heading: float = 0.8
    kd_yaw: float = 0.02
    max_error_for_full_rudder_deg: float = 70.0
    rudder_sign: float = 1.0
    throttle_action: float = 0.0


PRESETS: Dict[str, LOSAPFConfig] = {
    "los": LOSAPFConfig(name="los", method="los", k_rep=0.0, k_tangent=0.0),
    "balanced": LOSAPFConfig(
        name="balanced", method="los_apf", k_att=0.8, k_cte=0.10, k_rep=2.0,
        repulse_range=6.0, repulse_power=1.5, front_gain=2.0, kp_heading=0.9,
        max_error_for_full_rudder_deg=60.0, k_tangent=0.0, side_memory=False,
    ),
    "conservative": LOSAPFConfig(
        name="conservative", method="los_apf", k_att=0.7, k_cte=0.08, k_rep=3.0,
        repulse_range=7.0, repulse_power=1.5, front_gain=2.5, kp_heading=0.8,
        max_error_for_full_rudder_deg=70.0, k_tangent=0.0, side_memory=False,
    ),
    "strong_path": LOSAPFConfig(
        name="strong_path", method="los_apf", k_att=1.0, k_cte=0.15, k_rep=2.0,
        repulse_range=5.5, repulse_power=2.0, front_gain=2.0, kp_heading=0.9,
        max_error_for_full_rudder_deg=55.0, k_tangent=0.0, side_memory=False,
    ),
    "conservative_side": LOSAPFConfig(
        name="conservative_side", method="los_apf_side", k_att=0.7, k_cte=0.08, k_rep=2.5,
        repulse_range=7.0, repulse_power=1.5, front_gain=2.0, k_tangent=1.4,
        block_threshold=5.0, clear_threshold=7.0, kp_heading=0.8,
        max_error_for_full_rudder_deg=70.0, side_memory=True,
    ),
    "gap_side": LOSAPFConfig(
        name="gap_side", method="los_apf_side", k_att=0.8, k_cte=0.08, k_rep=1.8,
        repulse_range=6.5, repulse_power=1.2, front_gain=1.5, k_tangent=1.8,
        block_threshold=5.0, clear_threshold=7.0, kp_heading=0.8,
        max_error_for_full_rudder_deg=75.0, side_memory=True,
    ),
}


class LOSAPFController:
    def __init__(self, cfg: LOSAPFConfig):
        self.cfg = cfg
        self.active_side: Optional[str] = None

    def reset(self) -> None:
        self.active_side = None

    def _chosen_lidar(self, env: ASVLidarEnv):
        if self.cfg.lidar_source == "reward" and hasattr(env, "lidar_reward"):
            return env.lidar_reward
        return env.lidar_obs if hasattr(env, "lidar_obs") else env.lidar

    def _clearance_stats(self, env: ASVLidarEnv, ranges: np.ndarray, angles: np.ndarray) -> Tuple[float, float, float]:
        front_mask = np.abs(angles) <= 25.0
        left_mask = (angles <= -self.cfg.side_arc_min_deg) & (angles >= -self.cfg.side_arc_max_deg)
        right_mask = (angles >= self.cfg.side_arc_min_deg) & (angles <= self.cfg.side_arc_max_deg)

        def pct(mask, p=20.0, default=LIDAR_RANGE):
            vals = ranges[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return float(default)
            return float(np.percentile(vals, p))

        return pct(front_mask, 10.0), pct(left_mask, 20.0), pct(right_mask, 20.0)

    def action(self, env: ASVLidarEnv) -> np.ndarray:
        cfg = self.cfg
        pos = np.array([float(env.asv_x), float(env.asv_y)], dtype=np.float64)
        lookahead = np.array([float(env.lookahead_x), float(env.lookahead_y)], dtype=np.float64)
        closest = np.array([float(env.tgt_x), float(env.tgt_y)], dtype=np.float64)

        # LOS attraction toward lookahead.
        v_att = lookahead - pos
        n_att = float(np.linalg.norm(v_att))
        if n_att > 1e-9:
            v_att = cfg.k_att * v_att / n_att
        else:
            v_att = np.array([math.sin(math.radians(env.asv_h)), math.cos(math.radians(env.asv_h))], dtype=np.float64)

        # Small pull back to closest path point.
        v_cte = closest - pos
        n_cte = float(np.linalg.norm(v_cte))
        if n_cte > 1e-9:
            v_cte = cfg.k_cte * v_cte / max(n_cte, 1.0)
        else:
            v_cte = np.zeros(2, dtype=np.float64)

        lidar = self._chosen_lidar(env)
        ranges = np.asarray(lidar.ranges, dtype=np.float64).reshape(-1)
        angles = np.asarray(lidar.angles, dtype=np.float64).reshape(-1)
        n = min(ranges.size, angles.size)
        ranges = ranges[:n]
        angles = angles[:n]

        v_rep = np.zeros(2, dtype=np.float64)
        if cfg.method.lower() in {"los_apf", "los_apf_side", "apf"} and cfg.k_rep > 0.0:
            d0 = max(float(cfg.repulse_range), 1e-6)
            for d, rel_ang in zip(ranges, angles):
                d = float(d)
                if not np.isfinite(d):
                    continue
                if d <= 1e-6 or d >= d0 or d >= float(LIDAR_RANGE) - 1e-6:
                    continue

                abs_ang = math.radians(float(env.asv_h) + float(rel_ang))
                beam_dir = np.array([math.sin(abs_ang), math.cos(abs_ang)], dtype=np.float64)
                mag = cfg.k_rep * (1.0 / d - 1.0 / d0) / (d ** max(float(cfg.repulse_power), 1.0))
                front = max(0.0, math.cos(math.radians(float(rel_ang))))
                mag *= (1.0 + float(cfg.front_gain) * front * front)
                v_rep -= mag * beam_dir

            rep_norm = float(np.linalg.norm(v_rep))
            if rep_norm > cfg.max_rep_norm > 0.0:
                v_rep = v_rep / rep_norm * cfg.max_rep_norm

        # Underactuated ASV tangential pass-side cue.
        v_tan = np.zeros(2, dtype=np.float64)
        if cfg.method.lower() == "los_apf_side" and cfg.k_tangent > 0.0:
            front_clear, left_clear, right_clear = self._clearance_stats(env, ranges, angles)

            if self.active_side is not None and front_clear > cfg.clear_threshold:
                self.active_side = None

            if front_clear < cfg.block_threshold and self.active_side is None:
                if abs(right_clear - left_clear) < cfg.side_tie:
                    self.active_side = cfg.default_side
                elif right_clear > left_clear:
                    self.active_side = "right"
                else:
                    self.active_side = "left"

            if self.active_side is not None:
                # Forward vector in world frame.
                h = math.radians(float(env.asv_h))
                fwd = np.array([math.sin(h), math.cos(h)], dtype=np.float64)
                # Right/starboard is [cos(h), -sin(h)] in the current x-right/y-up convention.
                right_vec = np.array([math.cos(h), -math.sin(h)], dtype=np.float64)
                side_vec = right_vec if self.active_side == "right" else -right_vec
                block_alpha = float(np.clip((cfg.block_threshold - front_clear) / max(cfg.block_threshold - 1.0, 1e-6), 0.0, 1.0))
                # Add mostly side motion, with a little forward component to avoid backwards APF minima.
                v_tan = cfg.k_tangent * block_alpha * (0.85 * side_vec + 0.15 * fwd)

        desired = v_att + v_cte + v_rep + v_tan
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


def evaluate_one(env: ASVLidarEnv, scenario: Dict[str, Any], controller: LOSAPFController, *, max_steps: int) -> Dict[str, Any]:
    reset_to_scenario(env, scenario)
    controller.reset()

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
    delta_rudder: List[float] = []
    front_clearance: List[float] = []
    border_clearance: List[float] = []
    min_lidar: List[float] = []
    prev_rudder: Optional[float] = None

    while step_count < max_steps:
        action = controller.action(env)
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
        rudder_deg = float(action[0]) * 40.0
        rudder_abs.append(abs(rudder_deg))
        if prev_rudder is not None:
            delta_rudder.append(abs(rudder_deg - prev_rudder))
        prev_rudder = rudder_deg
        front_clearance.append(float(getattr(env, "front_clearance", np.nan)))
        border_clearance.append(float(getattr(env, "true_border_clearance", np.nan)))
        try:
            lidar = controller._chosen_lidar(env)
            min_lidar.append(float(np.min(lidar.ranges)))
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
    cte_sumsq = float(np.sum(np.square(cte_signed)) if cte_signed else 0.0)

    return {
        "case_id": int(scenario.get("case_id", -1)),
        "group": str(scenario.get("group", f"obs_{scenario.get('obstacle_count', 0)}")),
        "obstacle_count": int(scenario.get("obstacle_count", len(scenario.get("obstacles", [])))),
        "controller": controller.cfg.name,
        "method": controller.cfg.method,
        "lidar_source": controller.cfg.lidar_source,
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
        "mean_abs_delta_rudder_deg": mean(delta_rudder),
        "min_front_clearance": smin(front_clearance),
        "min_border_clearance": smin(border_clearance),
        "min_lidar": smin(min_lidar),
        "reference_path_length": ref_len,
        "actual_path_length": actual_len,
        "path_efficiency": efficiency,
        "d_end": float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y)),
    }


def summarize(rows: List[Dict[str, Any]], *, config_name: str = "") -> List[Dict[str, Any]]:
    groups: List[Any] = sorted(set(int(r["obstacle_count"]) for r in rows)) + ["all"]
    summary: List[Dict[str, Any]] = []

    for g in groups:
        subset = rows if g == "all" else [r for r in rows if int(r["obstacle_count"]) == int(g)]
        if not subset:
            continue
        success_subset = [r for r in subset if int(r.get("success", 0)) == 1]

        def rate(reason: str) -> float:
            return float(np.mean([1 if r["term_reason"] == reason else 0 for r in subset]))

        def avg(key: str, src: Optional[List[Dict[str, Any]]] = None) -> float:
            src = subset if src is None else src
            vals = []
            for r in src:
                try:
                    v = float(r[key])
                    if np.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            return float(np.mean(vals)) if vals else float("nan")

        def pooled_std_cte(src: Optional[List[Dict[str, Any]]] = None) -> float:
            src = subset if src is None else src
            n = float(sum(int(r.get("_cte_count", 0)) for r in src))
            if n <= 0.0:
                return float("nan")
            cte_sum = float(sum(float(r.get("_cte_sum", 0.0)) for r in src))
            cte_sumsq = float(sum(float(r.get("_cte_sumsq", 0.0)) for r in src))
            mean_cte = cte_sum / n
            variance = max(0.0, cte_sumsq / n - mean_cte * mean_cte)
            return float(math.sqrt(variance))

        summary.append({
            "config": config_name or str(subset[0].get("controller", "")),
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
            "success_mean_abs_cte": avg("mean_abs_cte", success_subset),
            "success_mean_elapsed_time_s": avg("elapsed_time_s", success_subset),
            "success_mean_path_efficiency": avg("path_efficiency", success_subset),
            "mean_abs_delta_rudder_deg": avg("mean_abs_delta_rudder_deg"),
            "mean_min_front_clearance": avg("min_front_clearance"),
            "mean_min_border_clearance": avg("min_border_clearance"),
        })
    return summary


def score_summary(summary_rows: List[Dict[str, Any]]) -> float:
    all_row = next((r for r in summary_rows if r.get("group") == "all"), summary_rows[-1])
    return float(
        10.0 * all_row.get("success_rate", 0.0)
        - 5.0 * all_row.get("obstacle_rate", 0.0)
        - 3.0 * all_row.get("border_rate", 0.0)
        - 2.0 * all_row.get("timeout_rate", 0.0)
        - 0.8 * all_row.get("success_mean_abs_cte", all_row.get("mean_abs_cte", 0.0))
        - 0.02 * all_row.get("mean_abs_delta_rudder_deg", 0.0)
    )


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Tuning presets/grid
# ---------------------------------------------------------------------------


def config_from_args(args: argparse.Namespace) -> LOSAPFConfig:
    if args.preset not in PRESETS:
        raise ValueError(f"Unknown preset {args.preset}. Options: {sorted(PRESETS)}")
    cfg = replace(PRESETS[args.preset])
    cfg.lidar_source = args.lidar_source
    cfg.rudder_sign = args.rudder_sign
    cfg.throttle_action = args.throttle_action

    # CLI overrides; None means use preset value.
    for attr in [
        "k_att", "k_cte", "k_rep", "repulse_range", "repulse_power", "front_gain",
        "k_tangent", "block_threshold", "clear_threshold", "side_tie",
        "kp_heading", "kd_yaw", "max_error_for_full_rudder_deg",
    ]:
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, float(val))
    if args.no_side_memory:
        cfg.side_memory = False
    return cfg


def grid_configs(lidar_source: str, rudder_sign: float, throttle_action: float) -> List[LOSAPFConfig]:
    """Small practical grid: presets plus a few parameter variations."""
    configs: List[LOSAPFConfig] = []
    base_names = ["balanced", "conservative", "strong_path", "conservative_side", "gap_side"]
    for name in base_names:
        cfg = replace(PRESETS[name])
        cfg.lidar_source = lidar_source
        cfg.rudder_sign = rudder_sign
        cfg.throttle_action = throttle_action
        configs.append(cfg)

    # Focused variations around side-memory configurations.
    for k_rep in [1.5, 2.5, 3.5]:
        for k_tan in [0.8, 1.4, 2.0]:
            for rr in [5.5, 7.0]:
                cfg = replace(PRESETS["conservative_side"])
                cfg.name = f"side_kr{k_rep}_kt{k_tan}_rr{rr}"
                cfg.lidar_source = lidar_source
                cfg.rudder_sign = rudder_sign
                cfg.throttle_action = throttle_action
                cfg.k_rep = k_rep
                cfg.k_tangent = k_tan
                cfg.repulse_range = rr
                configs.append(cfg)
    return configs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-json", default="eval_suite/asv_eval_suite.json")
    ap.add_argument("--out-dir", default="eval_results/los_apf_tuning")
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument("--per-group-limit", type=int, default=None,
                    help="Use only first N scenarios from each obstacle-count group; useful for tuning.")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--map-width", type=float, default=10.0)
    ap.add_argument("--map-height", type=float, default=25.0)
    ap.add_argument("--path-mode", default="straight")

    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="conservative_side")
    ap.add_argument("--grid-search", action="store_true", help="Evaluate a small set of candidate configurations.")
    ap.add_argument("--method", choices=["los", "los_apf", "los_apf_side", "apf"], default=None,
                    help="Optional override of preset method.")
    ap.add_argument("--lidar-source", choices=["obs", "reward"], default="reward")

    # Optional numeric overrides. Default None means use preset.
    for name in [
        "k-att", "k-cte", "k-rep", "repulse-range", "repulse-power", "front-gain",
        "k-tangent", "block-threshold", "clear-threshold", "side-tie",
        "kp-heading", "kd-yaw", "max-error-for-full-rudder-deg",
    ]:
        ap.add_argument(f"--{name}", type=float, default=None)

    ap.add_argument("--no-side-memory", action="store_true")
    ap.add_argument("--rudder-sign", type=float, default=1.0)
    ap.add_argument("--throttle-action", type=float, default=0.0,
                    help="0.0 should correspond to cruise RPM in your current residual-speed env.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.suite_json, "r") as f:
        suite = json.load(f)
    scenarios = scenario_list_from_json(suite)
    scenarios = subsample_per_group(scenarios, args.per_group_limit)
    if args.limit_scenarios is not None:
        scenarios = scenarios[: int(args.limit_scenarios)]
    if not scenarios:
        raise RuntimeError(f"No scenarios found in {args.suite_json}")

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

    if args.grid_search:
        configs = grid_configs(args.lidar_source, args.rudder_sign, args.throttle_action)
    else:
        cfg = config_from_args(args)
        if args.method is not None:
            cfg.method = args.method
        configs = [cfg]

    all_tuning_rows: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_name = ""

    for cfg_i, cfg in enumerate(configs, 1):
        controller = LOSAPFController(cfg)
        rows: List[Dict[str, Any]] = []
        n = len(scenarios)
        print(f"\n=== Config {cfg_i}/{len(configs)}: {cfg.name}  method={cfg.method} lidar={cfg.lidar_source} ===")
        for i, sc in enumerate(scenarios, 1):
            row = evaluate_one(env, sc, controller, max_steps=int(args.max_steps))
            rows.append(row)
            if (not args.grid_search) and (i % 25 == 0 or i == n):
                print(
                    f"{i:4d}/{n} case={row['case_id']} obs={row['obstacle_count']} "
                    f"succ={row['success']} reason={row['term_reason']} "
                    f"cte={row['mean_abs_cte']:.2f} R={row['ep_reward']:.1f}"
                )

        summary = summarize(rows, config_name=cfg.name)
        score = score_summary(summary)
        all_row = next((r for r in summary if r["group"] == "all"), summary[-1])
        tuning_row = dict(all_row)
        tuning_row["score"] = float(score)
        tuning_row.update({f"param_{k}": v for k, v in asdict(cfg).items()})
        all_tuning_rows.append(tuning_row)

        if score > best_score:
            best_score = score
            best_name = cfg.name

        if args.grid_search:
            print(
                f"score={score:.3f} success={all_row['success_rate']:.3f} "
                f"obst={all_row['obstacle_rate']:.3f} border={all_row['border_rate']:.3f} "
                f"success_cte={all_row['success_mean_abs_cte']:.3f}"
            )
        else:
            detail_rows = [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]
            with open(os.path.join(args.out_dir, f"{cfg.name}_details.json"), "w") as f:
                json.dump(detail_rows, f, indent=2)
            with open(os.path.join(args.out_dir, f"{cfg.name}_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(args.out_dir, f"{cfg.name}_config.json"), "w") as f:
                json.dump(asdict(cfg), f, indent=2)
            write_csv(os.path.join(args.out_dir, f"{cfg.name}_details.csv"), detail_rows)
            write_csv(os.path.join(args.out_dir, f"{cfg.name}_summary.csv"), summary)

            print("\nSummary:")
            for s in summary:
                print(
                    f"{s['group']:>6s}: eps={s['episodes']:3d} "
                    f"success={s['success_rate']:.3f} obst={s['obstacle_rate']:.3f} "
                    f"border={s['border_rate']:.3f} timeout={s['timeout_rate']:.3f} "
                    f"mean|cte|={s['mean_abs_cte']:.3f} success_cte={s['success_mean_abs_cte']:.3f} "
                    f"time_success={s['success_mean_elapsed_time_s']:.1f}s eff_success={s['success_mean_path_efficiency']:.3f}"
                )

    # Grid/tuning summary is saved for both modes.
    all_tuning_rows = sorted(all_tuning_rows, key=lambda r: float(r.get("score", -1e9)), reverse=True)
    write_csv(os.path.join(args.out_dir, "tuning_summary.csv"), all_tuning_rows)
    with open(os.path.join(args.out_dir, "tuning_summary.json"), "w") as f:
        json.dump(all_tuning_rows, f, indent=2)
    with open(os.path.join(args.out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nBest config by score: {best_name}  score={best_score:.3f}")
    print(f"Saved tuning summary: {os.path.join(args.out_dir, 'tuning_summary.csv')}")


if __name__ == "__main__":
    main()
