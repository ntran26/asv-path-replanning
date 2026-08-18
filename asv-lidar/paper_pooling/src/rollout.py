"""Driving a trained policy through one episode and recording what happened.

`run_episode` is the single place that owns the act/step loop; the training
callback and the suite evaluator each summarise the returned traces their own
way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ship import MAX_RUD_ANGLE

FRONT_ARC_DEG = 45.0

# info keys copied verbatim into the per-step traces.
_INFO_SERIES = (
    "rpm", "lam", "r_pf", "r_oa", "r_local", "r_center", "r_border", "r_exist",
    "min_lidar", "min_sector_range", "p10_sector_range", "mean_sector_pen",
    "front_clearance", "block_alpha", "local_target_cte", "true_border_clearance",
)


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def std(values: Sequence[float]) -> float:
    return float(np.std(values)) if len(values) else 0.0


def smallest(values: Sequence[float]) -> float:
    return float(np.min(values)) if len(values) else float("inf")


def largest(values: Sequence[float]) -> float:
    return float(np.max(values)) if len(values) else 0.0


def abs_mean(values: Sequence[float]) -> float:
    return mean([abs(v) for v in values])


def abs_max(values: Sequence[float]) -> float:
    return largest([abs(v) for v in values])


def front_clearance_stats(env) -> Dict[str, float]:
    """Minimum and 10th-percentile raw beam range, forward arc only."""
    ranges = np.array(env.lidar.ranges, dtype=np.float32)
    angles = np.array(env.lidar.angles, dtype=np.float32)
    finite = ranges[np.isfinite(ranges)]

    front = ranges[np.abs(angles) <= FRONT_ARC_DEG]
    front = front[np.isfinite(front)]
    return {
        "min_lidar_all": float(np.min(finite)) if finite.size else float("inf"),
        "p10_front": float(np.percentile(front, 10)) if front.size else float("inf"),
    }


def termination_reason(env, info: Dict[str, Any], truncated: bool, hit_max_steps: bool) -> str:
    if info.get("reached_goal", False):
        return "goal"
    if info.get("timeout", False) or truncated or hit_max_steps:
        return "timeout"
    if info.get("collided", False):
        return "border" if env.hit_border() else "obstacle"
    return "terminated"


@dataclass
class Episode:
    reward: float = 0.0
    steps: int = 0
    reason: str = "terminated"
    d_start: float = 0.0
    d_end: float = 0.0
    collision_steps: int = 0
    last_info: Dict[str, Any] = field(default_factory=dict)
    series: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def success(self) -> int:
        return int(self.reason == "goal")

    @property
    def progress(self) -> float:
        return self.d_start - self.d_end

    def track(self, key: str) -> List[float]:
        return self.series.get(key, [])


def run_episode(model, env, *, deterministic: bool = True, max_steps: int = 2000,
                reset_kwargs: Optional[dict] = None,
                action_filter: Optional[Callable[[np.ndarray, dict], np.ndarray]] = None) -> Episode:
    """Run one episode and return its traces.

    `action_filter(action, obs)` may post-process the policy output, which is
    how the interactive test mode applies its side/path consistency guard.
    """
    obs, _ = env.reset(**(reset_kwargs or {}))

    episode = Episode(d_start=float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y)))
    series: Dict[str, List[float]] = {k: [] for k in _INFO_SERIES}
    series.update({k: [] for k in (
        "speed_mps", "u_body", "v_body", "rudder_deg",
        "cross_track_error", "course_error", "lookahead_course_error",
        "min_lidar_all", "p10_front",
    )})

    done = False
    while episode.steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_filter is not None:
            action = action_filter(action, obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        episode.reward += float(reward)
        episode.steps += 1
        episode.last_info = info

        for key in _INFO_SERIES:
            series[key].append(float(info[key]))
        series["speed_mps"].append(float(env.speed_mps))
        series["u_body"].append(float(env.u_body))
        series["v_body"].append(float(env.v_body))
        series["rudder_deg"].append(float(action[0]) * MAX_RUD_ANGLE)
        series["cross_track_error"].append(float(env.cross_track_error))
        series["course_error"].append(float(env.course_error))
        series["lookahead_course_error"].append(float(env.lookahead_course_error))

        clearance = front_clearance_stats(env)
        series["min_lidar_all"].append(clearance["min_lidar_all"])
        series["p10_front"].append(clearance["p10_front"])

        if info.get("collided", False):
            episode.collision_steps += 1
        if done:
            break

    hit_max_steps = episode.steps >= max_steps and not done
    episode.reason = termination_reason(env, episode.last_info, bool(episode.last_info.get("timeout")), hit_max_steps)
    episode.d_end = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))
    episode.series = series
    return episode
