"""Per-episode metric computation and aggregation for the baseline comparison.

Split out of `evaluate.py` so the tuning search can import the scoring pieces
without pulling in the CLI.

Geometry note
-------------
`min_obstacle_clearance` is computed here rather than read from `info`, because
the environment does not compute it.  The nearest thing it exports,
`info["min_lidar_reward"]`, is a beam range from a sensor mounted 0.8625 m
forward of the vessel origin -- neither footprint-based nor a surface distance.
On a spot check it read 0.523 m where the true footprint-to-surface distance was
0.091 m.

What is computed instead is the exact distance between the inflated hull polygon
and each obstacle polygon, zero on intersection.  Both are convex, so the
minimum is attained at a vertex of one against an edge of the other, and the
vertex/edge sweep below is exact.  `env.hull_polygon()` and `env.obstacles` are
already public, so this needs no change to `env.py`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import config as cfg
from env import _polygons_intersect
from ship import MAX_RUD_ANGLE

# |action[0]| at or above this counts as a saturated rudder command.
SATURATION_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _point_segment_distances(points: np.ndarray, seg_a: np.ndarray,
                             seg_b: np.ndarray) -> np.ndarray:
    """Distances from each point to each segment.  points (n,2), segs (m,2)."""
    p = points[:, None, :]                      # (n, 1, 2)
    a = seg_a[None, :, :]                       # (1, m, 2)
    b = seg_b[None, :, :]
    ab = b - a
    denom = np.einsum("ijk,ijk->ij", ab, ab)
    t = np.where(denom > 1e-12,
                 np.einsum("ijk,ijk->ij", p - a, ab) / np.maximum(denom, 1e-12),
                 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, :, None] * ab
    return np.linalg.norm(p - closest, axis=2)


def polygon_distance(poly_a: Sequence, poly_b: Sequence) -> float:
    """Exact minimum distance between two convex polygons; 0.0 if they overlap."""
    if _polygons_intersect(list(poly_a), list(poly_b)):
        return 0.0
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    d1 = _point_segment_distances(a, b, np.roll(b, -1, axis=0))
    d2 = _point_segment_distances(b, a, np.roll(a, -1, axis=0))
    return float(min(d1.min(), d2.min()))


def min_obstacle_clearance(hull: Sequence, obstacles: Sequence) -> float:
    """Smallest hull-to-obstacle surface distance, or NaN when there are none."""
    if not obstacles:
        return float("nan")
    return min(polygon_distance(hull, obs) for obs in obstacles)


def lateral_border_clearance(hull: Sequence, map_width: float) -> float:
    """Distance from the hull to the nearer *side* wall.

    The environment's `true_border_clearance` minimises over all four walls,
    which makes it dominated by the start pose: START_Y = 2.0 minus an inflated
    hull half-length of 1.0125 floors it at ~0.9875 m in every episode before
    the controller acts.  The lateral component is the one that actually
    measures corridor keeping.  See BASELINES_NOTES.md section 9.
    """
    xs = [p[0] for p in hull]
    return float(min(min(xs), map_width - max(xs)))


# ---------------------------------------------------------------------------
# Per-episode accumulator
# ---------------------------------------------------------------------------
class EpisodeRecorder:
    """Accumulates the per-step traces one episode's metrics are built from."""

    def __init__(self, env, record: Dict[str, Any]) -> None:
        self.env = env
        self.record = record
        self.cte: List[float] = []
        self.speed: List[float] = []
        self.rpm: List[float] = []
        self.rudder_cmd: List[float] = []       # action[0], in [-1, 1]
        self.rudder_deg: List[float] = []       # commanded angle
        self.rudder_deg_actual: List[float] = []  # achieved angle, after the servo
        self.course_error: List[float] = []
        self.min_obs_clear = float("inf")
        self.min_border_clear = float("inf")
        self.min_lateral_clear = float("inf")
        self.min_front_clear = float("inf")
        self.reward = 0.0
        self.steps = 0
        self.last_info: Dict[str, Any] = {}

    def observe(self, action: np.ndarray, reward: float, info: Dict[str, Any]) -> None:
        env = self.env
        a0 = float(np.clip(action[0], -1.0, 1.0))

        self.rudder_cmd.append(a0)
        self.rudder_deg.append(a0 * MAX_RUD_ANGLE)
        self.rudder_deg_actual.append(float(env.model.rudder_deg))
        self.cte.append(float(env.cross_track_error))
        self.speed.append(float(env.speed_mps))
        self.rpm.append(float(env.rpm))
        self.course_error.append(float(env.course_error))

        hull = env.hull_polygon()
        if env.obstacles:
            self.min_obs_clear = min(self.min_obs_clear,
                                     min_obstacle_clearance(hull, env.obstacles))
        self.min_border_clear = min(self.min_border_clear,
                                    float(env.true_border_clearance))
        self.min_lateral_clear = min(self.min_lateral_clear,
                                     lateral_border_clearance(hull, env.map_width))
        self.min_front_clear = min(self.min_front_clear,
                                   float(info.get("front_clearance", float("inf"))))

        self.reward += float(reward)
        self.steps += 1
        self.last_info = info

    # -- derived quantities -------------------------------------------------
    def _rudder_rate(self, series: List[float]) -> float:
        """Mean |d(rudder angle)/dt| in deg/s."""
        if len(series) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(series))) / cfg.UPDATE_RATE)

    def finish(self, truncated: bool, hit_max_steps: bool) -> Dict[str, Any]:
        env = self.env
        info = self.last_info
        cte = np.asarray(self.cte, dtype=np.float64)

        reached_goal = bool(info.get("reached_goal", False))
        collided = bool(info.get("collided", False))
        timed_out = bool(info.get("timeout", False)) or truncated or hit_max_steps

        # `_collided` is true for border contact as well, so the two are split
        # the same way `rollout.termination_reason` splits them.
        border_collision = bool(collided and env.hit_border())
        obstacle_collision = bool(collided and not border_collision)

        if reached_goal:
            reason = "goal"
        elif timed_out:
            reason = "timeout"
        elif border_collision:
            reason = "border"
        elif obstacle_collision:
            reason = "obstacle"
        else:
            reason = "terminated"

        ref_len = _polyline_length(self.record.get("path", []))
        act_len = _polyline_length(env.asv_path)
        sat = np.abs(np.asarray(self.rudder_cmd)) >= SATURATION_THRESHOLD

        return {
            "episode_id": int(self.record["case_id"]),
            "obstacle_count": int(self.record.get(
                "obstacle_count", len(self.record.get("obstacles", [])))),
            "group": str(self.record.get("group", "")),
            "seed": int(self.record.get("seed", 0)),

            "success": int(reason == "goal"),
            "obstacle_collision": int(obstacle_collision),
            "border_collision": int(border_collision),
            "timeout": int(reason == "timeout"),
            "term_reason": reason,

            # Cross-track error.  `mean_cte` is the mean *absolute* error, which
            # is what the existing suite evaluator reports as `mean_abs_cte`;
            # the signed mean is kept alongside it because it shows side bias.
            "rms_cte": float(np.sqrt(np.mean(cte ** 2))) if cte.size else float("nan"),
            "mean_cte": float(np.mean(np.abs(cte))) if cte.size else float("nan"),
            "mean_signed_cte": float(np.mean(cte)) if cte.size else float("nan"),
            "max_cte": float(np.max(np.abs(cte))) if cte.size else float("nan"),
            "std_cte": float(np.std(cte)) if cte.size else float("nan"),

            "min_obstacle_clearance": (float(self.min_obs_clear)
                                       if np.isfinite(self.min_obs_clear) else float("nan")),
            "min_border_clearance": float(self.min_border_clear),
            "min_lateral_border_clearance": float(self.min_lateral_clear),
            "min_front_clearance": (float(self.min_front_clear)
                                    if np.isfinite(self.min_front_clear) else float("nan")),

            # Stated in both units, per the brief's request.
            "path_completion_steps": int(self.steps),
            "path_completion_time_s": float(self.steps * cfg.UPDATE_RATE),

            "mean_speed": float(np.mean(self.speed)) if self.speed else 0.0,
            "mean_rpm": float(np.mean(self.rpm)) if self.rpm else 0.0,

            # Integral of the squared normalised rudder command over the episode.
            "control_effort": float(np.sum(np.square(self.rudder_cmd)) * cfg.UPDATE_RATE),
            "control_effort_deg2s": float(
                np.sum(np.square(self.rudder_deg)) * cfg.UPDATE_RATE),

            # Commanded rate is the actuator demand; achieved rate is what the
            # servo delivered after its 20 deg/s limit.  Both are reported
            # because the reviewer comment is about actuator behaviour.
            "mean_abs_rudder_rate": self._rudder_rate(self.rudder_deg),
            "mean_abs_rudder_rate_achieved": self._rudder_rate(self.rudder_deg_actual),
            "mean_abs_rudder_deg": float(np.mean(np.abs(self.rudder_deg))) if self.rudder_deg else 0.0,
            "rudder_saturation_fraction": float(np.mean(sat)) if sat.size else 0.0,

            "mean_abs_course_error": float(np.mean(np.abs(self.course_error))) if self.course_error else 0.0,
            "ep_reward": float(self.reward),
            "reference_path_length": ref_len,
            "actual_path_length": act_len,
            "path_efficiency": act_len / ref_len if ref_len > 1e-6 else float("nan"),
            "d_end": float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y)),
        }


def _polyline_length(points) -> float:
    if points is None or len(points) < 2:
        return 0.0
    p = np.asarray(points, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50 % of the data."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if v.size < 4:
        return float(np.mean(v))
    lo, hi = np.percentile(v, [25.0, 75.0])
    middle = v[(v >= lo) & (v <= hi)]
    return float(np.mean(middle)) if middle.size else float(np.mean(v))


def stratified_bootstrap_ci(values: np.ndarray, strata: np.ndarray,
                            statistic=np.mean, n_boot: int = 10_000,
                            alpha: float = 0.05,
                            rng: Optional[np.random.Generator] = None):
    """Percentile bootstrap CI, resampling within strata.

    Layout difficulty varies systematically with obstacle count, so resampling
    the pooled set would let a draw over- or under-represent a group and inflate
    the interval.  Resampling within each obstacle-count group preserves the
    design of the evaluation set.
    """
    rng = np.random.default_rng(12345) if rng is None else rng
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(strata)
    keep = np.isfinite(v)
    v, s = v[keep], s[keep]
    if v.size == 0:
        return float("nan"), float("nan")

    groups = [np.flatnonzero(s == g) for g in np.unique(s)]
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picks = np.concatenate([
            g[rng.integers(0, g.size, g.size)] for g in groups if g.size
        ])
        stats[b] = statistic(v[picks])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# Metrics that get the full mean/median/IQM/CI treatment in the summary.
HEADLINE_METRICS = (
    "success", "obstacle_collision", "border_collision", "timeout",
    "rms_cte", "mean_cte", "max_cte",
    "min_obstacle_clearance", "min_border_clearance", "min_lateral_border_clearance",
    "path_completion_steps", "path_completion_time_s", "mean_speed",
    "control_effort", "mean_abs_rudder_rate", "rudder_saturation_fraction",
    "path_efficiency",
)


def summarise(rows: List[Dict[str, Any]], *, method: str, deterministic: bool,
              n_boot: int = 10_000, seed: int = 12345) -> Dict[str, Any]:
    """Per-method summary: mean, median, IQM and stratified bootstrap 95 % CIs."""
    rng = np.random.default_rng(seed)
    strata = np.asarray([r["obstacle_count"] for r in rows])

    out: Dict[str, Any] = {
        "method": method,
        "n_episodes": len(rows),
        "deterministic": bool(deterministic),
        "bootstrap_resamples": int(n_boot),
        "bootstrap_stratified_by": "obstacle_count",
        "metrics": {},
    }

    for key in HEADLINE_METRICS:
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        entry: Dict[str, Any] = {
            "n_finite": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "iqm": iqm(vals),
            "std": float(np.std(finite)) if finite.size else float("nan"),
        }
        # A deterministic controller has no sampling variability to bootstrap
        # over; reporting a CI for it would imply a spread that does not exist.
        if not deterministic and finite.size:
            lo, hi = stratified_bootstrap_ci(vals, strata, np.mean, n_boot, rng=rng)
            entry["ci95_mean"] = [lo, hi]
            lo, hi = stratified_bootstrap_ci(vals, strata, iqm, n_boot, rng=rng)
            entry["ci95_iqm"] = [lo, hi]
        out["metrics"][key] = entry

    by_group: Dict[str, Any] = {}
    for g in sorted(set(strata.tolist())):
        subset = [r for r in rows if r["obstacle_count"] == g]
        by_group[f"obs_{g}"] = {
            "n": len(subset),
            "success_rate": float(np.mean([r["success"] for r in subset])),
            "obstacle_collision_rate": float(np.mean([r["obstacle_collision"] for r in subset])),
            "border_collision_rate": float(np.mean([r["border_collision"] for r in subset])),
            "timeout_rate": float(np.mean([r["timeout"] for r in subset])),
            "mean_rms_cte": float(np.nanmean([r["rms_cte"] for r in subset])),
        }
    out["by_obstacle_count"] = by_group
    return out
