from __future__ import annotations

import heapq
import math
from typing import Dict as TypingDict

import numpy as np
from gymnasium.spaces import Box, Dict

from asv_lidar import LIDAR_MIN_RANGE, LIDAR_RANGE, LIDAR_SWATH
from rl_env import (
    ASVLidarEnv,
    DHDG_MAX_DPS,
    MAX_IN,
    MIN_IN,
    RPM_MAX,
    RPM_MIN,
    U_MAX,
    UPDATE_RATE,
)
from ship_model import HULL_MARGIN, MAX_RUD_ANGLE, VESSEL_LENGTH, VESSEL_WIDTH

GOAL_DIST_MAX = float(np.hypot(10.0, 25.0))
FRONT_HALF_ANGLE_DEG = 45.0
OBS_SECTORS = 15
SECTOR_PERCENTILE = 10
CENTER_HALF_ANGLE_DEG = 18.0
OPEN_PERCENTILE = 80
OPEN_CLEARANCE_M = 3.0
SECTOR_CLEAR_RECOVERY_M = 0.20
LOOKAHEAD_STEPS = 4
GUIDANCE_CLEARANCE_MARGIN_M = float(0.5 * VESSEL_WIDTH + HULL_MARGIN + 0.05)
GUIDANCE_LOOKAHEAD_M = 0.8
GUIDANCE_BASE_RPM = 9.0
GUIDANCE_TURN_RPM_DROP = 3.0
GUIDANCE_RUDDER_GAIN = 1.30
GUIDANCE_YAW_DAMP = 0.008
GUIDANCE_MIN_RPM = 3.5
GUIDANCE_MAX_RPM = 12.0

REWARD_VARIANTS: TypingDict[str, TypingDict[str, float]] = {
    "baseline": {
        "gamma_e": 0.05,
        "goal_reward": 50.0,
        "exist_penalty": -1.0,
        "collision_penalty": -1000.0,
        "lambda_fixed": 0.50,
        "warn_clearance": 4.5,
        "crit_clearance": 1.5,
        "near_clearance": 1.8,
        "center_gain": 0.0,
        "dir_gain": 0.0,
        "dir_scale": 1.5,
        "near_gain": 0.0,
        "speed_gain": 0.0,
        "lam_clear": 0.50,
        "lam_threat": 0.50,
        "heading_gain": 1.0,
        "progress_gain": 0.0,
        "stall_gain": 0.0,
        "stall_speed": 0.18,
        "wrong_turn_gain": 0.0,
        "turn_commit_gain": 0.0,
    },
    "threat_adaptive": {
        "gamma_e": 0.10,
        "goal_reward": 180.0,
        "exist_penalty": -0.12,
        "collision_penalty": -1200.0,
        "warn_clearance": 4.5,
        "crit_clearance": 1.5,
        "near_clearance": 1.8,
        "center_gain": 2.2,
        "dir_gain": 0.80,
        "dir_scale": 1.4,
        "near_gain": 2.0,
        "speed_gain": 0.90,
        "lam_clear": 0.75,
        "lam_threat": 0.28,
        "heading_gain": 0.10,
        "progress_gain": 0.0,
        "stall_gain": 0.0,
        "stall_speed": 0.18,
        "wrong_turn_gain": 0.0,
        "turn_commit_gain": 0.0,
    },
    "threat_progress": {
        "gamma_e": 0.10,
        "goal_reward": 220.0,
        "exist_penalty": -0.08,
        "collision_penalty": -1400.0,
        "warn_clearance": 4.5,
        "crit_clearance": 1.5,
        "near_clearance": 1.8,
        "center_gain": 2.4,
        "dir_gain": 0.95,
        "dir_scale": 1.35,
        "near_gain": 2.3,
        "speed_gain": 0.95,
        "lam_clear": 0.78,
        "lam_threat": 0.22,
        "heading_gain": 0.18,
        "progress_gain": 2.5,
        "stall_gain": 0.18,
        "stall_speed": 0.22,
        "wrong_turn_gain": 0.0,
        "turn_commit_gain": 0.0,
    },
    "turn_guided": {
        "gamma_e": 0.10,
        "goal_reward": 240.0,
        "exist_penalty": -0.08,
        "collision_penalty": -1500.0,
        "warn_clearance": 4.5,
        "crit_clearance": 1.5,
        "near_clearance": 1.8,
        "center_gain": 2.5,
        "dir_gain": 1.10,
        "dir_scale": 1.25,
        "near_gain": 2.5,
        "speed_gain": 1.00,
        "lam_clear": 0.80,
        "lam_threat": 0.18,
        "heading_gain": 0.18,
        "progress_gain": 2.8,
        "stall_gain": 0.18,
        "stall_speed": 0.22,
        "wrong_turn_gain": 0.55,
        "turn_commit_gain": 0.28,
    },
    "guided_path": {
        "gamma_e": 0.10,
        "goal_reward": 240.0,
        "exist_penalty": -0.06,
        "collision_penalty": -1500.0,
        "warn_clearance": 4.5,
        "crit_clearance": 1.5,
        "near_clearance": 1.8,
        "center_gain": 1.8,
        "dir_gain": 0.60,
        "dir_scale": 1.25,
        "near_gain": 2.2,
        "speed_gain": 0.80,
        "lam_clear": 0.86,
        "lam_threat": 0.28,
        "heading_gain": 0.22,
        "progress_gain": 3.2,
        "stall_gain": 0.16,
        "stall_speed": 0.20,
        "wrong_turn_gain": 0.20,
        "turn_commit_gain": 0.12,
    },
    "teacher_guided": {
        "gamma_e": 0.10,
        "goal_reward": 320.0,
        "exist_penalty": -0.02,
        "collision_penalty": -300.0,
        "warn_clearance": 4.5,
        "crit_clearance": 1.5,
        "near_clearance": 1.8,
        "center_gain": 1.4,
        "dir_gain": 0.50,
        "dir_scale": 1.20,
        "near_gain": 2.0,
        "speed_gain": 0.70,
        "lam_clear": 0.88,
        "lam_threat": 0.35,
        "heading_gain": 0.24,
        "progress_gain": 3.2,
        "stall_gain": 0.12,
        "stall_speed": 0.20,
        "wrong_turn_gain": 0.15,
        "turn_commit_gain": 0.10,
        "ref_rudder_gain": 1.25,
        "ref_throttle_gain": 0.45,
        "ref_base_gain": 0.30,
        "ref_turn_gain": 0.70,
        "ref_threat_gain": 0.55,
    },
}


class ASVRewardSearchEnv(ASVLidarEnv):
    def __init__(
        self,
        render_mode: str | None = "human",
        obs_mode: str = "compact",
        reward_mode: str = "turn_guided",
        reward_cfg: TypingDict[str, float] | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode)
        if obs_mode not in {"baseline", "compact", "teacher_compact"}:
            raise ValueError(f"Unsupported obs_mode: {obs_mode}")
        if reward_mode not in REWARD_VARIANTS:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")

        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.reward_cfg = dict(REWARD_VARIANTS[reward_mode])
        if reward_cfg:
            self.reward_cfg.update(reward_cfg)

        self.lookahead_heading_error = 0.0
        self.goal_dist_norm = 1.0
        self.left_clear_norm = 1.0
        self.center_clear_norm = 1.0
        self.right_clear_norm = 1.0
        self.left_clear_instant_norm = 1.0
        self.center_clear_instant_norm = 1.0
        self.right_clear_instant_norm = 1.0
        self.gap_asymmetry = 0.0
        self.threat_obs = 0.0
        self.turn_pref = 0.0
        self.rudder_state_norm = 0.0
        self.rpm_state_norm = 0.0
        self.ref_heading_error = 0.0
        self.ref_rudder_cmd = 0.0
        self.ref_throttle_cmd = 0.0
        self.sector_obs = np.ones(OBS_SECTORS, dtype=np.float32)
        self.front_clear_m = float(LIDAR_RANGE)
        self.left_clear_m = float(LIDAR_RANGE)
        self.center_clear_m = float(LIDAR_RANGE)
        self.right_clear_m = float(LIDAR_RANGE)
        self.path_progress = 0.0
        self.prev_path_progress = 0.0
        self.prev_distance_to_goal = 0.0
        self._edge_cache: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.train_case_pool: list[int] | None = None
        self.use_guidance_path = self.reward_mode in {"guided_path", "teacher_guided"}
        self.guidance_path = self.path.copy()
        self.guidance_waypoint_idx = 1

        self.observation_space = self._build_observation_space()

    def set_train_case_pool(self, case_pool) -> None:
        self.train_case_pool = None if case_pool is None else [int(case) for case in case_pool]

    def _build_observation_space(self) -> Dict:
        if self.obs_mode == "baseline":
            return Dict(
                {
                    "sectors": Box(low=0.0, high=1.0, shape=(OBS_SECTORS,), dtype=np.float32),
                    "speed": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "dhdg": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "tgt": Box(low=-float(self.map_width), high=float(self.map_width), shape=(1,), dtype=np.float32),
                    "heading_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                }
            )

        if self.obs_mode == "teacher_compact":
            return Dict(
                {
                    "speed": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "dhdg": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "tgt": Box(low=-float(self.map_width), high=float(self.map_width), shape=(1,), dtype=np.float32),
                    "heading_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                    "lookahead_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                    "goal_dist": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "front_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "left_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "center_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "right_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "gap_asymmetry": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "threat": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "turn_pref": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "rudder_state": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "rpm_state": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "ref_heading_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                    "ref_rudder": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "ref_throttle": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                }
            )

        return Dict(
            {
                "sectors": Box(low=0.0, high=1.0, shape=(OBS_SECTORS,), dtype=np.float32),
                "speed": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "dhdg": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "tgt": Box(low=-float(self.map_width), high=float(self.map_width), shape=(1,), dtype=np.float32),
                "heading_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                "lookahead_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                "goal_dist": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "front_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "left_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "center_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "right_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "left_clear_instant": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "center_clear_instant": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "right_clear_instant": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "gap_asymmetry": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "threat": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "turn_pref": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "rudder_state": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "rpm_state": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _get_obs(self):
        scalar_obs = {
            "sectors": self.sector_obs.astype(np.float32),
            "speed": np.array([np.clip(self.speed_mps / max(U_MAX, 1e-6), 0.0, 1.0)], dtype=np.float32),
            "dhdg": np.array([np.clip(self.asv_w / DHDG_MAX_DPS, -1.0, 1.0)], dtype=np.float32),
            "tgt": np.array([self.tgt], dtype=np.float32),
            "heading_error": np.array([self.heading_error], dtype=np.float32),
        }
        if self.obs_mode == "baseline":
            return scalar_obs

        warn = max(float(self.reward_cfg["warn_clearance"]), 1e-6)
        common = {
            "speed": scalar_obs["speed"],
            "dhdg": scalar_obs["dhdg"],
            "tgt": scalar_obs["tgt"],
            "heading_error": scalar_obs["heading_error"],
            "lookahead_error": np.array([self.lookahead_heading_error], dtype=np.float32),
            "goal_dist": np.array([self.goal_dist_norm], dtype=np.float32),
            "front_clear": np.array([np.clip(self.front_clear_m / warn, 0.0, 1.0)], dtype=np.float32),
            "left_clear": np.array([self.left_clear_norm], dtype=np.float32),
            "center_clear": np.array([self.center_clear_norm], dtype=np.float32),
            "right_clear": np.array([self.right_clear_norm], dtype=np.float32),
            "gap_asymmetry": np.array([self.gap_asymmetry], dtype=np.float32),
            "threat": np.array([self.threat_obs], dtype=np.float32),
            "turn_pref": np.array([self.turn_pref], dtype=np.float32),
            "rudder_state": np.array([self.rudder_state_norm], dtype=np.float32),
            "rpm_state": np.array([self.rpm_state_norm], dtype=np.float32),
        }
        if self.obs_mode == "teacher_compact":
            common.update(
                {
                    "ref_heading_error": np.array([self.ref_heading_error], dtype=np.float32),
                    "ref_rudder": np.array([self.ref_rudder_cmd], dtype=np.float32),
                    "ref_throttle": np.array([self.ref_throttle_cmd], dtype=np.float32),
                }
            )
            return common

        scalar_obs.update(common)
        scalar_obs.update(
            {
                "left_clear_instant": np.array([self.left_clear_instant_norm], dtype=np.float32),
                "center_clear_instant": np.array([self.center_clear_instant_norm], dtype=np.float32),
                "right_clear_instant": np.array([self.right_clear_instant_norm], dtype=np.float32),
            }
        )
        return scalar_obs

    def _rebuild_edge_cache(self) -> None:
        edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for poly in self.obstacles:
            for i in range(len(poly)):
                edges.append((poly[i], poly[(i + 1) % len(poly)]))
        for border in self.map_border:
            for i in range(len(border)):
                edges.append((border[i], border[(i + 1) % len(border)]))
        self._edge_cache = edges

    def _polys_intersect_sat(self, poly_a, poly_b) -> bool:
        def project(poly, ax, ay):
            dots = [p[0] * ax + p[1] * ay for p in poly]
            return min(dots), max(dots)

        for poly in (poly_a, poly_b):
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                ax = -(y2 - y1)
                ay = x2 - x1
                min_a, max_a = project(poly_a, ax, ay)
                min_b, max_b = project(poly_b, ax, ay)
                if max_a < min_b or max_b < min_a:
                    return False
        return True

    def _check_collision_geom(self) -> bool:
        hull = self._hull_polygon_world()
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]

        if min(xs) < 0 or max(xs) > self.map_width or min(ys) < 0 or max(ys) > self.map_height:
            return True

        hx0, hx1 = min(xs), max(xs)
        hy0, hy1 = min(ys), max(ys)
        for obs in self.obstacles:
            oxs = [p[0] for p in obs]
            oys = [p[1] for p in obs]
            ox0, ox1 = min(oxs), max(oxs)
            oy0, oy1 = min(oys), max(oys)
            if hx1 < ox0 or ox1 < hx0 or hy1 < oy0 or oy1 < hy0:
                continue
            if self._polys_intersect_sat(hull, obs):
                return True
        return False

    def _pool_lidar_sectors(self) -> np.ndarray:
        ranges = np.asarray(self.lidar.ranges, dtype=np.float32)
        angles = np.asarray(self.lidar.angles, dtype=np.float32)
        sector_edges = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, OBS_SECTORS + 1, dtype=np.float32)
        pooled = np.full(OBS_SECTORS, LIDAR_RANGE, dtype=np.float32)

        for idx in range(OBS_SECTORS):
            lo = sector_edges[idx]
            hi = sector_edges[idx + 1]
            if idx == OBS_SECTORS - 1:
                mask = (angles >= lo) & (angles <= hi)
            else:
                mask = (angles >= lo) & (angles < hi)
            sector_ranges = ranges[mask]
            if sector_ranges.size == 0:
                continue
            pooled[idx] = float(np.percentile(sector_ranges, SECTOR_PERCENTILE))

        return pooled

    def _path_progress_along_track(self, asv_x, asv_y) -> float:
        if self.use_guidance_path:
            return self._polyline_progress(asv_x, asv_y, self.guidance_path)
        path_dx = float(self.goal_x - self.start_x)
        path_dy = float(self.goal_y - self.start_y)
        path_norm = float(np.hypot(path_dx, path_dy))
        if path_norm < 1e-6:
            return 0.0
        rel_x = float(asv_x - self.start_x)
        rel_y = float(asv_y - self.start_y)
        along = (rel_x * path_dx + rel_y * path_dy) / path_norm
        return float(np.clip(along, 0.0, path_norm))

    def _polyline_progress(self, asv_x, asv_y, path: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0

        point = np.array([float(asv_x), float(asv_y)], dtype=np.float32)
        cumulative = 0.0
        best_progress = 0.0
        best_dist = float("inf")
        for idx in range(len(path) - 1):
            p0 = np.asarray(path[idx], dtype=np.float32)
            p1 = np.asarray(path[idx + 1], dtype=np.float32)
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            t = float(np.clip(np.dot(point - p0, seg) / max(seg_len * seg_len, 1e-6), 0.0, 1.0))
            proj = p0 + t * seg
            dist = float(np.linalg.norm(point - proj))
            if dist < best_dist:
                best_dist = dist
                best_progress = cumulative + t * seg_len
            cumulative += seg_len
        return best_progress

    def _point_on_polyline(self, path: np.ndarray, target_progress: float) -> tuple[float, float]:
        if len(path) == 0:
            return float(self.goal_x), float(self.goal_y)
        if len(path) == 1:
            return float(path[0][0]), float(path[0][1])

        remaining = float(max(target_progress, 0.0))
        for idx in range(len(path) - 1):
            p0 = np.asarray(path[idx], dtype=np.float32)
            p1 = np.asarray(path[idx + 1], dtype=np.float32)
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            if remaining <= seg_len:
                frac = remaining / seg_len
                point = p0 + frac * seg
                return float(point[0]), float(point[1])
            remaining -= seg_len
        return float(path[-1][0]), float(path[-1][1])

    def _signed_cross_track_to_path(self, asv_x, asv_y, path: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0

        point = np.array([float(asv_x), float(asv_y)], dtype=np.float32)
        best_cross = 0.0
        best_dist = float("inf")
        for idx in range(len(path) - 1):
            p0 = np.asarray(path[idx], dtype=np.float32)
            p1 = np.asarray(path[idx + 1], dtype=np.float32)
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            t = float(np.clip(np.dot(point - p0, seg) / max(seg_len * seg_len, 1e-6), 0.0, 1.0))
            proj = p0 + t * seg
            dist = float(np.linalg.norm(point - proj))
            if dist < best_dist:
                best_dist = dist
                cross = float(seg[0] * (point[1] - p0[1]) - seg[1] * (point[0] - p0[0]))
                best_cross = cross / seg_len
        return best_cross

    def _guidance_inflated_rects(self) -> list[tuple[float, float, float, float]]:
        margin = GUIDANCE_CLEARANCE_MARGIN_M
        rects: list[tuple[float, float, float, float]] = []
        for obs in self.obstacles:
            xs = [float(p[0]) for p in obs]
            ys = [float(p[1]) for p in obs]
            rects.append(
                (
                    float(min(xs) - margin),
                    float(max(xs) + margin),
                    float(min(ys) - margin),
                    float(max(ys) + margin),
                )
            )
        return rects

    @staticmethod
    def _point_in_rect(point: tuple[float, float], rect: tuple[float, float, float, float], *, strict: bool = True) -> bool:
        x, y = point
        x0, x1, y0, y1 = rect
        if strict:
            return bool(x0 < x < x1 and y0 < y < y1)
        return bool(x0 <= x <= x1 and y0 <= y <= y1)

    @staticmethod
    def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @staticmethod
    def _on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return bool(
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    def _segments_intersect(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
    ) -> bool:
        eps = 1e-9
        o1 = self._orientation(a, b, c)
        o2 = self._orientation(a, b, d)
        o3 = self._orientation(c, d, a)
        o4 = self._orientation(c, d, b)
        if abs(o1) < eps and self._on_segment(a, b, c):
            return True
        if abs(o2) < eps and self._on_segment(a, b, d):
            return True
        if abs(o3) < eps and self._on_segment(c, d, a):
            return True
        if abs(o4) < eps and self._on_segment(c, d, b):
            return True
        return bool((o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0))

    @staticmethod
    def _same_point(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-6) -> bool:
        return bool(abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol)

    def _segment_clear_of_rects(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        rects: list[tuple[float, float, float, float]],
    ) -> bool:
        keep_out = GUIDANCE_CLEARANCE_MARGIN_M
        samples = (a, b, ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5))
        for point in samples:
            x, y = point
            if not (keep_out <= x <= self.map_width - keep_out and keep_out <= y <= self.map_height - keep_out):
                return False

        for rect in rects:
            if self._point_in_rect(a, rect, strict=True) or self._point_in_rect(b, rect, strict=True):
                return False
            x0, x1, y0, y1 = rect
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            for edge_start, edge_end in zip(corners, corners[1:] + corners[:1]):
                if (
                    self._same_point(a, edge_start)
                    or self._same_point(a, edge_end)
                    or self._same_point(b, edge_start)
                    or self._same_point(b, edge_end)
                ):
                    continue
                if self._segments_intersect(a, b, edge_start, edge_end):
                    return False
            if self._point_in_rect(samples[-1], rect, strict=True):
                return False
        return True

    def _build_guidance_path(self) -> np.ndarray:
        start = (float(self.start_x), float(self.start_y))
        goal = (float(self.goal_x), float(self.goal_y))
        rects = self._guidance_inflated_rects()

        nodes = [start, goal]
        for rect in rects:
            x0, x1, y0, y1 = rect
            nodes.extend([(x0, y0), (x0, y1), (x1, y0), (x1, y1)])

        g_score = {0: 0.0}
        f_score = {0: float(np.hypot(goal[0] - start[0], goal[1] - start[1]))}
        came_from: dict[int, int] = {}
        open_heap: list[tuple[float, int]] = [(f_score[0], 0)]
        closed: set[int] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            if current == 1:
                break

            current_node = nodes[current]
            for nxt_idx, nxt_node in enumerate(nodes):
                if nxt_idx == current:
                    continue
                if not self._segment_clear_of_rects(current_node, nxt_node, rects):
                    continue
                tentative = g_score[current] + float(np.hypot(nxt_node[0] - current_node[0], nxt_node[1] - current_node[1]))
                if tentative < g_score.get(nxt_idx, float("inf")):
                    came_from[nxt_idx] = current
                    g_score[nxt_idx] = tentative
                    f_score[nxt_idx] = tentative + float(np.hypot(goal[0] - nxt_node[0], goal[1] - nxt_node[1]))
                    heapq.heappush(open_heap, (f_score[nxt_idx], nxt_idx))

        if 1 not in g_score:
            return self.path.copy()

        path_nodes = [nodes[1]]
        current = 1
        while current in came_from:
            current = came_from[current]
            path_nodes.append(nodes[current])
        path_nodes.reverse()
        return np.asarray(path_nodes, dtype=np.float32)

    def _line_is_clear(self, p0: np.ndarray, p1: np.ndarray, occ: np.ndarray) -> bool:
        delta = p1 - p0
        steps = max(int(np.ceil(np.linalg.norm(delta) * 2.0)), 1)
        for i in range(steps + 1):
            t = i / steps
            pt = p0 + t * delta
            x = int(round(float(pt[0])))
            y = int(round(float(pt[1])))
            if y < 0 or y >= occ.shape[0] or x < 0 or x >= occ.shape[1]:
                return False
            if occ[y, x] != 0:
                return False
        return True

    def _smooth_guidance_cells(self, cells: np.ndarray, occ: np.ndarray) -> np.ndarray:
        if len(cells) <= 2:
            return cells

        simplified = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            next_idx = anchor + 1
            farthest = next_idx
            while next_idx < len(cells) and self._line_is_clear(cells[anchor], cells[next_idx], occ):
                farthest = next_idx
                next_idx += 1
            simplified.append(cells[farthest])
            anchor = farthest

        path = np.asarray(simplified, dtype=np.float32)
        if len(path) <= 2:
            return path

        smoothed = path.copy()
        for idx in range(1, len(path) - 1):
            smoothed[idx] = 0.25 * path[idx - 1] + 0.5 * path[idx] + 0.25 * path[idx + 1]
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        return smoothed

    def _geometry_ranges_for_angles(self, rel_angles_deg: np.ndarray) -> np.ndarray:
        if rel_angles_deg.size == 0:
            return np.zeros(0, dtype=np.float32)

        lidar_offset = VESSEL_LENGTH / 2.0
        sensor_x = float(self.asv_x) + lidar_offset * math.sin(math.radians(float(self.asv_h)))
        sensor_y = float(self.asv_y) + lidar_offset * math.cos(math.radians(float(self.asv_h)))
        geom_ranges = np.full(rel_angles_deg.shape, float(LIDAR_RANGE), dtype=np.float32)
        for idx, rel_angle in enumerate(rel_angles_deg):
            absolute_angle = math.radians(float(self.asv_h + rel_angle))
            end_x = sensor_x + LIDAR_RANGE * math.sin(absolute_angle)
            end_y = sensor_y + LIDAR_RANGE * math.cos(absolute_angle)
            closest = float(LIDAR_RANGE)
            for edge in self._edge_cache:
                intersection = self.lidar.line_intersection((sensor_x, sensor_y), (end_x, end_y), edge[0], edge[1])
                if intersection:
                    dist = float(np.hypot(intersection[0] - sensor_x, intersection[1] - sensor_y))
                    closest = min(closest, dist)
            geom_ranges[idx] = closest
        return geom_ranges

    def _sector_clear_with_memory(self, ranges: np.ndarray, prev_clear_m: float) -> TypingDict[str, float]:
        finite = ranges[np.isfinite(ranges)]
        if finite.size == 0:
            finite = np.array([LIDAR_RANGE], dtype=np.float32)
        valid = finite[finite < float(LIDAR_RANGE) - 1e-6]
        blocked = float(np.percentile(valid, SECTOR_PERCENTILE)) if valid.size else float(LIDAR_RANGE)
        open_clear = float(np.percentile(finite, OPEN_PERCENTILE))
        open_fraction = float(np.mean(finite >= OPEN_CLEARANCE_M))
        no_return_fraction = float(np.mean(finite >= float(LIDAR_RANGE) - 1e-6))
        openness = float(np.clip(max(open_fraction, no_return_fraction), 0.0, 1.0))
        instant = float(np.clip(blocked + openness * max(0.0, open_clear - blocked), 0.0, float(LIDAR_RANGE)))
        smoothed = float(min(instant, float(prev_clear_m) + SECTOR_CLEAR_RECOVERY_M))
        return {
            "blocked_clear_m": blocked,
            "open_clear_m": open_clear,
            "open_fraction": open_fraction,
            "no_return_fraction": no_return_fraction,
            "instant_clear_m": instant,
            "smoothed_clear_m": smoothed,
        }

    def _compute_lidar_sector_features(self) -> TypingDict[str, float]:
        theta_deg = np.asarray(self.lidar.angles, dtype=np.float32)
        lidar_d = np.asarray(self.lidar.ranges, dtype=np.float32)
        front_mask = np.abs(theta_deg) <= FRONT_HALF_ANGLE_DEG
        front_theta = theta_deg[front_mask] if np.any(front_mask) else theta_deg
        front_ranges = lidar_d[front_mask] if np.any(front_mask) else lidar_d
        left_mask = (front_theta >= -FRONT_HALF_ANGLE_DEG) & (front_theta < -CENTER_HALF_ANGLE_DEG)
        center_mask = np.abs(front_theta) <= CENTER_HALF_ANGLE_DEG
        right_mask = (front_theta > CENTER_HALF_ANGLE_DEG) & (front_theta <= FRONT_HALF_ANGLE_DEG)
        left_ranges = front_ranges[left_mask] if np.any(left_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        center_ranges = front_ranges[center_mask] if np.any(center_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        right_ranges = front_ranges[right_mask] if np.any(right_mask) else np.array([LIDAR_RANGE], dtype=np.float32)

        left_stats = self._sector_clear_with_memory(left_ranges, self.left_clear_m)
        center_stats = self._sector_clear_with_memory(center_ranges, self.center_clear_m)
        right_stats = self._sector_clear_with_memory(right_ranges, self.right_clear_m)
        self.left_clear_m = left_stats["smoothed_clear_m"]
        self.center_clear_m = center_stats["smoothed_clear_m"]
        self.right_clear_m = right_stats["smoothed_clear_m"]

        return {
            "left_lidar_clear_m": float(self.left_clear_m),
            "center_lidar_clear_m": float(self.center_clear_m),
            "right_lidar_clear_m": float(self.right_clear_m),
            "left_lidar_instant_m": float(left_stats["instant_clear_m"]),
            "center_lidar_instant_m": float(center_stats["instant_clear_m"]),
            "right_lidar_instant_m": float(right_stats["instant_clear_m"]),
        }

    def _compute_reward_clearance_features(self) -> TypingDict[str, float]:
        theta_deg = np.asarray(self.lidar.angles, dtype=np.float32)
        lidar_d = np.asarray(self.lidar.ranges, dtype=np.float32)
        raw_front_mask = np.abs(theta_deg) <= FRONT_HALF_ANGLE_DEG
        raw_front_ranges = lidar_d[raw_front_mask] if np.any(raw_front_mask) else lidar_d
        raw_front_ranges = raw_front_ranges[np.isfinite(raw_front_ranges)]
        if raw_front_ranges.size == 0:
            raw_front_ranges = np.array([LIDAR_RANGE], dtype=np.float32)
        raw_front_clear = float(np.percentile(raw_front_ranges, SECTOR_PERCENTILE))

        geom_theta_deg = theta_deg[raw_front_mask] if np.any(raw_front_mask) else theta_deg
        geom_ranges = self._geometry_ranges_for_angles(geom_theta_deg)
        if geom_ranges.size == 0:
            geom_theta_deg = np.array([0.0], dtype=np.float32)
            geom_ranges = np.array([LIDAR_RANGE], dtype=np.float32)

        left_mask = (geom_theta_deg >= -FRONT_HALF_ANGLE_DEG) & (geom_theta_deg < -CENTER_HALF_ANGLE_DEG)
        center_mask = np.abs(geom_theta_deg) <= CENTER_HALF_ANGLE_DEG
        right_mask = (geom_theta_deg > CENTER_HALF_ANGLE_DEG) & (geom_theta_deg <= FRONT_HALF_ANGLE_DEG)
        left_ranges = geom_ranges[left_mask] if np.any(left_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        center_ranges = geom_ranges[center_mask] if np.any(center_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        right_ranges = geom_ranges[right_mask] if np.any(right_mask) else np.array([LIDAR_RANGE], dtype=np.float32)

        return {
            "raw_front_clear": raw_front_clear,
            "geom_front_clear": float(np.percentile(geom_ranges, SECTOR_PERCENTILE)),
            "left_clear_m": float(np.percentile(left_ranges, SECTOR_PERCENTILE)),
            "center_clear_m": float(np.percentile(center_ranges, SECTOR_PERCENTILE)),
            "right_clear_m": float(np.percentile(right_ranges, SECTOR_PERCENTILE)),
        }

    def _update_reference_action(self) -> None:
        path_ref = self.guidance_path if self.use_guidance_path else self.path
        if len(path_ref) == 0:
            ref_x = float(self.goal_x)
            ref_y = float(self.goal_y)
        else:
            pos = np.array([self.asv_x, self.asv_y], dtype=np.float32)
            idx = int(np.clip(self.guidance_waypoint_idx, 0, max(len(path_ref) - 1, 0)))
            while idx < len(path_ref) - 1 and float(np.linalg.norm(pos - path_ref[idx])) < GUIDANCE_CLEARANCE_MARGIN_M:
                idx += 1

            target_idx = idx
            for cand_idx in range(idx, len(path_ref)):
                if float(np.linalg.norm(path_ref[cand_idx] - pos)) >= GUIDANCE_LOOKAHEAD_M or cand_idx == len(path_ref) - 1:
                    target_idx = cand_idx
                    break

            self.guidance_waypoint_idx = target_idx
            ref_x = float(path_ref[target_idx][0])
            ref_y = float(path_ref[target_idx][1])

        self.tgt_x = ref_x
        self.tgt_y = ref_y
        self.lookahead_heading_error = float(self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, ref_x, ref_y))
        self.ref_heading_error = float(self.lookahead_heading_error)
        self.ref_rudder_cmd = float(
            np.clip(
                -GUIDANCE_RUDDER_GAIN * (self.ref_heading_error / max(MAX_RUD_ANGLE, 1e-6))
                - GUIDANCE_YAW_DAMP * self.asv_w,
                -1.0,
                1.0,
            )
        )

        desired_rpm = GUIDANCE_BASE_RPM - GUIDANCE_TURN_RPM_DROP * min(abs(self.ref_heading_error) / 45.0, 1.0)
        if self.front_clear_m < 4.0:
            desired_rpm -= 1.0
        if self.front_clear_m < 3.0:
            desired_rpm -= 2.0
        if self.front_clear_m < 2.0:
            desired_rpm -= 2.0
        desired_rpm = float(np.clip(desired_rpm, GUIDANCE_MIN_RPM, GUIDANCE_MAX_RPM))
        if RPM_MAX <= RPM_MIN:
            self.ref_throttle_cmd = -1.0
        else:
            self.ref_throttle_cmd = float(
                np.clip(((desired_rpm - RPM_MIN) / (RPM_MAX - RPM_MIN)) * 2.0 - 1.0, -1.0, 1.0)
            )

    def _update_search_state(self, rudder_cmd: float, rpm_norm: float) -> TypingDict[str, float]:
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        if self.use_guidance_path:
            self.tgt = float(self._signed_cross_track_to_path(self.asv_x, self.asv_y, self.guidance_path))
        else:
            self.tgt = float(self._signed_cross_track_error(self.asv_x, self.asv_y))
        self.heading_error = float(self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y))

        self.path_progress = self._path_progress_along_track(self.asv_x, self.asv_y)

        self.goal_dist_norm = float(np.clip(self.distance_to_goal / GOAL_DIST_MAX, 0.0, 1.0))
        self.sector_obs = np.clip(self._pool_lidar_sectors() / float(LIDAR_RANGE), 0.0, 1.0).astype(np.float32)

        ranges = np.asarray(self.lidar.ranges, dtype=np.float32)
        angles = np.asarray(self.lidar.angles, dtype=np.float32)
        front_mask = np.abs(angles) <= FRONT_HALF_ANGLE_DEG
        front_ranges = ranges[front_mask] if np.any(front_mask) else ranges
        self.front_clear_m = float(np.percentile(front_ranges, SECTOR_PERCENTILE)) if front_ranges.size else float(LIDAR_RANGE)

        sector_features = self._compute_lidar_sector_features()
        warn = max(float(self.reward_cfg["warn_clearance"]), 1e-6)
        self.left_clear_norm = float(np.clip(sector_features["left_lidar_clear_m"] / warn, 0.0, 1.0))
        self.center_clear_norm = float(np.clip(sector_features["center_lidar_clear_m"] / warn, 0.0, 1.0))
        self.right_clear_norm = float(np.clip(sector_features["right_lidar_clear_m"] / warn, 0.0, 1.0))
        self.left_clear_instant_norm = float(np.clip(sector_features["left_lidar_instant_m"] / warn, 0.0, 1.0))
        self.center_clear_instant_norm = float(np.clip(sector_features["center_lidar_instant_m"] / warn, 0.0, 1.0))
        self.right_clear_instant_norm = float(np.clip(sector_features["right_lidar_instant_m"] / warn, 0.0, 1.0))
        self.gap_asymmetry = float(
            np.clip((sector_features["right_lidar_clear_m"] - sector_features["left_lidar_clear_m"]) / warn, -1.0, 1.0)
        )
        threat = (float(self.reward_cfg["warn_clearance"]) - sector_features["center_lidar_clear_m"]) / max(
            float(self.reward_cfg["warn_clearance"]) - float(self.reward_cfg["crit_clearance"]),
            1e-6,
        )
        self.threat_obs = float(np.clip(threat, 0.0, 1.0))
        self.turn_pref = float(np.tanh(self.gap_asymmetry * 2.0))
        self.rudder_state_norm = float(np.clip(rudder_cmd, -1.0, 1.0))
        self.rpm_state_norm = float(np.clip(rpm_norm, 0.0, 1.0))
        self._update_reference_action()
        return sector_features

    def reset(self, seed=None, options=None):
        if self.test_case is None and self.train_case_pool:
            self.test_case = int(np.random.choice(self.train_case_pool))
        obs, info = super().reset(seed=seed, options=options)
        if self.train_case_pool is not None:
            self.test_case = None
        self.guidance_path = self._build_guidance_path() if self.use_guidance_path else self.path.copy()
        self.guidance_waypoint_idx = 1 if len(self.guidance_path) > 1 else 0
        self.left_clear_m = float(LIDAR_RANGE)
        self.center_clear_m = float(LIDAR_RANGE)
        self.right_clear_m = float(LIDAR_RANGE)
        self._rebuild_edge_cache()
        self.prev_distance_to_goal = float(self.distance_to_goal)
        self.prev_path_progress = self._path_progress_along_track(self.asv_x, self.asv_y)
        self._update_search_state(rudder_cmd=0.0, rpm_norm=0.0)
        self.lambda_reward = float(self.reward_cfg.get("lambda_fixed", self.reward_cfg.get("lam_clear", 0.5)))
        return self._get_obs(), info

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], MIN_IN, MAX_IN))
        throttle_cmd = float(np.clip(action[1], MIN_IN, MAX_IN))
        rudder = rudder_cmd * 100.0
        rpm = (throttle_cmd - MIN_IN) * ((RPM_MAX - RPM_MIN) / (MAX_IN - MIN_IN)) + RPM_MIN
        rpm_norm = float(rpm / RPM_MAX) if RPM_MAX > 0 else 0.0

        x_prev = float(self.asv_x)
        y_prev = float(self.asv_y)
        dist_prev = float(self.distance_to_goal)
        progress_prev = float(self.path_progress)

        dx, dy, h, w = self.model.update(rpm, rudder, UPDATE_RATE)
        self.asv_x += float(dx)
        self.asv_y += float(dy)
        self.asv_h = float(h)
        self.asv_w = float(w)
        self.speed_mps = float(np.hypot(self.asv_x - x_prev, self.asv_y - y_prev) / UPDATE_RATE)

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)
        self.asv_path.append((self.asv_x, self.asv_y))

        sector_features = self._update_search_state(rudder_cmd=rudder_cmd, rpm_norm=rpm_norm)
        clearance = self._compute_reward_clearance_features()
        collided = bool(self._check_collision_geom())
        reached_goal = bool(self.distance_to_goal <= (VESSEL_LENGTH / 2.0))

        u_norm = float(np.clip(self.speed_mps / max(U_MAX, 1e-6), 0.0, 1.0))
        r_goal = float(self.reward_cfg["goal_reward"] if reached_goal else 0.0)
        r_exist = float(self.reward_cfg["exist_penalty"])

        if self.reward_mode == "baseline":
            r_heading = float(np.cos(np.radians(self.heading_error)))
            r_pf = float(np.exp(-self.reward_cfg["gamma_e"] * abs(self.tgt)))
            lidar_d = np.asarray(self.lidar.ranges, dtype=np.float32)
            theta_deg = np.asarray(self.lidar.angles, dtype=np.float32)
            weights = 1.0 / (1.0 + np.abs(theta_deg))
            r_oa = float(-np.mean(weights / np.maximum(lidar_d, LIDAR_MIN_RANGE)))
            lam = float(self.reward_cfg["lambda_fixed"])
            threat = center_norm = near_norm = gap_bias = 0.0
            r_dir = r_wrong = r_turn_mag = r_progress = r_stall = r_ref = 0.0
        else:
            if u_norm > 1e-6:
                course_deg = float(np.degrees(np.arctan2(self.asv_x - x_prev, self.asv_y - y_prev)))
            else:
                course_deg = float(self.asv_h)
            if self.use_guidance_path:
                path_course_deg = float(self.asv_h + self.lookahead_heading_error)
            else:
                path_course_deg = float(np.degrees(np.arctan2(self.goal_x - self.start_x, self.goal_y - self.start_y)))
            chi_tilde_deg = float((course_deg - path_course_deg + 180.0) % 360.0 - 180.0)
            r_pf = float(-1.0 + (u_norm * np.cos(np.radians(chi_tilde_deg)) + 1.0) * (np.exp(-self.reward_cfg["gamma_e"] * abs(self.tgt)) + 1.0))

            left_clear = float(clearance["left_clear_m"])
            center_clear = float(clearance["center_clear_m"])
            right_clear = float(clearance["right_clear_m"])
            warn = float(self.reward_cfg["warn_clearance"])
            crit = float(self.reward_cfg["crit_clearance"])
            near_clear = float(self.reward_cfg["near_clearance"])
            center_norm = float(np.clip((warn - center_clear) / max(warn - crit, 1e-6), 0.0, 1.0))
            near_norm = float(np.clip((near_clear - center_clear) / max(near_clear, 1e-6), 0.0, 1.0))
            gap_bias = float(np.tanh((right_clear - left_clear) / max(self.reward_cfg["dir_scale"], 1e-6)))
            turn_cmd = float(-rudder_cmd)
            turn_align = float(gap_bias * turn_cmd)
            r_center = float(-self.reward_cfg["center_gain"] * (center_norm ** 2))
            r_dir = float(self.reward_cfg["dir_gain"] * center_norm * gap_bias * np.tanh(2.0 * turn_cmd))
            r_near = float(-self.reward_cfg["near_gain"] * (near_norm ** 2))
            threat = float(max(center_norm, near_norm))
            r_speed_threat = float(-self.reward_cfg["speed_gain"] * threat * (u_norm ** 2))
            r_wrong = float(-self.reward_cfg["wrong_turn_gain"] * center_norm * max(0.0, -turn_align))
            r_turn_mag = float(self.reward_cfg["turn_commit_gain"] * center_norm * max(0.0, turn_align))
            r_oa = float(r_center + r_dir + r_near + r_speed_threat + r_wrong + r_turn_mag)
            lam = float(self.reward_cfg["lam_clear"] - (self.reward_cfg["lam_clear"] - self.reward_cfg["lam_threat"]) * threat)
            r_heading = float(self.reward_cfg["heading_gain"] * (1.0 - threat) * np.cos(np.radians(self.lookahead_heading_error)))
            progress_delta = float(np.clip(self.path_progress - progress_prev, -0.50, 0.50))
            dist_delta = float(np.clip(dist_prev - self.distance_to_goal, -0.50, 0.50))
            r_progress = float(self.reward_cfg["progress_gain"] * (0.5 * progress_delta + 0.5 * dist_delta))
            r_stall = float(-self.reward_cfg["stall_gain"] * (1.0 - threat) * max(0.0, self.reward_cfg["stall_speed"] - u_norm))
            if self.reward_mode == "teacher_guided":
                ref_turn_weight = float(
                    self.reward_cfg["ref_base_gain"]
                    + self.reward_cfg["ref_turn_gain"] * min(abs(self.ref_heading_error) / 45.0, 1.0)
                    + self.reward_cfg["ref_threat_gain"] * threat
                )
                rudder_match = float(max(0.0, 1.0 - abs(rudder_cmd - self.ref_rudder_cmd)))
                throttle_match = float(max(0.0, 1.0 - abs(throttle_cmd - self.ref_throttle_cmd)))
                r_ref = float(
                    ref_turn_weight
                    * (
                        self.reward_cfg["ref_rudder_gain"] * rudder_match
                        + self.reward_cfg["ref_throttle_gain"] * throttle_match
                    )
                )
            else:
                r_ref = 0.0

        if collided:
            reward = float(self.reward_cfg["collision_penalty"])
        elif reached_goal:
            reward = float(r_goal)
        else:
            reward = float(lam * r_pf + (1.0 - lam) * r_oa + r_heading + r_progress + r_stall + r_ref + r_exist)

        self.reward = reward
        self.lambda_reward = float(lam)
        self.prev_distance_to_goal = float(self.distance_to_goal)
        self.prev_path_progress = float(self.path_progress)

        info = {
            "r_pf": float(r_pf),
            "r_oa": float(r_oa),
            "r_heading": float(r_heading),
            "r_exist": float(r_exist),
            "r_goal": float(r_goal),
            "r_progress": float(r_progress),
            "r_stall": float(r_stall),
            "r_ref": float(r_ref),
            "r_dir": float(r_dir),
            "r_wrong": float(r_wrong),
            "r_turn_mag": float(r_turn_mag),
            "lambda_reward": float(lam),
            "cross_track_error": float(self.tgt),
            "heading_error": float(self.heading_error),
            "lookahead_heading_error": float(self.lookahead_heading_error),
            "speed_mps": float(self.speed_mps),
            "rpm": float(rpm),
            "rudder_deg": float(0.3 * rudder),
            "x": float(self.asv_x),
            "y": float(self.asv_y),
            "heading_deg": float(self.asv_h),
            "dhdg_raw": float(self.asv_w),
            "front_clear": float(clearance["geom_front_clear"]),
            "raw_front_clear": float(clearance["raw_front_clear"]),
            "geom_front_clear": float(clearance["geom_front_clear"]),
            "left_p10": float(clearance["left_clear_m"]),
            "center_p10": float(clearance["center_clear_m"]),
            "right_p10": float(clearance["right_clear_m"]),
            "center_norm": float(center_norm),
            "near_norm": float(near_norm),
            "threat": float(threat),
            "gap_bias": float(gap_bias),
            "goal_dist_norm": float(self.goal_dist_norm),
            "left_clear": float(self.left_clear_norm),
            "center_clear": float(self.center_clear_norm),
            "right_clear": float(self.right_clear_norm),
            "left_clear_instant": float(self.left_clear_instant_norm),
            "center_clear_instant": float(self.center_clear_instant_norm),
            "right_clear_instant": float(self.right_clear_instant_norm),
            "gap_asymmetry": float(self.gap_asymmetry),
            "turn_pref": float(self.turn_pref),
            "rudder_state": float(self.rudder_state_norm),
            "rpm_state": float(self.rpm_state_norm),
            "ref_heading_error": float(self.ref_heading_error),
            "ref_rudder_cmd": float(self.ref_rudder_cmd),
            "ref_throttle_cmd": float(self.ref_throttle_cmd),
            "reward_mode": self.reward_mode,
            "obs_mode": self.obs_mode,
        }
        info.update(sector_features)

        if self.render_mode in self.metadata["render_modes"]:
            self.render()
        return self._get_obs(), reward, bool(collided or reached_goal), False, info


if __name__ == "__main__":
    env = ASVRewardSearchEnv(render_mode="human")
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        obs, rew, term, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
        total_reward += rew
        if term:
            print(f"Elapsed time: {env.elapsed_time:.1f}s, Reward: {total_reward:0.2f}")
            env.close()
            break
