from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium.spaces import Box, Dict as DictSpace

from asv_lidar import Lidar, LIDAR_RANGE, LIDAR_SECTORS, LIDAR_SWATH
from images import BOAT_ICON
from ship_model import (
    HULL_FORWARD_SHIFT,
    HULL_MARGIN,
    THRUST_COEF,
    DRAG_COEF,
    VESSEL_LENGTH,
    VESSEL_WIDTH,
    ShipModel,
)
from test_run import TestCase

RENDER_SCALE = 25
TEST_CASE = None

# ---------------------------------------------------------------------------
# Focused local-planner training setup
# ---------------------------------------------------------------------------
UPDATE_RATE = 0.1
RENDER_FPS = 10
MAP_WIDTH = 10
MAP_HEIGHT = 25
MAX_OBS = 6

PATH_MODE = "straight"      # keep the nominal global/reference path straight
CURVE_PROB = 0.0
LOOKAHEAD_FRACTION = 0.25

# Fixed lambda for this practical local-planner run.
LAMBDA_MIN = 1e-4
LAMBDA_MAX = 1.0
DEFAULT_EVAL_LAMBDA = 0.7

# Reward parameters
GAMMA_E = 0.15
GAMMA_THETA = 4.0
GAMMA_X = 0.005
EPSILON_X = 1.0
R_COLLISION = -1000.0
R_TIMEOUT = -1000.0
R_GOAL = 50.0
R_EXIST = -0.35

# Fixed-speed steering task. The action space remains 2D for compatibility,
# but throttle is ignored while FIXED_RPM=True.
RPM_MIN = 0
RPM_MAX = 24
CRUISE_RPM = 12.0
FIXED_RPM = True
U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * RPM_MAX)
MAX_IN = 1.0
MIN_IN = -1.0

MAX_EPISODE_STEPS = 700

# Obstacle curriculum: one obstacle near the path. Most episodes are slightly
# offset left/right so the policy learns a pass; some are exactly centered.
OBSTACLE_SIZE = 1.0
OBSTACLE_PATH_START_FRAC = 0.25
OBSTACLE_PATH_END_FRAC = 0.70
OBSTACLE_CENTER_PROB = 0.30
OBSTACLE_LATERAL_OFFSET_MIN = 0.25
OBSTACLE_LATERAL_OFFSET_MAX = 0.95

OBSTACLE_MODE = "random_multi"
# OBSTACLE_MODE = "single_near_path"

# Local lidar-based bypass cue. This is not global planning: it uses only the
# lidar sector ranges to choose the clearer side when the path ahead is blocked.
BLOCK_D_SAFE = 5.0
BLOCK_D_CRIT = 2.0
BLOCK_FRONT_DEG = 25.0
SIDE_ARC_MIN_DEG = 15.0
SIDE_ARC_MAX_DEG = 100.0
SIDE_CLEAR_TIE = 0.25
BYPASS_CTE = 0.7
K_LOCAL_TARGET = 1.5
K_CENTER_BLOCK = 1.0
K_BORDER = 1.5


class ASVLidarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        map_width: float = MAP_WIDTH,
        map_height: float = MAP_HEIGHT,
        max_obs: int = MAX_OBS,
        path_mode: str = PATH_MODE,
        curve_prob: float = CURVE_PROB,
        lookahead_fraction: float = LOOKAHEAD_FRACTION,
        lambda_override: Optional[float] = None,
        test_case: Optional[int] = TEST_CASE,
        record_video: bool = False,
    ) -> None:
        super().__init__()
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.max_obs = int(max_obs)
        self.path_mode = str(path_mode)
        self.curve_prob = float(curve_prob)
        self.lookahead_fraction = float(lookahead_fraction)
        self.lambda_override = lambda_override
        self.test_case = test_case
        self.record_video = bool(record_video)

        pygame.init()
        self.render_mode = render_mode
        self.render_scale = float(RENDER_SCALE)
        self.window_size = (int(round(self.map_width * self.render_scale)), int(round(self.map_height * self.render_scale)))
        self.world_size = (self.map_width, self.map_height)
        self.obstacle_mode = OBSTACLE_MODE

        self.display = None
        self.surface = None
        self.status = None
        self.icon = None
        self.icon_scaled = None
        self._icon_scaled_size = None
        self.fps_clock = pygame.time.Clock()
        if render_mode in self.metadata["render_modes"]:
            self.surface = pygame.Surface(self.window_size)
            self.status = pygame.freetype.SysFont(pygame.font.get_default_font(), size=10)

        self.video_writer = None
        self.frame_size = self.window_size
        self.video_fps = RENDER_FPS

        self.model = ShipModel()
        self.lidar = Lidar()
        self.scenario = TestCase()

        self.elapsed_time = 0.0
        self.step_count = 0
        self.asv_x = 0.0
        self.asv_y = 0.0
        self.asv_h = 0.0
        self.asv_w = 0.0
        self.speed_mps = 0.0
        self.u_body = 0.0
        self.v_body = 0.0

        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.distance_to_goal = 0.0

        self.path = np.zeros((2, 2), dtype=np.float32)
        self.path_s = np.zeros(2, dtype=np.float32)
        self.lookahead_distance = 1.0
        self.path_mode_used = "straight"

        self.cross_track_error = 0.0
        self.course_error = 0.0
        self.lookahead_course_error = 0.0
        self.closest_idx = 0
        self.lookahead_idx = 0
        self.tgt_x = 0.0
        self.tgt_y = 0.0
        self.lookahead_x = 0.0
        self.lookahead_y = 0.0

        self.front_clearance = LIDAR_RANGE
        self.left_clearance = LIDAR_RANGE
        self.right_clearance = LIDAR_RANGE
        self.block_alpha = 0.0
        self.local_target_cte = 0.0

        self.asv_path: List[Tuple[float, float]] = []
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.current_lambda = DEFAULT_EVAL_LAMBDA
        self.current_log10_lambda = float(np.log10(DEFAULT_EVAL_LAMBDA))

        self.observation_space = DictSpace({
            "lidar": Box(low=0.0, high=1.0, shape=(LIDAR_SECTORS,), dtype=np.float32),
            "u": Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32),
            "v": Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32),
            "yaw_rate": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            "cross_track_error": Box(low=-max(self.map_width, self.map_height), high=max(self.map_width, self.map_height), shape=(1,), dtype=np.float32),
            "course_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            "lookahead_course_error": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            "front_clearance": Box(low=0.0, high=LIDAR_RANGE, shape=(1,), dtype=np.float32),
            "side_clearance_diff": Box(low=-LIDAR_RANGE, high=LIDAR_RANGE, shape=(1,), dtype=np.float32),
            "local_target_cte": Box(low=-max(self.map_width, self.map_height), high=max(self.map_width, self.map_height), shape=(1,), dtype=np.float32),
            "log10_lambda": Box(low=-4.0, high=0.0, shape=(1,), dtype=np.float32),
        })

        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.map_border = [
            [(0.0, 0.0), (0.0, self.map_height)],
            [(0.0, self.map_height), (self.map_width, self.map_height)],
            [(self.map_width, self.map_height), (self.map_width, 0.0)],
            [(self.map_width, 0.0), (0.0, 0.0)],
        ]

    @staticmethod
    def _wrap180(a: float) -> float:
        return (float(a) + 180.0) % 360.0 - 180.0

    def _canonical_scale(self) -> Tuple[float, float]:
        return self.map_width / 10.0, self.map_height / 25.0

    def _scale_case_position(self, x: float, y: float) -> Tuple[float, float]:
        sx, sy = self._canonical_scale()
        return x * sx, y * sy

    def _scale_case_obstacles(self, obstacles):
        sx, sy = self._canonical_scale()
        return [[(float(px) * sx, float(py) * sy) for px, py in obs] for obs in obstacles]

    def _sample_lambda(self) -> None:
        lam = DEFAULT_EVAL_LAMBDA if self.lambda_override is None else float(self.lambda_override)
        self.current_lambda = float(np.clip(lam, LAMBDA_MIN, LAMBDA_MAX))
        self.current_log10_lambda = float(np.log10(self.current_lambda))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "lidar": self.lidar.sector_closeness.astype(np.float32),
            "u": np.array([self.u_body], dtype=np.float32),
            "v": np.array([self.v_body], dtype=np.float32),
            "yaw_rate": np.array([self.asv_w], dtype=np.float32),
            "cross_track_error": np.array([self.cross_track_error], dtype=np.float32),
            "course_error": np.array([self.course_error], dtype=np.float32),
            "lookahead_course_error": np.array([self.lookahead_course_error], dtype=np.float32),
            "front_clearance": np.array([self.front_clearance], dtype=np.float32),
            "side_clearance_diff": np.array([self.right_clearance - self.left_clearance], dtype=np.float32),
            "local_target_cte": np.array([self.local_target_cte], dtype=np.float32),
            "log10_lambda": np.array([self.current_log10_lambda], dtype=np.float32),
        }

    def _hull_polygon_world(self):
        L = VESSEL_LENGTH + 2.0 * HULL_MARGIN
        W = VESSEL_WIDTH + 2.0 * HULL_MARGIN
        shift = HULL_FORWARD_SHIFT
        half_L = 0.5 * L
        half_W = 0.5 * W
        h = math.radians(float(self.asv_h))
        sin_h = math.sin(h)
        cos_h = math.cos(h)
        local = [
            (+half_L + shift, +half_W),
            (+half_L + shift, -half_W),
            (-half_L + shift, -half_W),
            (-half_L + shift, +half_W),
        ]
        poly = []
        for x_forward, y_left in local:
            x = self.asv_x + x_forward * sin_h - y_left * cos_h
            y = self.asv_y + x_forward * cos_h + y_left * sin_h
            poly.append((x, y))
        return poly

    def _polys_intersect_sat(self, polyA, polyB) -> bool:
        def project(poly, ax, ay):
            dots = [p[0] * ax + p[1] * ay for p in poly]
            return min(dots), max(dots)
        for poly in (polyA, polyB):
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                ax = -(y2 - y1)
                ay = x2 - x1
                minA, maxA = project(polyA, ax, ay)
                minB, maxB = project(polyB, ax, ay)
                if maxA < minB or maxB < minA:
                    return False
        return True

    def _check_collision_geom(self) -> bool:
        hull = self._hull_polygon_world()
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        if min(xs) < 0.0 or max(xs) > self.map_width or min(ys) < 0.0 or max(ys) > self.map_height:
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

    def _check_border_collision_only(self) -> bool:
        hull = self._hull_polygon_world()
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        return min(xs) < 0.0 or max(xs) > self.map_width or min(ys) < 0.0 or max(ys) > self.map_height

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------
    def _start_goal_random(self) -> Tuple[float, float, float, float]:
        # Focused curriculum: mostly vertical path in the 10x25 task area.
        margin_x = max(2.0, 0.25 * self.map_width)
        x = float(np.random.uniform(margin_x, self.map_width - margin_x))
        return x, 2.0, x, self.map_height - 3.0

    def _choose_path_mode(self) -> str:
        if self.path_mode == "mixed":
            return "curve" if np.random.rand() < self.curve_prob else "straight"
        return self.path_mode

    def _generate_straight_path(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> np.ndarray:
        path_length = max(40, int(np.hypot(goal_x - start_x, goal_y - start_y) * 5.0))
        path_x = np.linspace(start_x, goal_x, path_length, dtype=np.float32)
        path_y = np.linspace(start_y, goal_y, path_length, dtype=np.float32)
        return np.column_stack((path_x, path_y)).astype(np.float32)

    def _generate_curve_path(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> np.ndarray:
        # Kept for later. The local-planner default uses straight paths.
        start = np.array([start_x, start_y], dtype=np.float32)
        goal = np.array([goal_x, goal_y], dtype=np.float32)
        vec = goal - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return self._generate_straight_path(start_x, start_y, goal_x, goal_y)
        tangent = vec / length
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        mid = 0.5 * (start + goal)
        offset = float(np.random.uniform(-0.18 * self.map_width, 0.18 * self.map_width))
        control = mid + offset * normal
        control[0] = float(np.clip(control[0], 1.5, self.map_width - 1.5))
        control[1] = float(np.clip(control[1], 1.5, self.map_height - 1.5))
        n = max(60, int(length * 5.0))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        pts = ((1 - t)[:, None] ** 2) * start[None, :] + 2 * (1 - t)[:, None] * t[:, None] * control[None, :] + (t[:, None] ** 2) * goal[None, :]
        return pts.astype(np.float32)

    def _generate_path(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> np.ndarray:
        self.path_mode_used = self._choose_path_mode()
        path = self._generate_curve_path(start_x, start_y, goal_x, goal_y) if self.path_mode_used == "curve" else self._generate_straight_path(start_x, start_y, goal_x, goal_y)
        self.path = path.astype(np.float32)
        diffs = np.diff(self.path, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self.path_s = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)
        total_length = float(self.path_s[-1]) if len(self.path_s) > 0 else 1.0
        self.lookahead_distance = max(2.0, self.lookahead_fraction * total_length)
        return self.path

    def _path_tangent(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, len(self.path) - 1))
        if len(self.path) < 2:
            return np.array([0.0, 1.0], dtype=np.float32)
        if idx == 0:
            vec = self.path[1] - self.path[0]
        elif idx == len(self.path) - 1:
            vec = self.path[-1] - self.path[-2]
        else:
            vec = self.path[idx + 1] - self.path[idx - 1]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return np.array([0.0, 1.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def _bearing_deg(self, from_xy: np.ndarray, to_xy: np.ndarray) -> float:
        dx = float(to_xy[0] - from_xy[0])
        dy = float(to_xy[1] - from_xy[1])
        return float(math.degrees(math.atan2(dx, dy)))

    def _update_path_relative_states(self, course_deg: float) -> None:
        asv_pos = np.array([self.asv_x, self.asv_y], dtype=np.float32)
        d = np.linalg.norm(self.path - asv_pos, axis=1)
        self.closest_idx = int(np.argmin(d))
        cte_abs = float(d[self.closest_idx])
        self.tgt_x, self.tgt_y = map(float, self.path[self.closest_idx])
        tangent = self._path_tangent(self.closest_idx)
        closest_pt = self.path[self.closest_idx]
        rel = asv_pos - closest_pt
        cross_z = float(tangent[0] * rel[1] - tangent[1] * rel[0])
        sign = 1.0 if cross_z > 0.0 else (-1.0 if cross_z < 0.0 else 0.0)
        self.cross_track_error = sign * cte_abs
        path_course_deg = float(math.degrees(math.atan2(float(tangent[0]), float(tangent[1]))))
        self.course_error = self._wrap180(path_course_deg - course_deg)
        s_here = float(self.path_s[self.closest_idx]) if len(self.path_s) > 0 else 0.0
        s_target = min(float(self.path_s[-1]), s_here + self.lookahead_distance)
        self.lookahead_idx = int(np.searchsorted(self.path_s, s_target, side="left"))
        self.lookahead_idx = int(np.clip(self.lookahead_idx, 0, len(self.path) - 1))
        self.lookahead_x, self.lookahead_y = map(float, self.path[self.lookahead_idx])
        lookahead_pt = self.path[self.lookahead_idx]
        self.lookahead_course_error = self._wrap180(self._bearing_deg(asv_pos, lookahead_pt) - course_deg)

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    def _make_box(self, cx: float, cy: float, size: float) -> List[Tuple[float, float]]:
        half = 0.5 * float(size)
        return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]

    def _generate_single_near_path_obstacle(self) -> List[List[Tuple[float, float]]]:
        if len(self.path) < 2:
            return []
        s_total = float(self.path_s[-1]) if len(self.path_s) else 0.0
        if s_total <= 1e-6:
            return []
        s_min = OBSTACLE_PATH_START_FRAC * s_total
        s_max = OBSTACLE_PATH_END_FRAC * s_total
        feasible = np.where((self.path_s >= s_min) & (self.path_s <= s_max))[0]
        if feasible.size == 0:
            return []
        for _ in range(100):
            idx = int(np.random.choice(feasible))
            center = self.path[idx].astype(np.float32)
            tangent = self._path_tangent(idx)
            normal_left = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            if np.random.rand() < OBSTACLE_CENTER_PROB:
                lateral = 0.0
            else:
                side = -1.0 if np.random.rand() < 0.5 else 1.0
                lateral = side * float(np.random.uniform(OBSTACLE_LATERAL_OFFSET_MIN, OBSTACLE_LATERAL_OFFSET_MAX))
            center = center + lateral * normal_left
            cx = float(center[0])
            cy = float(center[1])
            half = 0.5 * OBSTACLE_SIZE
            margin = half + 0.25
            if margin <= cx <= self.map_width - margin and margin <= cy <= self.map_height - margin:
                if np.hypot(cx - self.start_x, cy - self.start_y) > 3.0 and np.hypot(cx - self.goal_x, cy - self.goal_y) > 3.0:
                    return [self._make_box(cx, cy, OBSTACLE_SIZE)]
        return []

    def _generate_obstacles(self, num_obs: int, test_case: Optional[int] = None):
        if test_case is not None:
            raw = self.scenario.obstacles(test_case=test_case)
            return self._scale_case_obstacles(raw)

        if self.obstacle_mode == "single_near_path":
            return self._generate_single_near_path_obstacle()

        # random multiple obstacles around the map / around the path like before
        obstacles: List[List[Tuple[float, float]]] = []
        if num_obs <= 0:
            return obstacles

        path_len = len(self.path)
        start_margin = int(0.15 * path_len)
        end_margin = int(0.85 * path_len)
        min_center_dist = 1.6
        max_tries = 200
        tries = 0

        while len(obstacles) < num_obs and tries < max_tries:
            tries += 1

            idx = int(np.random.randint(max(1, start_margin), max(2, end_margin)))
            center = self.path[idx].astype(np.float32)

            tangent = self._path_tangent(idx)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            lateral_sigma = 0.18 * self.map_width
            lateral = float(np.random.normal(loc=0.0, scale=lateral_sigma))
            center = center + lateral * normal

            w = float(np.random.uniform(0.8, 1.2))
            h = float(np.random.uniform(0.8, 1.2))

            x0 = float(center[0] - 0.5 * w)
            x1 = float(center[0] + 0.5 * w)
            y0 = float(center[1] - 0.5 * h)
            y1 = float(center[1] + 0.5 * h)

            margin = 1.0
            if x0 < margin or x1 > self.map_width - margin or y0 < margin or y1 > self.map_height - margin:
                continue

            c = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            start = np.array([self.start_x, self.start_y], dtype=np.float32)
            goal = np.array([self.goal_x, self.goal_y], dtype=np.float32)

            if float(np.linalg.norm(c - start)) < 2.0 or float(np.linalg.norm(c - goal)) < 2.0:
                continue

            too_close = False
            for obs in obstacles:
                oc = np.mean(np.array(obs, dtype=np.float32), axis=0)
                if float(np.linalg.norm(c - oc)) < min_center_dist:
                    too_close = True
                    break
            if too_close:
                continue

            obstacles.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

        return obstacles
    


    # ------------------------------------------------------------------
    # Local planner features from LiDAR only
    # ------------------------------------------------------------------
    def _update_local_planner_features(self) -> None:
        sector_d = self.lidar.sector_ranges.astype(np.float32)
        sector_angles = self.lidar.sector_angles.astype(np.float32)
        front_mask = np.abs(sector_angles) <= BLOCK_FRONT_DEG
        left_mask = (sector_angles <= -SIDE_ARC_MIN_DEG) & (sector_angles >= -SIDE_ARC_MAX_DEG)
        right_mask = (sector_angles >= SIDE_ARC_MIN_DEG) & (sector_angles <= SIDE_ARC_MAX_DEG)

        def pctl(mask, p=20.0):
            vals = sector_d[mask]
            return float(np.percentile(vals, p)) if vals.size else float(LIDAR_RANGE)

        self.front_clearance = pctl(front_mask, 10.0)
        self.left_clearance = pctl(left_mask, 20.0)
        self.right_clearance = pctl(right_mask, 20.0)
        self.block_alpha = float(np.clip((BLOCK_D_SAFE - self.front_clearance) / (BLOCK_D_SAFE - BLOCK_D_CRIT), 0.0, 1.0))

        if self.block_alpha <= 1e-6 or self.front_clearance > BLOCK_D_SAFE:
            self.local_target_cte = 0.0
            return

        clear_diff = self.right_clearance - self.left_clearance

        # If both sides are nearly equal, only apply the default side bias when the path
        # is clearly blocked. Otherwise, stay centered.
        if abs(clear_diff) < SIDE_CLEAR_TIE:
            if self.block_alpha < 0.5:
                self.local_target_cte = 0.0
                return
            side_cte_sign = -1.0   # default starboard/right only when truly blocked
        elif clear_diff > 0.0:
            side_cte_sign = -1.0
        else:
            side_cte_sign = +1.0

        # make the requested bypass grow more gently
        self.local_target_cte = float(side_cte_sign * BYPASS_CTE * (self.block_alpha ** 1.5))

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.elapsed_time = 0.0
        self.asv_h = 0.0
        self.asv_w = 0.0
        self.speed_mps = 0.0
        self.u_body = 0.0
        self.v_body = 0.0
        self.cross_track_error = 0.0
        self.course_error = 0.0
        self.lookahead_course_error = 0.0
        self.front_clearance = LIDAR_RANGE
        self.left_clearance = LIDAR_RANGE
        self.right_clearance = LIDAR_RANGE
        self.block_alpha = 0.0
        self.local_target_cte = 0.0
        self.model = ShipModel()
        self.lidar.reset()
        self._sample_lambda()

        if self.test_case is None:
            self.start_x, self.start_y, self.goal_x, self.goal_y = self._start_goal_random()
        else:
            sx, sy, gx, gy = self.scenario.position(test_case=self.test_case)
            self.start_x, self.start_y = self._scale_case_position(float(sx), float(sy))
            self.goal_x, self.goal_y = self._scale_case_position(float(gx), float(gy))

        self.asv_x = float(self.start_x)
        self.asv_y = float(self.start_y)
        self.path = self._generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        if self.test_case is None:
            if self.obstacle_mode == "single_near_path":
                num_obs = 1
            else:
                num_obs = int(np.random.randint(0, self.max_obs + 1))
        else:
            num_obs = 0

        self.obstacles = self._generate_obstacles(num_obs, self.test_case)

        # self.obstacles = self._generate_obstacles(1, self.test_case)
        self.asv_path = [(self.asv_x, self.asv_y)]
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=None)
        self._update_path_relative_states(course_deg=self.asv_h)
        self._update_local_planner_features()
        if self.render_mode in self.metadata["render_modes"]:
            self.render()
        return self._get_obs(), {}

    def check_done(self) -> bool:
        if self._check_collision_geom():
            return True
        if self.distance_to_goal <= (VESSEL_LENGTH * 0.5):
            return True
        return False

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], MIN_IN, MAX_IN))
        throttle_cmd = float(np.clip(action[1], MIN_IN, MAX_IN))
        rudder = rudder_cmd * 100.0
        rpm = CRUISE_RPM if FIXED_RPM else (throttle_cmd - MIN_IN) * ((RPM_MAX - RPM_MIN) / (MAX_IN - MIN_IN)) + RPM_MIN

        x_prev = float(self.asv_x)
        y_prev = float(self.asv_y)
        dx, dy, h, w = self.model.update(rpm, rudder, UPDATE_RATE)
        self.asv_x += dx
        self.asv_y += dy
        self.asv_h = float(h)
        self.asv_w = float(w)
        dx_pos = float(self.asv_x) - x_prev
        dy_pos = float(self.asv_y) - y_prev
        self.speed_mps = float(math.hypot(dx_pos, dy_pos) / UPDATE_RATE)
        self.u_body = float(getattr(self.model, "_v", self.speed_mps))
        self.v_body = float(getattr(self.model, "_v_sway", 0.0))
        course_deg = float(math.degrees(math.atan2(dx_pos, dy_pos))) if self.speed_mps > 1e-6 else float(self.asv_h)

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=None)
        self._update_path_relative_states(course_deg=course_deg)
        self._update_local_planner_features()
        self.asv_path.append((self.asv_x, self.asv_y))
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        collided = bool(self._check_collision_geom())
        reached_goal = bool(self.distance_to_goal <= (VESSEL_LENGTH * 0.5))

        ye = abs(float(self.cross_track_error))
        U_norm = float(np.clip(self.speed_mps / U_MAX, 0.0, 1.5))
        cos_chi = float(np.cos(np.radians(self.course_error)))

        # Simple path-following term. It rewards staying close to path, aligned, and moving forward.
        r_pf = float(1.2 * math.exp(-GAMMA_E * ye) + 0.4 * cos_chi + 0.3 * U_norm * cos_chi)

        # Obstacle penalty from sector distances.
        sector_d = self.lidar.sector_ranges.astype(np.float32)
        sector_angles_deg = self.lidar.sector_angles.astype(np.float32)
        theta_rad = np.radians(sector_angles_deg)
        w_sec = 1.0 / (1.0 + np.abs(GAMMA_THETA * theta_rad))
        x = np.maximum(sector_d, EPSILON_X)
        pen = 1.0 / (GAMMA_X * (x ** 2))
        r_oa = -float(np.sum(w_sec * pen) / (np.sum(w_sec) + 1e-6))

        # Local planner penalty: when front is blocked, staying on path is bad;
        # moving toward the lidar-selected bypass side is good.
        target_err = float(self.cross_track_error - self.local_target_cte)

        # softer local-target penalty so the agent does not overcommit to a wide bypass
        r_local = -float(
            K_LOCAL_TARGET * self.block_alpha * min((target_err / max(BYPASS_CTE, 1e-6)) ** 2, 2.5)
        )

        # only penalize being centered when really centered on a blocked path
        r_center = -float(
            K_CENTER_BLOCK * self.block_alpha * math.exp(-ye / 0.35)
        )

        min_border = min(self.asv_x, self.map_width - self.asv_x, self.asv_y, self.map_height - self.asv_y)
        r_border = -float(K_BORDER * max(0.0, 1.0 - min_border / 1.5) ** 2)

        r_exist = R_EXIST
        if collided:
            reward = float(R_COLLISION)
        else:
            reward = float(self.current_lambda * r_pf + (1.0 - self.current_lambda) * r_oa + r_local + r_center + r_border + r_exist)
            if reached_goal:
                reward += R_GOAL

        terminated = self.check_done()
        self.step_count += 1
        truncated = False
        if self.step_count >= MAX_EPISODE_STEPS and not terminated:
            truncated = True
            reward += R_TIMEOUT

        info = {
            "lam": float(self.current_lambda),
            "log10_lambda": float(self.current_log10_lambda),
            "r_pf": float(r_pf),
            "r_oa": float(r_oa),
            "r_local": float(r_local),
            "r_center": float(r_center),
            "r_border": float(r_border),
            "r_exist": float(r_exist),
            "reward": float(reward),
            "ye": float(ye),
            "speed_mps": float(self.speed_mps),
            "u_body_mps": float(self.u_body),
            "v_body_mps": float(self.v_body),
            "course_error": float(self.course_error),
            "lookahead_course_error": float(self.lookahead_course_error),
            "cross_track_error": float(self.cross_track_error),
            "front_clearance": float(self.front_clearance),
            "left_clearance": float(self.left_clearance),
            "right_clearance": float(self.right_clearance),
            "block_alpha": float(self.block_alpha),
            "local_target_cte": float(self.local_target_cte),
            "min_lidar": float(np.min(self.lidar.ranges)) if len(self.lidar.ranges) > 0 else float("inf"),
            "min_sector_range": float(np.min(sector_d)),
            "p10_sector_range": float(np.percentile(sector_d, 10)),
            "mean_sector_pen": float(np.mean(pen)),
            "rpm": float(rpm),
            "rudder_deg": float(rudder_cmd * 40.0),
            "distance_to_goal": float(self.distance_to_goal),
            "collided": bool(collided),
            "reached_goal": bool(reached_goal),
            "timeout": bool(truncated),
            "path_mode": self.path_mode_used,
        }

        if self.render_mode in self.metadata["render_modes"]:
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.display is None:
            self.display = pygame.display.set_mode(self.window_size)
        scale = float(self.render_scale)

        def scale_point(xy):
            x, y = float(xy[0]), float(xy[1])
            px = int(round(x * scale))
            py = int(round((self.map_height - y) * scale))
            py = max(0, min(self.window_size[1] - 1, py))
            return (px, py)

        self.surface.fill((0, 0, 0))
        W = self.window_size[0] - 1
        H = self.window_size[1] - 1
        pygame.draw.rect(self.surface, (200, 0, 0), pygame.Rect(0, 0, W, H), width=2)
        for obs in self.obstacles:
            pygame.draw.polygon(self.surface, (200, 0, 0), [scale_point(p) for p in obs])
        self.lidar.render(self.surface, scale_point)
        path_px = [scale_point(p) for p in self.path]
        if len(path_px) >= 2:
            pygame.draw.lines(self.surface, (0, 200, 0), False, path_px, 2)
        pygame.draw.circle(self.surface, (100, 0, 0), scale_point((self.tgt_x, self.tgt_y)), 3)
        pygame.draw.circle(self.surface, (0, 220, 220), scale_point((self.lookahead_x, self.lookahead_y)), 3)
        pygame.draw.circle(self.surface, (200, 0, 200), scale_point((self.goal_x, self.goal_y)), 6)

        # Draw the local lidar-selected target CTE point near the closest path point.
        if abs(self.local_target_cte) > 1e-4 and len(self.path) > 1:
            tangent = self._path_tangent(self.closest_idx)
            normal_left = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            target = self.path[self.closest_idx] + self.local_target_cte * normal_left
            pygame.draw.circle(self.surface, (255, 180, 0), scale_point(target), 5)

        if self.icon is None:
            self.icon = pygame.image.frombytes(BOAT_ICON['bytes'], BOAT_ICON['size'], BOAT_ICON['format'])
            self.icon_scaled = None
            self._icon_scaled_size = None
        icon_width = max(1, int(round(VESSEL_WIDTH * scale)))
        icon_length = max(1, int(round(VESSEL_LENGTH * scale)))
        icon_size = (icon_width, icon_length)
        if self.icon_scaled is None or self._icon_scaled_size != icon_size:
            self.icon_scaled = pygame.transform.smoothscale(self.icon, icon_size)
            self._icon_scaled_size = icon_size
        os = pygame.transform.rotozoom(self.icon_scaled, -self.asv_h, 1)
        self.surface.blit(os, os.get_rect(center=scale_point((self.asv_x, self.asv_y))))
        pygame.draw.polygon(self.surface, (255, 0, 0), [scale_point(p) for p in self._hull_polygon_world()], width=2)
        if self.status is not None:
            status_line_1, _ = self.status.render(
                f"{self.elapsed_time:05.1f}s u:{self.u_body:0.2f} v:{self.v_body:+0.2f} r:{self.asv_w:+0.1f}",
                (255, 255, 255), (0, 0, 0),
            )
            status_line_2, _ = self.status.render(
                f"cte:{self.cross_track_error:+0.2f} tgt:{self.local_target_cte:+0.2f} front:{self.front_clearance:0.1f} L/R:{self.left_clearance:0.1f}/{self.right_clearance:0.1f}",
                (255, 255, 255), (0, 0, 0),
            )
            self.surface.blit(status_line_1, (10, self.window_size[1] - 28))
            self.surface.blit(status_line_2, (10, self.window_size[1] - 14))
        self.display.blit(self.surface, (0, 0))
        pygame.display.update()
        self.fps_clock.tick(RENDER_FPS)
        if self.record_video:
            frame = pygame.surfarray.array3d(self.surface)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter("asv_lidar.mp4", fourcc, self.video_fps, self.frame_size)
            self.video_writer.write(frame)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.display is not None:
            pygame.display.quit()
            self.display = None


if __name__ == "__main__":
    env = ASVLidarEnv(render_mode="human", map_width=10, map_height=25, path_mode="straight", test_case=1)
    obs, _ = env.reset()
    while True:
        action = np.array([-0.2, 0.0], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)
        if term or trunc:
            print(f"Done: reward={rew:.2f} info={info}")
            env.close()
            pygame.quit()
            break
