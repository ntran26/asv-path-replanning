from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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
# Default "paper mode" geometry
# ---------------------------------------------------------------------------
UPDATE_RATE = 0.1   # 10 Hz
RENDER_FPS = 10
MAP_WIDTH = 25
MAP_HEIGHT = 50
MAX_OBS = 6
OBSTACLE_MODE = "single_on_path"
# OBSTACLE_MODE = "old_path_relative"
OBSTACLE_SIZE = 1.0
OBSTACLE_PATH_START_FRAC = 0.20
OBSTACLE_PATH_END_FRAC = 0.80

# Path generation
PATH_MODE = "mixed"       # "straight", "curve", "mixed"
CURVE_PROB = 0.5
LOOKAHEAD_FRACTION = 0.25  # paper-like ratio: lookahead ≈ 0.25 * path length

# Reward parameters (paper-like)
GAMMA_E = 0.05
GAMMA_THETA = 4.0
ALPHA_R = 0.1
R_COLLISION = -1000.0
R_EXIST = -0.6
GAMMA_X = 0.005
EPSILON_X = 1.0

# Lambda conditioning (paper-style, one lambda per episode)
LAMBDA_MIN = 1e-4
LAMBDA_MAX = 1.0
DEFAULT_EVAL_LAMBDA = 1.0

# Speed control (rpm)
RPM_MIN = 0
RPM_MAX = 24
U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * RPM_MAX)
MAX_IN = 1.0
MIN_IN = -1.0

# Timeout condition
MAX_EPISODE_STEPS = 1000
R_TIMEOUT = -1000.0


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
        record_video: bool = True,
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
        self.obstacle_mode = OBSTACLE_MODE
        self.obstacle_size = OBSTACLE_SIZE

        pygame.init()
        self.render_mode = render_mode
        self.render_scale = float(RENDER_SCALE)
        self.window_size = (int(round(self.map_width * self.render_scale)), int(round(self.map_height * self.render_scale)))
        self.world_size = (self.map_width, self.map_height)

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

        # Episode state
        self.elapsed_time = 0.0
        self.asv_x = 0.0
        self.asv_y = 0.0
        self.asv_h = 0.0
        self.asv_w = 0.0  # yaw rate [deg/s]
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

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
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
        out = []
        for obs in obstacles:
            out.append([(float(px) * sx, float(py) * sy) for px, py in obs])
        return out

    def _sample_lambda(self) -> None:
        if self.lambda_override is not None:
            lam = float(self.lambda_override)
        else:
            # Paper-style sampling: -log10(lambda) ~ Gamma(1, 2)
            g = float(np.random.gamma(shape=1.0, scale=2.0))
            lam = 10.0 ** (-g)
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
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
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

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------
    def _start_goal_random(self) -> Tuple[float, float, float, float]:
        margin_x = max(2.0, 0.08 * self.map_width)
        start_x = float(np.random.uniform(margin_x, self.map_width - margin_x))
        goal_x = float(np.random.uniform(margin_x, self.map_width - margin_x))
        start_y = 2.0
        goal_y = self.map_height - 3.0
        return start_x, start_y, goal_x, goal_y

    def _choose_path_mode(self) -> str:
        if self.path_mode == "mixed":
            return "curve" if np.random.rand() < self.curve_prob else "straight"
        return self.path_mode

    def _generate_straight_path(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> np.ndarray:
        path_length = max(20, int(np.hypot(goal_x - start_x, goal_y - start_y) * 4.0))
        path_x = np.linspace(start_x, goal_x, path_length, dtype=np.float32)
        path_y = np.linspace(start_y, goal_y, path_length, dtype=np.float32)
        return np.column_stack((path_x, path_y)).astype(np.float32)

    def _generate_curve_path(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> np.ndarray:
        start = np.array([start_x, start_y], dtype=np.float32)
        goal = np.array([goal_x, goal_y], dtype=np.float32)
        vec = goal - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return self._generate_straight_path(start_x, start_y, goal_x, goal_y)
        tangent = vec / length
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        mid = 0.5 * (start + goal)
        max_offset = 0.25 * min(self.map_width, self.map_height)
        offset = float(np.random.uniform(-max_offset, max_offset))
        control = mid + offset * normal
        control[0] = float(np.clip(control[0], 1.5, self.map_width - 1.5))
        control[1] = float(np.clip(control[1], 1.5, self.map_height - 1.5))

        n = max(40, int(length * 5.0))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        one_minus_t = 1.0 - t
        pts = (
            (one_minus_t[:, None] ** 2) * start[None, :]
            + 2.0 * one_minus_t[:, None] * t[:, None] * control[None, :]
            + (t[:, None] ** 2) * goal[None, :]
        )
        return pts.astype(np.float32)

    def _generate_path(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> np.ndarray:
        self.path_mode_used = self._choose_path_mode()
        if self.path_mode_used == "curve":
            path = self._generate_curve_path(start_x, start_y, goal_x, goal_y)
        else:
            path = self._generate_straight_path(start_x, start_y, goal_x, goal_y)
        diffs = np.diff(path, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self.path_s = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)
        total_length = float(self.path_s[-1]) if len(self.path_s) > 0 else 1.0
        self.lookahead_distance = max(2.0, self.lookahead_fraction * total_length)
        return path

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
        lookahead_bearing_deg = self._bearing_deg(asv_pos, lookahead_pt)
        self.lookahead_course_error = self._wrap180(lookahead_bearing_deg - course_deg)

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    def _generate_single_on_path_obstacle(self) -> List[List[Tuple[float, float]]]:
        """Generate exactly one rectangular obstacle with its centre on the path.

        This is intended for the focused curriculum where the agent must learn
        that a blocked reference path requires a temporary bypass. The obstacle
        is sampled uniformly along the middle portion of the path so it is not
        placed immediately at the start or goal.
        """
        if len(self.path) < 2:
            return []

        half = 0.5 * float(self.obstacle_size)
        margin = half + 0.10

        s_total = float(self.path_s[-1]) if len(self.path_s) > 0 else 0.0
        if s_total <= 1e-6:
            return []

        s_min = OBSTACLE_PATH_START_FRAC * s_total
        s_max = OBSTACLE_PATH_END_FRAC * s_total

        # Build a list of feasible path indices where the whole obstacle stays
        # inside the map and is not too close to the start/goal.
        feasible_indices: List[int] = []
        start = np.array([self.start_x, self.start_y], dtype=np.float32)
        goal = np.array([self.goal_x, self.goal_y], dtype=np.float32)

        for idx, (px, py) in enumerate(self.path):
            if len(self.path_s) > idx:
                s_val = float(self.path_s[idx])
                if s_val < s_min or s_val > s_max:
                    continue

            cx = float(px)
            cy = float(py)
            if cx - half < margin or cx + half > self.map_width - margin:
                continue
            if cy - half < margin or cy + half > self.map_height - margin:
                continue

            c = np.array([cx, cy], dtype=np.float32)
            if float(np.linalg.norm(c - start)) < 2.0:
                continue
            if float(np.linalg.norm(c - goal)) < 2.0:
                continue

            feasible_indices.append(idx)

        if not feasible_indices:
            # In normal 25x50 / 10x25 maps this should not happen. Returning
            # empty is safer than silently moving the obstacle off the path.
            return []

        idx = int(np.random.choice(feasible_indices))
        cx = float(self.path[idx, 0])
        cy = float(self.path[idx, 1])

        x0 = cx - half
        x1 = cx + half
        y0 = cy - half
        y1 = cy + half
        return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]

    def _generate_obstacles(self, num_obs: int, test_case: Optional[int] = None):
        if test_case is not None:
            raw = self.scenario.obstacles(test_case=test_case)
            return self._scale_case_obstacles(raw)
        
        # Switch self.obstacle_mode / OBSTACLE_MODE back to "old_path_relative"
        # to restore the previous random path-relative obstacle generator.
        if self.obstacle_mode == "single_on_path":
            return self._generate_single_on_path_obstacle()

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
            num_obs = int(np.random.randint(0, self.max_obs + 1))
        else:
            num_obs = 0
        self.obstacles = self._generate_obstacles(num_obs, self.test_case)
        self.asv_path = [(self.asv_x, self.asv_y)]
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=None)
        self._update_path_relative_states(course_deg=self.asv_h)

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
        rpm = (throttle_cmd - MIN_IN) * ((RPM_MAX - RPM_MIN) / (MAX_IN - MIN_IN)) + RPM_MIN

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

        if self.speed_mps > 1e-6:
            course_deg = float(math.degrees(math.atan2(dx_pos, dy_pos)))
        else:
            course_deg = float(self.asv_h)

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=None)
        self._update_path_relative_states(course_deg=course_deg)

        self.asv_path.append((self.asv_x, self.asv_y))
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        collided = bool(self._check_collision_geom())
        reached_goal = bool(self.distance_to_goal <= (VESSEL_LENGTH * 0.5))

        # ------------------------------------------------------------------
        # Reward: lambda-conditioned trade-off (paper-style)
        # ------------------------------------------------------------------
        ye = abs(float(self.cross_track_error))
        U_norm = float(np.clip(self.speed_mps / U_MAX, 0.0, 1.5))
        cos_chi = float(np.cos(np.radians(self.course_error)))
        r_pf = float(-1.0 + (U_norm * cos_chi + 1.0) * (math.exp(-GAMMA_E * ye) + 1.0))

        # sector_closeness = self.lidar.sector_closeness.astype(np.float32)
        # sector_angles_deg = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, LIDAR_SECTORS, dtype=np.float32)
        # sector_weights = 1.0 / (1.0 + np.abs(GAMMA_THETA * np.radians(sector_angles_deg)))
        # r_oa = -float(np.sum(sector_weights * sector_closeness) / (np.sum(sector_weights) + 1e-6))
        
        sector_d = self.lidar.sector_ranges.astype(np.float32)

        sector_angles_deg = np.linspace(
            -LIDAR_SWATH / 2.0,
            LIDAR_SWATH / 2.0,
            LIDAR_SECTORS,
            dtype=np.float32,
        )

        theta_rad = np.radians(sector_angles_deg)
        w = 1.0 / (1.0 + np.abs(GAMMA_THETA * theta_rad))

        x = np.maximum(sector_d, EPSILON_X)
        pen = 1.0 / (GAMMA_X * (x ** 2))

        r_oa = -float(np.sum(w * pen) / (np.sum(w) + 1e-6))
        
        r_exist = -self.current_lambda * (2.0 * ALPHA_R + 1.0)

        if collided:
            reward = R_COLLISION
        else:
            reward = self.current_lambda * r_pf + (1.0 - self.current_lambda) * r_oa + r_exist
            if reached_goal:
                reward += 50.0

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
            "r_exist": float(r_exist),
            "reward": float(reward),
            "ye": float(ye),
            "speed_mps": float(self.speed_mps),
            "u_body_mps": float(self.u_body),
            "v_body_mps": float(self.v_body),
            "course_error": float(self.course_error),
            "lookahead_course_error": float(self.lookahead_course_error),
            "cross_track_error": float(self.cross_track_error),
            "min_lidar": float(np.min(self.lidar.ranges)) if len(self.lidar.ranges) > 0 else float("inf"),
            "min_sector_range": float(np.min(sector_d)),
            "p10_sector_range": float(np.percentile(sector_d, 10)),
            "mean_sector_pen": float(np.mean(pen)),
            # "mean_sector_closeness": float(np.mean(sector_closeness)) if len(sector_closeness) > 0 else 0.0,
            "rpm": float(rpm),
            "rudder_deg": float(rudder_cmd * 40.0),
            "distance_to_goal": float(self.distance_to_goal),
            "collided": bool(collided),
            "reached_goal": bool(reached_goal),
            "path_mode": self.path_mode_used,
            "timeout": bool(truncated),
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
        bw = 2
        W = self.window_size[0] - 1
        H = self.window_size[1] - 1
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (0, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, H), (W, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (W, 0), (W, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (W, 0), bw)

        for obs in self.obstacles:
            pygame.draw.polygon(self.surface, (200, 0, 0), [scale_point(p) for p in obs])

        self.lidar.render(self.surface, scale_point)

        path_px = [scale_point(p) for p in self.path]
        if len(path_px) >= 2:
            pygame.draw.lines(self.surface, (0, 200, 0), False, path_px, 2)
        pygame.draw.circle(self.surface, (100, 0, 0), scale_point((self.tgt_x, self.tgt_y)), 3)
        pygame.draw.circle(self.surface, (0, 220, 220), scale_point((self.lookahead_x, self.lookahead_y)), 3)
        pygame.draw.circle(self.surface, (200, 0, 200), scale_point((self.goal_x, self.goal_y)), 6)

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
        ship_outline = self._hull_polygon_world()
        pygame.draw.polygon(self.surface, (255, 0, 0), [scale_point(p) for p in ship_outline], width=2)

        if self.status is not None:
            status_line_1, _ = self.status.render(
                f"{self.elapsed_time:05.1f}s  u:{self.u_body:0.2f}  v:{self.v_body:+0.2f}  r:{self.asv_w:+0.1f}",
                (255, 255, 255),
                (0, 0, 0),
            )
            status_line_2, _ = self.status.render(
                f"cte:{self.cross_track_error:+0.2f}  ce:{self.course_error:+0.1f}  la:{self.lookahead_course_error:+0.1f}  log10λ:{self.current_log10_lambda:+0.2f}",
                (255, 255, 255),
                (0, 0, 0),
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


if __name__ == "__main__":
    env = ASVLidarEnv(render_mode="human", map_width=25, map_height=50, path_mode="mixed")
    obs, _ = env.reset()
    while True:
        action = np.array([-1.0, 1.0], dtype=np.float32)
        obs, rew, term, _, info = env.step(action)
        if term:
            print(f"Done: reward={rew:.2f}  info={info}")
            pygame.display.quit()
            pygame.quit()
            break
