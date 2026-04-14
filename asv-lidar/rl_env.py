import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import pygame
import pygame.freetype
from ship_model_selector import (
    ShipModel,
    MODEL_RPM_MAX,
    MODEL_U_MAX,
    model_u_body,
    model_v_body,
    model_rudder_deg,
    MAX_RUD_ANGLE,
    MAX_SURGE_SPEED,
    MAX_SWAY_SPEED,
    VESSEL_LENGTH,
    VESSEL_WIDTH,
    HULL_MARGIN,
    HULL_FORWARD_SHIFT,
)
from asv_lidar import Lidar, LIDAR_MIN_RANGE, LIDAR_RANGE, LIDAR_BEAMS, LIDAR_SWATH
from test_run import TestCase
from images import BOAT_ICON
import cv2

RENDER_SCALE = 25

# Environment / system
UPDATE_RATE = 0.1
RENDER_FPS = 10
MAP_WIDTH = 10
MAP_HEIGHT = 25
MAX_OBS = 5
DHDG_MAX_DPS = 180.0
OBS_SECTORS = 18
SECTOR_PERCENTILE = 10.0
GOAL_RADIUS = VESSEL_LENGTH / 2.0
TRAIN_STAGE_CHOICES = (1, 2, 3)

STAGE_FIXED_START_X = MAP_WIDTH / 2.0
STAGE_FIXED_START_Y = 2.0
STAGE_FIXED_GOAL_X = MAP_WIDTH / 2.0
STAGE_FIXED_GOAL_Y = float(MAP_HEIGHT - 5)
STAGE1_OBSTACLE = [(3.5, 12.0), (4.5, 12.0), (4.5, 13.0), (3.5, 13.0)]
STAGE2_OBS_X_MIN = 2.0
STAGE2_OBS_X_MAX = MAP_WIDTH - 2.0
STAGE2_OBS_Y_MIN = 8.0
STAGE2_OBS_Y_MAX = MAP_HEIGHT - 8.0

# Paper-style reward parameters.
GAMMA_E = 0.05
GAMMA_THETA = 4.0
GAMMA_X = 0.005
EPSILON_X = 1.0
ALPHA_R = 0.1
LAMBDA_REWARD = 0.50
R_COLLISION = -2000.0

# Speed control (rpm)
RPM_MIN = 0
RPM_MAX = float(MODEL_RPM_MAX)
U_MAX = float(MODEL_U_MAX)
MAX_IN = 1.0
MIN_IN = -1.0


class ASVLidarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = "human",
        fixed_lambda: float | None = LAMBDA_REWARD,
        train_stage: int = 3,
    ) -> None:
        self.map_width = MAP_WIDTH
        self.map_height = MAP_HEIGHT

        pygame.init()
        self.render_mode = render_mode
        self.render_scale = RENDER_SCALE
        self.window_size = (
            int(self.map_width * self.render_scale),
            int(self.map_height * self.render_scale),
        )

        self.display = None
        self.surface = None
        self.status = None
        self.icon = None
        self.icon_scaled = None
        self._icon_scaled_size = None
        self.fps_clock = pygame.time.Clock()
        if render_mode in self.metadata["render_modes"]:
            self.surface = pygame.Surface(self.window_size)
            self.status = pygame.freetype.SysFont(
                pygame.font.get_default_font(), size=10
            )

        self.model = ShipModel()
        self.lidar = Lidar()
        self.max_obs = MAX_OBS
        self.test_case = None
        self.asv_path = []

        self.map_border = [
            [(0.0, 0.0), (0.0, float(self.map_height))],
            [(0.0, float(self.map_height)), (float(self.map_width), float(self.map_height))],
            [(float(self.map_width), float(self.map_height)), (float(self.map_width), 0.0)],
            [(0.0, 0.0), (float(self.map_width), 0.0)],
        ]

        self.record_video = bool(render_mode in self.metadata["render_modes"])
        self.video_writer = None
        self.frame_size = self.window_size
        self.video_fps = RENDER_FPS

        self.elapsed_time = 0.0
        self.reward = 0.0
        self.fixed_lambda = LAMBDA_REWARD if fixed_lambda is None else float(np.clip(fixed_lambda, 0.0, 1.0))
        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.asv_x = 0.0
        self.asv_y = 0.0
        self.asv_h = 0.0
        self.asv_w = 0.0
        self.speed_mps = 0.0
        self.tgt = 0.0
        self.tgt_x = 0.0
        self.tgt_y = 0.0
        self.path = np.zeros((2, 2), dtype=np.float32)
        self.distance_to_goal = 0.0
        self.heading_error = 0.0
        self.lambda_reward = float(self.fixed_lambda)
        self.obstacles = []
        self.sector_lidar = np.full(OBS_SECTORS, float(LIDAR_RANGE), dtype=np.float32)
        self.train_stage = 3
        self.set_train_stage(train_stage)

        self.observation_space = Dict(
            {
                "lidar": Box(low=0.0, high=float(LIDAR_RANGE), shape=(LIDAR_BEAMS,), dtype=np.float32),
                "lidar_sectors": Box(low=0.0, high=float(LIDAR_RANGE), shape=(OBS_SECTORS,), dtype=np.float32),
                "pos": Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([float(self.map_width), float(self.map_height)], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "hdg": Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
                "u_body": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "v_body": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "rudder_state": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "speed": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "dhdg": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "tgt": Box(low=-float(self.map_width), high=float(self.map_width), shape=(1,), dtype=np.float32),
                "target_heading": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                "distance_to_goal": Box(
                    low=0.0,
                    high=float(np.hypot(self.map_width, self.map_height)),
                    shape=(1,),
                    dtype=np.float32,
                ),
                "front_min": Box(low=0.0, high=float(LIDAR_RANGE), shape=(1,), dtype=np.float32),
                "front_p10": Box(low=0.0, high=float(LIDAR_RANGE), shape=(1,), dtype=np.float32),
                "left_min": Box(low=0.0, high=float(LIDAR_RANGE), shape=(1,), dtype=np.float32),
                "right_min": Box(low=0.0, high=float(LIDAR_RANGE), shape=(1,), dtype=np.float32),
                "near_flag": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def set_train_stage(self, stage: int) -> None:
        stage = int(stage)
        if stage not in TRAIN_STAGE_CHOICES:
            raise ValueError(f"Unsupported train_stage={stage}. Use one of {TRAIN_STAGE_CHOICES}.")
        self.train_stage = stage

    def _get_obs(self):
        obs = {
            "lidar": self.lidar.ranges.astype(np.float32),
            "lidar_sectors": self.sector_lidar.astype(np.float32),
            "pos": np.array([self.asv_x, self.asv_y], dtype=np.float32),
            "hdg": np.array([self.asv_h], dtype=np.float32),
            "u_body": np.array(
                [np.clip(model_u_body(self.model) / max(float(MAX_SURGE_SPEED), 1e-6), 0.0, 1.0)],
                dtype=np.float32,
            ),
            "v_body": np.array(
                [np.clip(model_v_body(self.model) / max(float(MAX_SWAY_SPEED), 1e-6), -1.0, 1.0)],
                dtype=np.float32,
            ),
            "rudder_state": np.array(
                [np.clip(model_rudder_deg(self.model) / max(float(MAX_RUD_ANGLE), 1e-6), -1.0, 1.0)],
                dtype=np.float32,
            ),
            "speed": np.array([np.clip(self.speed_mps / max(U_MAX, 1e-6), 0.0, 1.0)], dtype=np.float32),
            "dhdg": np.array([np.clip(self.asv_w / DHDG_MAX_DPS, -1.0, 1.0)], dtype=np.float32),
            "tgt": np.array([self.tgt], dtype=np.float32),
            "target_heading": np.array([self.heading_error], dtype=np.float32),
            "distance_to_goal": np.array([self.distance_to_goal], dtype=np.float32),
        }
        obs.update(self._lidar_features())
        return obs

    def _hull_polygon_world(self):
        length = VESSEL_LENGTH + 2 * HULL_MARGIN
        width = VESSEL_WIDTH + 2 * HULL_MARGIN
        shift = HULL_FORWARD_SHIFT
        half_l = 0.5 * length
        half_w = 0.5 * width
        h = np.radians(float(self.asv_h))
        sin_h = np.sin(h)
        cos_h = np.cos(h)

        local = [
            (+half_l + shift, +half_w),
            (+half_l + shift, -half_w),
            (-half_l + shift, -half_w),
            (-half_l + shift, +half_w),
        ]

        poly = []
        for x_forward, y_left in local:
            x = float(self.asv_x) + x_forward * sin_h - y_left * cos_h
            y = float(self.asv_y) + x_forward * cos_h + y_left * sin_h
            poly.append((x, y))
        return poly

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

    def _check_collision_geom(self) -> tuple[bool, str | None]:
        hull = self._hull_polygon_world()
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]

        if min(xs) < 0.0 or max(xs) > self.map_width or min(ys) < 0.0 or max(ys) > self.map_height:
            return True, "border"

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
                return True, "obstacle"
        return False, None

    def _pool_lidar_sectors(self) -> np.ndarray:
        ranges = np.asarray(self.lidar.ranges, dtype=np.float32)
        angles = np.asarray(self.lidar.angles, dtype=np.float32)
        sector_edges = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, OBS_SECTORS + 1, dtype=np.float32)
        pooled = np.full(OBS_SECTORS, float(LIDAR_RANGE), dtype=np.float32)

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

    def _lidar_features(self):
        lidar_d = self.lidar.ranges.astype(np.float32)
        angles = self.lidar.angles.astype(np.float32)

        valid = (lidar_d >= float(LIDAR_MIN_RANGE)) & (lidar_d <= float(LIDAR_RANGE))

        def sector(mask, mode="min"):
            mask = mask & valid
            if not np.any(mask):
                return float(LIDAR_RANGE)
            vals = lidar_d[mask]
            if mode == "min":
                return float(np.min(vals))
            if mode == "p10":
                return float(np.percentile(vals, 10))
            raise ValueError(f"Unknown sector mode: {mode}")

        front = np.abs(angles) <= 30.0
        left = (angles > 30.0) & (angles <= 90.0)
        right = (angles < -30.0) & (angles >= -90.0)

        front_min = sector(front, "min")
        front_p10 = sector(front, "p10")
        left_min = sector(left, "min")
        right_min = sector(right, "min")
        near_flag = 1.0 if front_min < 2.0 else 0.0

        return {
            "front_min": np.array([front_min], dtype=np.float32),
            "front_p10": np.array([front_p10], dtype=np.float32),
            "left_min": np.array([left_min], dtype=np.float32),
            "right_min": np.array([right_min], dtype=np.float32),
            "near_flag": np.array([near_flag], dtype=np.float32),
        }

    def _generate_path(self, start_x, start_y, goal_x, goal_y):
        path_length = max(2, int(np.hypot(goal_x - start_x, goal_y - start_y)))
        path_x = np.round(np.linspace(start_x, goal_x, path_length)).astype(int)
        path_y = np.round(np.linspace(start_y, goal_y, path_length)).astype(int)
        return np.column_stack((path_x, path_y))

    def _generate_obstacles(self, num_obs, test_case=None):
        if test_case is not None:
            return TestCase().obstacles(test_case=test_case)

        obstacles = []
        tries = 0
        while len(obstacles) < num_obs and tries < 200:
            tries += 1
            x = np.random.randint(1, self.map_width - 1)
            y = np.random.randint(1, self.map_height - 1)
            if np.hypot(x - self.start_x, y - self.start_y) <= 1.5:
                continue
            if np.hypot(x - self.goal_x, y - self.goal_y) <= 1.5:
                continue
            obstacles.append([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
        return obstacles

    def _sample_stage2_obstacle(self):
        tries = 0
        while tries < 200:
            tries += 1
            x = float(np.random.uniform(STAGE2_OBS_X_MIN, STAGE2_OBS_X_MAX))
            y = float(np.random.uniform(STAGE2_OBS_Y_MIN, STAGE2_OBS_Y_MAX))
            if np.hypot(x - self.start_x, y - self.start_y) <= 2.0:
                continue
            if np.hypot(x - self.goal_x, y - self.goal_y) <= 2.0:
                continue
            return [(x, y), (x + 1.0, y), (x + 1.0, y + 1.0), (x, y + 1.0)]
        return list(STAGE1_OBSTACLE)

    def _calculate_angle(self, asv_x, asv_y, heading, goal_x, goal_y):
        dx = goal_x - asv_x
        dy = goal_y - asv_y
        target_angle = np.degrees(np.arctan2(dx, dy))
        return float((target_angle - heading + 180.0) % 360.0 - 180.0)

    def _signed_cross_track_error(self, asv_x, asv_y):
        path_dx = float(self.goal_x - self.start_x)
        path_dy = float(self.goal_y - self.start_y)
        path_norm = float(np.hypot(path_dx, path_dy))
        if path_norm < 1e-6:
            return 0.0
        rel_x = float(asv_x - self.start_x)
        rel_y = float(asv_y - self.start_y)
        cross = path_dx * rel_y - path_dy * rel_x
        return float(cross / path_norm)

    def _update_compact_state(self):
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        self.tgt = float(self._signed_cross_track_error(self.asv_x, self.asv_y))
        self.heading_error = float(
            self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y)
        )
        self.sector_lidar = self._pool_lidar_sectors()

        asv_pos = np.array([self.asv_x, self.asv_y], dtype=np.float32)
        distance = np.linalg.norm(self.path - asv_pos, axis=1)
        closest_idx = int(np.argmin(distance))
        self.tgt_x, self.tgt_y = self.path[closest_idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.elapsed_time = 0.0
        self.reward = 0.0
        self.model = ShipModel()
        self.lidar.reset()
        self.asv_path = []
        self.asv_h = 0.0
        self.asv_w = 0.0
        self.speed_mps = 0.0

        selected_case = self.test_case

        if selected_case is None:
            if self.train_stage in (1, 2):
                self.start_x = float(STAGE_FIXED_START_X)
                self.start_y = float(STAGE_FIXED_START_Y)
                self.goal_x = float(STAGE_FIXED_GOAL_X)
                self.goal_y = float(STAGE_FIXED_GOAL_Y)
            else:
                self.start_x = float(np.random.randint(2, self.map_width - 2))
                self.start_y = 2.0
                self.goal_x = float(np.random.randint(2, self.map_width - 2))
                self.goal_y = float(self.map_height - 5)
        else:
            self.start_x, self.start_y, self.goal_x, self.goal_y = map(float, TestCase().position(test_case=selected_case))

        self.asv_x = float(self.start_x)
        self.asv_y = float(self.start_y)
        self.path = self._generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        if selected_case is None:
            if self.train_stage == 1:
                self.num_obs = 1
                self.obstacles = [list(STAGE1_OBSTACLE)]
            elif self.train_stage == 2:
                self.num_obs = 1
                self.obstacles = [self._sample_stage2_obstacle()]
            else:
                self.num_obs = int(np.random.randint(0, self.max_obs + 1))
                self.obstacles = self._generate_obstacles(self.num_obs, selected_case)
        else:
            self.num_obs = len(TestCase().obstacles(test_case=selected_case))
            self.obstacles = self._generate_obstacles(self.num_obs, selected_case)
        self.asv_path.append((self.asv_x, self.asv_y))

        self.lambda_reward = float(self.fixed_lambda)
        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)
        self._update_compact_state()

        if self.render_mode in self.metadata["render_modes"]:
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], MIN_IN, MAX_IN))
        throttle_cmd = float(np.clip(action[1], MIN_IN, MAX_IN))

        rudder = rudder_cmd * 100.0
        rpm = (throttle_cmd - MIN_IN) * ((RPM_MAX - RPM_MIN) / (MAX_IN - MIN_IN)) + RPM_MIN

        x_prev = float(self.asv_x)
        y_prev = float(self.asv_y)

        dx, dy, h, w = self.model.update(rpm, rudder, UPDATE_RATE)
        self.asv_x += float(dx)
        self.asv_y += float(dy)
        self.asv_h = float(h)
        self.asv_w = float(w)

        dx_pos = float(self.asv_x - x_prev)
        dy_pos = float(self.asv_y - y_prev)
        self.speed_mps = float(np.hypot(dx_pos, dy_pos) / UPDATE_RATE)

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)
        self.asv_path.append((self.asv_x, self.asv_y))
        self._update_compact_state()

        collided, collision_type = self._check_collision_geom()
        reached_goal = bool(self.distance_to_goal <= GOAL_RADIUS)

        lam = float(self.lambda_reward)
        ye = float(abs(self.tgt))
        U = float(self.speed_mps)
        U_norm = float(np.clip(U / max(U_MAX, 1e-6), 0.0, 1.0))

        if U > 1e-6:
            course_deg = float(np.degrees(np.arctan2(dx_pos, dy_pos)))
        else:
            course_deg = float(self.asv_h)

        path_dx = float(self.goal_x - self.start_x)
        path_dy = float(self.goal_y - self.start_y)
        path_course_deg = float(np.degrees(np.arctan2(path_dx, path_dy)))
        chi_tilde_deg = float((course_deg - path_course_deg + 180.0) % 360.0 - 180.0)
        cos_chi = float(np.cos(np.radians(chi_tilde_deg)))
        r_pf = float(-1.0 + (U_norm * cos_chi + 1.0) * (np.exp(-GAMMA_E * ye) + 1.0))

        lidar_true = self.lidar.true_ranges.astype(np.float32)
        theta = np.radians(self.lidar.angles.astype(np.float32))
        weights = 1.0 / (1.0 + np.abs(GAMMA_THETA * theta))
        x = np.clip(lidar_true, EPSILON_X, LIDAR_RANGE)
        penalties = 1.0 / (GAMMA_X * x**2)
        r_oa = -float(np.sum(weights * penalties) / (np.sum(weights) + 1e-6))

        r_exist = float(-lam * (2.0 * ALPHA_R + 1.0))

        if collided:
            reward = float((1.0 - lam) * R_COLLISION)
        else:
            reward = float(lam * r_pf + (1.0 - lam) * r_oa + r_exist)

        self.reward = reward
        terminated = bool(collided or reached_goal)
        lidar_feat = self._lidar_features()
        rudder_deg = float(rudder_cmd * MAX_RUD_ANGLE)

        info = {
            "lam": float(lam),
            "r_pf": float(r_pf),
            "r_oa": float(r_oa),
            "r_exist": float(r_exist),
            "lambda_reward": float(lam),
            "reward": float(reward),
            "ye": float(ye),
            "U": float(U),
            "U_norm": float(U_norm),
            "course_deg": float(course_deg),
            "path_course_deg": float(path_course_deg),
            "chi_tilde_deg": float(chi_tilde_deg),
            "distance_to_goal": float(self.distance_to_goal),
            "cos_chi": float(cos_chi),
            "cross_track_error": float(self.tgt),
            "heading_error": float(self.heading_error),
            "tgt": float(self.tgt),
            "front_min": float(lidar_feat["front_min"][0]),
            "front_p10": float(lidar_feat["front_p10"][0]),
            "left_min": float(lidar_feat["left_min"][0]),
            "right_min": float(lidar_feat["right_min"][0]),
            "near_flag": float(lidar_feat["near_flag"][0]),
            "collided": bool(collided),
            "collision": bool(collided),
            "collision_type": collision_type,
            "reached_goal": bool(reached_goal),
            "speed_mps": float(self.speed_mps),
            "rpm": float(rpm),
            "rudder_deg": rudder_deg,
            "x": float(self.asv_x),
            "y": float(self.asv_y),
            "heading_deg": float(self.asv_h),
            "dhdg_raw": float(self.asv_w),
        }

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

        return self._get_obs(), reward, terminated, False, info

    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=10, exclude_corner=True):
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        length = np.linalg.norm(end_pos - start_pos)
        dash_amount = max(2, int(length / max(dash_length, 1)))
        dash_knots = np.array([
            np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)
        ]).transpose()
        return [
            pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n + 1]), width)
            for n in range(int(exclude_corner), dash_amount - int(exclude_corner), 2)
        ]

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
            return px, py

        self.surface.fill((0, 0, 0))

        bw = max(2, int(round(2)))
        width = self.window_size[0] - 1
        height = self.window_size[1] - 1
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (0, height), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, height), (width, height), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (width, 0), (width, height), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (width, 0), bw)

        for obs in self.obstacles:
            obs_px = [scale_point(p) for p in obs]
            pygame.draw.polygon(self.surface, (200, 0, 0), obs_px)

        self.lidar.render(self.surface, scale_point)

        self._draw_dashed_line(
            self.surface,
            (0, 200, 0),
            scale_point((self.start_x, self.start_y)),
            scale_point((self.goal_x, self.goal_y)),
            width=2,
            dash_length=int(np.clip(scale, 8, 30)),
        )
        pygame.draw.circle(self.surface, (100, 0, 0), scale_point((self.tgt_x, self.tgt_y)), radius=3)
        pygame.draw.circle(
            self.surface,
            (200, 0, 200),
            scale_point((self.goal_x, self.goal_y)),
            max(4, int(round(6))),
        )

        if self.icon is None:
            self.icon = pygame.image.frombytes(BOAT_ICON["bytes"], BOAT_ICON["size"], BOAT_ICON["format"])
            self.icon_scaled = None
            self._icon_scaled_size = None

        icon_width = max(1, int(round(VESSEL_WIDTH * scale)))
        icon_length = max(1, int(round(VESSEL_LENGTH * scale)))
        icon_size = (icon_width, icon_length)
        if self.icon_scaled is None or self._icon_scaled_size != icon_size:
            self.icon_scaled = pygame.transform.smoothscale(self.icon, icon_size)
            self._icon_scaled_size = icon_size

        rotated = pygame.transform.rotozoom(self.icon_scaled, -self.asv_h, 1)
        self.surface.blit(rotated, rotated.get_rect(center=scale_point((self.asv_x, self.asv_y))))
        ship_outline_px = [scale_point(p) for p in self._hull_polygon_world()]
        pygame.draw.polygon(self.surface, (255, 0, 0), ship_outline_px, width=max(2, int(round(2))))

        if self.status is not None:
            status_1, _ = self.status.render(
                f"{self.elapsed_time:005.1f}s  V:{self.speed_mps:0.2f}m/s  "
                f"HDG:{self.asv_h:+004.0f}({self.asv_w:+03.0f})  TGT:{self.tgt:+05.2f}",
                (255, 255, 255),
                (0, 0, 0),
            )
            status_2, _ = self.status.render(
                f"HDG_ERR:{self.heading_error:+05.1f}  LAM:{self.lambda_reward:0.2f}  R:{self.reward:0.2f}",
                (255, 255, 255),
                (0, 0, 0),
            )
            self.surface.blit(status_1, (10, self.window_size[1] - 30))
            self.surface.blit(status_2, (10, self.window_size[1] - 15))

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
    env = ASVLidarEnv(render_mode="human")
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        action = np.array([-1.0, 1.0], dtype=np.float32)
        obs, rew, term, _, _ = env.step(action)
        total_reward += rew
        if term:
            print(f"Elapsed time: {env.elapsed_time:.1f}s, Reward: {total_reward:0.2f}")
            env.close()
            pygame.quit()
            break
