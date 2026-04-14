import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import pygame
import pygame.freetype
from ship_model import (
    ShipModel,
    THRUST_COEF,
    DRAG_COEF,
    MAX_RUD_ANGLE,
    VESSEL_LENGTH,
    VESSEL_WIDTH,
    HULL_MARGIN,
    HULL_FORWARD_SHIFT,
)
from asv_lidar import Lidar, LIDAR_RANGE, LIDAR_BEAMS
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

# Paper-style reward parameters.
GAMMA_E = 0.05
GAMMA_THETA = 4.0
GAMMA_X = 0.005
EPSILON_X = 1.0
ALPHA_R = 0.1
LAMBDA_REWARD = 0.50
R_GOAL = 500.0
R_COLLISION = -2000.0
COLLISION_RANGE = 1.0

# Speed control (rpm)
RPM_MIN = 0
RPM_MAX = 24
U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * RPM_MAX)
MAX_IN = 1.0
MIN_IN = -1.0


class ASVLidarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = "human", fixed_lambda: float | None = LAMBDA_REWARD) -> None:
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

        self.observation_space = Dict(
            {
                "lidar": Box(low=0.0, high=float(LIDAR_RANGE), shape=(LIDAR_BEAMS,), dtype=np.float32),
                "pos": Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([float(self.map_width), float(self.map_height)], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "hdg": Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32),
                "speed": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "dhdg": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "tgt": Box(low=-float(self.map_width), high=float(self.map_width), shape=(1,), dtype=np.float32),
                "target_heading": Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self):
        return {
            "lidar": self.lidar.ranges.astype(np.float32),
            "pos": np.array([self.asv_x, self.asv_y], dtype=np.float32),
            "hdg": np.array([self.asv_h], dtype=np.float32),
            "speed": np.array([np.clip(self.speed_mps / max(U_MAX, 1e-6), 0.0, 1.0)], dtype=np.float32),
            "dhdg": np.array([np.clip(self.asv_w / DHDG_MAX_DPS, -1.0, 1.0)], dtype=np.float32),
            "tgt": np.array([self.tgt], dtype=np.float32),
            "target_heading": np.array([self.heading_error], dtype=np.float32),
        }

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

    def _check_collision_lidar(self) -> bool:
        ranges = np.asarray(self.lidar.ranges, dtype=np.float32)
        finite = ranges[np.isfinite(ranges)]
        if finite.size == 0:
            return False
        return bool(np.min(finite) < COLLISION_RANGE)

    def _update_compact_state(self):
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        self.tgt = float(self._signed_cross_track_error(self.asv_x, self.asv_y))
        self.heading_error = float(
            self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y)
        )

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
            self.start_x = float(np.random.randint(2, self.map_width - 2))
            self.start_y = 2.0
            self.goal_x = float(np.random.randint(2, self.map_width - 2))
            self.goal_y = float(self.map_height - 5)
        else:
            self.start_x, self.start_y, self.goal_x, self.goal_y = map(float, TestCase().position(test_case=selected_case))

        self.asv_x = float(self.start_x)
        self.asv_y = float(self.start_y)
        self.path = self._generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        self.num_obs = int(np.random.randint(0, self.max_obs + 1))
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

        # Legacy geometry collision check intentionally disabled in favor of
        # paper-style LiDAR-threshold collision detection.
        collided = bool(self._check_collision_lidar())
        reached_goal = bool(self.distance_to_goal <= (VESSEL_LENGTH / 2.0))

        u_norm = float(np.clip(self.speed_mps / max(U_MAX, 1e-6), 0.0, 1.0))
        chi_tilde = float(np.radians(self.heading_error))
        cos_chi = float(np.cos(chi_tilde))
        ye = float(abs(self.tgt))
        r_pf = float(-1.0 + (u_norm * cos_chi + 1.0) * (float(np.exp(-GAMMA_E * ye)) + 1.0))

        lidar_d = self.lidar.ranges.astype(np.float32)
        theta_rad = np.radians(self.lidar.angles.astype(np.float32))
        weights = 1.0 / (1.0 + np.abs(GAMMA_THETA * theta_rad))
        penalties = 1.0 / (GAMMA_X * (np.maximum(lidar_d, EPSILON_X) ** 2))
        r_oa = float(-np.sum(weights * penalties) / (np.sum(weights) + 1e-6))

        lam = float(self.lambda_reward)
        r_exist = float(-lam * (2.0 * ALPHA_R + 1.0))
        r_goal = float(R_GOAL if reached_goal else 0.0)

        if collided:
            reward = float((1.0 - lam) * R_COLLISION)
        elif reached_goal:
            reward = float(R_GOAL)
        else:
            reward = float(lam * r_pf + (1.0 - lam) * r_oa + r_exist)

        self.reward = reward
        terminated = bool(collided or reached_goal)

        info = {
            "r_pf": float(r_pf),
            "r_oa": float(r_oa),
            "r_exist": float(r_exist),
            "r_goal": float(r_goal),
            "lambda_reward": float(lam),
            "cross_track_error": float(self.tgt),
            "heading_error": float(self.heading_error),
            "distance_to_goal": float(self.distance_to_goal),
            "cos_chi": float(cos_chi),
            "u_norm": float(u_norm),
            "collision": bool(collided),
            "speed_mps": float(self.speed_mps),
            "rpm": float(rpm),
            "rudder_deg": float((rudder / 100.0) * MAX_RUD_ANGLE),
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
        action = np.array([0.0, 1.0], dtype=np.float32)
        obs, rew, term, _, _ = env.step(action)
        total_reward += rew
        if term:
            print(f"Elapsed time: {env.elapsed_time:.1f}s, Reward: {total_reward:0.2f}")
            env.close()
            pygame.quit()
            break
