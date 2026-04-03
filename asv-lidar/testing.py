import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import pygame
import pygame.freetype
from ship_model import ShipModel, THRUST_COEF, DRAG_COEF, VESSEL_LENGTH, VESSEL_WIDTH, HULL_MARGIN, HULL_FORWARD_SHIFT
from asv_lidar import Lidar, LIDAR_RANGE, LIDAR_BEAMS
from test_run import TestCase
from images import BOAT_ICON
import cv2
import json
import os
from datetime import datetime

RENDER_SCALE = 25
TEST_CASE = None
START_X = 9
START_Y = 2

# System parameters
UPDATE_RATE = 0.1   # 10 Hz
RENDER_FPS = 10
MAP_WIDTH = 10
MAP_HEIGHT = 25
MAX_OBS = 1

# Reward shaping parameters
LAMBDA_REWARD = 0.5
GAMMA_E = 0.05
GAMMA_THETA = 4.0
GAMMA_X = 0.005
EPSILON_X = 1.0
ALPHA_R = 0.1
R_COLLISION = -2000.0

# Speed control (rpm)
RPM_MIN = 0
RPM_MAX = 24
U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * RPM_MAX)
print(f"Estimated U_MAX = {U_MAX:.3f} m/s")
MAX_IN = 1
MIN_IN = -1

# Actions
PORT = 0
CENTER = 1
STBD = 2
rudder_action = {
    PORT: -25,
    CENTER: 0,
    STBD: 25
}

# ----------------------------
# Test scenarios
# ----------------------------
TEST_SCENARIOS = {
    "straight_accel": {
        "rudder_cmd": 0.0,      # normalized [-1,1]
        "throttle_cmd": 1.0,    # normalized [-1,1]
        "duration_s": 30.0,
        "record_video": False,
    },
    "turning_circle_port": {
        "rudder_cmd": -0.30,    # with current mapping *100 => -30 deg
        "throttle_cmd": 1.0,
        "duration_s": 60.0,
        "record_video": True,
    },
    "turning_circle_stbd": {
        "rudder_cmd": 0.30,     # +30 deg
        "throttle_cmd": 1.0,
        "duration_s": 60.0,
        "record_video": True,
    },
}

class ASVLidarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.map_width = MAP_WIDTH
        self.map_height = MAP_HEIGHT

        self.asv_path = []

        pygame.init()
        self.render_mode = render_mode
        self.world_size = (self.map_width, self.map_height)
        self.render_scale = RENDER_SCALE
        self.window_size = (int(self.map_width * self.render_scale),
                            int(self.map_height * self.render_scale))

        self.icon = None
        self.fps_clock = pygame.time.Clock()

        self.display = None
        self.surface = None
        self.status = None
        if render_mode in self.metadata["render_modes"]:
            self.surface = pygame.Surface(self.window_size)
            self.status = pygame.freetype.SysFont(pygame.font.get_default_font(), size=10)

        # State
        self.elapsed_time = 0.0
        self.tgt_x = 0
        self.tgt_y = 0
        self.tgt = 0
        self.asv_y = 0
        self.asv_x = 0
        self.asv_h = 0
        self.asv_w = 0
        self.angle_diff = 0
        self.prev_x = None
        self.prev_y = None
        self.speed_mps = 0.0

        self.model = ShipModel()
        self.scenario = TestCase()

        self.observation_space = Dict(
            {
                "lidar": Box(low=0, high=LIDAR_RANGE, shape=(LIDAR_BEAMS,), dtype=np.float32),
                "pos": Box(low=np.array([0, 0]), high=np.array(self.world_size), shape=(2,), dtype=np.float32),
                "hdg": Box(low=0, high=360, shape=(1,), dtype=np.float32),
                "dhdg": Box(low=-360, high=360, shape=(1,), dtype=np.float32),
                "speed": Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "tgt": Box(low=-50, high=50, shape=(1,), dtype=np.float32),
                "target_heading": Box(low=-180, high=180, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.lidar = Lidar()
        self.max_obs = MAX_OBS

        self.map_border = [
            [(0, 0), (0, self.map_height), (0, 0), (0, self.map_height)],
            [(0, self.map_height), (self.map_width, self.map_height), (0, self.map_height), (self.map_width, self.map_height)],
            [(self.map_width, self.map_height), (self.map_width, 0), (self.map_width, self.map_height), (self.map_width, 0)],
            [(0, 0), (self.map_width, 0), (0, 0), (self.map_width, 0)],
        ]

        self.record_video = True
        self.video_writer = None
        self.frame_size = self.window_size
        self.video_fps = RENDER_FPS

        self.test_case = TEST_CASE

    def _get_obs(self):
        return {
            "lidar": self.lidar.ranges.astype(np.float32),
            "pos": np.array([self.asv_x, self.asv_y], dtype=np.float32),
            "hdg": np.array([self.asv_h], dtype=np.float32),
            "dhdg": np.array([self.asv_w], dtype=np.float32),
            "speed": np.array([self.speed_mps], dtype=np.float32),
            "tgt": np.array([self.tgt], dtype=np.float32),
            "target_heading": np.array([self.angle_diff], dtype=np.float32),
        }

    def _hull_polygon_world(self):
        L = VESSEL_LENGTH + 2 * HULL_MARGIN
        W = VESSEL_WIDTH + 2 * HULL_MARGIN
        shift = HULL_FORWARD_SHIFT

        half_L = 0.5 * L
        half_W = 0.5 * W

        h = np.radians(float(self.asv_h))
        sin_h = np.sin(h)
        cos_h = np.cos(h)

        local = [
            (+half_L + shift, +half_W),
            (+half_L + shift, -half_W),
            (-half_L + shift, -half_W),
            (-half_L + shift, +half_W),
        ]

        cx = float(self.asv_x)
        cy = float(self.asv_y)

        poly = []
        for x_forward, y_left in local:
            x = cx + x_forward * sin_h - y_left * cos_h
            y = cy + x_forward * cos_h + y_left * sin_h
            poly.append((x, y))
        return poly

    def _polys_intersect_sat(self, polyA, polyB):
        def project(poly, ax, ay):
            dots = [p[0] * ax + p[1] * ay for p in poly]
            return min(dots), max(dots)

        for poly in (polyA, polyB):
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                ax = -(y2 - y1)
                ay = (x2 - x1)

                minA, maxA = project(polyA, ax, ay)
                minB, maxB = project(polyB, ax, ay)

                if maxA < minB or maxB < minA:
                    return False
        return True

    def _check_collision_geom(self):
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

    def _generate_path(self, start_x, start_y, goal_x, goal_y):
        path_length = max(2, int(np.hypot(abs(goal_x - start_x), abs(goal_y - start_y))))
        path_x = np.round(np.linspace(start_x, goal_x, path_length)).astype(int)
        path_y = np.round(np.linspace(start_y, goal_y, path_length)).astype(int)
        return np.column_stack((path_x, path_y))

    def _generate_obstacles(self, num_obs, test_case=None):
        obstacles = []
        if test_case is None:
            for _ in range(num_obs):
                x = np.random.randint(1, self.map_width - 1)
                y = np.random.randint(1, self.map_height - 1)
                if np.linalg.norm([x - self.start_x, y - self.start_y]) > 1 and \
                   np.linalg.norm([x - self.goal_x, y - self.goal_y]) > 1:
                    obstacles.append([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
        else:
            obstacles = self.scenario.obstacles(test_case=test_case)
        return obstacles

    def _calculate_angle(self, asv_x, asv_y, heading, goal_x, goal_y):
        dx = goal_x - asv_x
        dy = goal_y - asv_y
        target_angle = np.degrees(np.arctan2(dx, dy))
        return (target_angle - heading + 180) % 360 - 180

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.elapsed_time = 0.0
        self.asv_h = 0.0
        self.asv_w = 0.0
        self.tgt = 0.0
        self.angle_diff = 0.0

        self.model = ShipModel()
        self.model._v = 0.0
        self.lidar.reset()

        if self.test_case is None:
            self.start_x = START_X
            self.start_y = START_Y
            self.goal_x = 1
            self.goal_y = 20

        self.asv_x = self.start_x
        self.asv_y = self.start_y

        self.prev_x = float(self.asv_x)
        self.prev_y = float(self.asv_y)
        self.speed_mps = 0.0

        self.path = self._generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        # for repeatable dynamics tests, keep zero obstacles
        self.num_obs = 0
        self.obstacles = []

        self.asv_path = [(self.asv_x, self.asv_y)]
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        self.reward = 0

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

        return self._get_obs(), {}

    def check_done(self, position):
        if self._check_collision_geom():
            return True
        return False

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], MIN_IN, MAX_IN))
        throttle_cmd = float(np.clip(action[1], MIN_IN, MAX_IN))

        # IMPORTANT:
        # if you want real degrees here, use 30 not 100.
        # With action[0]=0.30 -> 30 deg
        rudder = rudder_cmd * 100

        rpm = (throttle_cmd - MIN_IN) * ((RPM_MAX - RPM_MIN) / (MAX_IN - MIN_IN)) + RPM_MIN

        x_prev = float(self.asv_x)
        y_prev = float(self.asv_y)

        dx, dy, h, w = self.model.update(rpm, rudder, UPDATE_RATE)
        self.asv_x += dx
        self.asv_y += dy
        self.asv_h = h
        self.asv_w = w

        dx_pos = float(self.asv_x) - x_prev
        dy_pos = float(self.asv_y) - y_prev
        self.speed_mps = float(np.sqrt(dx_pos * dx_pos + dy_pos * dy_pos) / float(UPDATE_RATE))

        asv_pos = np.array([self.asv_x, self.asv_y])
        distance = np.linalg.norm(self.path - asv_pos, axis=1)
        self.tgt = np.min(distance)

        closest_idx = np.argmin(distance)
        self.tgt_x, self.tgt_y = self.path[closest_idx]

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)
        self.angle_diff = self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y)

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

        self.asv_path.append((self.asv_x, self.asv_y))
        self.distance_to_goal = np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y])

        collided = bool(self._check_collision_geom())
        reached_goal = bool(self.distance_to_goal <= VESSEL_LENGTH)

        r_exist = -0.1
        angle_diff_rad = np.radians(self.angle_diff)
        r_heading = np.cos(angle_diff_rad)
        r_pf = np.exp(-0.05 * abs(self.tgt))

        lidar_list = self.lidar.ranges.astype(np.float32)
        r_oa = 0
        for i, dist in enumerate(lidar_list):
            theta = self.lidar.angles[i]
            weight = 1 / (1 + abs(theta))
            r_oa += weight / max(dist, 1)
        r_oa = -r_oa / len(lidar_list)

        r_goal = 50 if reached_goal else 0
        lambda_ = 0.5

        if collided:
            reward = -1000
        else:
            reward = lambda_ * r_pf + (1 - lambda_) * r_oa + r_heading + r_exist + r_goal

        terminated = self.check_done((self.asv_x, self.asv_y))
        info = {
            "distance_to_goal": float(self.distance_to_goal),
            "tgt": float(self.tgt),
            "collided": bool(collided),
            "reached_goal": bool(reached_goal),
            "rpm": float(rpm),
            "rudder_deg": float(rudder),
        }

        return self._get_obs(), reward, terminated, False, info

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

        bw = max(2, int(round(2)))
        W = self.window_size[0] - 1
        H = self.window_size[1] - 1
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (0, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, H), (W, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (W, 0), (W, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (W, 0), bw)

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

        os = pygame.transform.rotozoom(self.icon_scaled, -self.asv_h, 1)
        self.surface.blit(os, os.get_rect(center=scale_point((self.asv_x, self.asv_y))))
        ship_outline = self._hull_polygon_world()
        ship_outline_px = [scale_point(p) for p in ship_outline]
        pygame.draw.polygon(self.surface, (255, 0, 0), ship_outline_px, width=max(2, int(round(2))))

        if self.status is not None:
            status_surf_1, _ = self.status.render(
                f"{self.elapsed_time:005.1f}s  V:{self.speed_mps:0.2f}m/s  "
                f"HDG:{self.asv_h:+004.0f}  "
                f"DHDG:{self.asv_w:+0.2f}",
                (255, 255, 255),
                (0, 0, 0),
            )
            status_surf_2, _ = self.status.render(
                f"TGT:{self.tgt:+004.0f}  "
                f"TGT_HDG:{self.angle_diff:.2f}  "
                f"GOAL:{self.distance_to_goal:.2f}",
                (255, 255, 255),
                (0, 0, 0),
            )
            self.surface.blit(status_surf_1, (10, self.window_size[1] - 30))
            self.surface.blit(status_surf_2, (10, self.window_size[1] - 15))

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
        pygame.quit()


# ----------------------------
# Metric helpers
# ----------------------------
def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def cumulative_unwrapped_heading_deg(heading_samples_deg):
    if len(heading_samples_deg) == 0:
        return []
    out = [float(heading_samples_deg[0])]
    for i in range(1, len(heading_samples_deg)):
        d = wrap_to_180(float(heading_samples_deg[i]) - float(heading_samples_deg[i - 1]))
        out.append(out[-1] + d)
    return out

def circle_diameter_from_path(path_xy):
    if len(path_xy) < 10:
        return None
    pts = np.asarray(path_xy, dtype=np.float64)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(2.0 * np.mean(radii))

def first_crossing_time(samples_t, samples_val, threshold):
    for i in range(1, len(samples_val)):
        if samples_val[i - 1] < threshold <= samples_val[i]:
            return float(samples_t[i])
    return None

def first_abs_crossing_time(samples_t, samples_val, threshold):
    for i in range(1, len(samples_val)):
        if abs(samples_val[i - 1]) < threshold <= abs(samples_val[i]):
            return float(samples_t[i])
    return None


# ----------------------------
# Run one scenario and export JSON
# ----------------------------
def run_test_scenario(env, scenario_name, scenario_cfg, out_dir="test_results"):
    os.makedirs(out_dir, exist_ok=True)

    env.record_video = bool(scenario_cfg.get("record_video", False))
    obs, _ = env.reset()

    action = np.array(
        [scenario_cfg["rudder_cmd"], scenario_cfg["throttle_cmd"]],
        dtype=np.float32
    )
    duration_s = float(scenario_cfg["duration_s"])
    max_steps = int(np.ceil(duration_s / UPDATE_RATE))

    series = {
        "t_sec": [],
        "x_m": [],
        "y_m": [],
        "heading_deg": [],
        "yaw_rate_degps": [],
        "speed_mps": [],
        "rpm": [],
        "rudder_deg": [],
    }

    total_reward = 0.0
    terminated = False

    for _ in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break

        if terminated:
            break

        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew

        series["t_sec"].append(float(env.elapsed_time))
        series["x_m"].append(float(env.asv_x))
        series["y_m"].append(float(env.asv_y))
        series["heading_deg"].append(float(env.asv_h))
        series["yaw_rate_degps"].append(float(env.asv_w))
        series["speed_mps"].append(float(env.speed_mps))
        series["rpm"].append(float(info["rpm"]))
        series["rudder_deg"].append(float(info["rudder_deg"]))

        if term or trunc:
            terminated = True
            break

    # summary metrics
    peak_speed = float(max(series["speed_mps"])) if series["speed_mps"] else 0.0
    steady_speed = float(np.mean(series["speed_mps"][-20:])) if len(series["speed_mps"]) >= 20 else peak_speed
    threshold_90 = 0.9 * peak_speed if peak_speed > 0 else None
    t_90_speed = first_crossing_time(series["t_sec"], series["speed_mps"], threshold_90) if threshold_90 else None

    unwrapped_hdg = cumulative_unwrapped_heading_deg(series["heading_deg"])
    delta_hdg = [h - unwrapped_hdg[0] for h in unwrapped_hdg] if unwrapped_hdg else []
    t_90_turn = first_abs_crossing_time(series["t_sec"], delta_hdg, 90.0)
    t_180_turn = first_abs_crossing_time(series["t_sec"], delta_hdg, 180.0)

    steady_yaw_rate = float(np.mean(np.abs(series["yaw_rate_degps"][-20:]))) if len(series["yaw_rate_degps"]) >= 20 else 0.0
    turning_diameter = circle_diameter_from_path(list(zip(series["x_m"], series["y_m"])))

    result = {
        "meta": {
            "scenario_name": scenario_name,
            "timestamp": datetime.now().isoformat(),
            "UPDATE_RATE": UPDATE_RATE,
            "RPM_MIN": RPM_MIN,
            "RPM_MAX": RPM_MAX,
            "THRUST_COEF": float(THRUST_COEF),
            "DRAG_COEF": float(DRAG_COEF),
            "estimated_U_MAX_mps": float(U_MAX),
            "duration_cmd_s": duration_s,
            "rudder_cmd_norm": float(scenario_cfg["rudder_cmd"]),
            "throttle_cmd_norm": float(scenario_cfg["throttle_cmd"]),
        },
        "summary": {
            "elapsed_time_s": float(env.elapsed_time),
            "terminated_early": bool(terminated and env.elapsed_time < duration_s),
            "total_reward": float(total_reward),
            "peak_speed_mps": peak_speed,
            "steady_speed_mps": steady_speed,
            "speed_90pct_threshold_mps": float(threshold_90) if threshold_90 is not None else None,
            "time_to_90pct_peak_speed_s": t_90_speed,
            "steady_yaw_rate_degps": steady_yaw_rate,
            "time_to_90deg_heading_change_s": t_90_turn,
            "time_to_180deg_heading_change_s": t_180_turn,
            "turning_diameter_m": turning_diameter,
            "final_x_m": float(env.asv_x),
            "final_y_m": float(env.asv_y),
            "final_heading_deg": float(env.asv_h),
        },
        "timeseries": series,
    }

    out_path = os.path.join(out_dir, f"{scenario_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {out_path}")
    print("Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    env = ASVLidarEnv(render_mode="human")

    try:
        # scenario_name = "straight_accel"
        scenario_name = "turning_circle_port"
        # scenario_name = "turning_circle_stbd"

        scenario_cfg = TEST_SCENARIOS[scenario_name]
        run_test_scenario(env, scenario_name, scenario_cfg, out_dir="test_results")

    finally:
        env.close()

    # # Save path taken as image
    # path_surface = pygame.Surface((env.map_width, env.map_height))
    # path_surface.fill((255,255,255))

    # for i in range(1, len(env.asv_path)):
    #     pygame.draw.circle(path_surface, (0, 0, 200), env.asv_path[i], 3)
    
    # pygame.image.save(path_surface, "asv_path_result.png")
