"""
rl_env_reward_v2.py
===================

Derivative of ``rl_env.py`` with the reward function repaired per the
reward-analysis recommendations. Dynamics, observation space, action space and
scenario generation are byte-for-byte identical to ``rl_env.py`` so the 1M /
94% checkpoint (``models/sac_model_1M.zip``) remains load-compatible for
short-increment fine-tuning.

Only the reward changed. All changes stay inside the equations already printed
in the manuscript (Eq. 19-26), so the paper remains a valid reference:

  TIER 1 - fix the obstacle-proximity term r_oa (manuscript Eq. 23)
    * OA_DIST_FLOOR: 1.0 -> 0.3 m   (restore the near-field gradient; the old
      hard 1.0 m floor made every obstacle < 1 m produce an identical penalty)
    * angular weight w_i uses RADIANS, not degrees, so off-bow obstacles at
      30-60 deg still register (old degree kernel was ~0 beyond a few degrees)
    * OA_GAIN: rescale so a frontal contact is O(0.3-0.5), comparable to r_pf,
      making the lambda = 0.5 blend in Eq. 19 a genuine 50/50 balance

  TIER 2 - remove the two undocumented "repair" terms (not in the manuscript;
           staged experiments showed they degraded the baseline)
    * r_cte_recovery  (K_CTE_RECOVERY) -> removed
    * r_wrong_side    (K_WRONG_SIDE_ACTION) -> removed

The adaptive gamma_e (block_alpha blend) is KEPT: it is what gives the baseline
its tight clear-water tracking, and removing r_cte_recovery removes the only
term it previously contradicted. If strict Table-1 fidelity is wanted instead,
set GAMMA_E_CLEAR = GAMMA_E_BLOCKED = 0.05.

Every dense term is now logged in `info` (including the ones that were
previously invisible), so ablations are fully traceable.
"""
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

try:
    from asv_lidar import LIDAR_POOLING_MODE
except Exception:
    LIDAR_POOLING_MODE = "unknown"

try:
    from asv_lidar import FEASIBILITY_SAFE_WIDTH
except Exception:
    FEASIBILITY_SAFE_WIDTH = VESSEL_WIDTH + 2.0 * HULL_MARGIN

RENDER_SCALE = 25
TEST_CASE = None

# ---------------------------------------------------------------------------
# Focused local-planner training setup
# ---------------------------------------------------------------------------
UPDATE_RATE = 0.1
RENDER_FPS = 10
MAP_WIDTH = 10
MAP_HEIGHT = 25
MAX_OBS = 5

PATH_MODE = "straight"      # keep the nominal global/reference path straight
CURVE_PROB = 0.0
LOOKAHEAD_FRACTION = 0.25

# Fixed lambda for this practical local-planner run.
LAMBDA_MIN = 1e-4
LAMBDA_MAX = 1.0
DEFAULT_EVAL_LAMBDA = 0.5

# Reward parameters
GAMMA_E = 0.05
GAMMA_THETA = 4.0
GAMMA_X = 0.005
EPSILON_X = 1.0

# --- Obstacle-proximity term r_oa (manuscript Eq. 23) -- REWARD_V2 changes ---
# Distance floor: manuscript's epsilon only prevents blow-up for tiny ranges.
# The original code hard-clamped at 1.0 m, flattening the gradient across the
# whole danger zone (a 0.5 m obstacle scored identically to a 1.0 m one).
OA_DIST_FLOOR = 0.3
# Angular weight uses radians so the forward arc contributes, not just the few
# beams within a couple of degrees of dead-ahead. This is what lets an obstacle
# being passed at 30-90 deg off the bow push the vessel to the open side.
OA_ANGLE_IN_RADIANS = True
# Reference the inverse-distance penalty to the sensor's max range (the classic
# 1/d - 1/d0 potential-field form). This makes open water the zero point so the
# term does not add a large constant per-step offset (the naive mean of 1/d over
# all beams does), while staying an inverse-distance function of range as in
# Eq. 23. Beams at/near max range contribute ~0.
OA_INFLUENCE_DIST = float(LIDAR_RANGE)
# Scale so a frontal contact is O(0.5), comparable to r_pf, making the
# lambda=0.5 blend in Eq. 19 a genuine balance. Verified: open water -> 0.000,
# obstacle 1.0 m ahead -> -0.29, 0.3 m contact -> -1.00 (blended -0.50).
OA_GAIN = 8.0

R_COLLISION = -1000.0
R_TIMEOUT = -1000.0
R_GOAL = 50.0
R_EXIST = -0.5
U_REWARD_REF = 0.8

# Adaptive path-tracking sensitivity:
# - clear path: stricter tracking to avoid no-obstacle detours
# - blocked path: more tolerant CTE while bypassing obstacles
GAMMA_E_CLEAR = 0.20
GAMMA_E_BLOCKED = 0.05

# Explicit goal-region definition
GOAL_RADIUS = 0.5
GOAL_ALONG_DIST = 1.25
GOAL_CTE_RADIUS = 1.60

# Fixed-speed steering task. The action space remains 2D for compatibility,
# but throttle is ignored while FIXED_RPM=True.
RPM_MIN = 0
RPM_MAX = 24
CRUISE_RPM = 12.0
FIXED_RPM = False
U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * RPM_MAX)
MAX_IN = 1.0
MIN_IN = -1.0

MAX_EPISODE_STEPS = 700

RPM_STAGE = 1

if RPM_STAGE == 1:
    RPM_DELTA = 3.0       
    RPM_FLOOR = 9.0
    RPM_CEIL = 15.0
elif RPM_STAGE == 2:
    RPM_DELTA = 4.0       
    RPM_FLOOR = 8.0
    RPM_CEIL = 16.0
elif RPM_STAGE == 3:
    RPM_DELTA = 6.0      
    RPM_FLOOR = 6.0
    RPM_CEIL = 18.0
elif RPM_STAGE == 4:
    RPM_DELTA = 12.0      
    RPM_FLOOR = 0.0
    RPM_CEIL = 24.0

U_MIN_REWARD = 0.30
K_PROGRESS = 0.6
K_SLOW = 0.10
K_THRUST_DEV = 0.025

# Later speed-control stages, if needed:
# stage 1: RPM_DELTA=3.0, RPM_FLOOR=9.0, RPM_CEIL=15.0
# stage 2: RPM_DELTA=4.0, RPM_FLOOR=8.0, RPM_CEIL=16.0
# stage 3: RPM_DELTA=6.0, RPM_FLOOR=6.0, RPM_CEIL=18.0
# stage 4: RPM_DELTA=12.0, RPM_FLOOR=0.0, RPM_CEIL=24.0

# Observation LiDAR border-visibility mode.
# - "none": no borders/walls in LiDAR
# - "asymmetric": left pool edge visible, right pool edge invisible, far right wall visible
# - "both": both true pool borders visible
# - "mixed": randomize across the three cases above
OBS_BORDER_MODE = "none"
OBS_BORDER_P_NONE = 0.10
OBS_BORDER_P_ASYMMETRIC = 0.60
OBS_BORDER_P_BOTH = 0.30
RIGHT_WALL_OFFSET = 1.0
SOFT_BORDER_SAFE_DIST = 1.0
K_BORDER_SOFT = 0.40

# Obstacle curriculum: one obstacle near the path. Most episodes are slightly
# offset left/right so the policy learns a pass; some are exactly centered.
OBSTACLE_SIZE = 1.0
OBSTACLE_PATH_START_FRAC = 0.25
OBSTACLE_PATH_END_FRAC = 0.70
OBSTACLE_CENTER_PROB = 0.30
OBSTACLE_LATERAL_OFFSET_MIN = 0.25
OBSTACLE_LATERAL_OFFSET_MAX = 0.95

# Phase-1 training distribution: protect path/border behaviour first, then
# increase obstacle density in later phases. forced_num_obs overrides this.
TRAIN_OBS_COUNTS = [0, 1, 2, 3, 4]
# Conservative repair distribution for continuing from the 1M / 94% policy.
# Compared with the previous policy's path-protection curriculum, this gives
# more weight to 2-obstacle gate/pass-through layouts without making the whole
# run gate-only. Keep this moderate to avoid destroying the already-good policy.
TRAIN_OBS_PROBS = [0.15, 0.15, 0.45, 0.15, 0.10]

# Targeted side-choice repair distribution.
# The previous conservative run did not solve the hand-made failure case. This
# branch deliberately oversamples target-side/gate layouts, but keeps enough
# normal cases to reduce catastrophic forgetting. Train only short increments
# from the protected 1M/94% policy and evaluate before continuing.
TRAIN_SCENARIO_MODES = ["normal", "target_side", "field_repair", "gate", "offpath"]
TRAIN_SCENARIO_PROBS = [0.40, 0.35, 0.15, 0.05, 0.05]

# Gate generation parameters. Gap is measured between obstacle inner faces.
GATE_GAP_RANGE = (1.35, 2.25)
GATE_PATH_FRAC_RANGE = (0.35, 0.70)
GATE_CENTER_JITTER_ALONG = 0.45
GATE_CENTER_JITTER_LATERAL = 0.20
GATE_LATERAL_EXTRA = (0.05, 0.30)

# Field/failure-repair generator: path-relative perturbations of the layout
# where the old policy committed to the wrong/wide side even when a path-side
# corridor remained open.
FIELD_REPAIR_PATH_FRACS = (0.43, 0.66, 0.66)
FIELD_REPAIR_LATERALS = (0.0, +1.95, -1.95)
FIELD_REPAIR_FRAC_JITTER = 0.035
FIELD_REPAIR_LAT_JITTER = 0.25

# Target-side generator: creates layouts where the clearer/path-recovery side is
# deliberately passable.  This directly targets the failure mode where the actor
# committed to the left/wide side although the right/path-side corridor was open.
TARGET_SIDE_PATH_FRAC_RANGE = (0.38, 0.68)
TARGET_SIDE_CORRIDOR_OFFSET_RANGE = (0.65, 1.05)
TARGET_SIDE_BLOCKED_OFFSET_RANGE = (1.40, 2.30)
TARGET_SIDE_ALONG_JITTER = 0.45
TARGET_SIDE_LATERAL_JITTER = 0.20
TARGET_SIDE_RIGHT_PROB = 0.50   # mirror the layout so repair is not one-sided

# Very small reward additions for the targeted repair. These do not change the
# observation/action space, so the 1M checkpoint remains load-compatible.  They
# encourage recovery toward the path and discourage obviously wrong-side rudder
# only when path-recovery direction and side-clearance agree.
# REWARD_V2: r_cte_recovery and r_wrong_side removed from the reward (not in
# the manuscript; staged experiments degraded the baseline). Coefficients kept
# at 0.0 so the terms compute to 0 and the code path stays intact/reversible.
K_CTE_RECOVERY = 0.0
K_WRONG_SIDE_ACTION = 0.0
WRONG_SIDE_CTE_MIN = 0.25
WRONG_SIDE_DIFF_MIN = 1.00
WRONG_SIDE_FRONT_MIN = 1.20

# Off-path distractors teach the policy that not every visible obstacle should
# trigger a large bypass away from the path.
OFFPATH_LATERAL_MIN = 1.4
OFFPATH_LATERAL_MAX = 3.2

# Local lidar-based bypass cue. This is not global planning: it uses only the
# lidar sector ranges to choose the clearer side when the path ahead is blocked.
BLOCK_D_SAFE = 4.5
BLOCK_D_CRIT = 2.0
BLOCK_FRONT_DEG = 15.0
SIDE_ARC_MIN_DEG = 15.0
SIDE_ARC_MAX_DEG = 100.0
SIDE_CLEAR_TIE = 0.15
BYPASS_CTE = 0.5
K_LOCAL_TARGET = 1.0
K_CENTER_BLOCK = 0.0
K_BORDER = 0.0
# OBSTACLE_MODE = "single_near_path"
OBSTACLE_MODE = "multi_obs"


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
        # Policy observation LiDAR: obstacles + currently visible borders/walls.
        self.lidar_obs = Lidar()
        # Obstacle-only LiDAR: used for obstacle reward and front blockage.
        self.lidar_reward = Lidar()
        # Border-only LiDAR: used to keep local bypass side-choice boundary-aware.
        self.lidar_border_guard = Lidar()
        # Keep self.lidar pointing at the observation lidar so existing eval/render code still works.
        self.lidar = self.lidar_obs
        self.scenario = TestCase()
        self.obstacle_mode = OBSTACLE_MODE
        self.scenario_mode_used = "normal"

        self.elapsed_time = 0.0
        self.step_count = 0
        self.asv_x = 0.0
        self.asv_y = 0.0
        self.asv_h = 0.0
        self.asv_w = 0.0
        self.speed_mps = 0.0
        self.u_body = 0.0
        self.v_body = 0.0

        self.rudder = None
        self.rpm = None

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
        self.side_clearance_diff = 0.0
        self.block_alpha = 0.0
        self.local_target_cte = 0.0

        self.asv_path: List[Tuple[float, float]] = []
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.current_lambda = DEFAULT_EVAL_LAMBDA
        self.current_log10_lambda = float(np.log10(DEFAULT_EVAL_LAMBDA))
        self.obs_border_mode_used = "none"
        self.true_border_clearance = min(self.map_width, self.map_height)

        self.forced_num_obs = None

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
        # Fixed path-vs-obstacle reward balance for this practical local planner.
        # lambda_override is kept for API compatibility but intentionally ignored.
        self.current_lambda = float(DEFAULT_EVAL_LAMBDA)
        self.current_log10_lambda = float(np.log10(self.current_lambda))


    def _sample_obs_border_mode(self) -> None:
        if OBS_BORDER_MODE == "none":
            self.obs_border_mode_used = "none"
            return
        if OBS_BORDER_MODE == "asymmetric":
            self.obs_border_mode_used = "asymmetric"
            return
        if OBS_BORDER_MODE == "both":
            self.obs_border_mode_used = "both"
            return

        r = float(np.random.rand())
        if r < OBS_BORDER_P_NONE:
            self.obs_border_mode_used = "none"
        elif r < OBS_BORDER_P_NONE + OBS_BORDER_P_ASYMMETRIC:
            self.obs_border_mode_used = "asymmetric"
        else:
            self.obs_border_mode_used = "both"

    def _get_obs_lidar_border(self):
        if self.obs_border_mode_used == "none":
            return None
        if self.obs_border_mode_used == "both":
            return self.map_border

        # Asymmetric pool geometry: left pool edge is visible,
        # right pool edge is not visible, but a farther right wall is visible.
        # Keep the top wall visible as it exists in the pool view.
        right_x = self.map_width + RIGHT_WALL_OFFSET
        return [
            [(0.0, 0.0), (0.0, self.map_height)],
            [(0.0, self.map_height), (self.map_width, self.map_height)],
            [(right_x, 0.0), (right_x, self.map_height)],
        ]

    def _border_clearance_true(self) -> float:
        hull = self._hull_polygon_world()
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        return float(min(min(xs), self.map_width - max(xs), min(ys), self.map_height - max(ys)))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "lidar": self.lidar_obs.sector_closeness.astype(np.float32),
            "u": np.array([self.u_body], dtype=np.float32),
            "v": np.array([self.v_body], dtype=np.float32),
            "yaw_rate": np.array([self.asv_w], dtype=np.float32),
            "cross_track_error": np.array([self.cross_track_error], dtype=np.float32),
            "course_error": np.array([self.course_error], dtype=np.float32),
            "lookahead_course_error": np.array([self.lookahead_course_error], dtype=np.float32),
            "front_clearance": np.array([self.front_clearance], dtype=np.float32),
            "side_clearance_diff": np.array([self.side_clearance_diff], dtype=np.float32),
            "local_target_cte": np.array([self.local_target_cte], dtype=np.float32),
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
        margin_x = max(2.0, 0.25 * self.map_width)

        # 70%: vertical path
        if np.random.rand() < 0.7:
            x = float(np.random.uniform(margin_x, self.map_width - margin_x))
            return x, 2.0, x, self.map_height - 3.0

        # 30%: diagonal/slanted path
        start_x = float(np.random.uniform(margin_x, self.map_width - margin_x))
        goal_x = float(np.random.uniform(margin_x, self.map_width - margin_x))
        return start_x, 2.0, goal_x, self.map_height - 3.0

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
    
    def _path_frame_at_frac(self, frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return point, tangent, left-normal and arc-length at a path fraction."""
        if len(self.path_s) == 0 or len(self.path) < 2:
            p = np.array([self.start_x, self.start_y], dtype=np.float32)
            t = np.array([0.0, 1.0], dtype=np.float32)
            n = np.array([-1.0, 0.0], dtype=np.float32)
            return p, t, n, 0.0

        s_total = float(self.path_s[-1])
        s = float(np.clip(frac, 0.0, 1.0)) * s_total
        idx = int(np.searchsorted(self.path_s, s, side="left"))
        idx = int(np.clip(idx, 0, len(self.path) - 1))
        p = self.path[idx].astype(np.float32)
        t = self._path_tangent(idx).astype(np.float32)
        n = np.array([-t[1], t[0]], dtype=np.float32)
        return p, t, n, s

    def _make_box_rect(self, cx: float, cy: float, w: float, h: float) -> List[Tuple[float, float]]:
        """Axis-aligned rectangular obstacle helper."""
        hw = 0.5 * float(w)
        hh = 0.5 * float(h)
        return [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]

    def _box_inside_map(self, obs, margin_extra: float = 0.05) -> bool:
        xs = [p[0] for p in obs]
        ys = [p[1] for p in obs]
        return (
            min(xs) >= margin_extra
            and max(xs) <= self.map_width - margin_extra
            and min(ys) >= margin_extra
            and max(ys) <= self.map_height - margin_extra
        )

    def _append_if_valid(self, obstacles, obs, *, pad: float = 0.25) -> bool:
        if not self._box_inside_map(obs):
            return False
        if any(self._boxes_overlap(obs, existing, pad=pad) for existing in obstacles):
            return False
        # Keep obstacles away from exact start/goal regions.
        cx = float(np.mean([p[0] for p in obs]))
        cy = float(np.mean([p[1] for p in obs]))
        if np.hypot(cx - self.start_x, cy - self.start_y) < 2.5:
            return False
        if np.hypot(cx - self.goal_x, cy - self.goal_y) < 2.5:
            return False
        obstacles.append(obs)
        return True

    def _generate_target_side_obstacles(self, num_obs: int) -> List[List[Tuple[float, float]]]:
        """Generate the specific side-choice repair family.

        The layout leaves one path-side corridor intentionally usable and places
        one or more obstacles on the opposite/wide side.  The side is mirrored
        randomly.  The aim is to teach the actor that if the path-recovery side
        is also clearer, it should not commit to the opposite side.
        """
        num_obs = int(max(0, num_obs))
        if num_obs <= 0:
            return []
        if num_obs == 1:
            # For one obstacle, use a slightly off-centre near-path obstacle so
            # the policy must choose the clearer path-side rather than always
            # making a large bypass.
            obstacles: List[List[Tuple[float, float]]] = []
            for _ in range(80):
                frac = float(np.random.uniform(*TARGET_SIDE_PATH_FRAC_RANGE))
                center, tangent, normal_left, _ = self._path_frame_at_frac(frac)
                side_sign = -1.0 if np.random.rand() < TARGET_SIDE_RIGHT_PROB else +1.0
                lateral = side_sign * float(np.random.uniform(0.35, 0.75))
                along = float(np.random.uniform(-TARGET_SIDE_ALONG_JITTER, TARGET_SIDE_ALONG_JITTER))
                p = center + along * tangent + lateral * normal_left
                obs = self._make_box(float(p[0]), float(p[1]), float(np.random.uniform(0.85, 1.05) * OBSTACLE_SIZE))
                if self._append_if_valid(obstacles, obs, pad=0.20):
                    return obstacles
            return self._generate_normal_obstacles(num_obs)

        obstacles: List[List[Tuple[float, float]]] = []

        # Pick which side is the intended open/recovery side. For a vertical
        # path, normal_left is approximately negative x.  open_sign=-1 leaves a
        # right-side corridor; open_sign=+1 mirrors it.
        open_sign = -1.0 if np.random.rand() < TARGET_SIDE_RIGHT_PROB else +1.0
        blocked_sign = -open_sign

        # A lower near-path obstacle nudges the vessel but should still leave a
        # passable corridor on open_sign side.  Upper obstacles on the opposite
        # side create the tempting/wrong wide bypass.
        frac0 = float(np.random.uniform(*TARGET_SIDE_PATH_FRAC_RANGE))
        center0, tangent0, normal0, _ = self._path_frame_at_frac(frac0)

        candidates = []
        # Near-path / slightly blocking obstacle.
        lateral0 = blocked_sign * float(np.random.uniform(0.05, 0.35))
        p0 = center0 + float(np.random.uniform(-0.25, 0.25)) * tangent0 + lateral0 * normal0
        candidates.append((p0, float(np.random.uniform(0.85, 1.05) * OBSTACLE_SIZE)))

        # Two farther side obstacles, one on blocked side and one near centre but
        # shifted so the open side remains passable.
        for frac_shift, lat_range, sign in [
            (0.12, TARGET_SIDE_BLOCKED_OFFSET_RANGE, blocked_sign),
            (0.17, TARGET_SIDE_CORRIDOR_OFFSET_RANGE, open_sign),
        ]:
            frac = float(np.clip(frac0 + frac_shift + np.random.uniform(-0.035, 0.035), 0.28, 0.78))
            center, tangent, normal_left, _ = self._path_frame_at_frac(frac)
            lateral = sign * float(np.random.uniform(*lat_range))
            lateral += float(np.random.uniform(-TARGET_SIDE_LATERAL_JITTER, TARGET_SIDE_LATERAL_JITTER))
            along = float(np.random.uniform(-TARGET_SIDE_ALONG_JITTER, TARGET_SIDE_ALONG_JITTER))
            p = center + along * tangent + lateral * normal_left
            candidates.append((p, float(np.random.uniform(0.85, 1.05) * OBSTACLE_SIZE)))

        for p, size in candidates[:min(num_obs, len(candidates))]:
            obs = self._make_box(float(p[0]), float(p[1]), size)
            self._append_if_valid(obstacles, obs, pad=0.20)

        # Add extra obstacles as off-path distractors, not new centre blockers.
        tries = 0
        while len(obstacles) < num_obs and tries < 200:
            tries += 1
            candidate = self._generate_offpath_obstacle()
            if candidate:
                self._append_if_valid(obstacles, candidate[0], pad=0.25)

        return obstacles[:num_obs] if len(obstacles) >= num_obs else self._generate_gate_obstacles(num_obs)

    def _generate_gate_obstacles(self, num_obs: int) -> List[List[Tuple[float, float]]]:
        """Generate a pass-through gate/corridor layout.

        This is the key conservative repair scenario. It creates two obstacles
        on opposite sides of the path while leaving a passable gap around the
        reference path. Extra obstacles, if requested, are added as distractors.
        """
        num_obs = int(max(0, num_obs))
        if num_obs <= 0:
            return []
        if num_obs == 1:
            return self._generate_single_near_path_obstacle()

        obstacles: List[List[Tuple[float, float]]] = []
        for _ in range(80):
            frac = float(np.random.uniform(*GATE_PATH_FRAC_RANGE))
            center, tangent, normal_left, _ = self._path_frame_at_frac(frac)

            gap = float(np.random.uniform(*GATE_GAP_RANGE))
            size = float(OBSTACLE_SIZE)
            extra = float(np.random.uniform(*GATE_LATERAL_EXTRA))
            lateral_mag = 0.5 * gap + 0.5 * size + extra
            along_jitter = float(np.random.uniform(-GATE_CENTER_JITTER_ALONG, GATE_CENTER_JITTER_ALONG))
            lat_jitter = float(np.random.uniform(-GATE_CENTER_JITTER_LATERAL, GATE_CENTER_JITTER_LATERAL))

            left_c = center + along_jitter * tangent + (lateral_mag + lat_jitter) * normal_left
            right_c = center + along_jitter * tangent - (lateral_mag - lat_jitter) * normal_left

            left_obs = self._make_box(float(left_c[0]), float(left_c[1]), size)
            right_obs = self._make_box(float(right_c[0]), float(right_c[1]), size)

            trial: List[List[Tuple[float, float]]] = []
            if self._append_if_valid(trial, left_obs, pad=0.15) and self._append_if_valid(trial, right_obs, pad=0.15):
                obstacles = trial
                break

        # Fall back to normal generation if no valid gate was found.
        if len(obstacles) < min(2, num_obs):
            return self._generate_normal_obstacles(num_obs)

        # Add requested extra obstacles as off-path/distractor or normal near-path cases.
        tries = 0
        while len(obstacles) < num_obs and tries < 200:
            tries += 1
            if np.random.rand() < 0.60:
                candidate = self._generate_offpath_obstacle()
            else:
                candidate = self._generate_single_near_path_obstacle()
            if not candidate:
                continue
            self._append_if_valid(obstacles, candidate[0], pad=0.25)

        return obstacles[:num_obs]

    def _generate_field_repair_obstacles(self, num_obs: int) -> List[List[Tuple[float, float]]]:
        """Generate perturbations of the observed side-choice failure layout."""
        num_obs = int(max(0, num_obs))
        if num_obs <= 0:
            return []
        obstacles: List[List[Tuple[float, float]]] = []

        # For 1 obstacle, use a standard near-path case; for 2+ use the repair family.
        if num_obs == 1:
            return self._generate_single_near_path_obstacle()

        for k in range(min(num_obs, len(FIELD_REPAIR_PATH_FRACS))):
            base_frac = FIELD_REPAIR_PATH_FRACS[k]
            base_lat = FIELD_REPAIR_LATERALS[k]
            frac = float(np.clip(base_frac + np.random.uniform(-FIELD_REPAIR_FRAC_JITTER, FIELD_REPAIR_FRAC_JITTER), 0.25, 0.75))
            center, tangent, normal_left, _ = self._path_frame_at_frac(frac)
            lateral = float(base_lat + np.random.uniform(-FIELD_REPAIR_LAT_JITTER, FIELD_REPAIR_LAT_JITTER))
            along = float(np.random.uniform(-0.35, 0.35))
            p = center + along * tangent + lateral * normal_left
            size = float(np.random.uniform(0.85, 1.05) * OBSTACLE_SIZE)
            obs = self._make_box(float(p[0]), float(p[1]), size)
            self._append_if_valid(obstacles, obs, pad=0.20)

        tries = 0
        while len(obstacles) < num_obs and tries < 200:
            tries += 1
            candidate = self._generate_single_near_path_obstacle() if np.random.rand() < 0.5 else self._generate_offpath_obstacle()
            if candidate:
                self._append_if_valid(obstacles, candidate[0], pad=0.25)

        return obstacles[:num_obs] if len(obstacles) >= num_obs else self._generate_normal_obstacles(num_obs)

    def _generate_offpath_obstacle(self) -> List[List[Tuple[float, float]]]:
        """Generate one obstacle away from the path as a distractor."""
        if len(self.path) < 2:
            return []
        for _ in range(100):
            frac = float(np.random.uniform(OBSTACLE_PATH_START_FRAC, OBSTACLE_PATH_END_FRAC))
            center, tangent, normal_left, _ = self._path_frame_at_frac(frac)
            side = -1.0 if np.random.rand() < 0.5 else 1.0
            lateral = side * float(np.random.uniform(OFFPATH_LATERAL_MIN, OFFPATH_LATERAL_MAX))
            along = float(np.random.uniform(-0.50, 0.50))
            p = center + along * tangent + lateral * normal_left
            obs = self._make_box(float(p[0]), float(p[1]), OBSTACLE_SIZE)
            if self._box_inside_map(obs):
                return [obs]
        return []

    def _generate_normal_obstacles(self, num_obs: int) -> List[List[Tuple[float, float]]]:
        """Original multi-obstacle generator, kept for continuity."""
        num_obs = int(max(0, num_obs))
        if num_obs <= 0:
            return []

        obstacles: List[List[Tuple[float, float]]] = []
        max_tries = 300
        tries = 0

        while len(obstacles) < num_obs and tries < max_tries:
            tries += 1

            candidate = self._generate_single_near_path_obstacle()
            if not candidate:
                continue

            obs = candidate[0]

            # Reject if overlapping or too close to existing obstacles.
            if any(self._boxes_overlap(obs, existing, pad=0.25) for existing in obstacles):
                continue

            obstacles.append(obs)

        return obstacles

    def _generate_obstacles(self, num_obs: int, test_case: Optional[int] = None):
        if test_case is not None:
            raw = self.scenario.obstacles(test_case=test_case)
            return self._scale_case_obstacles(raw)

        num_obs = int(max(0, num_obs))
        if num_obs <= 0:
            self.scenario_mode_used = "none"
            return []

        # Conservative repair: mostly keep the old normal distribution, with
        # some gate/failure layouts mixed in.  This avoids the large distribution
        # shift that damaged the previous 200k continuation.
        probs = np.asarray(TRAIN_SCENARIO_PROBS, dtype=np.float64)
        probs = probs / np.sum(probs)
        mode = str(np.random.choice(TRAIN_SCENARIO_MODES, p=probs))
        self.scenario_mode_used = mode

        if mode == "target_side":
            return self._generate_target_side_obstacles(num_obs)
        if mode == "gate":
            return self._generate_gate_obstacles(num_obs)
        if mode == "field_repair":
            return self._generate_field_repair_obstacles(num_obs)
        if mode == "offpath":
            obstacles: List[List[Tuple[float, float]]] = []
            tries = 0
            while len(obstacles) < num_obs and tries < 300:
                tries += 1
                candidate = self._generate_offpath_obstacle()
                if candidate:
                    self._append_if_valid(obstacles, candidate[0], pad=0.25)
            if len(obstacles) >= num_obs:
                return obstacles
            return self._generate_normal_obstacles(num_obs)

        return self._generate_normal_obstacles(num_obs)

    def _boxes_overlap(self, a, b, pad: float = 0.15) -> bool:
        ax = [p[0] for p in a]
        ay = [p[1] for p in a]
        bx = [p[0] for p in b]
        by = [p[1] for p in b]

        return not (
            max(ax) + pad < min(bx) or
            max(bx) + pad < min(ax) or
            max(ay) + pad < min(by) or
            max(by) + pad < min(ay)
        )

    # ------------------------------------------------------------------
    # Local planner features from LiDAR only
    # ------------------------------------------------------------------
    def _update_local_planner_features(self) -> None:
        # Front blockage must be obstacle-only. This prevents visible walls/borders
        # from creating false "obstacle ahead" cues in no-obstacle episodes.
        obs_d = self.lidar_reward.sector_ranges.astype(np.float32)

        # Side choice should still be aware of true borders/geofence.
        # This prevents selecting a bypass side that has obstacle clearance but no wall clearance.
        border_d = self.lidar_border_guard.sector_ranges.astype(np.float32)
        side_d = np.minimum(obs_d, border_d)

        sector_angles = self.lidar_reward.sector_angles.astype(np.float32)
        front_mask = np.abs(sector_angles) <= BLOCK_FRONT_DEG
        left_mask = (sector_angles <= -SIDE_ARC_MIN_DEG) & (sector_angles >= -SIDE_ARC_MAX_DEG)
        right_mask = (sector_angles >= SIDE_ARC_MIN_DEG) & (sector_angles <= SIDE_ARC_MAX_DEG)

        def pctl(values: np.ndarray, mask: np.ndarray, p: float = 20.0) -> float:
            vals = values[mask]
            return float(np.percentile(vals, p)) if vals.size else float(LIDAR_RANGE)

        self.front_clearance = pctl(obs_d, front_mask, 10.0)
        self.left_clearance = pctl(side_d, left_mask, 20.0)
        self.right_clearance = pctl(side_d, right_mask, 20.0)
        self.side_clearance_diff = float(self.right_clearance - self.left_clearance)

        self.block_alpha = float(np.clip(
            (BLOCK_D_SAFE - self.front_clearance) / (BLOCK_D_SAFE - BLOCK_D_CRIT),
            0.0,
            1.0,
        ))

        if self.block_alpha <= 1e-6:
            self.local_target_cte = 0.0
            return

        # In this coordinate/sign convention, starboard/right of the path has negative CTE.
        if abs(self.right_clearance - self.left_clearance) < SIDE_CLEAR_TIE:
            side_cte_sign = -1.0
        elif self.right_clearance > self.left_clearance:
            side_cte_sign = -1.0
        else:
            side_cte_sign = +1.0

        self.local_target_cte = float(side_cte_sign * BYPASS_CTE * self.block_alpha)

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
        self.side_clearance_diff = 0.0
        self.block_alpha = 0.0
        self.local_target_cte = 0.0

        self.model = ShipModel()
        self.lidar_obs.reset()
        self.lidar_reward.reset()
        self.lidar_border_guard.reset()
        self.lidar = self.lidar_obs

        self._sample_lambda()
        self._sample_obs_border_mode()

        self.rudder = 0
        self.rpm = 0

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
            if self.forced_num_obs is not None:
                num_obs = int(self.forced_num_obs)
            else:
                probs = np.asarray(TRAIN_OBS_PROBS, dtype=np.float64)
                probs = probs / np.sum(probs)
                num_obs = int(np.random.choice(TRAIN_OBS_COUNTS, p=probs))
        else:
            num_obs = 0

        self.num_obs = int(num_obs)
        self.obstacles = self._generate_obstacles(num_obs, self.test_case)

        self.asv_path = [(self.asv_x, self.asv_y)]
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))

        self.lidar_obs.scan(
            (self.asv_x, self.asv_y),
            self.asv_h,
            obstacles=self.obstacles,
            map_border=self._get_obs_lidar_border(),
        )
        self.lidar_reward.scan(
            (self.asv_x, self.asv_y),
            self.asv_h,
            obstacles=self.obstacles,
            map_border=None,
        )
        self.lidar_border_guard.scan(
            (self.asv_x, self.asv_y),
            self.asv_h,
            obstacles=None,
            map_border=self.map_border,
        )
        self.lidar = self.lidar_obs

        self.true_border_clearance = self._border_clearance_true()
        self._update_path_relative_states(course_deg=self.asv_h)
        self._update_local_planner_features()

        if self.render_mode in self.metadata["render_modes"]:
            self.render()
        return self._get_obs(), {}

    def _reached_goal_region(self) -> bool:
        # Circular goal acceptance region.
        if self.distance_to_goal <= GOAL_RADIUS:
            return True

        # Finish-capsule acceptance near the end of the path. This catches cases
        # where the ASV has reached the goal region after avoidance but does not
        # pass exactly through the point goal.
        if len(self.path_s) > 0:
            s_here = float(self.path_s[self.closest_idx])
            s_end = float(self.path_s[-1])
            near_end_along_path = (s_end - s_here) <= GOAL_ALONG_DIST
            near_end_lateral = abs(float(self.cross_track_error)) <= GOAL_CTE_RADIUS
            if near_end_along_path and near_end_lateral:
                return True

        return False

    def check_done(self) -> bool:
        if self._check_collision_geom():
            return True
        if self._reached_goal_region():
            return True
        return False

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], MIN_IN, MAX_IN))
        throttle_cmd = float(np.clip(action[1], MIN_IN, MAX_IN))
        rudder = rudder_cmd * 100.0

        if FIXED_RPM:
            rpm = CRUISE_RPM
        else:
            rpm = float(np.clip(CRUISE_RPM + RPM_DELTA * throttle_cmd, RPM_FLOOR, RPM_CEIL))
        
        self.rudder = rudder
        self.rpm = rpm

        d_prev = float(self.distance_to_goal)
        ye_prev = abs(float(self.cross_track_error))
        x_prev = float(self.asv_x)
        y_prev = float(self.asv_y)

        # dynamic update
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

        self.lidar_obs.scan(
            (self.asv_x, self.asv_y),
            self.asv_h,
            obstacles=self.obstacles,
            map_border=self._get_obs_lidar_border(),
        )
        self.lidar_reward.scan(
            (self.asv_x, self.asv_y),
            self.asv_h,
            obstacles=self.obstacles,
            map_border=None,
        )
        self.lidar_border_guard.scan(
            (self.asv_x, self.asv_y),
            self.asv_h,
            obstacles=None,
            map_border=self.map_border,
        )
        self.lidar = self.lidar_obs
        self._update_path_relative_states(course_deg=course_deg)
        self._update_local_planner_features()
        self.asv_path.append((self.asv_x, self.asv_y))
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))

        # collision/goal flag
        collided = bool(self._check_collision_geom())
        reached_goal = bool(self._reached_goal_region())

        ye = abs(float(self.cross_track_error))
        U_norm = float(np.clip(self.speed_mps / U_MAX, 0.0, 1.5))

        # Simple old-style path reward with heading alignment, but using the current richer observation.
        # Use a speed gate so sitting still on the path is not profitable.
        u_gate = float(np.clip(max(self.u_body, 0.0) / U_REWARD_REF, 0.0, 1.0))
        gamma_e_eff = (
            (1.0 - float(self.block_alpha)) * GAMMA_E_CLEAR
            + float(self.block_alpha) * GAMMA_E_BLOCKED
        )
        r_pf = float(math.exp(-gamma_e_eff * ye))
        heading_err_deg = 0.7 * float(self.lookahead_course_error) + 0.3 * float(self.course_error)
        r_heading = float(math.cos(math.radians(heading_err_deg)))

        # Obstacle proximity (manuscript Eq. 23), REWARD_V2:
        #  - radian angular weight so the forward arc contributes (not just a
        #    few dead-ahead beams),
        #  - distance floor OA_DIST_FLOOR (0.3 m) instead of 1.0 m so the
        #    near-field gradient survives,
        #  - OA_GAIN so a frontal contact is comparable to r_pf.
        lidar_d = self.lidar_reward.ranges.astype(np.float32)
        lidar_angles_deg = self.lidar_reward.angles.astype(np.float32)
        angle_metric = np.radians(lidar_angles_deg) if OA_ANGLE_IN_RADIANS else lidar_angles_deg
        w_raw = 1.0 / (1.0 + np.abs(angle_metric))
        # inverse-distance proximity referenced to sensor max range (>=0)
        inv_prox = np.maximum(
            0.0, 1.0 / np.maximum(lidar_d, OA_DIST_FLOOR) - 1.0 / OA_INFLUENCE_DIST
        )
        r_oa = -OA_GAIN * float(np.sum(w_raw * inv_prox) / max(len(lidar_d), 1))

        # Keep local-planner shaping terms available for debug, but remove them from the reward for now.
        r_local = 0.0
        r_center = 0.0
        r_border = 0.0

        sector_d = self.lidar_reward.sector_ranges.astype(np.float32)
        pen = 1.0 / (GAMMA_X * (np.maximum(sector_d, EPSILON_X) ** 2))

        self.true_border_clearance = self._border_clearance_true()
        r_border = -float(K_BORDER_SOFT * max(0.0, 1.0 - self.true_border_clearance / SOFT_BORDER_SAFE_DIST) ** 2)

        # step penalty
        r_exist = R_EXIST

        # progress reward - keep moving forward
        r_progress = K_PROGRESS * (d_prev - self.distance_to_goal)

        # low-speed penalty
        r_slow = -(K_SLOW * max(0.0, U_MIN_REWARD - max(self.u_body, 0.0)))

        # stay near cruise RPM
        r_thrust = -float(K_THRUST_DEV * abs(rpm - CRUISE_RPM) / max(RPM_DELTA, 1e-6))

        # REWARD_V2: r_cte_recovery and r_wrong_side removed (not in manuscript;
        # degraded the baseline in staged experiments). Kept as explicit zeros
        # so the info dict / any downstream reader stays stable.
        r_cte_recovery = 0.0
        r_wrong_side = 0.0

        if collided:
            reward = float(R_COLLISION)
        else:
            reward = float(
                self.current_lambda * (u_gate * r_pf)      # path following, Eq. 20-21
                + (1.0 - self.current_lambda) * r_oa       # obstacle proximity, Eq. 23
                + 0.35 * u_gate * r_heading                # heading alignment, Eq. 22
                + r_exist                                  # living penalty, Eq. 25
                + r_border                                 # border clearance, Eq. 24
                + r_progress                               # progress, Eq. 26
                + r_slow                                   # low-speed penalty, Eq. 26
                + r_thrust                                 # cruise-RPM regularizer, Eq. 26
            )
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
            "r_pf": float(r_pf),
            "r_heading": float(r_heading),
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
            "side_clearance_diff": float(self.side_clearance_diff),
            "min_lidar": float(np.min(self.lidar_obs.ranges)) if len(self.lidar_obs.ranges) > 0 else float("inf"),
            "min_lidar_reward": float(np.min(self.lidar_reward.ranges)) if len(self.lidar_reward.ranges) > 0 else float("inf"),
            "min_sector_range": float(np.min(self.lidar_obs.sector_ranges.astype(np.float32))),
            "min_sector_range_reward": float(np.min(sector_d)),
            "p10_sector_range": float(np.percentile(sector_d, 10)),
            "mean_sector_pen": float(np.mean(pen)),
            "rpm": float(rpm),
            "rudder_deg": float(rudder_cmd * 40.0),
            "distance_to_goal": float(self.distance_to_goal),
            "collided": bool(collided),
            "reached_goal": bool(reached_goal),
            "timeout": bool(truncated),
            "path_mode": self.path_mode_used,
            "obs_border_mode": self.obs_border_mode_used,
            "scenario_mode": str(getattr(self, "scenario_mode_used", "normal")),
            "lidar_pooling_mode": str(LIDAR_POOLING_MODE),
            "feasibility_safe_width": float(FEASIBILITY_SAFE_WIDTH),
            "true_border_clearance": float(self.true_border_clearance),
            "goal_radius": float(GOAL_RADIUS),
            "goal_along_dist": float(GOAL_ALONG_DIST),
            "goal_cte_radius": float(GOAL_CTE_RADIUS),
            "gamma_e_eff": float(gamma_e_eff),
            "r_progress": float(r_progress),
            "r_slow": float(r_slow),
            "r_thrust": float(r_thrust),
            # REWARD_V2: log every dense term, including the now-zeroed repairs,
            # and the weighted contributions actually summed into the reward.
            "r_cte_recovery": float(r_cte_recovery),
            "r_wrong_side": float(r_wrong_side),
            "r_pf_weighted": float(self.current_lambda * u_gate * r_pf),
            "r_oa_weighted": float((1.0 - self.current_lambda) * r_oa),
            "r_heading_weighted": float(0.35 * u_gate * r_heading),
            "u_gate": float(u_gate),
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
                f"cte:{self.cross_track_error:+0.2f} tgt:{self.local_target_cte:+0.2f} front:{self.front_clearance:0.1f} L/R:{self.left_clearance:0.1f}/{self.right_clearance:0.1f} bc:{self.true_border_clearance:0.2f}",
                (255, 255, 255), (0, 0, 0),
            )
            status_line_3, _ = self.status.render(
                f"rud:{self.rudder:+0.2f}   thr:{self.rpm:+0.2f}",
                (255, 255, 255), (0, 0, 0)
            )
            self.surface.blit(status_line_1, (10, self.window_size[1] - 28))
            self.surface.blit(status_line_2, (10, self.window_size[1] - 14))
            self.surface.blit(status_line_3, (10, self.window_size[1] - 600))
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
    env = ASVLidarEnv(render_mode="human", map_width=10, map_height=25, path_mode="straight", test_case=None)
    obs, _ = env.reset()
    while True:
        action = np.array([-0.2, 0.0], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)
        if term or trunc:
            print(f"Done: reward={rew:.2f} info={info}")
            env.close()
            pygame.quit()
            break
