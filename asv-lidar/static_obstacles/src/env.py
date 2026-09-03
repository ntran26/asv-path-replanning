"""Gymnasium environment: ASV path following with static obstacle avoidance.

A 3-DOF vessel starts at one end of a rectangular basin and has to reach a goal
at the other end while staying near a straight reference path and avoiding
0-4 static obstacles.

Observation (Dict, 34 dims)
    lidar                    (25,) pooled sector closeness, 1 = touching
    u, v, yaw_rate           body velocities
    cross_track_error        signed, positive = left of the path
    course_error             deg
    lookahead_course_error   deg, lookahead at 25% of the path length
    front_clearance          m, from the obstacle-only LiDAR
    side_clearance_diff      m, right minus left
    local_target_cte         m, lateral bypass cue

Action (Box(2), [-1, 1])
    [rudder, throttle].  Rudder becomes a percentage command; throttle trims
    RPM around cruise.

Three LiDARs are simulated, because the observation and the reward need
different views of the world:
    lidar_obs            obstacles plus whichever walls OBS_BORDER_MODE exposes
    lidar_reward         obstacles only, so a visible wall cannot fake an
                         "obstacle ahead" cue in an empty episode
    lidar_border_guard   walls only, so the bypass side-choice stays inside the
                         basin
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

import config as cfg
from lidar import (
    FEASIBILITY_SAFE_WIDTH,
    LIDAR_POOLING_MODE,
    LIDAR_RANGE,
    LIDAR_SECTORS,
    Lidar,
)
from obstacles import ObstacleSampler
from path import ReferencePath, curved_points, straight_points
from scenarios import TestCase
from ship import HULL_MARGIN, MAX_RUD_ANGLE, VESSEL_LENGTH, VESSEL_WIDTH, ShipModel


class ASVLidarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        map_width: float = cfg.MAP_WIDTH,
        map_height: float = cfg.MAP_HEIGHT,
        max_obs: int = cfg.MAX_OBS,
        path_mode: str = cfg.PATH_MODE,
        curve_prob: float = cfg.CURVE_PROB,
        lookahead_fraction: float = cfg.LOOKAHEAD_FRACTION,
        test_case: Optional[int] = None,
        record_video: bool = False,
    ) -> None:
        super().__init__()
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.max_obs = int(max_obs)
        self.path_mode = str(path_mode)
        self.curve_prob = float(curve_prob)
        self.lookahead_fraction = float(lookahead_fraction)
        self.test_case = test_case
        self.render_mode = render_mode
        self.record_video = bool(record_video)

        self.model = ShipModel()
        self.lidar_obs = Lidar()
        self.lidar_reward = Lidar()
        self.lidar_border_guard = Lidar()
        self.lidar = self.lidar_obs          # what rendering and logging look at
        self.scenario = TestCase()

        # Overrides the sampled obstacle count when set (used by eval sweeps).
        self.forced_num_obs: Optional[int] = None

        self.map_border = [
            [(0.0, 0.0), (0.0, self.map_height)],
            [(0.0, self.map_height), (self.map_width, self.map_height)],
            [(self.map_width, self.map_height), (self.map_width, 0.0)],
            [(self.map_width, 0.0), (0.0, 0.0)],
        ]

        span = max(self.map_width, self.map_height)
        self.observation_space = DictSpace({
            "lidar": Box(0.0, 1.0, shape=(LIDAR_SECTORS,), dtype=np.float32),
            "u": Box(0.0, 5.0, shape=(1,), dtype=np.float32),
            "v": Box(-3.0, 3.0, shape=(1,), dtype=np.float32),
            "yaw_rate": Box(-180.0, 180.0, shape=(1,), dtype=np.float32),
            "cross_track_error": Box(-span, span, shape=(1,), dtype=np.float32),
            "course_error": Box(-180.0, 180.0, shape=(1,), dtype=np.float32),
            "lookahead_course_error": Box(-180.0, 180.0, shape=(1,), dtype=np.float32),
            "front_clearance": Box(0.0, LIDAR_RANGE, shape=(1,), dtype=np.float32),
            "side_clearance_diff": Box(-LIDAR_RANGE, LIDAR_RANGE, shape=(1,), dtype=np.float32),
            "local_target_cte": Box(-span, span, shape=(1,), dtype=np.float32),
        })
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.renderer = None
        self.scenario_mode_used = "normal"
        self.obs_border_mode_used = cfg.OBS_BORDER_MODE
        self.path_mode_used = self.path_mode
        self.current_lambda = cfg.DEFAULT_EVAL_LAMBDA
        self._clear_state()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _clear_state(self) -> None:
        self.step_count = 0
        self.elapsed_time = 0.0
        self.asv_x = self.asv_y = 0.0
        self.asv_h = self.asv_w = 0.0
        self.speed_mps = self.u_body = self.v_body = 0.0
        self.rudder = 0.0
        self.rpm = 0.0

        self.start_x = self.start_y = 0.0
        self.goal_x = self.goal_y = 0.0
        self.distance_to_goal = 0.0
        self.asv_path: List[Tuple[float, float]] = []
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.path: Optional[ReferencePath] = None

        self.cross_track_error = 0.0
        self.course_error = 0.0
        self.lookahead_course_error = 0.0
        self.closest_idx = 0
        self.lookahead_idx = 0
        self.tgt_x = self.tgt_y = 0.0
        self.lookahead_x = self.lookahead_y = 0.0

        self.front_clearance = LIDAR_RANGE
        self.left_clearance = LIDAR_RANGE
        self.right_clearance = LIDAR_RANGE
        self.side_clearance_diff = 0.0
        self.block_alpha = 0.0
        self.local_target_cte = 0.0
        self.true_border_clearance = min(self.map_width, self.map_height)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        `options["scenario"]` pins the layout to a saved dict with "start",
        "goal", "obstacles" and optionally "path" -- this is how the fixed
        holdout suite is replayed.  Otherwise the layout comes from `test_case`,
        or is sampled at random.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._clear_state()
        self.model.reset()
        self.lidar_obs.reset()
        self.lidar_reward.reset()
        self.lidar_border_guard.reset()

        self.current_lambda = cfg.DEFAULT_EVAL_LAMBDA
        self.obs_border_mode_used = self._sample_obs_border_mode()

        scenario = (options or {}).get("scenario")
        if scenario is not None:
            self._load_scenario(scenario)
        elif self.test_case is not None:
            self._load_test_case(self.test_case)
        else:
            self._sample_layout()

        self.asv_x, self.asv_y = self.start_x, self.start_y
        self.asv_path = [(self.asv_x, self.asv_y)]
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))

        self._scan_lidars()
        self.true_border_clearance = self._border_clearance(self.hull_polygon())
        self._update_path_errors(course_deg=self.asv_h)
        self._update_local_planner_features()

        self.render()
        return self._get_obs(), {}

    def _sample_layout(self) -> None:
        self.start_x, self.start_y, self.goal_x, self.goal_y = self._random_start_goal()
        self._build_path()

        if self.forced_num_obs is not None:
            num_obs = int(self.forced_num_obs)
        else:
            probs = np.asarray(cfg.TRAIN_OBS_PROBS, dtype=np.float64)
            num_obs = int(np.random.choice(cfg.TRAIN_OBS_COUNTS, p=probs / np.sum(probs)))
        self.obstacles = self.sample_obstacles(num_obs)

    def _load_test_case(self, test_case: int) -> None:
        sx, sy, gx, gy = self.scenario.position(test_case)
        self.start_x, self.start_y = self._scale_position(sx, sy)
        self.goal_x, self.goal_y = self._scale_position(gx, gy)
        self._build_path()
        self.obstacles = [
            [self._scale_position(px, py) for px, py in obs]
            for obs in self.scenario.obstacles(test_case)
        ]

    def _load_scenario(self, scenario: dict) -> None:
        self.start_x, self.start_y = (float(v) for v in scenario["start"])
        self.goal_x, self.goal_y = (float(v) for v in scenario["goal"])

        saved_path = scenario.get("path")
        if saved_path is not None and len(saved_path) >= 2:
            self.path = ReferencePath(saved_path, self.lookahead_fraction)
        else:
            self._build_path()

        self.obstacles = [
            [(float(x), float(y)) for x, y in obs]
            for obs in scenario.get("obstacles", [])
        ]

    def sample_obstacles(self, num_obs: int) -> List[List[Tuple[float, float]]]:
        """Draw a random obstacle layout around the current reference path."""
        sampler = ObstacleSampler(
            self.path, self.map_width, self.map_height,
            (self.start_x, self.start_y), (self.goal_x, self.goal_y),
        )
        layout = sampler.sample(num_obs)
        self.scenario_mode_used = sampler.mode_used
        return layout

    def _random_start_goal(self) -> Tuple[float, float, float, float]:
        margin_x = max(cfg.START_X_MARGIN_MIN, cfg.START_X_MARGIN_FRAC * self.map_width)
        goal_y = self.map_height - cfg.GOAL_Y_MARGIN

        if np.random.rand() < cfg.VERTICAL_PATH_PROB:
            x = float(np.random.uniform(margin_x, self.map_width - margin_x))
            return x, cfg.START_Y, x, goal_y

        start_x = float(np.random.uniform(margin_x, self.map_width - margin_x))
        goal_x = float(np.random.uniform(margin_x, self.map_width - margin_x))
        return start_x, cfg.START_Y, goal_x, goal_y

    def _build_path(self) -> None:
        if self.path_mode == "mixed":
            self.path_mode_used = "curve" if np.random.rand() < self.curve_prob else "straight"
        else:
            self.path_mode_used = self.path_mode

        args = (self.start_x, self.start_y, self.goal_x, self.goal_y)
        points = (curved_points(*args, self.map_width, self.map_height)
                  if self.path_mode_used == "curve" else straight_points(*args))
        self.path = ReferencePath(points, self.lookahead_fraction)

    def _scale_position(self, x: float, y: float) -> Tuple[float, float]:
        """Map a canonical 10 x 25 test-case coordinate onto the actual basin."""
        return float(x) * (self.map_width / 10.0), float(y) * (self.map_height / 25.0)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _sample_obs_border_mode(self) -> str:
        if cfg.OBS_BORDER_MODE != "mixed":
            return cfg.OBS_BORDER_MODE
        r = float(np.random.rand())
        if r < cfg.OBS_BORDER_P_NONE:
            return "none"
        if r < cfg.OBS_BORDER_P_NONE + cfg.OBS_BORDER_P_ASYMMETRIC:
            return "asymmetric"
        return "both"

    def _obs_lidar_border(self):
        if self.obs_border_mode_used == "none":
            return None
        if self.obs_border_mode_used == "both":
            return self.map_border

        # Asymmetric pool geometry: the left edge and the top wall are visible,
        # the right edge is not, but a wall further to starboard is.
        right_x = self.map_width + cfg.RIGHT_WALL_OFFSET
        return [
            [(0.0, 0.0), (0.0, self.map_height)],
            [(0.0, self.map_height), (self.map_width, self.map_height)],
            [(right_x, 0.0), (right_x, self.map_height)],
        ]

    def _scan_lidars(self) -> None:
        pos = (self.asv_x, self.asv_y)
        self.lidar_obs.scan(pos, self.asv_h, obstacles=self.obstacles, map_border=self._obs_lidar_border())
        self.lidar_reward.scan(pos, self.asv_h, obstacles=self.obstacles, map_border=None)
        self.lidar_border_guard.scan(pos, self.asv_h, obstacles=None, map_border=self.map_border)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        def scalar(value):
            return np.array([value], dtype=np.float32)

        return {
            "lidar": self.lidar_obs.sector_closeness.astype(np.float32),
            "u": scalar(self.u_body),
            "v": scalar(self.v_body),
            "yaw_rate": scalar(self.asv_w),
            "cross_track_error": scalar(self.cross_track_error),
            "course_error": scalar(self.course_error),
            "lookahead_course_error": scalar(self.lookahead_course_error),
            "front_clearance": scalar(self.front_clearance),
            "side_clearance_diff": scalar(self.side_clearance_diff),
            "local_target_cte": scalar(self.local_target_cte),
        }

    def _update_path_errors(self, course_deg: float) -> None:
        state = self.path.project(self.asv_x, self.asv_y, course_deg)
        self.closest_idx = state.closest_idx
        self.cross_track_error = state.cross_track_error
        self.course_error = state.course_error
        self.tgt_x, self.tgt_y = state.target
        self.lookahead_idx = state.lookahead_idx
        self.lookahead_x, self.lookahead_y = state.lookahead
        self.lookahead_course_error = state.lookahead_course_error

    def _update_local_planner_features(self) -> None:
        # Blockage ahead must come from obstacles only, otherwise a visible wall
        # reads as an obstacle in an empty episode.  Side choice, on the other
        # hand, has to respect the basin walls.
        obstacle_d = self.lidar_reward.sector_ranges.astype(np.float32)
        side_d = np.minimum(obstacle_d, self.lidar_border_guard.sector_ranges.astype(np.float32))
        angles = self.lidar_reward.sector_angles.astype(np.float32)

        ahead = np.abs(angles) <= cfg.BLOCK_FRONT_DEG
        to_port = (angles <= -cfg.SIDE_ARC_MIN_DEG) & (angles >= -cfg.SIDE_ARC_MAX_DEG)
        to_starboard = (angles >= cfg.SIDE_ARC_MIN_DEG) & (angles <= cfg.SIDE_ARC_MAX_DEG)

        def percentile(values, mask, p):
            selected = values[mask]
            return float(np.percentile(selected, p)) if selected.size else float(LIDAR_RANGE)

        self.front_clearance = percentile(obstacle_d, ahead, 10.0)
        self.left_clearance = percentile(side_d, to_port, 20.0)
        self.right_clearance = percentile(side_d, to_starboard, 20.0)
        self.side_clearance_diff = float(self.right_clearance - self.left_clearance)

        self.block_alpha = float(np.clip(
            (cfg.BLOCK_D_SAFE - self.front_clearance) / (cfg.BLOCK_D_SAFE - cfg.BLOCK_D_CRIT), 0.0, 1.0))

        if self.block_alpha <= 1e-6:
            self.local_target_cte = 0.0
            return

        # Starboard of the path is negative CTE here, so bypass to starboard
        # unless the port side is clearly more open.
        port_clearly_better = (self.left_clearance - self.right_clearance) >= cfg.SIDE_CLEAR_TIE
        sign = 1.0 if port_clearly_better else -1.0
        self.local_target_cte = float(sign * cfg.BYPASS_CTE * self.block_alpha)

    # ------------------------------------------------------------------
    # Collision geometry
    # ------------------------------------------------------------------
    def hull_polygon(self) -> List[Tuple[float, float]]:
        """Inflated vessel footprint in world coordinates."""
        half_l = 0.5 * (VESSEL_LENGTH + 2.0 * HULL_MARGIN)
        half_w = 0.5 * (VESSEL_WIDTH + 2.0 * HULL_MARGIN)
        h = math.radians(self.asv_h)
        sin_h, cos_h = math.sin(h), math.cos(h)
        return [
            (self.asv_x + fwd * sin_h - left * cos_h, self.asv_y + fwd * cos_h + left * sin_h)
            for fwd, left in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))
        ]

    def _border_clearance(self, hull) -> float:
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        return float(min(min(xs), self.map_width - max(xs), min(ys), self.map_height - max(ys)))

    def _hits_border(self, hull) -> bool:
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        return min(xs) < 0.0 or max(xs) > self.map_width or min(ys) < 0.0 or max(ys) > self.map_height

    def _collided(self, hull) -> bool:
        if self._hits_border(hull):
            return True

        hx0, hx1 = min(p[0] for p in hull), max(p[0] for p in hull)
        hy0, hy1 = min(p[1] for p in hull), max(p[1] for p in hull)
        for obs in self.obstacles:
            oxs = [p[0] for p in obs]
            oys = [p[1] for p in obs]
            # Cheap bounding-box reject before the exact separating-axis test.
            if hx1 < min(oxs) or max(oxs) < hx0 or hy1 < min(oys) or max(oys) < hy0:
                continue
            if _polygons_intersect(hull, obs):
                return True
        return False

    def hit_border(self) -> bool:
        """True while the hull is outside the basin.  Evaluation uses this to
        separate border collisions from obstacle collisions."""
        return self._hits_border(self.hull_polygon())

    def _reached_goal(self) -> bool:
        if self.distance_to_goal <= cfg.GOAL_RADIUS:
            return True
        # Capsule around the end of the path, so an episode that finishes beside
        # the goal point after an avoidance manoeuvre still counts.
        remaining = self.path.length - float(self.path.s[self.closest_idx])
        return remaining <= cfg.GOAL_ALONG_DIST and abs(self.cross_track_error) <= cfg.GOAL_CTE_RADIUS

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        self.elapsed_time += cfg.UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        self.rudder = rudder_cmd * 100.0
        self.rpm = cfg.CRUISE_RPM if cfg.FIXED_RPM else float(np.clip(
            cfg.CRUISE_RPM + cfg.RPM_DELTA * throttle_cmd, cfg.RPM_FLOOR, cfg.RPM_CEIL))

        distance_before = self.distance_to_goal
        cte_before = abs(self.cross_track_error)
        x_before, y_before = self.asv_x, self.asv_y

        dx, dy, heading, yaw_rate = self.model.update(self.rpm, self.rudder, cfg.UPDATE_RATE)
        self.asv_x += dx
        self.asv_y += dy
        self.asv_h = heading
        self.asv_w = yaw_rate
        self.u_body = self.model.u
        self.v_body = self.model.v

        moved_x = self.asv_x - x_before
        moved_y = self.asv_y - y_before
        self.speed_mps = float(math.hypot(moved_x, moved_y) / cfg.UPDATE_RATE)
        course_deg = math.degrees(math.atan2(moved_x, moved_y)) if self.speed_mps > 1e-6 else self.asv_h

        self._scan_lidars()
        self._update_path_errors(course_deg)
        self._update_local_planner_features()
        self.asv_path.append((self.asv_x, self.asv_y))
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))

        hull = self.hull_polygon()
        self.true_border_clearance = self._border_clearance(hull)
        collided = self._collided(hull)
        reached_goal = self._reached_goal()

        dense, terms = self._reward(rudder_cmd, cte_before, distance_before)
        if collided:
            reward = float(cfg.R_COLLISION)
        else:
            reward = dense + cfg.R_GOAL if reached_goal else dense

        terminated = collided or reached_goal
        self.step_count += 1
        truncated = self.step_count >= cfg.MAX_EPISODE_STEPS and not terminated
        if truncated:
            reward += cfg.R_TIMEOUT

        info = self._build_info(terms, reward, rudder_cmd, collided, reached_goal, truncated)
        self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def _reward(self, rudder_cmd: float, cte_before: float, distance_before: float):
        """Return (dense reward, per-term values for logging).

        Ten terms.  The path/heading pair is gated by speed so loitering on the
        path is not profitable, and the path-tracking sensitivity `gamma_e` is
        relaxed as the way ahead becomes blocked.
        """
        cte = abs(self.cross_track_error)
        u_gate = float(np.clip(max(self.u_body, 0.0) / cfg.U_REWARD_REF, 0.0, 1.0))

        gamma_e = (1.0 - self.block_alpha) * cfg.GAMMA_E_CLEAR + self.block_alpha * cfg.GAMMA_E_BLOCKED
        r_pf = math.exp(-gamma_e * cte)
        r_heading = math.cos(math.radians(0.7 * self.lookahead_course_error + 0.3 * self.course_error))

        # Raw beams give a smoother obstacle gradient than the pooled sectors.
        beam_d = self.lidar_reward.ranges.astype(np.float32)
        beam_w = 1.0 / (1.0 + np.abs(self.lidar_reward.angles.astype(np.float32)))
        r_oa = -float(np.sum(beam_w / np.maximum(beam_d, 1.0)) / max(len(beam_d), 1))

        wall_overlap = max(0.0, 1.0 - self.true_border_clearance / cfg.SOFT_BORDER_SAFE_DIST)
        r_border = -cfg.K_BORDER_SOFT * wall_overlap ** 2
        r_progress = cfg.K_PROGRESS * (distance_before - self.distance_to_goal)
        r_slow = -cfg.K_SLOW * max(0.0, cfg.U_MIN_REWARD - max(self.u_body, 0.0))
        r_thrust = -cfg.K_THRUST_DEV * abs(self.rpm - cfg.CRUISE_RPM) / cfg.RPM_DELTA
        r_cte_recovery = cfg.K_CTE_RECOVERY * (cte_before - cte)
        r_wrong_side = self._wrong_side_penalty(rudder_cmd)

        dense = float(
            self.current_lambda * (u_gate * r_pf)
            + (1.0 - self.current_lambda) * r_oa
            + cfg.W_HEADING * u_gate * r_heading
            + cfg.R_EXIST
            + r_border
            + r_progress
            + r_slow
            + r_thrust
            + r_cte_recovery
            + r_wrong_side
        )

        terms = {
            "r_pf": r_pf,
            "r_heading": r_heading,
            "r_oa": r_oa,
            "r_border": r_border,
            "r_exist": cfg.R_EXIST,
            "r_progress": r_progress,
            "r_slow": r_slow,
            "r_thrust": r_thrust,
            "r_cte_recovery": r_cte_recovery,
            "r_wrong_side": r_wrong_side,
            "gamma_e_eff": gamma_e,
        }
        return dense, terms

    def _wrong_side_penalty(self, rudder_cmd: float) -> float:
        """Penalise rudder that contradicts an unambiguously better side.

        Only fires when the path-recovery direction and the clearer side agree
        and the way ahead is not already tight.  Negative rudder turns to port.
        """
        if self.front_clearance <= cfg.WRONG_SIDE_FRONT_MIN:
            return 0.0

        diff = self.right_clearance - self.left_clearance
        starboard_favoured = self.cross_track_error > cfg.WRONG_SIDE_CTE_MIN and diff > cfg.WRONG_SIDE_DIFF_MIN
        port_favoured = self.cross_track_error < -cfg.WRONG_SIDE_CTE_MIN and diff < -cfg.WRONG_SIDE_DIFF_MIN

        if (starboard_favoured and rudder_cmd < 0.0) or (port_favoured and rudder_cmd > 0.0):
            return -cfg.K_WRONG_SIDE_ACTION * min(abs(rudder_cmd), 1.0)
        return 0.0

    def _build_info(self, terms, reward, rudder_cmd, collided, reached_goal, truncated) -> Dict:
        sector_d = self.lidar_reward.sector_ranges.astype(np.float32)
        sector_pen = 1.0 / (cfg.GAMMA_X * (np.maximum(sector_d, cfg.EPSILON_X) ** 2))

        info = {
            "lam": float(self.current_lambda),
            # r_local and r_center are retired shaping terms, kept at zero so the
            # log schema (and the CSV headers built from it) does not change.
            "r_local": 0.0,
            "r_center": 0.0,
            "reward": float(reward),
            "ye": float(abs(self.cross_track_error)),
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
            "min_lidar": float(np.min(self.lidar_obs.ranges)),
            "min_lidar_reward": float(np.min(self.lidar_reward.ranges)),
            "min_sector_range": float(np.min(self.lidar_obs.sector_ranges.astype(np.float32))),
            "min_sector_range_reward": float(np.min(sector_d)),
            "p10_sector_range": float(np.percentile(sector_d, 10)),
            "mean_sector_pen": float(np.mean(sector_pen)),
            "rpm": float(self.rpm),
            "rudder_deg": float(rudder_cmd * MAX_RUD_ANGLE),
            "distance_to_goal": float(self.distance_to_goal),
            "collided": bool(collided),
            "reached_goal": bool(reached_goal),
            "timeout": bool(truncated),
            "path_mode": self.path_mode_used,
            "obs_border_mode": self.obs_border_mode_used,
            "scenario_mode": self.scenario_mode_used,
            "lidar_pooling_mode": LIDAR_POOLING_MODE,
            "feasibility_safe_width": float(FEASIBILITY_SAFE_WIDTH),
            "true_border_clearance": float(self.true_border_clearance),
            "goal_radius": float(cfg.GOAL_RADIUS),
            "goal_along_dist": float(cfg.GOAL_ALONG_DIST),
            "goal_cte_radius": float(cfg.GOAL_CTE_RADIUS),
        }
        # r_cte_recovery and r_wrong_side are computed but deliberately left out
        # here, matching the original log schema.  Drop the filter to log them.
        info.update({k: float(v) for k, v in terms.items()
                     if k not in ("r_cte_recovery", "r_wrong_side")})
        return info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.renderer is None:
            from render import Renderer  # imported lazily: training needs no display
            self.renderer = Renderer(self.map_width, self.map_height, record_video=self.record_video)
        self.renderer.draw(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def _polygons_intersect(poly_a, poly_b) -> bool:
    """Separating-axis test for two convex polygons."""
    for poly in (poly_a, poly_b):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            axis_x, axis_y = -(y2 - y1), x2 - x1

            a = [p[0] * axis_x + p[1] * axis_y for p in poly_a]
            b = [p[0] * axis_x + p[1] * axis_y for p in poly_b]
            if max(a) < min(b) or max(b) < min(a):
                return False
    return True


if __name__ == "__main__":
    env = ASVLidarEnv(render_mode="human")
    env.reset()
    while True:
        _, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0], dtype=np.float32))
        if terminated or truncated:
            print(f"Done: reward={reward:.2f} info={info}")
            env.close()
            break
