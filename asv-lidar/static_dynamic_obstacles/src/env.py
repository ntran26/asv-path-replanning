"""Gymnasium environment: path following with static obstacles and one target vessel.

**Revision 2** — two-vessel encounters.  One dynamic target, `N_MAX_TARGETS`
configurable so a multi-vessel extension costs a retrain, not a redesign (S1).

Scope of this file in task 01
-----------------------------
Perception and observation.  Specifically:

* **Reward is a placeholder.**  `_reward` is sparse terminal payoff and nothing
  else.  02 owns the reward design -- six carried-over terms, five COLREGs
  terms, the Rule 9 precedence table and the mandatory scale audit -- and per
  D10 it is redesigned rather than patched, so no Paper 2 shaping term is
  carried across.
* **Target motion is constant-velocity and the spawn is a placeholder.**
  Constant velocity is decision D1 for training; reactive and non-compliant
  targets are evaluation-only and belong to 03.  `_sample_target` places a
  single head-on target beyond the sensor horizon purely so the perception path
  is exercised in situ -- 03 owns the real encounter geometry.
* **The corridor is a straight inset rectangle.**  03 owns variable width along
  the path, bends, and deliberately off-centre reference paths.  Until those
  land the boundary branch is an affine function of cross-track error and must
  not be ablated (01 §3.3).

What is fully built here is the perception path:

    raycast (obstacles only, aft mask, dropout, 1 m dead zone)
        -> gate against the boundary polygon
        -> cluster -> track -> Kalman -> static/dynamic split
        -> CPA/CRI -> encounter class (shared with 02)
        -> five-branch Dict observation

Collision and termination stay geometric and exact, as in Paper 2.  What the
policy *sees* and what *counts* as a collision are deliberately separate: the
policy sees a gated, pooled, noise-perturbed view, while termination uses true
hull geometry.

Study 2 degradation axes (04 §6) are constructor parameters, all nominal-zero:
pose drift, detection dropout, occlusion duration, velocity noise.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

import boundary_raycast as br
import constants as cfg
import tracking as trk
from asv_lidar import Lidar
from observation import ObservationBuilder, observation_space
from obstacles import ObstacleSampler
from path import ReferencePath, curved_points, straight_points
from ship import HULL_MARGIN, MAX_RUD_ANGLE, VESSEL_LENGTH, VESSEL_WIDTH, ShipModel


class TargetShip:
    """A COLREGs target vessel.

    Oriented hull rather than a circle (03 §3): required for ship-domain metrics
    and for correct aspect-angle computation in the encounter classifier.
    Constant velocity (D1); 03 replaces the motion model for the reactive and
    non-compliant evaluation strata.
    """

    def __init__(self, x: float, y: float, heading_deg: float, speed: float,
                 length: float = cfg.LOA, width: float = cfg.BREADTH) -> None:
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading_deg)
        self.speed = float(speed)
        self.length = float(length)
        self.width = float(width)

    @property
    def velocity(self) -> np.ndarray:
        a = math.radians(self.heading)
        return np.array([self.speed * math.sin(a), self.speed * math.cos(a)])

    def step(self, dt: float) -> None:
        v = self.velocity
        self.x += float(v[0]) * dt
        self.y += float(v[1]) * dt

    def hull(self) -> List[Tuple[float, float]]:
        half_l, half_w = 0.5 * self.length, 0.5 * self.width
        h = math.radians(self.heading)
        sin_h, cos_h = math.sin(h), math.cos(h)
        return [
            (self.x + fwd * sin_h - lat * cos_h, self.y + fwd * cos_h + lat * sin_h)
            for fwd, lat in ((half_l, half_w), (half_l, -half_w),
                             (-half_l, -half_w), (-half_l, half_w))
        ]


class ASVLidarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None, *,
                 map_width: float = cfg.MAP_WIDTH,
                 map_height: float = cfg.MAP_HEIGHT,
                 corridor_width: Optional[float] = None,
                 max_obs: int = cfg.MAX_OBS,
                 path_mode: str = cfg.PATH_MODE,
                 curve_prob: float = cfg.CURVE_PROB,
                 lookahead_fraction: float = cfg.LOOKAHEAD_FRACTION,
                 n_max_targets: int = cfg.N_MAX_TARGETS,
                 no_target_prob: float = cfg.NO_TARGET_EPISODE_PROB,
                 pose_noise: bool = True,
                 # --- Study 2 degradation axes (04 §6) ---
                 detection_dropout_p: float = cfg.DETECTION_DROPOUT_P,
                 track_velocity_noise: float = cfg.TRACK_VELOCITY_NOISE,
                 lidar_dropout_p: float = cfg.LIDAR_DROPOUT_P,
                 aft_mask_half_deg: float = cfg.LIDAR_AFT_MASK_HALF_DEG,
                 ego_speed_noise: float = cfg.EGO_SPEED_NOISE,
                 ego_yaw_rate_noise_dps: float = cfg.EGO_YAW_RATE_NOISE_DPS) -> None:
        super().__init__()
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        # Study 1 sweeps this; the corridor is inset in the basin, so every
        # simulated width is physically reproducible (03 §5).
        self.corridor_width = float(corridor_width if corridor_width is not None else map_width)
        self.max_obs = int(max_obs)
        self.path_mode = str(path_mode)
        self.curve_prob = float(curve_prob)
        self.lookahead_fraction = float(lookahead_fraction)
        self.n_max_targets = int(n_max_targets)
        self.no_target_prob = float(no_target_prob)
        self.render_mode = render_mode

        self.ego_speed_noise = float(ego_speed_noise)
        self.ego_yaw_rate_noise_dps = float(ego_yaw_rate_noise_dps)
        self._rng = np.random.default_rng()

        self.model = ShipModel()
        self.lidar = Lidar(aft_mask_half_deg=aft_mask_half_deg,
                           dropout_p=lidar_dropout_p, rng=self._rng)
        self.tracker = trk.Tracker(dropout_p=detection_dropout_p,
                                   velocity_noise=track_velocity_noise,
                                   rng=self._rng)
        self.observer = ObservationBuilder(self.n_max_targets)
        self._pose_noise = br.PoseNoise(self._rng) if pose_noise else None

        self.boundary_polygon = self._build_corridor()

        self.forced_num_obs: Optional[int] = None
        self.forced_targets: Optional[List[TargetShip]] = None
        self.observation_space = observation_space()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.renderer = None
        self.scenario_mode_used = "normal"
        self.path_mode_used = self.path_mode
        self._clear_state()

    # ------------------------------------------------------------------
    # Corridor geometry
    # ------------------------------------------------------------------
    def _build_corridor(self) -> List[Tuple[float, float]]:
        """The navigable channel, inset in the basin and centred on it.

        03 replaces this with a generator producing variable width along the
        path, bends, and off-centre reference paths -- all three are required
        before the boundary branch carries information (01 §3.3).
        """
        inset = 0.5 * (self.map_width - self.corridor_width)
        return br.rectangle(self.corridor_width, self.map_height, x0=inset, y0=0.0)

    @property
    def corridor_breadths(self) -> float:
        """Channel width in ship breadths -- the scale-explicit unit (03 §4)."""
        return self.corridor_width / cfg.BREADTH

    def corridor_bounds_x(self) -> Tuple[float, float]:
        inset = 0.5 * (self.map_width - self.corridor_width)
        return inset, inset + self.corridor_width

    def local_channel_width(self) -> float:
        """Channel width at the vessel's current station.

        Constant while the corridor is a straight inset rectangle.  03's
        generator introduces variation along the path, at which point this stops
        being a constant and `r_pf`'s normalisation starts to matter.
        """
        return self.corridor_width

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _clear_state(self) -> None:
        self.step_count = 0
        self.elapsed_time = 0.0
        self.asv_x = self.asv_y = 0.0
        self.asv_h = self.asv_w = 0.0
        self.u_body = self.v_body = 0.0
        self.speed_mps = 0.0
        self.rudder = 0.0
        self.rpm = 0.0

        self.start_x = self.start_y = 0.0
        self.goal_x = self.goal_y = 0.0
        self.distance_to_goal = 0.0
        self.asv_path: List[Tuple[float, float]] = []
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.targets: List[TargetShip] = []
        self.path: Optional[ReferencePath] = None

        self.cross_track_error = 0.0
        self.course_error = 0.0
        self.r_path = 0.0
        self.lookahead_course_error = 0.0
        self.closest_idx = 0
        self.lookahead_idx = 0
        self.tgt_x = self.tgt_y = 0.0
        self.lookahead_x = self.lookahead_y = 0.0

        self.sector_closeness = np.zeros(cfg.LIDAR_SECTORS, dtype=np.float32)
        self.boundary_closeness = np.zeros(cfg.BOUNDARY_RAYS, dtype=np.float32)
        self.tracks: List[trk.Track] = []
        self.true_border_clearance = min(self.corridor_width, self.map_height)

        # Perception metrics (04 §7): reported for the nominal case and across
        # the Study 2 sweep.
        self.acquisition_range: Optional[float] = None
        self.steps_target_visible = 0
        self.steps_target_tracked = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            self._rng = np.random.default_rng(seed)
            self.lidar.rng = self._rng
            self.tracker.rng = self._rng
            if self._pose_noise is not None:
                self._pose_noise.rng = self._rng

        self._clear_state()
        self.model.reset()
        self.lidar.reset()
        self.tracker.reset()
        self.observer.reset()
        if self._pose_noise is not None:
            self._pose_noise.reset()

        scenario = (options or {}).get("scenario")
        if scenario is not None:
            self._load_scenario(scenario)
        else:
            self._sample_layout()

        self.asv_x, self.asv_y = self.start_x, self.start_y
        self.asv_path = [(self.asv_x, self.asv_y)]
        self.distance_to_goal = float(np.hypot(self.asv_x - self.goal_x,
                                               self.asv_y - self.goal_y))

        self._perceive()
        self.true_border_clearance = self._border_clearance(self.hull_polygon())
        self._update_path_errors(course_deg=self.asv_h)

        self.render()
        return self._get_obs(), {}

    def _sample_layout(self) -> None:
        self.start_x, self.start_y, self.goal_x, self.goal_y = self._random_start_goal()
        self._build_path()

        if self.forced_num_obs is not None:
            num_obs = int(self.forced_num_obs)
        else:
            probs = np.asarray(cfg.TRAIN_OBS_PROBS, dtype=np.float64)
            num_obs = int(np.random.choice(cfg.TRAIN_OBS_COUNTS, p=probs / probs.sum()))
        self.obstacles = self.sample_obstacles(num_obs)
        self.targets = (list(self.forced_targets) if self.forced_targets is not None
                        else self._sample_target())

    def _sample_target(self) -> List[TargetShip]:
        """PLACEHOLDER spawn -- 03 owns the real encounter geometry.

        Places at most one head-on target beyond the sensor horizon, so the
        perception path is exercised in situ.  Four properties are kept because
        they are cheap now and expensive later:

        * spawned **outside** LiDAR range, so acquisition is part of the task
          and track acquisition range is a measurable quantity (03 §3);
        * a fraction of episodes carry **no target at all**, or the static-only
          configuration is out of distribution (01 §6.2);
        * the hull is oriented, not a circle (03 §3);
        * **a spawn lateral offset is sampled** (02a §11.1).

        That last one is not cosmetic.  Solving backwards for a spawn position
        without sampling an offset puts the target on the own ship's projected
        track in *every* episode, so `DCPA ~ 0` and the own ship must always
        produce the whole required separation.  The agent then never meets the
        case 02 §3.2 calls the normal one -- target correctly on its own side,
        channel-keeping already satisfies Rule 14, holding course is right -- and
        would learn "always alter" rather than "when to alter".  02a §11.1 makes
        this a blocking hand-off to 04; sampling it here keeps the placeholder
        from baking the same bias into everything built on top of it.

        TODO(03)/TODO(04): replace with the encounter generator -- head-on,
        crossing from both bows, overtaking, being overtaken -- with spawn DCPA
        and TCPA as explicit sampled axes, plus the reactive and non-compliant
        evaluation strata.
        """
        if self.n_max_targets < 1 or float(np.random.rand()) < self.no_target_prob:
            return []

        spawn_dist = cfg.LIDAR_RANGE + cfg.TARGET_SPAWN_MARGIN
        frac = min(1.0, spawn_dist / max(self.path.length, 1e-6))
        point, tangent, normal = self.path.frame_at_frac(frac)
        heading = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))
        speed = float(np.random.uniform(*cfg.TARGET_SPEED_RANGE))

        # Half the draws put the target on its own starboard side of the
        # fairway (positionally 9(a)-compliant, DCPA >= d_req); the rest leave
        # it near the centreline.  Both head-on regimes then appear.
        lo, hi = self.corridor_bounds_x()
        if float(np.random.rand()) < cfg.TARGET_COMPLIANT_SPAWN_PROB:
            offset = float(np.random.uniform(cfg.DOMAIN_LATERAL * 2.0,
                                             cfg.DOMAIN_LATERAL * 3.0))
        else:
            offset = float(np.random.uniform(-cfg.DOMAIN_LATERAL, cfg.DOMAIN_LATERAL))

        # `normal` is the path's left normal, so a negative multiple puts the
        # target to starboard of the own ship's track -- which is its own port
        # side, i.e. the side Rule 9(a) sends it to on a reciprocal course.
        x = float(point[0]) - offset * float(normal[0])
        y = float(point[1]) - offset * float(normal[1])
        margin = 0.5 * cfg.BREADTH
        x = float(np.clip(x, lo + margin, hi - margin))

        return [TargetShip(x, y, (heading + 180.0) % 360.0, speed)]

    def _load_scenario(self, scenario: dict) -> None:
        self.start_x, self.start_y = (float(v) for v in scenario["start"])
        self.goal_x, self.goal_y = (float(v) for v in scenario["goal"])

        if "corridor_width" in scenario:
            self.corridor_width = float(scenario["corridor_width"])
            self.boundary_polygon = self._build_corridor()

        saved_path = scenario.get("path")
        if saved_path is not None and len(saved_path) >= 2:
            self.path = ReferencePath(saved_path, self.lookahead_fraction)
        else:
            self._build_path()

        self.obstacles = [[(float(x), float(y)) for x, y in obs]
                          for obs in scenario.get("obstacles", [])]
        self.targets = [TargetShip(**spec) for spec in scenario.get("targets", [])]

    def sample_obstacles(self, num_obs: int) -> List[List[Tuple[float, float]]]:
        sampler = ObstacleSampler(self.path, self.map_width, self.map_height,
                                  (self.start_x, self.start_y),
                                  (self.goal_x, self.goal_y))
        layout = sampler.sample(num_obs)
        self.scenario_mode_used = sampler.mode_used
        return layout

    def _random_start_goal(self) -> Tuple[float, float, float, float]:
        lo, hi = self.corridor_bounds_x()
        margin_x = max(cfg.START_X_MARGIN_MIN, cfg.START_X_MARGIN_FRAC * self.corridor_width)
        margin_x = min(margin_x, 0.45 * self.corridor_width)
        goal_y = self.map_height - cfg.GOAL_Y_MARGIN

        if np.random.rand() < cfg.VERTICAL_PATH_PROB:
            x = float(np.random.uniform(lo + margin_x, hi - margin_x))
            return x, cfg.START_Y, x, goal_y

        start_x = float(np.random.uniform(lo + margin_x, hi - margin_x))
        goal_x = float(np.random.uniform(lo + margin_x, hi - margin_x))
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

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def estimated_pose(self) -> Tuple[float, float, float]:
        """The pose the localiser would report.

        One estimate feeds both the boundary raycast and the tracker, which is
        the field arrangement: they share rf2o's output and therefore share its
        drift.  Using ground truth for the tracker and a noisy pose for the
        boundary would understate the coupling 01 §4 step 3 warns about.
        """
        if self._pose_noise is None:
            return self.asv_x, self.asv_y, self.asv_h
        return self._pose_noise.perturb(self.asv_x, self.asv_y, self.asv_h)

    def _perceive(self) -> None:
        """Raycast -> gate -> pool -> cluster -> track."""
        scene = list(self.obstacles) + [t.hull() for t in self.targets]
        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=scene)

        est_x, est_y, est_h = self.estimated_pose()

        # Gate beyond-boundary returns.  A no-op in simulation, where the
        # raycast never sees the border, and a real filter in the field -- which
        # is exactly what makes the two pipelines equivalent (01 §3.4).
        #
        # Note the ordering the field pipeline must follow (01 §3.4): localise
        # on the FULL scan including the facility walls, because their fixed
        # features are the only along-track constraint scan-to-map registration
        # has, and only then gate for the tracker.  The walls are a liability
        # for tracking and an asset for localisation.
        gated = br.gate_beams(self.lidar.ranges, self.lidar.bearings,
                              est_x, est_y, est_h, self.boundary_polygon)

        self.sector_closeness = self.lidar.sector_closeness
        self.boundary_closeness = br.boundary_scan(
            self.asv_x, self.asv_y, self.asv_h, self.boundary_polygon,
            pose_noise=self._pose_noise,
        )

        detections = trk.cluster_scan(gated, self.lidar.bearings, est_x, est_y, est_h)
        self.tracker.update(detections, cfg.UPDATE_RATE)
        self.tracks = self.tracker.dynamic_tracks()
        self._update_perception_metrics()

    def _update_perception_metrics(self) -> None:
        """Track acquisition range, visibility and track uptime (04 §7).

        Tracking is attributed to the target by proximity rather than by "any
        dynamic track exists".  A static panel promoted to a dynamic track by
        pose drift is a false positive, and counting it as a detection would
        report uptime above 1.0 and flatter the N1 claim.
        """
        if not self.targets:
            return
        target = self.targets[0]
        gap = float(np.hypot(target.x - self.asv_x, target.y - self.asv_y))
        if gap > cfg.LIDAR_RANGE:
            return

        self.steps_target_visible += 1
        centre = np.array([target.x, target.y])
        matched = any(float(np.linalg.norm(t.position - centre)) <= cfg.TARGET_MATCH_RADIUS
                      for t in self.tracks)
        if matched:
            self.steps_target_tracked += 1
            if self.acquisition_range is None:
                self.acquisition_range = gap

    def _get_obs(self) -> Dict[str, np.ndarray]:
        u, v, r = self._measured_ego()
        return self.observer.build(
            sector_closeness=self.sector_closeness,
            boundary_scan=self.boundary_closeness,
            u=u, v=v, yaw_rate_degps=r,
            cross_track_error=self.cross_track_error,
            course_error_deg=self.course_error,
            lookahead_course_error_deg=self.lookahead_course_error,
            tracks=self.tracks,
            p_os=(self.asv_x, self.asv_y),
            v_os=self._own_velocity(),
            heading_os_deg=self.asv_h,
        )

    def _measured_ego(self) -> Tuple[float, float, float]:
        """u, v and r as the vessel would actually measure them.

        **An IMU is confirmed** (05 §4.7), which changes this gap rather than
        closing it.  `r` comes from the gyro directly, so its residual is the
        sensor noise floor rather than pose-differentiation error -- and the
        yaw-rate criterion 02 §4.2 depends on becomes directly measurable in the
        field instead of inferred.  `u` and `v` are largely rescued by the
        accelerometer but are still fused rather than measured, so a residual
        remains.  Both magnitudes are nominal-zero until 05 characterises them.
        """
        u, v, r = self.u_body, self.v_body, self.asv_w
        if self.ego_speed_noise > 0.0:
            u += float(self._rng.normal(0.0, self.ego_speed_noise))
            v += float(self._rng.normal(0.0, self.ego_speed_noise))
        if self.ego_yaw_rate_noise_dps > 0.0:
            r += float(self._rng.normal(0.0, self.ego_yaw_rate_noise_dps))
        return u, v, r

    def _own_velocity(self) -> np.ndarray:
        a = math.radians(self.asv_h)
        # Body -> world: surge along the heading, sway to starboard of it.
        return np.array([
            self.u_body * math.sin(a) + self.v_body * math.cos(a),
            self.u_body * math.cos(a) - self.v_body * math.sin(a),
        ])

    def _update_path_errors(self, course_deg: float) -> None:
        state = self.path.project(self.asv_x, self.asv_y, course_deg)
        self.closest_idx = state.closest_idx
        self.cross_track_error = state.cross_track_error
        self.course_error = state.course_error
        self.tgt_x, self.tgt_y = state.target
        self.lookahead_idx = state.lookahead_idx
        self.lookahead_x, self.lookahead_y = state.lookahead
        self.lookahead_course_error = state.lookahead_course_error
        # Yaw rate the path itself demands, rad/s (02b T3).  Zero while the
        # corridor is straight; 03's bends make it live.
        self.r_path = self.path.yaw_rate_for_tracking(self.closest_idx, self.u_body)

    # ------------------------------------------------------------------
    # Collision geometry -- carried over unchanged (Bucket A)
    # ------------------------------------------------------------------
    def hull_polygon(self) -> List[Tuple[float, float]]:
        half_l = 0.5 * (VESSEL_LENGTH + 2.0 * HULL_MARGIN)
        half_w = 0.5 * (VESSEL_WIDTH + 2.0 * HULL_MARGIN)
        h = math.radians(self.asv_h)
        sin_h, cos_h = math.sin(h), math.cos(h)
        return [
            (self.asv_x + fwd * sin_h - lat * cos_h, self.asv_y + fwd * cos_h + lat * sin_h)
            for fwd, lat in ((half_l, half_w), (half_l, -half_w),
                             (-half_l, -half_w), (-half_l, half_w))
        ]

    def _border_clearance(self, hull) -> float:
        """Distance to the nearest channel limit.  Negative once outside."""
        lo, hi = self.corridor_bounds_x()
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        return float(min(min(xs) - lo, hi - max(xs), min(ys), self.map_height - max(ys)))

    def _hits_border(self, hull) -> bool:
        return self._border_clearance(hull) < 0.0

    def hit_border(self) -> bool:
        return self._hits_border(self.hull_polygon())

    def _hits_obstacle(self, hull) -> bool:
        return any(_overlaps(hull, obs) for obs in self.obstacles)

    def _hits_target(self, hull) -> bool:
        return any(_overlaps(hull, t.hull()) for t in self.targets)

    def collision_kind(self, hull) -> Optional[str]:
        """Which of the three collision types happened, if any.

        Reported separately per the evaluation protocol: static obstacle,
        boundary and target vessel are distinct failures and must not be pooled.
        """
        if self._hits_border(hull):
            return "boundary"
        if self._hits_obstacle(hull):
            return "obstacle"
        if self._hits_target(hull):
            return "target"
        return None

    def _reached_goal(self) -> bool:
        if self.distance_to_goal <= cfg.GOAL_RADIUS:
            return True
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

        x_before, y_before = self.asv_x, self.asv_y

        dx, dy, heading, yaw_rate = self.model.update(self.rpm, self.rudder, cfg.UPDATE_RATE)
        self.asv_x += dx
        self.asv_y += dy
        self.asv_h = heading
        self.asv_w = yaw_rate
        self.u_body = self.model.u
        self.v_body = self.model.v

        for target in self.targets:
            target.step(cfg.UPDATE_RATE)

        moved_x = self.asv_x - x_before
        moved_y = self.asv_y - y_before
        self.speed_mps = float(math.hypot(moved_x, moved_y) / cfg.UPDATE_RATE)
        course_deg = (math.degrees(math.atan2(moved_x, moved_y))
                      if self.speed_mps > 1e-6 else self.asv_h)

        self._perceive()
        self._update_path_errors(course_deg)
        self.asv_path.append((self.asv_x, self.asv_y))
        self.distance_to_goal = float(np.hypot(self.asv_x - self.goal_x,
                                               self.asv_y - self.goal_y))

        hull = self.hull_polygon()
        self.true_border_clearance = self._border_clearance(hull)
        collision = self.collision_kind(hull)
        reached_goal = self._reached_goal()

        terminated = collision is not None or reached_goal
        self.step_count += 1
        truncated = self.step_count >= cfg.MAX_EPISODE_STEPS and not terminated

        reward = self._reward(collision, reached_goal, truncated)
        info = self._build_info(reward, rudder_cmd, collision, reached_goal, truncated)
        self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def _reward(self, collision: Optional[str], reached_goal: bool,
                truncated: bool) -> float:
        """PLACEHOLDER.  Terminal payoff only.

        TODO(02): 02 owns the entire reward -- six carried-over terms, five
        COLREGs terms (wrong-side passing, port turn in head-on, bow crossing,
        course-keeping hold, late-or-insufficient action), the Rule 9 precedence
        table and the mandatory per-term scale audit.  Deliberately sparse here
        so nothing in this file can be mistaken for a design decision, and so no
        Paper 2 shaping term is inherited rather than chosen (D10).

        02 §4.1 states collision -200, goal +100, timeout via value
        bootstrapping.  The constants still carry Paper 2's magnitudes and are
        marked TODO(02) in `constants.py`.

        A policy trained against this reward alone will not learn the task.
        That is intended: 01 ships perception, not a trainable agent.
        """
        if collision is not None:
            return float(cfg.R_COLLISION)
        reward = 0.0
        if reached_goal:
            reward += float(cfg.R_GOAL)
        if truncated:
            reward += float(cfg.R_TIMEOUT)
        return float(reward)

    def _build_info(self, reward, rudder_cmd, collision, reached_goal, truncated) -> Dict:
        classes = self.observer.encounter_classes
        held = list(classes.values())
        return {
            "reward": float(reward),
            "cross_track_error": float(self.cross_track_error),
            "ye": float(abs(self.cross_track_error)),
            "course_error": float(self.course_error),
            "lookahead_course_error": float(self.lookahead_course_error),
            "speed_mps": float(self.speed_mps),
            "u_body_mps": float(self.u_body),
            "v_body_mps": float(self.v_body),
            "yaw_rate_dps": float(self.asv_w),
            # Both in rad/s so 02a `R-8`'s `r - r_path` cannot be taken in
            # mixed units -- the env's own yaw rate is degrees per second.
            "yaw_rate_radps": float(math.radians(self.asv_w)),
            "r_path_radps": float(self.r_path),
            "rpm": float(self.rpm),
            "rudder_deg": float(rudder_cmd * MAX_RUD_ANGLE),
            "distance_to_goal": float(self.distance_to_goal),
            "min_lidar": float(np.min(self.lidar.ranges)),
            "min_sector_range": float(np.min(self.lidar.sector_ranges)),
            "max_boundary_closeness": float(np.max(self.boundary_closeness)),
            "true_border_clearance": float(self.true_border_clearance),
            "corridor_width": float(self.corridor_width),
            "corridor_breadths": float(self.corridor_breadths),
            # `W_local` is the width at the vessel's current station.  Equal to
            # `corridor_width` while the channel is a straight inset rectangle;
            # 03's generator makes the two diverge.  02a's `r_pf` normalises on
            # it and R-10 overrides it to 10.0 for the open-water benchmark.
            "W_local": float(self.local_channel_width()),
            # Perception metrics (04 §7) -- the N1 evidence.
            "n_tracks": int(len(self.tracks)),
            "n_targets": int(len(self.targets)),
            "acquisition_range": (float(self.acquisition_range)
                                  if self.acquisition_range is not None else float("nan")),
            "max_coast_steps": int(self.tracker.max_coast),
            "dropped_detections": int(self.tracker.dropped_detections),
            "steps_target_visible": int(self.steps_target_visible),
            "steps_target_tracked": int(self.steps_target_tracked),
            "encounter_class": held[0] if held else "none",
            "encounter_classes": dict(classes),
            "crossing_sides": self.observer.crossing_sides,
            # Reported separately: static obstacle / boundary / target vessel.
            "collision_kind": collision,
            "collided": collision is not None,
            "collided_boundary": collision == "boundary",
            "collided_obstacle": collision == "obstacle",
            "collided_target": collision == "target",
            "reached_goal": bool(reached_goal),
            "timeout": bool(truncated),
            "path_mode": self.path_mode_used,
            "scenario_mode": self.scenario_mode_used,
        }

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.renderer is None:
            from render import Renderer
            self.renderer = Renderer(self.map_width, self.map_height)
        self.renderer.draw(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def _overlaps(poly_a: Sequence, poly_b: Sequence) -> bool:
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


# Retained under the Paper 2 name because `metrics.py` imports it.
_polygons_intersect = _overlaps


if __name__ == "__main__":
    # Quick start: `python src/env.py` drops straight into manual control.
    # `src/play.py` has the full CLI -- random actions, the width sweep, the
    # Study 2 degradation knobs, and a headless smoke test.
    import sys

    from play import main

    raise SystemExit(main(sys.argv[1:]))
