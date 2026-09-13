"""Gymnasium environment: path following with static obstacles and one target vessel.

**Revision 2** — two-vessel encounters.  One dynamic target, `N_MAX_TARGETS`
configurable so a multi-vessel extension costs a retrain, not a redesign (S1).

Scope of this file in task 01
-----------------------------
Perception and observation.  Specifically:

* **The reward is live** (02b T4).  Eight dense terms plus terminals, in
  `reward/`, redesigned rather than patched per D10 so no Paper 2 shaping term
  is carried across.  This file assembles the `RewardState` -- it is the only
  object holding both the simulated truth and the map -- and owns none of the
  reward's design.
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
import corridor as corr
import targets as tgtmod
import tracking as trk
from asv_lidar import Lidar
from observation import ObservationBuilder, observation_space
from obstacles import ObstacleSampler
from path import ReferencePath, curved_points, straight_points
from reward import RewardConfig, RewardFunction
from reward import terms as rterms
from ship import HULL_MARGIN, MAX_RUD_ANGLE, VESSEL_LENGTH, VESSEL_WIDTH, ShipModel


# 03a §5.1 replaces the four-corner box with an oriented 11-vertex hull, and
# 03a §5.3 adds the behaviour models.  Both live in `targets.py`; the Paper 2
# name is kept as an alias because the play harness, the tests and 04a's named
# cases all construct targets by it, and renaming a constructor across the tree
# to gain nothing is churn.
TargetShip = tgtmod.Target


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
                 ego_yaw_rate_noise_dps: float = cfg.EGO_YAW_RATE_NOISE_DPS,
                 reward_config: Optional[RewardConfig] = None,
                 open_water: bool = False,
                 channel: Optional[object] = None,
                 facility_walls: bool = cfg.SIMULATE_FACILITY_WALLS) -> None:
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
        # `R-10`: the 04 §4.1 benchmark also runs open water, where `r_pf`
        # normalises on a reference width, `r_bnd` is zero and both alterations
        # are admissible.  A flag rather than a subclass, because the difference
        # is three constants and not a different environment.
        self.open_water = bool(open_water)
        self.reward_fn = RewardFunction(reward_config or RewardConfig())
        self._pose_noise = br.PoseNoise(self._rng) if pose_noise else None

        # 03a §3.1: the corridor is a generated channel, not an inset
        # rectangle.  A fixed one can be passed in for a named evaluation case
        # or a Study 1 width level; otherwise each episode samples one.
        self.channel = channel if channel is not None else corr.rectangle(
            self.corridor_width)
        self.resample_channel = channel is None
        self.boundary_polygon = self.channel.polygon()

        # 03a §1.2: the out-of-corridor world, so the gate has something to do.
        self.facility_walls = bool(facility_walls)
        self.wall_polygon = corr.facility_walls((self.map_width, self.map_height))

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
        """The navigable channel polygon, from the current `Corridor`."""
        return self.channel.polygon()

    @property
    def corridor_breadths(self) -> float:
        """Channel width in ship breadths -- the scale-explicit unit (03 §4)."""
        return self.corridor_width / cfg.BREADTH

    def corridor_bounds_x(self) -> Tuple[float, float]:
        """Lateral extent of the channel, for metrics that need a scalar pair.

        Exact while the channel is straight; an envelope once it bends.  Every
        clearance that matters is measured against the polygon itself.
        """
        left, right = self.channel.edges()
        xs = np.concatenate([left[:, 0], right[:, 0]])
        return float(xs.min()), float(xs.max())

    def local_channel_width(self) -> float:
        """Channel width at the vessel's current station, metres.

        Now genuinely local: 03a's generator varies width along `s`, so this and
        `corridor_width` diverge, and `r_pf`'s width normalisation starts doing
        the job it was specified for -- holding the path term's range constant
        across the Study 1 sweep instead of letting the gradient move with the
        channel.
        """
        # Interpolated from the known arclength rather than searched for: the
        # path projection already located the vessel this step, and an argmin
        # over 500 stations every step is a cost with no information in it.
        return self.channel.width_at_s(self.s_along + self.path_start_s)

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
        self.s_along = 0.0
        self.prev_s_along = 0.0
        self.path_start_s = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.reward_fn.reset()
        self.last_reward = None
        self.last_reward_state = None
        self.last_panel = None
        self._clearances = (float("inf"), float("inf"))
        self.episode_seed: Optional[int] = None
        self.target_spawn_regime = "none"
        self.clip_steps = {branch: 0 for branch in
                           ("lidar", "boundary", "ego", "path", "target")}
        self.dim_clip_steps: Dict[str, np.ndarray] = {}
        self.branch_extremes: Dict[str, Tuple[float, float]] = {}
        self.obs_steps = 0
        self.misclassified_steps = 0
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
        self._pending_seed = seed
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

        self.episode_seed = self._pending_seed
        self._perceive()
        self.true_border_clearance = self._border_clearance(self.hull_polygon())
        self._update_path_errors(course_deg=self.asv_h)
        # Prime the progress increment, or the first step would score the whole
        # distance from the path origin to the start position as advance.
        self.prev_s_along = self.s_along

        self.render()
        obs = self._get_obs()
        self._record_obs_health(obs)
        return obs, {}

    def _sample_layout(self) -> None:
        if self.resample_channel:
            self.channel = corr.sample(
                self._rng, width_range=(self.corridor_width, self.corridor_width))
            self.boundary_polygon = self.channel.polygon()
        self._build_path_from_channel()

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
            self.target_spawn_regime = "COMPLIANT"
        else:
            offset = float(np.random.uniform(-cfg.DOMAIN_LATERAL, cfg.DOMAIN_LATERAL))
            self.target_spawn_regime = "DISPLACED"

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

    def _build_path_from_channel(self) -> None:
        """Reference path from the channel's Rule 9(a) station (03a §3.2).

        The path is the centreline offset by a fraction of the **local**
        half-width, with a positive mean -- so the vessel is trained to hold the
        starboard side of the fairway rather than the middle of it, which is the
        behaviour the whole Rule 9 precedence argument is about.
        """
        # **Start the path inside the corridor, not at its mouth.**  The hull is
        # 1.73 m long and the collision test is on the polygon, so a vessel whose
        # origin sits exactly on the corridor's first station has its stern
        # outside the channel and terminates on step 1 -- which is what happened
        # the first time the generator was wired in.  The corridor is 25 m and
        # the path 20 m, so the inset is free.
        inset = 0.5 * VESSEL_LENGTH + HULL_MARGIN
        self.path_start_s = float(inset)
        points = self.channel.reference_path_points(start_s=inset)
        self.path = ReferencePath(points, self.lookahead_fraction)
        self.start_x, self.start_y = (float(points[0][0]), float(points[0][1]))
        self.goal_x, self.goal_y = (float(points[-1][0]), float(points[-1][1]))
        self.path_mode_used = "corridor"

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
        # 03a §1.2.  The facility walls are **returned by the sensor and then
        # gated**, which is the whole point: until they existed the gate had
        # nothing to remove in simulation and was load-bearing only in the
        # field -- a sim-to-real gap in the exact component 01 §3 exists to
        # remove one from.  The corridor boundary is NOT in this list; it is a
        # map polygon and is invisible to the sensor (01 §3.1).
        scene = list(self.obstacles) + [t.hull() for t in self.targets]
        if self.facility_walls:
            scene = scene + [self.wall_polygon]
        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=scene)
        self.raw_ranges = self.lidar.ranges.copy()

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

        # **Pool from the gated scan, not the raw one.**  The method docstring
        # above has always said "raycast -> gate -> pool -> cluster"; the code
        # pooled before gating, which returned the same answer for as long as
        # the gate had nothing to remove.  With the facility walls returned it
        # does not: 624 of 720 beams reached the obstacle branch.
        self.lidar.repool(gated)
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
            # The map side of `R-1`.  This environment is the only object
            # holding both the perception output and the boundary polygon, so
            # the admissibility predicate is fed from here rather than
            # rediscovered inside the observation builder.
            r_path=self.r_path,
            path=self.path,
            boundary_polygon=self.boundary_polygon,
            s_along=self.s_along,
            true_targets=self.targets,
            open_water=self.open_water,
        )

    @property
    def encounter_contexts(self):
        """The per-step `EncounterContext` per track, as of the last observation."""
        return self.observer.encounter_contexts

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
        self.s_along = float(state.s_along)
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

        own_state = {"x": self.asv_x, "y": self.asv_y,
                     "velocity": self._own_velocity(), "heading": self.asv_h}
        for target in self.targets:
            target.step(cfg.UPDATE_RATE, own=own_state)
            # Confined classes keep the fairway; a crossing target under Rule
            # 9(d) is not a channel user and is left alone (03a §5.2).
            tgtmod.clamp_to_corridor(target, self.channel)

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

        # **Observation first.**  The per-step `EncounterContext` objects are
        # built while the observation is assembled, and the reward reads them
        # back rather than deriving its own -- which is the whole point of
        # 02a §10.1.  Building the observation after the reward would mean two
        # derivations of the same encounter from the same inputs.
        obs = self._get_obs()
        self._record_obs_health(obs)

        breakdown = self._reward(action, collision, reached_goal, truncated)
        info = self._build_info(breakdown, rudder_cmd, throttle_cmd,
                                collision, reached_goal, truncated)
        self.prev_action = np.array([rudder_cmd, throttle_cmd], dtype=np.float32)
        self.prev_s_along = self.s_along
        self.render()
        return obs, float(breakdown.total), terminated, truncated, info

    def _reward_state(self, action) -> rterms.RewardState:
        """Assemble one step's `RewardState`: the own ship and the scene.

        This environment is the only object holding both the simulated truth and
        the map, which is why the ground-truth clearances are measured here and
        handed over rather than looked up inside a term.  Everything below is a
        physical fact under `R-1` -- the agent pays for hitting things whether
        or not it saw them.
        """
        rudder_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))
        hull = self.hull_polygon()
        # Held for the step: `_panel_view` needs the same two numbers, and both
        # are O(hull x polygon edges).  Recomputing them for the display would
        # also let the panel and the reward disagree, which is the failure the
        # whole single-source arrangement exists to prevent.
        self._clearances = (self._hull_boundary_distance(hull),
                            self._nearest_obstacle_clearance(hull))

        return rterms.RewardState(
            u=float(self.u_body),
            v=float(self.v_body),
            # The environment reports yaw in degrees per second; every COLREGs
            # threshold is in rad/s.  Converted once, here, so `r - r_path` can
            # never be taken in mixed units.
            r=float(math.radians(self.asv_w)),
            heading_deg=float(self.asv_h),
            e_y=float(self.cross_track_error),
            chi=float(math.radians(self.course_error)),
            chi_la=float(math.radians(self.lookahead_course_error)),
            w_local=float(self.local_channel_width()),
            r_path=float(self.r_path),
            ds=float(self.s_along - self.prev_s_along),
            l_path=float(self.path.length),
            d_bnd=self._clearances[0],
            d_clear=self._clearances[1],
            dom_intrusion=rterms.domain_intrusion(
                (self.asv_x, self.asv_y), self.asv_h, self.targets,
                self.reward_fn.cfg),
            d_rudder=rudder_cmd - float(self.prev_action[0]),
            d_throttle=throttle_cmd - float(self.prev_action[1]),
            step_index=int(self.step_count),
            open_water=self.open_water,
        )

    def _reward(self, action, collision: Optional[str], reached_goal: bool,
                truncated: bool):
        """The 02a reward.  Returns the full breakdown, not just the scalar.

        02 owns the design; this method owns nothing but the call.  The
        breakdown is retained because `04 §7`'s metric set is a *read* of these
        keys rather than a separate computation -- Paper 2's concessions came
        from metrics that were not designed in before the campaign ran.
        """
        self.last_reward_state = self._reward_state(action)
        breakdown = self.reward_fn(
            self.last_reward_state, self.observer.encounter_contexts,
            collision=collision, reached_goal=reached_goal, truncated=truncated)
        self.last_reward = breakdown
        return breakdown

    # ------------------------------------------------------------------
    # Ground-truth clearances for the safety terms
    # ------------------------------------------------------------------
    def _hull_boundary_distance(self, hull) -> float:
        """Hull polygon to channel boundary, metres; 0 once any corner is out.

        Measured against the boundary *polygon* rather than the axis-aligned
        channel bounds, because 03's generator produces bends and variable
        width, at which point the two stop agreeing and only one of them is the
        channel.
        """
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        inside = br.points_in_polygon(np.asarray(xs), np.asarray(ys),
                                      self.boundary_polygon)
        if not bool(np.all(inside)):
            return 0.0
        return float(np.min(br.points_boundary_distance(
            np.asarray(xs), np.asarray(ys), self.boundary_polygon)))

    def _nearest_obstacle_clearance(self, hull) -> float:
        """Minimum hull-to-hull clearance to a static obstacle within the swath.

        Restricted to +/-`OBS_SWATH_HALF_DEG`, matching the `c_t` swath, so the
        agent is never charged for proximity it cannot observe and an obstacle
        passed astern generates no signal against an action space with no
        reverse (02a §5.4).
        """
        best = float("inf")
        for obstacle in self.obstacles:
            gap, point = _polygon_gap(hull, obstacle)
            bearing = _wrap180(math.degrees(math.atan2(
                point[0] - self.asv_x, point[1] - self.asv_y)) - self.asv_h)
            if abs(bearing) <= cfg.OBS_SWATH_HALF_DEG:
                best = min(best, gap)
        return best

    def _record_obs_health(self, obs) -> None:
        """Per-branch clip rate: the F19 detector (RENDER_PANEL_SPEC §6).

        A branch pinned at its normaliser's clip carries no gradient.  That is
        exactly the bug where the `ego` surge feature sat at 1.0 for 45% of a
        run, which was invisible in every training curve and obvious the moment
        anyone counted.

        The lower bound is counted only for **signed** branches.  A pooled
        LiDAR sector reading 0.0 means "nothing within range", which is the
        normal state of most sectors most of the time and not a saturation --
        counting it would put every run permanently in the red.

        The `target` branch's class one-hot and presence bit are excluded for
        the same reason, and the panel found that one itself: an occupied slot
        sets a one-hot element and the presence bit to exactly 1.0 every step,
        so counting them reported a 90% clip rate on a branch whose ten
        continuous features were nowhere near their bounds.  An indicator
        variable sitting at 1 is the indicator working.
        """
        self.obs_steps += 1
        for branch, box in self.observation_space.spaces.items():
            raw = np.asarray(obs[branch], dtype=np.float64)
            values = _normalised_dims(branch, raw)
            hi = float(np.max(box.high))
            lo = float(np.min(box.low))
            at_clip = values >= hi - 1e-6
            if lo < 0.0:
                at_clip = at_clip | (values <= lo + 1e-6)

            if bool(np.any(at_clip)):
                self.clip_steps[branch] += 1
            counts = self.dim_clip_steps.get(branch)
            if counts is None:
                counts = np.zeros(raw.size, dtype=np.int64)
                self.dim_clip_steps[branch] = counts
            counts[_normalised_indices(branch, raw.size)] += at_clip.astype(np.int64)

            prev = self.branch_extremes.get(branch)
            here = (float(np.min(values)), float(np.max(values)))
            self.branch_extremes[branch] = here if prev is None else (
                min(prev[0], here[0]), max(prev[1], here[1]))

    def clip_fractions(self) -> Dict[str, float]:
        steps = max(self.obs_steps, 1)
        return {branch: count / steps for branch, count in self.clip_steps.items()}

    def _build_info(self, breakdown, rudder_cmd, throttle_cmd, collision,
                    reached_goal, truncated) -> Dict:
        """One step's `info`: the reward keys, the metrics, and the panel.

        **The panel is a view on this, never a separate computation**
        (`RENDER_PANEL_SPEC` §0).  If the panel needs a number, it is added here
        rather than derived in `render.py` -- the same principle as the single
        `EncounterContext`, and for the same reason: two consumers, one source,
        or they diverge.

        The flat `reward/...` and `colregs/...` keys are `02a §10.3`'s logging
        schema.  `00 §4.2`'s metric set should be a *read* of these keys rather
        than a separate computation; Paper 2's concessions came from metrics
        that were not designed in before the campaign ran.
        """
        contexts = self.observer.encounter_contexts
        classes = {tid: ctx.cls for tid, ctx in contexts.items()}
        held = list(classes.values())
        governing = self._governing_context(contexts)
        if governing is not None and governing.misclassified:
            self.misclassified_steps += 1

        info = {
            "reward": float(breakdown.total),
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
            "action_rudder": float(rudder_cmd),
            "action_throttle": float(throttle_cmd),
            "d_action_rudder": float(rudder_cmd - self.prev_action[0]),
            "d_action_throttle": float(throttle_cmd - self.prev_action[1]),
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
            # it and `R-10` overrides it to 10.0 for the open-water benchmark.
            "W_local": float(self.local_channel_width()),
            "s_along": float(self.s_along),
            "path_length": float(self.path.length),
            "seed": self.episode_seed,
            "step": int(self.step_count),
            "max_steps": int(cfg.MAX_EPISODE_STEPS),
            "elapsed_time": float(self.elapsed_time),
            "open_water": bool(self.open_water),
            "spawn_regime": str(self.target_spawn_regime),
            # Perception metrics (04 §7) -- the N1 evidence.
            "n_tracks": int(len(self.tracks)),
            "n_targets": int(len(self.targets)),
            "acquisition_range": (float(self.acquisition_range)
                                  if self.acquisition_range is not None else float("nan")),
            "max_coast_steps": int(self.tracker.max_coast),
            "dropped_detections": int(self.tracker.dropped_detections),
            "steps_target_visible": int(self.steps_target_visible),
            "steps_target_tracked": int(self.steps_target_tracked),
            "misclassified_steps": int(self.misclassified_steps),
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
        info.update(breakdown.as_info())
        # Built once and held, because `render.py` reads it off the environment
        # rather than off `info` -- the renderer is handed the env, not the step
        # tuple.  Same object either way, which is the point: the panel is a
        # view on this step's `info` and never a second computation.
        self.last_panel = self._panel_view(breakdown, governing)
        info["panel"] = self.last_panel
        return info

    def _governing_context(self, contexts):
        """The target the panel and the metrics speak about.

        The one the COLREGs group scored against when it scored anything, and
        otherwise the closest.  At `N_MAX_TARGETS = 1` this is "the target"; the
        selection exists so the panel does not silently start describing a
        different vessel when the scope widens.
        """
        if not contexts:
            return None
        track_id = self.last_reward.colregs_track if self.last_reward else None
        if track_id in contexts:
            return contexts[track_id]
        return min(contexts.values(), key=lambda c: c.rng)

    def _panel_view(self, breakdown, ctx) -> Dict:
        """Everything `render.py`'s left panel draws, computed once, here.

        Structured rather than flat because the panel's blocks are structured;
        the flat `02a §10.3` keys above are the logging schema and this is the
        display schema, both read off the same step.
        """
        cfgr = self.reward_fn.cfg
        u_ref_eff = float(breakdown.u_ref_eff)
        g_u = float(np.clip(max(self.u_body, 0.0) / max(u_ref_eff, 1e-9), 0.0, 1.0))

        panel = {
            "run": {
                "seed": self.episode_seed,
                "scenario": self.scenario_mode_used,
                "step": int(self.step_count),
                "max_steps": int(cfg.MAX_EPISODE_STEPS),
                "t": float(self.elapsed_time),
                "corridor_w": float(self.corridor_width),
                "corridor_b": float(self.corridor_breadths),
                "w_local": float(self.local_channel_width()),
                "targets": int(len(self.targets)),
                "spawn": str(self.target_spawn_regime),
                "open_water": bool(self.open_water),
            },
            "ego": {
                "x": float(self.asv_x), "y": float(self.asv_y),
                "hdg": float(self.asv_h),
                "u": float(self.u_body), "v": float(self.v_body),
                "r_dps": float(self.asv_w),
                "u_ref": float(cfgr.u_ref), "u_ref_eff": u_ref_eff,
                "u_ref_rule": breakdown.u_ref_rule,
                "u_ref_reason": breakdown.u_ref_reason,
                "g_u": g_u,
                "g_u_sat": bool(g_u >= 1.0 - 1e-9),
                "r_path": float(self.r_path),
                "r_err": float(math.radians(self.asv_w) - self.r_path),
                "rudder": float(self.prev_action[0]),
                "throttle": float(self.prev_action[1]),
                "rudder_deg": float(self.rudder),
                "rpm": float(self.rpm),
                "d_rudder": float(getattr(self.last_reward_state, "d_rudder", 0.0)),
                "d_throttle": float(getattr(self.last_reward_state, "d_throttle", 0.0)),
                "kappa_delta": float(cfgr.kappa_delta),
                "sigma": float(breakdown.sigma_smooth),
            },
            "reward": {
                "rows": self._reward_rows(breakdown),
                "step_total": float(breakdown.total),
                "dense": float(breakdown.dense),
                "terminal": float(breakdown.terminal),
                "dominant": breakdown.dominant,
                "episode_total": float(self.reward_fn.audit.episode_total),
                "episode_dominant": self.reward_fn.audit.episode_dominant(),
                "hierarchy": self.reward_fn.audit.hierarchy_violations(),
            },
            "obs_health": self._obs_health_rows(),
            "clearance": {
                "boundary": self._clearances[0],
                "obstacle": self._clearances[1],
                "goal": float(self.distance_to_goal),
                "steps_left": int(cfg.MAX_EPISODE_STEPS - self.step_count),
            },
            "colregs": None,
            "perception": None,
        }
        # The domain margin is reported from **truth**, not from the context,
        # so the panel can show an intrusion the tracker has not seen -- which
        # is precisely the case `r_dom` charges for and the case a reader would
        # otherwise have no way to explain (F29).
        panel["clearance"]["domain_margin"] = self._true_domain_margin()
        if ctx is not None:
            panel["colregs"] = self._colregs_block(breakdown, ctx)
            panel["perception"] = self._perception_block(ctx)
            panel["clearance"]["target"] = float(ctx.d_ts_true)
        return panel

    def _reward_rows(self, breakdown) -> list:
        """The four columns of block [5], and the fourth is the important one.

        `inst` / `xw` / `Sep` / **`range(ep)`**.  The range column is the direct
        detector for the Paper 2 scale bug: a term whose episode range is
        `[-0.44, -0.41]` varies by less than 10% of its own value -- a constant
        offset wearing a shaping term's costume, invisible in the other three
        columns.
        """
        rows = []
        for row in self.reward_fn.audit.rows():
            name = row["name"]
            rows.append({
                "name": name,
                "inst": float(breakdown.term.get(name, 0.0)),
                "xw": float(breakdown.weighted.get(name, 0.0)),
                "sum": float(row["sum"]),
                "lo": float(row["lo"]),
                "hi": float(row["hi"]),
                "flat": bool(row["flat"]),
            })
        return rows

    def _obs_health_rows(self) -> list:
        """Per-branch clip rate, and **which dimension is responsible**.

        `RENDER_PANEL_SPEC` §6 asks for a per-dimension drill-down on a keypress,
        "since a single saturating dimension inside a 27-dim branch will not move
        the branch aggregate much".  Naming the worst dimension inline is better
        than a keypress: it is one string, and it is exactly the thing wanted at
        the moment the aggregate goes red.
        """
        from observation import branch_feature_names

        rows = []
        fractions = self.clip_fractions()
        steps = max(self.obs_steps, 1)
        for branch, box in self.observation_space.spaces.items():
            lo, hi = self.branch_extremes.get(branch, (0.0, 0.0))
            counts = self.dim_clip_steps.get(branch)
            worst, worst_frac = "", 0.0
            if counts is not None and counts.size:
                index = int(np.argmax(counts))
                worst_frac = float(counts[index]) / steps
                names = branch_feature_names(branch)
                worst = names[index] if index < len(names) else f"dim{index}"
            rows.append({
                "name": branch,
                "dim": int(np.prod(box.shape)),
                "lo": float(lo),
                "hi": float(hi),
                "clip": float(fractions.get(branch, 0.0)),
                "worst": worst,
                "worst_clip": worst_frac,
            })
        return rows

    def _colregs_block(self, breakdown, ctx) -> Dict:
        cfgr = self.reward_fn.cfg
        state = self._reward_state(np.asarray(self.prev_action))
        parts = rterms.r8_parts(state, ctx, cfgr)
        return {
            "cls": ctx.cls,
            "cls_true": ctx.cls_true,
            "state": ctx.state,
            "engaged_at": int(ctx.t_engage),
            "engaged_for": (float((self.step_count - ctx.t_engage) * cfg.UPDATE_RATE)
                            if ctx.t_engage >= 0 else 0.0),
            "sense": {1: "STBD", -1: "PORT", 0: "none"}[int(ctx.compliant_turn_sense)],
            "a_stbd": bool(ctx.a_stbd),
            "a_port": bool(ctx.a_port),
            "known": bool(ctx.admissibility_known),
            "dy_req": float(ctx.dy_req),
            "r_stbd": float(ctx.r_stbd),
            "r_port": float(ctx.r_port),
            "d_req": float(cfgr.d_req),
            "rho": float(ctx.rho),
            "a_req": float(parts["a_req"]),
            "a_t": float(parts["a_t"]),
            "urgency": float(parts["urgency"]),
            "in_extremis": bool(ctx.in_extremis),
            "terms": dict(breakdown.colregs),
            "why": rterms.explain_colregs(state, ctx, cfgr),
            "group": float(-breakdown.term.get("col", 0.0)),
            "pre_clip": float(breakdown.colregs_pre_clip),
        }

    def _perception_block(self, ctx) -> Dict:
        """Block [3]: estimate against truth, which is `R-1` made visible.

        Under `R-1` the safety terms read truth and the COLREGs gating reads the
        estimate.  The panel must show both or that decision is invisible, and a
        misclassification -- the failure `04 §6` names as the one that matters --
        is otherwise almost impossible to spot in a replay.
        """
        target = self.targets[0] if self.targets else None
        true = {}
        if target is not None:
            import cpa_cri as cc
            p_true = (target.x, target.y)
            v_os = self._own_velocity()
            dcpa, tcpa = cc.cpa((self.asv_x, self.asv_y), v_os, p_true, target.velocity)
            true = {
                "range": float(np.hypot(target.x - self.asv_x, target.y - self.asv_y)),
                "bearing": float(cc.relative_bearing_deg(
                    (self.asv_x, self.asv_y), self.asv_h, p_true)),
                "speed": float(target.speed),
                "heading": float(target.heading_deg),
                "dcpa": float(dcpa),
                "tcpa": float(tcpa),
                "cls": ctx.cls_true,
            }
        track = next((t for t in self.tracks if t.id == ctx.track_id), None)
        return {
            "est": {
                "range": float(ctx.rng),
                "bearing": float(ctx.alpha if ctx.alpha <= 180.0 else ctx.alpha - 360.0),
                "speed": float(ctx.speed_ts),
                "heading": float(track.course_deg) if track is not None else float("nan"),
                "dcpa": float(ctx.dcpa),
                "tcpa": float(ctx.tcpa),
                "cls": ctx.cls,
            },
            "true": true,
            "track": {
                "age": int(track.age) if track is not None else 0,
                "hits": int(track.hits) if track is not None else 0,
                "misses": int(track.misses) if track is not None else 0,
                "coast": float(self.tracker.max_coast * cfg.UPDATE_RATE),
            },
            "dropped": int(self.tracker.dropped_detections),
            "mismatched_steps": int(self.misclassified_steps),
        }

    def _true_domain_margin(self) -> Optional[float]:
        """Signed clearance to the nearest true target's domain boundary, metres.

        Negative means intruding.  `None` when there is no target at all -- as
        opposed to a target the tracker has lost, which still reports.
        """
        import cpa_cri as cc
        if not self.targets:
            return None
        margins = []
        for target in self.targets:
            gap = float(np.hypot(target.x - self.asv_x, target.y - self.asv_y))
            bearing = cc.relative_bearing_deg((self.asv_x, self.asv_y), self.asv_h,
                                              (target.x, target.y))
            margins.append(gap - cc.domain_scale(bearing))
        return float(min(margins))

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


def _polygon_gap(poly_a: Sequence, poly_b: Sequence):
    """Closest distance between two convex polygons, and where on `b` it is.

    Exact for convex polygons: the minimum distance between two disjoint convex
    sets is realised at a vertex of one and a point on an edge of the other, so
    checking both directions covers every case.  Returns 0 for overlapping
    polygons.
    """
    if _overlaps(poly_a, poly_b):
        centre = np.mean(np.asarray(poly_b, dtype=np.float64), axis=0)
        return 0.0, (float(centre[0]), float(centre[1]))

    best, at = float("inf"), (0.0, 0.0)
    for points, edges, on_b in ((poly_a, poly_b, True), (poly_b, poly_a, False)):
        pts = np.asarray(points, dtype=np.float64)
        poly = np.asarray(edges, dtype=np.float64)
        seg = np.roll(poly, -1, axis=0) - poly
        len_sq = np.einsum("ij,ij->i", seg, seg)
        offset = pts[:, None, :] - poly[None, :, :]
        t = np.clip(np.einsum("nmj,mj->nm", offset, seg)
                    / np.where(len_sq < 1e-18, 1.0, len_sq), 0.0, 1.0)
        foot = poly[None, :, :] + t[..., None] * seg[None, :, :]
        dist = np.linalg.norm(pts[:, None, :] - foot, axis=-1)
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        if float(dist[i, j]) < best:
            best = float(dist[i, j])
            here = foot[i, j] if on_b else pts[i]
            at = (float(here[0]), float(here[1]))
    return best, at


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _normalised_indices(branch: str, size: int):
    """Which dimensions of a branch are normalisers rather than definitions.

    Only these can meaningfully *clip*.  For `target` that excludes the sin/cos
    pairs, the class one-hot and the presence bit: they reach their bounds
    because of what they are, not because information was lost.
    """
    if branch != "target":
        return np.arange(size)
    from observation import NORMALISED_SLOT_INDICES
    return np.array([slot * cfg.TARGET_FEATURES + i
                     for slot in range(cfg.N_MAX_TARGETS)
                     for i in NORMALISED_SLOT_INDICES], dtype=np.int64)


def _normalised_dims(branch: str, values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[_normalised_indices(branch, values.size)]


# Retained under the Paper 2 name because `metrics.py` imports it.
_polygons_intersect = _overlaps


if __name__ == "__main__":
    # Quick start: `python src/env.py` drops straight into manual control.
    # `src/play.py` has the full CLI -- random actions, the width sweep, the
    # Study 2 degradation knobs, and a headless smoke test.
    import sys

    from play import main

    raise SystemExit(main(sys.argv[1:]))
