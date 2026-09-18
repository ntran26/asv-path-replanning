"""Gymnasium environment: path following with static obstacles and one target vessel.

CODEX revision 3 uses one synchronized perceived state per decision and exposes
the encounter latch and previous executed action to the policy. Scenarios may
be supplied explicitly or generated with variable corridor geometry. The
training wrapper controls the empty/static/dynamic/combined scene mixture.

    raycast (obstacles only, aft mask, dropout, 1 m dead zone)
        -> gate against the boundary polygon
        -> cluster -> track -> Kalman -> static/dynamic split
        -> CPA/CRI -> encounter class (shared with 02)
        -> six-branch Dict observation (70 values for one target)

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
import emergency_stop as estop_mod
import feasibility as feas
import scenario as scn
import targets as tgtmod
import tracking as trk
from asv_lidar import Lidar
from observation import ObservationBuilder, observation_space
from obstacles import ObstacleSampler
from path import ReferencePath, curved_points, straight_points
from reward import RewardConfig, RewardFunction
from reward import terms as rterms
import ship as shipmod
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
                 facility_walls: bool = cfg.SIMULATE_FACILITY_WALLS,
                 emergency_stop: bool = cfg.EMERGENCY_STOP_ENABLED,
                 low_speed_start_frac: float = 0.0,
                 vessel_randomisation: Optional[float] = cfg.VESSEL_RANDOMISATION_SCALE,
                 command_rate_limit: bool = cfg.RUDDER_COMMAND_LIMIT,
                 pose_stale_prob: float = cfg.POSE_STALE_PROB,
                 motion_classifier: str = cfg.MOTION_CLASSIFIER,
                 scenario_stage: Optional[int] = None,
                 scenario_namespace: str = "training",
                 geometry_mode: Optional[str] = None) -> None:
        super().__init__()
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        # Study 1 sweeps this; the corridor is a map polygon inside the basin, so
        # every simulated width is physically reproducible (03 §5).  F36: a
        # supplied channel reports its own width, not the default.
        if channel is not None:
            self.corridor_width = float(channel.nominal_width)
        else:
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

        # The identified hull (05 part 1).  `vessel_randomisation` draws a new
        # one per episode from the bootstrap; see `reset`.
        self.model = ShipModel()
        self.vessel_randomisation = (None if vessel_randomisation is None
                                     else float(vessel_randomisation))
        # The bridge's optional 50 %/s rudder command limiter.  Off by default in
        # both (`constants.RUDDER_COMMAND_LIMIT`): it stood in for the old
        # simulator's servo, and the identified model predicts raw-command runs
        # at least as well as limited ones.  Whatever this is, the bridge's
        # `--rudder-limit` must match it.
        self.command_rate_limit = bool(command_rate_limit)
        self.estop_enabled = bool(emergency_stop)
        self.estop = estop_mod.EmergencyStop(
            stop_speed=cfg.ESTOP_STOP_SPEED, min_hold_s=cfg.ESTOP_MIN_HOLD_S,
            max_hold_s=cfg.ESTOP_MAX_HOLD_S, max_brake_s=cfg.ESTOP_MAX_BRAKE_S)
        self.lidar = Lidar(aft_mask_half_deg=aft_mask_half_deg,
                           dropout_p=lidar_dropout_p, rng=self._rng)
        self.tracker = trk.Tracker(dropout_p=detection_dropout_p,
                                   velocity_noise=track_velocity_noise,
                                   classifier=motion_classifier,
                                   rng=self._rng)
        # Measured pose staleness (`constants.POSE_STALE_PROB`).  Its own stream,
        # so switching it on or off does not reshuffle any other draw.
        self.pose_stale_prob = float(pose_stale_prob)
        self._stale_rng = np.random.default_rng()
        # F68: a fraction of generated episodes starts slow or at rest, so a
        # policy resuming after a supervisor stop is in distribution.  Its own
        # stream, like staleness, so it reshuffles no other draw.
        self.low_speed_start_frac = float(low_speed_start_frac)
        self._start_rng = np.random.default_rng()
        self.start_speed = float(cfg.U_NOM)

        # C1: episodes from 04a's scenario generator.  `None` keeps the head-on
        # placeholder `_sample_target`, which the older tests are written against.
        self.scenario_stage = None if scenario_stage is None else int(scenario_stage)
        self.scenario_namespace = str(scenario_namespace)
        self._generator: Optional[scn.ScenarioGenerator] = None
        self.scenario = None
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
        # F74: with neither a channel nor a width given, the default geometry is
        # the basin (Paper 2's layout, 06).  A width asks for a channel.
        if geometry_mode is None:
            geometry_mode = ("channel" if channel is not None or corridor_width is not None
                             else cfg.DEFAULT_GEOMETRY_MODE)
        self.geometry_mode = str(geometry_mode)
        if channel is not None:
            self.channel = channel
        elif self.geometry_mode == "basin":
            self.channel = corr.sample_basin(np.random.default_rng(0))
        else:
            self.channel = corr.rectangle(self.corridor_width)
        self.resample_channel = channel is None
        self.boundary_polygon = self.channel.polygon()

        # 03a §1.2: the out-of-corridor world, so the gate has something to do.
        self.facility_walls = bool(facility_walls)
        self.wall_polygon = corr.facility_walls((self.map_width, self.map_height))

        self.forced_num_obs: Optional[int] = None
        self.forced_targets: Optional[List[TargetShip]] = None
        self.observation_space = observation_space(self.n_max_targets)
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
        #
        # 06 M-5: twice the clearance on the side of the deviation.  In a
        # channel `h_+ = h_- = W/2`, so this is exactly `W(s)` there (T14).
        return 2.0 * self.channel.half_width_on_side(
            self.s_along + self.path_start_s, self.cross_track_error)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _clear_state(self) -> None:
        self.step_count = 0
        self._confine_geom = None                # 06 §3.5, set by a generated reset
        self._confine_poly = None
        self.elapsed_time = 0.0
        self.asv_x = self.asv_y = 0.0
        self.asv_h = self.asv_w = 0.0
        self.u_body = self.v_body = 0.0
        self.speed_mps = 0.0
        self.rudder = 0.0
        self.rpm = 0.0
        self.propulsion_s2 = 0.0
        self._estop_request: Optional[str] = None
        self._estop_reverse_dv = 0.0
        self.estop.reset()

        # Deployment timing: whether this frame's pose is the previous frame's,
        # and the previous frame's pose-derived quantities to serve if it is.
        self.pose_stale = False
        self.stale_frames = 0
        self._has_frame = False
        self._pose_hold: Optional[Tuple[float, float, float]] = None
        self._boundary_hold: Optional[np.ndarray] = None
        self._obs_hold: Optional[Tuple[float, ...]] = None
        self._obs_fresh: Optional[Tuple[float, ...]] = None
        self._ego_hold: Optional[Tuple[float, float, float]] = None
        self._obs_cache: Optional[Dict[str, np.ndarray]] = None
        self._executed_action = np.zeros(2, dtype=np.float32)
        self._tracker_dt = 0.0
        self._estop_started = False
        self.scenario = None

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
        self.clip_steps = {branch: 0 for branch in self.observation_space.spaces}
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
        self.gated_ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE)
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
            self._stale_rng = np.random.default_rng(int(seed) + 7_919)
            self._start_rng = np.random.default_rng(int(seed) + 15_485_863)

        self._clear_state()
        if self.vessel_randomisation is not None:
            # A separate stream, so switching randomisation on does not reshuffle
            # the corridor, target and obstacle draws of an otherwise identical
            # seeded episode.
            hull_rng = (np.random.default_rng(int(seed) + 104_729) if seed is not None
                        else self._rng)
            self.model.set_params(shipmod.sample_params(
                hull_rng, scale=self.vessel_randomisation))
        self.model.reset()
        self.lidar.reset()
        self.tracker.reset()
        self.observer.reset()
        if self._pose_noise is not None:
            self._pose_noise.reset()

        options = options or {}
        scenario = options.get("scenario")
        if scenario is not None:
            self._load_scenario(scenario)
        elif options.get("generated") is not None or self.scenario_stage is not None:
            self._load_generated(options.get("generated"))
        else:
            self._sample_layout()

        self.asv_x, self.asv_y = self.start_x, self.start_y
        if self.scenario is not None:
            # 04a's backward solve places the target for an own ship already at
            # `U_nom` on the path heading at t = 0.  Starting from rest instead
            # would delay the own ship by its acceleration and move every CPA.
            self.asv_h = float(self.scenario.own_heading) % 360.0
            self.model._s[3, 0] = math.radians(self.asv_h)
            start_speed = float(cfg.U_NOM)
            if self.low_speed_start_frac > 0.0 and self._start_rng.uniform() < self.low_speed_start_frac:
                start_speed = (0.0 if self._start_rng.uniform() < cfg.LOW_SPEED_START_ZERO_SHARE
                               else float(self._start_rng.uniform(0.0, 0.5 * cfg.U_NOM)))
            self.start_speed = start_speed
            self.model._s[0, 0] = start_speed
            self.u_body = start_speed
        if "obstacles" in options:
            self.obstacles = [[tuple(map(float, p)) for p in poly]
                              for poly in options["obstacles"]]
        self._apply_initial_conditions(options)
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
        self._obs_hold = self._obs_fresh
        self._record_obs_health(obs)
        return obs, {}

    def _apply_initial_conditions(self, options: dict) -> None:
        """Explicit, reproducible starts for recovery training and replay.

        Generated target encounters retain their backward-solved start unless
        the caller requests a perturbation. Invalid requested recovery poses
        raise instead of silently becoming a different training case.
        """
        if "initial_heading_deg" in options:
            self.asv_h = float(options["initial_heading_deg"]) % 360.0
        delta = float(options.get("recovery_heading_deg",
                                  options.get("initial_heading_error_deg", 0.0)))
        half_width = 0.5 * self.channel.width_at_s(self.path_start_s)
        lateral = float(options.get("initial_lateral_offset_m",
                                    float(options.get("recovery_fraction", 0.0)) * half_width))
        tangent = self.path.tangent(0)
        self.asv_x += lateral * float(tangent[1])
        self.asv_y -= lateral * float(tangent[0])
        self.asv_h = (self.asv_h + delta) % 360.0
        self.model._s[3, 0] = math.radians(self.asv_h)
        if "initial_speed" in options:
            speed = float(options["initial_speed"])
            if not np.isfinite(speed) or speed < 0.0:
                raise ValueError("initial_speed must be finite and nonnegative")
            self.start_speed = self.u_body = speed
            self.model._s[0, 0] = speed
        if not np.isfinite([self.asv_x, self.asv_y, self.asv_h]).all():
            raise ValueError("initial pose must be finite")
        if lateral or delta or "initial_heading_deg" in options:
            if self.collision_kind(self.hull_polygon()) is not None:
                raise ValueError("requested recovery start intersects a boundary or obstacle")

    def _sample_layout(self) -> None:
        if self.resample_channel:
            self.channel = (corr.sample_basin(self._rng) if self.geometry_mode == "basin"
                            else corr.sample(self._rng, width_range=(self.corridor_width,
                                                                     self.corridor_width)))
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

    # ------------------------------------------------------------------
    # 04a's scenario generator (C1)
    # ------------------------------------------------------------------
    def set_scenario_stage(self, stage: Optional[int]) -> None:
        """Switch the curriculum stage; takes effect at the next reset."""
        self.scenario_stage = None if stage is None else int(stage)
        self._generator = None

    def _load_generated(self, built=None) -> None:
        """One episode from `scenario.ScenarioGenerator`, or the one supplied.

        Seeds are drawn inside the episode's namespace (04a §9.2) from the
        environment's own stream, so a seeded reset reproduces the episode.
        """
        if built is None:
            stage = self.scenario_stage if self.scenario_stage is not None else 5
            if self._generator is None or self._generator.stage != stage:
                self._generator = scn.ScenarioGenerator(
                    stage=stage, seed_namespace=self.scenario_namespace)
            # C16: the class is drawn once, then retried on cap-out.  Redrawing it
            # with each seed let hard classes lose their share: null caps out on
            # about half its draws and trained at 4.3 % against an intended 11 %.
            stage_spec = cfg.CURRICULUM_STAGES[stage]
            classes = stage_spec["classes"]
            weights = np.array([cfg.CLASS_SAMPLE_WEIGHTS[c] for c in classes], dtype=float)
            cls = str(self._rng.choice(classes, p=weights / weights.sum()))
            for _ in range(20):
                index = int(self._rng.integers(0, 10 ** 9))
                built = self._generator.sample(scn.seed_for(self.scenario_namespace, index),
                                               encounter_class=cls)
                if built is not None:
                    break
            if built is None:
                raise RuntimeError("the scenario generator capped out 20 times running")

        self.scenario = built
        self.channel = built.channel
        self.boundary_polygon = self.channel.polygon()
        self.corridor_width = float(self.channel.nominal_width)
        # 06 §3.5: confined targets keep the path band in basin mode.
        self._confine_geom = scn.confinement_geometry(built.encounter_class, self.channel)
        self._confine_poly = self._confine_geom.polygon()

        start_s = scn.own_start_s(built.encounter_class, self.channel)
        self.path_start_s = float(start_s)
        points = self.channel.reference_path_points(start_s=start_s)
        self.path = ReferencePath(points, self.lookahead_fraction)
        self.start_x, self.start_y = float(points[0][0]), float(points[0][1])
        self.goal_x, self.goal_y = float(points[-1][0]), float(points[-1][1])
        self.path_mode_used = "corridor"
        self.scenario_mode_used = f"generated:{built.encounter_class}"
        self.target_spawn_regime = built.encounter_class

        self.targets = [] if built.encounter_class == "no_target" else [TargetShip(
            float(built.target_spawn[0]), float(built.target_spawn[1]),
            float(built.target_heading), float(built.target_speed),
            behaviour=built.target_behaviour, encounter_class=built.encounter_class,
            confined=bool(built.target_confined))]

        count = int(self.forced_num_obs) if self.forced_num_obs is not None else int(built.n_obstacles)
        self.obstacles = self._obstacles_clear_of_encounter(count, built)

    def _obstacles_clear_of_encounter(self, count: int, built) -> List[List[Tuple[float, float]]]:
        """Static clutter that does not decide the encounter for the policy.

        04a §3.6 keeps `+/- OBSTACLE_CPA_GUARD_FRAC * T_0` of own-ship travel
        around the CPA clear, and a panel on the target's spawn would hide or
        block it.  Obstacles violating either are dropped rather than moved, so
        the realised count can fall short of the request; it is reported in
        `info["num_obs"]`.  Occlusion and conflict placement (04a §3.6's flagged
        cases) are still not written.
        """
        layout = self._feasible_layout(count)
        flagged = self._flagged_panels(built, layout)
        if not layout or built.encounter_class == "no_target":
            return layout + flagged
        guard = cfg.OBSTACLE_CPA_GUARD_FRAC * max(float(built.tcpa_s), 0.0) * cfg.U_NOM
        s_cpa = max(float(built.tcpa_s), 0.0) * cfg.U_NOM
        tx, ty = float(built.target_spawn[0]), float(built.target_spawn[1])
        kept = []
        for poly in layout:
            cx = float(np.mean([p[0] for p in poly]))
            cy = float(np.mean([p[1] for p in poly]))
            s_obs = float(self.path.project(cx, cy, 0.0).s_along)
            if guard > 0.0 and abs(s_obs - s_cpa) <= guard:
                continue
            if float(np.hypot(cx - tx, cy - ty)) < 2.0:
                continue
            kept.append(poly)
        return kept + flagged

    def _flagged_panels(self, built, layout) -> List[List[Tuple[float, float]]]:
        """04a §3.6's flagged panels, placed on purpose (F75).

        * `conflict`: a panel beside the path, just before CPA, on the side the
          compliant alteration would use -- so the textbook turn runs out of
          water and the vessel must slow, alter less, or pass the other way.
        * `occlusion`: a panel on the initial line of sight to the target, off
          the path, so the target is hidden until the geometry opens.  Its
          realised occlusion time is measured by the tracker, not set here.

        Both are kept only if the layout stays A*-feasible, moving outward in
        0.25 m steps until it is.  They are exempt from the CPA guard, which is
        exactly what they are for.
        """
        flags = dict(getattr(built, "flags", {}) or {})
        if built.encounter_class == "no_target" or not (flags.get("conflict") or flags.get("occlusion")):
            return []
        from colregs.context import compliant_turn_sense
        start, goal = (self.start_x, self.start_y), (self.goal_x, self.goal_y)
        pts = np.asarray(self.path.points, dtype=float)
        t = (pts[-1] - pts[0]) / max(float(np.linalg.norm(pts[-1] - pts[0])), 1e-9)
        starboard = np.array([t[1], -t[0]])
        half = 0.5 * cfg.OBSTACLE_SIZE
        out: List[List[Tuple[float, float]]] = []

        def box(c):
            return [(float(c[0] - half), float(c[1] - half)), (float(c[0] + half), float(c[1] - half)),
                    (float(c[0] + half), float(c[1] + half)), (float(c[0] - half), float(c[1] + half))]

        def place(centre, step_dir):
            for k in range(8):
                panel = box(centre + step_dir * 0.25 * k)
                inside = br.points_in_polygon(np.array([p[0] for p in panel]),
                                              np.array([p[1] for p in panel]), self.boundary_polygon)
                if np.all(inside) and feas.layout_feasible(start, goal, self.boundary_polygon,
                                                           layout + out + [panel]):
                    return panel
            return None

        s_cpa = max(float(built.tcpa_s), 0.0) * cfg.U_NOM
        if flags.get("conflict"):
            side = ("port" if float(getattr(built, "ct_deg", 0.0)) < 180.0 else "starboard")
            sense = compliant_turn_sense(built.encounter_class, side) or 1
            p = pts[0] + t * max(s_cpa - 1.0, 2.0)
            lateral = starboard * float(sense)
            panel = place(p + lateral * 1.5, lateral)
            if panel is not None:
                out.append(panel)
        if flags.get("occlusion"):
            own = pts[0]
            target = np.array(built.target_spawn, dtype=float)
            centre = own + 0.4 * (target - own)
            away = centre - (pts[0] + t * float(np.dot(centre - pts[0], t)))
            norm = float(np.linalg.norm(away))
            away = away / norm if norm > 1e-6 else starboard
            if norm < 1.2:
                centre = centre + away * (1.2 - norm)
            panel = place(centre, away)
            if panel is not None:
                out.append(panel)
        self.flagged_panels = len(out)
        return out

    def _feasible_layout(self, count: int) -> List[List[Tuple[float, float]]]:
        """A layout that leaves a route from start to goal (F74).

        Redrawn up to `FEASIBILITY_REDRAWS` times, then thinned nearest-the-path
        first.  `info["layout_redraws"]` and `info["layout_thinned"]` record it.
        """
        start, goal = (self.start_x, self.start_y), (self.goal_x, self.goal_y)
        nav = self.boundary_polygon
        self.layout_redraws, self.layout_thinned = 0, 0
        layout = self.sample_obstacles(count)
        while layout and not feas.layout_feasible(start, goal, nav, layout):
            if self.layout_redraws >= int(cfg.FEASIBILITY_REDRAWS):
                kept = feas.thin_to_feasible(start, goal, nav, layout, self.path.points)
                self.layout_thinned = len(layout) - len(kept)
                return kept
            self.layout_redraws += 1
            layout = self.sample_obstacles(count)
        return layout

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
        # outside the channel and terminates on step 1.  F35: the inset must also
        # clear `d_safe`, or `r_bnd` charges the stern for sitting on the end
        # edge for the first 16-19 steps of every episode.
        # A basin leg starts at Paper 2's start y, already clear of the wall.
        inset = 0.0 if getattr(self.channel, "mode", "channel") == "basin" else cfg.SPAWN_INSET_M
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
        """The last received pose; reading it never draws more sensor noise."""
        if self._pose_hold is None:
            raise RuntimeError("reset the environment before reading its estimated pose")
        return self._pose_hold

    def _perceive(self) -> None:
        """Raycast -> gate -> pool -> cluster -> track."""
        self._obs_cache = None
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

        # **Deployment timing.**  On a stale frame the bridge has this frame's
        # scan and the previous frame's pose.  Every pose-derived quantity
        # therefore repeats, and the tracker is not fed: the bridge can tell,
        # because the pose timestamp has not moved, and lifting a fresh scan with
        # an old pose would move every static object by a step's travel.
        self.pose_stale = bool(self._has_frame and self.pose_stale_prob > 0.0
                               and self._stale_rng.random() < self.pose_stale_prob)
        if self.pose_stale:
            self.stale_frames += 1
        else:
            true_pose = (self.asv_x, self.asv_y, self.asv_h)
            self._pose_hold = (self._pose_noise.perturb(*true_pose)
                               if self._pose_noise is not None else true_pose)
            self._ego_hold = self._sample_ego()
        # Consecutive stale frames hold the last estimate actually received,
        # not an unavailable fresh estimate from the preceding simulation step.
        est_x, est_y, est_h = self.estimated_pose()
        self._has_frame = True

        # Returns are lifted from the **sensor**, which the raycast casts from
        # `LIDAR_OFFSET_M` ahead of the vessel origin.  Lifting from the origin
        # moved every point 0.86 m along the heading, so static objects appeared
        # to move whenever the vessel turned.
        sensor_x, sensor_y = trk.sensor_origin(est_x, est_y, est_h)

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
                              sensor_x, sensor_y, est_h, self.boundary_polygon)
        self.gated_ranges = gated.copy()

        # **Pool from the gated scan, not the raw one.**  The method docstring
        # above has always said "raycast -> gate -> pool -> cluster"; the code
        # pooled before gating, which returned the same answer for as long as
        # the gate had nothing to remove.  With the facility walls returned it
        # does not: 624 of 720 beams reached the obstacle branch.
        self.lidar.repool(gated)
        self.sector_closeness = self.lidar.sector_closeness
        self.boundary_closeness = br.boundary_scan(
            est_x, est_y, est_h, self.boundary_polygon)
        self._boundary_hold = self.boundary_closeness

        if self.pose_stale:
            self._tracker_dt += cfg.UPDATE_RATE
        else:
            detections = trk.segment_scan(gated, self.lidar.bearings,
                                          sensor_x, sensor_y, est_h)
            scan = trk.ScanFrame.from_scan(self.raw_ranges, self.lidar.bearings,
                                           (sensor_x, sensor_y), est_h)
            self.tracker.update(detections, self._tracker_dt + cfg.UPDATE_RATE, scan=scan)
            self._tracker_dt = 0.0
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
        if self._obs_cache is not None:
            return {key: value.copy() for key, value in self._obs_cache.items()}
        u, v, r = self._measured_ego()
        x, y, heading = self.estimated_pose()
        a = math.radians(heading)
        velocity = np.array([u * math.sin(a) + v * math.cos(a),
                             u * math.cos(a) - v * math.sin(a)])
        # Course is not observable at rest. Use heading below the estimate's
        # noise floor instead of turning near-zero velocity jitter into 180 deg.
        course = (math.degrees(math.atan2(velocity[0], velocity[1]))
                  if math.hypot(u, v) > max(0.10, 2.0 * self.ego_speed_noise)
                  else heading)
        perceived = self.path.project(x, y, course)
        cte, chi, chi_la = (perceived.cross_track_error, perceived.course_error,
                            perceived.lookahead_course_error)
        self._obs_fresh = (u, v, r, cte, chi, chi_la)
        self.perceived_path_state = perceived
        self.perceived_velocity = velocity
        # The surge the controller perceived in this observation: what the stop
        # latch reads next step (F68).  It used to draw its own noisy estimate
        # from the shared stream, so switching the supervisor on shifted every
        # later noise draw and an on/off comparison changed more than the stop.
        self._observed_surge = float(u)
        observation = self.observer.build(
            sector_closeness=self.sector_closeness,
            boundary_scan=self.boundary_closeness,
            u=u, v=v, yaw_rate_degps=r,
            cross_track_error=cte,
            course_error_deg=chi,
            lookahead_course_error_deg=chi_la,
            tracks=self.tracks,
            p_os=(x, y),
            v_os=velocity,
            heading_os_deg=heading,
            # The map side of `R-1`.  This environment is the only object
            # holding both the perception output and the boundary polygon, so
            # the admissibility predicate is fed from here rather than
            # rediscovered inside the observation builder.
            r_path=self.path.yaw_rate_for_tracking(perceived.closest_idx, u),
            path=self.path,
            boundary_polygon=self.boundary_polygon,
            s_along=perceived.s_along,
            true_targets=(),
            open_water=self.open_water,
            previous_action=self._executed_action,
            cross_track_scale=self.channel.half_width_on_side(
                perceived.s_along + self.path_start_s, cte),
        )
        # Diagnostic truth uses a wholly true reference frame. It never enters
        # the policy branches or the context's perceived CPA/admissibility.
        self.observer.contexts.attach_truth(
            tracks=self.tracks, p_os=(self.asv_x, self.asv_y),
            v_os=self._own_velocity(), heading_os_deg=self.asv_h,
            true_targets=self.targets, path=self.path, s_along=self.s_along)
        self._obs_cache = observation
        return {key: value.copy() for key, value in observation.items()}

    @property
    def encounter_contexts(self):
        """The per-step `EncounterContext` per track, as of the last observation."""
        return self.observer.encounter_contexts

    def _measured_ego(self) -> Tuple[float, float, float]:
        """Last synchronized ego estimate, also read by the stop latch."""
        if self._ego_hold is None:
            raise RuntimeError("reset the environment before reading its ego estimate")
        return self._ego_hold

    def _sample_ego(self) -> Tuple[float, float, float]:
        """Draw once when a fresh telemetry frame arrives."""
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
        # Signed CTE is zero on the exact extension of a straight path, even
        # beyond its endpoint. Do not let that degenerate sign report success
        # after overshooting the goal capture region.
        delta = np.array([self.asv_x, self.asv_y]) - self.path.points[-1]
        if (float(delta @ self.path.tangent(len(self.path) - 1)) > 0.0
                and float(np.linalg.norm(delta)) > cfg.GOAL_CTE_RADIUS):
            return False
        remaining = self.path.length - float(self.path.s[self.closest_idx])
        return remaining <= cfg.GOAL_ALONG_DIST and abs(self.cross_track_error) <= cfg.GOAL_CTE_RADIUS

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        self.elapsed_time += cfg.UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Rudder: the bridge's 50 %/s command limiter, when it is enabled.
        commanded = rudder_cmd * 100.0
        if self.command_rate_limit:
            max_step = shipmod.COMMAND_RATE_PCT_S * cfg.UPDATE_RATE
            commanded = self.rudder + float(np.clip(commanded - self.rudder,
                                                    -max_step, max_step))
        self.rudder = commanded

        # Propulsion: the policy's forward command, unless the stop latch holds.
        policy_rpm = cfg.CRUISE_RPM if cfg.FIXED_RPM else float(np.clip(
            cfg.CRUISE_RPM + cfg.RPM_DELTA * throttle_cmd, cfg.RPM_FLOOR, cfg.RPM_CEIL))
        self._estop_started = False
        override = self._emergency_stop_override()
        if override is None:
            self.rpm = policy_rpm
            self.propulsion_s2 = estop_mod.rpm_to_s2(policy_rpm)
        else:
            self.propulsion_s2 = float(override)
            self.rpm = estop_mod.s2_to_rpm(override)
        self._executed_action = np.array([
            np.clip(self.rudder / 100.0, -1.0, 1.0),
            np.clip((self.rpm - cfg.CRUISE_RPM) / max(cfg.RPM_DELTA, 1e-6), -1.0, 1.0),
        ], dtype=np.float32)

        x_before, y_before = self.asv_x, self.asv_y

        # **The command holds for the whole decision interval; the physics and
        # the collision test do not.**  At 2 Hz a closing target moves up to
        # 1.3 m per step, more than a hull's beam, so testing only at the
        # decision instant would let hulls pass through each other.
        n_sub = max(1, int(round(cfg.UPDATE_RATE / cfg.PHYSICS_DT)))
        h = cfg.UPDATE_RATE / n_sub
        astern_impulse = 0.0
        sub_collision = None
        for _ in range(n_sub):
            dx, dy, heading, yaw_rate = self.model.update(self.rpm, self.rudder, h)
            astern_impulse += self.model.last_astern_impulse
            self.asv_x += dx
            self.asv_y += dy
            self.asv_h = heading
            self.asv_w = yaw_rate
            self.u_body = self.model.u
            self.v_body = self.model.v

            own_state = {"x": self.asv_x, "y": self.asv_y,
                         "velocity": self._own_velocity(), "heading": self.asv_h}
            for target in self.targets:
                target.step(h, own=own_state)
                # Confined classes keep the fairway; a crossing target under Rule
                # 9(d) is not a channel user and is left alone (03a §5.2).
                tgtmod.clamp_to_corridor(target, self._confine_geom or self.channel,
                                         self._confine_poly or self.boundary_polygon)
            sub_collision = self.collision_kind(self.hull_polygon())
            if sub_collision is not None:
                break
        if self.estop.state == estop_mod.BRAKING:
            self._estop_reverse_dv += astern_impulse / shipmod.M11

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
        collision = sub_collision
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
        self._obs_hold = self._obs_fresh
        self._record_obs_health(obs)

        breakdown = self._reward(action, collision, reached_goal, truncated)
        info = self._build_info(breakdown, rudder_cmd, throttle_cmd,
                                collision, reached_goal, truncated)
        self.prev_action = np.array([rudder_cmd, throttle_cmd], dtype=np.float32)
        self.prev_s_along = self.s_along
        self.render()
        return obs, float(breakdown.total), terminated, truncated, info

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------
    def request_emergency_stop(self, reason: str = "manual") -> None:
        """Ask for a stop on the next step -- the play harness's E key, tests."""
        self._estop_request = str(reason)

    def _emergency_stop_override(self) -> Optional[float]:
        """Run the stop latch for this step; return the S2 override or None.

        Reads the contexts from the **previous** observation, which is what the
        controller perceived when it chose this step's action -- and speed as
        the controller measures it, noise included, because the field latch
        will only ever have the estimate.
        """
        if not self.estop_enabled:
            self._estop_request = None
            return None

        contexts = list(self.observer.encounter_contexts.values())
        speed = float(getattr(self, "_observed_surge", self.u_body))
        was_braking = self.estop.state == estop_mod.BRAKING

        reason, self._estop_request = self._estop_request, None
        if reason is None and cfg.ESTOP_TRIGGER == "supervisor":
            reason = estop_mod.stop_required(contexts)
        if reason is not None and self.estop.request(reason, t=self.elapsed_time,
                                                     speed=speed):
            self._estop_reverse_dv = 0.0
            self._estop_started = True

        return self.estop.update(
            t=self.elapsed_time, dt=cfg.UPDATE_RATE, speed=speed,
            distance_step_m=self.speed_mps * cfg.UPDATE_RATE if was_braking else 0.0,
            release_ok=estop_mod.danger_passed(contexts))

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
            perceived_heading_deg=float(self.estimated_pose()[2]),
            perceived_u=float(self._measured_ego()[0]),
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
            estop_active=bool(self.estop.active),
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
            collision=collision, reached_goal=reached_goal, truncated=truncated,
            estop_triggered=bool(self._estop_started))
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
            "propulsion_s2": float(self.propulsion_s2),
            "estop/state": self.estop.state,
            "estop/active": bool(self.estop.active),
            "estop/events": int(len(self.estop.events)),
            "estop/reason": self.estop.events[-1].reason if self.estop.active else "",
            "estop/braking_force_n": float(self.model.last_braking_force),
            # Full astern applied after surge reached zero, as the astern speed it
            # would have produced.  The hull clips surge at zero, so the overshoot
            # is not simulated -- this is what the 2 Hz latch would impart on the
            # water before it sees the vessel stopped.
            "estop/reverse_dv_est_mps": float(self._estop_reverse_dv),
            "pose_stale": bool(self.pose_stale),
            "stale_frames": int(self.stale_frames),
            # The rudder the vessel is commanded, after the bridge limiter --
            # not the policy's action, which `action_rudder` records.
            "rudder_deg": float(self.rudder / 100.0 * MAX_RUD_ANGLE),
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
            "scenario_class": (self.scenario.encounter_class if self.scenario is not None
                               else "placeholder"),
            "num_obs": int(len(self.obstacles)),
            "layout_redraws": int(getattr(self, "layout_redraws", 0)),
            "layout_thinned": int(getattr(self, "layout_thinned", 0)),
            "geometry_mode": str(getattr(self.channel, "mode", "channel")),
            "start_speed": float(self.start_speed),
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
                "s2": float(self.propulsion_s2),
                "estop_state": self.estop.state,
                "estop_reason": self.estop.events[-1].reason if self.estop.active else "",
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
                names = branch_feature_names(branch, n_slots=self.n_max_targets)
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
    if branch == "context":
        from observation import CONTEXT_FEATURES, NORMALISED_CONTEXT_INDICES
        return np.array([slot * CONTEXT_FEATURES + i
                         for slot in range((size - 2) // CONTEXT_FEATURES)
                         for i in NORMALISED_CONTEXT_INDICES], dtype=np.int64)
    if branch != "target":
        return np.arange(size)
    from observation import NORMALISED_SLOT_INDICES
    return np.array([slot * cfg.TARGET_FEATURES + i
                     for slot in range(size // cfg.TARGET_FEATURES)
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
