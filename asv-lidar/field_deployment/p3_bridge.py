"""
Paper 3 field bridge, core: the deployed policy and the E1 safety layer on live sensor frames.

No sockets and no display here.  `udp_live_p3.py` is the live bridge built on this module;
the bench scripts (`p3_parity_replay.py`, `p3_shadow_replay.py`, `p3_timing.py`) drive it
directly.  The design is `static_dynamic_obstacles/planning/FIELD_TRIALS_PLAN.md` sections 3
and 5.

Observation.  `LiveEnv` is the simulator's `ASVLidarEnv` with its sensor models replaced by
supplied frames: the raycast by the measured scan, the pose estimate and the ego measurement
by the measured ones, the staleness draw by the bridge's pose-sync flag.  The environment's
own `_perceive` and `_get_obs` then run unchanged: the gate against the map polygon, the 27
pooled sectors, the 7 map-boundary rays at the estimated pose, clustering, the tracker,
encounter classification and latch, and the 70-value baseline-v3 observation.  Nothing of
the perception is copied into this file.

Safety layer.  E1 (`safety_v21.SafetyFilterV21EC`, W = 2.5 s, boundary-and-targets trigger,
no braking fallback) reads the same `LiveEnv` as its environment interface: estimated pose,
measured ego motion, tracks and encounter contexts, the boundary polygon, the gated scan,
and the issued command through `observe_issued_command`.  In shadow mode (decision D-4) it
decides and is logged at every decision, and nothing it decides reaches the vessel: its
brake request is cleared, its actuator history is advanced with the command actually issued
(the policy's), and the issued-command callback receives that command.  Its plan memory and
its stop commitment remain its own, so the first would-be stop is permanent, as on the water.

Command.  `resolve_command` is `ASVLidarEnv.step`'s command resolution: rudder
`100 * clip(a0)` with no limiter (`RUDDER_COMMAND_LIMIT = False`), propulsion
`CRUISE_RPM + RPM_DELTA * a1` clipped to the deployed stage's 0-12 rpm-units, a stop
override (the operator's latch, or the layer's brake in active mode) as an S2 value, and the
executed action that the next observation carries.

Conventions at the field boundary (`frame_to_sensor`): the decoder's 720 beams run
clockwise from `lidar_index0_deg`, the simulator's from dead ahead; the decoder's sway is
positive to port, the simulator's to starboard; headings are compass degrees in both.
"""

from __future__ import annotations

import os

# The evaluated configuration of the safety layer runs one native thread (the V21 records);
# set before numpy loads.  An environment that already sets them keeps its values.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import copy
import csv
import dataclasses
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.normpath(os.path.join(HERE, "..", "static_dynamic_obstacles"))
SIM_SRC = os.path.join(SIM_ROOT, "src")

# This directory carries Paper 2's own `asv_lidar.py` and `lidar_pooling.py`.  The
# simulator's modules of the same names must win, so its source goes first on the path, and
# an import that already resolved to this directory's copies is refused rather than mixed.
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
for _name in ("asv_lidar", "lidar_pooling"):
    _mod = sys.modules.get(_name)
    if _mod is not None and os.path.dirname(os.path.abspath(_mod.__file__)) != os.path.abspath(SIM_SRC):
        raise ImportError(f"{_name} is already imported from {_mod.__file__}; import p3_bridge first")

import asv_lidar as sim_lidar  # noqa: E402
import constants as cfg  # noqa: E402
import corridor as corr  # noqa: E402
import curriculum  # noqa: E402
import emergency_stop as estop_mod  # noqa: E402
import observation as obsmod  # noqa: E402
import safety_estop  # noqa: E402
import safety_v21  # noqa: E402
import scenario as scn  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

# The deployed policy's propulsion stage (0-12 rpm-units about CRUISE_RPM = 6), as every
# evaluation of it applies.
curriculum.apply_stage(tf.PROPULSION_STAGE)

DEFAULT_MODEL = os.path.join(SIM_ROOT, "runs", "sac_formulation_seed0_bl3", "best_model.zip")
EXPECTED_MODEL_SHA256 = "993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8"
TEST_SET_DIR = os.path.join(SIM_ROOT, "results", "test_set", "v4")

# E1: V16 with the permanent stop, boundary-and-targets trigger, collision-hull detector,
# W = 2.5 s (FIELD_TRIALS_PLAN.md section 2; mode `v21_e1_estop_bt` of the V21 records).
E1_CLASS = safety_v21.SafetyFilterV21EC
E1_OPTIONS = {"estop_window_s": 2.5, "use_static_points": False, "braking_fallback": False}
# Bulky per-decision diagnostics left out of the decision log unless asked for
# (the compact traces of the V21 records drop the same keys).
COMPACT_DROP = ("risk_monitor", "track_persistence", "motion_axis_geometry", "policy_search",
                "escape_search", "v11_track_admission", "v11_track_persistence")

RUDDER_SIGN = -1.0          # the vessel's rudder command has the opposite sign to the simulator's
S2_LIMITS = (estop_mod.S2_FULL_ASTERN, 100.0)


# ---------------------------------------------------------------------------
# Sensor frames
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SensorFrame:
    """One decision's sensor input, in the simulator's frame and units.

    ranges        (720,) m, the simulator's beam order: bin 0 dead ahead, clockwise, 0.5 deg
    pose          estimated (x, y) in the map frame, m, and heading, compass deg
    ego           measured surge, sway (positive to starboard), m/s, and yaw rate, deg/s
                  (positive clockwise)
    pose_stale    the frame carries the previous frame's pose (no new pose line arrived)
    missed_cycles whole decision intervals since the previous frame, less one
    """
    t: float
    ranges: np.ndarray
    pose: Tuple[float, float, float]
    ego: Tuple[float, float, float]
    pose_stale: bool = False
    missed_cycles: int = 0


class _Draw:
    """Stands in for the staleness stream: staleness is supplied, never drawn."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self) -> float:
        return self.value


class LiveEnv(ASVLidarEnv):
    """`ASVLidarEnv` whose sensor frames come from outside: the vessel, or a recorded stream.

    Nothing is simulated: no physics, no targets, no reward.  The pose and ego attributes the
    environment would hold as truth hold the estimate instead, since the field has no other.
    The Study 2 degradation draws (beam dropout, track dropout, velocity noise) are off: the
    supplied data carry their own.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(render_mode=None, emergency_stop=False, vessel_randomisation=None,
                         low_speed_start_frac=0.0, lidar_dropout_p=0.0, detection_dropout_p=0.0,
                         track_velocity_noise=0.0, pose_stale_prob=0.0, **kwargs)
        self._pose_noise = None                 # the supplied pose is already the estimate
        self.lidar.scan = self._supplied_scan
        self._stale_rng = _Draw(0.0)
        self._frame: Optional[SensorFrame] = None
        self.frames = 0

    # -- the sensor models, replaced ------------------------------------
    def _supplied_scan(self, pos, heading_deg: float, obstacles=None) -> np.ndarray:
        """The raycast's place: the measured scan through the simulator's sensor model
        (the 1 m dead zone and the aft mask; dropout is off), then pooled."""
        lidar = self.lidar
        lidar.heading = float(heading_deg)
        ranges = np.asarray(self._frame.ranges, dtype=np.float64)
        if ranges.shape != (cfg.LIDAR_BEAMS,) or not np.all(np.isfinite(ranges)):
            raise ValueError(f"a scan needs {cfg.LIDAR_BEAMS} finite ranges, got shape {ranges.shape}")
        lidar.ranges = lidar._degrade(sim_lidar.apply_min_range(ranges))
        lidar._pool()
        return lidar.ranges

    def _sample_ego(self) -> Tuple[float, float, float]:
        return self.u_body, self.v_body, self.asv_w

    def _load_generated(self, built=None, _depth: int = 0) -> None:
        super()._load_generated(built, _depth)
        self.targets = []                        # the target is real; the scene holds none

    def _perceive(self) -> None:
        frame = self._frame
        if frame is None:
            raise RuntimeError("supply a sensor frame before perceiving")
        # The environment never treats its first frame as stale, so that frame's pose is used
        # whatever the flag says (it is the latest the decoder holds).
        if not frame.pose_stale or not self._has_frame:
            self.asv_x, self.asv_y, self.asv_h = (float(v) for v in frame.pose)
            self.u_body, self.v_body, self.asv_w = (float(v) for v in frame.ego)
        # The environment's staleness rule, with the flag supplied instead of drawn.
        self.pose_stale_prob = 1.0 if frame.pose_stale else 0.0
        # A lost frame: the tracker predicts over the whole gap, as over a stale frame.
        self._tracker_dt += cfg.UPDATE_RATE * int(frame.missed_cycles)
        super()._perceive()
        self.frames += 1

    # -- the bridge's two calls ------------------------------------------
    def begin(self, built, frame: SensorFrame, *, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Load the scenario's map and path and perceive the first frame (the reset observation)."""
        self._frame = frame
        obs, _ = self.reset(seed=seed, options={"generated": copy.deepcopy(built)})
        self._stale_rng = _Draw(0.0)             # a seeded reset replaces the stream
        return obs

    def observe(self, frame: SensorFrame, executed_action) -> Dict[str, np.ndarray]:
        """Perceive the next frame; `executed_action` is the command issued since the last one."""
        self._frame = frame
        self._executed_action = np.asarray(executed_action, dtype=np.float32).reshape(2).copy()
        self.elapsed_time += cfg.UPDATE_RATE * (1 + int(frame.missed_cycles))
        self.step_count += 1
        self._perceive()
        obs = self._get_obs()
        self._obs_hold = self._obs_fresh
        self._record_obs_health(obs)
        return obs


# ---------------------------------------------------------------------------
# Stopping profiles, precomputed
# ---------------------------------------------------------------------------
# The encounter context's slowdown test (`colregs.context.slowdown_clears`) integrates a stop
# on the identified hull, `stopping._profile`, cached per cm/s of measured surge: about 0.55 s
# per new value.  A long simulation warms that cache; a bridge starts cold, and measured
# surge from 0.1 m poses jumps between values, so each new one held a decision for half a
# second (July logs: 32 of 49 decisions over 0.5 s).  The table holds the function's own
# results for every key up to STOPPING_TABLE_MAX_CMPS, computed by the function itself
# (`build_stopping_table`), and serves them in its place; any other key still goes to the
# function.  It is refused when the sources or constants it depends on have changed.
STOPPING_TABLE = os.path.join(HERE, "p3_stopping_profiles.pkl")
STOPPING_TABLE_MAX_CMPS = 300
_STOPPING_CONSTANTS = ("STOP_TEST_DT_S", "ESTOP_MAX_BRAKE_S", "ESTOP_STOP_SPEED", "POLICY_SLOWDOWN_RPM",
                       "SLOWDOWN_TEST_MAX_S")


def _stopping_sources() -> List[str]:
    files = [os.path.join(SIM_SRC, n) for n in ("stopping.py", "ship.py", "emergency_stop.py", "constants.py")]
    bluefin = os.path.join(SIM_ROOT, "bluefin")
    return files + sorted(os.path.join(bluefin, n) for n in os.listdir(bluefin) if n.endswith((".py", ".json")))


def stopping_fingerprint() -> str:
    h = hashlib.sha256()
    for path in _stopping_sources():
        h.update(os.path.relpath(path, SIM_ROOT).replace(os.sep, "/").encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    h.update(repr([(n, getattr(cfg, n)) for n in _STOPPING_CONSTANTS]).encode())
    return h.hexdigest()


class _TabledProfile:
    """`stopping._profile` with precomputed results; keys outside the table go to the function."""

    def __init__(self, native, table: Dict) -> None:
        self.native, self.table = native, table
        self.hits = self.misses = 0

    def __call__(self, u0_cmps: int, mode: str = "latch"):
        hit = self.table.get((int(u0_cmps), mode))
        if hit is None:
            self.misses += 1
            return self.native(u0_cmps, mode)
        self.hits += 1
        return hit


def _native_profile():
    import stopping
    f = stopping._profile
    return f.native if isinstance(f, _TabledProfile) else f


def build_stopping_table(path: str = STOPPING_TABLE, max_cmps: int = STOPPING_TABLE_MAX_CMPS) -> Dict:
    native = _native_profile()
    table = {}
    start = time.perf_counter()
    for mode in ("latch", "coast"):
        for u in range(int(max_cmps) + 1):
            t, s = native(u, mode)
            table[(u, mode)] = (np.array(t), np.array(s))
    with open(path, "wb") as fh:
        pickle.dump({"fingerprint": stopping_fingerprint(), "max_cmps": int(max_cmps), "table": table}, fh)
    return {"path": path, "entries": len(table), "seconds": time.perf_counter() - start}


def use_stopping_table(path: str = STOPPING_TABLE, *, spot_checks: int = 3) -> Dict:
    """Serve `stopping._profile` from the table; spot-check some entries against the function."""
    import stopping
    if isinstance(stopping._profile, _TabledProfile):
        return {"installed": True, "path": path, "already": True}
    if not os.path.exists(path):
        return {"installed": False, "reason": f"no table at {path}"}
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if data.get("fingerprint") != stopping_fingerprint():
        return {"installed": False, "reason": "the table predates a change to its sources or constants"}
    native, table = stopping._profile, data["table"]
    keys = sorted(table)
    for key in [keys[int(i)] for i in np.linspace(0, len(keys) - 1, max(0, spot_checks)).round()]:
        t, s = native(*key)
        if not (np.array_equal(t, table[key][0]) and np.array_equal(s, table[key][1])):
            return {"installed": False, "reason": f"entry {key} differs from the function"}
    stopping._profile = _TabledProfile(native, table)
    return {"installed": True, "path": path, "entries": len(table), "max_cmps": data["max_cmps"]}


@dataclasses.dataclass
class _NativeStopping:
    """Context manager: the function itself, for a simulator reference pass."""
    saved: Any = None

    def __enter__(self):
        import stopping
        self.saved = stopping._profile
        stopping._profile = _native_profile()
        return self

    def __exit__(self, *exc):
        import stopping
        stopping._profile = self.saved
        return False


def native_stopping() -> _NativeStopping:
    return _NativeStopping()


# ---------------------------------------------------------------------------
# Scenario geometry
# ---------------------------------------------------------------------------
def field_scenario(start, goal, *, panels: Sequence = (), case_id: str = "field") -> scn.Scenario:
    """A basin leg from start to goal, as the simulator builds one (`corridor.build_basin`).

    `panels` (polygons) are the layout's known panels; perception does not read them (the
    scan does), they are kept for the log and the display.
    """
    channel = corr.build_basin(tuple(map(float, start)), tuple(map(float, goal)))
    dx, dy = float(goal[0]) - float(start[0]), float(goal[1]) - float(start[1])
    flags = {"fixed_obstacles": [[tuple(map(float, p)) for p in poly] for poly in panels]} if len(panels) else {}
    built = scn.Scenario(case_id=case_id, encounter_class="no_target", seed=0, n_obstacles=0,
                         flags=flags, geometry_mode="basin",
                         own_heading=math.degrees(math.atan2(dx, dy)) % 360.0)
    built.channel = channel
    return built


def layout_scenario(name: str) -> scn.Scenario:
    """Paper 2's basin layouts L1-L3 (`paper2_set.LAYOUTS`), the field trials' layouts."""
    import paper2_set
    layout = paper2_set.LAYOUTS[name.upper()]
    return field_scenario(layout["start"], layout["goal"], panels=paper2_set.panel_polygons(name.upper()),
                          case_id=f"layout-{name.upper()}")


def load_test_case(test_id: str) -> Tuple[scn.Scenario, int]:
    """A test set v4 scenario and its episode seed, checked against the set's definition."""
    test_id = test_id.split(":", 1)[-1]
    with open(os.path.join(TEST_SET_DIR, "definition.csv"), newline="", encoding="utf-8") as fh:
        rows = {row["test_id"]: row for row in csv.DictReader(fh)}
    if test_id not in rows:
        raise KeyError(f"{test_id} is not in test set v4")
    with open(os.path.join(TEST_SET_DIR, "set_v4.0.pkl"), "rb") as fh:
        kept, _ = pickle.load(fh)
    entry = next(item for item in kept if item["test_id"] == test_id)
    built, seed = entry["built"], int(entry["episode_seed"])
    if built.digest() != rows[test_id]["digest"] or seed != int(rows[test_id]["episode_seed"]):
        raise ValueError(f"{test_id}: cached scenario does not match the set definition")
    return built, seed


def geometry_record(env: LiveEnv) -> Dict[str, Any]:
    built = env.scenario
    return {"case_id": getattr(built, "case_id", None),
            "encounter_class": getattr(built, "encounter_class", None),
            "start": [env.start_x, env.start_y], "goal": [env.goal_x, env.goal_y],
            "path_start_s": env.path_start_s,
            "boundary_polygon": jsonable(env.boundary_polygon),
            "path": jsonable(np.asarray(env.path.points)),
            "panels": jsonable(env.obstacles)}


# ---------------------------------------------------------------------------
# Field telemetry -> sensor frame
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PoseTransform:
    """Localisation frame -> map frame: rotate clockwise by `rotation_deg` about the origin,
    then translate by (dx, dy).  Headings gain `rotation_deg`."""
    rotation_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0

    def apply(self, x: float, y: float, heading_deg: float) -> Tuple[float, float, float]:
        a = math.radians(self.rotation_deg)
        xm = float(x) * math.cos(a) + float(y) * math.sin(a) + self.dx
        ym = -float(x) * math.sin(a) + float(y) * math.cos(a) + self.dy
        return xm, ym, (float(heading_deg) + self.rotation_deg) % 360.0

    @classmethod
    def anchored(cls, telemetry_xy, map_xy, rotation_deg: float = 0.0) -> "PoseTransform":
        """The transform that puts `telemetry_xy` at `map_xy` (Paper 2's origin rule)."""
        base = cls(rotation_deg=rotation_deg).apply(telemetry_xy[0], telemetry_xy[1], 0.0)
        return cls(rotation_deg, float(map_xy[0]) - base[0], float(map_xy[1]) - base[1])


def frame_to_sensor(frame, transform: PoseTransform, *, lidar_index0_deg: float = 0.0,
                    pose_stale: bool = False, missed_cycles: int = 0, pose_point: str = "origin") -> SensorFrame:
    """A decoded `log_parser.BluefinFrame` in the simulator's frame and conventions.

    `pose_point` names the point the telemetry pose belongs to: "origin", the simulator's
    vessel reference point (the July bridge's assumption), or "sensor", the LiDAR,
    `LIDAR_OFFSET_M` ahead of it on the centreline; a sensor pose is moved back to the
    reference point and its sway freed of the lever arm's `r * offset`.
    """
    lidar = np.asarray(frame.lidar_m, dtype=np.float32)
    if lidar.shape != (cfg.LIDAR_BEAMS,):
        raise ValueError(f"the decoder must emit {cfg.LIDAR_BEAMS} beams, got {lidar.shape}")
    shift = int(round(float(lidar_index0_deg) / cfg.LIDAR_BEAM_RES_DEG))
    ranges = np.roll(lidar, shift).astype(np.float64)
    x, y, h = transform.apply(frame.x_m, frame.y_m, frame.yaw_deg)
    u, v, r = float(frame.u_body_mps), -float(frame.v_body_mps), float(frame.yaw_rate)
    if pose_point == "sensor":
        from ship import LIDAR_OFFSET_M
        a = math.radians(h)
        x, y = x - LIDAR_OFFSET_M * math.sin(a), y - LIDAR_OFFSET_M * math.cos(a)
        v -= math.radians(r) * LIDAR_OFFSET_M
    elif pose_point != "origin":
        raise ValueError(f"pose_point {pose_point!r}")
    return SensorFrame(t=float(frame.t_sec), ranges=ranges, pose=(x, y, h), ego=(u, v, r),
                       pose_stale=bool(pose_stale), missed_cycles=int(missed_cycles))


def missed_cycles(t_now: float, t_prev: Optional[float]) -> int:
    if t_prev is None:
        return 0
    return max(0, int(round((float(t_now) - float(t_prev)) / cfg.UPDATE_RATE)) - 1)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Policy:
    """The deployed checkpoint, refused unless it is the one Paper 3 reports."""

    def __init__(self, path: str = DEFAULT_MODEL, *, expected_sha256: str = EXPECTED_MODEL_SHA256) -> None:
        import torch
        torch.set_num_threads(1)
        self.path = os.path.abspath(path)
        self.sha256 = sha256_file(self.path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(f"{self.path}: SHA-256 {self.sha256[:8]}..., expected {expected_sha256[:8]}...")
        config_path = os.path.join(os.path.dirname(self.path), "config.json")
        self.config = json.load(open(config_path)) if os.path.exists(config_path) else {}
        self.config_sha256 = sha256_file(config_path) if os.path.exists(config_path) else None
        problems = check_formulation(self.config)
        if problems:
            raise ValueError("the checkpoint's recorded formulation differs from the code: " + "; ".join(problems))
        self.model = tf.ALGORITHMS[self.config.get("algo", "sac")].load(self.path, device="cpu")
        space = obsmod.observation_space(cfg.N_MAX_TARGETS)
        got = {k: tuple(v.shape) for k, v in self.model.observation_space.spaces.items()}
        want = {k: tuple(v.shape) for k, v in space.spaces.items()}
        if got != want:
            raise ValueError(f"policy observation space {got} is not the code's {want}")
        self.actor = tf.EpisodeActor(self.model)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(self.actor(obs), dtype=np.float32).reshape(2)


def check_formulation(config: Dict) -> List[str]:
    """The run record against the constants the bridge runs with."""
    problems = []
    if bool(cfg.RUDDER_COMMAND_LIMIT):
        problems.append("RUDDER_COMMAND_LIMIT is on; the bridge issues the command unlimited")
    if not config:
        return problems + ["no config.json beside the checkpoint"]
    if config.get("observation_schema") != cfg.OBSERVATION_SCHEMA_VERSION:
        problems.append(f"observation schema {config.get('observation_schema')!r}")
    if int(config.get("observation_dim", -1)) != 70:
        problems.append(f"observation dim {config.get('observation_dim')!r}")
    if int(config.get("propulsion_stage", -1)) != int(cfg.RPM_STAGE):
        problems.append(f"propulsion stage {config.get('propulsion_stage')!r}, applied {cfg.RPM_STAGE}")
    params = config.get("rpm_stage_params")
    if params is not None and [float(v) for v in params] != [cfg.RPM_DELTA, cfg.RPM_FLOOR, cfg.RPM_CEIL]:
        problems.append(f"rpm stage params {params}, code {[cfg.RPM_DELTA, cfg.RPM_FLOOR, cfg.RPM_CEIL]}")
    for name, value in (config.get("constants") or {}).items():
        if hasattr(cfg, name) and getattr(cfg, name) != value:
            problems.append(f"{name}: run {value!r}, code {getattr(cfg, name)!r}")
    return problems


# ---------------------------------------------------------------------------
# Command resolution (ASVLidarEnv.step)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Command:
    rudder_pct: float           # simulator convention, +-100
    rpm: float                  # rpm-units; negative under a stop override
    s2: float                   # propulsion command S2
    executed: np.ndarray        # the executed action the next observation carries
    source: str                 # "policy", "layer" (active mode), "layer brake", "operator"

    @property
    def vessel_rudder(self) -> float:
        return RUDDER_SIGN * self.rudder_pct

    def message(self) -> str:
        return f"$CMD,{self.vessel_rudder:.2f},{float(np.clip(self.s2, *S2_LIMITS)):.2f}"


def resolve_command(action, *, brake: bool = False, observed_surge: float = 0.0,
                    override_s2: Optional[float] = None, source: str = "policy") -> Command:
    """`ASVLidarEnv.step` from the (filtered) action to the issued command, without physics.

    `override_s2` is the operator stop latch's S2 when it holds; otherwise the layer's brake
    (active mode only) commands full astern above the stop speed and zero below it.
    """
    rudder_cmd = float(np.clip(action[0], -1.0, 1.0))
    throttle_cmd = float(np.clip(action[1], -1.0, 1.0))
    if cfg.RUDDER_COMMAND_LIMIT:
        raise RuntimeError("RUDDER_COMMAND_LIMIT is on; the bridge implements the unlimited command only")
    rudder = rudder_cmd * 100.0
    policy_rpm = cfg.CRUISE_RPM if cfg.FIXED_RPM else float(np.clip(
        cfg.CRUISE_RPM + cfg.RPM_DELTA * throttle_cmd, cfg.RPM_FLOOR, cfg.RPM_CEIL))
    override = None if override_s2 is None else float(override_s2)
    if override is not None:
        source = "operator"
    elif brake:
        override = (estop_mod.S2_FULL_ASTERN if float(observed_surge) > cfg.ESTOP_STOP_SPEED else 0.0)
        source = "layer brake"
    if override is None:
        rpm, s2 = policy_rpm, estop_mod.rpm_to_s2(policy_rpm)
    else:
        s2, rpm = float(override), estop_mod.s2_to_rpm(override)
    executed = np.array([np.clip(rudder / 100.0, -1.0, 1.0),
                         np.clip((rpm - cfg.CRUISE_RPM) / max(cfg.RPM_DELTA, 1e-6), -1.0, 1.0)],
                        dtype=np.float32)
    return Command(rudder_pct=rudder, rpm=rpm, s2=s2, executed=executed, source=source)


# ---------------------------------------------------------------------------
# Safety layer
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SafetyDecision:
    out: np.ndarray             # the command the layer issues (active) or would issue (shadow)
    changed: bool               # differs from the policy's (brake included)
    brake: bool
    why: str
    mode: str
    committed: Optional[int]    # decision index of the stop commitment, if any
    seconds: float
    details: Dict[str, Any]
    plan: Optional[np.ndarray]


class SafetyLayer:
    """E1 on one environment interface: `"shadow"` (decision D-4) or `"active"`."""

    def __init__(self, mode: str = "shadow") -> None:
        if mode not in ("shadow", "active"):
            raise ValueError(f"safety mode {mode!r}")
        if dict(safety_estop.BOUNDARY_TARGET_OPTIONS, braking_fallback=False) != E1_OPTIONS:
            raise ValueError("E1's options no longer match the boundary-and-targets trigger")
        self.mode = mode
        self.filter = E1_CLASS(**E1_OPTIONS)
        self._before = None
        self._last: Optional[SafetyDecision] = None
        self.decisions = 0
        self.changes = 0
        self.first_change: Optional[int] = None

    def decide(self, env, action) -> SafetyDecision:
        """One decision on the environment's current perception (as `env.step` calls it)."""
        self._before = copy.deepcopy(self.filter.actuators)
        start = time.perf_counter()
        out, changed = self.filter.filter(env, action)
        seconds = time.perf_counter() - start
        brake = bool(getattr(env, "_v2_brake", False))
        if self.mode == "shadow":
            env._v2_brake = False                # never acted on
        last = self.filter.last
        decision = SafetyDecision(
            out=np.asarray(out, dtype=np.float32).reshape(2).copy(), changed=bool(changed), brake=brake,
            why=str(last.get("why", "idle")), mode=str(last.get("mode", "idle")),
            committed=self.filter.estop_committed_decision, seconds=seconds, details=last,
            plan=None if getattr(self.filter, "plan", None) is None else np.array(self.filter.plan))
        if decision.changed and self.first_change is None:
            self.first_change = self.decisions
        self.decisions += 1
        self.changes += int(decision.changed)
        self._last = decision
        return decision

    def issued(self, env, rudder_pct: float, rpm: float) -> None:
        """The command that went out after this decision (`env.step`'s issued-command call)."""
        rudder = float(rudder_pct) / 100.0
        if (self.mode == "shadow" and self._last is not None
                and abs(rudder - float(self._last.out[0])) > 1e-9):
            # The layer's model of its own rudder follows the command issued, not its proposal.
            self.filter.actuators = copy.deepcopy(self._before)
            self.filter.actuators.issue(env, rudder)
        self.filter.observe_issued_command(rudder, float(rpm))


# ---------------------------------------------------------------------------
# The bridge: one decision per sensor frame
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Decision:
    k: int
    sensor: SensorFrame
    executed_prev: np.ndarray
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    command: Command
    safety: Optional[SafetyDecision]
    seconds: Dict[str, float]


class Bridge:
    """Perception, policy and command for each sensor frame.

    `layer` given in active mode puts E1 between the policy and the command.  Shadow mode
    does not pass through here: the shadow layer reads the bridge's environment after the
    command has gone (`ShadowRunner` or `shadow_inline`), so its run time never delays it.
    """

    def __init__(self, built, policy: Policy, *, seed: Optional[int] = None,
                 layer: Optional[SafetyLayer] = None) -> None:
        if layer is not None and layer.mode != "active":
            raise ValueError("only an active layer sits in the command path")
        self.built = built
        self.seed = seed
        self.policy = policy
        self.layer = layer
        self.env = LiveEnv()
        self.executed = np.zeros(2, dtype=np.float32)
        self.k = 0

    def decide(self, sensor: SensorFrame, *, override_s2: Optional[float] = None) -> Decision:
        t0 = time.perf_counter()
        executed_prev = self.executed.copy()
        obs = (self.env.begin(self.built, sensor, seed=self.seed) if self.k == 0
               else self.env.observe(sensor, executed_prev))
        t1 = time.perf_counter()
        action = self.policy.act(obs)
        t2 = time.perf_counter()
        safety, out, brake = None, action, False
        if self.layer is not None:
            safety = self.layer.decide(self.env, action)
            out, brake = safety.out, safety.brake
        t3 = time.perf_counter()
        command = resolve_command(out, brake=brake, observed_surge=self.env._observed_surge,
                                  override_s2=override_s2,
                                  source="layer" if safety is not None and safety.changed else "policy")
        if self.layer is not None:
            self.layer.issued(self.env, command.rudder_pct, command.rpm)
        self.executed = command.executed.copy()
        seconds = {"observation": t1 - t0, "policy": t2 - t1, "layer_active": t3 - t2,
                   "command": time.perf_counter() - t3}
        decision = Decision(self.k, sensor, executed_prev, obs, action, command, safety, seconds)
        self.k += 1
        return decision


class ShadowRunner:
    """E1 in shadow from decision inputs: its own `LiveEnv`, fed the bridge's frames.

    The bridge's worker process and the offline replay of a decision log both run this.
    Perception is deterministic, so its environment holds the bridge's state exactly; `obs`
    given is checked against its own.
    """

    def __init__(self, built, *, seed: Optional[int] = None) -> None:
        self.built, self.seed = built, seed
        self.env = LiveEnv()
        self.layer = SafetyLayer("shadow")
        self.k = 0

    def process(self, sensor: SensorFrame, executed_prev, action, rudder_pct: float, rpm: float,
                obs: Optional[Dict[str, np.ndarray]] = None) -> Tuple[SafetyDecision, bool, float]:
        start = time.perf_counter()
        mine = (self.env.begin(self.built, sensor, seed=self.seed) if self.k == 0
                else self.env.observe(sensor, executed_prev))
        perceive = time.perf_counter() - start
        same = obs is None or obs_equal(mine, obs)
        decision = shadow_inline(self.layer, self.env, action, rudder_pct, rpm)
        self.k += 1
        return decision, same, perceive


def shadow_inline(layer: SafetyLayer, env, action, rudder_pct: float, rpm: float) -> SafetyDecision:
    """The shadow layer's decision on the environment's current perception, then the issued
    command it is told about (the order of `env.step`)."""
    decision = layer.decide(env, action)
    layer.issued(env, rudder_pct, rpm)
    return decision


def shadow_worker(inbox, path: str, built, seed: Optional[int], header: Dict, outbox=None) -> None:
    """Worker process: the shadow layer off the control path, one record per decision.

    `inbox` carries one dict per decision (`shadow_item`) and None to finish; `outbox`, if
    given, receives "ready" once loaded and a short summary per decision for the display.
    `t_queued` is `time.time()`, comparable across processes.
    """
    table = use_stopping_table()
    runner = ShadowRunner(built, seed=seed)
    if outbox is not None:
        outbox.put({"type": "ready", "stopping_table": table})
    with open(path, "a", encoding="utf-8", buffering=1) as out:
        out.write(json.dumps(jsonable(dict(header, type="shadow_header")), allow_nan=False) + "\n")
        while True:
            item = inbox.get()
            if item is None:
                break
            received = time.time()
            decision, same, perceive = runner.process(
                item["sensor"], item["executed_prev"], item["action"], item["rudder_pct"], item["rpm"],
                obs=item.get("obs"))
            record = {"type": "shadow", "run_id": header.get("run_id"), "k": item["k"],
                      "t_frame": item["sensor"].t, "obs_matches_bridge": bool(same),
                      "seconds": {"perception": perceive, "layer": decision.seconds,
                                  "queue_wait": max(0.0, received - float(item.get("t_queued", received))),
                                  "behind_frame": time.time() - float(item.get("t_queued", received))},
                      "safety": safety_record(decision, compact=bool(header.get("compact", True)))}
            out.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
            if outbox is not None:
                outbox.put({"type": "shadow", "k": item["k"], "why": decision.why, "changed": decision.changed,
                            "brake": decision.brake, "committed": decision.committed,
                            "t_star": decision.details.get("estop_t_star"), "seconds": decision.seconds,
                            "obs_matches_bridge": bool(same)})
    if outbox is not None:
        outbox.put({"type": "done", "decisions": runner.layer.decisions, "changes": runner.layer.changes,
                    "first_change": runner.layer.first_change,
                    "committed": runner.layer.filter.estop_committed_decision})


def shadow_item(decision: "Decision") -> Dict[str, Any]:
    """What the shadow worker needs from one bridge decision (and the observation to check)."""
    c = decision.command
    return {"k": decision.k, "sensor": decision.sensor, "executed_prev": decision.executed_prev,
            "action": decision.action, "rudder_pct": c.rudder_pct, "rpm": c.rpm, "obs": decision.obs,
            "t_queued": time.time()}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def jsonable(value):
    """JSON-ready copy: arrays to lists, non-finite floats to strings, dataclasses to dicts."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def without_keys(value, drop: Sequence[str]):
    if isinstance(value, dict):
        return {k: without_keys(v, drop) for k, v in value.items() if k not in drop}
    return value


def obs_equal(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def obs_record(obs: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    return {k: np.asarray(v, dtype=np.float32).tolist() for k, v in obs.items()}


def track_record(track, dynamic_ids) -> Dict[str, Any]:
    return {"id": int(track.id), "position": track.position.tolist(), "velocity": track.velocity.tolist(),
            "speed": track.speed, "course_deg": track.course_deg, "confirmed": bool(track.confirmed),
            "dynamic": int(track.id) in dynamic_ids, "hits": int(track.hits), "misses": int(track.misses),
            "age": int(track.age), "fit_heading_deg": track.last_fit_heading_deg}


def perception_record(env: LiveEnv) -> Dict[str, Any]:
    """The tracker's tracks and the encounter contexts of this decision (truth fields are
    empty in the field and left out)."""
    dynamic = {int(t.id) for t in env.tracks}
    contexts = {}
    for tid, ctx in env.encounter_contexts.items():
        fields = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)
                  if not f.name.endswith("_true")}
        contexts[str(tid)] = jsonable(fields)
    return {"tracks": [track_record(t, dynamic) for t in env.tracker.tracks],
            "contexts": contexts, "pose_stale": bool(env.pose_stale),
            "estimated_pose": list(env.estimated_pose()), "measured_ego": list(env._measured_ego())}


def safety_record(decision: SafetyDecision, *, compact: bool = True) -> Dict[str, Any]:
    details = without_keys(decision.details, COMPACT_DROP) if compact else decision.details
    return {"why": decision.why, "mode": decision.mode, "changed": decision.changed,
            "brake": decision.brake, "out": decision.out.tolist(), "committed": decision.committed,
            "estop_t_star": details.get("estop_t_star"), "checked_clearance": details.get("checked_clearance"),
            "seconds": decision.seconds, "plan": decision.plan, "details": jsonable(details)}


def decision_record(decision: Decision, *, run_id: str) -> Dict[str, Any]:
    c = decision.command
    s = decision.sensor
    return {"type": "decision", "run_id": run_id, "k": decision.k, "t_frame": s.t,
            "sensor": {"ranges": np.asarray(s.ranges, dtype=np.float64).tolist(), "pose": list(s.pose),
                       "ego": list(s.ego), "pose_stale": bool(s.pose_stale), "missed_cycles": int(s.missed_cycles)},
            "executed_prev": decision.executed_prev.tolist(),
            "observation": obs_record(decision.obs),
            "action": decision.action.tolist(),
            "command": {"rudder_pct": c.rudder_pct, "rpm": c.rpm, "s2": c.s2, "vessel_rudder": c.vessel_rudder,
                        "source": c.source, "executed": c.executed.tolist(), "message": c.message()},
            "perception": None, "safety": None,
            "seconds": dict(decision.seconds)}


def source_hashes() -> Dict[str, str]:
    out = {}
    for base in (SIM_SRC, HERE):
        for root, _dirs, files in os.walk(base):
            if "__pycache__" in root:
                continue
            for name in sorted(files):
                if name.endswith(".py"):
                    path = os.path.join(root, name)
                    out[os.path.relpath(path, os.path.dirname(SIM_ROOT)).replace(os.sep, "/")] = sha256_file(path)
    return out


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def constants_record() -> Dict[str, Any]:
    names = ("UPDATE_RATE", "CRUISE_RPM", "RPM_STAGE", "RPM_DELTA", "RPM_FLOOR", "RPM_CEIL", "FIXED_RPM",
             "RUDDER_COMMAND_LIMIT", "ESTOP_STOP_SPEED", "LIDAR_BEAMS", "LIDAR_RANGE", "LIDAR_MIN_RANGE",
             "LIDAR_AFT_MASK_HALF_DEG", "LIDAR_SECTORS", "BOUNDARY_RAYS", "MOTION_CLASSIFIER",
             "OBSERVATION_SCHEMA_VERSION", "N_MAX_TARGETS", "EGO_SPEED_NOISE")
    return {n: jsonable(getattr(cfg, n, None)) for n in names}


def run_header(*, run_id: str, policy: Policy, env: LiveEnv, safety: str, extra: Optional[Dict] = None) -> Dict:
    return jsonable({
        "type": "header", "run_id": run_id, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"path": policy.path, "sha256": policy.sha256, "config_sha256": policy.config_sha256},
        "safety": {"mode": safety, "class": f"{E1_CLASS.__module__}.{E1_CLASS.__name__}", "options": E1_OPTIONS},
        "constants": constants_record(), "geometry": geometry_record(env),
        "git_head": (extra or {}).get("git_head") or git_head(),
        "source_sha256": (extra or {}).get("source_sha256") or source_hashes(), **(extra or {})})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="p3_bridge utilities")
    ap.add_argument("--build-stopping-table", action="store_true",
                    help=f"precompute stopping._profile for 0..{STOPPING_TABLE_MAX_CMPS} cm/s (a few minutes)")
    ap.add_argument("--max-cmps", type=int, default=STOPPING_TABLE_MAX_CMPS)
    cli = ap.parse_args()
    if cli.build_stopping_table:
        print(build_stopping_table(max_cmps=cli.max_cmps))
        print(use_stopping_table(spot_checks=5))
