"""Shared machinery for the classical comparators (B8).

Both comparators see exactly what the learned policies and the reference
controller see, and nothing else: the known path and map polygon, the held
pose and ego estimates, the gated scan and the tracker's tracks with their
encounter contexts.  Simulator truth (`env.targets`, `env.asv_x`, the context's
ground-truth fields) never enters action selection.

What lives here:

* `Perception` -- one onboard snapshot per decision, with a short scan memory
  so the aft mask and the dead zone do not make a passed obstacle vanish;
* LOS guidance (Fossen, Breivik & Skjetne 2003) and a heading PID -- the
  path-following half of both comparators;
* `Actuators` -- the controller's own copy of the rudder delay line and servo,
  so predictions start from the rudder that is actually coming, as the
  reference controller does;
* `rollout` -- a batched closed-loop prediction on the identified hull;
* hull-level clearances to scan points, the map polygon and tracked targets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

import constant_temp as ct
import constants as cfg
from reference_controller import hull_separation
from ship import HULL_MARGIN, IDENTIFIED, LIDAR_OFFSET_M, VESSEL_LENGTH, VESSEL_WIDTH, dyn, steady_speed

HALF_L = 0.5 * VESSEL_LENGTH + HULL_MARGIN
HALF_W = 0.5 * VESSEL_WIDTH + HULL_MARGIN
MAX_RUDDER_RAD = ct.CLASSICAL_MAX_RUDDER_RAD

# LOS lookahead.  1.6 Lpp, inside the 1.5-2.5 L band LOS papers use, and the
# value the reference controller settled on in the same channels.
LOS_LOOKAHEAD_M = ct.CLASSICAL_LOS_LOOKAHEAD_M
MAX_SIDESLIP_RAD = ct.CLASSICAL_MAX_SIDESLIP_RAD

# Heading PID, in rudder-units (full rudder = 1) per degree.  P and D are the
# reference controller's tuning on this hull; the integral is slow and gated to
# small errors so it trims a steady offset without winding up in an avoidance.
PID_KP = ct.CLASSICAL_PID_KP
PID_KD = ct.CLASSICAL_PID_KD   # on measured yaw rate, deg/s
PID_KI = ct.CLASSICAL_PID_KI   # per deg.s
PID_I_GATE_DEG = ct.CLASSICAL_PID_I_GATE_DEG
PID_I_LIMIT = ct.CLASSICAL_PID_I_LIMIT   # rudder units

# Identified steady turn: ~10.4 deg/s per unit rudder at cruise (probed on the
# simulator, linear in rudder to full deflection).  Used as the feed-forward of
# the yaw-rate loop.
YAW_RATE_GAIN_DPS = ct.CLASSICAL_YAW_RATE_GAIN_DPS
YAW_RATE_KP = ct.CLASSICAL_YAW_RATE_KP   # rudder units per deg/s of yaw-rate error

U_PER_RPM = steady_speed(12.0) / 12.0     # 0.093 m/s per rpm-unit, linear
SPEED_KP_RPM = ct.CLASSICAL_SPEED_KP_RPM  # rpm-units per m/s of speed error

# Prediction step.  The rudder servo is solved exactly and the body dynamics
# evolve over seconds, so 0.125 s holds the collision checks at ~0.14 m of
# closing per sample at the fastest head-on closing speed.
PRED_DT = ct.CLASSICAL_PRED_DT
SUBSTEPS = int(round(cfg.UPDATE_RATE / PRED_DT))
DELAY_STEPS = max(0, int(round(IDENTIFIED["rud_delay"] / PRED_DT)))

# Scan memory: 4 decisions = 2 s, enough to carry a passed obstacle through
# the aft mask.  The reference controller's 10 s smeared a target's returns
# into a phantom wall along its track before the tracker confirmed it dynamic.
SCAN_MEMORY_FRAMES = ct.CLASSICAL_SCAN_MEMORY_FRAMES
TRACK_EXCLUSION_M = ct.CLASSICAL_TRACK_EXCLUSION_M  # returns this close to a track are the target's own


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------
@dataclass
class TrackView:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float                  # rad, compass
    ctx: object = None              # EncounterContext, or None before it exists

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))


@dataclass
class Snapshot:
    x: float
    y: float
    heading: float                  # rad, compass (0 = +y, clockwise)
    u: float
    v: float
    r: float                        # rad/s
    tangent: np.ndarray
    right: np.ndarray
    centre: np.ndarray
    base_heading: float             # rad
    lateral: float                  # m, +ve to starboard of the path
    remaining: float                # m along the path to its end
    points: np.ndarray              # (P, 2) static scan returns, world frame
    tracks: List[TrackView] = field(default_factory=list)
    edges_a: np.ndarray = None      # (E, 2) map polygon edge starts
    edges_b: np.ndarray = None

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def sideslip(self) -> float:
        return float(np.clip(math.atan2(self.v, max(0.15, self.u)), -MAX_SIDESLIP_RAD, MAX_SIDESLIP_RAD))


class Perception:
    """Builds one `Snapshot` per decision from onboard-available state only."""

    def __init__(self, memory_frames: int = SCAN_MEMORY_FRAMES):
        self.memory_frames = int(memory_frames)
        self.memory: List[Tuple[int, np.ndarray]] = []
        self.frames = 0

    def snapshot(self, env) -> Snapshot:
        x, y, heading_deg = env.estimated_pose()
        u, v, yaw_dps = env._measured_ego()
        heading = math.radians(heading_deg)
        state = env.path.project(x, y, heading_deg)
        tangent = env.path.tangent(state.closest_idx).astype(float)
        right = np.array([tangent[1], -tangent[0]])
        centre = np.asarray(state.target, dtype=float)
        lateral = float((np.array([x, y]) - centre) @ right)
        remaining = float((env.path.points[-1] - [x, y]) @ tangent)

        tracks = []
        for t in env.tracks:
            fitted = getattr(t, "last_fit_heading_deg", None)
            vel = np.asarray(t.velocity, dtype=float)
            hdg = (math.radians(fitted) if fitted is not None
                   else math.atan2(vel[0], vel[1]) if np.hypot(*vel) > 0.05 else 0.0)
            tracks.append(TrackView(int(t.id), np.asarray(t.position, dtype=float), vel, hdg,
                                    env.encounter_contexts.get(t.id)))

        ranges = np.asarray(getattr(env, "gated_ranges", env.lidar.ranges), dtype=float)
        mask = ranges < cfg.LIDAR_RANGE - 1e-5
        bearings = np.radians(np.asarray(env.lidar.bearings)[mask]) + heading
        origin = np.array([x, y]) + LIDAR_OFFSET_M * np.array([math.sin(heading), math.cos(heading)])
        pts = origin + ranges[mask, None] * np.stack([np.sin(bearings), np.cos(bearings)], axis=1)
        pts = _drop_near_tracks(pts, tracks, TRACK_EXCLUSION_M)
        self.frames += 1
        if len(pts):
            self.memory.append((self.frames, pts))
        self.memory = [(k, p) for k, p in self.memory if self.frames - k < self.memory_frames]
        pts = np.concatenate([p for _, p in self.memory]) if self.memory else np.empty((0, 2))
        # Remembered returns can lie where a target has since moved to.
        pts = _drop_near_tracks(pts, tracks, TRACK_EXCLUSION_M + 0.2)
        if len(pts):
            _, keep = np.unique(np.round(pts / 0.25).astype(int), axis=0, return_index=True)
            pts = pts[keep]

        poly = np.asarray(env.boundary_polygon, dtype=float)
        return Snapshot(float(x), float(y), heading, float(u), float(v), math.radians(yaw_dps),
                        tangent, right, centre, math.atan2(tangent[0], tangent[1]),
                        lateral, remaining, pts, tracks, poly, np.roll(poly, -1, axis=0))


def _drop_near_tracks(points: np.ndarray, tracks, radius: float) -> np.ndarray:
    for t in tracks:
        if len(points):
            points = points[np.linalg.norm(points - t.position, axis=1) > radius]
    return points


# ---------------------------------------------------------------------------
# LOS guidance and PID
# ---------------------------------------------------------------------------
def los_heading(snap: Snapshot, positions: Optional[np.ndarray] = None,
                sideslip=None, lookahead: float = LOS_LOOKAHEAD_M):
    """Proportional LOS: the heading that steers the ground course back onto
    the path over `lookahead` metres, less the measured sideslip.

    `positions` (…, 2) evaluates it at predicted positions, for the rollouts;
    the path is straight between stations, so the snapshot's tangent holds.
    """
    if positions is None:
        lateral = snap.lateral
        sideslip = snap.sideslip if sideslip is None else sideslip
    else:
        lateral = (np.asarray(positions) - snap.centre) @ snap.right
        sideslip = 0.0 if sideslip is None else sideslip
    return snap.base_heading + np.arctan2(-lateral, lookahead) - sideslip


class HeadingPID:
    """Rudder from heading error: P on error, D on measured yaw rate, gated I."""

    def __init__(self):
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def __call__(self, error_rad: float, yaw_rate_rad: float, dt: float = cfg.UPDATE_RATE) -> float:
        e = math.degrees(float(error_rad))
        if abs(e) < PID_I_GATE_DEG:
            self.integral = float(np.clip(self.integral + PID_KI * e * dt, -PID_I_LIMIT, PID_I_LIMIT))
        else:
            self.integral *= 0.9          # bleed rather than hold across an alteration
        return float(np.clip(PID_KP * e - PID_KD * math.degrees(yaw_rate_rad) + self.integral, -1.0, 1.0))


def pd_rudder(error_rad, yaw_rate_rad):
    """The PID without its integral, vectorised -- for predicted states."""
    return np.clip(PID_KP * np.degrees(error_rad) - PID_KD * np.degrees(yaw_rate_rad), -1.0, 1.0)


def yaw_rate_rudder(r_des_rad, r_rad):
    """Yaw-rate loop: steady-turn feed-forward plus proportional correction."""
    r_des, r = np.degrees(r_des_rad), np.degrees(r_rad)
    return np.clip(r_des / YAW_RATE_GAIN_DPS + YAW_RATE_KP * (r_des - r), -1.0, 1.0)


def course_rudder(error_rad, r_rad, gain: float = 0.3, max_rate_dps: float = 6.0):
    """Cascaded course autopilot: heading error -> yaw-rate demand (capped) ->
    the yaw-rate loop.  Turns at a known rate, which a planner that sweeps a
    turn-rate model (the VO) needs; the PID above turns as fast as its damping
    allows, which is right for holding a path and too slow to execute a
    planned alteration (the VO's first version picked the right alteration
    and turned at ~2 deg/s instead of the ~6 it planned for)."""
    r_des = np.clip(gain * np.asarray(error_rad), -np.radians(max_rate_dps), np.radians(max_rate_dps))
    return yaw_rate_rudder(r_des, r_rad)


def speed_rpm(u_des, u):
    """Speed loop: steady-speed feed-forward plus proportional correction."""
    rpm = np.asarray(u_des) / U_PER_RPM + SPEED_KP_RPM * (np.asarray(u_des) - np.asarray(u))
    rpm = np.where(np.asarray(u_des) <= 1e-6, cfg.RPM_FLOOR, rpm)
    return np.clip(rpm, cfg.RPM_FLOOR, cfg.RPM_CEIL)


def throttle_for_rpm(rpm: float) -> float:
    return float(np.clip((float(rpm) - cfg.CRUISE_RPM) / max(cfg.RPM_DELTA, 1e-6), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Actuators and prediction
# ---------------------------------------------------------------------------
class Actuators:
    """The controller's model of its own rudder: delay line, then servo.

    Advanced with every command issued, so a rollout starts from the rudder
    angle and the in-flight commands the vessel will actually execute.
    """

    def __init__(self):
        self.buffer: Optional[List[float]] = None
        self.servo = 0.0
        self.executed = 0.0

    def issue(self, env, rudder: float) -> None:
        if bool(getattr(env, "command_rate_limit", False)):
            self.executed += float(np.clip(rudder - self.executed, -0.25, 0.25))
        else:
            self.executed = float(rudder)
        delta = -MAX_RUDDER_RAD * self.executed
        if self.buffer is None:
            self.buffer = [delta] * DELAY_STEPS
        for _ in range(SUBSTEPS):
            self.buffer.append(delta)
            delayed = self.buffer.pop(0)
            self.servo = float(dyn.advance_rudder(np.array([self.servo]), np.array([delayed]),
                                                  IDENTIFIED, PRED_DT)[0])


@dataclass
class Rollout:
    positions: np.ndarray            # (K, n, 2), one row per PRED_DT, t = PRED_DT .. horizon
    headings: np.ndarray             # (K, n) rad
    speeds: np.ndarray               # (K, n)
    times: np.ndarray                # (K,)


def rollout(snap: Snapshot, act: Actuators, n: int, horizon_s: float,
            law: Callable[[int, np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> Rollout:
    """Closed-loop prediction of `n` candidates on the identified hull.

    `law(k, state)` returns `(rpm, rudder)` arrays at decision `k`; it is held
    for the decision interval, as the environment holds a command.  `state` is
    the model's `[u, v, r, psi, delta, x, y]` by `n`.
    """
    decisions = int(math.ceil(horizon_s / cfg.UPDATE_RATE))
    state = np.zeros((7, n))
    state[0:4] = np.array([max(0.0, snap.u), snap.v, snap.r, snap.heading])[:, None]
    state[4] = act.servo
    state[5], state[6] = snap.x, snap.y
    params = {k: np.full(n, v) for k, v in IDENTIFIED.items()}
    pending = ([np.full(n, d) for d in act.buffer] if act.buffer is not None
               else None)
    K = decisions * SUBSTEPS
    pos, hdg, spd = np.empty((K, n, 2)), np.empty((K, n)), np.empty((K, n))
    j = 0
    for k in range(decisions):
        rpm, rudder = law(k, state)
        delta = -MAX_RUDDER_RAD * np.asarray(rudder, dtype=float) * np.ones(n)
        rpm = np.asarray(rpm, dtype=float) * np.ones(n)
        if pending is None:
            pending = [delta.copy() for _ in range(DELAY_STEPS)]
        for _ in range(SUBSTEPS):
            pending.append(delta)
            state = dyn.rk4_step(state, rpm, pending.pop(0), params, PRED_DT)
            pos[j] = state[5:7].T
            hdg[j] = state[3]
            spd[j] = np.hypot(state[0], state[1])
            j += 1
    return Rollout(pos, hdg, spd, PRED_DT * np.arange(1, K + 1))


# ---------------------------------------------------------------------------
# Clearances (hull level, metres; negative is overlap)
# ---------------------------------------------------------------------------
def point_clearance(positions: np.ndarray, headings: np.ndarray, points: np.ndarray,
                    reach: float = None) -> np.ndarray:
    """Distance from the inflated own-hull rectangle to the nearest scan point.

    `positions` (K, n, 2), `headings` (K, n); returns (K, n).
    """
    K, n = headings.shape
    if len(points) == 0:
        return np.full((K, n), np.inf)
    if reach is not None:
        centre = positions.reshape(-1, 2)
        lo, hi = centre.min(axis=0) - reach, centre.max(axis=0) + reach
        points = points[np.all((points >= lo) & (points <= hi), axis=1)]
        if len(points) == 0:
            return np.full((K, n), np.inf)
    rel = points[None, None, :, :] - positions[:, :, None, :]
    s, c = np.sin(headings)[..., None], np.cos(headings)[..., None]
    lon = np.abs(rel[..., 0] * s + rel[..., 1] * c) - HALF_L
    lat = np.abs(rel[..., 0] * c - rel[..., 1] * s) - HALF_W
    outside = np.hypot(np.maximum(lon, 0.0), np.maximum(lat, 0.0))
    inside = np.minimum(np.maximum(lon, lat), 0.0)
    return np.min(outside + inside, axis=-1)


def boundary_clearance(positions: np.ndarray, headings: np.ndarray,
                       a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from the inflated hull to the map polygon, (K, n).

    Per edge: the centre's distance to the segment less the hull's support
    along the edge normal.  Exact for a rectangle against a long straight edge,
    which is every edge of a basin and of an unbent channel.
    """
    edge = b - a
    length2 = np.maximum(np.sum(edge * edge, axis=1), 1e-12)
    normal = np.stack([-edge[:, 1], edge[:, 0]], axis=1) / np.sqrt(length2)[:, None]
    rel = positions[:, :, None, :] - a                          # (K, n, E, 2)
    t = np.clip(np.sum(rel * edge, axis=-1) / length2, 0.0, 1.0)
    dist = np.linalg.norm(rel - t[..., None] * edge, axis=-1)
    fwd = np.stack([np.sin(headings), np.cos(headings)], axis=-1)       # (K, n, 2)
    stbd = np.stack([np.cos(headings), -np.sin(headings)], axis=-1)
    support = HALF_L * np.abs(fwd @ normal.T) + HALF_W * np.abs(stbd @ normal.T)
    return np.min(dist - support, axis=-1)


def target_gap(positions: np.ndarray, headings: np.ndarray, times: np.ndarray,
               track: TrackView) -> np.ndarray:
    """SAT gap between the own hull and the track's hull at constant velocity, (K, n)."""
    future = track.position + times[:, None] * track.velocity             # (K, 2)
    return hull_separation(positions, headings, future[:, None, :], track.heading, margin=HULL_MARGIN)
