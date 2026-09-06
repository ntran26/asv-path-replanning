"""Target tracking: gated returns -> tracks with estimated velocity.

Why this exists (01 §4)
-----------------------
Sector closeness is velocity-blind.  A wall at 4 m and a vessel closing at
1 m/s at 4 m produce an identical `c_t`.  Explicit tracking is the chosen
resolution rather than frame-stacking a pooled scan, because it also delivers
information parity with the VO and DWA baselines, which need tracked target
state anyway.

Pipeline, in order:

1. **Gate** beyond-boundary returns          -> `boundary_raycast.gate_beams`
2. **Cluster** the survivors                 -> `cluster_scan`
3. **Ego-motion compensate** using odometry  -> `Tracker.update`, via the pose
4. **Associate** clusters to tracks          -> nearest-neighbour
5. **Estimate** velocity per track           -> constant-velocity Kalman filter
6. **Classify** static vs dynamic            -> speed threshold with hysteresis

Static clusters keep feeding `c_t`; dynamic tracks feed the target branch.

The coupling that matters
-------------------------
Step 3 is where 01 and 05 meet.  Clusters are lifted into the world frame using
the *estimated* pose, so odometry drift appears directly as apparent velocity on
genuinely **static** objects.  Worse, scan-matching odometry is itself corrupted
by the moving objects in the scan, so the error is correlated with exactly the
situation the tracker exists to handle.  `Tracker` therefore takes the estimated
pose, never a ground-truth one.

That is why **the static/dynamic threshold is a property of localisation
quality, not of obstacle behaviour** (01 §4 step 6, 03 §4a).  The field
obstacles are suspended panels, confirmed from video to hang stably, so apparent
motion of a static object comes almost entirely from ego-pose error -- which
affects every object in the scan identically.  The threshold is set from
measured pose noise (05 §4) and retightened as registration improves, and it is
biased toward **under**-detection: promoting a static panel to a target ship is
a false positive with COLREGs consequences.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

import constants as cfg

_next_track_id = itertools.count(1)


# ---------------------------------------------------------------------------
# Step 2 -- clustering
# ---------------------------------------------------------------------------
def scan_to_points(ranges, bearings_deg, x: float, y: float,
                   heading_deg: float, *, max_range: float = cfg.LIDAR_RANGE) -> np.ndarray:
    """Lift a body-frame scan into world coordinates, dropping no-returns.

    This is the ego-motion compensation step: expressing every return in a
    common frame is what makes a velocity estimate meaningful across scans.
    Pass the **estimated** pose, not ground truth -- see the module docstring.
    """
    r = np.asarray(ranges, dtype=np.float64)
    b = np.asarray(bearings_deg, dtype=np.float64)
    valid = r < float(max_range) - 1e-6
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    r, b = r[valid], b[valid]
    a = np.radians(float(heading_deg) + b)
    return np.column_stack((float(x) + r * np.sin(a), float(y) + r * np.cos(a)))


def cluster_scan(ranges, bearings_deg, x: float, y: float, heading_deg: float, *,
                 eps: float = cfg.CLUSTER_EPS,
                 min_points: int = cfg.CLUSTER_MIN_POINTS,
                 beam_res_deg: float = cfg.LIDAR_BEAM_RES_DEG,
                 max_range: float = cfg.LIDAR_RANGE) -> List[np.ndarray]:
    """Adaptive-breakpoint segmentation over the angularly ordered scan.

    Two consecutive returns join the same cluster when they are closer than
    `eps` **plus** the arc a single beam subtends at that range.  The adaptive
    term matters: at 0.5 deg spacing, neighbouring beams are 0.017 m apart at
    2 m but 0.14 m apart at 16 m, so a fixed threshold either over-segments
    distant objects or merges nearby ones.

    Returns cluster centroids in the world frame.  Preferred over DBSCAN here
    because a laser scan is already ordered by bearing, which turns the problem
    into a single linear pass and drops the scikit-learn dependency.
    """
    r = np.asarray(ranges, dtype=np.float64)
    b = np.asarray(bearings_deg, dtype=np.float64)
    valid = np.flatnonzero(r < float(max_range) - 1e-6)
    if valid.size == 0:
        return []

    pts = scan_to_points(r, b, x, y, heading_deg, max_range=max_range)
    beam_arc = np.radians(float(beam_res_deg)) * r[valid]

    groups: List[List[int]] = [[0]]
    for k in range(1, valid.size):
        # Adjacent in the scan? A skipped beam means a no-return in between,
        # which is itself evidence of a boundary between objects.
        contiguous = (valid[k] - valid[k - 1]) == 1
        gap = float(np.linalg.norm(pts[k] - pts[k - 1]))
        threshold = float(eps) + float(beam_arc[k])
        if contiguous and gap <= threshold:
            groups[-1].append(k)
        else:
            groups.append([k])

    # A full revolution wraps, so a cluster straddling the 180/-180 seam would
    # otherwise be split in two.
    if len(groups) > 1 and (valid[0] % len(r)) == 0 and (valid[-1] == len(r) - 1):
        gap = float(np.linalg.norm(pts[0] - pts[-1]))
        if gap <= float(eps) + float(beam_arc[-1]):
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    return [pts[g].mean(axis=0) for g in groups if len(g) >= int(min_points)]


# ---------------------------------------------------------------------------
# Step 5 -- constant-velocity Kalman filter
# ---------------------------------------------------------------------------
class Track:
    """One tracked object: constant-velocity Kalman filter over [x, y, vx, vy].

    `slot` is assigned by the observation layer on first publication and held
    until track loss (01 §6.2), so it is stored here rather than recomputed.

    `max_coast` records the longest run of consecutive missed updates: the
    tracker's occlusion tolerance, and a Study 2 reported metric.
    """

    def __init__(self, position, *, dt: float = cfg.UPDATE_RATE,
                 process_accel: float = cfg.KF_PROCESS_NOISE_ACCEL,
                 meas_noise: float = cfg.KF_MEAS_NOISE_POS) -> None:
        self.id = next(_next_track_id)
        self.dt = float(dt)
        self.q = float(process_accel) ** 2
        self.r = float(meas_noise) ** 2

        self.state = np.array([position[0], position[1], 0.0, 0.0], dtype=np.float64)
        self.cov = np.diag([self.r, self.r, cfg.KF_INIT_VEL_VAR, cfg.KF_INIT_VEL_VAR])

        self.hits = 1
        self.misses = 0
        self.max_coast = 0
        self.age = 1
        self.slot: Optional[int] = None
        # Velocity-estimate noise (Study 2 axis).  Held for the step rather than
        # resampled per read, so every consumer in one step sees one value.
        self._vel_noise = np.zeros(2, dtype=np.float64)

        # Hysteretic static/dynamic state (step 6).
        self.is_dynamic = False
        self._pending: Optional[bool] = None
        self._pending_steps = 0

    # -- accessors ---------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.state[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:] + self._vel_noise

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def course_deg(self) -> float:
        """Compass course of the estimated velocity; 0 deg = +y, clockwise."""
        vx, vy = self.velocity
        return float(np.degrees(np.arctan2(vx, vy))) % 360.0

    @property
    def confirmed(self) -> bool:
        return self.hits >= cfg.TRACK_MIN_HITS

    # -- filter ------------------------------------------------------------
    def _matrices(self, dt: float):
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        # Piecewise-constant white acceleration.
        q = self.q * np.array([
            [dt ** 4 / 4, 0.0, dt ** 3 / 2, 0.0],
            [0.0, dt ** 4 / 4, 0.0, dt ** 3 / 2],
            [dt ** 3 / 2, 0.0, dt ** 2, 0.0],
            [0.0, dt ** 3 / 2, 0.0, dt ** 2],
        ])
        return f, q

    def predict(self, dt: Optional[float] = None) -> None:
        dt = self.dt if dt is None else float(dt)
        f, q = self._matrices(dt)
        self.state = f @ self.state
        self.cov = f @ self.cov @ f.T + q
        self.age += 1

    def update(self, measurement) -> None:
        h = np.zeros((2, 4))
        h[0, 0] = h[1, 1] = 1.0
        r = self.r * np.eye(2)

        innovation = np.asarray(measurement, dtype=np.float64) - h @ self.state
        s = h @ self.cov @ h.T + r
        gain = self.cov @ h.T @ np.linalg.inv(s)
        self.state = self.state + gain @ innovation
        self.cov = (np.eye(4) - gain @ h) @ self.cov

        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1
        self.max_coast = max(self.max_coast, self.misses)

    def set_velocity_noise(self, noise) -> None:
        """Inject velocity-estimate error for this step (Study 2 axis).

        Physically this stands in for scan motion distortion plus filter
        residual: the sweep is captured across a range of poses, so a target's
        apparent displacement between revolutions carries an error that lands
        directly on the one quantity the tracker exists to produce.
        """
        self._vel_noise = np.asarray(noise, dtype=np.float64)

    # -- step 6 ------------------------------------------------------------
    def update_motion_class(self) -> None:
        """Static/dynamic split with hysteresis, so a track cannot chatter.

        Two thresholds, not one: a static track must exceed `DYNAMIC_SPEED_ON`
        to become dynamic, and a dynamic track must fall below
        `DYNAMIC_SPEED_OFF` to go back.  A candidate flip must also persist for
        `DYNAMIC_HOLD_STEPS` before it is applied.
        """
        speed = self.speed
        candidate = speed > cfg.DYNAMIC_SPEED_ON if not self.is_dynamic \
            else speed > cfg.DYNAMIC_SPEED_OFF

        if candidate == self.is_dynamic:
            self._pending = None
            self._pending_steps = 0
            return

        if self._pending == candidate:
            self._pending_steps += 1
        else:
            self._pending = candidate
            self._pending_steps = 1

        if self._pending_steps >= cfg.DYNAMIC_HOLD_STEPS:
            self.is_dynamic = candidate
            self._pending = None
            self._pending_steps = 0


# ---------------------------------------------------------------------------
# Steps 3-6 -- the tracker
# ---------------------------------------------------------------------------
class Tracker:
    """Nearest-neighbour multi-target tracker.

    Nearest-neighbour rather than JPDA: 01 §4 states it is sufficient at one
    target, and JPDA's advantage appears in clutter densities this problem does
    not reach once beyond-boundary returns are gated out.

    Two Study 2 degradation axes live here (04 §6).  Both default to the nominal
    zero case, so the tracker is exact unless a sweep asks otherwise:

    * `dropout_p` -- per-detection probability of a miss, standing in for the
      no-return process characterised from the field logs.  Sweeping it to the
      point of track loss is the "detection dropout" axis.
    * `velocity_noise` -- 1-sigma error added to each track's velocity estimate,
      standing in for scan motion distortion plus filter residual.  Sweeping it
      to the point of encounter misclassification is the "velocity noise" axis.

    The remaining two axes are injected upstream: pose drift through the
    estimated pose passed to `cluster_scan`, and occlusion through the scenario
    geometry, measured here as `max_coast`.
    """

    def __init__(self, *, dt: float = cfg.UPDATE_RATE,
                 gate_dist: float = cfg.TRACK_GATE_DIST,
                 max_misses: int = cfg.TRACK_MAX_MISSES,
                 dropout_p: float = cfg.DETECTION_DROPOUT_P,
                 velocity_noise: float = cfg.TRACK_VELOCITY_NOISE,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.dt = float(dt)
        self.gate_dist = float(gate_dist)
        self.max_misses = int(max_misses)
        self.dropout_p = float(dropout_p)
        self.velocity_noise = float(velocity_noise)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.tracks: List[Track] = []
        self.dropped_detections = 0

    def reset(self) -> None:
        self.tracks = []
        self.dropped_detections = 0

    def update(self, detections: Sequence[np.ndarray],
               dt: Optional[float] = None) -> List[Track]:
        """Advance one step against world-frame cluster centroids."""
        dt = self.dt if dt is None else float(dt)
        detections = self._apply_dropout(detections)

        for track in self.tracks:
            track.predict(dt)

        matches, unmatched = self._associate(detections)
        for track, det in matches:
            track.update(det)
        for track in self.tracks:
            if track not in [t for t, _ in matches]:
                track.mark_missed()

        for det in unmatched:
            self.tracks.append(Track(det, dt=dt))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for track in self.tracks:
            track.set_velocity_noise(self._draw_velocity_noise())
            track.update_motion_class()
        return self.confirmed_tracks()

    def _apply_dropout(self, detections):
        """Randomly discard detections (Study 2 axis)."""
        dets = list(detections)
        if self.dropout_p <= 0.0 or not dets:
            return dets
        keep = self.rng.random(len(dets)) >= self.dropout_p
        self.dropped_detections += int(len(dets) - int(keep.sum()))
        return [d for d, k in zip(dets, keep) if k]

    def _draw_velocity_noise(self) -> np.ndarray:
        if self.velocity_noise <= 0.0:
            return np.zeros(2, dtype=np.float64)
        return self.rng.normal(0.0, self.velocity_noise, size=2)

    @property
    def max_coast(self) -> int:
        """Longest occlusion any live track has survived, in steps."""
        return max((t.max_coast for t in self.tracks), default=0)

    def _associate(self, detections) -> Tuple[List[Tuple[Track, np.ndarray]], List[np.ndarray]]:
        """Greedy nearest-neighbour on the global distance matrix.

        Greedy rather than Hungarian: at three targets the two agree except in
        contrived geometries, and greedy keeps the association order stable and
        inspectable, which matters more here than optimality.
        """
        dets = [np.asarray(d, dtype=np.float64) for d in detections]
        if not self.tracks or not dets:
            return [], dets

        cost = np.array([[float(np.linalg.norm(t.position - d)) for d in dets]
                         for t in self.tracks])
        matches: List[Tuple[Track, np.ndarray]] = []
        used_t, used_d = set(), set()

        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for ti, di in order:
            ti, di = int(ti), int(di)
            if ti in used_t or di in used_d:
                continue
            if cost[ti, di] > self.gate_dist:
                break
            matches.append((self.tracks[ti], dets[di]))
            used_t.add(ti)
            used_d.add(di)

        unmatched = [d for i, d in enumerate(dets) if i not in used_d]
        return matches, unmatched

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]

    def dynamic_tracks(self) -> List[Track]:
        """Tracks that feed the target branch."""
        return [t for t in self.confirmed_tracks() if t.is_dynamic]

    def static_tracks(self) -> List[Track]:
        """Tracks whose returns continue to feed `c_t`."""
        return [t for t in self.confirmed_tracks() if not t.is_dynamic]
