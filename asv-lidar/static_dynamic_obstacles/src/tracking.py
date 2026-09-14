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
2. **Cluster** the survivors                 -> `segment_scan`
3. **Ego-motion compensate** using odometry  -> world-frame points, via the pose
4. **Associate** clusters to tracks          -> nearest-neighbour
5. **Estimate** velocity per track           -> constant-velocity Kalman filter
6. **Classify** static vs dynamic            -> free-space consistency (F31)

Static clusters keep feeding `c_t`; dynamic tracks feed the target branch.

Step 6, and why it is not a speed threshold (F31)
-------------------------------------------------
01 classified by the Kalman speed of the cluster **centroid**.  A centroid is
the mean of whatever face the sensor can see, and that face changes as the own
ship moves past a panel: the centroid slides along it at 0.17-0.24 m/s with pose
noise off, and panels were promoted to target vessels on 28-33 % of frames.  No
threshold separates that from a 0.39 m/s target, so the fix is a different
question, not a different number.

`MotionEvidence` asks what motion physically is.  Over a window `T_w` it counts

* **appear** -- returns in space a ray passed straight through `T_w` ago, with
  no return near them then;
* **vacate** -- the track's returns from `T_w` ago, in space a ray now passes
  straight through, with no return near them now.

A static solid can do neither from any viewpoint: a ray passing through a point
on its boundary must already have hit it.  A newly revealed face was occluded,
not empty, so revealing it produces nothing.  Centroid sliding therefore cannot
reach the classifier at all.  What can is pose error, which moves every point
by the same amount -- and that is what `MOTION_PASS_TOL_M` is sized from,
exactly as 03a §6.3 sizes its velocity threshold.  The speed classifier is kept
(`classifier="speed"`) as 01's original and as the ablation.

The coupling that matters
-------------------------
Step 3 is where 01 and 05 meet.  Clusters are lifted into the world frame using
the *estimated* pose, so odometry drift appears directly as apparent motion of
genuinely **static** objects, and scan-matching odometry is itself corrupted by
the moving objects in the scan.  `Tracker` therefore takes the estimated pose,
never a ground-truth one -- and lifts from the **sensor** origin, which sits
`LIDAR_OFFSET_M` ahead of the vessel reference point.  Lifting from the vessel
origin moved every point 0.86 m along the heading, so a static object appeared
to move by 0.86 m per radian of heading change.

Biased toward **under**-detection throughout: promoting a static panel to a
target ship is a false positive with COLREGs consequences, and now an emergency
stop.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

import constants as cfg

_next_track_id = itertools.count(1)
_next_scan_serial = itertools.count(1)

# The lateral pose margin in `ScanFrame.passes_through` spans `atan(tol/rho)`
# either side of a point; at the 1 m dead-zone edge and 0.25 m that is 14 deg,
# 28 beams.  Capped so a point near the sensor cannot demand a quarter-sweep.
_MAX_LATERAL_BEAMS = 30


# ---------------------------------------------------------------------------
# Step 2 -- clustering
# ---------------------------------------------------------------------------
def scan_to_points(ranges, bearings_deg, x: float, y: float,
                   heading_deg: float, *, max_range: float = cfg.LIDAR_RANGE) -> np.ndarray:
    """Lift a body-frame scan into world coordinates, dropping no-returns.

    `(x, y)` is the **sensor** position.  This is the ego-motion compensation
    step: expressing every return in a common frame is what makes motion
    meaningful across scans.  Pass the **estimated** pose -- see the module
    docstring.
    """
    r = np.asarray(ranges, dtype=np.float64)
    b = np.asarray(bearings_deg, dtype=np.float64)
    valid = r < float(max_range) - 1e-6
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    r, b = r[valid], b[valid]
    a = np.radians(float(heading_deg) + b)
    return np.column_stack((float(x) + r * np.sin(a), float(y) + r * np.cos(a)))


def sensor_origin(x: float, y: float, heading_deg: float,
                  offset: float = None) -> Tuple[float, float]:
    """Where the LiDAR sits for a vessel reference point and heading."""
    if offset is None:
        from ship import LIDAR_OFFSET_M
        offset = LIDAR_OFFSET_M
    a = np.radians(float(heading_deg))
    return float(x) + float(offset) * float(np.sin(a)), float(y) + float(offset) * float(np.cos(a))


@dataclass
class Cluster:
    """One segment of the scan: its centroid and the world-frame returns in it."""

    centroid: np.ndarray
    points: np.ndarray

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.centroid, dtype=dtype)

    def __getitem__(self, index):
        return self.centroid[index]

    def __len__(self) -> int:
        return len(self.centroid)


def segment_scan(ranges, bearings_deg, x: float, y: float, heading_deg: float, *,
                 eps: float = cfg.CLUSTER_EPS,
                 min_points: int = cfg.CLUSTER_MIN_POINTS,
                 beam_res_deg: float = cfg.LIDAR_BEAM_RES_DEG,
                 max_range: float = cfg.LIDAR_RANGE) -> List[Cluster]:
    """Adaptive-breakpoint segmentation over the angularly ordered scan.

    Two consecutive returns join the same cluster when they are closer than
    `eps` **plus** the arc a single beam subtends at that range.  The adaptive
    term matters: at 0.5 deg spacing, neighbouring beams are 0.017 m apart at
    2 m but 0.14 m apart at 16 m, so a fixed threshold either over-segments
    distant objects or merges nearby ones.

    Preferred over DBSCAN here because a laser scan is already ordered by
    bearing, which turns the problem into a single linear pass and drops the
    scikit-learn dependency.
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

    return [Cluster(pts[g].mean(axis=0), pts[g]) for g in groups if len(g) >= int(min_points)]


def cluster_scan(ranges, bearings_deg, x: float, y: float, heading_deg: float,
                 **kwargs) -> List[np.ndarray]:
    """`segment_scan`, returning centroids only -- the 01 interface."""
    return [c.centroid for c in segment_scan(ranges, bearings_deg, x, y, heading_deg, **kwargs)]


# ---------------------------------------------------------------------------
# Step 6 -- free-space consistency
# ---------------------------------------------------------------------------
@dataclass
class ScanFrame:
    """One revolution as the free-space test needs it: where every ray went.

    Built from the **raw** scan -- facility walls included -- because a ray that
    ends on a wall has passed through everything in front of it, and that
    pass-through is the evidence.

    **A no-return is not evidence of anything.**  The first version counted it
    as free space to the horizon, and T8 found the flaw at 1.1 % of frames: the
    C1 reports nothing for a surface inside its 1 m dead zone, so a panel the
    vessel was brushing past read as empty, and its far face "appeared" there
    two seconds later.  Dropout, the aft mask and the matte black wall produce
    the same ambiguity.  Only a ray that returned from beyond a point has seen
    through it.
    """

    origin: np.ndarray
    heading_deg: float
    ranges: np.ndarray
    points: np.ndarray
    serial: int = 0
    max_range: float = cfg.LIDAR_RANGE

    @classmethod
    def from_scan(cls, ranges, bearings_deg, origin_xy, heading_deg: float, *,
                  max_range: float = cfg.LIDAR_RANGE) -> "ScanFrame":
        b = np.asarray(bearings_deg, dtype=np.float64)
        n = b.size
        # The beam lookup below assumes the C1's layout: bin 0 dead ahead,
        # bins advancing clockwise at 360/n.  `lidar_pooling.beam_bearings`.
        assert abs(b[0]) < 1e-9 and abs(((b[1] - b[0]) % 360.0) - 360.0 / n) < 1e-9
        r = np.asarray(ranges, dtype=np.float64).copy()
        origin = np.asarray(origin_xy, dtype=np.float64)
        return cls(origin=origin, heading_deg=float(heading_deg), ranges=r,
                   points=scan_to_points(r, b, origin[0], origin[1], heading_deg,
                                         max_range=max_range),
                   serial=next(_next_scan_serial), max_range=float(max_range))

    def passes_through(self, points: np.ndarray, tol: float) -> np.ndarray:
        """True where this scan's rays went at least `tol` beyond each point.

        The beam toward the point **and both neighbours** must clear it, so a
        point between two beams is judged by the more conservative of them and a
        grazing corner cannot count.  Points inside the dead zone are unknown.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if pts.size == 0:
            return np.zeros(0, dtype=bool)
        n = self.ranges.size
        rel = pts - self.origin[None, :]
        rho = np.hypot(rel[:, 0], rel[:, 1])
        bearing = np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) - self.heading_deg
        idx = np.rint((bearing % 360.0) / (360.0 / n)).astype(np.int64) % n
        ok = rho >= cfg.LIDAR_MIN_RANGE
        # The beam toward the point must have *returned* from beyond it: that is
        # the only certificate of empty space.  An empty beam certifies nothing
        # but blocks nothing, so the others need only not be short of it --
        # demanding returns from all of them cost 1.5 s of head-on latency,
        # because a 0.5 m beam at 15 m spans four beams and the outer two look
        # past the hull into open water.
        centre = self.ranges[idx]
        ok &= (centre < self.max_range - 1e-6) & (centre > rho + float(tol))
        # **The certificate must survive a pose error of `tol` in any direction,
        # not only along the ray.**  Checking the two adjacent beams guarded the
        # radial direction alone, and T8 with 3 cm of pose noise found the gap at
        # 0.17 % of frames: a panel face seen edge-on, as the vessel runs
        # parallel to it, needs only a centimetre of sideways error for the ray
        # toward a face point to graze past the corner and return from the wall.
        # So every beam within `tol` laterally of the point must clear it too.
        spread = np.degrees(np.arctan2(float(tol), np.maximum(rho, 1e-6)))
        half = np.clip(np.ceil(spread / (360.0 / n)), 1, _MAX_LATERAL_BEAMS).astype(np.int64)
        for k in range(1, int(half.max()) + 1):
            within = half >= k
            ok &= ~within | ((self.ranges[(idx + k) % n] > rho + float(tol))
                             & (self.ranges[(idx - k) % n] > rho + float(tol)))
        return ok

    def explains(self, points: np.ndarray, radius: float) -> np.ndarray:
        """True where this scan had a return within `radius` of each point."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if pts.size == 0 or self.points.size == 0:
            return np.zeros(len(pts), dtype=bool)
        lo = pts.min(axis=0) - radius
        hi = pts.max(axis=0) + radius
        near = self.points[np.all((self.points >= lo) & (self.points <= hi), axis=1)]
        if near.size == 0:
            return np.zeros(len(pts), dtype=bool)
        d2 = np.min(np.sum((pts[:, None, :] - near[None, :, :]) ** 2, axis=-1), axis=1)
        return d2 <= float(radius) ** 2


@dataclass
class MotionEvidence:
    """One update's verdict on whether a track moved over the window."""

    appear: int
    vacate: int
    compared: int

    @property
    def violations(self) -> int:
        return self.appear + self.vacate

    @property
    def moving(self) -> bool:
        return (self.violations >= cfg.MOTION_MIN_POINTS
                and self.violations >= cfg.MOTION_MIN_FRAC * max(self.compared, 1))


def motion_evidence(current_points: np.ndarray, current_scan: ScanFrame,
                    reference_points: np.ndarray, reference_scan: ScanFrame, *,
                    tol: float = cfg.MOTION_PASS_TOL_M,
                    explain: float = cfg.MOTION_EXPLAIN_M) -> MotionEvidence:
    """Appear and vacate counts between a track's two observations."""
    cur = np.atleast_2d(np.asarray(current_points, dtype=np.float64))
    ref = np.atleast_2d(np.asarray(reference_points, dtype=np.float64))
    appear = vacate = 0
    if cur.size:
        appear = int(np.sum(reference_scan.passes_through(cur, tol)
                            & ~reference_scan.explains(cur, explain)))
    if ref.size:
        vacate = int(np.sum(current_scan.passes_through(ref, tol)
                            & ~current_scan.explains(ref, explain)))
    return MotionEvidence(appear, vacate, len(cur) + len(ref))


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

        # Step 6 state.  `history` holds (scan serial, world points) for the
        # last `MOTION_WINDOW_STEPS` detections, oldest first.
        self.is_dynamic = False
        self._pending: Optional[bool] = None
        self._pending_steps = 0
        self.history: Deque[Tuple[int, np.ndarray]] = deque(maxlen=cfg.MOTION_WINDOW_STEPS)
        self.evidence_run = 0
        self.quiet_run = 0
        self.last_evidence: Optional[MotionEvidence] = None

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
        """01's speed classifier: two thresholds, held for `DYNAMIC_HOLD_STEPS`.

        A static track must exceed `DYNAMIC_SPEED_ON` to become dynamic, and a
        dynamic track must fall below `DYNAMIC_SPEED_OFF` to go back.  Kept as
        the ablation of the free-space classifier and for detections that carry
        no points.
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

    def apply_evidence(self, evidence: Optional[MotionEvidence]) -> None:
        """03a §6.3's asymmetric hysteresis, over free-space evidence.

        Promote after `DYNAMIC_PROMOTE_STEPS` consecutive updates showing motion;
        demote after `DYNAMIC_DEMOTE_STEPS` consecutive updates showing none.
        `None` -- a track too young to have a reference -- changes nothing.
        """
        self.last_evidence = evidence
        if evidence is None:
            return
        if evidence.moving:
            self.evidence_run += 1
            self.quiet_run = 0
        else:
            self.quiet_run += 1
            self.evidence_run = 0
        if not self.is_dynamic and self.evidence_run >= cfg.DYNAMIC_PROMOTE_STEPS:
            self.is_dynamic = True
        elif self.is_dynamic and self.quiet_run >= cfg.DYNAMIC_DEMOTE_STEPS:
            self.is_dynamic = False


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
    estimated pose passed to `segment_scan`, and occlusion through the scenario
    geometry, measured here as `max_coast`.
    """

    def __init__(self, *, dt: float = cfg.UPDATE_RATE,
                 gate_dist: float = cfg.TRACK_GATE_DIST,
                 max_misses: int = cfg.TRACK_MAX_MISSES,
                 dropout_p: float = cfg.DETECTION_DROPOUT_P,
                 velocity_noise: float = cfg.TRACK_VELOCITY_NOISE,
                 classifier: str = cfg.MOTION_CLASSIFIER,
                 rng: Optional[np.random.Generator] = None) -> None:
        if classifier not in ("free_space", "speed"):
            raise ValueError(f"unknown motion classifier {classifier!r}")
        self.dt = float(dt)
        self.gate_dist = float(gate_dist)
        self.max_misses = int(max_misses)
        self.dropout_p = float(dropout_p)
        self.velocity_noise = float(velocity_noise)
        self.classifier = classifier
        self.rng = rng if rng is not None else np.random.default_rng()
        self.tracks: List[Track] = []
        self.dropped_detections = 0
        self._scans: Deque[ScanFrame] = deque(maxlen=cfg.MOTION_WINDOW_STEPS + 1)

    def reset(self) -> None:
        self.tracks = []
        self.dropped_detections = 0
        self._scans.clear()

    def update(self, detections: Sequence, dt: Optional[float] = None, *,
               scan: Optional[ScanFrame] = None) -> List[Track]:
        """Advance one step against world-frame detections.

        `detections` are `Cluster`s, or bare centroids for callers that have no
        points.  The free-space classifier needs both the clusters' points and
        this revolution's `scan`; without them a track falls back to 01's speed
        classifier for that update.
        """
        dt = self.dt if dt is None else float(dt)
        detections = self._apply_dropout(detections)

        for track in self.tracks:
            track.predict(dt)

        matches, unmatched = self._associate(detections)
        matched_ids = {id(t) for t, _ in matches}
        for track, det in matches:
            track.update(_centroid(det))
        for track in self.tracks:
            if id(track) not in matched_ids:
                track.mark_missed()

        use_space = self.classifier == "free_space" and scan is not None
        for track, det in matches:
            points = _points(det)
            if use_space and points is not None:
                track.apply_evidence(self._evidence(track, points, scan))
                track.history.append((scan.serial, points))

        for det in unmatched:
            fresh = Track(_centroid(det), dt=dt)
            points = _points(det)
            if use_space and points is not None:
                fresh.history.append((scan.serial, points))
            self.tracks.append(fresh)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for track in self.tracks:
            track.set_velocity_noise(self._draw_velocity_noise())
            if not use_space:
                track.update_motion_class()

        if scan is not None:
            self._scans.append(scan)
        return self.confirmed_tracks()

    def _evidence(self, track: Track, points: np.ndarray,
                  scan: ScanFrame) -> Optional[MotionEvidence]:
        """Compare against the track's oldest observation inside the window."""
        if not track.history:
            return None
        serial, reference_points = track.history[0]
        reference = next((s for s in self._scans if s.serial == serial), None)
        if reference is None:
            return None
        return motion_evidence(points, scan, reference_points, reference)

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

    def _associate(self, detections):
        """Greedy nearest-neighbour on the global distance matrix.

        Greedy rather than Hungarian: at three targets the two agree except in
        contrived geometries, and greedy keeps the association order stable and
        inspectable, which matters more here than optimality.
        """
        dets = list(detections)
        if not self.tracks or not dets:
            return [], dets

        centres = np.array([_centroid(d) for d in dets])
        cost = np.array([[float(np.linalg.norm(t.position - c)) for c in centres]
                         for t in self.tracks])
        matches = []
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


def _centroid(detection) -> np.ndarray:
    if isinstance(detection, Cluster):
        return np.asarray(detection.centroid, dtype=np.float64)
    return np.asarray(detection, dtype=np.float64)


def _points(detection) -> Optional[np.ndarray]:
    return detection.points if isinstance(detection, Cluster) else None
