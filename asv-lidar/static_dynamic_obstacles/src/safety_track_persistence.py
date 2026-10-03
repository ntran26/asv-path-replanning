"""Experimental safety-only motion evidence and bounded target persistence.

The motivating saved traces contain full-length moving hull returns against
open water, where finite-background free-space evidence cannot establish motion.
Partial occlusion then shortens the observed cluster and biases its centroid
velocity towards zero before the raw tracker deletes its ID.

This adapter instead requires coherent translation of BOTH observed longitudinal
ends across at least three substantially full-length clusters. It preserves that
measured motion through partial visibility/raw-ID loss. It does not change the
shared tracker, dynamic labels, policy observations, or static-return ownership.

Related architectural ideas: Zhang et al., IV2017, DOI10.1109/IVS.2017.7995698
(oriented extent fitting); Yoon et al., arXiv:1809.06972, Sec.III-C (finite-ray
free-space consistency); Nuss et al., arXiv:1605.02406 (dynamic occupancy
prediction/update). This small deterministic adapter implements none of their
full algorithms. Its extent/translation gates and finite coast duration are
engineering hypotheses, not calibrated existence probabilities or safety bounds.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math

import numpy as np

import constants as cfg
import tracking
from classical import common as cc
from safety_provisional_tracks import _current_fit


@dataclass
class _Observation:
    frame: int
    serial: int
    points: np.ndarray


@dataclass
class _Anchor:
    source_id: int
    synthetic_id: int
    frame: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float


class TrackPersistencePerception:
    """Append distinguishable hypotheses to one existing onboard snapshot.

``enable_admission`` permits the independent observed-motion rule to add a raw
ID absent from the base snapshot. ``enable_persistence`` permits an anchor to
survive subsequent missing/partial measurements, independently of raw-ID life.
Age always advances from the last qualifying ACTUAL observation. No-return,
stale pose, occlusion, and the minimum-range dead zone cannot clear an anchor;
fully contradicted predicted hulls and the explicit age limit can.

Construct a fresh adapter at episode reset. The base snapshot is called once.
No truth state, target object, scenario label, or hidden behaviour is accessed.
    """

    def __init__(self, base_perception, *, enable_admission=True,
                 enable_persistence=True, max_coast_s=8.0,
                 min_motion_observations=3):
        if not np.isfinite(max_coast_s) or max_coast_s < 0:
            raise ValueError("max_coast_s must be finite and nonnegative")
        if (int(min_motion_observations) != min_motion_observations
                or min_motion_observations < 3):
            raise ValueError("min_motion_observations must be an integer >=3")
        self.base_perception = base_perception
        self.enable_admission = bool(enable_admission)
        self.enable_persistence = bool(enable_persistence)
        self.max_coast_s = float(max_coast_s)
        self.min_motion_observations = int(min_motion_observations)
        self._frame = 0
        self._observations = {}
        self._anchors = {}
        self._next_id = -1000000
        self.last_track_persistence_stats = {}

    def __getattr__(self, name):
        delegate = self.__dict__.get("base_perception")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def _motion(self, track):
        fit, reason = _current_fit(track)
        if fit is None:
            return None, reason
        centre, heading = fit
        forward = np.array([math.sin(heading), math.cos(heading)])
        axes = np.column_stack((forward, [forward[1], -forward[0]]))
        velocity = np.asarray(track.velocity, dtype=float)
        if (velocity.shape != (2,) or not np.isfinite(velocity).all()
                or float(velocity @ forward) <= cfg.TRACK_FIT_PRIOR_SPEED):
            return None, "low_or_inconsistent_measured_speed"
        tol = float(cfg.TRACK_FIT_FULL_EXTENT_TOL_M)
        good = []
        for sample in self._observations.get(int(track.id), ()):
            projected = sample.points @ axes
            low, high = projected.min(axis=0), projected.max(axis=0)
            extent = high - low
            if (len(sample.points) >= cfg.TRACK_FIT_MIN_POINTS
                    and cfg.LOA - tol <= extent[0] <= cfg.LOA + tol
                    and extent[1] <= cfg.BREADTH + tol):
                good.append((sample.frame, low, high))
        if not good or good[-1][0] != self._frame:
            return None, "current_extent_partial"
        if len(good) < self.min_motion_observations:
            return None, "insufficient_full_extent_observations"
        # A changing visible segment can slide its centroid. Both fully observed
        # longitudinal ends must instead have moved beyond the pose tolerance.
        endpoint_motion = np.array([good[-1][i][0] - good[0][i][0] for i in (1, 2)])
        if np.any(endpoint_motion <= cfg.MOTION_PASS_TOL_M):
            return None, "insufficient_endpoint_translation"
        times = np.array([item[0] - good[0][0] for item in good]) * cfg.UPDATE_RATE
        centres = np.array([0.5 * (item[1] + item[2]) for item in good])
        dt = times - times.mean()
        local_velocity = (dt[:, None] * (centres - centres.mean(axis=0))).sum(axis=0) / (dt @ dt)
        residual = np.linalg.norm(centres - centres.mean(axis=0) - dt[:, None] * local_velocity, axis=1)
        if float(residual.max()) > tol:
            return None, "inconsistent_translation"
        measured_velocity = axes @ local_velocity
        if (local_velocity[0] <= cfg.TRACK_FIT_PRIOR_SPEED
                or np.linalg.norm(measured_velocity) > cfg.TRACK_GATE_DIST / cfg.UPDATE_RATE):
            return None, "translation_speed_outside_tracking_gate"
        return (centre, measured_velocity, heading, {
            "observations": len(good), "observation_frames": [item[0] for item in good],
            "endpoint_translation_m": endpoint_motion.tolist(),
            "max_translation_residual_m": float(residual.max()),
        }), None

    @staticmethod
    def _scan(env, snap):
        raw = getattr(env, "raw_ranges", None)
        bearings = getattr(getattr(env, "lidar", None), "bearings", None)
        if raw is None or bearings is None:
            return None
        raw, bearings = np.asarray(raw, dtype=float), np.asarray(bearings, dtype=float)
        if (raw.ndim != 1 or raw.shape != bearings.shape or len(raw) < 2
                or not np.isfinite(raw).all() or not np.isfinite(bearings).all()):
            return None
        origin = np.asarray(tracking.sensor_origin(snap.x, snap.y, math.degrees(snap.heading)))
        points = tracking.scan_to_points(raw, bearings, *origin, math.degrees(snap.heading))
        # Direct construction avoids advancing the shared scan-serial counter.
        return tracking.ScanFrame(origin, math.degrees(snap.heading), raw.copy(), points,
                                  serial=0, max_range=float(cfg.LIDAR_RANGE))

    @staticmethod
    def _contradicted(anchor, position, scan):
        forward = np.array([math.sin(anchor.heading), math.cos(anchor.heading)])
        right = np.array([forward[1], -forward[0]])
        samples = np.array([position] + [
            position + a * cfg.LOA / 2 * forward + b * cfg.BREADTH / 2 * right
            for a in (-1, 1) for b in (-1, 1)])
        return bool(scan.passes_through(samples, cfg.MOTION_PASS_TOL_M).all()
                    and not scan.explains(samples, cfg.MOTION_EXPLAIN_M).any())

    def snapshot(self, env):
        snap = self.base_perception.snapshot(env)
        self._frame += 1
        frame = self._frame
        stale = bool(getattr(env, "pose_stale", False))
        current_ids = {int(view.id) for view in snap.tracks}
        reasons, updates, expired, contradicted = Counter(), [], [], []
        window = float(cfg.MOTION_WINDOW_S)
        self._observations = {
            key: recent for key, observations in self._observations.items()
            if (recent := [s for s in observations if (frame-s.frame)*cfg.UPDATE_RATE <= window])
        }
        for track in getattr(getattr(env, "tracker", None), "tracks", ()):
            source_id = int(track.id)
            if stale or int(getattr(track, "misses", 0)):
                reasons["pose_stale" if stale else "missed_detection"] += 1
                continue
            history = getattr(track, "history", ())
            if not history:
                reasons["missing_cluster"] += 1
                continue
            serial, points = history[-1]
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
                reasons["invalid_cluster"] += 1
                continue
            samples = self._observations.setdefault(source_id, [])
            if samples and samples[-1].serial == int(serial):
                reasons["repeated_observation"] += 1
                continue
            samples.append(_Observation(frame, int(serial), points.copy()))
            if int(getattr(track, "hits", 0)) < cfg.TRACK_MIN_HITS:
                reasons["unconfirmed"] += 1
                continue
            if not self.enable_admission and source_id not in current_ids and source_id not in self._anchors:
                reasons["independent_admission_disabled"] += 1
                continue
            motion, reason = self._motion(track)
            if motion is None:
                reasons[reason] += 1
                continue
            centre, velocity, heading, evidence = motion
            previous = self._anchors.get(source_id)
            if previous is None:
                occupied = current_ids | {a.synthetic_id for a in self._anchors.values()}
                while self._next_id in occupied:
                    self._next_id -= 1
                synthetic_id, self._next_id = self._next_id, self._next_id - 1
            else:
                synthetic_id = previous.synthetic_id
            self._anchors[source_id] = _Anchor(source_id, synthetic_id, frame,
                                               centre.copy(), velocity.copy(), heading)
            updates.append(dict(source_id=source_id, synthetic_id=synthetic_id,
                                kind="admission" if previous is None else "measured_refresh",
                                base_admitted=source_id in current_ids, **evidence))

        scan = None if stale else self._scan(env, snap)
        additions, details = [], []
        for source_id, anchor in list(self._anchors.items()):
            age_s = (frame-anchor.frame) * float(cfg.UPDATE_RATE)
            if age_s > self.max_coast_s or (age_s > 0 and not self.enable_persistence):
                expired.append(source_id)
                del self._anchors[source_id]
                continue
            position = anchor.position + age_s * anchor.velocity
            if age_s > 0 and scan is not None and self._contradicted(anchor, position, scan):
                contradicted.append(source_id)
                del self._anchors[source_id]
                continue
            if anchor.synthetic_id in current_ids:
                raise ValueError("Base perception reused a persistent hypothesis ID")
            additions.append(cc.TrackView(anchor.synthetic_id, position.copy(),
                                           anchor.velocity.copy(), anchor.heading, None))
            details.append(dict(source_id=source_id, id=anchor.synthetic_id,
                                anchor_frame=anchor.frame, age_s=age_s,
                                position=position.tolist(), velocity=anchor.velocity.tolist()))
        self.last_track_persistence_stats = {
            "frame": frame, "pose_stale": stale, "enable_admission": self.enable_admission,
            "enable_persistence": self.enable_persistence, "max_coast_s": self.max_coast_s,
            "min_motion_observations": self.min_motion_observations,
            "base_track_count": len(snap.tracks), "added_hypotheses": len(additions),
            "updates": updates, "hypotheses": details, "rejections": dict(reasons),
            "expired_source_ids": expired, "contradicted_source_ids": contradicted,
            "static_points_preserved": len(snap.points),
        }
        return replace(snap, tracks=list(snap.tracks) + additions)
