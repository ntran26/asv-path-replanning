"""Experimental safety-only geometry for fresh, represented partial hulls.

A moving bow can span the beam while exposing little of the length. Its return
centroid is not the vessel centre, and a single-scan rectangular fit can select
the wrong axis. This adapter uses measured course only after coherent observed
translation and beam-extent checks; known-size visible-face completion follows
``tracking.hull_fit_centre`` (Zhang et al., IV2017,
https://doi.org/10.1109/IVS.2017.7995698).

Separating spatial measurement location from object centre/shape is motivated by
Granstrom, Baum & Reuter, Extended Object Tracking (2016),
https://arxiv.org/abs/1604.00970. This deterministic engineering adaptation is
not that survey's probabilistic filter or a calibrated uncertainty bound.
Course can differ from hull heading; stable occlusion can mimic a visible face.

Only an existing exact raw source ID is replaced. Persistent/synthetic views,
velocities, contexts, ordering and static points are preserved. No truth,
scenario identifiers, policy calls, extra sensor calls, or raw tracker mutations
are used. Failed or absent evidence keeps the base view exactly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math

import numpy as np

import constants as cfg
from classical import common as cc


@dataclass
class _Observation:
    frame: int
    time_s: float
    serial: int
    points: np.ndarray
    origin: np.ndarray


def _array(value, shape=None):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        return None
    return result


def _face(sample, heading):
    """Known full beam and partial length; existing fit tolerances only."""
    forward = np.array([math.sin(heading), math.cos(heading)])
    axes = np.column_stack((forward, [forward[1], -forward[0]]))
    projected = sample.points @ axes
    low, high = projected.min(axis=0), projected.max(axis=0)
    extent = high - low
    tol = float(cfg.TRACK_FIT_FULL_EXTENT_TOL_M)
    if not cfg.BREADTH - tol <= extent[1] <= cfg.BREADTH + tol:
        return None, "beam_extent_incomplete_or_oversized"
    if not 0. < extent[0] < cfg.LOA - tol:
        return None, "not_partial_length"
    if np.linalg.norm(sample.points - sample.origin, axis=1).min() <= cfg.LIDAR_MIN_RANGE + cfg.TRACK_FIT_DEAD_ZONE_TOL_M:
        return None, "minimum_range_clipping"
    sensor = sample.origin @ axes
    if low[0] <= sensor[0] <= high[0]:
        return None, "observer_inside_length_span"
    longitudinal = low[0] + cfg.LOA / 2 if sensor[0] < low[0] else high[0] - cfg.LOA / 2
    centre = axes @ np.array([longitudinal, .5 * (low[1] + high[1])])
    return (centre, low, high, extent), None


class MotionAxisPerception:
    """Wrap one base snapshot; accumulate only observations actually received.

Frames are local decision calls and times use their configured decision period.
Scan serials identify duplicates only; their numeric gaps never imply time.
Historical track clusters are never backfilled on construction or a later call.
No corrected view is coasted: current evidence is required at every decision.
    """

    def __init__(self, base_perception):
        self.base_perception = base_perception
        self._frame = 0
        self._observations = {}
        self._last_serial = {}
        self.last_motion_axis_stats = {}

    def __getattr__(self, name):
        delegate = self.__dict__.get("base_perception")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def _motion(self, track, sample, velocity):
        if len(sample.points) < cfg.TRACK_FIT_MIN_POINTS:
            return None, "insufficient_current_fit_points"
        speed = float(np.linalg.norm(velocity))
        if speed <= cfg.TRACK_FIT_PRIOR_SPEED:
            return None, "low_speed"
        evidence = getattr(track, "last_evidence", None)
        if (evidence is None or getattr(evidence, "violations", 0) < cfg.MOTION_MIN_POINTS
                or getattr(evidence, "appear", 0) <= 0):
            return None, "missing_motion_evidence"
        heading = math.atan2(velocity[0], velocity[1])
        current, reason = _face(sample, heading)
        if current is None:
            return None, reason
        good = []
        for observation in self._observations[int(track.id)]:
            fitted, _ = _face(observation, heading)
            if fitted is not None:
                good.append((observation, fitted))
        # Three measured observations is the existing persistence admission
        # requirement. Historical clusters use CLUSTER_MIN_POINTS; only the
        # current geometry additionally requires TRACK_FIT_MIN_POINTS.
        if len(good) < 3:
            return None, "insufficient_motion_observations"
        times = np.array([item[0].time_s for item in good])
        centres = np.array([item[1][0] for item in good])
        endpoint_motion = np.array([good[-1][1][i][0] - good[0][1][i][0] for i in (1, 2)])
        if np.any(endpoint_motion <= cfg.MOTION_PASS_TOL_M):
            return None, "insufficient_endpoint_translation"
        dt = times - times.mean()
        if not dt @ dt > 0:
            return None, "nonpositive_observation_duration"
        fitted_velocity = (dt[:, None] * (centres - centres.mean(axis=0))).sum(axis=0) / (dt @ dt)
        residual = np.linalg.norm(centres - centres.mean(axis=0) - dt[:, None] * fitted_velocity, axis=1)
        measured_residual = np.linalg.norm(centres - centres[-1] - (times-times[-1])[:, None] * velocity, axis=1)
        tol = float(cfg.TRACK_FIT_FULL_EXTENT_TOL_M)
        if residual.max() > tol or measured_residual.max() > tol:
            return None, "inconsistent_translation"
        forward = velocity / speed
        if (fitted_velocity @ forward <= cfg.TRACK_FIT_PRIOR_SPEED
                or np.linalg.norm(fitted_velocity) > cfg.TRACK_GATE_DIST / cfg.UPDATE_RATE):
            return None, "translation_speed_outside_tracking_gate"
        return (current[0], heading, {
            "observation_frames": [item[0].frame for item in good],
            "observation_times_s": times.tolist(),
            "scan_serials": [item[0].serial for item in good],
            "point_counts": [len(item[0].points) for item in good],
            "endpoint_translation_m": endpoint_motion.tolist(),
            "max_translation_residual_m": float(residual.max()),
            "max_measured_velocity_residual_m": float(measured_residual.max()),
            "fitted_velocity_diagnostic": fitted_velocity.tolist(),
            "observed_extents": current[3].tolist(),
            "measured_speed_mps": speed,
        }), None

    def snapshot(self, env):
        snap = self.base_perception.snapshot(env)
        self._frame += 1
        time_s = self._frame * float(cfg.UPDATE_RATE)
        window = float(cfg.MOTION_WINDOW_S)
        self._observations = {key: recent for key, values in self._observations.items()
            if (recent := [s for s in values if time_s-s.time_s <= window])}
        stale = bool(getattr(env, "pose_stale", False))
        tracker = getattr(env, "tracker", None)
        scans = list(getattr(tracker, "_scans", ()))
        raw_tracks = list(getattr(tracker, "tracks", ()))
        counts = Counter(int(t.id) for t in raw_tracks)
        represented = {int(v.id) for v in snap.tracks if int(v.id) >= 0}
        corrected, updates, reasons = {}, [], Counter()
        for track in raw_tracks:
            source = int(track.id)
            if source not in represented:
                continue
            if counts[source] != 1:
                reasons["duplicate_raw_id"] += 1
                continue
            if stale or int(getattr(track, "misses", 0)):
                reasons["pose_stale" if stale else "missed_detection"] += 1
                continue
            if not getattr(track, "confirmed", False) or int(getattr(track, "hits", 0)) < cfg.TRACK_MIN_HITS:
                reasons["unconfirmed"] += 1
                continue
            velocity = _array(track.velocity, (2,))
            position = _array(track.position, (2,))
            covariance = _array(getattr(track, "cov", None), (4, 4))
            if velocity is None or position is None or covariance is None:
                reasons["invalid_track_state"] += 1
                continue
            history = getattr(track, "history", ())
            if not history or not scans:
                reasons["missing_current_cluster_or_scan"] += 1
                continue
            serial, points = history[-1]
            if int(serial) != int(scans[-1].serial):
                reasons["cluster_is_not_current_scan"] += 1
                continue
            if self._last_serial.get(source) == int(serial):
                reasons["repeated_observation"] += 1
                continue
            points, origin = _array(points), _array(scans[-1].origin, (2,))
            if (points is None or points.ndim != 2 or points.shape[1] != 2
                    or len(points) < cfg.CLUSTER_MIN_POINTS or origin is None):
                reasons["invalid_cluster_or_origin"] += 1
                continue
            sample = _Observation(self._frame, time_s, int(serial), points.copy(), origin.copy())
            self._observations.setdefault(source, []).append(sample)
            self._last_serial[source] = int(serial)
            motion, reason = self._motion(track, sample, velocity)
            if motion is None:
                reasons[reason] += 1
                continue
            centre, heading, evidence = motion
            corrected[source] = (centre, heading)
            updates.append(dict(source_id=source, centre=centre.tolist(), heading_rad=heading, **evidence))
        # Metadata for vanished IDs is bounded too, but stale/repeated current
        # raw IDs retain their last serial so an old cluster cannot be refreshed.
        live_ids = {int(t.id) for t in raw_tracks}
        self._last_serial = {k: v for k, v in self._last_serial.items() if k in live_ids or k in self._observations}
        self.last_motion_axis_stats = {
            "frame": self._frame, "time_s": time_s, "pose_stale": stale,
            "base_track_count": len(snap.tracks), "replaced_source_ids": sorted(corrected),
            "updates": updates, "rejections": dict(reasons),
            "stored_observation_counts": {k: len(v) for k, v in self._observations.items()},
            "static_points_preserved": len(snap.points),
        }
        if not corrected:
            return snap
        views = [cc.TrackView(view.id, corrected[int(view.id)][0].copy(), view.velocity,
                              corrected[int(view.id)][1], view.ctx)
                 if int(view.id) in corrected else view for view in snap.tracks]
        return replace(snap, tracks=views)
