"""Optional safety-only target geometry and return ownership.

The tracker deliberately estimates the visible-return centroid.  Its existing
LiDAR hull fit supplies a separate geometric centre/axis; this adapter never
feeds that correction into the track's Kalman filter, velocity or contexts.
The fit uses Zhang et al., "Efficient L-shape fitting for vehicle detection
using laser scanners", IV 2017, DOI:10.1109/IVS.2017.7995698:
https://publications.ri.cmu.edu/efficient-l-shape-fitting-for-vehicle-detection-using-laser-scanners
Our fitted-footprint return mask is an engineering association rule, not that
paper's algorithm or a safety guarantee.  Unexplained returns remain static
evidence.  Historical memory is only permanently cleared by SafetyPerception's
existing finite-ray evidence or normal expiry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import constants as cfg
import tracking
from classical import common as cc
from safety_perception import SafetyPerception


@dataclass
class _HullFit:
    centre: np.ndarray
    heading: float
    frame: int


class _ScanView:
    """Suppress inherited radial masking without modifying any shared object."""

    def __init__(self, env, ranges):
        self._env = env
        self.tracks = []
        self.gated_ranges = ranges

    def __getattr__(self, name):
        # Guard: during deepcopy/unpickling the delegate is not set yet, and an
        # unguarded lookup recursed forever (2026-10-03, trigger counterfactuals).
        delegate = self.__dict__.get("_env")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)


def _explained_points(points, hulls):
    """Known-size oriented footprints, padded by existing measurement tolerance.

    This is an ownership mask, so it uses physical vessel dimensions and the
    tracker's return-explanation tolerance, not the safety hull inflation.
    A missing or expired fit supplies no evidence for excluding a return.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    explained = np.zeros(len(points), dtype=bool)
    tolerance = float(cfg.MOTION_EXPLAIN_M)
    for hull in hulls:
        rel = points - hull.centre
        forward = np.array([math.sin(hull.heading), math.cos(hull.heading)])
        right = np.array([forward[1], -forward[0]])
        explained |= ((np.abs(rel @ forward) <= 0.5 * cfg.LOA + tolerance)
                      & (np.abs(rel @ right) <= 0.5 * cfg.BREADTH + tolerance))
    return explained


class HullSafetyPerception(SafetyPerception):
    """Drop-in perception with independent geometric-centre and mask ablations.

    Current fits require a fresh pose, a matched track and finite centre/axis.
    On a missed fit or held pose, an earlier valid fit may coast for at most
    TRACK_MAX_MISSES decisions using the measured track velocity.  The axis is
    held, matching the existing constant-heading target predictor.  Otherwise
    target geometry falls back to the unmodified track position/heading and
    unassigned scan returns are retained.  Fit uncertainty is not estimated.
    """

    def __init__(self, memory_frames=cc.SCAN_MEMORY_FRAMES, *,
                 fitted_track_centre=True, hull_return_mask=True):
        super().__init__(memory_frames=memory_frames)
        self.fitted_track_centre = bool(fitted_track_centre)
        self.hull_return_mask = bool(hull_return_mask)
        self._hull_fits = {}
        self.last_hull_stats = {}

    def _track_views(self, env):
        frame = self.frames + 1
        stale = bool(getattr(env, "pose_stale", False))
        views, original_views, hulls = [], [], []
        current_count = coast_count = corrected_count = 0
        live_ids = set()
        for track in env.tracks:
            track_id = int(track.id)
            live_ids.add(track_id)
            position = np.asarray(track.position, dtype=float).copy()
            velocity = np.asarray(track.velocity, dtype=float).copy()
            axis = getattr(track, "last_fit_heading_deg", None)
            finite_axis = axis is not None and np.isfinite(axis)
            heading = (math.radians(float(axis)) if finite_axis else
                       math.atan2(velocity[0], velocity[1])
                       if np.hypot(*velocity) > 0.05 else 0.0)
            context = env.encounter_contexts.get(track_id)
            original_views.append(cc.TrackView(track_id, position.copy(), velocity.copy(),
                                               heading, context))
            centre = getattr(track, "last_fit_centre", None)
            centre = None if centre is None else np.asarray(centre, dtype=float)
            current = (not stale and int(getattr(track, "misses", 0)) == 0
                       and centre is not None and centre.shape == (2,)
                       and np.isfinite(centre).all() and finite_axis)
            if current:
                self._hull_fits[track_id] = _HullFit(centre.copy(), heading, frame)
                current_count += 1
            cached = self._hull_fits.get(track_id)
            age = frame - cached.frame if cached is not None else math.inf
            hull = None
            if (cached is not None and age <= int(cfg.TRACK_MAX_MISSES)
                    and np.isfinite(velocity).all()):
                hull = _HullFit(cached.centre + age * cfg.UPDATE_RATE * velocity,
                                cached.heading, frame)
                hulls.append(hull)
                coast_count += int(age > 0)
            elif cached is not None:
                del self._hull_fits[track_id]
            if self.fitted_track_centre and hull is not None:
                position, heading = hull.centre.copy(), hull.heading
                corrected_count += 1
            views.append(cc.TrackView(track_id, position, velocity, heading, context))
        self._hull_fits = {key: fit for key, fit in self._hull_fits.items() if key in live_ids}
        return views, original_views, hulls, {
            "current_hull_fits": current_count, "coasted_hull_fits": coast_count,
            "corrected_track_centres": corrected_count,
            "tracks_without_hull_fit": len(views) - len(hulls),
        }

    def snapshot(self, env):
        if not self.fitted_track_centre and not self.hull_return_mask:
            self.last_hull_stats = {"enabled": False}
            return super().snapshot(env)
        views, original_views, hulls, stats = self._track_views(env)
        stale = bool(getattr(env, "pose_stale", False))
        ranges = np.asarray(getattr(env, "gated_ranges", env.lidar.ranges), dtype=float).copy()
        fresh_removed = 0
        if not stale:
            indices = np.flatnonzero(ranges < cfg.LIDAR_RANGE - 1e-5)
            x, y, heading = env.estimated_pose()
            origin = np.asarray(tracking.sensor_origin(x, y, heading))
            bearings = np.radians(np.asarray(env.lidar.bearings)[indices] + heading)
            points = origin + ranges[indices, None] * np.stack(
                [np.sin(bearings), np.cos(bearings)], axis=1)
            if self.hull_return_mask:
                remove = _explained_points(points, hulls)
            else:
                # Centre-only ablation preserves the original exclusion anchor.
                remove = np.zeros(len(points), dtype=bool)
                for track in original_views:
                    remove |= np.linalg.norm(points - track.position, axis=1) <= cc.TRACK_EXCLUSION_M
            ranges[indices[remove]] = cfg.LIDAR_RANGE
            fresh_removed = int(remove.sum())
        # The superclass still sees the original raw scan for free-space clearing,
        # and still suppresses fresh ingestion and clearing when the pose is stale.
        snap = super().snapshot(_ScanView(env, ranges))
        before_mask = len(snap.points)
        if self.hull_return_mask:
            snap.points = snap.points[~_explained_points(snap.points, hulls)]
        else:
            snap.points = cc._drop_near_tracks(snap.points, original_views,
                                              cc.TRACK_EXCLUSION_M + 0.2)
        snap.tracks = views
        self.last_memory_stats["snapshot_points"] = len(snap.points)
        self.last_hull_stats = dict(stats, enabled=True, pose_stale=stale,
                                   fresh_returns_excluded=fresh_removed,
                                   remembered_returns_masked=before_mask - len(snap.points))
        return snap
