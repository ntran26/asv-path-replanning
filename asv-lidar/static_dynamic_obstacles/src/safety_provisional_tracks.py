"""Safety hypotheses from onboard tracks before/after dynamic publication.

Dynamic classification needs persistent free-space evidence.  Its publication
gate is appropriate for encounter labels but can hide a confirmed approaching
cluster from the safety predictor.  This adapter adds conservative hypotheses
after ordinary perception, preserving every static return from that snapshot.
It never changes tracking, policy observations, or encounter classification.

The existing fitted extent follows Zhang et al., IV 2017,
DOI:10.1109/IVS.2017.7995698.  The admission/retention rules here are engineering
rules, not that paper's algorithm or a guarantee: a visible static edge may
still resemble a vessel, and centroid velocity need not equal vessel velocity.
The optional initial-admission evidence gate reuses existing finite-ray motion
counts. Related prior work: Yoon, Tang and Barfoot (2018), Sec. III-C,
https://arxiv.org/abs/1809.06972. Their free-space check rejects viewpoint-induced
motion labels; this adapter does not implement their 3D detection pipeline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math

import numpy as np

import constants as cfg
from classical import common as cc


@dataclass
class _Hypothesis:
    position: np.ndarray
    raw_position: np.ndarray
    velocity: np.ndarray
    heading: float
    frame: int
    was_dynamic: bool = False


def _vector(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.shape == (2,) and np.isfinite(array).all() else None


def _current_fit(track):
    """Return current onboard fit only when its observed extent fits a hull.

Both oriented extents must fit physical LOA/BREADTH plus the tracker's existing
full-extent tolerance.  No lower extent is required: partial visibility is
normal, so this rejects oversized panels without claiming object recognition.
    """
    centre = _vector(getattr(track, "last_fit_centre", None))
    axis = getattr(track, "last_fit_heading_deg", None)
    if centre is None or axis is None or not np.isfinite(axis):
        return None, "invalid_fit"
    history = getattr(track, "history", ())
    if not history:
        return None, "missing_cluster"
    points = np.asarray(history[-1][1], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2
            or len(points) < int(cfg.TRACK_FIT_MIN_POINTS)
            or not np.isfinite(points).all()):
        return None, "invalid_cluster"
    heading = math.radians(float(axis))
    forward = np.array([math.sin(heading), math.cos(heading)])
    right = np.array([forward[1], -forward[0]])
    rel = points - centre
    tolerance = float(cfg.TRACK_FIT_FULL_EXTENT_TOL_M)
    if (np.ptp(rel @ forward) > float(cfg.LOA) + tolerance
            or np.ptp(rel @ right) > float(cfg.BREADTH) + tolerance):
        return None, "oversized_cluster"
    return (centre.copy(), heading), None


class ProvisionalTrackPerception:
    """Wrap a perception instance without adding provisional return ownership.

New hypotheses require a matched confirmed raw track, a fresh pose, speed above
TRACK_FIT_PRIOR_SPEED, and a current vessel-sized fit.  Their last valid fit may
coast for TRACK_MAX_MISSES decisions.  A previously published dynamic ID remains
represented while the raw tracker retains it, even after dynamic demotion; its
geometric offset is carried with the measured raw track.  Held poses never admit
or refresh hypotheses, and coasting then expires after TRACK_MAX_MISSES decisions.

    The caller must construct/reset this adapter with the underlying perception at
    episode reset, just as for the existing perception's scan memory.

    With require_motion_evidence=True, a new provisional ID also needs at least
    MOTION_MIN_POINTS appear/vacate violations.  The publication fraction and
    hysteresis are not required.  An admitted ID may refresh a valid current fit
    without repeated evidence, since visibility changes can hide later motion.
    Published-track retention is independent of this optional admission gate.
    """

    def __init__(self, base_perception, *, require_motion_evidence=False):
        self.base_perception = base_perception
        self.require_motion_evidence = bool(require_motion_evidence)
        self._frames = 0
        self._hypotheses = {}
        self.last_provisional_stats = {}

    def __getattr__(self, name):
        # Guard: during deepcopy/unpickling the delegate is not set yet, and an
        # unguarded lookup recursed forever (2026-10-03, trigger counterfactuals).
        delegate = self.__dict__.get("base_perception")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def snapshot(self, env):
        snap = self.base_perception.snapshot(env)
        self._frames += 1
        frame = self._frames
        stale = bool(getattr(env, "pose_stale", False))
        raw_tracks = list(getattr(getattr(env, "tracker", None), "tracks", ()))
        raw_by_id = {int(track.id): track for track in raw_tracks}
        self._hypotheses = {key: value for key, value in self._hypotheses.items()
                            if key in raw_by_id}
        published = {int(track.id): track for track in snap.tracks}
        additions = []
        reasons = Counter()
        current_ids, coasted_ids, retained_ids = [], [], []
        admission_events = []

        for track_id, track in raw_by_id.items():
            raw_position = _vector(track.position)
            velocity = _vector(track.velocity)
            if raw_position is None or velocity is None:
                reasons["invalid_state"] += 1
                continue
            cached = self._hypotheses.get(track_id)
            if track_id in published:
                if not stale:
                    view = published[track_id]
                    self._hypotheses[track_id] = _Hypothesis(
                        np.asarray(view.position, dtype=float).copy(), raw_position.copy(),
                        np.asarray(view.velocity, dtype=float).copy(), float(view.heading),
                        frame, was_dynamic=True)
                continue

            misses = int(getattr(track, "misses", 0))
            if misses > int(cfg.TRACK_MAX_MISSES):
                reasons["expired_raw_track"] += 1
                self._hypotheses.pop(track_id, None)
                continue
            confirmed = int(getattr(track, "hits", 0)) >= int(cfg.TRACK_MIN_HITS)
            fit, rejection = (None, "pose_stale") if stale else (
                (None, "missed_detection") if misses else _current_fit(track))
            was_dynamic = cached is not None and cached.was_dynamic
            evidence = getattr(track, "last_evidence", None)
            evidence_ok = (not self.require_motion_evidence or cached is not None
                           or (evidence is not None
                               and evidence.violations >= int(cfg.MOTION_MIN_POINTS)))

            if not stale and was_dynamic:
                # Raw KF position already advances on missed detections.  Carry
                # the last geometric offset rather than integrating twice.
                position = raw_position + cached.position - cached.raw_position
                heading = cached.heading
                if fit is not None:
                    position, heading = fit
                cached = _Hypothesis(position.copy(), raw_position.copy(), velocity.copy(),
                                     heading, frame, was_dynamic=True)
                self._hypotheses[track_id] = cached
                retained_ids.append(track_id)
            elif (not stale and not misses and confirmed
                  and float(np.linalg.norm(velocity)) > float(cfg.TRACK_FIT_PRIOR_SPEED)
                  and fit is not None and evidence_ok):
                position, heading = fit
                if cached is None:
                    admission_events.append({
                        "id": track_id, "frame": frame,
                        "motion_evidence": None if evidence is None else {
                            "appear": int(evidence.appear), "vacate": int(evidence.vacate),
                            "compared": int(evidence.compared),
                            "violations": int(evidence.violations)},
                    })
                cached = _Hypothesis(position.copy(), raw_position.copy(), velocity.copy(),
                                     heading, frame)
                self._hypotheses[track_id] = cached
                current_ids.append(track_id)
            else:
                if not confirmed:
                    rejection = "unconfirmed"
                elif not stale and not misses and np.linalg.norm(velocity) <= cfg.TRACK_FIT_PRIOR_SPEED:
                    rejection = "low_speed"
                elif fit is not None and not evidence_ok:
                    rejection = "insufficient_motion_evidence"
                reasons[rejection or "not_admitted"] += 1
                if cached is None or frame - cached.frame > int(cfg.TRACK_MAX_MISSES):
                    self._hypotheses.pop(track_id, None)
                    continue
                coasted_ids.append(track_id)
                if cached.was_dynamic:
                    retained_ids.append(track_id)

            age = frame - cached.frame
            position = cached.position + age * float(cfg.UPDATE_RATE) * cached.velocity
            # A safety hypothesis has no encounter context or COLREG label.
            additions.append(cc.TrackView(track_id, position.copy(), cached.velocity.copy(),
                                          cached.heading, None))

        self.last_provisional_stats = {
            "enabled": True, "pose_stale": stale, "raw_track_count": len(raw_tracks),
            "require_motion_evidence": self.require_motion_evidence,
            "new_admissions": admission_events,
            "published_track_count": len(published), "added_track_count": len(additions),
            "current_fit_ids": current_ids, "coasted_ids": coasted_ids,
            "retained_dynamic_ids": retained_ids, "added_ids": [view.id for view in additions],
            "rejections": dict(reasons), "static_points_preserved": len(snap.points),
        }
        # Return a new dataclass/list.  Underlying snapshots, points, memory and
        # shared tracks are never modified by the augmentation.
        return replace(snap, tracks=list(snap.tracks) + additions)
