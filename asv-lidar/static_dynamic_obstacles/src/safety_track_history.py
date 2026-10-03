"""Finite-lived kinematic hypotheses from previously admitted onboard tracks.

Motivation: Wabersich & Zeilinger (2021), predictive safety filtering with
uncertainty, Sec.4.2, https://arxiv.org/abs/1812.05506 ; Granstrom, Baum & Reuter
(2017), extended-object kinematics/extent estimation from partial measurements,
https://arxiv.org/abs/1604.00970 . The history rule below is our empirical
engineering approximation, not either paper's algorithm or a certified error
set. An old velocity can be wrong after a real manoeuvre and reject a safe plan.

This adapter appends previous measured TrackViews without replacing a current
view or removing a static return. Anchors use only an already admitted base
view, fresh matched raw track, finite current hull fit and appearance-backed
motion evidence. No simulator truth, policy observation or context is changed.
"""
from collections import Counter
from dataclasses import dataclass, replace
import math

import numpy as np

import constants as cfg
from classical import common as cc


@dataclass
class _Anchor:
    source_id: int
    synthetic_id: int
    frame: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float


def _vector(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.shape == (2,) and np.isfinite(array).all() else None


def _admission_reason(raw, view):
    if raw is None:
        return "missing_raw_track"
    if int(getattr(raw, "misses", 0)) != 0:
        return "missed_detection"
    if int(getattr(raw, "hits", 0)) < int(cfg.TRACK_MIN_HITS):
        return "unconfirmed"
    centre = _vector(getattr(raw, "last_fit_centre", None))
    axis = getattr(raw, "last_fit_heading_deg", None)
    if centre is None or axis is None or not np.isfinite(axis):
        return "missing_current_finite_fit"
    evidence = getattr(raw, "last_evidence", None)
    appear = getattr(evidence, "appear", 0)
    vacate = getattr(evidence, "vacate", 0)
    if not (np.isfinite(appear) and np.isfinite(vacate)) or appear <= 0:
        return "no_appearance_evidence"
    if appear + vacate < int(cfg.MOTION_MIN_POINTS):
        return "insufficient_motion_evidence"
    if (_vector(view.position) is None or _vector(view.velocity) is None
            or not np.isfinite(view.heading)):
        return "invalid_admitted_state"
    return None


class TrackHistoryPerception:
    """Append at most TRACK_MAX_MISSES previous views per original track ID.

At the default3-decision limit an anchor is used at ages1,2,3, then expires.
Age advances even with a held pose or absent raw ID. An anchor is never renewed
from a coasted or synthetic view. Current views/points/contexts are preserved;
only appended hypotheses have unique negative IDs and no encounter context.
Construct a fresh adapter at episode reset, as for other perception memory.
"""

    def __init__(self, base_perception):
        self.base_perception = base_perception
        self._frame = 0
        self._next_id = -1
        self._anchors = {}
        self.last_track_history_stats = {}

    def __getattr__(self, name):
        # Guard: during deepcopy/unpickling the delegate is not set yet, and an
        # unguarded lookup recursed forever (2026-10-03, trigger counterfactuals).
        delegate = self.__dict__.get("base_perception")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def snapshot(self, env):
        snap = self.base_perception.snapshot(env)
        self._frame += 1
        frame, ttl = self._frame, max(0, int(cfg.TRACK_MAX_MISSES))
        current_ids = {int(view.id) for view in snap.tracks}
        additions, details = [], []
        retained = {}
        for source_id, anchors in self._anchors.items():
            live = [a for a in anchors if 0 < frame-a.frame <= ttl]
            if live:
                retained[source_id] = live
            for anchor in live:
                age = frame-anchor.frame
                # Synthetic IDs stay attached to their anchor, never to an age
                # slot that could accidentally identify a different hypothesis.
                if anchor.synthetic_id in current_ids:
                    raise ValueError("Base perception reused a reserved history-track ID")
                position = anchor.position + age*float(cfg.UPDATE_RATE)*anchor.velocity
                additions.append(cc.TrackView(anchor.synthetic_id, position.copy(),
                                               anchor.velocity.copy(), anchor.heading, None))
                details.append({"id": anchor.synthetic_id, "source_id": source_id,
                                "anchor_frame": anchor.frame, "age_decisions": age})
        self._anchors = retained

        raw_by_id = {int(t.id): t for t in getattr(getattr(env, "tracker", None), "tracks", ())}
        stale = bool(getattr(env, "pose_stale", False))
        admitted, reasons, seen = [], Counter(), set()
        for view in snap.tracks:
            source_id = int(view.id)
            if source_id < 0 or source_id in seen:
                reasons["synthetic_or_duplicate_view"] += 1
                continue
            seen.add(source_id)
            reason = "pose_stale" if stale else _admission_reason(raw_by_id.get(source_id), view)
            if reason is not None:
                reasons[reason] += 1
                continue
            if ttl == 0:
                reasons["history_disabled_by_zero_age_limit"] += 1
                continue
            while self._next_id in current_ids:
                self._next_id -= 1
            anchor = _Anchor(source_id, self._next_id, frame,
                             np.asarray(view.position, dtype=float).copy(),
                             np.asarray(view.velocity, dtype=float).copy(), float(view.heading))
            self._next_id -= 1
            self._anchors.setdefault(source_id, []).append(anchor)
            self._anchors[source_id] = self._anchors[source_id][-ttl:]
            admitted.append(source_id)

        self.last_track_history_stats = {
            "enabled": True, "frame": frame, "max_age_decisions": ttl,
            "pose_stale": stale, "current_track_count": len(snap.tracks),
            "added_hypotheses": len(additions), "hypotheses": details,
            "admitted_source_ids": admitted, "admission_rejections": dict(reasons),
            "stored_anchor_count": sum(len(a) for a in self._anchors.values()),
            "static_points_preserved": len(snap.points),
        }
        return replace(snap, tracks=list(snap.tracks) + additions)
