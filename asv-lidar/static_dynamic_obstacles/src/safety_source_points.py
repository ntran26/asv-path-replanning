"""Experimental transfer of source-owned current returns to a moving view.

A stationary occupancy assumption should not also freeze returns assigned to an
observed moving object: Nuss et al., A Random Finite Set Approach for Dynamic
Occupancy Grid Maps (2016), https://arxiv.org/abs/1605.02406. Explicit measurement
association and extended-object shape modelling are reviewed by Granstrom,
Baum & Reuter (2016), https://arxiv.org/abs/1604.00970.

This small deterministic adapter is NOT either paper's probabilistic filter.
It reuses the existing persistence adapter's full-extent, endpoint-translation
and residual gates, then transfers only exactly equal current measurement rows.
It introduces no grid, distance radius or numerical matching tolerance. Old
memory rows, uncertain ownership, stale/coasted sources and unrelated returns
remain static. The moving hypothesis remains in the snapshot. No input memory,
tracker, policy observation or sensor is modified or queried again.

Mixed clusters can still contain an unrecognised static surface even after the
existing motion/shape gates pass. This is an empirical ownership hypothesis,
not a proof that every transferred point belongs to the moving vessel.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math

import numpy as np

import constants as cfg
from safety_provisional_tracks import _current_fit


def _points(value):
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None
    return points


def _keys(points):
    return {tuple(row) for row in points}


def _memory_sets(perception):
    """Keep provenance from the current batch distinct from older memory."""
    seen = set()
    while perception is not None and id(perception) not in seen:
        seen.add(id(perception))
        values = vars(perception)
        if "memory" in values and "frames" in values:
            current, old = set(), set()
            frame = values["frames"]
            for batch_frame, batch in values["memory"]:
                points = _points(batch)
                if points is None or batch_frame > frame:
                    return None
                (current if batch_frame == frame else old).update(_keys(points))
            return current, old
        perception = values.get("base_perception")
    return None


def _motion_update_ok(update, frame, minimum_observations):
    """Recheck the existing admission certificate's measured-motion gates."""
    if update.get("kind") not in ("admission", "measured_refresh"):
        return False
    frames = update.get("observation_frames", ())
    ends = np.asarray(update.get("endpoint_translation_m", ()), dtype=float)
    residual = update.get("max_translation_residual_m")
    try:
        return (len(frames) >= minimum_observations and len(frames) == update.get("observations")
                and len(set(frames)) == len(frames) and list(frames) == sorted(frames)
                and frames[-1] == frame and ends.shape == (2,) and np.isfinite(ends).all()
                and bool(np.all(ends > cfg.MOTION_PASS_TOL_M))
                and residual is not None and math.isfinite(residual)
                and 0. <= residual <= cfg.TRACK_FIT_FULL_EXTENT_TOL_M)
    except (TypeError, ValueError):
        return False


class SourcePointPerception:
    """Call base once, then transfer only freshly admitted, exact current rows.

``track_persistence`` is the existing inner adapter, never a new tracker. Missing
diagnostics fail closed. Duplicate sources, hypotheses, serials or coordinates
are rejected. Repeated observation serials never authorize another transfer.
    """

    def __init__(self, base_perception, track_persistence):
        self.base_perception = base_perception
        self.track_persistence = track_persistence
        self._frame = 0
        self._seen_serial = {}
        self.last_source_point_stats = {}

    def __getattr__(self, name):
        delegate = self.__dict__.get("base_perception")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def snapshot(self, env):
        snap = self.base_perception.snapshot(env)
        self._frame += 1
        points = _points(snap.points)
        reasons, transfers = Counter(), []
        removed = np.zeros(len(snap.points), dtype=bool)
        stale = bool(getattr(env, "pose_stale", False))
        tracker = getattr(env, "tracker", None)
        tracks = list(getattr(tracker, "tracks", ()))
        scans = list(getattr(tracker, "_scans", ()))
        stats = getattr(self.track_persistence, "last_track_persistence_stats", {})
        memory = _memory_sets(self.base_perception)
        if stale:
            reasons["pose_stale"] += 1
        elif points is None or memory is None:
            reasons["missing_point_provenance"] += 1
        elif not scans or not stats or stats.get("pose_stale", True):
            reasons["missing_fresh_persistence"] += 1
        else:
            frame = stats.get("frame")
            serial = int(scans[-1].serial)
            ids = Counter(int(t.id) for t in tracks)
            view_ids = Counter(int(t.id) for t in snap.tracks)
            updates = stats.get("updates", ())
            hypotheses = stats.get("hypotheses", ())
            update_ids = Counter(u.get("source_id") for u in updates)
            hypothesis_ids = Counter(h.get("source_id") for h in hypotheses)
            current_memory, old_memory = memory
            point_keys = [tuple(p) for p in points]
            for track in tracks:
                source = int(track.id)
                if source < 0 or ids[source] != 1:
                    reasons["duplicate_or_invalid_raw_id"] += 1
                    continue
                history = getattr(track, "history", ())
                if not history or int(history[-1][0]) != serial or sum(int(s.serial) == serial for s in scans) != 1:
                    reasons["cluster_is_not_unique_current_scan"] += 1
                    continue
                if sum(int(s) == serial for s, _ in history) != 1 or self._seen_serial.get(source) == serial:
                    reasons["repeated_observation"] += 1
                    continue
                self._seen_serial[source] = serial
                if (not bool(getattr(track, "confirmed", False))
                        or int(getattr(track, "hits", 0)) < cfg.TRACK_MIN_HITS
                        or int(getattr(track, "misses", 0)) != 0):
                    reasons["not_fresh_confirmed"] += 1
                    continue
                evidence = getattr(track, "last_evidence", None)
                if (evidence is None or evidence.appear <= 0
                        or evidence.violations < cfg.MOTION_MIN_POINTS):
                    reasons["insufficient_current_motion_evidence"] += 1
                    continue
                fit, rejection = _current_fit(track)
                if fit is None:
                    reasons[rejection or "invalid_fit"] += 1
                    continue
                cluster = _points(history[-1][1])
                cluster_keys = _keys(cluster)
                heading = fit[1]
                forward = np.array([math.sin(heading), math.cos(heading)])
                if np.ptp(cluster @ forward) < cfg.LOA - cfg.TRACK_FIT_FULL_EXTENT_TOL_M:
                    reasons["current_extent_partial"] += 1
                    continue
                if len(cluster_keys) != len(cluster):
                    reasons["duplicate_cluster_points"] += 1
                    continue
                if update_ids[source] != 1 or hypothesis_ids[source] != 1:
                    reasons["missing_unique_motion_update"] += 1
                    continue
                update = next(u for u in updates if u.get("source_id") == source)
                hypothesis = next(h for h in hypotheses if h.get("source_id") == source)
                if not _motion_update_ok(update, frame, getattr(self.track_persistence, "min_motion_observations", 3)):
                    reasons["invalid_motion_update"] += 1
                    continue
                target_id = hypothesis.get("id")
                if (not isinstance(target_id, int) or target_id >= 0 or view_ids[target_id] != 1
                        or hypothesis.get("age_s") != 0. or hypothesis.get("anchor_frame") != frame
                        or update.get("synthetic_id") != target_id):
                    reasons["no_unique_fresh_published_target"] += 1
                    continue
                view = next(t for t in snap.tracks if int(t.id) == target_id)
                position, velocity = np.asarray(view.position), np.asarray(view.velocity)
                if (position.shape != (2,) or velocity.shape != (2,) or not np.isfinite(position).all()
                        or not np.isfinite(velocity).all() or not np.isfinite(view.heading)
                        or not np.array_equal(position, np.asarray(hypothesis.get("position")))
                        or not np.array_equal(velocity, np.asarray(hypothesis.get("velocity")))):
                    reasons["inconsistent_published_target"] += 1
                    continue
                view_forward = np.array([math.sin(view.heading), math.cos(view.heading)])
                axes = np.column_stack((view_forward, [view_forward[1], -view_forward[0]]))
                extent = np.abs((cluster-position) @ axes).max(axis=0)
                if np.any(extent > np.array([cfg.LOA, cfg.BREADTH])/2 + cfg.TRACK_FIT_FULL_EXTENT_TOL_M):
                    reasons["cluster_outside_published_hull"] += 1
                    continue
                ambiguous = False
                for other in tracks:
                    if other is track:
                        continue
                    other_history = getattr(other, "history", ())
                    if not other_history:
                        continue
                    other_points = _points(other_history[-1][1])
                    if other_points is None or cluster_keys & _keys(other_points):
                        ambiguous = True
                        break
                if ambiguous:
                    reasons["overlapping_or_unknown_raw_source"] += 1
                    continue
                transferable = cluster_keys & current_memory - old_memory
                mask = np.array([key in transferable for key in point_keys], dtype=bool)
                removed |= mask
                if mask.any():
                    transfers.append({"source_id": source, "target_id": target_id, "scan_serial": serial,
                                      "cluster_points": len(cluster), "removed_points": int(mask.sum()),
                                      "remembered_equal_points_retained": sum(k in cluster_keys and k in old_memory for k in point_keys)})
        self.last_source_point_stats = {
            "frame": self._frame, "enabled": True, "pose_stale": stale,
            "coordinate_match": "exact_float64", "static_points_before": len(snap.points),
            "static_points_after": int(len(snap.points)-removed.sum()),
            "removed_point_count": int(removed.sum()), "source_ids_transferred": [t["source_id"] for t in transfers],
            "transfers": transfers, "rejections": dict(reasons), "underlying_memory_unchanged": True,
        }
        return replace(snap, points=points[~removed]) if removed.any() else snap
