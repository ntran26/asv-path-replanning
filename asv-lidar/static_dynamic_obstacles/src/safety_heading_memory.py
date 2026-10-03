"""Bounded hull-axis memory for an existing safety track with a missing fit.

The tracker clears its fitted hull heading when a fit is unavailable. Ordinary
perception then uses velocity course (or zero at low speed) as hull orientation.
A partly observed/stopping vessel can consequently rotate its predicted collision
rectangle without evidence that its hull rotated. This adapter holds a recent,
validated hull axis independently of translational velocity for the SAME raw ID.

Related modelling inspiration: Granstrom, Baum & Reuter, "Extended Object
Tracking: Introduction, Overview and Applications", Sec. II (kinematic state,
extent and spatial measurements), https://arxiv.org/abs/1604.00970 . The bounded
hold here is an engineering hypothesis, not that paper's probabilistic estimator
or an uncertainty bound. A genuinely turning but poorly observed hull can make
the held axis wrong. This does not solve target manoeuvre prediction or ID loss.

Only fresh, matched, confirmed observations with a new scan serial can supply
or use memory. Fits use the existing vessel-extent gates; age uses the existing
TRACK_MAX_MISSES decision budget. There is no association across IDs. Synthetic
persistent views and current V16 motion-axis corrections remain untouched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math

import numpy as np

import constants as cfg
from safety_provisional_tracks import _current_fit


@dataclass(frozen=True)
class _Heading:
    heading: float
    frame: int
    serial: int


class HeadingMemoryPerception:
    """Wrap one perception call without changing admission or point ownership.

    Construct a new instance at episode reset. Missing IDs, decreasing raw age/
    hit count and expired fits discard memory. The raw tracker normally assigns
    monotonic IDs; undetectable same-ID reuse remains its identity assumption.
    """

    def __init__(self, base_perception):
        self.base_perception = base_perception
        self._frame = 0
        self._headings = {}
        self._seen = {}
        self.last_heading_memory_stats = {}

    def __getattr__(self, name):
        delegate = self.__dict__.get("base_perception")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def snapshot(self, env):
        snap = self.base_perception.snapshot(env)
        self._frame += 1
        frame = self._frame
        stale = bool(getattr(env, "pose_stale", False))
        raw_tracks = [t for t in getattr(getattr(env, "tracker", None), "tracks", ())
                      if int(t.id) >= 0]
        raw = {int(t.id): t for t in raw_tracks}
        base_ids = [int(view.id) for view in snap.tracks if int(view.id) >= 0]
        if len(raw) != len(raw_tracks) or len(base_ids) != len(set(base_ids)):
            raise ValueError("Duplicate source ID in heading-memory perception")
        expired = sorted(source for source, cached in self._headings.items()
                         if source not in raw or frame-cached.frame > int(cfg.TRACK_MAX_MISSES))
        self._headings = {source: cached for source, cached in self._headings.items()
                          if source not in expired}
        self._seen = {source: seen for source, seen in self._seen.items() if source in raw}
        corrected_now = set(getattr(self.base_perception, "last_motion_axis_stats", {})
                            .get("replaced_source_ids", ()))
        represented = set(base_ids)
        replacements, updates, reasons, resets = {}, [], Counter(), []
        for source, track in raw.items():
            age, hits = int(getattr(track, "age", 0)), int(getattr(track, "hits", 0))
            seen = self._seen.get(source)
            if seen is not None and (age < seen[1] or hits < seen[2]):
                self._headings.pop(source, None)
                self._seen.pop(source, None)
                seen = None
                resets.append(source)
            if stale or int(getattr(track, "misses", 0)):
                reasons["pose_stale" if stale else "missed_detection"] += 1
                continue
            history = getattr(track, "history", ())
            if not history:
                reasons["missing_cluster"] += 1
                continue
            serial, points = history[-1]
            serial, points = int(serial), np.asarray(points, dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                    or not np.isfinite(points).all()):
                reasons["invalid_cluster"] += 1
                continue
            if seen is not None and serial == seen[0]:
                reasons["repeated_observation"] += 1
                continue
            self._seen[source] = (serial, age, hits)
            if hits < int(cfg.TRACK_MIN_HITS):
                reasons["unconfirmed"] += 1
                continue
            fit, reason = _current_fit(track)
            if fit is not None:
                self._headings[source] = _Heading(float(fit[1]), frame, serial)
                updates.append(source)
                continue
            axis = getattr(track, "last_fit_heading_deg", None)
            if axis is not None and math.isfinite(float(axis)):
                # A finite but rejected fit is contradictory evidence, not a
                # missing fit. Discard the prior rather than resurrect it if
                # the next observation has no fit.
                self._headings.pop(source, None)
                reasons[reason or "rejected_finite_fit"] += 1
                continue
            cached = self._headings.get(source)
            if source in corrected_now:
                reasons["current_motion_axis_geometry"] += 1
            elif source in represented and cached is not None:
                replacements[source] = cached
            else:
                reasons["no_recent_same_source_fit"] += 1

        details = [dict(source_id=source, heading_rad=cached.heading,
                        fit_age_steps=frame-cached.frame, fit_serial=cached.serial)
                   for source, cached in sorted(replacements.items())]
        self.last_heading_memory_stats = {
            "frame": frame, "pose_stale": stale,
            "max_age_steps": int(cfg.TRACK_MAX_MISSES),
            "updated_source_ids": updates, "replaced_source_ids": sorted(replacements),
            "replacements": details, "expired_source_ids": expired,
            "identity_reset_source_ids": resets, "rejections": dict(reasons),
            "stored_heading_count": len(self._headings), "static_points_preserved": len(snap.points),
        }
        if not replacements:
            return snap
        return replace(snap, tracks=[replace(view, heading=replacements[int(view.id)].heading)
                                     if int(view.id) in replacements else view for view in snap.tracks])
