"""Remembered static points inside a moving track's recent footprint (V21, refinement R4).

The safety memory keeps every static scan return for the whole episode (`MEMORY_S`),
excluding at ingestion only the returns within 1.8 m of a current track, and clears a
point only when a later ray passes beyond it.  Returns of the target stored before its
track was confirmed, or while the track estimate lagged, therefore persist; those
within the LiDAR's 1 m dead zone around the hull can never be cleared, which is where
they bind the filter's plans.  On the baseline-v3 validation set (2026-10-25), 2,315
near-hull point sightings at V16's active decisions lie on the target's true earlier
track, in 34 of 270 episodes (planning_records/SAFETY_V21_PLAN.md, mechanism 4).

`TrailExclusionPerception` records, at each decision, the footprint of every track
moving faster than `MIN_SPEED_MPS` (its estimated centre and hull heading, the known
hull dimensions inflated by `INFLATE_M`) for the last `HORIZON_S`, and leaves out of
the snapshot every remembered point inside one of those footprints.  A static obstacle
cannot occupy space a target's hull has occupied, so only the track estimate's error
can make this remove a real obstacle point; `INFLATE_M` is kept small for that reason.
Memory itself is unchanged, and only onboard estimates are used.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ship import VESSEL_LENGTH, VESSEL_WIDTH

HORIZON_S = 20.0
MIN_SPEED_MPS = 0.15        # the existing moving-track threshold of the safety perception
INFLATE_M = 0.10            # beyond the true hull, for the track's position and heading error
DECISION_S = 0.5


class TrailExclusionPerception:
    """Perception wrapper: drops snapshot points inside recent moving-track footprints."""

    def __init__(self, inner, horizon_s=HORIZON_S, inflate_m=INFLATE_M):
        self.inner = inner
        self.footprints = deque(maxlen=max(1, int(round(horizon_s / DECISION_S))))
        self.half_l = 0.5 * VESSEL_LENGTH + float(inflate_m)
        self.half_w = 0.5 * VESSEL_WIDTH + float(inflate_m)
        self.last_trail = {}

    def snapshot(self, env):
        snap = self.inner.snapshot(env)
        now = [(np.asarray(t.position, float), float(t.heading)) for t in snap.tracks
               if float(np.hypot(*np.asarray(t.velocity, float))) > MIN_SPEED_MPS]
        self.footprints.append(now)
        pts = np.asarray(snap.points, float).reshape(-1, 2)
        keep = np.ones(len(pts), dtype=bool)
        if len(pts):
            for decision in self.footprints:
                for centre, heading in decision:
                    rel = pts - centre
                    s, c = np.sin(heading), np.cos(heading)
                    lon = np.abs(rel[:, 0] * s + rel[:, 1] * c)
                    lat = np.abs(rel[:, 0] * c - rel[:, 1] * s)
                    keep &= ~((lon <= self.half_l) & (lat <= self.half_w))
        self.last_trail = {"removed_points": int((~keep).sum()),
                           "footprints": int(sum(len(d) for d in self.footprints))}
        snap.points = pts[keep]
        return snap

    def __getattr__(self, name):
        return getattr(self.inner, name)
