"""Safety-only scan memory with conservative evidence of vacated space.

The ordinary perception snapshot and its long memory are retained.  A remembered
point is removed only when the current raw scan returned from beyond it, with
the tracker's radial/lateral pose tolerance and no nearby supporting return.
No-return rays, occlusions and the LiDAR dead zone provide no clearing evidence.
The shared classical comparator perception and tracker are not modified.
"""
from __future__ import annotations

import time

import numpy as np

import constants as cfg
import tracking
from classical import common as cc


class _WithoutFreshReturns:
    """Held-pose snapshot view: retain all state, suppress scan ingestion."""

    def __init__(self, env):
        self._env = env
        self.gated_ranges = np.full_like(np.asarray(env.lidar.ranges, dtype=float), cfg.LIDAR_RANGE)

    def __getattr__(self, name):
        # Guard: during deepcopy/unpickling the delegate is not set yet, and an
        # unguarded lookup recursed forever (2026-10-03, trigger counterfactuals).
        delegate = self.__dict__.get("_env")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)


class SafetyPerception(cc.Perception):
    """Drop-in safety perception; pass the desired ``memory_frames`` explicitly.

    Clearing mutates every retained memory batch, so a vacated target's old
    returns cannot reappear when its current track moves away.  Only measured
    free-space evidence removes old points: proximity to a current track still
    affects the output exactly as in ``cc.Perception``, without deleting memory.
    """

    # Limit explains()'s point-by-return temporary array on long memories.
    CLEAR_CHUNK_POINTS = 1024

    def __init__(self, memory_frames=cc.SCAN_MEMORY_FRAMES):
        super().__init__(memory_frames=memory_frames)
        self.total_cleared_points = 0
        self.last_memory_stats = {}

    def _clear_vacated(self, scan):
        cleared, retained = 0, []
        for frame, points in self.memory:
            keep = np.ones(len(points), dtype=bool)
            for start in range(0, len(points), self.CLEAR_CHUNK_POINTS):
                chunk = points[start:start + self.CLEAR_CHUNK_POINTS]
                passed = scan.passes_through(chunk, cfg.MOTION_PASS_TOL_M)
                indices = np.flatnonzero(passed)
                if len(indices):
                    supported = scan.explains(chunk[indices], cfg.MOTION_EXPLAIN_M)
                    keep[start + indices[~supported]] = False
            cleared += int((~keep).sum())
            if keep.any():
                retained.append((frame, points[keep]))
        self.memory = retained
        return cleared

    def snapshot(self, env):
        started = time.perf_counter()
        # Match the inherited snapshot's decision-based expiry before spending
        # work on geometry that expires during this decision anyway.
        self.memory = [(frame, points) for frame, points in self.memory
                       if self.frames + 1 - frame < self.memory_frames]
        before = sum(len(points) for _, points in self.memory)
        stale = bool(getattr(env, "pose_stale", False))
        cleared = 0
        if stale:
            # The environment likewise refuses to feed a fresh scan with a
            # held pose to its tracker.  Retain old memory and advance its age.
            snap = super().snapshot(_WithoutFreshReturns(env))
        else:
            if before:
                x, y, heading = env.estimated_pose()
                origin = tracking.sensor_origin(x, y, heading)
                raw = np.asarray(getattr(env, "raw_ranges", env.lidar.ranges), dtype=float)
                scan = tracking.ScanFrame.from_scan(raw, env.lidar.bearings, origin, heading)
                cleared = self._clear_vacated(scan)
            # Add the current measured static returns only after clearing old
            # memory; this prevents current obstacle evidence being discarded.
            snap = super().snapshot(env)
        self.total_cleared_points += cleared
        self.last_memory_stats = {
            "pose_stale": stale,
            "remembered_before": before,
            "cleared_points": cleared,
            "remembered_after": sum(len(points) for _, points in self.memory),
            "snapshot_points": len(snap.points),
            "seconds": time.perf_counter() - started,
        }
        return snap
