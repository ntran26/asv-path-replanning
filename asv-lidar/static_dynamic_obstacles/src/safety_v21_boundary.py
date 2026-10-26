"""Scored boundary limits for the V21 safety filter (refinement R3).

The filter checks the hull against `env.boundary_polygon`, the navigable polygon.
The environment scores a boundary contact differently (`env._border_clearance`):
against the corridor's lateral limits (the polygon's sides) and against the map's
ends at y = 0 and y = map height.  In the basin the polygon is inset 0.4 m on all four
sides (x 0.4-9.6 m, y 0.4-24.6 m), so near either end wall the filter sees its hull
outside the boundary 0.4 m before a contact is scored, finds no plan that passes at its
first sample, and issues the policy's command unchecked (planning_records/
SAFETY_V21_PLAN.md, section 3, mechanism 3).  Channel polygons already span the map's
full length, so they are unchanged.

`ScoredBoundaryPerception` wraps the filter's perception and moves every polygon vertex
on the polygon's lowest y to y = 0, and every vertex on its highest y to the map height,
when those extremes lie inside the map.  The sides are untouched: at a channel bend the
polygon stays stricter than the environment's lateral envelope.  Only onboard map
knowledge (the polygon and the map's extent) is used.
"""
from __future__ import annotations

import numpy as np

END_TOLERANCE_M = 1e-6


def scored_polygon(polygon: np.ndarray, map_height: float) -> np.ndarray:
    poly = np.array(polygon, dtype=float, copy=True)
    if poly.ndim != 2 or len(poly) < 3:
        return poly
    low, high = poly[:, 1].min(), poly[:, 1].max()
    if low > END_TOLERANCE_M:
        poly[np.abs(poly[:, 1] - low) <= END_TOLERANCE_M, 1] = 0.0
    if high < map_height - END_TOLERANCE_M:
        poly[np.abs(poly[:, 1] - high) <= END_TOLERANCE_M, 1] = float(map_height)
    return poly


class ScoredBoundaryPerception:
    """Perception wrapper: the snapshot's boundary is the scored one; all else is the inner snapshot."""

    def __init__(self, inner):
        self.inner = inner
        self.last_scored_boundary = {}

    def snapshot(self, env):
        snap = self.inner.snapshot(env)
        poly = scored_polygon(snap.edges_a, float(env.map_height))
        moved = int(np.sum(np.any(poly != np.asarray(snap.edges_a, dtype=float), axis=1)))
        snap.edges_a, snap.edges_b = poly, np.roll(poly, -1, axis=0)
        self.last_scored_boundary = {"vertices_moved": moved}
        return snap

    def __getattr__(self, name):
        return getattr(self.inner, name)
