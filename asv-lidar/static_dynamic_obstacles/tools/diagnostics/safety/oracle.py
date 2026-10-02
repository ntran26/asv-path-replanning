"""Diagnostics-only truth adapter for the v2/v3 safety filters.

Call ``install_oracle(env._safety_v2)`` after constructing the filter following
each reset.  This changes only that instance.  The policy still observes its
normal sensors; the filter receives true pose, body velocity and target states,
without its ego EMA lag.  Static clearance uses complete convex polygons, not
sampled vertices or scan returns.  Inflated hull size, clearance gaps, target
constant-velocity prediction, identified dynamics and terminal rules are kept.

``exact_geometry=False`` isolates state/target estimation while retaining the
ordinary perception object's remembered scan points.  Neither mode supplies
future target motion or simulator dynamics/actuator state to the predictor.
"""
from __future__ import annotations

import math
from types import MethodType

import numpy as np

import safety_v2 as v2
from classical import common as cc


def true_snapshot(env) -> cc.Snapshot:
    """Truth at the current decision; reads no sensors and consumes no RNG."""
    x, y, heading_deg = float(env.asv_x), float(env.asv_y), float(env.asv_h)
    state = env.path.project(x, y, heading_deg)
    tangent = np.asarray(env.path.tangent(state.closest_idx), dtype=float)
    right = np.array([tangent[1], -tangent[0]])
    centre = np.asarray(state.target, dtype=float)
    position = np.array([x, y])
    tracks = [cc.TrackView(i, np.array([t.x, t.y], dtype=float),
                           np.asarray(t.velocity, dtype=float).copy(),
                           math.radians(t.heading_deg))
              for i, t in enumerate(env.targets)]
    boundary = np.asarray(env.boundary_polygon, dtype=float).copy()
    snap = cc.Snapshot(x, y, math.radians(heading_deg), float(env.u_body),
                       float(env.v_body), math.radians(env.asv_w), tangent, right,
                       centre, math.atan2(tangent[0], tangent[1]),
                       float((position - centre) @ right),
                       float((env.path.points[-1] - position) @ tangent),
                       np.empty((0, 2)), tracks, boundary,
                       np.roll(boundary, -1, axis=0))
    snap.static_polygons = tuple(np.asarray(p, dtype=float).copy() for p in env.obstacles)
    return snap


def _vertex_edge_distances(points, starts, ends):
    """Minimum vertex-to-segment distance with broadcastable batch axes."""
    segments = ends - starts
    offset = points[..., :, None, :] - starts[..., None, :, :]
    length2 = np.sum(segments * segments, axis=-1)
    along = np.clip(np.sum(offset * segments[..., None, :, :], axis=-1)
                    / np.maximum(length2[..., None, :], 1e-18), 0.0, 1.0)
    delta = offset - along[..., None] * segments[..., None, :, :]
    return np.sqrt(np.min(np.sum(delta * delta, axis=-1), axis=(-2, -1)))


def polygon_clearance(positions, headings, polygons) -> np.ndarray:
    """Inflated rectangular hull to complete convex obstacles, shape ``(K,n)``.

    Disjoint polygons have their exact Euclidean gap; touching returns zero.
    Overlap has negative SAT penetration depth.  Both polygon winding orders
    work.  The environment's obstacle collision test also assumes convexity.
    """
    positions = np.asarray(positions, dtype=float)
    headings = np.asarray(headings, dtype=float)
    forward = np.stack((np.sin(headings), np.cos(headings)), axis=-1)
    right = np.stack((np.cos(headings), -np.sin(headings)), axis=-1)
    corners = (positions[..., None, :]
               + np.array([1, 1, -1, -1])[:, None] * cc.HALF_L * forward[..., None, :]
               + np.array([1, -1, -1, 1])[:, None] * cc.HALF_W * right[..., None, :])
    corner_ends = np.roll(corners, -1, axis=-2)
    result = np.full(headings.shape, np.inf)
    for points in polygons:
        poly = np.asarray(points, dtype=float)
        if len(poly) < 3:
            raise ValueError("oracle obstacles must be convex polygons with at least three vertices")
        ends = np.roll(poly, -1, axis=0)
        edges = ends - poly
        lengths = np.linalg.norm(edges, axis=-1)
        axes = np.stack((-edges[:, 1], edges[:, 0]), axis=-1)
        axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
        # SAT gaps on obstacle normals and the two own-hull normals.
        own_proj = corners @ axes.T
        obs_proj = poly @ axes.T
        sep = np.maximum(obs_proj.min(axis=0) - own_proj.max(axis=-2),
                         own_proj.min(axis=-2) - obs_proj.max(axis=0)).max(axis=-1)
        rel = poly - positions[..., None, :]
        for axis, radius in ((forward, cc.HALF_L), (right, cc.HALF_W)):
            projection = np.sum(rel * axis[..., None, :], axis=-1)
            sep = np.maximum(sep, np.maximum(projection.min(axis=-1) - radius,
                                             -radius - projection.max(axis=-1)))
        distance = np.minimum(_vertex_edge_distances(corners, poly, ends),
                              _vertex_edge_distances(poly, corners, corner_ends))
        result = np.minimum(result, np.where(sep > 0.0, distance, sep))
    return result


def _centre_polygon_distance(position, poly):
    poly = np.asarray(poly, dtype=float)
    ends = np.roll(poly, -1, axis=0)
    edges, rel = ends - poly, position - poly
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    if np.all(cross >= -1e-12) or np.all(cross <= 1e-12):
        return 0.0
    return float(_vertex_edge_distances(position[None, :], poly, ends))


def _oracle_threat(self, snap):
    if self._oracle_original_threat(snap):
        return True
    forward = np.array([math.sin(snap.heading), math.cos(snap.heading)])
    for poly in snap.static_polygons:
        # Whole edges count, including a panel whose distant vertices are all
        # outside ENGAGE_RANGE_M.  Preserve the ordinary aft engagement gate.
        if (np.max((poly - snap.position) @ forward) > -1.0
                and _centre_polygon_distance(snap.position, poly) < v2.ENGAGE_RANGE_M):
            return True
    return False


def _oracle_evaluate(self, snap, ro):
    """v2's evaluator, replacing only static scan-point geometry."""
    clearance = polygon_clearance(ro.positions, ro.headings, snap.static_polygons) - v2.GAP_STATIC_M
    clearance = np.minimum(clearance, cc.boundary_clearance(
        ro.positions, ro.headings, snap.edges_a, snap.edges_b) - v2.GAP_BOUNDARY_M)
    for track in snap.tracks:
        clearance = np.minimum(clearance, cc.target_gap(
            ro.positions, ro.headings, ro.times, track) - v2.GAP_TARGET_M)
    bad = clearance < 0.0
    first = np.where(bad.any(axis=0), ro.times[np.argmax(bad, axis=0)], np.inf)
    minimum = clearance.min(axis=0)
    moving = ro.speeds[-1] > v2.TERMINAL_STOP_SPEED
    if moving.any():
        run = np.minimum(v2.TERMINAL_M, v2.TERMINAL_S * ro.speeds[-1, moving])
        distance = np.linspace(1 / 8, 1.0, 8)[:, None, None] * run[None, :, None]
        heading = ro.headings[-1, moving]
        extended = (ro.positions[-1, moving][None]
                    + distance * np.stack((np.sin(heading), np.cos(heading)), axis=1)[None])
        extended_heading = np.broadcast_to(heading, extended.shape[:2])
        terminal = polygon_clearance(extended, extended_heading, snap.static_polygons) - v2.GAP_STATIC_M
        if v2.TERMINAL_BOUNDARY:
            terminal = np.minimum(terminal, cc.boundary_clearance(
                extended, extended_heading, snap.edges_a, snap.edges_b) - v2.GAP_BOUNDARY_M)
        terminal_min = np.full(len(minimum), np.inf)
        terminal_min[moving] = terminal.min(axis=0)
        first = np.where(np.isinf(first) & (terminal_min < 0.0), ro.times[-1], first)
        minimum = np.minimum(minimum, terminal_min)
    return first, minimum


class _TruthPerception:
    def __init__(self, owner, original, exact_geometry):
        self.owner, self.original, self.exact_geometry = owner, original, exact_geometry

    def snapshot(self, env):
        snap = true_snapshot(env)
        if not self.exact_geometry:
            snap.points = self.original.snapshot(env).points
        # v2/v3 run their EMA immediately after this call.  Starting it at the
        # current true sample avoids turning the oracle into a lagged observer.
        self.owner.ego = np.array([snap.u, snap.v, snap.r])
        observer = getattr(self.owner, "observer", None)
        if observer is not None:
            # Oracle truth must bypass a model-based prior as well as the EMA.
            observer.estimate = self.owner.ego.copy()
            observer.prior = self.owner.ego.copy()
        return snap


def install_oracle(safety_filter, *, exact_geometry=True):
    """Install truth on one freshly constructed v2/v3 instance and return it.

    Install once per episode, after reset.  Repeated installation is rejected
    so a state-only diagnostic cannot silently retain full-oracle methods.
    """
    if isinstance(safety_filter.perception, _TruthPerception):
        raise ValueError("oracle is already installed on this filter")
    safety_filter.perception = _TruthPerception(safety_filter, safety_filter.perception, exact_geometry)
    if exact_geometry:
        safety_filter._oracle_original_threat = safety_filter._threat_in_reach
        safety_filter._threat_in_reach = MethodType(_oracle_threat, safety_filter)
        safety_filter._evaluate = MethodType(_oracle_evaluate, safety_filter)
    return safety_filter
