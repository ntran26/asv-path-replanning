"""Boundary branch: a virtual range scan against the known channel polygon.

Why the boundary is not in the LiDAR channel (01 §3.1)
-----------------------------------------------------
The physical LiDAR sits higher than the test-basin wall and cannot be moved, so
the field scan contains returns from *beyond* the wall -- equipment, railings,
people walking past at scan height.  The tracker estimates velocity from
clustered returns, so a person walking past becomes a phantom target ship with
entirely plausible kinematics.  Geometric gating is a prerequisite for the
tracker, not a convenience.

So the channel boundary comes from the map instead: ray-cast against a known
polygon from the *estimated* pose, at fixed body-frame bearings, normalised to
closeness by exactly the same function `c_t` uses.

Two things this module must get right
-------------------------------------
1. **Pose noise.**  In the field this scan is computed from map + estimated
   pose, so it inherits localisation error.  A noiseless boundary scan in
   training would open a second sim-to-real gap in the very place this design
   was meant to close one (01 §3.3).  The hook is here and always on the path;
   the magnitude is `TODO(05)` and currently 0.0.
2. **Identical gating both sides.**  `gate_beams` is applied to simulated and
   field scans alike, so the two pipelines cannot diverge (01 §3.4).

What this module does NOT do: the true collision boundary is still enforced
geometrically for termination and penalties, exactly as in Paper 2.  What the
policy sees and what counts as a collision stay separate.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

import constants as cfg
from lidar_pooling import closeness_from_ranges

Point = Tuple[float, float]
Polygon = Sequence[Point]


# ---------------------------------------------------------------------------
# Pose noise
# ---------------------------------------------------------------------------
class PoseNoise:
    """Localisation error injected into the boundary raycast.

    Holds a slowly-drifting offset plus per-step jitter, because rf2o error is
    dominated by accumulated drift rather than by independent noise -- a fresh
    draw each step would be far easier to average out than the real thing.

    All three magnitudes are `TODO(05)` and default to 0.0, i.e. disabled.  The
    hook exists so no consumer has to change when 05 lands its numbers.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None, *,
                 sigma_xy: float = cfg.BOUNDARY_POSE_NOISE_XY,
                 sigma_heading_deg: float = cfg.BOUNDARY_POSE_NOISE_HEADING_DEG,
                 walk: float = cfg.BOUNDARY_POSE_NOISE_WALK) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sigma_xy = float(sigma_xy)
        self.sigma_heading_deg = float(sigma_heading_deg)
        self.walk = float(walk)
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.sigma_xy > 0.0 or self.sigma_heading_deg > 0.0 or self.walk > 0.0

    def reset(self) -> None:
        self._drift = np.zeros(2, dtype=np.float64)

    def perturb(self, x: float, y: float, heading_deg: float) -> Tuple[float, float, float]:
        """Return the pose the estimator would have reported."""
        if not self.enabled:
            return float(x), float(y), float(heading_deg)

        if self.walk > 0.0:
            self._drift += self.rng.normal(0.0, self.walk, size=2)
        jitter = self.rng.normal(0.0, self.sigma_xy, size=2) if self.sigma_xy > 0.0 else 0.0
        dh = self.rng.normal(0.0, self.sigma_heading_deg) if self.sigma_heading_deg > 0.0 else 0.0

        return (
            float(x + self._drift[0] + (jitter[0] if self.sigma_xy > 0.0 else 0.0)),
            float(y + self._drift[1] + (jitter[1] if self.sigma_xy > 0.0 else 0.0)),
            float(heading_deg + dh),
        )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _segments(polygon: Polygon, closed: bool = True):
    n = len(polygon)
    last = n if closed else n - 1
    for i in range(last):
        yield polygon[i], polygon[(i + 1) % n]


def raycast_polygon(origin: Point, bearing_deg: float, polygon: Polygon,
                    max_range: float = cfg.BOUNDARY_MAX_RANGE) -> float:
    """Distance from `origin` along `bearing_deg` to the first polygon edge.

    Bearings are compass-style, matching the rest of the codebase: 0 deg is +y
    and angles increase clockwise.  Returns `max_range` if nothing is hit.
    """
    ox, oy = float(origin[0]), float(origin[1])
    a = np.radians(float(bearing_deg))
    dx, dy = np.sin(a), np.cos(a)

    best = float(max_range)
    for (x0, y0), (x1, y1) in _segments(polygon):
        ex, ey = x1 - x0, y1 - y0
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue                                   # parallel
        wx, wy = x0 - ox, y0 - oy
        t = (wx * ey - wy * ex) / denom                 # along the ray
        s = (wx * dy - wy * dx) / denom                 # along the edge
        if 0.0 <= t < best and 0.0 <= s <= 1.0:
            best = float(t)
    return best


def boundary_scan(x: float, y: float, heading_deg: float, polygon: Polygon, *,
                  bearings_deg: Sequence[float] = cfg.BOUNDARY_BEARINGS_DEG,
                  max_range: float = cfg.BOUNDARY_MAX_RANGE,
                  pose_noise: Optional[PoseNoise] = None) -> np.ndarray:
    """The `boundary` observation branch: 7 rays, normalised to closeness.

    Normalisation is `lidar_pooling.closeness_from_ranges`, the identical
    function `c_t` uses, so the two branches cannot drift apart (01 §3.2).
    """
    if pose_noise is not None:
        x, y, heading_deg = pose_noise.perturb(x, y, heading_deg)

    ranges = np.array(
        [raycast_polygon((x, y), heading_deg + b, polygon, max_range) for b in bearings_deg],
        dtype=np.float32,
    )
    return closeness_from_ranges(ranges, max_range)


def boundary_ranges(x: float, y: float, heading_deg: float, polygon: Polygon, *,
                    bearings_deg: Sequence[float] = cfg.BOUNDARY_BEARINGS_DEG,
                    max_range: float = cfg.BOUNDARY_MAX_RANGE) -> np.ndarray:
    """Raw metric ranges, for tests, rendering and diagnostics."""
    return np.array(
        [raycast_polygon((x, y), heading_deg + b, polygon, max_range) for b in bearings_deg],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Field-side gating (01 §3.4)
# ---------------------------------------------------------------------------
def points_in_polygon(px, py, polygon: Polygon) -> np.ndarray:
    """Vectorised ray-crossing test over many points at once.

    Loops over the polygon's edges (a handful) rather than over the points
    (720 per scan), which is what keeps `gate_beams` off the critical path.
    """
    px = np.atleast_1d(np.asarray(px, dtype=np.float64))
    py = np.atleast_1d(np.asarray(py, dtype=np.float64))
    poly = np.asarray(polygon, dtype=np.float64)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)

    inside = np.zeros(px.shape, dtype=bool)
    for i in range(len(poly)):
        straddles = (y0[i] > py) != (y1[i] > py)
        dy = y1[i] - y0[i]
        x_cross = (x1[i] - x0[i]) * (py - y0[i]) / (dy if abs(dy) > 1e-300 else 1e-300) + x0[i]
        inside ^= straddles & (px < x_cross)
    return inside


def point_in_polygon(px: float, py: float, polygon: Polygon) -> bool:
    """Scalar convenience wrapper over `points_in_polygon`."""
    return bool(points_in_polygon(px, py, polygon)[0])


def gate_beams(ranges, bearings_deg, x: float, y: float, heading_deg: float,
               polygon: Polygon, *, margin: float = cfg.BOUNDARY_GATE_MARGIN,
               max_range: float = cfg.LIDAR_RANGE) -> np.ndarray:
    """Discard returns whose endpoint falls outside the boundary polygon.

    Applied **identically** in simulation and in the field so the two pipelines
    are equivalent.  In simulation the raycast never sees the border in the
    first place, so this is a no-op there and a real filter in the field --
    which is exactly the property that makes the two comparable.

    The margin is drawn from localisation uncertainty.  Too tight gates out real
    obstacles near the wall; too loose lets beyond-wall clutter through.  Both
    failure modes are worth measuring and reporting: the gate is a load-bearing
    component of the perception stack, not a tidying step.
    """
    r = np.asarray(ranges, dtype=np.float64).copy()
    b = np.asarray(bearings_deg, dtype=np.float64)
    a = np.radians(float(heading_deg) + b)
    ex = float(x) + r * np.sin(a)
    ey = float(y) + r * np.cos(a)

    # A beam at max range never hit anything, so it has nothing to gate.  Only
    # the beams that actually returned need testing, which is usually well
    # under half of them.
    hit = r < float(max_range) - 1e-6
    if not np.any(hit):
        return r

    keep = np.ones(r.shape, dtype=bool)
    px, py = ex[hit], ey[hit]
    inside = points_in_polygon(px, py, polygon)
    if float(margin) > 0.0:
        near = points_boundary_distance(px, py, polygon) <= float(margin)
        inside = inside | near
    keep[hit] = inside
    return np.where(keep, r, float(max_range))


def points_boundary_distance(px, py, polygon: Polygon) -> np.ndarray:
    """Shortest distance from each point to the polygon's boundary.

    Used instead of inflating the polygon.  Inflation about the centroid gives
    an *anisotropic* margin -- a wall far from the centroid along one axis moves
    mostly along that axis, so the effective margin varies around the outline.
    The gate margin comes from localisation uncertainty, which is isotropic, so
    a true perpendicular distance is the right test.  It is also correct for
    non-convex outlines, which the channel generator in 03 will produce.
    """
    p = np.stack([np.atleast_1d(np.asarray(px, dtype=np.float64)),
                  np.atleast_1d(np.asarray(py, dtype=np.float64))], axis=-1)
    poly = np.asarray(polygon, dtype=np.float64)
    start = poly
    seg = np.roll(poly, -1, axis=0) - poly
    len_sq = np.einsum("ij,ij->i", seg, seg)

    offset = p[:, None, :] - start[None, :, :]                 # (N, M, 2)
    t = np.einsum("nmj,mj->nm", offset, seg) / np.where(len_sq < 1e-18, 1.0, len_sq)
    t = np.clip(t, 0.0, 1.0)
    closest = start[None, :, :] + t[..., None] * seg[None, :, :]
    return np.linalg.norm(p[:, None, :] - closest, axis=-1).min(axis=1)


def boundary_distance(px: float, py: float, polygon: Polygon) -> float:
    """Scalar convenience wrapper over `points_boundary_distance`."""
    return float(points_boundary_distance(px, py, polygon)[0])


def rectangle(width: float, height: float, x0: float = 0.0, y0: float = 0.0) -> Polygon:
    """The Paper 2 basin outline, as a boundary polygon."""
    return [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)]
