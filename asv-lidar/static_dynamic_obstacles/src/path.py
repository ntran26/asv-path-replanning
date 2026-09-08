"""Reference path geometry: construction, arc length, and vessel-relative errors.

**Cross-track error is positive to STARBOARD** (02b C4), the textbook LOS
convention.  This differs from Paper 2, which used positive-to-port; see
`project()` for why it was flipped.  Anything comparing numbers across the two
papers must account for it.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

import constants as cfg


def wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def bearing_deg(from_xy, to_xy) -> float:
    """Compass-style bearing (0 deg = +y, clockwise positive)."""
    return math.degrees(math.atan2(float(to_xy[0] - from_xy[0]), float(to_xy[1] - from_xy[1])))


class PathState(NamedTuple):
    """Where the vessel sits relative to the reference path."""
    closest_idx: int
    cross_track_error: float      # signed; positive = STARBOARD of the path
    course_error: float           # deg
    target: tuple                 # closest point on the path
    lookahead_idx: int
    lookahead: tuple
    lookahead_course_error: float  # deg


class ReferencePath:
    """A polyline reference path with cached arc length."""

    def __init__(self, points, lookahead_fraction: float = cfg.LOOKAHEAD_FRACTION) -> None:
        self.points = np.asarray(points, dtype=np.float32)
        if self.points.ndim != 2 or len(self.points) < 2:
            raise ValueError("a reference path needs at least two (x, y) points")

        seg_len = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.s = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)
        self.length = float(self.s[-1])
        self.lookahead_distance = max(2.0, lookahead_fraction * self.length)

    def __len__(self) -> int:
        return len(self.points)

    def tangent(self, idx: int) -> np.ndarray:
        """Unit tangent at a path index, by central difference where possible."""
        idx = int(np.clip(idx, 0, len(self.points) - 1))
        if idx == 0:
            vec = self.points[1] - self.points[0]
        elif idx == len(self.points) - 1:
            vec = self.points[-1] - self.points[-2]
        else:
            vec = self.points[idx + 1] - self.points[idx - 1]

        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return np.array([0.0, 1.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def curvature(self, idx: int) -> float:
        """Signed path curvature at a vertex, 1/m.  **Positive = to starboard.**

        Menger curvature over the three vertices centred on `idx`, signed so it
        matches the vessel's yaw-rate convention (`r > 0` is a starboard turn).
        A straight path returns 0.

        The sign comes from the cross product of the two segment vectors: in
        this frame (+y north, +x east, headings clockwise) a starboard turn
        rotates the tangent from +y toward +x, which gives a *negative* cross
        product -- hence the negation.
        """
        n = len(self.points)
        if n < 3:
            return 0.0
        i = int(np.clip(idx, 1, n - 2))
        a, b, c = (self.points[i - 1].astype(np.float64),
                   self.points[i].astype(np.float64),
                   self.points[i + 1].astype(np.float64))
        ab, bc, ac = b - a, c - b, c - a
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        denom = float(np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ac))
        if denom < 1e-12:
            return 0.0
        kappa = -2.0 * cross / denom
        # Deadband: see `constants.CURVATURE_EPS`.  float32 vertex storage
        # leaves ~2e-5 1/m on a straight path, which would otherwise make
        # `r_path` non-zero everywhere.
        return 0.0 if abs(kappa) < cfg.CURVATURE_EPS else float(kappa)

    def yaw_rate_for_tracking(self, idx: int, speed: float) -> float:
        """Yaw rate required to follow the path at `speed`, rad/s.

        `r_path = u * kappa` (02b T3).  02a `R-8` measures every yaw-based
        COLREGs term as `r - r_path`, so that following a channel which bends to
        port is not scored as an evasive port turn -- which is what withdrew
        `R-3` and removed the unwinnable state it was patching.

        Returns rad/s to match `r_ref = 0.20 rad/s` in 02a §6.2.  Note the
        environment reports its own yaw rate in **degrees** per second; `info`
        emits both in rad/s so the difference cannot be taken in mixed units.
        """
        return float(speed) * self.curvature(idx)

    def left_normal(self, idx: int) -> np.ndarray:
        t = self.tangent(idx)
        return np.array([-t[1], t[0]], dtype=np.float32)

    def index_at_frac(self, frac: float) -> int:
        s = float(np.clip(frac, 0.0, 1.0)) * self.length
        return int(np.clip(np.searchsorted(self.s, s, side="left"), 0, len(self.points) - 1))

    def frame_at_frac(self, frac: float):
        """Return (point, tangent, left_normal) at a fraction of the arc length."""
        idx = self.index_at_frac(frac)
        return self.points[idx].astype(np.float32), self.tangent(idx), self.left_normal(idx)

    def project(self, x: float, y: float, course_deg: float) -> PathState:
        """Locate the vessel on the path and compute the tracking errors."""
        pos = np.array([x, y], dtype=np.float32)
        distances = np.linalg.norm(self.points - pos, axis=1)
        closest_idx = int(np.argmin(distances))

        tangent = self.tangent(closest_idx)
        offset = pos - self.points[closest_idx]
        cross_z = float(tangent[0] * offset[1] - tangent[1] * offset[0])
        # Positive to STARBOARD -- the textbook LOS convention (02b C4).
        #
        # Paper 2 used positive-to-port, which its own `verify_los_apf.py`
        # flagged as non-standard.  Flipped here because this paper's
        # contribution is COLREGs *geometry*: 02a §6.4's passing-side term has
        # two opposite branches keyed on the sign of the lateral offset
        # (head-on penalises one, overtaking the other), and carrying a
        # non-standard lateral sign through them invites exactly the class of
        # error the paper is about.
        #
        # `cross_z > 0` means the vessel lies to the LEFT of the path tangent,
        # hence the negation.
        sign = -1.0 if cross_z > 0.0 else (1.0 if cross_z < 0.0 else 0.0)
        cte = sign * float(distances[closest_idx])

        path_course = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))

        s_target = min(self.length, float(self.s[closest_idx]) + self.lookahead_distance)
        lookahead_idx = int(np.clip(np.searchsorted(self.s, s_target, side="left"), 0, len(self.points) - 1))
        lookahead = self.points[lookahead_idx]

        return PathState(
            closest_idx=closest_idx,
            cross_track_error=cte,
            course_error=wrap180(path_course - course_deg),
            target=(float(self.points[closest_idx][0]), float(self.points[closest_idx][1])),
            lookahead_idx=lookahead_idx,
            lookahead=(float(lookahead[0]), float(lookahead[1])),
            lookahead_course_error=wrap180(bearing_deg(pos, lookahead) - course_deg),
        )


def straight_points(start_x: float, start_y: float, goal_x: float, goal_y: float) -> np.ndarray:
    n = max(40, int(np.hypot(goal_x - start_x, goal_y - start_y) * 5.0))
    return np.column_stack((
        np.linspace(start_x, goal_x, n, dtype=np.float32),
        np.linspace(start_y, goal_y, n, dtype=np.float32),
    )).astype(np.float32)


def curved_points(start_x: float, start_y: float, goal_x: float, goal_y: float,
                  map_width: float, map_height: float) -> np.ndarray:
    """Quadratic Bezier with a randomly offset control point.

    Only reachable with path_mode "curve"/"mixed"; the published runs use
    straight reference paths.
    """
    start = np.array([start_x, start_y], dtype=np.float32)
    goal = np.array([goal_x, goal_y], dtype=np.float32)
    vec = goal - start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return straight_points(start_x, start_y, goal_x, goal_y)

    tangent = vec / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    offset = float(np.random.uniform(-0.18 * map_width, 0.18 * map_width))
    control = 0.5 * (start + goal) + offset * normal
    control[0] = float(np.clip(control[0], 1.5, map_width - 1.5))
    control[1] = float(np.clip(control[1], 1.5, map_height - 1.5))

    t = np.linspace(0.0, 1.0, max(60, int(length * 5.0)), dtype=np.float32)[:, None]
    pts = (1 - t) ** 2 * start[None, :] + 2 * (1 - t) * t * control[None, :] + t ** 2 * goal[None, :]
    return pts.astype(np.float32)
