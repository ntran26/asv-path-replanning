"""Reference path geometry: construction, arc length, and vessel-relative errors."""

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
    cross_track_error: float      # signed; positive = left of the path
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
        sign = 1.0 if cross_z > 0.0 else (-1.0 if cross_z < 0.0 else 0.0)
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
