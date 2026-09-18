"""Static feasibility: is there a route through this layout at all?  (06, F74)

Paper 2 filtered its evaluation layouts with an A* search on an inflated grid,
and the Paper 3 environment never did: its obstacle sampler drew panels near the
path and the only guard kept them clear of the encounter's CPA. Nothing stopped
a gate narrower than the hull, or a panel against the wall a slanted leg closes
on, and a layout with no route teaches only that collision is unavoidable.

The check is kinematic, not a dynamics proof. It asks for a grid route from
start to goal that keeps the hull's half-breadth plus margin off every wall
and a little more off every panel, and no longer than
`FEASIBILITY_MAX_ROUTE_RATIO` times the leg (Paper 2's bounds). Layouts that
pass can still be hard -- a gate the vessel must thread, a panel on the side it
would recover to, a wall it approaches obliquely -- and that is the point:
challenge stays, impossibility does not.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

import boundary_raycast as br
import constants as cfg

Point = Tuple[float, float]
Polygon = Sequence[Point]


def free_grid(nav: Polygon, obstacles: Sequence[Polygon], *,
              res: float = None, wall: float = None, panel: float = None):
    """Boolean occupancy over `nav`'s bounding box: True where the hull fits."""
    res = cfg.FEASIBILITY_GRID_M if res is None else float(res)
    wall = cfg.FEASIBILITY_WALL_INFLATION_M if wall is None else float(wall)
    panel = cfg.FEASIBILITY_OBSTACLE_INFLATION_M if panel is None else float(panel)
    xs = [p[0] for p in nav]
    ys = [p[1] for p in nav]
    x0, y0 = min(xs), min(ys)
    nx = int(math.ceil((max(xs) - x0) / res)) + 1
    ny = int(math.ceil((max(ys) - y0) / res)) + 1
    gx, gy = np.meshgrid(x0 + res * np.arange(nx), y0 + res * np.arange(ny), indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    free = np.asarray(br.points_in_polygon(px, py, list(nav)), dtype=bool)
    idx = np.flatnonzero(free)
    if idx.size:
        clear = np.asarray(br.points_boundary_distance(px[idx], py[idx], list(nav))) >= wall
        free[idx] = clear
    for poly in obstacles:
        poly = list(poly)
        idx = np.flatnonzero(free)
        if not idx.size:
            break
        # Only cells near the panel can be blocked by it.
        pxs, pys = [p[0] for p in poly], [p[1] for p in poly]
        near = ((px[idx] >= min(pxs) - panel) & (px[idx] <= max(pxs) + panel)
                & (py[idx] >= min(pys) - panel) & (py[idx] <= max(pys) + panel))
        idx = idx[near]
        if not idx.size:
            continue
        inside = np.asarray(br.points_in_polygon(px[idx], py[idx], poly), dtype=bool)
        dist = np.asarray(br.points_boundary_distance(px[idx], py[idx], poly))
        free[idx[inside | (dist < panel)]] = False
    return free.reshape(nx, ny), (x0, y0, res)


def route_length(start: Point, goal: Point, nav: Polygon,
                 obstacles: Sequence[Polygon], **kw) -> Optional[float]:
    """A* route length in metres (8-connected), or None if there is none."""
    free, (x0, y0, res) = free_grid(nav, obstacles, **kw)
    nx, ny = free.shape

    def cell(p):
        return (int(round((p[0] - x0) / res)), int(round((p[1] - y0) / res)))

    s, g = cell(start), cell(goal)
    for c in (s, g):
        if not (0 <= c[0] < nx and 0 <= c[1] < ny) or not free[c]:
            return None
    moves = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
             if dx or dy]
    best = {s: 0.0}
    frontier = [(math.hypot(g[0] - s[0], g[1] - s[1]), 0.0, s)]
    while frontier:
        _, cost, c = heapq.heappop(frontier)
        if c == g:
            return cost * res
        if cost > best.get(c, math.inf):
            continue
        for dx, dy, step in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n]:
                continue
            # No corner-cutting between two blocked cells.
            if dx and dy and not (free[c[0] + dx, c[1]] and free[c[0], c[1] + dy]):
                continue
            new = cost + step
            if new < best.get(n, math.inf):
                best[n] = new
                heapq.heappush(frontier, (new + math.hypot(g[0] - n[0], g[1] - n[1]), new, n))
    return None


def layout_feasible(start: Point, goal: Point, nav: Polygon,
                    obstacles: Sequence[Polygon]) -> bool:
    """A route exists and is no longer than the Paper 2 ratio allows."""
    length = route_length(start, goal, nav, obstacles)
    if length is None:
        return False
    straight = max(math.hypot(goal[0] - start[0], goal[1] - start[1]), 1e-6)
    return length <= float(cfg.FEASIBILITY_MAX_ROUTE_RATIO) * straight


def thin_to_feasible(start: Point, goal: Point, nav: Polygon,
                     obstacles: List[Polygon], path_points) -> List[Polygon]:
    """Drop panels, nearest the reference path first, until a route exists."""
    kept = list(obstacles)
    pts = np.asarray(path_points, dtype=float)
    while kept and not layout_feasible(start, goal, nav, kept):
        dist = [float(np.min(np.hypot(pts[:, 0] - np.mean([p[0] for p in poly]),
                                      pts[:, 1] - np.mean([p[1] for p in poly]))))
                for poly in kept]
        kept.pop(int(np.argmin(dist)))
    return kept
