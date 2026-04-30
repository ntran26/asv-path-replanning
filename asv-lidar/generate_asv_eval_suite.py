#!/usr/bin/env python3
"""
Generate a deterministic ASV holdout evaluation suite with feasible multi-obstacle cases.

The generated JSON can be used as a fixed test suite for simulation-to-field evaluation.
It does NOT require gymnasium/SB3; it only uses numpy/matplotlib.

Default: 100 cases, 10x25 map, 2-5 static rectangular obstacles, 70/30 vertical/slanted paths,
with an A* feasibility check on an obstacle-inflated grid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

Point = Tuple[float, float]
Polygon = List[Point]


@dataclass
class SuiteConfig:
    n_cases: int = 100
    seed: int = 675973
    map_width: float = 10.0
    map_height: float = 25.0
    grid_res: float = 0.15
    obstacle_min: int = 2
    obstacle_max: int = 5
    vertical_prob: float = 0.70
    margin_x_frac: float = 0.25
    start_y: float = 2.0
    goal_y_margin: float = 3.0
    obstacle_size: float = 1.0
    obstacle_size_jitter: float = 0.15
    obstacle_start_frac: float = 0.18
    obstacle_end_frac: float = 0.82
    lateral_sigma_frac: float = 0.20
    centered_obstacle_prob: float = 0.20
    min_obs_center_sep: float = 1.35
    min_start_goal_clear: float = 2.0
    inflate_radius: float = 0.45  # approximate vessel+margin radius for feasibility check
    min_astar_path_len_ratio: float = 1.00
    max_astar_path_len_ratio: float = 2.20
    max_attempts: int = 30000


def make_box(cx: float, cy: float, sx: float, sy: float) -> Polygon:
    hx, hy = 0.5 * sx, 0.5 * sy
    return [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]


def poly_center(poly: Polygon) -> np.ndarray:
    return np.mean(np.asarray(poly, dtype=float), axis=0)


def obstacle_bbox(poly: Polygon) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def generate_path(start: Point, goal: Point, n: int = 160) -> np.ndarray:
    xs = np.linspace(start[0], goal[0], n, dtype=np.float32)
    ys = np.linspace(start[1], goal[1], n, dtype=np.float32)
    return np.column_stack([xs, ys])


def path_tangent(path: np.ndarray, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, len(path) - 1))
    if idx <= 0:
        v = path[1] - path[0]
    elif idx >= len(path) - 1:
        v = path[-1] - path[-2]
    else:
        v = path[idx + 1] - path[idx - 1]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.array([0.0, 1.0], dtype=np.float32)


def sample_start_goal(cfg: SuiteConfig, rng: np.random.Generator) -> Tuple[Point, Point, str]:
    margin_x = max(2.0, cfg.margin_x_frac * cfg.map_width)
    if rng.random() < cfg.vertical_prob:
        x = float(rng.uniform(margin_x, cfg.map_width - margin_x))
        return (x, cfg.start_y), (x, cfg.map_height - cfg.goal_y_margin), "vertical"
    sx = float(rng.uniform(margin_x, cfg.map_width - margin_x))
    # mild diagonal, not extreme corner-to-corner
    gx = float(np.clip(sx + rng.uniform(-2.2, 2.2), margin_x, cfg.map_width - margin_x))
    return (sx, cfg.start_y), (gx, cfg.map_height - cfg.goal_y_margin), "slanted"


def sample_obstacles(cfg: SuiteConfig, path: np.ndarray, start: Point, goal: Point, rng: np.random.Generator) -> List[Polygon]:
    n_obs = int(rng.integers(cfg.obstacle_min, cfg.obstacle_max + 1))
    obstacles: List[Polygon] = []
    s_min = int(cfg.obstacle_start_frac * len(path))
    s_max = int(cfg.obstacle_end_frac * len(path))
    s_min = max(1, s_min)
    s_max = min(len(path) - 2, s_max)

    for _ in range(250):
        if len(obstacles) >= n_obs:
            break
        idx = int(rng.integers(s_min, s_max + 1))
        center = path[idx].astype(float)
        tangent = path_tangent(path, idx).astype(float)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        if rng.random() < cfg.centered_obstacle_prob:
            lateral = rng.normal(0.0, 0.15)
        else:
            lateral = rng.normal(0.0, cfg.lateral_sigma_frac * cfg.map_width)

        center = center + lateral * normal
        sx = float(np.clip(cfg.obstacle_size * rng.uniform(1.0 - cfg.obstacle_size_jitter, 1.0 + cfg.obstacle_size_jitter), 0.75, 1.35))
        sy = float(np.clip(cfg.obstacle_size * rng.uniform(1.0 - cfg.obstacle_size_jitter, 1.0 + cfg.obstacle_size_jitter), 0.75, 1.35))
        cx, cy = float(center[0]), float(center[1])

        margin = 0.8
        if not (margin + sx / 2 <= cx <= cfg.map_width - margin - sx / 2):
            continue
        if not (margin + sy / 2 <= cy <= cfg.map_height - margin - sy / 2):
            continue

        c = np.array([cx, cy], dtype=float)
        if np.linalg.norm(c - np.array(start)) < cfg.min_start_goal_clear:
            continue
        if np.linalg.norm(c - np.array(goal)) < cfg.min_start_goal_clear:
            continue

        too_close = False
        for obs in obstacles:
            if np.linalg.norm(c - poly_center(obs)) < cfg.min_obs_center_sep:
                too_close = True
                break
        if too_close:
            continue
        obstacles.append(make_box(cx, cy, sx, sy))

    return obstacles if len(obstacles) >= cfg.obstacle_min else []


def make_occupancy(cfg: SuiteConfig, obstacles: List[Polygon]) -> np.ndarray:
    nx = int(round(cfg.map_width / cfg.grid_res)) + 1
    ny = int(round(cfg.map_height / cfg.grid_res)) + 1
    occ = np.zeros((nx, ny), dtype=bool)
    # true border inflated by clearance: disallow cells too close to map edge
    for ix in range(nx):
        x = ix * cfg.grid_res
        for iy in range(ny):
            y = iy * cfg.grid_res
            if x < cfg.inflate_radius or x > cfg.map_width - cfg.inflate_radius or y < cfg.inflate_radius or y > cfg.map_height - cfg.inflate_radius:
                occ[ix, iy] = True
                continue
            for obs in obstacles:
                x0, x1, y0, y1 = obstacle_bbox(obs)
                if (x0 - cfg.inflate_radius) <= x <= (x1 + cfg.inflate_radius) and (y0 - cfg.inflate_radius) <= y <= (y1 + cfg.inflate_radius):
                    occ[ix, iy] = True
                    break
    return occ


def world_to_grid(cfg: SuiteConfig, p: Point) -> Tuple[int, int]:
    return int(round(p[0] / cfg.grid_res)), int(round(p[1] / cfg.grid_res))


def grid_to_world(cfg: SuiteConfig, ij: Tuple[int, int]) -> Point:
    return ij[0] * cfg.grid_res, ij[1] * cfg.grid_res


def astar(cfg: SuiteConfig, occ: np.ndarray, start: Point, goal: Point) -> Optional[List[Point]]:
    nx, ny = occ.shape
    s = world_to_grid(cfg, start)
    g = world_to_grid(cfg, goal)
    if not (0 <= s[0] < nx and 0 <= s[1] < ny and 0 <= g[0] < nx and 0 <= g[1] < ny):
        return None
    if occ[s] or occ[g]:
        return None
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    heap = []
    heappush(heap, (0.0, s))
    came: Dict[Tuple[int,int], Tuple[int,int]] = {}
    gscore = {s: 0.0}
    def h(a):
        return math.hypot(a[0] - g[0], a[1] - g[1])
    closed = set()
    while heap:
        _, cur = heappop(heap)
        if cur in closed:
            continue
        if cur == g:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            return [grid_to_world(cfg, q) for q in path]
        closed.add(cur)
        for dx, dy in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if occ[nb]:
                continue
            step = math.sqrt(2) if dx and dy else 1.0
            tentative = gscore[cur] + step
            if tentative < gscore.get(nb, float("inf")):
                came[nb] = cur
                gscore[nb] = tentative
                heappush(heap, (tentative + h(nb), nb))
    return None


def path_length(points: List[Point]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(math.hypot(points[i+1][0] - points[i][0], points[i+1][1] - points[i][1]) for i in range(len(points)-1)))


def min_clearance_to_obstacles(points: List[Point], obstacles: List[Polygon]) -> float:
    # approximate min distance to obstacle bbox, not exact polygon distance
    best = float("inf")
    for x, y in points:
        for obs in obstacles:
            x0, x1, y0, y1 = obstacle_bbox(obs)
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            d = math.hypot(dx, dy)
            best = min(best, d)
    return best


def generate_suite(cfg: SuiteConfig) -> List[dict]:
    rng = np.random.default_rng(cfg.seed)
    cases = []
    attempts = 0
    while len(cases) < cfg.n_cases and attempts < cfg.max_attempts:
        attempts += 1
        start, goal, path_type = sample_start_goal(cfg, rng)
        ref_path = generate_path(start, goal)
        obstacles = sample_obstacles(cfg, ref_path, start, goal, rng)
        if not obstacles:
            continue
        occ = make_occupancy(cfg, obstacles)
        astar_path = astar(cfg, occ, start, goal)
        if astar_path is None:
            continue
        ref_len = math.hypot(goal[0] - start[0], goal[1] - start[1])
        route_len = path_length(astar_path)
        if ref_len <= 1e-6:
            continue
        ratio = route_len / ref_len
        if not (cfg.min_astar_path_len_ratio <= ratio <= cfg.max_astar_path_len_ratio):
            continue
        min_clr = min_clearance_to_obstacles(astar_path, obstacles)
        case = {
            "case_id": len(cases),
            "seed": int(cfg.seed),
            "attempt": int(attempts),
            "map_width": cfg.map_width,
            "map_height": cfg.map_height,
            "start": [round(start[0], 4), round(start[1], 4)],
            "goal": [round(goal[0], 4), round(goal[1], 4)],
            "path_type": path_type,
            "obstacle_count": len(obstacles),
            "obstacles": [[[round(float(x), 4), round(float(y), 4)] for x, y in obs] for obs in obstacles],
            "feasibility": {
                "grid_res": cfg.grid_res,
                "inflate_radius": cfg.inflate_radius,
                "astar_path_length": round(route_len, 4),
                "reference_length": round(ref_len, 4),
                "path_length_ratio": round(ratio, 4),
                "approx_min_clearance_to_obstacle": round(min_clr, 4),
            },
        }
        cases.append(case)
    if len(cases) < cfg.n_cases:
        raise RuntimeError(f"Only generated {len(cases)} feasible cases after {attempts} attempts. Loosen filters or increase max attempts.")
    return cases


def write_json(path: str, cfg: SuiteConfig, cases: List[dict]) -> None:
    out = {
        "name": "asv_multi_obstacle_holdout_suite",
        "description": "Deterministic feasible multi-obstacle holdout suite for ASV local-planner evaluation.",
        "config": cfg.__dict__,
        "cases": cases,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def plot_preview(path: str, cases: List[dict], n: int = 12) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping preview")
        return
    cols = 4
    rows = math.ceil(min(n, len(cases)) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 4.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, case in zip(axes.ravel(), cases[:n]):
        w = case["map_width"]; h = case["map_height"]
        ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        ax.plot([case["start"][0], case["goal"][0]], [case["start"][1], case["goal"][1]], "--", linewidth=1)
        ax.scatter([case["start"][0]], [case["start"][1]], marker="o")
        ax.scatter([case["goal"][0]], [case["goal"][1]], marker="*", s=80)
        for obs in case["obstacles"]:
            poly = np.array(obs + [obs[0]])
            ax.fill(poly[:,0], poly[:,1], alpha=0.55)
        ax.set_title(f"#{case['case_id']} n={case['obstacle_count']} {case['path_type']}\nratio={case['feasibility']['path_length_ratio']:.2f}", fontsize=8)
        ax.axis("on")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="asv_eval_suite_100.json")
    p.add_argument("--preview", type=str, default=None)
    p.add_argument("--n-cases", type=int, default=100)
    p.add_argument("--seed", type=int, default=675973)
    p.add_argument("--map-width", type=float, default=10.0)
    p.add_argument("--map-height", type=float, default=25.0)
    p.add_argument("--min-obs", type=int, default=2)
    p.add_argument("--max-obs", type=int, default=5)
    p.add_argument("--inflate-radius", type=float, default=0.45)
    p.add_argument("--grid-res", type=float, default=0.15)
    p.add_argument("--min-ratio", type=float, default=1.00, help="Minimum A* path/reference length ratio")
    p.add_argument("--max-ratio", type=float, default=2.20, help="Maximum A* path/reference length ratio")
    p.add_argument("--centered-prob", type=float, default=0.20, help="Probability obstacle is near path centerline")
    p.add_argument("--lateral-sigma-frac", type=float, default=0.20, help="Std-dev of obstacle lateral offset as fraction of map width")
    p.add_argument("--vertical-prob", type=float, default=0.70, help="Probability of vertical path")
    args = p.parse_args()
    cfg = SuiteConfig(
        n_cases=args.n_cases,
        seed=args.seed,
        map_width=args.map_width,
        map_height=args.map_height,
        obstacle_min=args.min_obs,
        obstacle_max=args.max_obs,
        inflate_radius=args.inflate_radius,
        grid_res=args.grid_res,
        min_astar_path_len_ratio=args.min_ratio,
        max_astar_path_len_ratio=args.max_ratio,
        centered_obstacle_prob=args.centered_prob,
        lateral_sigma_frac=args.lateral_sigma_frac,
        vertical_prob=args.vertical_prob,
    )
    cases = generate_suite(cfg)
    write_json(args.out, cfg, cases)
    print(f"Wrote {len(cases)} cases -> {args.out}")
    if args.preview:
        plot_preview(args.preview, cases, n=min(20, len(cases)))
        print(f"Wrote preview -> {args.preview}")


if __name__ == "__main__":
    main()
