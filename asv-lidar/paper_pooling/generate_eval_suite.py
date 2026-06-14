"""
Generate a fixed 600-episode ASV evaluation suite.

Purpose
-------
This replaces the old seed-only evaluation with a saved holdout set:
  - 100 scenarios with 0 obstacles
  - 100 scenarios with 1 obstacle
  - ...
  - 100 scenarios with 5 obstacles

The output is a single JSON file containing all start/goal/obstacle layouts.
Each scenario is also saved as an individual JSON file compatible with the
simple test_run.py case-99 style: {"start": ..., "goal": ..., "obstacles": ...}.
"""

from __future__ import annotations

import json
import math
import os
import heapq
from typing import Dict, List, Tuple

import numpy as np

from rl_env import ASVLidarEnv, DEFAULT_EVAL_LAMBDA

# -----------------------------
# User settings
# -----------------------------
OUT_DIR = "data/env_setup/eval_suite"
SUITE_JSON = os.path.join(OUT_DIR, "asv_eval_suite.json")
INDIVIDUAL_DIR = os.path.join(OUT_DIR, "cases")

N_PER_OBS_COUNT = 100
OBSTACLE_COUNTS = [0, 1, 2, 3, 4]
BASE_SEED = 675973

MAP_WIDTH = 10.0
MAP_HEIGHT = 25.0
PATH_MODE = "straight"
LAMBDA_VALUE = DEFAULT_EVAL_LAMBDA

# Start/goal curriculum for the holdout suite.
# 70% vertical, 30% mildly slanted.
VERTICAL_PROB = 0.70
X_MARGIN_FRAC = 0.25
X_MARGIN_MIN = 2.0
SLANT_MAX_DX = 2.0
START_Y = 2.0
GOAL_Y_MARGIN = 3.0

# Feasibility filtering.
# The A* check inflates obstacles and borders; it is deliberately conservative
# but not a full ship-dynamics proof.
GRID_RES = 0.25
BORDER_INFLATION = 0.35
OBSTACLE_INFLATION = 0.45
MIN_ROUTE_RATIO = 1.00       # lower bound on A* path/ref path ratio
MAX_ROUTE_RATIO = 2.25       # reject very convoluted layouts
MAX_ATTEMPTS_PER_CASE = 400

# If the current env's obstacle generator sometimes fails to produce the
# requested count, retry.
EXACT_OBSTACLE_COUNT = True


Point = Tuple[float, float]
Polygon = List[Point]


def _poly_to_json(poly: Polygon) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in poly]


def sample_start_goal(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    margin_x = max(X_MARGIN_MIN, X_MARGIN_FRAC * MAP_WIDTH)
    start_x = float(rng.uniform(margin_x, MAP_WIDTH - margin_x))

    if rng.random() < VERTICAL_PROB:
        goal_x = start_x
    else:
        goal_x = float(np.clip(
            start_x + rng.uniform(-SLANT_MAX_DX, SLANT_MAX_DX),
            margin_x,
            MAP_WIDTH - margin_x,
        ))

    return start_x, START_Y, goal_x, MAP_HEIGHT - GOAL_Y_MARGIN


def build_path_for_env(env: ASVLidarEnv, sx: float, sy: float, gx: float, gy: float) -> np.ndarray:
    env.start_x, env.start_y = float(sx), float(sy)
    env.goal_x, env.goal_y = float(gx), float(gy)
    return env._generate_path(env.start_x, env.start_y, env.goal_x, env.goal_y)


def generate_obstacles_exact(env: ASVLidarEnv, obs_count: int, seed: int) -> List[Polygon]:
    if obs_count == 0:
        return []
    # The env generator uses np.random directly, so seed it here.
    np.random.seed(seed)
    obs = env._generate_obstacles(obs_count, test_case=None)
    if EXACT_OBSTACLE_COUNT and len(obs) != obs_count:
        return []
    return obs


# -----------------------------
# Grid feasibility check
# -----------------------------
def point_in_inflated_rect(x: float, y: float, obs: Polygon, inflation: float) -> bool:
    xs = [p[0] for p in obs]
    ys = [p[1] for p in obs]
    return (min(xs) - inflation <= x <= max(xs) + inflation and
            min(ys) - inflation <= y <= max(ys) + inflation)


def is_free(x: float, y: float, obstacles: List[Polygon]) -> bool:
    if x < BORDER_INFLATION or x > MAP_WIDTH - BORDER_INFLATION:
        return False
    if y < BORDER_INFLATION or y > MAP_HEIGHT - BORDER_INFLATION:
        return False
    for obs in obstacles:
        if point_in_inflated_rect(x, y, obs, OBSTACLE_INFLATION):
            return False
    return True


def to_cell(p: Point) -> Tuple[int, int]:
    return int(round(p[0] / GRID_RES)), int(round(p[1] / GRID_RES))


def to_world(c: Tuple[int, int]) -> Point:
    return c[0] * GRID_RES, c[1] * GRID_RES


def astar_path_length(start: Point, goal: Point, obstacles: List[Polygon]) -> float | None:
    start_c = to_cell(start)
    goal_c = to_cell(goal)

    if not is_free(*to_world(start_c), obstacles):
        return None
    if not is_free(*to_world(goal_c), obstacles):
        return None

    max_ix = int(round(MAP_WIDTH / GRID_RES))
    max_iy = int(round(MAP_HEIGHT / GRID_RES))

    def h(c):
        return math.hypot(c[0] - goal_c[0], c[1] - goal_c[1]) * GRID_RES

    nbrs = [
        (-1, 0, GRID_RES), (1, 0, GRID_RES), (0, -1, GRID_RES), (0, 1, GRID_RES),
        (-1, -1, GRID_RES * math.sqrt(2)), (-1, 1, GRID_RES * math.sqrt(2)),
        (1, -1, GRID_RES * math.sqrt(2)), (1, 1, GRID_RES * math.sqrt(2)),
    ]

    open_q = [(h(start_c), 0.0, start_c)]
    best = {start_c: 0.0}

    while open_q:
        _, g, c = heapq.heappop(open_q)
        if c == goal_c:
            return float(g)
        if g > best.get(c, float("inf")) + 1e-9:
            continue
        for dx, dy, cost in nbrs:
            nc = (c[0] + dx, c[1] + dy)
            if not (0 <= nc[0] <= max_ix and 0 <= nc[1] <= max_iy):
                continue
            if not is_free(*to_world(nc), obstacles):
                continue
            ng = g + cost
            if ng < best.get(nc, float("inf")):
                best[nc] = ng
                heapq.heappush(open_q, (ng + h(nc), ng, nc))
    return None


def feasible(start: Point, goal: Point, obstacles: List[Polygon]) -> Tuple[bool, float | None]:
    route_len = astar_path_length(start, goal, obstacles)
    if route_len is None:
        return False, None
    ref_len = max(1e-6, math.hypot(goal[0] - start[0], goal[1] - start[1]))
    ratio = route_len / ref_len
    if ratio < MIN_ROUTE_RATIO or ratio > MAX_ROUTE_RATIO:
        return False, ratio
    return True, ratio


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

    env = ASVLidarEnv(
        render_mode=None,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        max_obs=max(OBSTACLE_COUNTS),
        path_mode=PATH_MODE,
        lambda_override=LAMBDA_VALUE,
        test_case=None,
        record_video=False,
    )
    env.reset(seed=BASE_SEED)

    scenarios: List[Dict] = []
    counts_done = {n: 0 for n in OBSTACLE_COUNTS}

    for obs_count in OBSTACLE_COUNTS:
        attempts = 0
        while counts_done[obs_count] < N_PER_OBS_COUNT:
            if attempts > N_PER_OBS_COUNT * MAX_ATTEMPTS_PER_CASE:
                raise RuntimeError(
                    f"Could not generate enough feasible scenarios for {obs_count} obstacles. "
                    f"Generated {counts_done[obs_count]}/{N_PER_OBS_COUNT}."
                )
            attempts += 1
            local_idx = counts_done[obs_count]
            seed = BASE_SEED + obs_count * 100000 + attempts
            rng = np.random.default_rng(seed)

            sx, sy, gx, gy = sample_start_goal(rng)
            path = build_path_for_env(env, sx, sy, gx, gy)
            obstacles = generate_obstacles_exact(env, obs_count, seed + 17)
            if EXACT_OBSTACLE_COUNT and len(obstacles) != obs_count:
                continue

            ok, route_ratio = feasible((sx, sy), (gx, gy), obstacles)
            if not ok:
                continue

            case_id = obs_count * 100 + local_idx
            scenario = {
                "case_id": int(case_id),
                "group": f"obs_{obs_count}",
                "obstacle_count": int(obs_count),
                "seed": int(seed),
                "start": [float(sx), float(sy)],
                "goal": [float(gx), float(gy)],
                "obstacles": [[list(p) for p in obs] for obs in obstacles],
                "path": path.tolist(),
                "map_width": float(MAP_WIDTH),
                "map_height": float(MAP_HEIGHT),
                "path_mode": PATH_MODE,
                "route_ratio_astar": None if route_ratio is None else float(route_ratio),
            }
            scenarios.append(scenario)

            # Individual test_run.py case-99 style file.
            individual_path = os.path.join(INDIVIDUAL_DIR, f"obs{obs_count}_case{local_idx:03d}.json")
            with open(individual_path, "w") as f:
                json.dump({
                    "start": scenario["start"],
                    "goal": scenario["goal"],
                    "obstacles": scenario["obstacles"],
                    "map_width": scenario["map_width"],
                    "map_height": scenario["map_height"],
                    "path_mode": scenario["path_mode"],
                    "path": scenario["path"],
                    "source_suite": SUITE_JSON,
                    "case_id": scenario["case_id"],
                    "obstacle_count": obs_count,
                }, f, indent=2)

            counts_done[obs_count] += 1
            if counts_done[obs_count] % 20 == 0:
                print(f"Generated {counts_done[obs_count]:3d}/{N_PER_OBS_COUNT} for {obs_count} obstacles")

    suite = {
        "metadata": {
            "description": "600-case ASV evaluation suite: 100 scenarios for each obstacle count 0..5.",
            "map_width": MAP_WIDTH,
            "map_height": MAP_HEIGHT,
            "path_mode": PATH_MODE,
            "n_per_obstacle_count": N_PER_OBS_COUNT,
            "obstacle_counts": OBSTACLE_COUNTS,
            "base_seed": BASE_SEED,
            "vertical_probability": VERTICAL_PROB,
            "grid_res": GRID_RES,
            "border_inflation": BORDER_INFLATION,
            "obstacle_inflation": OBSTACLE_INFLATION,
        },
        "scenarios": scenarios,
    }

    with open(SUITE_JSON, "w") as f:
        json.dump(suite, f, indent=2)

    print("\nDone.")
    print(f"Saved suite: {SUITE_JSON}")
    print(f"Saved individual scenarios: {INDIVIDUAL_DIR}")
    print("Counts:", counts_done)


if __name__ == "__main__":
    main()
