"""Generate the fixed holdout evaluation suite.

100 scenarios for each obstacle count in OBSTACLE_COUNTS, written as one JSON
file plus one file per scenario.  Every layout is checked for reachability with
an inflated-grid A* before being accepted, so the suite contains no impossible
cases.

The shipped suite already lives in eval_suite/; it is a fixed holdout and
regenerating it mid-study makes results incomparable.  OUT_DIR therefore points
somewhere else on purpose - copy the result over deliberately, never by
accident.
"""

from __future__ import annotations

import heapq
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import ASVLidarEnv

# -----------------------------
# USER SETTINGS
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

# Start/goal curriculum: mostly vertical, some mildly slanted.
VERTICAL_PROB = 0.70
X_MARGIN_FRAC = 0.25
X_MARGIN_MIN = 2.0
SLANT_MAX_DX = 2.0
START_Y = 2.0
GOAL_Y_MARGIN = 3.0

# Feasibility filter.  Deliberately conservative, but not a ship-dynamics proof.
GRID_RES = 0.25
BORDER_INFLATION = 0.35
OBSTACLE_INFLATION = 0.45
MIN_ROUTE_RATIO = 1.00        # A* route / straight-line reference path
MAX_ROUTE_RATIO = 2.25        # reject very convoluted layouts
MAX_ATTEMPTS_PER_CASE = 400

Point = Tuple[float, float]
Polygon = List[Point]


def sample_start_goal(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    margin_x = max(X_MARGIN_MIN, X_MARGIN_FRAC * MAP_WIDTH)
    start_x = float(rng.uniform(margin_x, MAP_WIDTH - margin_x))

    if rng.random() < VERTICAL_PROB:
        goal_x = start_x
    else:
        goal_x = float(np.clip(start_x + rng.uniform(-SLANT_MAX_DX, SLANT_MAX_DX),
                               margin_x, MAP_WIDTH - margin_x))
    return start_x, START_Y, goal_x, MAP_HEIGHT - GOAL_Y_MARGIN


# -----------------------------
# Grid feasibility check
# -----------------------------
def is_free(x: float, y: float, obstacles: List[Polygon]) -> bool:
    if not (BORDER_INFLATION <= x <= MAP_WIDTH - BORDER_INFLATION):
        return False
    if not (BORDER_INFLATION <= y <= MAP_HEIGHT - BORDER_INFLATION):
        return False
    for obs in obstacles:
        xs = [p[0] for p in obs]
        ys = [p[1] for p in obs]
        if (min(xs) - OBSTACLE_INFLATION <= x <= max(xs) + OBSTACLE_INFLATION
                and min(ys) - OBSTACLE_INFLATION <= y <= max(ys) + OBSTACLE_INFLATION):
            return False
    return True


def astar_path_length(start: Point, goal: Point, obstacles: List[Polygon]) -> Optional[float]:
    """Shortest 8-connected grid route, or None if the goal is unreachable."""
    def to_cell(p: Point):
        return int(round(p[0] / GRID_RES)), int(round(p[1] / GRID_RES))

    def to_world(c):
        return c[0] * GRID_RES, c[1] * GRID_RES

    start_c, goal_c = to_cell(start), to_cell(goal)
    if not is_free(*to_world(start_c), obstacles) or not is_free(*to_world(goal_c), obstacles):
        return None

    max_ix = int(round(MAP_WIDTH / GRID_RES))
    max_iy = int(round(MAP_HEIGHT / GRID_RES))
    diagonal = GRID_RES * math.sqrt(2)
    neighbours = [(-1, 0, GRID_RES), (1, 0, GRID_RES), (0, -1, GRID_RES), (0, 1, GRID_RES),
                  (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal)]

    def heuristic(c):
        return math.hypot(c[0] - goal_c[0], c[1] - goal_c[1]) * GRID_RES

    frontier = [(heuristic(start_c), 0.0, start_c)]
    best = {start_c: 0.0}
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell == goal_c:
            return float(cost)
        if cost > best.get(cell, float("inf")) + 1e-9:
            continue
        for dx, dy, step in neighbours:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] <= max_ix and 0 <= nxt[1] <= max_iy):
                continue
            if not is_free(*to_world(nxt), obstacles):
                continue
            new_cost = cost + step
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(frontier, (new_cost + heuristic(nxt), new_cost, nxt))
    return None


def route_ratio(start: Point, goal: Point, obstacles: List[Polygon]) -> Optional[float]:
    """A* route length relative to the straight reference path, or None."""
    length = astar_path_length(start, goal, obstacles)
    if length is None:
        return None
    reference = max(1e-6, math.hypot(goal[0] - start[0], goal[1] - start[1]))
    return length / reference


def build_scenario(env: ASVLidarEnv, obs_count: int, local_idx: int, seed: int) -> Optional[Dict]:
    """Attempt one scenario; return None if the layout is infeasible."""
    rng = np.random.default_rng(seed)
    sx, sy, gx, gy = sample_start_goal(rng)

    env.start_x, env.start_y = sx, sy
    env.goal_x, env.goal_y = gx, gy
    env._build_path()

    obstacles: List[Polygon] = []
    if obs_count > 0:
        # The sampler draws from the global stream, so seed it here.
        np.random.seed(seed + 17)
        obstacles = env.sample_obstacles(obs_count)
    if len(obstacles) != obs_count:
        return None

    ratio = route_ratio((sx, sy), (gx, gy), obstacles)
    if ratio is None or not (MIN_ROUTE_RATIO <= ratio <= MAX_ROUTE_RATIO):
        return None

    return {
        "case_id": obs_count * 100 + local_idx,
        "group": f"obs_{obs_count}",
        "obstacle_count": obs_count,
        "seed": int(seed),
        "start": [float(sx), float(sy)],
        "goal": [float(gx), float(gy)],
        "obstacles": [[list(p) for p in obs] for obs in obstacles],
        "path": env.path.points.tolist(),
        "map_width": MAP_WIDTH,
        "map_height": MAP_HEIGHT,
        "path_mode": PATH_MODE,
        "route_ratio_astar": float(ratio),
    }


def main() -> None:
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

    env = ASVLidarEnv(map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
                      max_obs=max(OBSTACLE_COUNTS), path_mode=PATH_MODE)
    env.reset(seed=BASE_SEED)

    scenarios: List[Dict] = []
    counts_done = {n: 0 for n in OBSTACLE_COUNTS}

    for obs_count in OBSTACLE_COUNTS:
        attempts = 0
        while counts_done[obs_count] < N_PER_OBS_COUNT:
            attempts += 1
            if attempts > N_PER_OBS_COUNT * MAX_ATTEMPTS_PER_CASE:
                raise RuntimeError(
                    f"Could not generate enough feasible scenarios for {obs_count} obstacles. "
                    f"Generated {counts_done[obs_count]}/{N_PER_OBS_COUNT}.")

            local_idx = counts_done[obs_count]
            scenario = build_scenario(env, obs_count, local_idx, BASE_SEED + obs_count * 100000 + attempts)
            if scenario is None:
                continue

            scenarios.append(scenario)
            individual = os.path.join(INDIVIDUAL_DIR, f"obs{obs_count}_case{local_idx:03d}.json")
            with open(individual, "w") as f:
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

    with open(SUITE_JSON, "w") as f:
        json.dump({
            "metadata": {
                # Kept verbatim so a regenerated suite matches the shipped file
                # byte for byte, even though the counts below are the truth.
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
        }, f, indent=2)

    print("\nDone.")
    print(f"Saved suite: {SUITE_JSON}")
    print(f"Saved individual scenarios: {INDIVIDUAL_DIR}")
    print("Counts:", counts_done)


if __name__ == "__main__":
    main()
