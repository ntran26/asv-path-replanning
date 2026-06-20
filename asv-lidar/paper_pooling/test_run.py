"""Built-in deterministic scenario set for quick ASV policy tests.

Scenario layout
---------------
The cases are grouped by obstacle count:

    00-09 : 0 obstacles, pure path tracking
    10-19 : 1 obstacle
    20-29 : 2 obstacles
    30-39 : 3 obstacles
    40-49 : 4 obstacles
    50-59 : 5 obstacles

Within each 10-case block, cases are arranged from easier to harder. Cases
ending in 9 use a deterministic curved reference path; all other cases use a
straight mostly-slanted path. Coordinates are defined for the canonical
10 m x 25 m map used by rl_env.py and are scaled by rl_env.py when a different
map size is requested.

This file is intentionally small and dependency-light because it is imported by
both the simulator environment and the UDP deployment/visualisation tools.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]
Polygon = List[Point]

MAP_WIDTH = 10.0
MAP_HEIGHT = 25.0
START_Y = 2.0
GOAL_Y = 22.0

# Cases are in blocks of 10. 0..5 obstacles => 60 cases total.
MIN_CASE_ID = 0
MAX_CASE_ID = 59
CASES_PER_GROUP = 10
MAX_OBSTACLES = 5


def _box(cx: float, cy: float, size: float = 1.0) -> Polygon:
    h = 0.5 * float(size)
    return [
        (float(cx - h), float(cy - h)),
        (float(cx + h), float(cy - h)),
        (float(cx + h), float(cy + h)),
        (float(cx - h), float(cy + h)),
    ]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _lerp(a: float, b: float, t: float) -> float:
    return float((1.0 - t) * a + t * b)


def _normalised(vx: float, vy: float) -> Point:
    n = (vx * vx + vy * vy) ** 0.5
    if n <= 1e-9:
        return (0.0, 1.0)
    return (float(vx / n), float(vy / n))


# Mostly slanted start/goal templates. The last local case in each block is
# curved through the control point specified separately.
_PATH_TEMPLATES: Dict[int, Dict[str, Point]] = {
    # straight and diagonal paths
    0: {"start": (5.0, START_Y), "goal": (5.0, GOAL_Y)},
    1: {"start": (2.0, START_Y), "goal": (8.0, GOAL_Y)},
    2: {"start": (8.0, START_Y), "goal": (2.0, GOAL_Y)},
    # Medium slant
    3: {"start": (3.9, START_Y), "goal": (6.1, GOAL_Y)},
    4: {"start": (6.1, START_Y), "goal": (3.9, GOAL_Y)},
    5: {"start": (3.6, START_Y), "goal": (6.4, GOAL_Y)},
    6: {"start": (6.4, START_Y), "goal": (3.6, GOAL_Y)},
    # Harder slants, closer to pool sides while still feasible
    7: {"start": (3.2, START_Y), "goal": (6.8, GOAL_Y)},
    8: {"start": (6.8, START_Y), "goal": (3.2, GOAL_Y)},
    # Curved path case at the end of each block
    9: {"start": (3.8, START_Y), "goal": (6.2, GOAL_Y), "control": (7.0, 12.0)},
}


_OBS_FRACTIONS: Dict[int, Tuple[float, ...]] = {
    0: (),
    1: (0.50,),
    2: (0.38, 0.62),
    3: (0.28, 0.50, 0.72),
    4: (0.22, 0.40, 0.60, 0.78),
    5: (0.18, 0.34, 0.50, 0.66, 0.82),
}


def _offset_pattern(local_id: int, obstacle_count: int) -> Tuple[float, ...]:
    """Lateral obstacle offsets from the reference path, easy -> hard.

    Positive offset means one side of the path; negative offset means the other
    side. Zero means the obstacle is centred on/very near the path.
    """
    if obstacle_count <= 0:
        return ()

    # Base side alternation, varied by local case so the policy is tested on
    # both port/starboard bypasses.
    side = 1.0 if local_id % 2 == 0 else -1.0

    if local_id <= 2:
        # Easy: obstacles are clearly offset from the path.
        amp = (1.30, 1.15, 1.00)[local_id]
        pattern = [side * amp * ((-1.0) ** i) for i in range(obstacle_count)]
    elif local_id <= 5:
        # Medium: closer to the path, still with obvious bypass side.
        amp = (0.85, 0.70, 0.55)[local_id - 3]
        pattern = [side * amp * ((-1.0) ** i) for i in range(obstacle_count)]
    elif local_id == 6:
        # Medium-hard: include one centred obstacle.
        base = [0.0, side * 0.85, -side * 0.85, side * 0.55, -side * 0.55]
        pattern = base[:obstacle_count]
    elif local_id == 7:
        # Hard: near-centre obstacles plus alternating bypass requirements.
        base = [side * 0.35, -side * 0.70, side * 0.70, -side * 0.35, 0.0]
        pattern = base[:obstacle_count]
    elif local_id == 8:
        # Hardest straight case: several path-blocking or near-blocking obstacles.
        base = [0.0, side * 0.45, -side * 0.45, 0.0, -side * 0.65]
        pattern = base[:obstacle_count]
    else:
        # Curved case: still challenging, but not impossible; obstacles are
        # placed around the curve and alternate sides.
        base = [side * 0.75, -side * 0.75, 0.0, side * 0.60, -side * 0.60]
        pattern = base[:obstacle_count]

    return tuple(float(x) for x in pattern)


def _case_ids(test_case: int) -> Tuple[int, int, int]:
    """Return (case_id, obstacle_count, local_id)."""
    tc = int(test_case)
    if MIN_CASE_ID <= tc <= MAX_CASE_ID:
        obstacle_count = tc // CASES_PER_GROUP
        local_id = tc % CASES_PER_GROUP
        return tc, obstacle_count, local_id

    # Keep case 99 as a simple legacy empty-path test.
    if tc == 99:
        return 99, 0, 0

    # Safe fallback.
    return 0, 0, 0


def _path_spec(local_id: int) -> Dict[str, Point]:
    return _PATH_TEMPLATES.get(int(local_id), _PATH_TEMPLATES[0])


def _point_on_path(local_id: int, t: float) -> Point:
    spec = _path_spec(local_id)
    sx, sy = spec["start"]
    gx, gy = spec["goal"]
    t = _clip(float(t), 0.0, 1.0)

    if local_id == 9 and "control" in spec:
        cx, cy = spec["control"]
        omt = 1.0 - t
        x = omt * omt * sx + 2.0 * omt * t * cx + t * t * gx
        y = omt * omt * sy + 2.0 * omt * t * cy + t * t * gy
        return (float(x), float(y))

    return (_lerp(sx, gx, t), _lerp(sy, gy, t))


def _path_tangent(local_id: int, t: float) -> Point:
    spec = _path_spec(local_id)
    sx, sy = spec["start"]
    gx, gy = spec["goal"]
    t = _clip(float(t), 0.0, 1.0)

    if local_id == 9 and "control" in spec:
        cx, cy = spec["control"]
        # Derivative of quadratic Bezier curve.
        dx = 2.0 * (1.0 - t) * (cx - sx) + 2.0 * t * (gx - cx)
        dy = 2.0 * (1.0 - t) * (cy - sy) + 2.0 * t * (gy - cy)
        return _normalised(dx, dy)

    return _normalised(gx - sx, gy - sy)


def _normal_left(local_id: int, t: float) -> Point:
    tx, ty = _path_tangent(local_id, t)
    return (-ty, tx)


def _sample_path(local_id: int, n: int = 100) -> List[Point]:
    n = max(2, int(n))
    return [_point_on_path(local_id, i / float(n - 1)) for i in range(n)]


def _scenario_obstacles(obstacle_count: int, local_id: int) -> List[Polygon]:
    obstacle_count = int(max(0, min(MAX_OBSTACLES, obstacle_count)))
    if obstacle_count <= 0:
        return []

    fracs = _OBS_FRACTIONS[obstacle_count]
    offsets = _offset_pattern(local_id, obstacle_count)

    obstacles: List[Polygon] = []
    for frac, offset in zip(fracs, offsets):
        px, py = _point_on_path(local_id, frac)
        nx, ny = _normal_left(local_id, frac)

        # Lateral offset from path, with clipping to keep boxes inside the map.
        cx = _clip(px + float(offset) * nx, 1.0, MAP_WIDTH - 1.0)
        cy = _clip(py + float(offset) * ny, 4.0, MAP_HEIGHT - 4.0)
        obstacles.append(_box(cx, cy, 1.0))

    return obstacles


class TestCase:
    """Preset scenarios for quick deterministic visual policy tests."""

    def count(self) -> int:
        return (MAX_OBSTACLES + 1) * CASES_PER_GROUP

    def obstacle_count(self, test_case: int = 0) -> int:
        _, obstacle_count, _ = _case_ids(test_case)
        return int(obstacle_count)

    def path_mode(self, test_case: int = 0) -> str:
        _, _, local_id = _case_ids(test_case)
        return "curve" if local_id == 9 else "straight"

    def difficulty(self, test_case: int = 0) -> str:
        _, _, local_id = _case_ids(test_case)
        if local_id <= 2:
            return "easy"
        if local_id <= 5:
            return "medium"
        if local_id <= 7:
            return "hard"
        return "very_hard" if local_id == 8 else "curved"

    def description(self, test_case: int = 0) -> str:
        case_id, obstacle_count, local_id = _case_ids(test_case)
        return (
            f"case {case_id}: obs_{obstacle_count}, "
            f"{self.path_mode(test_case)} path, {self.difficulty(test_case)}"
        )

    def position(self, test_case: int = 0) -> Tuple[float, float, float, float]:
        # Coordinates are in the canonical 10 m x 25 m map. rl_env.py scales
        # them if a different map size is requested.
        case_id, _, local_id = _case_ids(test_case)
        if case_id == 99:
            return (5.0, START_Y, 5.0, GOAL_Y)
        spec = _path_spec(local_id)
        sx, sy = spec["start"]
        gx, gy = spec["goal"]
        return (float(sx), float(sy), float(gx), float(gy))

    def path(self, test_case: int = 0, n: Optional[int] = None) -> List[Point]:
        """Return the deterministic reference path for this case.

        Current rl_env.py versions may only call position(); if you want the
        curved cases to be truly curved, update rl_env.py to use this method
        when test_case is not None.
        """
        case_id, _, local_id = _case_ids(test_case)
        if case_id == 99:
            local_id = 0
        if n is None:
            # Use denser samples for the curved path so the closest-point logic
            # remains smooth.
            n = 120 if local_id == 9 else 80
        return _sample_path(local_id, int(n))

    def obstacles(self, test_case: int = 0) -> List[Polygon]:
        case_id, obstacle_count, local_id = _case_ids(test_case)
        if case_id == 99:
            return []
        return _scenario_obstacles(obstacle_count, local_id)


if __name__ == "__main__":
    tc = TestCase()
    for group in range(MAX_OBSTACLES + 1):
        ids = range(group * CASES_PER_GROUP, group * CASES_PER_GROUP + CASES_PER_GROUP)
        print(f"obs_{group}:")
        for case_id in ids:
            print(f"  {tc.description(case_id)} start/goal={tc.position(case_id)} obstacles={len(tc.obstacles(case_id))}")
