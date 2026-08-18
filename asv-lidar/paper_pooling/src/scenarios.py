"""Deterministic hand-authored test cases.

Layouts are authored in the canonical 10 x 25 m basin; `ASVLidarEnv` rescales
them if it is configured with a different map size.

  0       centred straight path, no obstacles
  1       field scenario A: narrow gate then path recovery
  2       field scenario B: three-obstacle slalom.  This is the layout recorded
          in asv_lidar.mp4 / asv_data.json -- the obstacle at (6.0, 17.0) sits on
          the reference path near the goal, and the 1M policy drifts left into it
          instead of taking the ~0.5 m pass on the starboard/path side
  3       field scenario C: boundary-constrained bypass
  4       three scattered obstacles
  5       horizontal blockage across a diagonal path
  6, 7    no obstacles, goal offset left / right
  8, 9    no obstacles, curved reference path authored in CURVED_WAYPOINTS
  10-18   paper replica cases: gates, near-path passes, slalom, distractors
  19, 20  no obstacles, diagonal path
  99      layout loaded from RECORDED_SCENARIO
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

RECORDED_SCENARIO = "data/env_setup/survival_pool/env_0.json"

OBSTACLE_SIZE = 1.0

Box = List[Tuple[float, float]]


def _box(x: float, y: float, size: float = OBSTACLE_SIZE) -> Box:
    """Axis-aligned square obstacle centred on (x, y)."""
    return _rect(x, y, 0.5 * size, 0.5 * size)


def _rect(x: float, y: float, half_x: float, half_y: float) -> Box:
    return [(x - half_x, y - half_y), (x + half_x, y - half_y),
            (x + half_x, y + half_y), (x - half_x, y + half_y)]


# start_x, start_y, goal_x, goal_y
POSITIONS: Dict[int, Tuple[float, float, float, float]] = {
    0: (5, 2, 5, 22),
    1: (5, 2, 5, 22),
    2: (3, 2, 8, 22),
    3: (7, 2, 2, 22),
    4: (5, 2, 5, 22),
    5: (2, 2, 7, 22),
    6: (5, 2, 3, 22),
    7: (5, 2, 7, 22),
    8: (5, 2, 5, 22),
    9: (5, 2, 5, 22),
    10: (5, 2, 5, 22),
    11: (5, 2, 5, 22),
    12: (5, 2, 5, 22),
    13: (5, 2, 5, 22),
    14: (5, 2, 5, 22),
    15: (5, 2, 5, 22),
    16: (5, 2, 5, 22),
    17: (5, 2, 5, 22),
    18: (5, 2, 5, 22),
    19: (3, 2, 7, 22),
    20: (7, 2, 3, 22),
}

OBSTACLES: Dict[int, List[Box]] = {
    0: [],
    # Two obstacles form a gate; the third forces a recovery and a second correction.
    1: [_box(2.0, 16.5), _box(5.0, 8.0), _box(7.2, 15.7)],
    # Alternating offsets: sequential avoidance without excessive oscillation.
    2: [_box(1.5, 8.5), _box(6.0, 17.0), _box(7.2, 9.3)],
    # Side obstacles plus a centreline one: obstacle vs boundary trade-off.
    3: [_box(1.5, 8.5), _box(5.5, 17.0), _box(6.0, 9.3)],
    4: [_box(5, 8), _box(8, 15), _box(3, 18)],
    5: [_rect(3.5, 15, half_x=3.0, half_y=0.5 * OBSTACLE_SIZE), _box(9, 15)],
    6: [],
    7: [],
    8: [],
    9: [],
    # Close to the centre path; the gap encourages a controlled pass.
    10: [_box(4.45, 11.0), _box(5.55, 14.0)],
    # Near but not on the path: a small lateral correction should suffice.
    11: [_box(3.8, 10.5), _box(6.2, 14.5), _box(4.1, 18.0)],
    # Narrow centred gate; the vessel should pass between the obstacles.
    12: [_box(3.7, 12.5), _box(6.3, 12.5)],
    # One obstacle near the bend of the curved path.
    13: [_box(5.8, 13.0)],
    14: [_box(5.0, 7.5)],
    15: [_box(5.0, 17.5)],
    # Controlled slalom along the centre path.
    16: [_box(4.0, 8.0), _box(6.0, 13.0), _box(4.0, 18.0)],
    # Off-path distractors; neither should trigger a wide detour.
    17: [_box(2.7, 12.5)],
    18: [_box(7.3, 12.5)],
    19: [],
    20: [],
}

# Curved reference paths authored for cases 8, 9 and 13.  Kept as data: the
# environment currently generates a straight path for every test case, so these
# cases run with a straight start-to-goal reference until it consumes them.
CURVED_WAYPOINTS: Dict[int, List[Tuple[float, float]]] = {
    8: [(5.0, 2.0), (3.4, 7.5), (6.6, 15.5), (5.0, 22.0)],
    9: [(5.0, 2.0), (7.0, 8.0), (7.2, 15.0), (5.0, 22.0)],
    13: [(5.0, 2.0), (3.6, 8.0), (6.4, 15.0), (5.0, 22.0)],
}


class TestCase:
    """Look-up of start/goal and obstacles for a deterministic case id."""

    def __init__(self, recorded_scenario: str = RECORDED_SCENARIO) -> None:
        self.recorded_scenario = recorded_scenario

    def _load_recorded(self) -> dict:
        with open(self.recorded_scenario, "r") as f:
            return json.load(f)

    def position(self, test_case: int) -> Tuple[float, float, float, float]:
        if test_case == 99:
            data = self._load_recorded()
            return (*data["start"], *data["goal"])
        if test_case not in POSITIONS:
            raise ValueError(f"Invalid test case: {test_case}")
        return POSITIONS[test_case]

    def obstacles(self, test_case: int) -> List[Box]:
        if test_case == 99:
            return self._load_recorded()["obstacles"]
        if test_case not in OBSTACLES:
            raise ValueError(f"Invalid test case: {test_case}")
        # Copy so callers (and the env's rescaling) cannot mutate the table.
        return [list(obs) for obs in OBSTACLES[test_case]]
