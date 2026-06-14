"""Small built-in scenario set used by rl_env.py and udp_live_rl.py.

Training and fixed-suite evaluation normally use random/generated scenarios,
so this file is mainly for quick visual tests with --test-case.
"""
from __future__ import annotations
from typing import List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]


def _box(cx: float, cy: float, size: float = 1.0) -> Polygon:
    h = 0.5 * size
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


class TestCase:
    def position(self, test_case: int = 0):
        # Coordinates are in the canonical 10 m x 25 m map. rl_env.py scales
        # them if a different map size is requested.
        cases = {
            0: (5.0, 2.0, 5.0, 22.0),
            1: (5.0, 2.0, 5.0, 22.0),
            2: (4.0, 2.0, 6.0, 22.0),
            3: (6.0, 2.0, 4.0, 22.0),
            4: (3.5, 2.0, 6.5, 22.0),
            5: (6.5, 2.0, 3.5, 22.0),
            6: (5.0, 2.0, 5.0, 22.0),
            7: (4.2, 2.0, 5.8, 22.0),
            99: (5.0, 2.0, 5.0, 22.0),
        }
        return cases.get(int(test_case), cases[0])

    def obstacles(self, test_case: int = 0) -> List[Polygon]:
        cases = {
            0: [],
            1: [_box(5.0, 11.0, 1.0)],
            2: [_box(4.3, 11.0, 1.0)],
            3: [_box(5.7, 11.0, 1.0)],
            4: [_box(5.0, 9.0, 1.0), _box(4.0, 14.0, 1.0)],
            5: [_box(5.0, 9.0, 1.0), _box(6.0, 14.0, 1.0)],
            6: [_box(5.0, 8.5, 1.0), _box(4.1, 13.0, 1.0), _box(5.9, 17.0, 1.0)],
            7: [_box(4.7, 8.0, 1.0), _box(5.6, 12.0, 1.0), _box(4.4, 16.0, 1.0)],
            99: [],
        }
        return cases.get(int(test_case), [])
