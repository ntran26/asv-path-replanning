"""Fix 1 (2026-10-01): perceived static obstacles bound the room for a compliant turn."""
import numpy as np

import path as pathmod
from colregs import geometry as geo


def _straight():
    return pathmod.ReferencePath(pathmod.straight_points(5.0, 2.0, 5.0, 22.0)) \
        if hasattr(pathmod, "ReferencePath") else None


def test_obstacle_room_by_side_and_passage():
    import path as p
    cls = [getattr(p, n) for n in dir(p) if isinstance(getattr(p, n), type) and hasattr(getattr(p, n), "project")][0]
    leg = cls(p.straight_points(5.0, 2.0, 5.0, 22.0))          # heading north: starboard is +x
    pts = np.array([[6.2, 9.0], [6.0, 9.2],                     # starboard, ~1.0-1.2 m off, at s ~ 7 m
                    [3.0, 9.0],                                 # port, 2 m off
                    [7.0, 20.0]])                               # starboard but beyond the passage
    stbd, port = geo.obstacle_room(leg, pts, 5.0, 9.0, 0.0)
    assert abs(stbd - 1.0) < 0.05 and abs(port - 2.0) < 0.05
    stbd, port = geo.obstacle_room(leg, pts, 5.0, 9.0, 0.5)      # vessel 0.5 m to starboard
    assert abs(stbd - 0.5) < 0.05 and abs(port - 2.5) < 0.05
    assert geo.obstacle_room(leg, pts, 0.0, 3.0, 0.0) == (float("inf"), float("inf"))
    assert geo.obstacle_room(leg, None, 0.0, 20.0, 0.0) == (float("inf"), float("inf"))
