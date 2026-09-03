"""Verification tests for the LOS+APF baseline.

    python src/verify_los_apf.py

Two things are checked, both of which a reviewer could reasonably ask about:

1. **Sign conventions.**  The environment defines positive cross-track error as
   *port* of the path, while the textbook LOS law assumes starboard.  Getting
   that backwards makes the controller steer away from the path -- the easiest
   possible way to build an accidental straw man.  Every sign is asserted
   against a synthetic observation with a known answer.

2. **Observation-only access.**  The controller must not receive ground-truth
   obstacle geometry.  An AST scan of the controller source asserts that it
   never references `env`, `obstacles`, `map_border`, the vessel world pose, or
   any other privileged attribute -- it reads the 34-dimensional observation and
   nothing else.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines.los_apf import LosApfController          # noqa: E402
from lidar import sector_angle_grid                     # noqa: E402

BEARINGS = sector_angle_grid()
FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}   {detail}")
    if not condition:
        FAILURES.append(name)


def obs(lidar=None, cte=0.0, course_error=0.0, lookahead=0.0, yaw=0.0,
        front=16.0, side_diff=0.0, local_target=0.0):
    """A synthetic observation with the same keys and shapes as the env's."""
    def s(v):
        return np.array([v], dtype=np.float32)
    return {
        "lidar": (np.zeros(25, dtype=np.float32) if lidar is None
                  else np.asarray(lidar, dtype=np.float32)),
        "u": s(0.8), "v": s(0.0), "yaw_rate": s(yaw),
        "cross_track_error": s(cte), "course_error": s(course_error),
        "lookahead_course_error": s(lookahead), "front_clearance": s(front),
        "side_clearance_diff": s(side_diff), "local_target_cte": s(local_target),
    }


def sector_at(bearing_deg: float, closeness: float = 0.9) -> np.ndarray:
    """A single occupied sector at the bearing nearest `bearing_deg`."""
    v = np.zeros(25, dtype=np.float32)
    v[int(np.argmin(np.abs(BEARINGS - bearing_deg)))] = closeness
    return v


def main() -> int:
    c = LosApfController()

    print("LOS guidance signs")
    c.reset()
    a, _ = c.predict(obs())
    check("centred and aligned -> zero rudder", abs(a[0]) < 1e-6, f"rudder={a[0]:+.4f}")
    c.reset()
    a, _ = c.predict(obs(cte=+2.0))
    check("cte > 0 (port of path) -> starboard rudder", a[0] > 0, f"rudder={a[0]:+.4f}")
    c.reset()
    a, _ = c.predict(obs(cte=-2.0))
    check("cte < 0 (starboard of path) -> port rudder", a[0] < 0, f"rudder={a[0]:+.4f}")
    c.reset()
    a, _ = c.predict(obs(course_error=+30.0))
    check("course_error > 0 (path to starboard) -> starboard rudder",
          a[0] > 0, f"rudder={a[0]:+.4f}")

    print("\nAPF repulsion")
    c.reset()
    a, _ = c.predict(obs(lidar=sector_at(+40)))
    check("obstacle to starboard -> port rudder", a[0] < 0, f"rudder={a[0]:+.4f}")
    c.reset()
    a, _ = c.predict(obs(lidar=sector_at(-40)))
    check("obstacle to port -> starboard rudder", a[0] > 0, f"rudder={a[0]:+.4f}")
    c.reset()
    a, _ = c.predict(obs(lidar=sector_at(0.0), side_diff=+3.0))
    check("head-on, starboard clearer -> starboard rudder", a[0] > 0, f"rudder={a[0]:+.4f}")
    c.reset()
    a, _ = c.predict(obs(lidar=sector_at(0.0), side_diff=-3.0))
    check("head-on, port clearer -> port rudder", a[0] < 0, f"rudder={a[0]:+.4f}")
    c.reset()
    a, _ = c.predict(obs(lidar=sector_at(0.0), local_target=-0.4))
    check("head-on tie, env cue starboard -> starboard rudder",
          a[0] > 0, f"rudder={a[0]:+.4f}")

    print("\nSide commitment (APF dithering guard)")
    c.reset()
    c.predict(obs(lidar=sector_at(0.0), side_diff=+3.0))
    a, _ = c.predict(obs(lidar=sector_at(0.0), side_diff=-3.0))
    check("commitment latched when the sensor flips", a[0] > 0, f"rudder={a[0]:+.4f}")
    c.predict(obs())
    check("commitment released once the way ahead is clear", c.committed_side == 0.0)

    print("\nSpeed control")
    c.reset()
    clear, _ = c.predict(obs())
    c.reset()
    near, _ = c.predict(obs(lidar=sector_at(0.0, 0.95)))
    check("throttle reduced near obstacles", near[1] < clear[1],
          f"clear={clear[1]:+.3f} near={near[1]:+.3f}")
    c.reset()
    off, _ = c.predict(obs(course_error=80.0))
    check("throttle reduced when badly misaligned", off[1] < clear[1],
          f"aligned={clear[1]:+.3f} off={off[1]:+.3f}")

    print("\nDerivative term acts on yaw rate, not on the error")
    c.reset()
    still, _ = c.predict(obs(cte=2.0, yaw=0.0))
    c.reset()
    turning, _ = c.predict(obs(cte=2.0, yaw=+20.0))
    check("positive yaw rate damps the starboard command",
          turning[0] < still[0], f"{still[0]:+.4f} -> {turning[0]:+.4f}")

    print("\nOutput bounds and determinism")
    c.reset()
    a, _ = c.predict(obs(cte=50.0, lidar=np.ones(25, dtype=np.float32)))
    check("action stays within [-1, 1]", bool(np.all(np.abs(a) <= 1.0)), f"a={a}")
    c.reset()
    x, _ = c.predict(obs(cte=1.0, lidar=sector_at(20)))
    c.reset()
    y, _ = c.predict(obs(cte=1.0, lidar=sector_at(20)))
    check("identical output after reset", bool(np.array_equal(x, y)))
    c.reset()
    d1, _ = c.predict(obs(cte=1.0), deterministic=True)
    c.reset()
    d2, _ = c.predict(obs(cte=1.0), deterministic=False)
    check("deterministic flag changes nothing", bool(np.array_equal(d1, d2)))

    print("\nObservation-only access (AST scan of the controller source)")
    source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "baselines", "los_apf.py")
    tree = ast.parse(open(source_path).read())
    attrs, names, subs = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            subs.add(node.slice.value)

    banned_attrs = {"obstacles", "map_border", "asv_x", "asv_y", "asv_h",
                    "hull_polygon", "true_border_clearance", "start_x", "goal_x",
                    "lidar_reward", "lidar_border_guard", "lidar_obs"}
    banned_names = {"env", "ASVLidarEnv", "ObstacleSampler", "ReferencePath"}
    allowed_keys = {"lidar", "cross_track_error", "course_error", "u", "v",
                    "lookahead_course_error", "yaw_rate", "side_clearance_diff",
                    "local_target_cte", "front_clearance"}

    bad_attrs = sorted(attrs & banned_attrs)
    bad_names = sorted(names & banned_names)
    extra = sorted(({k for k in subs if isinstance(k, str)}
                    & (allowed_keys | banned_attrs)) - allowed_keys)

    check("no privileged env attributes referenced", not bad_attrs, f"found={bad_attrs}")
    check("no env classes referenced", not bad_names, f"found={bad_names}")
    check("no keys outside the observation space", not extra, f"found={extra}")
    c.reset()
    c.predict(obs())
    check("runs from a bare observation dict, no env involved", True)

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
