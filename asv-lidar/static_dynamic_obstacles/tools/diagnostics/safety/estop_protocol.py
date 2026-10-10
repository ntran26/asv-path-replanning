"""Evaluation-side rules of the emergency-stop protocol (planning/SAFETY_ESTOP_PROTOCOL.md).

The filter decides when to stop (src/safety_estop.py) and sets
``env.emergency_stop_committed``. This module applies the protocol's
environment rules from that moment, without changing env.py or its defaults:

* every target ship brakes in a straight line at TARGET_DECEL_MPS2 until at
  rest, ignoring its behaviour model (installed per target at reset, so the
  braking starts inside the same physics step as the commitment);
* the episode ends as ``stopped`` once the own ship is still (true planar
  speed and yaw rate below the thresholds); truth is used only here, for
  termination and scoring;
* a target contact after the commitment is ``collision:by_target``.
"""
from __future__ import annotations

import math

import safety_v2 as v2
import ship
import targets as tgtmod

# The simulator's strongest own-ship braking: reverse-thrust unit at efficiency 0.5.
TARGET_DECEL_MPS2 = 0.5 * ship.braking_thrust(v2.ASTERN_RPM, efficiency=1.0) / ship.M11
STILL_SPEED_MPS = 0.05
STILL_YAW_DEGPS = 1.0


def _braking_step(env, target, original):
    def step(dt, *, own=None):
        if not getattr(env, "emergency_stop_committed", False):
            return original(dt, own=own)
        target._t += float(dt)
        if target._stopped:
            return
        target.speed = max(0.0, float(target.speed) - TARGET_DECEL_MPS2 * float(dt))
        v = target.velocity
        nx, ny = target.x + float(v[0]) * dt, target.y + float(v[1]) * dt
        if target.stop_box is not None:
            x0, x1, y0, y1 = target.stop_box
            xs, ys = zip(*tgtmod.hull_polygon(nx, ny, target.heading))
            if min(xs) < x0 or max(xs) > x1 or min(ys) < y0 or max(ys) > y1:
                target.speed, target._stopped = 0.0, True
                return
        target.x, target.y = nx, ny
    return step


def install(env) -> None:
    """Call after reset: arm target braking and clear the protocol state."""
    env.emergency_stop_committed = False
    env.estop_protocol = {"commit_step": None, "outcome": None, "commit_speed_mps": None}
    for target in env.targets:
        target.step = _braking_step(env, target, target.step)


def still(env) -> bool:
    return (math.hypot(float(env.u_body), float(env.v_body)) <= STILL_SPEED_MPS
            and abs(float(env.asv_w)) <= STILL_YAW_DEGPS)


def after_step(env, result, step_index: int):
    """Apply termination and outcome rules; returns the (possibly terminated) step tuple."""
    obs, reward, terminated, truncated, info = result
    state = env.estop_protocol
    if not getattr(env, "emergency_stop_committed", False):
        return result
    if state["commit_step"] is None:
        state["commit_step"] = step_index
        state["commit_speed_mps"] = float(env.speed_mps)
    if info.get("collided"):
        state["outcome"] = ("collision:by_target" if info.get("collision_kind") == "target"
                            else f"collision:{info.get('collision_kind')}")
        return result
    if info.get("reached_goal"):
        state["outcome"] = "goal"
        return result
    if still(env):
        state["outcome"] = "stopped"
        return obs, reward, True, truncated, info
    if truncated:
        state["outcome"] = "timeout"
    return result
