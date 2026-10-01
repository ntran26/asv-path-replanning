"""Space-time solvability of an encounter among static panels (baseline-v3, 2026-09-28).

F74's A* filter guarantees a route around the *static* obstacles.  Once panels
may sit beside the encounter (baseline-v3 relaxes the CPA guard, and the field
layouts have none), that is no longer enough: a moving target can sweep the only
gap at the moment the own ship needs it.  This check asks the stronger question
-- does *some* trajectory reach the goal before the episode ends, keeping clear
of the panels and of the target at every instant?

**Method: forward reachability on a grid.**  The own ship is a point that can
move up to `speed` m/s in any direction or wait (holonomic), on the free-space
grid F74 already builds (walls and panels inflated for the hull).  Each step of
`dt` seconds the reachable set grows by `speed * dt`, and every cell within
`separation` of the target's position at that time is removed.  The scenario is
solvable if a cell within `GOAL_RADIUS` of the goal is reached by the horizon.

It is **optimistic** in one direction -- the real vessel has a turning radius,
cannot reverse and accelerates slowly -- and **conservative** in another -- the
target keeps `separation` (more than a hull length) from the own ship's centre.
It is a necessary-condition filter: it removes scenarios no controller could
solve; it does not promise the policy will.  The target moves as the scenario
defines it: constant heading, its speed profile (`T-VS`), and stopping short of
the wall (`stop_box`); reactive targets are not modelled (training targets
never react).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

import constants as cfg
import feasibility as feas
import targets as tgt
from corridor import nav_polygon

DT_S = 0.5                         # the decision interval
OWN_SPEED_MPS = 1.10               # the own ship's top speed (12 RPM, the July logs)
TARGET_SEPARATION_M = 1.3          # centre to centre; > one hull length (1.725 m) / 2 + beam, margin included
GOAL_TOL_M = cfg.GOAL_RADIUS


def target_track(built, horizon_s: float, dt: float = DT_S) -> np.ndarray:
    """(n, 2) target positions every `dt`, as the scenario moves it (no reaction)."""
    if built.encounter_class == "no_target":
        return np.zeros((0, 2))
    flags = built.flags or {}
    t = tgt.Target(float(built.target_spawn[0]), float(built.target_spawn[1]),
                   float(built.target_heading), float(built.target_speed),
                   speed_profile=tuple(flags["speed_profile"]) if flags.get("speed_profile") else None,
                   stop_box=tuple(flags["target_stop_box"]) if flags.get("target_stop_box") else None)
    out = [(t.x, t.y)]
    sub = 5
    for _ in range(int(round(horizon_s / dt))):
        for _ in range(sub):
            t.step(dt / sub)                        # own=None: no reaction
        out.append((t.x, t.y))
    return np.asarray(out)


def _dilate(mask: np.ndarray, r_cells: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r_cells):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def solvable(start: Tuple[float, float], goal: Tuple[float, float], panels: Sequence,
             built=None, *, nav: Optional[Sequence] = None, horizon_s: Optional[float] = None,
             speed: float = OWN_SPEED_MPS, separation: float = TARGET_SEPARATION_M,
             dt: float = DT_S, domain: bool = False) -> dict:
    """Is there a trajectory from `start` to `goal` clear of `panels` and of the
    target of `built` (if any) at every step?  Returns {"solvable", "t_goal_s",
    "static_route"}; `static_route` is F74's answer without the target.

    `domain=True` (rule-aware solvability, 2026-10-01): the own ship must also stay
    out of the target's asymmetric ship domain (fore 3.14 m, aft 1.57 m, abeam
    1.25 m -- the domain the reward and d_req use) at every step.  Longer ahead
    than astern, it forbids cutting close across the target's bow and allows a
    pass astern, so a case unsolvable with it has no encounter-consistent
    trajectory at all."""
    nav = list(nav_polygon((cfg.MAP_WIDTH, cfg.MAP_HEIGHT))) if nav is None else list(nav)
    horizon_s = cfg.MAX_EPISODE_STEPS * cfg.UPDATE_RATE if horizon_s is None else float(horizon_s)
    free, (x0, y0, res) = feas.free_grid(nav, [list(p) for p in panels])
    nx, ny = free.shape
    gx, gy = np.meshgrid(x0 + res * np.arange(nx), y0 + res * np.arange(ny), indexing="ij")
    goal_mask = np.hypot(gx - goal[0], gy - goal[1]) <= GOAL_TOL_M
    si = int(np.clip(round((start[0] - x0) / res), 0, nx - 1))
    sj = int(np.clip(round((start[1] - y0) / res), 0, ny - 1))
    reach = np.zeros_like(free)
    reach[si, sj] = True                           # the start itself (spawn is valid by construction)
    r_cells = max(1, int(round(speed * dt / res)))

    static = feas.layout_feasible(tuple(start), tuple(goal), nav, [list(p) for p in panels])
    track = target_track(built, horizon_s, dt) if built is not None else np.zeros((0, 2))
    if domain and len(track):
        h = np.radians(float(built.target_heading))
        fore, aft, abeam = cfg.DOMAIN_FORE, cfg.DOMAIN_AFT, cfg.DOMAIN_LATERAL
    steps = int(round(horizon_s / dt))
    for k in range(1, steps + 1):
        reach = _dilate(reach, r_cells) & free
        if len(track):
            px, py = track[min(k, len(track) - 1)]
            reach &= np.hypot(gx - px, gy - py) >= separation
            if domain:
                dx, dy = gx - px, gy - py
                along = dx * np.sin(h) + dy * np.cos(h)
                across = dx * np.cos(h) - dy * np.sin(h)
                a = np.where(along >= 0.0, fore, aft)
                reach &= (along / a) ** 2 + (across / abeam) ** 2 > 1.0
        if (reach & goal_mask).any():
            return {"solvable": True, "t_goal_s": k * dt, "static_route": bool(static)}
        if not reach.any():
            break
    return {"solvable": False, "t_goal_s": None, "static_route": bool(static)}


def scenario_solvable(built, panels: Sequence, **kw) -> dict:
    """`solvable` for a generated scenario: its own-ship spawn and leg goal."""
    goal = (built.flags or {}).get("basin_leg", [None, None])[1]
    if goal is None:
        pts = np.asarray(built.channel.reference_path_points(), dtype=float) if hasattr(built.channel, "reference_path_points") else None
        goal = tuple(pts[-1]) if pts is not None else None
    return solvable(tuple(map(float, built.own_spawn)), tuple(map(float, goal)), panels, built, **kw)
