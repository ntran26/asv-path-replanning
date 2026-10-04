"""Oracle feasibility of an episode: can a controller that knows the future solve it? (2026-10-03)

Decision (2026-10-03): near-impossible episodes should not be in the test set or
the development set, or only as a very small share, so they neither confuse the
policy nor mislead readers of the statistics.  `feasibility_st` is not enough to
find them: its own ship is a holonomic point (any direction, instantly), so it
passes encounters that a vessel with a turning circle and slow acceleration can
no longer escape.

**The oracle.**  From the episode's true initial state (same reset, same seed),
it rolls out a fixed library of manoeuvres in a copy of the environment, with
perfect foresight: the copy moves the target exactly as the episode will
(constant velocity, speed profile, reaction to the own ship, corridor clamp),
the own ship uses the true vessel model and the policy's own action space
(rudder and throttle in [-1, 1]; no astern -- only the safety layer has that).
Each manoeuvre is:

    route follower at cruise until t0
    ->  alteration: hold heading(t0) + dpsi at throttle tau      (dpsi != 0)
        or speed change: route follower at throttle tau          (dpsi == 0)
        for at least D s, and until every target has passed (TCPA <= 0 and
        `CLEAR_RANGE_M` away) or is `FAR_RANGE_M` away (at most `MAX_HOLD_S`)
    ->  route follower at cruise to the end

over t0 in `T0_S`, dpsi in `DPSI_DEG`, tau in `TAU`, D in `DURATION_S`.  The
**route follower** pursues an A* route from the start to the goal that keeps
`ROUTE_PANEL_M` off the panels and `ROUTE_WALL_M` off the walls (string-pulled,
pure pursuit `LOOKAHEAD_M` ahead), so the static layout alone never defeats the
oracle.  An episode is solved by a manoeuvre that reaches the goal without
contact within the episode cap.  Physics, the rudder command limiter, the target
model and the collision test are `env.step`'s own; perception, observation and
reward are skipped (they do not change the motion).  No COLREGs condition: this
measures whether the episode can be survived at all.

(v1 returned to the reference path on a timer and ignored the panels while
following it; on the development side SAC solved 9 of the 12 episodes v1 called
unsolvable -- turning back into the crossing target, and a panel on the path,
were the misses.  v1's rows: `results/feasibility/oracle_dev_v1_weak.csv`.)

It is a **lower bound** on solvability: a finite library can miss a solution
that needs, say, two successive alterations.  Hence the outputs are graded, not
just a verdict:
* `n_success` / `success_share` -- how much of the library works (solution-space size);
* `best_clearance_m` -- the widest minimum clearance of any solving manoeuvre;
* `latest_start_s` -- the latest t0 from which some manoeuvre still works: when
  the outcome is decided;
* success by family: port turns, starboard turns, speed only.

**Physically solvable is not the same as solvable onboard.**  A controller
cannot react before its perception tracks the target, and in the dense layouts
a panel can hide the target until late.  `track_time` runs the route follower
through the full `env.step` (LiDAR and tracker) to find when the target is
first tracked, and the library is also started at that time.  An episode is
then graded twice: `solved` (any start) and `solved_after_track` (only starts
at or after tracking).  An episode solvable only by manoeuvres that begin
before the target can be seen is near-impossible for any sensing controller;
`decision_window_s` (latest feasible start minus tracking time, full library
only) says how long an onboard controller has to decide.

The vessel model's RK4 integration is most of the cost (about 5 ms a decision),
so `assess` screens by default: it skips the library when the route follower
alone finishes with `MARGIN_M` clearance, and stops once `STOP_AFTER` solutions
keep that margin -- such an episode is clearly solvable and its counts are lower
bounds (`complete` is False).  Hard episodes, the ones the verdict is about,
always run the whole library; `full=True` runs it for every episode.
Checked against `env.step`: the same commands give the same own-ship and target
trajectories to 1e-7 m.

    import oracle_feasibility as of
    rep = of.assess(env, built, seed)              # env: an ASVLidarEnv; resets it
    rep = of.assess(env, built, seed, full=True)   # whole library: latest_start_s exact
"""
from __future__ import annotations

import copy
import heapq
import math
import time
from typing import Dict, List, Optional

import numpy as np

import constants as cfg
import feasibility as feas
import ship as shipmod
import targets as tgtmod

T0_S = (0.0, 2.0, 4.0, 6.0, 8.0, 11.0, 14.0, 18.0)   # manoeuvre start times
DPSI_DEG = (-70.0, -40.0, -20.0, 0.0, 20.0, 40.0, 70.0)  # course alteration; + is starboard; 0 = speed only
TAU = (-1.0, -0.5, 0.0, 1.0)                          # throttle while manoeuvring (0 = cruise)
DURATION_S = (4.0, 8.0, 14.0)                          # the least time the manoeuvre is held
HEADING_GAIN_DEG = 20.0                               # rudder = heading error / 20 deg (common.colregs_action)
GAP_CHECK_M = 3.0                                     # polygon gaps only for objects whose centre is this close
MARGIN_M = 0.20                                       # a solution "with margin" keeps this clearance throughout
STOP_AFTER = 12                                       # screening: stop once this many solutions with margin exist
ROUTE_PANEL_M = 0.80                                  # route clearance from panels (hull sweep while turning)
ROUTE_WALL_M = 0.60                                   # route clearance from walls
LOOKAHEAD_M = 1.5                                     # pure pursuit distance along the route
CLEAR_RANGE_M = 1.5                                   # a passed target must also be this far before returning
FAR_RANGE_M = 6.0                                     # ... or simply this far
MAX_HOLD_S = 30.0                                     # the longest a manoeuvre is held


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def follower(env, throttle: float = 0.0):
    """LOS reference-path follower (tools/tiers/common.follower_action), any throttle."""
    rudder = float(np.clip(env.lookahead_course_error / 25.0 + 0.4 * env.cross_track_error, -1.0, 1.0))
    return rudder, float(throttle)


def hold(env, heading_deg: float, throttle: float):
    return float(np.clip(_wrap180(heading_deg - env.asv_h) / HEADING_GAIN_DEG, -1.0, 1.0)), float(throttle)


def _astar(free: np.ndarray, s, g) -> Optional[List]:
    nx, ny = free.shape
    moves = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    best, parent = {s: 0.0}, {s: None}
    frontier = [(math.hypot(g[0] - s[0], g[1] - s[1]), 0.0, s)]
    while frontier:
        _, cost, c = heapq.heappop(frontier)
        if c == g:
            path = []
            while c is not None:
                path.append(c)
                c = parent[c]
            return path[::-1]
        if cost > best.get(c, math.inf):
            continue
        for dx, dy, step in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n]:
                continue
            if dx and dy and not (free[c[0] + dx, c[1]] and free[c[0], c[1] + dy]):
                continue
            new = cost + step
            if new < best.get(n, math.inf):
                best[n], parent[n] = new, c
                heapq.heappush(frontier, (new + math.hypot(g[0] - n[0], g[1] - n[1]), new, n))
    return None


class Route:
    """A* route around the panels, string-pulled, followed by pure pursuit.
    `points` is None when no route exists at either inflation; the
    reference-path follower is used then."""

    def __init__(self, env):
        self.points = None
        start, goal = (float(env.asv_x), float(env.asv_y)), (float(env.goal_x), float(env.goal_y))
        nav = list(env.boundary_polygon)
        for panel, wall in ((ROUTE_PANEL_M, ROUTE_WALL_M),
                            (cfg.FEASIBILITY_OBSTACLE_INFLATION_M, cfg.FEASIBILITY_WALL_INFLATION_M)):
            free, (x0, y0, res) = feas.free_grid(nav, [list(o) for o in env.obstacles], panel=panel, wall=wall)

            def cell(q):
                return (int(np.clip(round((q[0] - x0) / res), 0, free.shape[0] - 1)),
                        int(np.clip(round((q[1] - y0) / res), 0, free.shape[1] - 1)))
            s, g = cell(start), cell(goal)
            grid = free.copy()
            grid[s] = grid[g] = True                  # the start and goal themselves are valid by construction
            path = _astar(grid, s, g)
            if path is None:
                continue
            pts = [(x0 + res * i, y0 + res * j) for i, j in path]
            self.points = np.asarray(self._pull(grid, (x0, y0, res), [start] + pts[1:-1] + [goal]), dtype=float)
            break
        if self.points is not None:
            self.seg = np.diff(self.points, axis=0)
            self.len = np.hypot(self.seg[:, 0], self.seg[:, 1])
            self.s = np.r_[0.0, np.cumsum(self.len)]

    @staticmethod
    def _pull(grid, frame, pts):
        x0, y0, res = frame

        def free_line(a, b):
            n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * res)) + 1)
            for t in np.linspace(0.0, 1.0, n):
                i = int(round((a[0] + t * (b[0] - a[0]) - x0) / res))
                j = int(round((a[1] + t * (b[1] - a[1]) - y0) / res))
                if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]) or not grid[i, j]:
                    return False
            return True
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not free_line(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def action(self, env, throttle: float = 0.0):
        if self.points is None:
            return follower(env, throttle)
        p = np.array([env.asv_x, env.asv_y])
        rel = p - self.points[:-1]
        t = np.clip(np.einsum("ij,ij->i", rel, self.seg) / np.maximum(self.len ** 2, 1e-12), 0.0, 1.0)
        foot = self.points[:-1] + t[:, None] * self.seg
        k = int(np.argmin(np.hypot(foot[:, 0] - p[0], foot[:, 1] - p[1])))
        s_aim = min(self.s[k] + t[k] * self.len[k] + LOOKAHEAD_M, self.s[-1])
        m = int(np.clip(np.searchsorted(self.s, s_aim) - 1, 0, len(self.len) - 1))
        aim = self.points[m] + (s_aim - self.s[m]) / max(self.len[m], 1e-12) * self.seg[m]
        return hold(env, math.degrees(math.atan2(aim[0] - p[0], aim[1] - p[1])), throttle)


def targets_clear(env) -> bool:
    """Every target has passed (true TCPA <= 0, `CLEAR_RANGE_M` away) or is `FAR_RANGE_M` away."""
    v_own = env._own_velocity()
    for t in env.targets:
        p = np.array([t.x - env.asv_x, t.y - env.asv_y])
        rng = float(np.hypot(p[0], p[1]))
        if rng >= FAR_RANGE_M:
            continue
        v = np.asarray(t.velocity, dtype=float) - v_own
        vv = float(v @ v)
        tcpa = -float(p @ v) / vv if vv > 1e-9 else 0.0
        if tcpa <= 0.0 and rng >= CLEAR_RANGE_M:
            continue
        return False
    return True


def clearance(env) -> float:
    """Smallest gap from the own hull to a target hull, a panel or the wall, m."""
    from env import _polygon_gap
    hull = env.hull_polygon()
    best = float(env._border_clearance(hull))
    for t in env.targets:
        if math.hypot(t.x - env.asv_x, t.y - env.asv_y) < GAP_CHECK_M:
            best = min(best, _polygon_gap(hull, t.hull())[0])
    for obs in env.obstacles:
        c = np.mean(np.asarray(obs, dtype=float), axis=0)
        if math.hypot(c[0] - env.asv_x, c[1] - env.asv_y) < GAP_CHECK_M + 1.0:
            best = min(best, _polygon_gap(hull, obs)[0])
    return best


def physics_step(env, rudder_cmd: float, throttle_cmd: float) -> Optional[str]:
    """`env.step`'s command, physics, target motion and collision test, nothing else.
    Returns "goal", "timeout", a collision kind, or None to continue."""
    env.elapsed_time += cfg.UPDATE_RATE
    commanded = float(np.clip(rudder_cmd, -1.0, 1.0)) * 100.0
    if env.command_rate_limit:
        max_step = shipmod.COMMAND_RATE_PCT_S * cfg.UPDATE_RATE
        commanded = env.rudder + float(np.clip(commanded - env.rudder, -max_step, max_step))
    env.rudder = commanded
    env.rpm = cfg.CRUISE_RPM if cfg.FIXED_RPM else float(np.clip(
        cfg.CRUISE_RPM + cfg.RPM_DELTA * float(np.clip(throttle_cmd, -1.0, 1.0)), cfg.RPM_FLOOR, cfg.RPM_CEIL))
    x_before, y_before = env.asv_x, env.asv_y
    n_sub = max(1, int(round(cfg.UPDATE_RATE / cfg.PHYSICS_DT)))
    h = cfg.UPDATE_RATE / n_sub
    for _ in range(n_sub):
        dx, dy, heading, yaw_rate = env.model.update(env.rpm, env.rudder, h)
        env.asv_x += dx
        env.asv_y += dy
        env.asv_h = heading
        env.asv_w = yaw_rate
        env.u_body = env.model.u
        env.v_body = env.model.v
        own_state = {"x": env.asv_x, "y": env.asv_y, "velocity": env._own_velocity(), "heading": env.asv_h}
        for target in env.targets:
            target.step(h, own=own_state)
            tgtmod.clamp_to_corridor(target, env._confine_geom or env.channel,
                                     env._confine_poly or env.boundary_polygon)
        kind = env.collision_kind(env.hull_polygon())
        if kind is not None:
            return kind
    mx, my = env.asv_x - x_before, env.asv_y - y_before
    speed = math.hypot(mx, my) / cfg.UPDATE_RATE
    env._update_path_errors(math.degrees(math.atan2(mx, my)) if speed > 1e-6 else env.asv_h)
    env.distance_to_goal = float(np.hypot(env.asv_x - env.goal_x, env.asv_y - env.goal_y))
    env.step_count += 1
    if env._reached_goal():
        return "goal"
    if env.step_count >= cfg.MAX_EPISODE_STEPS:
        return "timeout"
    return None


def _run(env, plan) -> Dict:
    """Roll `plan(env, k)` (k = steps since the copy was taken) to the end."""
    k, clear = 0, float("inf")
    while True:
        r, tau = plan(env, k)
        end = physics_step(env, r, tau)
        k += 1
        if end is not None and end != "goal":
            return {"outcome": end, "clearance": 0.0 if end not in ("timeout",) else clear, "steps": k}
        clear = min(clear, clearance(env))
        if end == "goal":
            return {"outcome": "goal", "clearance": clear, "steps": k}


def manoeuvre(route: Route, dpsi: float, tau: float, duration_s: float):
    n_min = int(round(duration_s / cfg.UPDATE_RATE))
    n_max = int(round(MAX_HOLD_S / cfg.UPDATE_RATE))
    state = {"done": False}

    def plan(env, k):
        if k == 0:
            state["psi"] = float(env.asv_h) + dpsi
        if not state["done"] and (k < n_min or (k < n_max and not targets_clear(env))):
            return route.action(env, tau) if dpsi == 0.0 else hold(env, state["psi"], tau)
        state["done"] = True
        return route.action(env, 0.0)
    return plan


def _snap_time(t: float) -> float:
    """A start time on the decision grid, no earlier than `t`."""
    return math.ceil(float(t) / cfg.UPDATE_RATE - 1e-9) * cfg.UPDATE_RATE


def track_time(env, route: Route):
    """When the onboard perception first tracks a target while the own ship follows
    the route at cruise: (seconds, range m).  A full `env.step` run (LiDAR,
    tracker) on a copy, so occlusion by the panels counts.  0 s with no target;
    inf if it is never tracked before the run ends."""
    if not env.targets:
        return 0.0, float("nan")
    e = copy.deepcopy(env)
    k = 0
    while True:
        r, tau = route.action(e, 0.0)
        _, _, term, trunc, _ = e.step(np.array([r, tau], dtype=np.float32))
        k += 1
        if e.encounter_contexts:
            t = e.targets[0]
            return k * cfg.UPDATE_RATE, float(math.hypot(t.x - e.asv_x, t.y - e.asv_y))
        if term or trunc:
            return float("inf"), float("nan")


def library(t0s=T0_S, dpsis=DPSI_DEG, taus=TAU, durations=DURATION_S):
    """(t0, dpsi, tau, D) in start-time order; the route at cruise once per t0."""
    out = []
    for t0 in t0s:
        for dpsi in dpsis:
            for tau in taus:
                for dur in durations:
                    if dpsi == 0.0 and tau == 0.0 and dur != durations[0]:
                        continue
                    out.append((t0, dpsi, tau, dur))
    return out


def assess(env, built, seed: int, *, full: bool = False, lib=None) -> Dict:
    """Oracle report for one episode.  `full=False` (screening) stops once
    `STOP_AFTER` solutions keep `MARGIN_M`, and skips the library when the route
    follower alone (standing on at cruise) finishes with that margin; `full=True`
    runs the whole library (needed for `latest_start_s`)."""
    t_wall = time.time()
    lib = library() if lib is None else lib
    env.reset(seed=seed, options={"generated": built})
    env.estop_enabled = False
    route = Route(env)
    nominal = _run(copy.deepcopy(env), lambda e, k: route.action(e, 0.0))
    nominal_ok = nominal["outcome"] == "goal" and nominal["clearance"] >= MARGIN_M
    t_track, track_range = track_time(env, route)
    rows, complete = [], True
    if nominal_ok and not full:
        complete = False
    else:
        # The route follower's own run, a snapshot at every start time (it may end first).
        # The library is also started at the tracking time itself, the earliest an
        # onboard controller can react.
        if math.isfinite(t_track) and _snap_time(t_track) not in {m[0] for m in lib}:
            lib = lib + [(_snap_time(t_track), d, tau, dur) for (t0, d, tau, dur) in lib if t0 == lib[0][0]]
        t0s = sorted({m[0] for m in lib})
        snaps, pre, k_done = {}, env, 0
        for t0 in t0s:
            end = None
            while k_done < int(round(t0 / cfg.UPDATE_RATE)):
                end = physics_step(pre, *route.action(pre, 0.0))
                k_done += 1
                if end is not None:
                    break
            if end is not None:
                break
            snaps[t0] = copy.deepcopy(pre)
        # Starts after the target is tracked first: if they already give STOP_AFTER
        # solutions with margin, both verdicts are settled.
        order = [t for t in snaps if t >= t_track] + [t for t in snaps if t < t_track]
        n_margin, cut = 0, False
        for t0 in order:
            for (_, dpsi, tau, dur) in (m for m in lib if m[0] == t0):
                res = _run(copy.deepcopy(snaps[t0]), manoeuvre(route, dpsi, tau, dur))
                rows.append((t0, dpsi, tau, dur, res["outcome"] == "goal", res["clearance"], res["outcome"]))
                n_margin += int(rows[-1][4] and rows[-1][5] >= MARGIN_M)
                if not full and n_margin >= STOP_AFTER:
                    cut = True
                    break
            if cut:
                complete = False
                break
    ok = [r for r in rows if r[4]]
    okm = [r for r in ok if r[5] >= MARGIN_M]
    ok_t = [r for r in ok if r[0] >= t_track]
    okm_t = [r for r in okm if r[0] >= t_track]
    # The route follower alone needs no reaction, so it counts after tracking too.
    solved_t = bool(ok_t) or nominal["outcome"] == "goal"
    margin_t = bool(okm_t) or nominal_ok
    fam = lambda sel: int(sum(1 for r in ok if sel(r)))
    best = max(ok, key=lambda r: r[5]) if ok else None
    outcomes = {}
    for r in rows:
        outcomes[r[6]] = outcomes.get(r[6], 0) + 1
    latest = max(r[0] for r in ok) if ok else float("nan")
    return {
        "oracle_version": 2, "route": route.points is not None,
        "nominal_outcome": nominal["outcome"], "nominal_clearance_m": nominal["clearance"],
        "t_track_s": t_track, "track_range_m": track_range,
        "n_success_after_track": len(ok_t), "solved_after_track": solved_t, "margin_after_track": margin_t,
        "decision_window_s": (latest - t_track) if (ok and math.isfinite(t_track) and complete) else float("nan"),
        "complete": complete, "n_rollouts": len(rows), "n_success": len(ok), "n_success_margin": len(okm),
        "success_share": len(ok) / max(len(rows), 1),
        "best_clearance_m": (best[5] if best else (nominal["clearance"] if nominal["outcome"] == "goal" else 0.0)),
        "best_plan": (f"t0={best[0]:g} dpsi={best[1]:+g} tau={best[2]:+g} D={best[3]:g}" if best
                      else ("route" if nominal["outcome"] == "goal" else "")),
        "latest_start_s": latest,
        "latest_start_margin_s": max(r[0] for r in okm) if okm else float("nan"),
        "n_success_port": fam(lambda r: r[1] < 0), "n_success_starboard": fam(lambda r: r[1] > 0),
        "n_success_speed": fam(lambda r: r[1] == 0),
        "n_success_t0_0": fam(lambda r: r[0] == 0.0),
        "fail_target": outcomes.get("target", 0), "fail_obstacle": outcomes.get("obstacle", 0),
        "fail_boundary": outcomes.get("boundary", 0), "fail_timeout": outcomes.get("timeout", 0),
        "oracle_s": time.time() - t_wall,
    }
