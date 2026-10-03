"""Oracle feasibility of an episode: can a controller that knows the future solve it? (2026-10-03)

The user (2026-10-03): near-impossible episodes should not be in the test set or
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

    path follower at cruise until t0  ->  hold heading(t0) + dpsi at throttle tau for D s
    ->  path follower at cruise to the end

over t0 in `T0_S`, dpsi in `DPSI_DEG`, tau in `TAU`, D in `DURATION_S`.  An
episode is solved by a manoeuvre that reaches the goal without contact within
the episode cap.  Physics, the rudder command limiter, the target model and the
collision test are `env.step`'s own; perception, observation and reward are
skipped (they do not change the motion).  No COLREGs condition: this measures
whether the episode can be survived at all.

It is a **lower bound** on solvability: a finite library can miss a solution
that needs, say, two successive alterations.  Hence the outputs are graded, not
just a verdict:
* `n_success` / `success_share` -- how much of the library works (solution-space size);
* `best_clearance_m` -- the widest minimum clearance of any solving manoeuvre;
* `latest_start_s` -- the latest t0 from which some manoeuvre still works: when
  the outcome is decided;
* success by family: port turns, starboard turns, speed only.

The vessel model's RK4 integration is most of the cost (about 5 ms a decision),
so `assess` screens by default: it skips the library when the path follower
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
import math
import time
from typing import Dict, Optional

import numpy as np

import constants as cfg
import ship as shipmod
import targets as tgtmod

T0_S = (0.0, 2.0, 4.0, 6.0, 8.0, 11.0, 14.0, 18.0)   # manoeuvre start times
DPSI_DEG = (-70.0, -40.0, -20.0, 0.0, 20.0, 40.0, 70.0)  # course alteration; + is starboard
TAU = (-1.0, -0.5, 0.0, 1.0)                          # throttle while manoeuvring (0 = cruise)
DURATION_S = (4.0, 8.0, 14.0)                          # how long the alteration is held
HEADING_GAIN_DEG = 20.0                               # rudder = heading error / 20 deg (common.colregs_action)
GAP_CHECK_M = 3.0                                     # polygon gaps only for objects whose centre is this close
MARGIN_M = 0.20                                       # a solution "with margin" keeps this clearance throughout
STOP_AFTER = 12                                       # screening: stop once this many solutions with margin exist


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def follower(env, throttle: float = 0.0):
    """LOS path follower (tools/tiers/common.follower_action), any throttle."""
    rudder = float(np.clip(env.lookahead_course_error / 25.0 + 0.4 * env.cross_track_error, -1.0, 1.0))
    return rudder, float(throttle)


def hold(env, heading_deg: float, throttle: float):
    return float(np.clip(_wrap180(heading_deg - env.asv_h) / HEADING_GAIN_DEG, -1.0, 1.0)), float(throttle)


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


def manoeuvre(dpsi: float, tau: float, duration_s: float):
    n = int(round(duration_s / cfg.UPDATE_RATE))
    state = {}

    def plan(env, k):
        if k == 0:
            state["psi"] = float(env.asv_h) + dpsi
        if k < n:
            return hold(env, state["psi"], tau)
        return follower(env, 0.0)
    return plan


def library(t0s=T0_S, dpsis=DPSI_DEG, taus=TAU, durations=DURATION_S):
    """(t0, dpsi, tau, D) in start-time order; holding course at cruise once per t0."""
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
    `STOP_AFTER` solutions keep `MARGIN_M`, and skips the library when the path
    follower alone (standing on at cruise) finishes with that margin; `full=True`
    runs the whole library (needed for `latest_start_s`)."""
    t_wall = time.time()
    lib = library() if lib is None else lib
    env.reset(seed=seed, options={"generated": built})
    env.estop_enabled = False
    nominal = _run(copy.deepcopy(env), lambda e, k: follower(e, 0.0))
    nominal_ok = nominal["outcome"] == "goal" and nominal["clearance"] >= MARGIN_M
    rows, complete = [], True
    if nominal_ok and not full:
        complete = False
    else:
        pre, k_done, n_margin, cut = env, 0, 0, False
        for t0 in sorted({m[0] for m in lib}):
            k0 = int(round(t0 / cfg.UPDATE_RATE))
            end = None
            while k_done < k0:
                end = physics_step(pre, *follower(pre, 0.0))
                k_done += 1
                if end is not None:
                    break
            if end is not None:               # the follower's run ended before t0: no later start
                break
            for (_, dpsi, tau, dur) in (m for m in lib if m[0] == t0):
                res = _run(copy.deepcopy(pre), manoeuvre(dpsi, tau, dur))
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
    fam = lambda sel: int(sum(1 for r in ok if sel(r)))
    best = max(ok, key=lambda r: r[5]) if ok else None
    outcomes = {}
    for r in rows:
        outcomes[r[6]] = outcomes.get(r[6], 0) + 1
    return {
        "nominal_outcome": nominal["outcome"], "nominal_clearance_m": nominal["clearance"],
        "complete": complete, "n_rollouts": len(rows), "n_success": len(ok), "n_success_margin": len(okm),
        "success_share": len(ok) / max(len(rows), 1),
        "best_clearance_m": (best[5] if best else (nominal["clearance"] if nominal["outcome"] == "goal" else 0.0)),
        "best_plan": (f"t0={best[0]:g} dpsi={best[1]:+g} tau={best[2]:+g} D={best[3]:g}" if best
                      else ("follower" if nominal["outcome"] == "goal" else "")),
        "latest_start_s": max(r[0] for r in ok) if ok else float("nan"),
        "latest_start_margin_s": max(r[0] for r in okm) if okm else float("nan"),
        "n_success_port": fam(lambda r: r[1] < 0), "n_success_starboard": fam(lambda r: r[1] > 0),
        "n_success_speed": fam(lambda r: r[1] == 0),
        "n_success_t0_0": fam(lambda r: r[0] == 0.0),
        "fail_target": outcomes.get("target", 0), "fail_obstacle": outcomes.get("obstacle", 0),
        "fail_boundary": outcomes.get("boundary", 0), "fail_timeout": outcomes.get("timeout", 0),
        "oracle_s": time.time() - t_wall,
    }
