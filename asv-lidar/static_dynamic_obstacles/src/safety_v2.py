"""Safety layer v2: a predictive safety filter around the policy (2026-10-01).

`planning/SAFETY_LAYER_V2_PLAN.md`.  Safety layer v1 (`emergency_stop.py`) fires
on a rule test -- a give-way encounter in extremis whose compliant turn is
inadmissible -- and can only stop.  On the seed-0 frozen suite it stopped mostly
narrow-channel overtakes that would have passed clear, lost about one point of
success, and never acted in the crossings and head-ons where target collisions
happen.  v2 asks a physical question instead, every step:

    *If the vessel commits to this action for `COMMIT_S`, is there still a
    recovery manoeuvre that keeps the hull clear for the rest of `HORIZON_S`?*

Prediction uses only what the policy has: the static LiDAR returns (the
tracked target's own removed) and the map boundary, the tracked target at
constant velocity, and the identified hull with its rudder delay and servo
(`classical/common.py`, shared with the DWA and VO comparators), plus the hull's
reverse-braking model (`ship.braking_thrust`) for the one action only the safety
layer has: full astern.

Iteration 2 (2026-10-01), after the first check steered the vessel off static
obstacles into walls and left target contacts unchanged:

* **braking** is a candidate and a recovery: full astern for the step, re-decided
  every step (iteration 3; at first it went through the v1 stop latch, which held
  the vessel stopped in a target's path);
* the look-ahead is `HORIZON_S` = 8 s (was 6);
* **hold back:** when nothing is safe, the policy's own action stands unless an
  alternative delays the first contact by `HOLD_BACK_GAIN_S`;
* **hysteresis:** for `HOLD_STEPS` after an intervention the override is kept
  while it stays safe and the policy's action is safe only by a hair.

If the policy's action passes, it is executed untouched.  Selected by
`constants.SAFETY_VERSION = 2` at run time (absent means v1), so the frozen
baselines and every v1 evaluation are unchanged.  Interventions are counted on
the environment (`safety_v2_steps`) and never credited to the policy.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

import constants as cfg
import emergency_stop as estop_mod
import ship as shipmod
from classical import common as cc

COMMIT_S = 1.0                       # the candidate is held this long ...
HORIZON_S = 8.0                      # ... then a recovery manoeuvre, to here
ENGAGE_RANGE_M = 7.0                 # filter only with something this close
GAP_STATIC_M = 0.05                  # hull (with its margin) to a static return
GAP_BOUNDARY_M = 0.02                # hull to the map polygon (exact, so tighter)
GAP_TARGET_M = 0.10                  # hull to the target hull
MARGIN_M = 0.10                      # "safe only by a hair" for the hysteresis
TRIGGER_MARGIN_M = 0.15              # the policy passes only with this much turning margin
ROOM_SLACK_M = 0.20                  # ... or with this little less than the best candidate
TERMINAL_M = 2.0                     # clear run required ahead of a hull still moving at the horizon ...
TERMINAL_S = 3.0                     # ... or this many seconds of its speed there, if less (a fixed
                                     # 2 m deadlocked a slow hull beside an obstacle: only braking passed)
TERMINAL_STOP_SPEED = 0.10           # m/s: slower than this counts as stopped
TERMINAL_BOUNDARY = False            # the run-out against the map boundary too (dev set: 105 -> 109 goals off)
RUDDERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
THROTTLES = (-1.0, 0.0, 1.0)         # propulsion floor, cruise, ceiling
BRAKE = None                         # marker: full astern through the latch
BRAKE_RUDDERS = (-1.0, 0.0, 1.0)
# Turning recoveries keep cruise propulsion: the rudder only bites in the
# propeller's wash, and coasting with the rudder hard over barely turns this hull.
RECOVERY = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0),        # (rudder, throttle) after the commit,
            (-1.0, BRAKE), (0.0, BRAKE), (1.0, BRAKE))    # or full astern ...
RECOVERY_TURN_S = 2.0                # ... held this long, then the rudder centred
MEMORY_S = 90.0                      # static returns are remembered the whole episode: inside the 1 m
                                     # dead zone at the bow the LiDAR loses an obstacle it is about to hit
W_THROTTLE = 2.0                     # distance to the policy action: rudder^2 + w * throttle^2 -- a turn
                                     # is preferred to cutting the throttle, which costs steerage
EGO_SMOOTHING = 0.3                  # EMA weight of a new ego sample: the measured yaw rate's noise
                                     # (1 deg/s) alone swings the predicted margin by +-0.4 m over 8 s
W_PREVIOUS = 0.3                     # ... plus this times the distance to the last override
HOLD_BACK_GAIN_S = 1.0
HOLD_STEPS = 2
ASTERN_RPM = estop_mod.s2_to_rpm(estop_mod.S2_FULL_ASTERN)
# Reverse thrust is **not identified**: no field log commands S2 < 0, and the
# simulator assumes 0.5 x the forward law with no reversal delay
# (`ship.REVERSE_THRUST_EFFICIENCY`, basin crash stops S1-C2 to measure it).  The
# filter's own prediction is pessimistic -- a quarter of the forward law, and
# nothing for the first 0.75 s (about the rudder's effective delay) -- so it only
# chooses a brake that would still work if reverse turns out weaker and slower.
BRAKE_EFFICIENCY = 0.25
BRAKE_DELAY_S = 0.75


def _rpm(throttle) -> np.ndarray:
    """Throttle to rpm-units; NaN (the brake marker) to full astern."""
    t = np.asarray(throttle, dtype=float)
    fwd = (np.full_like(t, cfg.CRUISE_RPM) if cfg.FIXED_RPM else
           np.clip(cfg.CRUISE_RPM + cfg.RPM_DELTA * np.nan_to_num(t), cfg.RPM_FLOOR, cfg.RPM_CEIL))
    return np.where(np.isnan(t), ASTERN_RPM, fwd)


def rollout(snap, act, controls_commit, controls_after, commit_s: float, horizon_s: float):
    """`classical.common.rollout` with reverse braking: columns whose rpm is negative
    coast on the identified dynamics and lose surge to `ship.braking_thrust`.  The
    recovery `controls_after` is held for `RECOVERY_TURN_S`, then its rudder is
    centred (holding hard rudder to the horizon would put the hull in a circle)."""
    n = len(controls_commit)
    decisions = int(math.ceil(horizon_s / cfg.UPDATE_RATE))
    commit_k = int(round(commit_s / cfg.UPDATE_RATE))
    turn_k = commit_k + int(round(RECOVERY_TURN_S / cfg.UPDATE_RATE))
    straight_after = controls_after.copy()
    straight_after[:, 0] = 0.0
    state = np.zeros((7, n))
    state[0:4] = np.array([max(0.0, snap.u), snap.v, snap.r, snap.heading])[:, None]
    state[4] = act.servo
    state[5], state[6] = snap.x, snap.y
    params = {k: np.full(n, v) for k, v in cc.IDENTIFIED.items()}
    pending = [np.full(n, d) for d in act.buffer] if act.buffer is not None else None
    K = decisions * cc.SUBSTEPS
    pos, hdg, spd = np.empty((K, n, 2)), np.empty((K, n)), np.empty((K, n))
    brake_decel = shipmod.braking_thrust(ASTERN_RPM, efficiency=BRAKE_EFFICIENCY) / shipmod.M11
    brake_since = np.full(n, np.inf)                 # time each column started braking
    j = 0
    for k in range(decisions):
        ctrl = controls_commit if k < commit_k else (controls_after if k < turn_k else straight_after)
        rpm = _rpm(ctrl[:, 1])
        delta = -cc.MAX_RUDDER_RAD * ctrl[:, 0]
        braking = rpm < 0.0
        if pending is None:
            pending = [delta.copy() for _ in range(cc.DELAY_STEPS)]
        for _ in range(cc.SUBSTEPS):
            pending.append(delta)
            state = cc.dyn.rk4_step(state, np.maximum(rpm, 0.0), pending.pop(0), params, cc.PRED_DT)
            t_now = (j + 1) * cc.PRED_DT
            brake_since = np.where(braking, np.minimum(brake_since, t_now), np.inf)
            active = braking & (t_now - brake_since >= BRAKE_DELAY_S)
            if active.any():
                state[0] = np.where(active, np.maximum(0.0, state[0] - brake_decel * cc.PRED_DT), state[0])
            pos[j] = state[5:7].T
            hdg[j] = state[3]
            spd[j] = np.hypot(state[0], state[1])
            j += 1
    return cc.Rollout(pos, hdg, spd, cc.PRED_DT * np.arange(1, K + 1))


class SafetyFilterV2:
    def __init__(self) -> None:
        self.perception = cc.Perception(memory_frames=int(round(MEMORY_S / cfg.UPDATE_RATE)))
        self.actuators = cc.Actuators()
        self.last = {}
        self.hold = 0
        self.previous = None              # the last override, (rudder, throttle or NaN)
        self.ego = None                   # smoothed (u, v, r)

    def _threat_in_reach(self, snap) -> bool:
        if any(np.hypot(*(t.position - snap.position)) < ENGAGE_RANGE_M for t in snap.tracks):
            return True
        if len(snap.points):
            rel = snap.points - snap.position
            ahead = rel @ np.array([math.sin(snap.heading), math.cos(snap.heading)])
            if np.any((np.linalg.norm(rel, axis=1) < ENGAGE_RANGE_M) & (ahead > -1.0)):
                return True
        pos = snap.position[None, None, :]
        return bool(cc.boundary_clearance(pos, np.array([[snap.heading]]),
                                          snap.edges_a, snap.edges_b)[0, 0] < 1.5)

    def _evaluate(self, snap, ro) -> Tuple[np.ndarray, np.ndarray]:
        """(time of first violation or inf, minimum clearance) per rollout column."""
        c = cc.point_clearance(ro.positions, ro.headings, snap.points,
                               reach=cc.HALF_L + 1.0) - GAP_STATIC_M
        c = np.minimum(c, cc.boundary_clearance(ro.positions, ro.headings,
                                                snap.edges_a, snap.edges_b) - GAP_BOUNDARY_M)
        for track in snap.tracks:
            c = np.minimum(c, cc.target_gap(ro.positions, ro.headings, ro.times, track) - GAP_TARGET_M)
        bad = c < 0.0
        first = np.where(bad.any(axis=0), ro.times[np.argmax(bad, axis=0)], np.inf)
        clear = c.min(axis=0)
        # Terminal check: a hull still under way at the horizon must have the next
        # TERMINAL_M clear along its heading.  Without it a fixed time horizon
        # rewards slowing -- less distance run, so the obstacle falls past the end.
        moving = ro.speeds[-1] > TERMINAL_STOP_SPEED
        if moving.any():
            run = np.minimum(TERMINAL_M, TERMINAL_S * ro.speeds[-1][moving])
            d = np.linspace(1 / 8, 1.0, 8)[:, None, None] * run[None, :, None]
            hdg = ro.headings[-1][moving]
            ext = ro.positions[-1][moving][None] + d * np.stack([np.sin(hdg), np.cos(hdg)], axis=1)[None]
            hk = np.broadcast_to(hdg, ext.shape[:2])
            ce = cc.point_clearance(ext, hk, snap.points, reach=cc.HALF_L + 1.0) - GAP_STATIC_M
            if TERMINAL_BOUNDARY:
                ce = np.minimum(ce, cc.boundary_clearance(ext, hk, snap.edges_a, snap.edges_b) - GAP_BOUNDARY_M)
            ce = ce.min(axis=0)
            cm = np.full(len(clear), np.inf)
            cm[moving] = ce
            first = np.where(np.isinf(first) & (cm < 0.0), ro.times[-1], first)
            clear = np.minimum(clear, cm)
        return first, clear

    def filter(self, env, action) -> Tuple[np.ndarray, bool]:
        """Return (action to execute, whether the filter changed it); a chosen brake
        is requested from the environment's stop latch."""
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        snap = self.perception.snapshot(env)
        raw = np.array([snap.u, snap.v, snap.r])
        self.ego = raw if self.ego is None else self.ego + EGO_SMOOTHING * (raw - self.ego)
        snap.u, snap.v, snap.r = (float(z) for z in self.ego)
        env._v2_brake = False
        if not self._threat_in_reach(snap):
            self.actuators.issue(env, float(action[0]))
            self.last, self.hold, self.previous = {"mode": "idle"}, 0, None
            return action.astype(np.float32), False

        grid = [(r, t) for r in RUDDERS for t in THROTTLES]
        brakes = [(r, np.nan) for r in BRAKE_RUDDERS]
        cands = np.array([tuple(action)] + grid + brakes, dtype=float)    # row 0 = policy
        if self.previous is not None:
            cands = np.vstack([cands, np.asarray(self.previous, dtype=float)[None, :]])
        rec = np.array([(r, np.nan if t is BRAKE else t) for r, t in RECOVERY], dtype=float)
        m, nrec = len(cands), len(rec)
        ro = rollout(snap, self.actuators, np.repeat(cands, nrec, axis=0), np.tile(rec, (m, 1)),
                     COMMIT_S, HORIZON_S)
        first, clear = self._evaluate(snap, ro)
        first, clear = first.reshape(m, nrec), clear.reshape(m, nrec)
        turn_rec = ~np.isnan(rec[:, 1])
        is_brake = np.isnan(cands[:, 1])
        if not any(np.hypot(*(t.position - snap.position)) < ENGAGE_RANGE_M for t in snap.tracks):
            # Braking only for traffic: a target moves on, a wall or obstacle stays put,
            # and with no astern motion a hull stopped beside one is trapped (the dev
            # set's boundary collisions, iteration 5).  It also leans less on the
            # unidentified reverse thrust.
            first[:, ~turn_rec] = 0.0
            first[is_brake, :] = 0.0
        safe_turn = np.isinf(first[:, turn_rec]).any(axis=1)                # a turning recovery exists
        safe = np.isinf(first).any(axis=1)                                  # some recovery, braking included
        margin = np.where(np.isinf(first[:, turn_rec]), clear[:, turn_rec], -np.inf).max(axis=1)
        latest = first.max(axis=1)

        def dist(rows):
            thr = np.where(np.isnan(cands[rows, 1]), -1.5, cands[rows, 1])  # a brake sits beyond the floor
            d = (cands[rows, 0] - action[0]) ** 2 + W_THROTTLE * (thr - action[1]) ** 2
            if self.previous is not None:
                prev_thr = -1.5 if np.isnan(self.previous[1]) else self.previous[1]
                d = d + W_PREVIOUS * ((cands[rows, 0] - self.previous[0]) ** 2
                                      + W_THROTTLE * (thr - prev_thr) ** 2)
            return d

        # Braking is the reserve: an action passes if a *turning* recovery keeps it
        # clear; a turn that keeps one is preferred to the policy's action when the
        # latter needs the brake; the brake itself only when no turn works.
        idx = np.arange(m)
        choice = 0
        if self.hold > 0 and self.previous is not None and safe_turn[-1] and margin[0] < TRIGGER_MARGIN_M:
            choice = m - 1                                                  # hysteresis: keep the override
                                                                            # until the policy's own action is roomy
        elif safe_turn[0] and margin[0] >= TRIGGER_MARGIN_M:
            choice = 0
        else:
            pool = idx[safe_turn & ~is_brake]
            if len(pool):
                # The nearest action whose margin is roomy, or near the best on offer
                # (insisting on the full trigger margin would only ever pick slowing,
                # which buys margin now and costs steerage later).
                floor = min(TRIGGER_MARGIN_M, float(margin[pool].max()) - ROOM_SLACK_M)
                pool = pool[margin[pool] >= floor]
                choice = int(pool[np.argmin(dist(pool))])                  # the policy itself, if it is in it
            elif safe[0]:
                choice = 0                                                  # the brake is still in reserve
            else:
                pool = idx[safe & is_brake]
                if len(pool):
                    choice = int(pool[np.argmin(dist(pool))])              # brake now
                else:
                    best = int(np.lexsort((-clear.max(axis=1), -latest))[0])
                    if latest[best] - latest[0] >= HOLD_BACK_GAIN_S:        # else hold back: the policy stands
                        choice = best
        changed = choice != 0 and not np.allclose(cands[choice], action, equal_nan=False)
        out = cands[choice].copy()
        brake = bool(np.isnan(out[1]))
        env._v2_brake = brake        # full astern this step only, under the filter's control (no latch)
        if brake:
            out[1] = -1.0
        out = out.astype(np.float32)
        self.actuators.issue(env, float(out[0]))
        if changed:
            self.previous, self.hold = (float(out[0]), np.nan if brake else float(out[1])), HOLD_STEPS
        else:
            self.hold = max(0, self.hold - 1)
            if self.hold == 0:
                self.previous = None
        nb = ~is_brake
        self.last = {"mode": "filter", "policy_margin": round(float(margin[0]), 2),
                     "best_turn_margin": round(float(margin[nb].max()), 2), "n_safe_turn": int((safe_turn & nb).sum()),
                     "policy_safe": bool(safe_turn[0]), "any_safe": bool(safe.any()),
                     "changed": bool(changed), "brake": brake, "chosen": out.tolist()}

        return out, bool(changed)
