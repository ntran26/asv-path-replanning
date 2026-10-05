"""Safety layer v3: committed backup and a recovery mode (2026-10-02).

`planning/archive/safety/SAFETY_LAYER_V3_PLAN.md`.  v2 (`safety_v2.py`) checks every step
whether the policy's action leaves an escape manoeuvre, and replaces it with the
nearest action that does.  On the development set it removed obstacle contacts
but traded them for wall contacts and stalls: (a) when no new candidate passed
it fell back to braking or to the policy's own action, discarding the escape it
had certified a step earlier; and (b) it handed the helm back the moment the
policy's action passed again -- often slow, off the path and beside a wall, a
state the policy (trained without a safety layer) rarely saw.

v3 keeps v2's prediction (identified hull, rudder delay and servo, static LiDAR
returns, the map polygon, the tracked target at constant velocity, the
pessimistic braking model) and its terminal run-out check, and adds:

* **Committed backup** (model-predictive shielding, Bastani et al.; the
  gatekeeper idea of keeping a verified continuation).  The chosen action's
  whole certified plan -- the action for `COMMIT_S`, then its escape manoeuvre --
  is stored.  Each step it is re-checked from the new state as one candidate
  among the rest; when nothing passes, the stored plan is followed rather than
  braking blind or letting an unsafe action through.
* **Recovery mode.**  An override starts a recovery.  During it the reference
  action is not the policy's but a path-rejoin action (LOS course autopilot at
  cruise), so the filter finishes passing the hazard and heads back to the
  path; every action still has to pass the same check.  The helm goes back to
  the policy only when it is in a state the policy knows: the policy's own
  action passes with `HANDBACK_MARGIN_M`, the hull has steerage way
  (`HANDBACK_SPEED`), its heading is within `HANDBACK_HEADING_DEG` of the path,
  and the recovery has lasted `MIN_RECOVERY_STEPS` -- or nothing is in reach.
* Braking stays a traffic-only option (v2 iteration 6).

**What it is not.**  The basin is 10 m wide and the hull's steady turning circle
at any rpm is about 10.5 m across, with no astern motion: there is no loiter
circle or stop that is safe for ever, so no invariant terminal set exists and
the filter cannot be called *certified*.  Its terminal condition is a
finite-mission one (a clear run along the final heading).  It is a finite-
horizon predictive filter, evaluated empirically.

Selected by `constants.SAFETY_VERSION = 3` at run time.  Interventions are
counted on the environment as for v2 (`safety_v2_steps`).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

import constants as cfg
import safety_v2 as v2
import ship as shipmod
from classical import common as cc

HANDBACK_MARGIN_M = 0.30             # the policy's action must pass with this much room ...
HANDBACK_SPEED = 0.30                # ... the hull have steerage way (m/s) ...
HANDBACK_HEADING_DEG = 30.0          # ... head within this of the path ...
MIN_RECOVERY_STEPS = 4               # ... and the recovery have lasted 2 s
REJOIN_THROTTLE = 0.0                # the rejoin action's propulsion: cruise
W_CONTINUE = 0.25                    # preference for continuing the stored plan (distance units)
RECOVERY_REFERENCE = "policy"        # the action a recovery stays nearest to: "policy" or "rejoin" (dev set,
                                     # 2 Oct: rejoin 95 goals, policy 108 -- L1-L3 paths run through the panels)
REJOIN_LOOKAHEAD_M = cc.LOS_LOOKAHEAD_M   # LOS look-ahead of the rejoin action
LAST_CERT_MAX_STEPS = 4              # a stored plan that no longer passes is followed at most 2 s;
                                     # the first version followed expired plans for 13-140 steps


def _decisions() -> Tuple[int, int, int]:
    d = int(math.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))
    commit = int(round(v2.COMMIT_S / cfg.UPDATE_RATE))
    turn = commit + int(round(v2.RECOVERY_TURN_S / cfg.UPDATE_RATE))
    return d, commit, turn


def plan_for(cand, rec) -> np.ndarray:
    """(D, 2) controls: the candidate for the commit, the escape for its turn, then
    the escape's propulsion with the rudder centred."""
    d, commit, turn = _decisions()
    seq = np.empty((d, 2))
    seq[:commit] = cand
    seq[commit:turn] = rec
    seq[turn:] = (0.0, rec[1])
    return seq


def continuation(plan: np.ndarray) -> np.ndarray:
    """The stored plan one step on (its certified steps only; empty when spent),
    padded to the horizon for checking with the rudder centred and coasting
    propulsion -- never a brake held past the plan."""
    d = _decisions()[0]
    rest = plan[1:]
    last = 0.0 if not len(rest) or np.isnan(rest[-1, 1]) else rest[-1, 1]
    pad = np.tile((0.0, last), (d - len(rest), 1))
    return np.vstack([rest, pad]) if len(rest) else pad


def rollout_seq(snap, act, seqs: np.ndarray):
    """`safety_v2.rollout` for arbitrary per-decision control sequences, (n, D, 2);
    NaN throttle is full astern on the pessimistic braking model."""
    n, d = seqs.shape[0], seqs.shape[1]
    state = np.zeros((7, n))
    state[0:4] = np.array([max(0.0, snap.u), snap.v, snap.r, snap.heading])[:, None]
    state[4] = act.servo
    state[5], state[6] = snap.x, snap.y
    params = {k: np.full(n, v) for k, v in cc.IDENTIFIED.items()}
    pending = [np.full(n, x) for x in act.buffer] if act.buffer is not None else None
    K = d * cc.SUBSTEPS
    pos, hdg, spd = np.empty((K, n, 2)), np.empty((K, n)), np.empty((K, n))
    brake_decel = shipmod.braking_thrust(v2.ASTERN_RPM, efficiency=v2.BRAKE_EFFICIENCY) / shipmod.M11
    brake_since = np.full(n, np.inf)
    j = 0
    for k in range(d):
        ctrl = seqs[:, k, :]
        rpm = v2._rpm(ctrl[:, 1])
        delta = -cc.MAX_RUDDER_RAD * ctrl[:, 0]
        braking = rpm < 0.0
        if pending is None:
            pending = [delta.copy() for _ in range(cc.DELAY_STEPS)]
        for _ in range(cc.SUBSTEPS):
            pending.append(delta)
            state = cc.dyn.rk4_step(state, np.maximum(rpm, 0.0), pending.pop(0), params, cc.PRED_DT)
            t_now = (j + 1) * cc.PRED_DT
            brake_since = np.where(braking, np.minimum(brake_since, t_now), np.inf)
            active = braking & (t_now - brake_since >= v2.BRAKE_DELAY_S)
            if active.any():
                state[0] = np.where(active, np.maximum(0.0, state[0] - brake_decel * cc.PRED_DT), state[0])
            pos[j] = state[5:7].T
            hdg[j] = state[3]
            spd[j] = np.hypot(state[0], state[1])
            j += 1
    return cc.Rollout(pos, hdg, spd, cc.PRED_DT * np.arange(1, K + 1))


class SafetyFilterV3(v2.SafetyFilterV2):
    def __init__(self) -> None:
        super().__init__()
        self.mode = "nominal"
        self.plan: Optional[np.ndarray] = None        # (steps, 2) the stored certified plan
        self.recovery_steps = 0
        self.uncertified_steps = 0                    # steps the stored plan has been followed unchecked

    def _rejoin(self, snap) -> np.ndarray:
        err = cc.wrap_pi(cc.los_heading(snap, lookahead=REJOIN_LOOKAHEAD_M) - snap.heading)
        return np.array([float(cc.course_rudder(err, snap.r)), REJOIN_THROTTLE])

    def _handback_state(self, snap) -> bool:
        off = abs(math.degrees(cc.wrap_pi(snap.heading - snap.base_heading)))
        return snap.u >= HANDBACK_SPEED and off <= HANDBACK_HEADING_DEG

    def filter(self, env, action) -> Tuple[np.ndarray, bool]:
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        snap = self.perception.snapshot(env)
        raw = np.array([snap.u, snap.v, snap.r])
        self.ego = raw if self.ego is None else self.ego + v2.EGO_SMOOTHING * (raw - self.ego)
        snap.u, snap.v, snap.r = (float(z) for z in self.ego)
        env._v2_brake = False
        threat = self._threat_in_reach(snap)
        if not threat:
            # Nothing in reach: the policy has the helm (and a recovery ends).
            self._release()
            self.actuators.issue(env, float(action[0]))
            self.last = {"mode": "idle"}
            return action.astype(np.float32), False

        traffic = any(np.hypot(*(t.position - snap.position)) < v2.ENGAGE_RANGE_M for t in snap.tracks)
        rejoin = self._rejoin(snap)
        rows = [tuple(action)] + [(r, t) for r in v2.RUDDERS for t in v2.THROTTLES] + [tuple(rejoin)]
        if traffic:
            rows += [(r, np.nan) for r in v2.BRAKE_RUDDERS]
        cands = np.array(rows, dtype=float)
        recs = [rc for rc in v2.RECOVERY if traffic or rc[1] is not v2.BRAKE]
        recs = np.array([(r, np.nan if t is v2.BRAKE else t) for r, t in recs], dtype=float)
        m, nrec = len(cands), len(recs)
        seqs = [plan_for(c, r) for c in cands for r in recs]
        cont = continuation(self.plan) if self.plan is not None and len(self.plan) > 1 else None
        spent = self.plan is not None and len(self.plan) <= 1
        if cont is not None:
            seqs.append(cont)
        ro = rollout_seq(snap, self.actuators, np.asarray(seqs))
        first, clear = self._evaluate(snap, ro)
        cont_first, cont_clear = (first[-1], clear[-1]) if cont is not None else (0.0, -np.inf)
        first, clear = first[:m * nrec].reshape(m, nrec), clear[:m * nrec].reshape(m, nrec)
        ok = np.isinf(first)
        safe = ok.any(axis=1)
        margin = np.where(ok, clear, -np.inf).max(axis=1)
        best_rec = np.argmax(np.where(ok, clear, -np.inf), axis=1)
        is_brake = np.isnan(cands[:, 1])
        cont_ok = bool(np.isinf(cont_first))

        # Nominal: the policy passes with room.  Recovery: hand back only in a known state.
        if self.mode == "recovery":
            self.recovery_steps += 1
            if (safe[0] and margin[0] >= HANDBACK_MARGIN_M and self.recovery_steps >= MIN_RECOVERY_STEPS
                    and self._handback_state(snap)):
                return self._pass(env, action, margin, "handback")
            reference = rejoin if RECOVERY_REFERENCE == "rejoin" else action
        else:
            if safe[0] and margin[0] >= v2.TRIGGER_MARGIN_M:
                return self._pass(env, action, margin, "nominal")
            reference = action

        def dist(rows_idx):
            thr = np.where(np.isnan(cands[rows_idx, 1]), -1.5, cands[rows_idx, 1])
            return (cands[rows_idx, 0] - reference[0]) ** 2 + v2.W_THROTTLE * (thr - reference[1]) ** 2

        idx = np.arange(m)
        pool = idx[safe & ~is_brake]
        out, plan, why = None, None, ""
        if len(pool) or cont_ok:
            best = max(float(margin[pool].max()) if len(pool) else -np.inf, cont_clear if cont_ok else -np.inf)
            floor = min(v2.TRIGGER_MARGIN_M, best - v2.ROOM_SLACK_M)
            pool = pool[margin[pool] >= floor]
            score = dist(pool) if len(pool) else np.array([])
            use_cont = False
            if cont_ok and cont_clear >= floor:
                c = cont[0]
                thr = -1.5 if np.isnan(c[1]) else c[1]
                cont_score = (c[0] - reference[0]) ** 2 + v2.W_THROTTLE * (thr - reference[1]) ** 2 - W_CONTINUE
                use_cont = not len(pool) or cont_score <= score.min()
            if use_cont:
                out, plan, why = cont[0].copy(), self.plan[1:], "continue"
            else:
                i = int(pool[np.argmin(score)])
                if self.mode == "nominal" and i == 0:
                    return self._pass(env, action, margin, "nominal")     # the policy is as good as any
                out, plan, why = cands[i].copy(), plan_for(cands[i], recs[best_rec[i]]), "turn"
        elif traffic and (safe & is_brake).any():
            pool = idx[safe & is_brake]
            i = int(pool[np.argmin(dist(pool))])
            out, plan, why = cands[i].copy(), plan_for(cands[i], recs[best_rec[i]]), "brake"
        elif cont is not None and not spent and self.uncertified_steps < LAST_CERT_MAX_STEPS:
            out, plan, why = cont[0].copy(), self.plan[1:], "last certificate"   # nothing passes: keep the plan
            self.uncertified_steps += 1
        else:
            # Nothing passes and no live plan: v2's hold-back -- the policy's action stands
            # unless an alternative delays the first predicted contact by HOLD_BACK_GAIN_S.
            latest = first.max(axis=1)
            best = int(np.lexsort((-clear.max(axis=1), -latest))[0])
            if latest[best] - latest[0] < v2.HOLD_BACK_GAIN_S:
                return self._pass(env, action, margin, "no escape")
            out, plan, why = cands[best].copy(), None, "hold back"
        if why != "last certificate":
            self.uncertified_steps = 0

        self.mode, self.plan = "recovery", plan
        brake = bool(np.isnan(out[1]))
        env._v2_brake = brake
        if brake:
            out[1] = -1.0
        out = out.astype(np.float32)
        self.actuators.issue(env, float(out[0]))
        changed = not np.allclose(out, action)
        self.last = {"mode": "recovery", "why": why, "policy_margin": round(float(margin[0]), 2),
                     "best_margin": round(float(margin[~is_brake].max()), 2), "continuation_ok": cont_ok,
                     "changed": bool(changed), "brake": brake, "chosen": out.tolist(),
                     "recovery_steps": self.recovery_steps}
        return out, bool(changed)

    def _release(self) -> None:
        self.mode, self.plan, self.recovery_steps, self.uncertified_steps = "nominal", None, 0, 0

    def _pass(self, env, action, margin, why):
        self._release()
        self.actuators.issue(env, float(action[0]))
        self.last = {"mode": "filter", "why": why, "policy_margin": round(float(margin[0]), 2),
                     "changed": False, "brake": False, "chosen": action.tolist()}
        return action.astype(np.float32), False
