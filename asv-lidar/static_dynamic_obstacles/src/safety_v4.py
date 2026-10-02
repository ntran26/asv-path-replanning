"""Safety v4: model-based ego estimation and free-space memory clearing.

The selected development preset combines an issued-command model prior with
the existing sensor correction, and removes remembered LiDAR returns when
fresh finite rays contradict them. V3's prediction horizon, safety gaps,
candidate library and selection thresholds remain unchanged.

Countersteering backups, retention of checked nominal plans, and a second
braking response remain disabled experimental ablations. Set module switches
before constructing the filter; observer and perception are selected then.
See planning/SAFETY_LAYER_V4_PLAN.md for paired results and remaining wall
failures. These finite-horizon checks do not guarantee collision avoidance.
"""
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np

import safety_v2 as v2
import safety_v3 as v3

COUNTERSTEER_RECOVERY = False
RETAIN_NOMINAL_PLAN = False
DUAL_BRAKE_PREDICTION = False       # development ablation; see safety_prediction.py
FREE_SPACE_MEMORY = True           # clear only old returns contradicted by fresh finite rays
MODEL_EGO_OBSERVER = True           # model prior + existing .3 measurement correction


def recovery_templates(recoveries):
    """Keep v3 templates in order, followed by their countersteering variants."""
    result = [(np.asarray(rec, dtype=float), False) for rec in recoveries]
    if COUNTERSTEER_RECOVERY:
        result += [(np.asarray(rec, dtype=float), True) for rec in recoveries
                   if rec[0] != 0.0 and not np.isnan(rec[1])]
    return result


def plan_for(candidate, recovery, countersteer=False):
    """Build a v3 plan, optionally reversing its recovery turn before centring."""
    plan = v3.plan_for(candidate, recovery)
    if countersteer:
        _, commit, turn_end = v3._decisions()
        counter_end = min(len(plan), turn_end + (turn_end - commit))
        plan[turn_end:counter_end] = (-recovery[0], recovery[1])
    return plan


def sequence_bank(candidates, templates):
    """Candidate-major plans; retain this ordering for selecting the checked plan."""
    return np.asarray([plan_for(candidate, recovery, countersteer)
                       for candidate in candidates
                       for recovery, countersteer in templates])


class SafetyFilterV4(v3.SafetyFilterV3):
    def __init__(self):
        super().__init__()
        self.observer = None
        if MODEL_EGO_OBSERVER:
            from safety_observer import EgoObserver
            self.observer = EgoObserver()
        if FREE_SPACE_MEMORY:
            import constants as cfg
            from safety_perception import SafetyPerception
            self.perception = SafetyPerception(memory_frames=int(round(v2.MEMORY_S / cfg.UPDATE_RATE)))

    def _pass_checked(self, env, action, margin, why, plan, diagnostics):
        """Release recovery mode while keeping the checked plan for this action."""
        out, changed = super()._pass(env, action, margin, why)
        if RETAIN_NOMINAL_PLAN and plan is not None:
            self.plan = plan.copy()
        self.last.update(diagnostics)
        self.last.update({
            "checked_clearance": float(margin[0]),
            "nominal_plan_retained": self.plan is not None,
            "plan_steps": 0 if self.plan is None else len(self.plan),
        })
        return out, changed

    def filter(self, env, action) -> Tuple[np.ndarray, bool]:
        if self.observer is None:
            return self._filter(env, action)
        # Actuators.issue advances the command history during selection. The
        # observer must start from the history before this command was issued.
        self._observer_actuators = copy.deepcopy(self.actuators)
        out, changed = self._filter(env, action)
        self.last["model_ego_observer"] = True
        return out, changed

    def observe_issued_command(self, rudder, rpm):
        """Advance the prior after the bridge resolves limiting and stop/astern.

        Inputs are the issued normalized rudder and signed RPM, never plant
        state. The environment calls this before physics advances.
        """
        if self.observer is not None:
            self.observer.predict(self._observer_snapshot, self._observer_actuators, rudder, rpm)

    def _filter(self, env, action) -> Tuple[np.ndarray, bool]:
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        snap = self.perception.snapshot(env)
        raw = np.array([snap.u, snap.v, snap.r])
        self._raw_ego = raw.copy()
        if self.observer is not None:
            self.ego = self.observer.update(raw, fresh=not bool(getattr(env, "pose_stale", False)))
        else:
            self.ego = raw if self.ego is None else self.ego + v2.EGO_SMOOTHING * (raw - self.ego)
        snap.u, snap.v, snap.r = (float(z) for z in self.ego)
        self._observer_snapshot = snap
        env._v2_brake = False
        if not self._threat_in_reach(snap):
            self._release()
            self.actuators.issue(env, float(action[0]))
            self.last = {"mode": "idle", "plan_steps": 0}
            return action.astype(np.float32), False

        traffic = any(np.hypot(*(t.position - snap.position)) < v2.ENGAGE_RANGE_M
                      for t in snap.tracks)
        rejoin = self._rejoin(snap)
        rows = [tuple(action)] + [(r, t) for r in v2.RUDDERS for t in v2.THROTTLES] + [tuple(rejoin)]
        if traffic:
            rows += [(r, np.nan) for r in v2.BRAKE_RUDDERS]
        candidates = np.asarray(rows, dtype=float)
        recoveries = [(r, np.nan if t is v2.BRAKE else t)
                      for r, t in v2.RECOVERY if traffic or t is not v2.BRAKE]
        templates = recovery_templates(recoveries)
        count, nrec = len(candidates), len(templates)
        bank = sequence_bank(candidates, templates)
        continuation = (v3.continuation(self.plan)
                        if self.plan is not None and len(self.plan) > 1 else None)
        sequences = (np.concatenate((bank, continuation[None]), axis=0)
                     if continuation is not None else bank)
        dual_count, fast_rejections = 0, 0
        if DUAL_BRAKE_PREDICTION:
            from safety_prediction import evaluate_sequences
            envelope = evaluate_sequences(snap, self.actuators, sequences, self._evaluate)
            first, clear = envelope.first, envelope.clear
            dual_count = envelope.dual_count
            fast_rejections = int((np.isinf(envelope.weak_first)
                                   & ~np.isinf(envelope.fast_first)).sum())
        else:
            rollout = v3.rollout_seq(snap, self.actuators, sequences)
            first, clear = self._evaluate(snap, rollout)
        cont_first, cont_clear = ((first[-1], clear[-1]) if continuation is not None
                                  else (0.0, -np.inf))
        first = first[:count * nrec].reshape(count, nrec)
        clear = clear[:count * nrec].reshape(count, nrec)
        ok = np.isinf(first)
        safe = ok.any(axis=1)
        margin = np.where(ok, clear, -np.inf).max(axis=1)
        best_rec = np.argmax(np.where(ok, clear, -np.inf), axis=1)
        is_brake = np.isnan(candidates[:, 1])
        cont_ok = bool(np.isinf(cont_first))
        diagnostics = {
            "continuation_checked": continuation is not None,
            "continuation_ok": cont_ok,
            "continuation_clearance": float(cont_clear),
            "policy_safe": bool(safe[0]), "any_safe": bool(safe.any() or cont_ok),
            "n_safe_candidates": int(safe.sum()),
            "countersteer_recovery_enabled": bool(COUNTERSTEER_RECOVERY),
            "dual_brake_sequences": dual_count,
            "fast_brake_rejections": fast_rejections,
        }

        def checked_policy(why):
            index = int(best_rec[0])
            details = dict(diagnostics, recovery_template=(
                "countersteer" if templates[index][1] else "v3"))
            return self._pass_checked(env, action, margin, why, bank[index], details)

        if self.mode == "recovery":
            self.recovery_steps += 1
            if (safe[0] and margin[0] >= v3.HANDBACK_MARGIN_M
                    and self.recovery_steps >= v3.MIN_RECOVERY_STEPS
                    and self._handback_state(snap)):
                return checked_policy("handback")
            reference = rejoin if v3.RECOVERY_REFERENCE == "rejoin" else action
        else:
            if safe[0] and margin[0] >= v2.TRIGGER_MARGIN_M:
                return checked_policy("nominal")
            reference = action

        def distance(indices):
            throttle = np.where(np.isnan(candidates[indices, 1]), -1.5, candidates[indices, 1])
            return ((candidates[indices, 0] - reference[0]) ** 2
                    + v2.W_THROTTLE * (throttle - reference[1]) ** 2)

        indices = np.arange(count)
        pool = indices[safe & ~is_brake]
        out, plan, why = None, None, ""
        chosen_clearance = -np.inf
        template_name = None
        if len(pool) or cont_ok:
            best = max(float(margin[pool].max()) if len(pool) else -np.inf,
                       cont_clear if cont_ok else -np.inf)
            floor = min(v2.TRIGGER_MARGIN_M, best - v2.ROOM_SLACK_M)
            pool = pool[margin[pool] >= floor]
            score = distance(pool) if len(pool) else np.array([])
            use_continuation = False
            if cont_ok and cont_clear >= floor:
                command = continuation[0]
                throttle = -1.5 if np.isnan(command[1]) else command[1]
                cont_score = ((command[0] - reference[0]) ** 2
                              + v2.W_THROTTLE * (throttle - reference[1]) ** 2 - v3.W_CONTINUE)
                use_continuation = not len(pool) or cont_score <= score.min()
            if use_continuation:
                out, plan, why = continuation[0].copy(), self.plan[1:].copy(), "continue"
                chosen_clearance, template_name = float(cont_clear), "continuation"
            else:
                chosen = int(pool[np.argmin(score)])
                if self.mode == "nominal" and chosen == 0:
                    return checked_policy("nominal")
                rec = int(best_rec[chosen])
                out = candidates[chosen].copy()
                plan, why = bank[chosen * nrec + rec].copy(), "turn"
                chosen_clearance = float(margin[chosen])
                template_name = "countersteer" if templates[rec][1] else "v3"
        elif traffic and (safe & is_brake).any():
            pool = indices[safe & is_brake]
            chosen = int(pool[np.argmin(distance(pool))])
            rec = int(best_rec[chosen])
            out = candidates[chosen].copy()
            plan, why = bank[chosen * nrec + rec].copy(), "brake"
            chosen_clearance, template_name = float(margin[chosen]), "v3"
        elif continuation is not None and self.uncertified_steps < v3.LAST_CERT_MAX_STEPS:
            out, plan, why = continuation[0].copy(), self.plan[1:].copy(), "last certificate"
            self.uncertified_steps += 1
            chosen_clearance, template_name = float(cont_clear), "continuation"
        else:
            latest = first.max(axis=1)
            chosen = int(np.lexsort((-clear.max(axis=1), -latest))[0])
            if latest[chosen] - latest[0] < v2.HOLD_BACK_GAIN_S:
                return self._pass_checked(env, action, margin, "no escape", None, diagnostics)
            out, plan, why = candidates[chosen].copy(), None, "hold back"
            chosen_clearance = float(clear[chosen].max())
        if why != "last certificate":
            self.uncertified_steps = 0

        self.mode, self.plan = "recovery", plan
        brake = bool(np.isnan(out[1]))
        env._v2_brake = brake
        if brake:
            out[1] = -1.0
        out = out.astype(np.float32)
        self.actuators.issue(env, float(out[0]))
        # NaN becomes policy-space throttle -1 for transport, but full astern
        # still changes propulsion when the policy already requested its floor.
        changed = brake or not np.allclose(out, action)
        self.last = {
            "mode": "recovery", "why": why,
            "policy_margin": round(float(margin[0]), 2),
            "best_margin": round(float(margin[~is_brake].max()), 2),
            "checked_clearance": chosen_clearance,
            "recovery_template": template_name,
            "changed": bool(changed), "brake": brake, "chosen": out.tolist(),
            "recovery_steps": self.recovery_steps,
            "uncertified_steps": self.uncertified_steps,
            "nominal_plan_retained": False,
            "plan_steps": 0 if self.plan is None else len(self.plan),
            **diagnostics,
        }
        return out, bool(changed)
