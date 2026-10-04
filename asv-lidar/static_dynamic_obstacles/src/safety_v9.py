"""Experimental V8 follow-up: currently checked, beneficial interventions.

V8 disables hold-back but can still execute an expired stored backup. V9
rechecks the actual proposed command plus its whole backup before an override.
It also compares the exact same tail after the current SAC command. If both
pass and SAC has at least as much minimum clearance, SAC retains control.
No extra .15 m floor is imposed on a proposal accepted by V8's hard checks.

Inspiration: Wabersich & Zeilinger (2021), Sec. 4.1, minimizing deviation of
the first action subject to a currently feasible backup:
https://arxiv.org/abs/1812.05506v4 . The comparable-tail margin preference is
an engineering adaptation of V8 plan option (c), not their optimizer or
formal guarantee. Rejecting an unchecked override does NOT certify SAC safe.

Optional target hypotheses implement V8 plan option (a), with method sources
in safety_target_prediction.py. They are disabled by default because the
turn-rate hypothesis has not been calibrated. Every active check (including
CEM, continuation and V7 repair) shares the same per-decision target envelope.
Finite sampled turns do not cover every reactive target trajectory.

No new episode performance has been measured. Runtime SAFETY_VERSION=9 is
experimental; the evaluated V8 remains available as the comparison baseline.
"""
from __future__ import annotations

import copy
import math

import numpy as np

import safety_v4 as v4
import safety_v7 as v7
import safety_v8 as v8


# Zero preserves the existing CV target model. A positive explicit value is
# an uncalibrated finite hypothesis set, not a bound identified from the plant.
TARGET_TURN_RATE_DEG_S = 0.0
REQUIRE_CURRENT_PLAN = True
PREFER_POLICY_MARGIN = True


class SafetyFilterV9(v8.SafetyFilterV8):
    def __init__(self, *, target_turn_rate_deg_s=None, require_current_plan=None,
                 prefer_policy_margin=None):
        super().__init__()
        rate = (TARGET_TURN_RATE_DEG_S if target_turn_rate_deg_s is None
                else float(target_turn_rate_deg_s))
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError("Target turn-rate hypothesis must be finite and nonnegative")
        self.target_turn_rate_rad_s = math.radians(rate)
        self.require_current_plan = (REQUIRE_CURRENT_PLAN if require_current_plan is None
                                     else bool(require_current_plan))
        self.prefer_policy_margin = (PREFER_POLICY_MARGIN if prefer_policy_margin is None
                                     else bool(prefer_policy_margin))
        self._target_snapshot = None
        self._target_envelope = None

    def _evaluate(self, snap, rollout):
        first, clear = super()._evaluate(snap, rollout)
        if self.target_turn_rate_rad_s > 0.0 and snap.tracks:
            if self._target_snapshot is not snap:
                from safety_target_prediction import TargetPredictionEnvelope
                self._target_envelope = TargetPredictionEnvelope.from_snapshot(
                    snap, turn_rate_rad_s=self.target_turn_rate_rad_s)
                self._target_snapshot = snap
            target_first, target_clear = self._target_envelope.evaluate(rollout)
            first = np.minimum(first, target_first)
            clear = np.minimum(clear, target_clear)
        return first, clear

    def _evaluate_pair(self, snap, actuators, sequences):
        if v4.DUAL_BRAKE_PREDICTION:
            from safety_prediction import evaluate_sequences
            result = evaluate_sequences(snap, actuators, sequences, self._evaluate)
            return result.first, result.clear
        return self._evaluate(snap, self._rollout(snap, actuators, sequences))

    def _return_policy(self, env, policy, before, details, reason, plan=None, clearance=None):
        self.actuators = before
        self.actuators.issue(env, float(policy[0]))
        env._v2_brake = False
        self._release()
        self.plan = None if plan is None else plan.copy()
        self.last.update(details)
        self.last.update(mode="nominal", why=reason, changed=False, brake=False,
                         chosen=policy.tolist(), checked_clearance=clearance,
                         recovery_steps=0, uncertified_steps=0,
                         plan_steps=0 if self.plan is None else len(self.plan),
                         nominal_plan_retained=self.plan is not None,
                         v9_override_suppressed=True)
        return policy.copy(), False

    def _filter(self, env, action):
        policy = np.clip(np.asarray(action, dtype=float).reshape(2), -1., 1.).astype(np.float32)
        before = copy.deepcopy(self.actuators)
        self._target_snapshot = self._target_envelope = None
        out, changed = super()._filter(env, policy)
        details = {
            "v9_current_plan_checked": False,
            "v9_override_suppressed": False,
            "v9_parent_why": self.last.get("why", "idle"),
            "v9_parent_action": np.asarray(out).tolist(),
            "v9_parent_brake": bool(getattr(env, "_v2_brake", False)),
            "v9_target_turn_rate_deg_s": math.degrees(self.target_turn_rate_rad_s),
            "v9_target_hypotheses_enabled": self.target_turn_rate_rad_s > 0.0,
            "v9_require_current_plan": self.require_current_plan,
            "v9_prefer_policy_margin": self.prefer_policy_margin,
        }
        if not changed or not (self.require_current_plan or self.prefer_policy_margin):
            self.last.update(details)
            return out, changed
        if bool(getattr(env, "command_rate_limit", False)):
            # The inherited predictor does not model this limiter; do not label
            # its nominal action sequence as a current check of executed input.
            if self.require_current_plan:
                return self._return_policy(env, policy, before, details,
                                           "unmodelled command limit")
            self.last.update(details)
            return out, changed
        if self.plan is None:
            if self.require_current_plan:
                return self._return_policy(env, policy, before, details,
                                           "override without backup")
            self.last.update(details)
            return out, changed

        # Same state, actuator delay line, horizon and backup tail for both.
        # replace_first_action also pads shortened continuations to eight seconds.
        policy_plan = v7.replace_first_action(self.plan, policy)
        proposed_plan = policy_plan.copy()
        proposed_plan[0] = np.asarray(out, dtype=np.float32)
        if bool(getattr(env, "_v2_brake", False)):
            proposed_plan[0, 1] = np.nan  # Signed astern, not ordinary throttle -1.
        sequences = np.stack((proposed_plan, policy_plan))
        first, clearance = self._evaluate_pair(self._observer_snapshot, before, sequences)
        first, clearance = np.asarray(first, dtype=float), np.asarray(clearance, dtype=float)
        if first.shape != (2,) or clearance.shape != (2,):
            raise ValueError("Expected two aligned trajectory checks")
        passing = np.isposinf(first) & (clearance >= 0.)
        details.update(v9_current_plan_checked=True,
                       v9_proposed_first_violation=float(first[0]),
                       v9_policy_first_violation=float(first[1]),
                       v9_proposed_clearance=float(clearance[0]),
                       v9_policy_tail_clearance=float(clearance[1]),
                       v9_proposed_currently_passing=bool(passing[0]),
                       v9_policy_tail_passing=bool(passing[1]))
        if not passing[0]:
            if self.require_current_plan:
                return self._return_policy(
                    env, policy, before, details, "unchecked override suppressed",
                    plan=policy_plan if passing[1] else None,
                    clearance=float(clearance[1]) if passing[1] else None)
            self.last.update(details)
            return out, changed
        if self.prefer_policy_margin and passing[1] and clearance[1] >= clearance[0]:
            return self._return_policy(env, policy, before, details,
                                       "policy margin dominates", plan=policy_plan,
                                       clearance=float(clearance[1]))

        self.plan = proposed_plan.copy()
        self.uncertified_steps = 0
        self.last.update(details)
        self.last.update(checked_clearance=float(clearance[0]), plan_steps=len(self.plan),
                         uncertified_steps=0)
        return out, changed
