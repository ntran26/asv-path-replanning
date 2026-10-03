"""Experimental V8 follow-up: preserve SAC only with a passing backup.

Compare V8's proposed first command and SAC's current command followed by the
same complete backup tail, using the same onboard state and actuator history.
Two independent switches preserve SAC when (a) only its plan passes, or
(b) both pass and its minimum clearance is at least the proposal's. When both
fail, keep V8's action, stored plan and bounded fallback counters unchanged.
In particular, rejecting a backup is not evidence that SAC is safe.
An optional third switch, disabled by default, preserves any passing SAC
backup regardless of the proposal's larger clearance. It tests prioritizing
the learning input among model-feasible plans without adding a new margin.

Method inspiration: Wabersich & Zeilinger (2021), Sec. 4.1, Eq. (5a)-(5f):
https://arxiv.org/html/1812.05506v4#S4.SS1 . Their first-action deviation
objective motivates policy preservation with a feasible backup. Our paired
tail comparison and relative-clearance condition are engineering adaptations,
not their optimizer, uncertainty bounds or terminal safe-set construction.
Keeping a currently failing V8 fallback is explicitly uncertified. Sampled
model feasibility does not establish collision avoidance or goal preservation.

No margins, horizons or V2-V9 behavior are changed. No scenario identifiers,
outcomes, hidden plant parameters or future SAC observations enter this rule.
Target hypotheses remain disabled by default; inherited optional nonzero
hypotheses have the limitations described in safety_target_prediction.py.
"""
from __future__ import annotations

import copy
import math

import numpy as np

import safety_v7 as v7
import safety_v8 as v8
import safety_v9 as v9


PREFER_FEASIBLE_POLICY = True
PREFER_POLICY_MARGIN = True
PREFER_ANY_FEASIBLE_POLICY = False
TARGET_TURN_RATE_DEG_S = 0.0


class SafetyFilterV10(v9.SafetyFilterV9):
    """V8 proposals with independently ablatable, checked policy preferences."""

    def __init__(self, *, prefer_feasible_policy=None, prefer_policy_margin=None,
                 prefer_any_feasible_policy=None, target_turn_rate_deg_s=None):
        rate = (TARGET_TURN_RATE_DEG_S if target_turn_rate_deg_s is None
                else target_turn_rate_deg_s)
        # Reuse V9's target-envelope and pair evaluator, not its blanket veto.
        super().__init__(target_turn_rate_deg_s=rate, require_current_plan=False,
                         prefer_policy_margin=False)
        self.prefer_feasible_policy = (PREFER_FEASIBLE_POLICY if prefer_feasible_policy is None
                                       else bool(prefer_feasible_policy))
        self.prefer_policy_margin = (PREFER_POLICY_MARGIN if prefer_policy_margin is None
                                    else bool(prefer_policy_margin))
        self.prefer_any_feasible_policy = (
            PREFER_ANY_FEASIBLE_POLICY if prefer_any_feasible_policy is None
            else bool(prefer_any_feasible_policy))

    def _preserve_policy(self, env, policy, before, details, reason, plan, clearance):
        self.actuators = before
        self.actuators.issue(env, float(policy[0]))
        env._v2_brake = False
        self._release()
        self.plan = plan.copy()
        self.last.update(details)
        self.last.update(mode="nominal", why=reason, changed=False, brake=False,
                         chosen=policy.tolist(), checked_clearance=float(clearance),
                         recovery_steps=0, uncertified_steps=0,
                         plan_steps=len(self.plan), nominal_plan_retained=True,
                         v10_policy_preserved=True)
        return policy.copy(), False

    def _filter(self, env, action):
        policy = np.clip(np.asarray(action, dtype=float).reshape(2), -1., 1.).astype(np.float32)
        if not np.isfinite(policy).all():
            raise ValueError("Policy action must be finite")
        before = copy.deepcopy(self.actuators)
        self._target_snapshot = self._target_envelope = None
        out, changed = v8.SafetyFilterV8._filter(self, env, policy)
        details = {
            "v10_pair_checked": False,
            "v10_policy_preserved": False,
            "v10_parent_why": self.last.get("why", "idle"),
            "v10_parent_action": np.asarray(out).tolist(),
            "v10_parent_brake": bool(getattr(env, "_v2_brake", False)),
            "v10_prefer_feasible_policy": self.prefer_feasible_policy,
            "v10_prefer_policy_margin": self.prefer_policy_margin,
            "v10_prefer_any_feasible_policy": self.prefer_any_feasible_policy,
            "v10_target_turn_rate_deg_s": math.degrees(self.target_turn_rate_rad_s),
        }
        if not changed or not (self.prefer_feasible_policy or self.prefer_policy_margin
                               or self.prefer_any_feasible_policy):
            self.last.update(details)
            return out, changed
        if bool(getattr(env, "command_rate_limit", False)) or self.plan is None:
            # The current model does not simulate the optional command limiter.
            # Missing/unmodelled backup evidence cannot authorize a policy pass.
            details["v10_skip_reason"] = ("unmodelled command limit" if
                bool(getattr(env, "command_rate_limit", False)) else "no backup")
            self.last.update(details)
            return out, changed

        policy_plan = v7.replace_first_action(self.plan, policy)
        proposed_plan = policy_plan.copy()
        proposed_plan[0] = np.asarray(out, dtype=np.float32)
        if bool(getattr(env, "_v2_brake", False)):
            proposed_plan[0, 1] = np.nan  # Full astern, not policy throttle -1.
        first, clearance = self._evaluate_pair(
            self._observer_snapshot, before, np.stack((proposed_plan, policy_plan)))
        first, clearance = np.asarray(first, dtype=float), np.asarray(clearance, dtype=float)
        if first.shape != (2,) or clearance.shape != (2,):
            raise ValueError("Expected two aligned trajectory checks")
        passing = np.isposinf(first) & (clearance >= 0.)
        details.update(v10_pair_checked=True,
                       v10_proposed_first_violation=float(first[0]),
                       v10_policy_first_violation=float(first[1]),
                       v10_proposed_clearance=float(clearance[0]),
                       v10_policy_tail_clearance=float(clearance[1]),
                       v10_proposed_currently_passing=bool(passing[0]),
                       v10_policy_tail_passing=bool(passing[1]))
        reason = None
        if passing[1]:
            if self.prefer_any_feasible_policy:
                reason = "feasible policy preference"
            elif self.prefer_feasible_policy and not passing[0]:
                reason = "feasible policy backup"
            elif self.prefer_policy_margin and passing[0] and clearance[1] >= clearance[0]:
                reason = "policy margin dominates"
        if reason is not None:
            return self._preserve_policy(env, policy, before, details, reason,
                                         policy_plan, clearance[1])
        # Preserve the parent's plan/counters too: a failed paired check must
        # neither discard its emergency recovery nor renew its fallback budget.
        self.last.update(details)
        return out, changed
