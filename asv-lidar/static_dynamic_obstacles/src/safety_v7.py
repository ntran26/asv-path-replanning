"""Experimental minimal-intervention extension of V6; no episode validation yet.

Before an override, substitute SAC's current action for just the first decision
of the proposed recovery and re-evaluate the whole trajectory. Preserve SAC
only if the replacement passes the existing hard checks and trigger margin.
The future commands remain an explicit backup, not imagined SAC actions.

Method inspiration: Wabersich & Zeilinger (2021), predictive safety filtering,
https://arxiv.org/abs/1812.05506 : test the learning input against a feasible
backup and minimize intervention. This local substitution is an engineering
adaptation, not their optimizer, uncertainty treatment or safety guarantee.
The separate risk monitor cites its sources in safety_risk_monitor.py and
records evidence only. Its recommendation never authorizes a policy bypass.

No scenario ID, episode outcome, simulator geometry or future observation is
used. All tests are synthetic; test-set-v2 results contain no decision traces
from which to validate a success/failure classifier.
"""
from __future__ import annotations

import copy
import math

import numpy as np

import constants as cfg
import safety_v2 as v2
import safety_v4 as v4
import safety_v6 as v6


POLICY_PREFIX_REPAIR = True
RISK_MONITOR_ENABLED = True  # Diagnostics only, regardless of its recommendation.


def replace_first_action(plan, action):
    """Copy the backup, pad a spent tail to the full horizon, replace one action.

    Padding is V3's existing continuation convention: centre the rudder and
    retain the final throttle, or coast at cruise after a brake. Never shorten
    the prediction just because the stored recovery is nearly spent.
    """
    plan = np.asarray(plan, dtype=float)
    action = np.asarray(action, dtype=float)
    decisions = int(math.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))
    if plan.ndim != 2 or plan.shape[1] != 2 or not 1 <= len(plan) <= decisions:
        raise ValueError("Backup must contain one to a full horizon of 2D actions")
    if action.shape != (2,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
        raise ValueError("Policy action must be finite and normalized")
    if (not np.isfinite(plan[:, 0]).all() or (np.abs(plan[:, 0]) > 1).any()
            or np.isinf(plan[:, 1]).any()
            or (np.abs(plan[np.isfinite(plan[:, 1]), 1]) > 1).any()):
        raise ValueError("Backup commands must be normalized; only throttle may be NaN")
    throttle = 0.0 if np.isnan(plan[-1, 1]) else float(plan[-1, 1])
    padded = np.tile((0.0, throttle), (decisions, 1))
    padded[:len(plan)] = plan
    padded[0] = action
    return padded


class SafetyFilterV7(v6.SafetyFilterV6):
    """V6 proposals with a fully rechecked policy-first backup substitution."""

    def __init__(self):
        super().__init__()
        from safety_risk_monitor import ObservableRiskMonitor
        self.risk_monitor = ObservableRiskMonitor()

    def _check_repaired(self, snap, actuators, plan):
        sequences = plan[None]
        if v4.DUAL_BRAKE_PREDICTION:
            from safety_prediction import evaluate_sequences
            envelope = evaluate_sequences(snap, actuators, sequences, self._evaluate)
            first, clear = envelope.first, envelope.clear
        else:
            first, clear = self._evaluate(snap, self._rollout(snap, actuators, sequences))
        return float(first[0]), float(clear[0])

    def _filter(self, env, action):
        # Check exactly the float32 command that can be returned to the bridge.
        policy = np.clip(np.asarray(action, dtype=float).reshape(2), -1., 1.).astype(np.float32)
        if not np.isfinite(policy).all():
            raise ValueError("Policy action must be finite")
        before = copy.deepcopy(self.actuators)
        out, changed = super()._filter(env, policy)
        snap = self._observer_snapshot
        proposal = dict(self.last)
        details = {
            "v7_policy_repair_enabled": bool(POLICY_PREFIX_REPAIR),
            "v7_repair_checked": False,
            "v7_policy_preserved": False,
            "v6_proposed_change": bool(changed),
            "v6_proposal_why": proposal.get("why", "idle"),
            "v6_proposal_action": np.asarray(out).tolist(),
            "v6_proposal_brake": bool(getattr(env, "_v2_brake", False)),
        }
        if RISK_MONITOR_ENABLED:
            evidence = self.risk_monitor.update(
                snap, policy, before, self._rollout,
                fresh=not bool(getattr(env, "pose_stale", False)))
            details["risk_monitor"] = evidence.as_dict()
        # Current predictors do not model command-rate limiting. Do not use a
        # nominal unconstrained prediction to authorize an extra policy pass.
        rate_limited = bool(getattr(env, "command_rate_limit", False))
        details["v7_rate_limit_skip"] = rate_limited
        if POLICY_PREFIX_REPAIR and changed and self.plan is not None and not rate_limited:
            repaired = replace_first_action(self.plan, policy)
            first, clearance = self._check_repaired(snap, before, repaired)
            details.update(v7_repair_checked=True,
                           v7_repair_first_violation=first,
                           v7_repair_clearance=clearance)
            if np.isposinf(first) and clearance >= v2.TRIGGER_MARGIN_M:
                # V6 advanced only its internal command model. Replace that
                # proposal with one advance from the pre-decision history.
                self.actuators = before
                self.actuators.issue(env, float(policy[0]))
                env._v2_brake = False
                self.mode, self.plan = "nominal", repaired.copy()
                self.recovery_steps = self.uncertified_steps = 0
                out, changed = policy.copy(), False
                self.last.update(mode="nominal", why="policy prefix repaired",
                                 checked_clearance=clearance, changed=False, brake=False,
                                 chosen=out.tolist(), recovery_steps=0,
                                 uncertified_steps=0, nominal_plan_retained=True,
                                 plan_steps=len(repaired))
                details["v7_policy_preserved"] = True
        self.last.update(details)
        return out, changed
