"""Safety v5: evidence-led extensions to the selected v4 filter.

Methods are independently switchable for paired development ablations:
* One-decision commitment: Osbert Bastani, "Safe Reinforcement Learning with
  Nonlinear Dynamics via Model Predictive Shielding", 2019/2020, Sec. III, Algorithm 1, https://arxiv.org/abs/1905.10691 .
  Test the issued action for one actual control interval, then a backup.
* Soft recovery: Wabersich & Zeilinger, "Predictive control barrier functions:
  Enhanced safety mechanisms for learning-based control", IEEE TAC 2023,
  Sec. III, https://arxiv.org/abs/2105.10241 . Minimize predicted violations
  when the sampled hard-constrained bank is infeasible, then policy distance.
* Feedback backups: Eriksen et al., BC-MPC, J. Field Robotics 2019, Sec. 3.1,
  https://doi.org/10.1002/rob.21900 ; Chen et al., backup CBFs, 2021, Sec. III,
  https://arxiv.org/abs/2104.11332 . Predict course-holding feedback responses.
* Sideslip-compensated rescue: Fossen, Pettersen & Galeazzi, IEEE TCST 2015,
  Sec. II.B, https://doi.org/10.1109/TCST.2014.2338354 . Account for the
  difference between heading and velocity direction in one additional backup.
  This uses the kinematic relationship, not their adaptive LOS stability proof.
* Verified-policy priority: Wabersich & Zeilinger, predictive safety filter,
  Sec. 4.1 Eq. (5a), https://arxiv.org/abs/1812.05506 . Among the existing
  feasible, clearance-qualified actions, prefer zero deviation from the policy
  over a continuation bonus. Recovery mode and its checked plan are retained.
* Policy-only feedback certification: Bastani Sec. III and Chen et al. Sec.
  III (links above). Before an initial override, test richer backups of the
  unchanged policy action, without expanding the competing action pool.

These are finite-bank adaptations, without the invariant terminal sets or
uncertainty assumptions required by the cited safety guarantees. V4 remains
unchanged. See planning/SAFETY_LAYER_V5_PLAN.md for provenance and ablations.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import constants as cfg
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
from classical import common as cc

# Original expanded-benchmark preset is archived in quick_v5_budget100.
# The quick-test preset isolates verified-policy priority. Additional feedback
# and sideslip banks remain available for separate ablations, without crediting
# untested combinations for an improvement.
ONE_DECISION_COMMIT = False
SOFT_RECOVERY = False
FEEDBACK_BACKUPS = False
FEEDBACK_ONLY_INFEASIBLE = True
SIDESLIP_RESCUE = False
PREFER_CERTIFIED_POLICY = True
POLICY_FEEDBACK_PRESERVATION = True


def plan_for(candidate, recovery, countersteer=False):
    if not ONE_DECISION_COMMIT:
        return v4.plan_for(candidate, recovery, countersteer)
    count = v3._decisions()[0]
    commit = 1
    turn_end = min(count, commit + int(round(v2.RECOVERY_TURN_S / cfg.UPDATE_RATE)))
    plan = np.empty((count, 2))
    plan[:commit] = candidate
    plan[commit:turn_end] = recovery
    plan[turn_end:] = (0.0, recovery[1])
    if countersteer:
        counter_end = min(count, turn_end + turn_end - commit)
        plan[turn_end:counter_end] = (-recovery[0], recovery[1])
    return plan


def sequence_bank(candidates, templates):
    return np.asarray([plan_for(candidate, recovery, countersteer)
                       for candidate in candidates
                       for recovery, countersteer in templates])


class SafetyFilterV5(v4.SafetyFilterV4):
    """Keep v4's onboard observer and memory; change only explicit v5 methods."""

    def __init__(self):
        if v4.DUAL_BRAKE_PREDICTION:
            raise ValueError("v5 ablations require the selected v4 single-response predictor")
        super().__init__()

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
        templates = v4.recovery_templates(recoveries)
        count, nrec = len(candidates), len(templates)
        bank = sequence_bank(candidates, templates)
        template_names = ["countersteer" if counter else "v3" for _, counter in templates]
        continuation = (v3.continuation(self.plan)
                        if self.plan is not None and len(self.plan) > 1 else None)
        sequences = (np.concatenate((bank, continuation[None]), axis=0)
                     if continuation is not None else bank)
        dual_count, fast_rejections = 0, 0
        rollout = v3.rollout_seq(snap, self.actuators, sequences)
        first, clear = self._evaluate(snap, rollout)
        policy_ok = np.isinf(first[:nrec])
        policy_room = np.where(policy_ok, clear[:nrec], -np.inf).max()
        policy_feedback_evaluated = bool(
            POLICY_FEEDBACK_PRESERVATION and self.mode == "nominal"
            and policy_room < v2.TRIGGER_MARGIN_M and np.isinf(first).any())
        if policy_feedback_evaluated:
            # Test recoverability of the proposed learned action BEFORE replacing
            # it. Only four backups of row zero are evaluated; failed proposals
            # never enter the original alternative or fallback rankings.
            from safety_feedback import feedback_bank as policy_backups
            policy_rollout, policy_sequences, names = policy_backups(
                snap, self.actuators, candidates[:1],
                cfg.UPDATE_RATE if ONE_DECISION_COMMIT else v2.COMMIT_S)
            proposed_first, proposed_clear = self._evaluate(snap, policy_rollout)
            proposed_margin = np.where(np.isinf(proposed_first), proposed_clear, -np.inf)
            chosen_backup = int(np.argmax(proposed_margin))
            if proposed_margin[chosen_backup] >= v2.TRIGGER_MARGIN_M:
                diagnostics = {
                    "policy_safe": True, "any_safe": True,
                    "policy_feedback_evaluated": True, "policy_feedback_preserved": True,
                    "policy_feedback_original_margin": float(policy_room),
                    "recovery_template": names[chosen_backup],
                    "continuation_checked": continuation is not None,
                    "continuation_ok": bool(continuation is not None and np.isinf(first[-1])),
                    "verified_policy_priority_enabled": bool(PREFER_CERTIFIED_POLICY),
                    "verified_policy_preserved": False, "continuation_override_prevented": False,
                    "feedback_backups_enabled": bool(FEEDBACK_BACKUPS),
                    "feedback_backups_evaluated": False,
                    "sideslip_rescue_enabled": bool(SIDESLIP_RESCUE),
                    "sideslip_rescue_evaluated": False, "sideslip_rescue_admitted": False,
                }
                return self._pass_checked(env, action, proposed_margin[chosen_backup:chosen_backup + 1],
                                          "policy feedback", policy_sequences[0, chosen_backup], diagnostics)
        # Optional conservative integration: preserve every original hard-safe
        # choice, including the separately checked continuation. This gate is
        # an engineering ablation, not a consequence of either paper's theorem.
        feedback_needed = (not np.isinf(first).any() if FEEDBACK_ONLY_INFEASIBLE
                           else self.mode == "recovery" or policy_room < v2.TRIGGER_MARGIN_M)
        feedback_used = bool(FEEDBACK_BACKUPS and feedback_needed)
        def append_backups(feedback, feedback_sequences, names, fb_first, fb_clear):
            nonlocal bank, rollout, first, clear, sequences, nrec
            decisions = bank.shape[1]
            old_columns = count * nrec
            bank = np.concatenate((bank.reshape(count, nrec, decisions, 2),
                                   feedback_sequences), axis=1).reshape(-1, decisions, 2)

            def merge_columns(old, extra):
                # Preserve candidate-major/recovery-minor ordering, including
                # the separately checked continuation as the final column.
                tail = old.shape[2:]
                merged = np.concatenate((old[:, :old_columns].reshape(
                    (len(old), count, nrec) + tail), extra.reshape(
                    (len(extra), count, len(names)) + tail)), axis=2)
                merged = merged.reshape((len(old), -1) + tail)
                return (np.concatenate((merged, old[:, old_columns:]), axis=1)
                        if continuation is not None else merged)

            rollout = cc.Rollout(merge_columns(rollout.positions, feedback.positions),
                                 merge_columns(rollout.headings, feedback.headings),
                                 merge_columns(rollout.speeds, feedback.speeds), rollout.times)
            first = merge_columns(first[None], fb_first[None])[0]
            clear = merge_columns(clear[None], fb_clear[None])[0]
            sequences = (np.concatenate((bank, continuation[None]), axis=0)
                         if continuation is not None else bank)
            nrec += len(names)
            template_names.extend(names)

        if feedback_used:
            # BC-MPC / backup-CBF inspired feedback predictions; retain the
            # original bank and all its hard checks. New plans are additional.
            from safety_feedback import feedback_bank
            feedback, feedback_sequences, names = feedback_bank(
                snap, self.actuators, candidates,
                cfg.UPDATE_RATE if ONE_DECISION_COMMIT else v2.COMMIT_S)
            fb_first, fb_clear = self._evaluate(snap, feedback)
            append_backups(feedback, feedback_sequences, names, fb_first, fb_clear)

        sideslip_evaluated = bool(SIDESLIP_RESCUE and not np.isinf(first).any())
        sideslip_admitted = False
        if sideslip_evaluated:
            from safety_course_backup import feedback_bank as course_bank
            extra, extra_sequences, names = course_bank(
                snap, self.actuators, candidates,
                cfg.UPDATE_RATE if ONE_DECISION_COMMIT else v2.COMMIT_S)
            extra_first, extra_clear = self._evaluate(snap, extra)
            sideslip_admitted = bool(np.isinf(extra_first).any())
            # Failed extra trajectories must not alter the old latest-contact
            # ranking, soft recovery, or last-certificate behavior.
            if sideslip_admitted:
                append_backups(extra, extra_sequences, names, extra_first, extra_clear)
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
            "countersteer_recovery_enabled": bool(v4.COUNTERSTEER_RECOVERY),
            "dual_brake_sequences": dual_count,
            "one_decision_commit": bool(ONE_DECISION_COMMIT),
            "soft_recovery_enabled": bool(SOFT_RECOVERY),
            "feedback_backups_enabled": bool(FEEDBACK_BACKUPS),
            "feedback_only_infeasible": bool(FEEDBACK_ONLY_INFEASIBLE),
            "feedback_backups_evaluated": feedback_used,
            "sideslip_rescue_enabled": bool(SIDESLIP_RESCUE),
            "sideslip_rescue_evaluated": sideslip_evaluated,
            "sideslip_rescue_admitted": sideslip_admitted,
            "verified_policy_priority_enabled": bool(PREFER_CERTIFIED_POLICY),
            "verified_policy_preserved": False,
            "continuation_override_prevented": False,
            "policy_feedback_evaluated": policy_feedback_evaluated,
            "policy_feedback_preserved": False,
            "fast_brake_rejections": fast_rejections,
        }

        def checked_policy(why):
            index = int(best_rec[0])
            details = dict(diagnostics, recovery_template=(
                template_names[index]))
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
            prefer_policy = bool(PREFER_CERTIFIED_POLICY and self.mode == "recovery"
                                 and v3.RECOVERY_REFERENCE == "policy" and 0 in pool)
            use_continuation = False
            if cont_ok and cont_clear >= floor:
                command = continuation[0]
                throttle = -1.5 if np.isnan(command[1]) else command[1]
                cont_score = ((command[0] - reference[0]) ** 2
                              + v2.W_THROTTLE * (throttle - reference[1]) ** 2 - v3.W_CONTINUE)
                continuation_would_win = not len(pool) or cont_score <= score.min()
                use_continuation = continuation_would_win and not prefer_policy
                diagnostics["continuation_override_prevented"] = bool(
                    prefer_policy and continuation_would_win
                    and not np.allclose(command, action, equal_nan=True))
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
                if prefer_policy:
                    # Preserve recovery bookkeeping, but execute the already
                    # accepted zero-deviation action and retain its own plan.
                    why = "verified policy"
                    diagnostics["verified_policy_preserved"] = True
                chosen_clearance = float(margin[chosen])
                template_name = template_names[rec]
        elif traffic and (safe & is_brake).any():
            pool = indices[safe & is_brake]
            chosen = int(pool[np.argmin(distance(pool))])
            rec = int(best_rec[chosen])
            out = candidates[chosen].copy()
            plan, why = bank[chosen * nrec + rec].copy(), "brake"
            chosen_clearance, template_name = float(margin[chosen]), template_names[rec]
        elif SOFT_RECOVERY:
            # Wabersich & Zeilinger (2023), Eq. (9): recovery by constraint
            # slacks before policy proximity. This sampled adaptation has no
            # terminal CBF and cannot inherit their convergence guarantee.
            from safety_recovery import evaluate, choose
            metrics = evaluate(snap, rollout)
            selected = choose(sequences, metrics, reference)
            out, plan, why = sequences[selected, 0].copy(), None, "soft recovery"
            chosen_clearance = float(metrics.clear[selected])
            template_name = "soft continuation" if selected == len(bank) else "soft candidate"
            diagnostics.update({"soft_violation": float(metrics.violation[selected]),
                                "soft_immediate_violation": float(metrics.immediate_violation[selected]),
                                "soft_sequence_index": int(selected)})
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
