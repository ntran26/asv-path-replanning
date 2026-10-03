"""V6 development candidate: trajectory search and evidence-gated target views.

Keeps selected v4's observer, perception, actions and safety checks. The v4
selection implementation is deliberately frozen/copied here so its evaluated
source and behavior remain independently reproducible. No v5 mechanism is
implicitly enabled. The defaults reproduce the broad_v6_50 evaluated preset.
Select this experimental candidate with runtime constants.SAFETY_VERSION = 6;
the environment's existing default and established V4 filter are unchanged.
Calibration, fitted-hull geometry and track history remain disabled ablations.

Data-driven prediction is motivated by Wabersich & Zeilinger (2021),
"A predictive safety filter for learning-based control of constrained nonlinear
 dynamical systems", Sec. 4.2, https://arxiv.org/abs/1812.05506 . Calibration
is an empirical effective response, not their uncertainty set or guarantee.
The measured-response prediction-error fit follows Ljung, "Prediction Error
Estimation Methods", Sec. 1, technical report (2001):
https://liu.diva-portal.org/smash/get/diva2:316694/FULLTEXT01.pdf .
The optional geometry adapter uses the existing L-shape hull fit inspired by
Zhang et al. (2017), https://doi.org/10.1109/IVS.2017.7995698, and attributes
LiDAR returns to the target's fitted footprint rather than a surrounding disk.
Optional provisional targets and bounded cross-entropy trajectory search are
separate ablations; their modules document method citations and limits.
See planning/SAFETY_LAYER_V6_PLAN.md and the development calibration artifacts.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4

CALIBRATED_BRAKING = False
FITTED_HULL_PERCEPTION = False
FITTED_TRACK_CENTRE = True
HULL_RETURN_MASK = True
PROVISIONAL_TRACKS = True
PROVISIONAL_MOTION_EVIDENCE = True
TRAJECTORY_SEARCH = True
TRACK_HISTORY = False
# Latent-start, measured-state prediction-error fit on 53 development pulses:
# effective deceleration 0.4652061638 m/s^2 / identified unit-efficiency thrust.
# Source: results/safety_dev/brake_calibration/latent_delay_profile/.
# This is an effective response at -24 RPM only; delay and force are not
# separately identifiable. Zero is an equivalent immediate impulse model.
BRAKE_EFFICIENCY = 0.49389942357454414
BRAKE_DELAY_S = 0.0
CALIBRATION_RPM = -24.0


class SafetyFilterV6(v4.SafetyFilterV4):
    """Experimental v4-compatible filter with an explicit prediction backend."""

    def __init__(self):
        super().__init__()
        self._search_rng = np.random.default_rng(0)
        if CALIBRATED_BRAKING and v4.DUAL_BRAKE_PREDICTION:
            raise ValueError("Calibrated and old dual-response experiments are separate")
        if CALIBRATED_BRAKING and not (0.0 < BRAKE_EFFICIENCY <= 1.0):
            raise ValueError("A development-fitted brake response is required")
        if CALIBRATED_BRAKING and not (np.isfinite(BRAKE_DELAY_S) and BRAKE_DELAY_S >= 0.0):
            raise ValueError("Brake response delay must be finite and nonnegative")
        if CALIBRATED_BRAKING and v2.ASTERN_RPM != CALIBRATION_RPM:
            raise ValueError("Brake calibration is identified only at the recorded RPM")
        if FITTED_HULL_PERCEPTION:
            import constants as cfg
            from safety_hull_perception import HullSafetyPerception
            self.perception = HullSafetyPerception(
                memory_frames=int(round(v2.MEMORY_S / cfg.UPDATE_RATE)),
                fitted_track_centre=FITTED_TRACK_CENTRE,
                hull_return_mask=HULL_RETURN_MASK)
        if PROVISIONAL_TRACKS:
            from safety_provisional_tracks import ProvisionalTrackPerception
            self.perception = ProvisionalTrackPerception(
                self.perception, require_motion_evidence=PROVISIONAL_MOTION_EVIDENCE)
        if TRACK_HISTORY:
            from safety_track_history import TrackHistoryPerception
            self.perception = TrackHistoryPerception(self.perception)

    def _rollout(self, snap, actuators, sequences):
        if not CALIBRATED_BRAKING:
            return v3.rollout_seq(snap, actuators, sequences)
        from safety_prediction import rollout_seq
        return rollout_seq(snap, actuators, sequences,
                           brake_efficiency=BRAKE_EFFICIENCY,
                           brake_delay_s=BRAKE_DELAY_S)

    def _hold_back_gain_s(self):
        """Required delay gain; subclasses may isolate a fallback ablation."""
        return v2.HOLD_BACK_GAIN_S

    def _accept_search(self, env, action, result, diagnostics):
        """Issue the first command of the exact checked plan and retain its tail."""
        fixed = result.diagnostics["fixed_policy"]
        if self.mode == "recovery":
            self.recovery_steps += 1
        if not fixed:
            self.mode = "recovery"
        self.plan = result.plan.copy()
        self.uncertified_steps = 0
        out = self.plan[0].copy()
        brake = bool(np.isnan(out[1]))
        env._v2_brake = brake
        if brake:
            out[1] = -1.
        out = out.astype(np.float32)
        self.actuators.issue(env, float(out[0]))
        changed = brake or not np.allclose(out, action)
        self.last = dict(diagnostics, mode=self.mode,
                         why="searched policy" if fixed else "searched escape",
                         checked_clearance=float(result.clearance),
                         changed=bool(changed), brake=brake, chosen=out.tolist(),
                         recovery_steps=self.recovery_steps, uncertified_steps=0,
                         nominal_plan_retained=self.mode == "nominal",
                         plan_steps=len(self.plan))
        return out, bool(changed)

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
            self.last = {"mode": "idle", "plan_steps": 0,
                         "selection_floor": None, "selection_best_clearance": None, "selection_branch": None}
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
        bank = v4.sequence_bank(candidates, templates)
        continuation = (v3.continuation(self.plan)
                        if self.plan is not None and len(self.plan) > 1 else None)
        sequences = (np.concatenate((bank, continuation[None]), axis=0)
                     if continuation is not None else bank)
        dual_count, fast_rejections = 0, 0
        if v4.DUAL_BRAKE_PREDICTION:
            from safety_prediction import evaluate_sequences
            envelope = evaluate_sequences(snap, self.actuators, sequences, self._evaluate)
            first, clear = envelope.first, envelope.clear
            dual_count = envelope.dual_count
            fast_rejections = int((np.isinf(envelope.weak_first)
                                   & ~np.isinf(envelope.fast_first)).sum())
        else:
            rollout = self._rollout(snap, self.actuators, sequences)
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
            "selection_floor": None, "selection_best_clearance": None, "selection_branch": None,
            "continuation_checked": continuation is not None,
            "continuation_ok": cont_ok,
            "continuation_clearance": float(cont_clear),
            "policy_safe": bool(safe[0]), "any_safe": bool(safe.any() or cont_ok),
            "n_safe_candidates": int(safe.sum()),
            "countersteer_recovery_enabled": bool(v4.COUNTERSTEER_RECOVERY),
            "dual_brake_sequences": dual_count,
            "fast_brake_rejections": fast_rejections,
            "calibrated_braking": bool(CALIBRATED_BRAKING),
            "fitted_hull_perception": bool(FITTED_HULL_PERCEPTION),
            "provisional_tracks": bool(PROVISIONAL_TRACKS),
            "provisional_motion_evidence": bool(PROVISIONAL_MOTION_EVIDENCE),
            "trajectory_search": bool(TRAJECTORY_SEARCH),
            "track_history": bool(TRACK_HISTORY),
        }

        # CEM augments the finite primitive bank; it never relaxes a check.
        # Zheng et al. (2022), https://arxiv.org/abs/2102.12124 . Search before
        # accepting a millimetre-clearance fallback, while escape may exist.
        if TRAJECTORY_SEARCH and margin[0] < v2.TRIGGER_MARGIN_M:
            from safety_trajectory_search import search
            policy_seeds = bank[:nrec]
            if continuation is not None:
                policy_seeds = np.concatenate((policy_seeds, continuation[None]))
            result = search(snap, self.actuators, action, policy_seeds,
                            self._rollout, self._evaluate, self._search_rng,
                            fixed_policy=True, traffic=traffic)
            diagnostics["policy_search"] = result.diagnostics
            if result.plan is not None:
                return self._accept_search(env, action, result, diagnostics)
            best_original = max(float(margin.max()), cont_clear if cont_ok else -np.inf)
            if best_original < v2.TRIGGER_MARGIN_M:
                # Include strongest primitive seeds even if none is feasible.
                order = np.lexsort((-clear.ravel(), -first.ravel()))[:16]
                seeds = bank[order]
                if continuation is not None:
                    seeds = np.concatenate((seeds, continuation[None]))
                result = search(snap, self.actuators, action, seeds,
                                self._rollout, self._evaluate, self._search_rng,
                                fixed_policy=False, traffic=traffic)
                diagnostics["escape_search"] = result.diagnostics
                if result.plan is not None:
                    return self._accept_search(env, action, result, diagnostics)

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
            diagnostics.update(selection_branch="ordinary", selection_floor=float(floor),
                               selection_best_clearance=float(best))
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
            if latest[chosen] - latest[0] < self._hold_back_gain_s():
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
