"""Experimental V10 with observed-motion admission and bounded persistence.

TrackPersistencePerception adds safety-only motion hypotheses without changing
the policy observation or shared tracker. Its extent/motion/visibility method
sources and limitations are in safety_track_persistence.py: Zhang et al. (2017),
DOI10.1109/IVS.2017.7995698; Yoon et al., arXiv:1809.06972, Sec. III-C; and
Nuss et al., arXiv:1605.02406. The eight-second coast is an explicit engineering
setting, not an identified existence probability or a bounded reachable set.

Optional one-decision policy-prefix search is disabled by default. After an
intervention proposal or "no escape", it may preserve SAC with a newly checked
whole backup, fixing only the currently issued .5 s action. This adapts the
first-action objective of Wabersich & Zeilinger (2021), Sec. 4.1, Eq. (5):
https://arxiv.org/html/1812.05506v4#S4.SS1 . CEM optimizer attribution is Zheng
et al. (2022), https://arxiv.org/abs/2102.12124v3 . Full method details and finite
search limitations are in safety_policy_prefix.py. No check/margin is relaxed.

Failed extra searches retain V10's action, plan and fallback counters. Neither
persisted tracks nor a sampled feasible plan guarantee real collision avoidance
or goal preservation. No truth state, scenario ID or outcome enters selection.
Existing filters and environment dispatch are untouched by this module.
"""
from __future__ import annotations

import copy

import numpy as np

import safety_v2 as v2
import safety_v4 as v4
import safety_v10 as v10
from safety_track_persistence import TrackPersistencePerception


TRACK_ADMISSION = True
TRACK_PERSISTENCE = True
TRACK_MAX_COAST_S = 8.0
POLICY_PREFIX_SEARCH = False


class SafetyFilterV11(v10.SafetyFilterV10):
    def __init__(self, *, enable_admission=None, enable_persistence=None,
                 max_coast_s=None, policy_prefix_search=None, prefix_rng_seed=0,
                 **v10_options):
        super().__init__(**v10_options)
        self.enable_admission = TRACK_ADMISSION if enable_admission is None else bool(enable_admission)
        self.enable_persistence = TRACK_PERSISTENCE if enable_persistence is None else bool(enable_persistence)
        self.max_coast_s = float(TRACK_MAX_COAST_S if max_coast_s is None else max_coast_s)
        if not np.isfinite(self.max_coast_s) or self.max_coast_s < 0.:
            raise ValueError("max_coast_s must be finite and nonnegative")
        self.policy_prefix_search = (POLICY_PREFIX_SEARCH if policy_prefix_search is None
                                     else bool(policy_prefix_search))
        self.track_persistence = None
        if self.enable_admission or self.enable_persistence:
            self.track_persistence = TrackPersistencePerception(
                self.perception, enable_admission=self.enable_admission,
                enable_persistence=self.enable_persistence, max_coast_s=self.max_coast_s)
            self.perception = self.track_persistence
        # Independent of the inherited bank/escape search and global RNG.
        self._prefix_rng = np.random.default_rng(prefix_rng_seed)

    def _prefix_search_request(self):
        """Default request preserves the original trigger-margin search."""
        return True, None, "default_trigger_margin"

    def _filter(self, env, action):
        policy = np.clip(np.asarray(action, dtype=float).reshape(2), -1., 1.).astype(np.float32)
        if not np.isfinite(policy).all():
            raise ValueError("Policy action must be finite")
        before = copy.deepcopy(self.actuators)
        out, changed = super()._filter(env, policy)
        parent_why = self.last.get("why", "idle")
        details = {
            "v11_track_admission": self.enable_admission,
            "v11_track_persistence": self.enable_persistence,
            "v11_policy_prefix_search_enabled": self.policy_prefix_search,
            "v11_policy_prefix_search_checked": False,
            "v11_prefix_request_checked": False,
            "v11_policy_prefix_preserved": False,
            "v11_parent_why": parent_why,
            "v11_parent_changed": bool(changed),
            "v11_parent_action": np.asarray(out).tolist(),
            "v11_parent_brake": bool(getattr(env, "_v2_brake", False)),
            "track_persistence": copy.deepcopy(
                self.track_persistence.last_track_persistence_stats
                if self.track_persistence is not None else {}),
        }
        self.last.update(details)
        if not self.policy_prefix_search or not (changed or parent_why == "no escape"):
            return out, changed
        if bool(getattr(env, "command_rate_limit", False)):
            self.last["v11_prefix_skip_reason"] = "unmodelled command limit"
            return out, changed
        snap = getattr(self, "_observer_snapshot", None)
        if snap is None:
            self.last["v11_prefix_skip_reason"] = "no onboard snapshot"
            return out, changed
        allowed, minimum_clearance, request_reason = self._prefix_search_request()
        details.update(v11_prefix_request_checked=True,
                       v11_prefix_minimum_clearance=minimum_clearance,
                       v11_prefix_request_reason=request_reason)
        if not allowed:
            self.last.update(details, v11_prefix_skip_reason=request_reason)
            return out, changed
        traffic = any(np.hypot(*(track.position - snap.position)) < v2.ENGAGE_RANGE_M
                      for track in snap.tracks)
        rollout, evaluate = self._rollout, self._evaluate
        if v4.DUAL_BRAKE_PREDICTION:
            from safety_prediction import evaluate_sequences

            def rollout(snapshot, actuators, sequences):
                return evaluate_sequences(snapshot, actuators, sequences, self._evaluate)

            def evaluate(snapshot, envelope):
                return envelope.first, envelope.clear

        from safety_policy_prefix import search
        result = search(snap, before, policy, self.plan, rollout, evaluate,
                        self._prefix_rng, traffic=traffic,
                        minimum_clearance=minimum_clearance)
        details.update(v11_policy_prefix_search_checked=True,
                       policy_prefix_search=copy.deepcopy(result.diagnostics))
        if result.plan is None:
            self.last.update(details)
            return out, changed
        parent_preserved = bool(self.last.get("v10_policy_preserved", False))
        out, changed = self._preserve_policy(env, policy, before, details,
                                             "policy prefix searched", result.plan,
                                             result.clearance)
        # The inherited helper records a V10 handback; attribute this one to V11.
        self.last.update(v10_policy_preserved=parent_preserved,
                         v11_policy_prefix_preserved=True)
        return out, changed
