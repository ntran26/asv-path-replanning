"""Experimental V20: V16 plus a committed stop-terminal contingency.

V16 decides unchanged wherever its own checker passes a plan. At every
decision V20 certifies a short contingency that brings the predicted vessel to
rest with the rudder centred (safety_v20_contingency.py) and commits it. Only
in V16's two unchecked branches -- ``last certificate`` (a stored plan whose
recheck fails) and ``no escape`` (SAC's command passes unchecked) -- and for
unknown branch labels does V20 replace the command, issuing the first
certified option among: SAC's command, V16's command, V2's command grid
ordered by distance to SAC, and the committed contingency. If none is
certified, it follows the committed contingency anyway, or brakes with the
rudder centred, and labels the decision out of contract.

Saved-trace evidence for the target (21 of 22 V16 development contacts follow
unchecked decisions) and the method sources are in planning/SAFETY_V20_PLAN.md
and results/safety_dev/v20_development/phase1_saved_cascade_audit/. The
committed-backup structure follows model-predictive shielding (Bastani,
https://arxiv.org/abs/1905.10691) and gatekeeper (Agrawal, Chen and Panagou,
https://arxiv.org/abs/2211.14361); fail-safe stops follow Magdici and Althoff
(https://doi.org/10.1109/ITSC.2016.7795594). The finite tails, nominal model
and statistical allowances give a conditional, contract-based check, not a
collision-avoidance guarantee. No truth, scenario identity or outcome is read.

``certified_fallback=False`` returns V16's decision and state unchanged while
still logging certificates (parity mode).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

import constants as cfg
import safety_v2 as v2
import safety_v16 as v16
import safety_v20_contingency as sc
import safety_v20_tubes as tubes
import ship

CERTIFIED_FALLBACK = True
ALLOWANCE_TABLES = "calibrated"
OUT_OF_CONTRACT = "committed_then_stop"
TAIL_FAMILY = "stop"
ENFORCEMENT = "unchecked_only"

# V16 branches whose issued plan hard-passed the inherited checker at that decision.
CHECKED_BRANCHES = frozenset({
    "nominal", "handback", "turn", "continue", "brake", "searched policy",
    "searched escape", "policy prefix searched", "policy prefix repaired",
    "policy margin dominates", "feasible policy backup",
    "feasible policy preference"})
IDLE = "idle"
# V3/V7 helpers accept stored plans of at most the inherited horizon.
V16_PLAN_DECISIONS = int(math.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))


def _v16_plan(sequence):
    """The certified contingency in V16's plan encoding, truncated to its horizon."""
    return np.asarray(sequence, float)[:V16_PLAN_DECISIONS].copy()


def _same_command(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.isnan(a[1]) != np.isnan(b[1]):
        return False
    return abs(a[0] - b[0]) <= 1e-9 and (np.isnan(a[1]) or abs(a[1] - b[1]) <= 1e-9)


class SafetyFilterV20(v16.SafetyFilterV16):
    def __init__(self, *, certified_fallback=None, allowance_tables=None,
                 hold_horizon_s=None, out_of_contract=None, tail_family=None,
                 enforcement=None, **v16_options):
        super().__init__(**v16_options)
        self.certified_fallback = (CERTIFIED_FALLBACK if certified_fallback is None
                                   else bool(certified_fallback))
        tables = ALLOWANCE_TABLES if allowance_tables is None else str(allowance_tables)
        if tables not in ("calibrated", "none"):
            raise ValueError("allowance_tables must be 'calibrated' or 'none'")
        self.allowance_tables = tables
        self.own_table = tubes.OWN_TABLE if tables == "calibrated" else None
        self.target_table = tubes.TARGET_TABLE if tables == "calibrated" else None
        self.hold_horizon_s = float(sc.HOLD_HORIZON_S if hold_horizon_s is None else hold_horizon_s)
        if not math.isfinite(self.hold_horizon_s) or self.hold_horizon_s < 0.0:
            raise ValueError("hold_horizon_s must be finite and nonnegative")
        mode = OUT_OF_CONTRACT if out_of_contract is None else str(out_of_contract)
        if mode not in ("committed_then_stop", "v16"):
            raise ValueError("out_of_contract must be 'committed_then_stop' or 'v16'")
        self.out_of_contract = mode
        family = TAIL_FAMILY if tail_family is None else str(tail_family)
        if family not in sc.TAIL_FAMILIES:
            raise ValueError("tail_family must be 'stop' or 'extended'")
        self.tail_family = family
        enforce = ENFORCEMENT if enforcement is None else str(enforcement)
        if enforce not in ("unchecked_only", "gatekeeper"):
            raise ValueError("enforcement must be 'unchecked_only' or 'gatekeeper'")
        self.enforcement = enforce
        self.v20_committed = None           # (DECISIONS, 2) contingency starting at the next decision
        self._v20_previous_tracks = None

    # ------------------------------------------------------------------
    def _contract_residuals(self, snap):
        """One-decision constant-velocity residual for tracks seen at both decisions."""
        current = {int(t.id): t for t in snap.tracks}
        out = []
        if self._v20_previous_tracks:
            for tid, (pos, vel) in self._v20_previous_tracks.items():
                if tid in current:
                    predicted = pos + 0.5 * vel
                    out.append(float(np.hypot(*(predicted - current[tid].position))))
        self._v20_previous_tracks = {tid: (np.asarray(t.position, float).copy(),
                                           np.asarray(t.velocity, float).copy())
                                     for tid, t in current.items()}
        return out

    def _observed_free(self, env, snap, sequence):
        try:
            ro = sc.rollout_states(snap, self._v20_before, np.asarray(sequence, float)[None], *sc.WEAK)
            origin = np.array([snap.x, snap.y]) + ship.LIDAR_OFFSET_M * np.array(
                [math.sin(snap.heading), math.cos(snap.heading)])
            ranges = np.asarray(getattr(env, "gated_ranges", env.lidar.ranges), dtype=float)
            return sc.observed_free_fraction(ro, 0, origin, math.degrees(snap.heading),
                                             env.lidar.bearings, ranges, cfg.LIDAR_RANGE,
                                             cfg.LIDAR_MIN_RANGE)
        except Exception as error:  # diagnostic only; never affects the decision
            return {"error": type(error).__name__}

    def _issue_replacement(self, env, policy, command, sequence, source, details):
        """Replace V16's issued command; keep actuator model and V16 state consistent."""
        self.actuators = copy.deepcopy(self._v20_before)
        self.actuators.issue(env, float(command[0]))
        brake = bool(np.isnan(command[1]))
        env._v2_brake = brake
        out = np.array([command[0], -1.0 if brake else command[1]], dtype=np.float32)
        changed = brake or not np.allclose(out, policy)
        if source == "sac":
            self._release()
            self.mode = "nominal"
        else:
            self.mode = "recovery"
        self.plan = None if sequence is None else _v16_plan(sequence)
        self.uncertified_steps = 0
        self.last.update(details)
        self.last.update(why=f"v20 {source}", changed=bool(changed), brake=brake,
                         chosen=out.tolist(), mode=self.mode,
                         plan_steps=0 if self.plan is None else len(self.plan))
        return out, bool(changed)

    def _filter(self, env, action):
        policy = np.clip(np.asarray(action, dtype=float).reshape(2), -1., 1.).astype(np.float32)
        if not np.isfinite(policy).all():
            raise ValueError("Policy action must be finite")
        self._v20_before = copy.deepcopy(self.actuators)
        out, changed = super()._filter(env, policy)
        started = time.perf_counter()
        snap = getattr(self, "_observer_snapshot", None)
        why = self.last.get("why", IDLE)
        brake = bool(getattr(env, "_v2_brake", False))
        v16_command = np.array([float(out[0]), np.nan if brake else float(out[1])])
        sac_command = np.asarray(policy, dtype=float)
        details = {"v20_certified_fallback": self.certified_fallback,
                   "v20_allowance_tables": self.allowance_tables,
                   "v20_tail_family": self.tail_family, "v20_enforcement": self.enforcement,
                   "v20_v16_why": why, "v20_v16_command": np.asarray(out).tolist(),
                   "v20_v16_brake": brake}
        if bool(getattr(env, "command_rate_limit", False)) or snap is None:
            # The contingency predictor does not model the optional command limiter.
            self.v20_committed = None
            details.update(v20_level="not_evaluated",
                           v20_skip_reason="command limit" if snap is not None else "no snapshot")
            self.last.update(details)
            return out, changed

        checker = sc.ContingencyChecker(snap, self._v20_before, self.own_table,
                                        self.target_table, self.hold_horizon_s,
                                        sc.TAIL_FAMILIES[self.tail_family])
        residuals = self._contract_residuals(snap)
        rho_half = float(sc._lookup(self.target_table, np.array([0.5]))[0])
        details.update(v20_contract_residual_max=max(residuals) if residuals else None,
                       v20_contract_violations=int(sum(r > rho_half for r in residuals)) if self.target_table else None)

        previous = self.v20_committed
        cert_previous = None
        if previous is not None:
            cert_previous = checker.certify_sequence(previous)
            details.update(v20_previous_recheck=bool(cert_previous.certified),
                           v20_previous_slack=cert_previous.slack)
        cert_v16 = checker.certify(v16_command)
        details.update(v20_v16_certified=bool(cert_v16.certified), v20_v16_slack=cert_v16.slack)
        if _same_command(sac_command, v16_command):
            cert_sac = cert_v16
        else:
            cert_sac = None

        checked = why == IDLE or why in CHECKED_BRANCHES
        level, source, cert, command = None, None, None, None
        if not self.certified_fallback or (checked and cert_v16.certified):
            level = "v16_unchanged_certified" if cert_v16.certified else "v16_unchanged_finite_only"
            cert = cert_v16 if cert_v16.certified else None
        elif checked and self.enforcement == "unchecked_only":
            level = "v16_unchanged_finite_only"
        elif checked:
            # Gatekeeper: V16's own checked command is not certified. Keep V16's
            # intent with the nearest certified command; otherwise the
            # committed contingency; otherwise V16's checked command.
            reference = v16_command if np.isfinite(v16_command[1]) else np.array([v16_command[0], -1.0])
            grid = sc.projection_candidates(reference)
            batch = checker.certify_many(grid)
            details["v20_candidates_evaluated"] = len(grid)
            for candidate, certificate in zip(grid, batch):
                if certificate.certified:
                    source, cert, command = "gatekeeper", certificate, candidate
                    break
            if cert is None and cert_previous is not None and cert_previous.certified:
                source, cert, command = "committed", cert_previous, previous[0]
            if cert is not None:
                level = "gatekeeper_replaced" if source == "gatekeeper" else "committed_continuation"
            else:
                level = "gatekeeper_uncertified_v16"
        else:
            options = [("sac", sac_command, cert_sac), ("v16", v16_command, cert_v16)]
            options += [("projection", c, None) for c in sc.projection_candidates(sac_command)]
            pending = [i for i, (_, _, known) in enumerate(options) if known is None]
            batch = checker.certify_many(np.array([options[i][1] for i in pending])) if pending else []
            certificates = {i: c for i, c in zip(pending, batch)}
            details["v20_candidates_evaluated"] = len(pending)
            for i, (name, candidate, known) in enumerate(options):
                certificate = known if known is not None else certificates[i]
                if name == "sac":
                    cert_sac = certificate
                if certificate.certified:
                    source, cert, command = name, certificate, candidate
                    break
            if cert is None and cert_previous is not None and cert_previous.certified:
                source, cert, command = "committed", cert_previous, previous[0]
            if cert is not None:
                level = {"sac": "replaced_by_sac", "v16": "replaced_by_v16_certified",
                         "projection": "replaced_by_projection",
                         "committed": "committed_continuation"}[source]
            elif self.out_of_contract == "v16":
                level = "out_of_contract_v16"
            elif previous is not None:
                level, source, command = "out_of_contract_committed", "committed", previous[0]
            else:
                level, source, command = "out_of_contract_stop", "stop", np.array([0.0, np.nan])
        if cert_sac is not None:
            details.update(v20_sac_certified=bool(cert_sac.certified), v20_sac_slack=cert_sac.slack)

        # Commitment for the next decision.
        if cert is not None:
            sequence = cert.sequence
            self.v20_committed = sc.shift_sequence(sequence)
        elif level == "out_of_contract_committed":
            sequence = previous
            self.v20_committed = sc.shift_sequence(previous)
        else:
            sequence = None
            self.v20_committed = None
        details.update(v20_level=level, v20_source=source,
                       v20_committed_slack=None if cert is None else cert.slack,
                       v20_committed_rest_s=None if cert is None else cert.rest_time_s,
                       v20_committed_tail=None if cert is None else cert.tail_index,
                       v20_committed_parts=None if cert is None else cert.diagnostics)
        if sequence is not None:
            details["v20_observed_free"] = self._observed_free(env, snap, sequence)

        replace = (self.certified_fallback and command is not None
                   and not (source == "v16" or _same_command(command, v16_command)))
        details["v20_seconds"] = time.perf_counter() - started
        if replace:
            return self._issue_replacement(env, policy, command, sequence, source, details)
        if self.certified_fallback and level in ("replaced_by_v16_certified", "replaced_by_sac",
                                                 "replaced_by_projection", "committed_continuation",
                                                 "gatekeeper_replaced"):
            # Same command as V16 in an unchecked branch: keep it, but follow the
            # certified contingency instead of V16's failed stored plan.
            self.plan = _v16_plan(sequence)
            self.uncertified_steps = 0
        self.last.update(details)
        return out, changed
