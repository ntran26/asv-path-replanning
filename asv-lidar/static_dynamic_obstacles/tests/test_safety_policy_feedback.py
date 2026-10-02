"""Policy-only feedback preservation with synthetic predictions; no episodes."""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_feedback
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v5 as v5


POLICY = np.array([0.2, 0.1])
HEADS = ("edge_parallel", "current_heading", "port_30", "starboard_30")


def _rollout(tags):
    tags = np.asarray(tags, dtype=float)
    positions = np.stack((tags, np.zeros(len(tags))), axis=-1)[None]
    return cc.Rollout(np.repeat(positions, 2, axis=0), np.zeros((2, len(tags))),
                      np.zeros((2, len(tags))), np.array([0.125, 0.25]))


@pytest.fixture
def feedback_case(monkeypatch):
    for name in ("COUNTERSTEER_RECOVERY", "RETAIN_NOMINAL_PLAN", "DUAL_BRAKE_PREDICTION",
                 "FREE_SPACE_MEMORY", "MODEL_EGO_OBSERVER"):
        monkeypatch.setattr(v4, name, False)
    for name in ("ONE_DECISION_COMMIT", "SOFT_RECOVERY", "FEEDBACK_BACKUPS",
                 "FEEDBACK_ONLY_INFEASIBLE", "SIDESLIP_RESCUE", "PREFER_CERTIFIED_POLICY"):
        monkeypatch.setattr(v5, name, False)
    monkeypatch.setattr(v5, "POLICY_FEEDBACK_PRESERVATION", True)
    monkeypatch.setattr(cfg, "UPDATE_RATE", 0.5)
    monkeypatch.setattr(v2, "TRIGGER_MARGIN_M", 0.15)
    monkeypatch.setattr(v2, "ROOM_SLACK_M", 0.20)
    monkeypatch.setattr(v3, "RECOVERY_REFERENCE", "policy")

    def make(*, original="turn", mode="nominal", extra_first=None,
             extra_clear=None, stored=True):
        filter_ = v5.SafetyFilterV5()
        snap = cc.Snapshot(
            x=5.0, y=5.0, heading=0.0, u=0.1, v=0.0, r=0.0,
            tangent=np.array([0.0, 1.0]), right=np.array([1.0, 0.0]),
            centre=np.array([5.0, 5.0]), base_heading=0.0,
            lateral=0.0, remaining=20.0, points=np.empty((0, 2)))
        monkeypatch.setattr(filter_.perception, "snapshot", lambda env: snap)
        monkeypatch.setattr(filter_, "_threat_in_reach", lambda snap: True)
        monkeypatch.setattr(filter_, "_rejoin", lambda snap: np.zeros(2))
        captured = {"feedback_calls": 0, "original_evaluations": 0, "issued": []}
        monkeypatch.setattr(filter_.actuators, "issue",
                            lambda env, rudder: captured["issued"].append(rudder))
        filter_.mode = mode
        filter_.recovery_steps = 7
        filter_.uncertified_steps = 1
        if stored:
            filter_.plan = v3.plan_for((0.75, 0.8), (-1.0, 0.0))
        previous = None if filter_.plan is None else filter_.plan.copy()
        nrec = sum(throttle is not v2.BRAKE for _, throttle in v2.RECOVERY)
        first_extra = np.asarray(extra_first if extra_first is not None
                                 else [1.0, np.inf, np.inf, 0.0], dtype=float)
        clear_extra = np.asarray(extra_clear if extra_clear is not None
                                 else [5.0, 0.20, 0.45, -1.0], dtype=float)

        def original_rollout(snapshot, actuators, sequences):
            captured["original_sequences"] = sequences.copy()
            return _rollout(np.arange(len(sequences)))

        def feedback(snapshot, actuators, candidates, commit_s):
            captured["feedback_calls"] += 1
            # The new branch must only ask for backups of the exact policy,
            # never generate a competing action grid or shorten commitment.
            np.testing.assert_array_equal(candidates, POLICY[None])
            expected_commit = cfg.UPDATE_RATE if v5.ONE_DECISION_COMMIT else v2.COMMIT_S
            assert commit_s == expected_commit
            decisions = v3._decisions()[0]
            commit = int(round(commit_s / cfg.UPDATE_RATE))
            sequences = np.repeat(candidates[:, None, None, :], len(HEADS), axis=1)
            sequences = np.repeat(sequences, decisions, axis=2)
            for head in range(len(HEADS)):
                sequences[0, head, commit:] = (0.1 * (head + 1), 0.0)
            np.testing.assert_array_equal(sequences[0, :, :commit],
                                          np.tile(POLICY, (len(HEADS), commit, 1)))
            captured["feedback_sequences"] = sequences.copy()
            return _rollout(1000 + np.arange(len(HEADS))), sequences, HEADS

        def evaluate(snapshot, rollout):
            tags = rollout.positions[0, :, 0]
            if tags[0] >= 1000:
                np.testing.assert_array_equal(tags, 1000 + np.arange(len(HEADS)))
                return first_extra.copy(), clear_extra.copy()
            captured["original_evaluations"] += 1
            first = np.zeros(len(tags))
            clear = np.full(len(tags), -1.0)
            if original in ("turn", "low_policy"):
                first[3 * nrec + 1], clear[3 * nrec + 1] = np.inf, 0.4
            if original == "low_policy":
                first[1], clear[1] = np.inf, 0.14
            elif original == "nominal":
                first[1], clear[1] = np.inf, v2.TRIGGER_MARGIN_M
            elif original == "continuation":
                assert stored
                first[-1], clear[-1] = np.inf, 0.4
            elif original not in ("turn", "none"):
                raise AssertionError(original)
            return first, clear

        monkeypatch.setattr(v3, "rollout_seq", original_rollout)
        monkeypatch.setattr(safety_feedback, "feedback_bank", feedback)
        monkeypatch.setattr(filter_, "_evaluate", evaluate)
        return SimpleNamespace(filter=filter_, env=SimpleNamespace(_v2_brake=True),
                               captured=captured, previous=previous)

    return make


@pytest.mark.parametrize("original", ["turn", "low_policy", "continuation"])
def test_safe_policy_feedback_preserves_exact_action_and_discards_old_plan(feedback_case, original):
    case = feedback_case(original=original)
    out, changed = case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY.astype(np.float32))
    assert not changed and not case.filter.last["changed"]
    assert not case.env._v2_brake and not case.filter.last["brake"]
    assert case.filter.mode == "nominal"
    assert case.filter.plan is None
    assert case.filter.recovery_steps == case.filter.uncertified_steps == 0
    assert case.filter.last["why"] == "policy feedback"
    assert case.filter.last["policy_feedback_evaluated"]
    assert case.filter.last["policy_feedback_preserved"]
    # A larger clearance on an unsafe head must not be borrowed.
    assert case.filter.last["checked_clearance"] == 0.45
    assert case.captured["feedback_calls"] == 1
    assert case.captured["original_evaluations"] == 1
    assert case.captured["issued"] == [float(POLICY[0])]


@pytest.mark.parametrize("original,mode", [("nominal", "nominal"),
                                           ("none", "nominal"),
                                           ("turn", "recovery")])
def test_policy_feedback_gate_skips_unneeded_infeasible_and_recovery_cases(
        feedback_case, original, mode):
    case = feedback_case(original=original, mode=mode)
    case.filter.filter(case.env, POLICY)
    assert case.captured["feedback_calls"] == 0
    assert not case.filter.last["policy_feedback_evaluated"]
    assert not case.filter.last["policy_feedback_preserved"]
    if mode == "recovery":
        assert case.filter.mode == "recovery"
        assert case.filter.recovery_steps == 8


@pytest.mark.parametrize("original", ["turn", "low_policy", "continuation"])
@pytest.mark.parametrize("rejection", ["unsafe", "below_margin"])
def test_failed_extra_bank_preserves_original_selector_exactly(
        feedback_case, monkeypatch, original, rejection):
    first = [10.0, 20.0, 30.0, 40.0] if rejection == "unsafe" else [np.inf] * 4
    # Unsafe paths look tempting by clearance or contact time. They must not
    # enter any existing selection/fallback array. Safe but small-margin paths
    # must also be discarded rather than bypassing the acceptance threshold.
    clear = [5.0, 6.0, 7.0, 8.0] if rejection == "unsafe" else [0.01, 0.14, 0.149, 0.02]
    reference = feedback_case(original=original, extra_first=first, extra_clear=clear)
    monkeypatch.setattr(v5, "POLICY_FEEDBACK_PRESERVATION", False)
    expected, expected_changed = reference.filter.filter(reference.env, POLICY)
    assert reference.captured["feedback_calls"] == 0
    candidate = feedback_case(original=original, extra_first=first, extra_clear=clear)
    monkeypatch.setattr(v5, "POLICY_FEEDBACK_PRESERVATION", True)
    out, changed = candidate.filter.filter(candidate.env, POLICY)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(candidate.filter.plan, reference.filter.plan)
    assert changed == expected_changed
    assert candidate.env._v2_brake == reference.env._v2_brake
    assert candidate.filter.mode == reference.filter.mode
    assert candidate.filter.recovery_steps == reference.filter.recovery_steps
    assert candidate.filter.uncertified_steps == reference.filter.uncertified_steps
    original_diagnostics = {k: v for k, v in reference.filter.last.items()
                            if not k.startswith("policy_feedback")}
    candidate_diagnostics = {k: v for k, v in candidate.filter.last.items()
                             if not k.startswith("policy_feedback")}
    assert candidate_diagnostics == original_diagnostics
    assert candidate.captured["feedback_calls"] == 1
    assert candidate.filter.last["policy_feedback_evaluated"]
    assert not candidate.filter.last["policy_feedback_preserved"]


def test_feedback_acceptance_uses_the_existing_trigger_boundary(feedback_case):
    case = feedback_case(extra_first=[0.0, np.inf, 0.0, 0.0],
                         extra_clear=[1.0, v2.TRIGGER_MARGIN_M, 1.0, 1.0])
    out, changed = case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY.astype(np.float32))
    assert not changed
    assert case.filter.last["why"] == "policy feedback"
    assert case.filter.last["checked_clearance"] == v2.TRIGGER_MARGIN_M


def test_optional_retention_keeps_only_the_new_checked_feedback_plan(feedback_case, monkeypatch):
    monkeypatch.setattr(v4, "RETAIN_NOMINAL_PLAN", True)
    case = feedback_case()
    case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(case.filter.plan, case.captured["feedback_sequences"][0, 2])
    assert not np.array_equal(case.filter.plan, case.previous)
    assert case.filter.last["nominal_plan_retained"]
