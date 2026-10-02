"""Safety v4 unit regressions; no policies or scenario episodes are run."""
from types import SimpleNamespace

import numpy as np
import pytest

import safety_v3 as v3
import safety_v4 as v4
from classical import common as cc


def test_countersteer_preserves_commit_and_reverses_recovery_turn():
    candidate, recovery = (0.5, -1.0), (1.0, 0.0)
    original = v3.plan_for(candidate, recovery)
    plan = v4.plan_for(candidate, recovery, countersteer=True)
    _, commit, turn_end = v3._decisions()
    counter_end = turn_end + turn_end - commit
    np.testing.assert_array_equal(plan[:turn_end], original[:turn_end])
    np.testing.assert_array_equal(plan[turn_end:counter_end],
                                  np.tile((-1.0, 0.0), (turn_end - commit, 1)))
    np.testing.assert_array_equal(plan[counter_end:], original[counter_end:])


def test_sequence_bank_retains_all_v3_templates_and_excludes_brake_countersteer(monkeypatch):
    recoveries = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (-1.0, np.nan)]
    candidates = np.array([(0.1, 0.2), (-0.7, -0.3)])
    monkeypatch.setattr(v4, "COUNTERSTEER_RECOVERY", True)
    templates = v4.recovery_templates(recoveries)
    assert len(templates) == 6
    bank = v4.sequence_bank(candidates, templates)
    for ci, candidate in enumerate(candidates):
        for ri, recovery in enumerate(recoveries):
            np.testing.assert_array_equal(bank[ci * len(templates) + ri],
                                          v3.plan_for(candidate, recovery))
    assert all(not np.isnan(rec[1]) for rec, counter in templates if counter)
    monkeypatch.setattr(v4, "COUNTERSTEER_RECOVERY", False)
    assert len(v4.recovery_templates(recoveries)) == len(recoveries)


@pytest.fixture
def mocked_filter(monkeypatch):
    # Exercise each mechanism explicitly, independent of the selected dev preset.
    monkeypatch.setattr(v4, "COUNTERSTEER_RECOVERY", True)
    monkeypatch.setattr(v4, "RETAIN_NOMINAL_PLAN", True)
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", False)
    monkeypatch.setattr(v4, "FREE_SPACE_MEMORY", False)
    monkeypatch.setattr(v4, "MODEL_EGO_OBSERVER", False)
    filter_ = v4.SafetyFilterV4()
    snap = cc.Snapshot(5.0, 5.0, 0.0, 1.0, 0.0, 0.0,
                       np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([5.0, 5.0]),
                       0.0, 0.0, 20.0, np.empty((0, 2)))
    monkeypatch.setattr(filter_.perception, "snapshot", lambda env: snap)
    monkeypatch.setattr(filter_, "_threat_in_reach", lambda snap: True)
    monkeypatch.setattr(filter_, "_rejoin", lambda snap: np.zeros(2))
    monkeypatch.setattr(filter_.actuators, "issue", lambda env, rudder: None)
    # Pass the sequence bank directly to the mocked clearance evaluator.
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, sequences: sequences)
    return filter_, SimpleNamespace(_v2_brake=False)


@pytest.mark.parametrize("initial_mode,reason", [("nominal", "nominal"), ("recovery", "handback")])
def test_checked_policy_plan_survives_pass_and_advances_one_step(mocked_filter, monkeypatch,
                                                               initial_mode, reason):
    filter_, env = mocked_filter
    monkeypatch.setattr(v4, "RETAIN_NOMINAL_PLAN", True)
    filter_.mode = initial_mode
    filter_.recovery_steps = v3.MIN_RECOVERY_STEPS
    chosen_plan = None

    def all_safe(snap, sequences):
        nonlocal chosen_plan
        margins = np.ones(len(sequences))
        margins[1] = 2.0  # Choose the second policy recovery, not the first.
        chosen_plan = sequences[1].copy()
        return np.full(len(sequences), np.inf), margins

    monkeypatch.setattr(filter_, "_evaluate", all_safe)
    action = np.array([0.4, 0.2])
    out, changed = filter_.filter(env, action)
    assert not changed
    np.testing.assert_allclose(out, action)
    np.testing.assert_array_equal(filter_.plan, chosen_plan)
    assert filter_.last["why"] == reason
    assert filter_.last["nominal_plan_retained"]
    original_plan = chosen_plan.copy()

    def only_continuation_safe(snap, sequences):
        np.testing.assert_array_equal(sequences[-1], v3.continuation(original_plan))
        first = np.zeros(len(sequences))
        first[-1] = np.inf
        clear = np.full(len(sequences), -1.0)
        clear[-1] = 0.4
        return first, clear

    monkeypatch.setattr(filter_, "_evaluate", only_continuation_safe)
    out, _ = filter_.filter(env, action)
    np.testing.assert_array_equal(out, original_plan[1].astype(np.float32))
    np.testing.assert_array_equal(filter_.plan, original_plan[1:])
    assert filter_.last["why"] == "continue"
    assert filter_.last["checked_clearance"] == 0.4


def test_nominal_plan_retention_can_be_disabled(mocked_filter, monkeypatch):
    filter_, env = mocked_filter
    monkeypatch.setattr(v4, "RETAIN_NOMINAL_PLAN", False)
    monkeypatch.setattr(filter_, "_evaluate", lambda snap, seq: (np.full(len(seq), np.inf),
                                                               np.ones(len(seq))))
    filter_.filter(env, np.zeros(2))
    assert filter_.plan is None
    assert not filter_.last["nominal_plan_retained"]


def test_selected_countersteer_backup_is_the_plan_retained(mocked_filter, monkeypatch):
    filter_, env = mocked_filter
    monkeypatch.setattr(v4, "COUNTERSTEER_RECOVERY", True)
    monkeypatch.setattr(v4, "RETAIN_NOMINAL_PLAN", True)

    def only_countersteer_safe(snap, sequences):
        first = np.zeros(len(sequences))
        first[3] = np.inf  # First appended recovery for the policy candidate.
        clear = np.full(len(sequences), -1.0)
        clear[3] = 0.5
        return first, clear

    monkeypatch.setattr(filter_, "_evaluate", only_countersteer_safe)
    action = np.array([0.2, 0.1])
    filter_.filter(env, action)
    np.testing.assert_array_equal(filter_.plan, v4.plan_for(action, (-1.0, 0.0), True))
    assert filter_.last["recovery_template"] == "countersteer"
    assert filter_.last["checked_clearance"] == 0.5


@pytest.mark.parametrize("mode", ["policy_safe", "override", "none_safe"])
def test_switches_off_match_v3_selection(mocked_filter, monkeypatch, mode):
    filter_, env = mocked_filter
    monkeypatch.setattr(v4, "COUNTERSTEER_RECOVERY", False)
    monkeypatch.setattr(v4, "RETAIN_NOMINAL_PLAN", False)
    reference = v3.SafetyFilterV3()
    reference.perception = filter_.perception
    reference.actuators = filter_.actuators
    reference._threat_in_reach = filter_._threat_in_reach
    reference._rejoin = filter_._rejoin

    def evaluate(snap, sequences):
        first = np.zeros(len(sequences))
        clear = np.full(len(sequences), -1.0)
        if mode == "policy_safe":
            first[1], clear[1] = np.inf, 0.8
        elif mode == "override":
            first[10], clear[10] = np.inf, 0.8
        return first, clear

    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    monkeypatch.setattr(reference, "_evaluate", evaluate)
    action = np.array([0.2, 0.1])
    output, changed = filter_.filter(env, action)
    expected, expected_changed = reference.filter(env, action)
    np.testing.assert_array_equal(output, expected)
    assert changed == expected_changed
    assert filter_.last["why"] == reference.last["why"]
    assert filter_.mode == reference.mode
    if reference.plan is None:
        assert filter_.plan is None
    else:
        np.testing.assert_array_equal(filter_.plan, reference.plan)


def test_failed_continuation_keeps_v3_unchecked_step_limit(mocked_filter, monkeypatch):
    filter_, env = mocked_filter
    filter_.plan = v3.plan_for((0.5, 0.0), (1.0, 0.0))
    monkeypatch.setattr(filter_, "_evaluate", lambda snap, seq: (np.zeros(len(seq)),
                                                               np.full(len(seq), -1.0)))
    for count in range(1, v3.LAST_CERT_MAX_STEPS + 1):
        filter_.filter(env, np.zeros(2))
        assert filter_.last["why"] == "last certificate"
        assert filter_.uncertified_steps == count
    filter_.filter(env, np.zeros(2))
    assert filter_.last["why"] == "no escape"
    assert filter_.plan is None


def test_expired_one_action_plan_is_not_reused(mocked_filter, monkeypatch):
    filter_, env = mocked_filter
    filter_.plan = np.array([[1.0, 0.0]])
    monkeypatch.setattr(filter_, "_evaluate", lambda snap, seq: (np.zeros(len(seq)),
                                                               np.full(len(seq), -1.0)))
    filter_.filter(env, np.zeros(2))
    assert filter_.last["why"] == "no escape"
    assert not filter_.last["continuation_checked"]
    assert filter_.plan is None


def test_braking_counts_as_intervention_when_transport_action_matches_policy(mocked_filter, monkeypatch):
    filter_, env = mocked_filter
    snap = filter_.perception.snapshot(env)
    snap.tracks = [SimpleNamespace(position=np.array([5., 7.]))]
    monkeypatch.setattr(v4, "COUNTERSTEER_RECOVERY", False)
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", False)

    def only_brake_safe(snap, sequences):
        first = np.zeros(len(sequences))
        clear = np.full(len(sequences), -1.)
        # 1 policy + 15 grid + 1 rejoin, each with 6 recovery templates.
        first[17 * 6], clear[17 * 6] = np.inf, .5
        return first, clear

    monkeypatch.setattr(filter_, "_evaluate", only_brake_safe)
    action = np.array([-1., -1.])
    out, changed = filter_.filter(env, action)
    np.testing.assert_array_equal(out, action)
    assert env._v2_brake and changed and filter_.last["changed"]
