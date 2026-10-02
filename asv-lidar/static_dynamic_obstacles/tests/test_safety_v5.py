"""V5 selection and bank-order regressions using synthetic predictions only."""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_feedback
import safety_recovery
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v5 as v5


@pytest.fixture
def filter_factory(monkeypatch):
    for name in ("COUNTERSTEER_RECOVERY", "RETAIN_NOMINAL_PLAN", "DUAL_BRAKE_PREDICTION",
                 "FREE_SPACE_MEMORY", "MODEL_EGO_OBSERVER"):
        monkeypatch.setattr(v4, name, False)
    for name in ("ONE_DECISION_COMMIT", "SOFT_RECOVERY", "FEEDBACK_BACKUPS",
                 "FEEDBACK_ONLY_INFEASIBLE", "SIDESLIP_RESCUE", "PREFER_CERTIFIED_POLICY"):
        monkeypatch.setattr(v5, name, False)
    monkeypatch.setattr(v5, "POLICY_FEEDBACK_PRESERVATION", False)
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, sequences: sequences)

    def make(cls=v5.SafetyFilterV5, traffic=False, stored=False):
        filter_ = cls()
        snap = cc.Snapshot(5.0, 5.0, 0.0, 1.0, 0.0, 0.0,
                           np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([5.0, 5.0]),
                           0.0, 0.0, 20.0, np.empty((0, 2)))
        if traffic:
            snap.tracks = [SimpleNamespace(position=np.array([5.0, 7.0]))]
        monkeypatch.setattr(filter_.perception, "snapshot", lambda env: snap)
        monkeypatch.setattr(filter_, "_threat_in_reach", lambda snap: True)
        monkeypatch.setattr(filter_, "_rejoin", lambda snap: np.zeros(2))
        monkeypatch.setattr(filter_.actuators, "issue", lambda env, rudder: None)
        if stored:
            filter_.plan = v3.plan_for((0.75, 0.1), (-1.0, 0.0))
        return filter_, SimpleNamespace(_v2_brake=False)

    return make


def _hard_metrics(sequences, mode, continuation=False, traffic=False):
    nrec = sum(traffic or throttle is not v2.BRAKE for _, throttle in v2.RECOVERY)
    first, clear = np.zeros(len(sequences)), np.full(len(sequences), -1.0)
    if mode == "nominal":
        first[1], clear[1] = np.inf, 0.8
    elif mode == "turn":
        first[3 * nrec + 1], clear[3 * nrec + 1] = np.inf, 0.8
    elif mode == "continue":
        assert continuation
        first[-1], clear[-1] = np.inf, 0.8
    elif mode == "brake":
        row = 2 + len(v2.RUDDERS) * len(v2.THROTTLES)
        first[row * nrec], clear[row * nrec] = np.inf, 0.8
    elif mode == "hold back":
        first[3 * nrec] = 2.0 + v2.HOLD_BACK_GAIN_S
    return first, clear


@pytest.mark.parametrize("mode", ["nominal", "turn", "continue", "brake",
                                  "no escape", "hold back", "last certificate"])
def test_all_v5_methods_disabled_preserve_v4_selection(filter_factory, monkeypatch, mode):
    stored, traffic = mode in ("continue", "last certificate"), mode == "brake"
    reference, reference_env = filter_factory(v4.SafetyFilterV4, traffic, stored)
    filter_, env = filter_factory(traffic=traffic, stored=stored)

    def evaluate(snap, sequences):
        return _hard_metrics(sequences, mode, stored, traffic)

    monkeypatch.setattr(reference, "_evaluate", evaluate)
    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    action = np.array([0.2, 0.1])
    out, changed = filter_.filter(env, action)
    expected, expected_changed = reference.filter(reference_env, action)
    np.testing.assert_array_equal(out, expected)
    assert changed == expected_changed
    assert env._v2_brake == reference_env._v2_brake
    assert filter_.mode == reference.mode
    assert filter_.last["why"] == reference.last["why"] == mode
    assert filter_.uncertified_steps == reference.uncertified_steps
    if reference.plan is None:
        assert filter_.plan is None
    else:
        np.testing.assert_array_equal(filter_.plan, reference.plan)


@pytest.mark.parametrize("countersteer", [False, True])
def test_one_decision_commit_holds_only_the_issued_control_interval(monkeypatch, countersteer):
    monkeypatch.setattr(v5, "ONE_DECISION_COMMIT", True)
    candidate, recovery = (0.4, 0.2), (-1.0, 0.0)
    plan = v5.plan_for(candidate, recovery, countersteer)
    turn_count = int(round(v2.RECOVERY_TURN_S / cfg.UPDATE_RATE))
    np.testing.assert_array_equal(plan[0], candidate)
    np.testing.assert_array_equal(plan[1:1 + turn_count], np.tile(recovery, (turn_count, 1)))
    next_command = (1.0, 0.0) if countersteer else (0.0, 0.0)
    np.testing.assert_array_equal(plan[1 + turn_count], next_command)
    assert len(plan) == v3._decisions()[0]
    monkeypatch.setattr(v5, "ONE_DECISION_COMMIT", False)
    np.testing.assert_array_equal(v5.plan_for(candidate, recovery, countersteer),
                                  v4.plan_for(candidate, recovery, countersteer))


@pytest.mark.parametrize("continuation_wins", [False, True])
def test_soft_recovery_ranks_complete_plans_without_old_plan_privilege(
        filter_factory, monkeypatch, continuation_wins):
    filter_, env = filter_factory(stored=True)
    monkeypatch.setattr(v5, "SOFT_RECOVERY", True)
    previous = filter_.plan.copy()
    filter_.uncertified_steps = 1
    captured = {}
    monkeypatch.setattr(filter_, "_evaluate", lambda snap, seq: _hard_metrics(seq, "no escape"))

    def soft(snap, sequences):
        np.testing.assert_array_equal(sequences[-1], v3.continuation(previous))
        captured["sequences"] = sequences.copy()
        cost = np.full(len(sequences), 10.0)
        cost[10], cost[-1] = 0.1, 0.05 if continuation_wins else 0.2
        # Better clearance on another recovery must not be borrowed by index 10.
        clear = np.full(len(sequences), -0.1)
        clear[10], clear[-1] = -3.25, -2.5
        return safety_recovery.RecoveryMetrics(np.zeros(len(sequences)), clear,
                                              cost, cost / 10.0)

    monkeypatch.setattr(safety_recovery, "evaluate", soft)
    out, _ = filter_.filter(env, np.array([0.2, 0.1]))
    selected = len(captured["sequences"]) - 1 if continuation_wins else 10
    np.testing.assert_array_equal(out, captured["sequences"][selected, 0].astype(np.float32))
    assert filter_.last["soft_sequence_index"] == selected
    assert filter_.last["checked_clearance"] == (-2.5 if continuation_wins else -3.25)
    assert filter_.last["recovery_template"] == ("soft continuation" if continuation_wins else "soft candidate")
    assert filter_.last["why"] == "soft recovery"
    assert filter_.plan is None
    assert filter_.last["plan_steps"] == 0
    assert filter_.uncertified_steps == 0


@pytest.mark.parametrize("mode", ["nominal", "turn", "continue", "brake"])
def test_soft_recovery_does_not_change_hard_feasible_selection(filter_factory, monkeypatch, mode):
    stored, traffic = mode == "continue", mode == "brake"
    reference, reference_env = filter_factory(traffic=traffic, stored=stored)
    filter_, env = filter_factory(traffic=traffic, stored=stored)

    def evaluate(snap, sequences):
        return _hard_metrics(sequences, mode, stored, traffic)

    monkeypatch.setattr(reference, "_evaluate", evaluate)
    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    action = np.array([0.2, 0.1])
    expected, expected_changed = reference.filter(reference_env, action)
    monkeypatch.setattr(v5, "SOFT_RECOVERY", True)

    def forbidden(*args):
        pytest.fail("Soft scoring must not run when a hard-safe option exists")

    monkeypatch.setattr(safety_recovery, "evaluate", forbidden)
    out, changed = filter_.filter(env, action)
    np.testing.assert_array_equal(out, expected)
    assert changed == expected_changed
    assert filter_.last["why"] == mode
    if reference.plan is None:
        assert filter_.plan is None
    else:
        np.testing.assert_array_equal(filter_.plan, reference.plan)


def test_soft_braking_is_an_intervention_even_when_transport_action_matches(filter_factory, monkeypatch):
    filter_, env = filter_factory(traffic=True)
    monkeypatch.setattr(v5, "SOFT_RECOVERY", True)
    monkeypatch.setattr(filter_, "_evaluate", lambda snap, seq: _hard_metrics(seq, "no escape", traffic=True))

    def soft(snap, sequences):
        cost = np.full(len(sequences), 10.0)
        braking = np.flatnonzero(np.isnan(sequences[:, 0, 1]) & (sequences[:, 0, 0] == -1.0))[0]
        cost[braking] = 0.1
        return safety_recovery.RecoveryMetrics(np.zeros(len(sequences)), -cost, cost, cost)

    monkeypatch.setattr(safety_recovery, "evaluate", soft)
    action = np.array([-1.0, -1.0])
    out, changed = filter_.filter(env, action)
    np.testing.assert_array_equal(out, action)
    assert changed and filter_.last["changed"] and filter_.last["brake"] and env._v2_brake
    assert filter_.plan is None


def _tagged_rollout(tags):
    tags = np.asarray(tags, dtype=float)
    positions = np.stack((tags, tags + 10000.0), axis=-1)[None]
    return cc.Rollout(np.repeat(positions, 2, axis=0), np.tile(tags + 20000.0, (2, 1)),
                      np.tile(tags + 30000.0, (2, 1)), np.array([0.125, 0.25]))


@pytest.mark.parametrize("only_infeasible", [False, True])
@pytest.mark.parametrize("winner", ["original", "feedback", "continuation"])
def test_feedback_merge_keeps_rollouts_sequences_and_continuation_aligned(
        filter_factory, monkeypatch, winner, only_infeasible):
    filter_, env = filter_factory(stored=True)
    monkeypatch.setattr(v5, "FEEDBACK_BACKUPS", True)
    monkeypatch.setattr(v5, "FEEDBACK_ONLY_INFEASIBLE", only_infeasible)
    monkeypatch.setattr(v5, "SOFT_RECOVERY", True)
    previous = filter_.plan.copy()
    captured = {}

    def rollout(snap, act, sequences):
        captured["old_sequences"] = sequences.copy()
        return _tagged_rollout(np.arange(len(sequences)))

    def feedback(snap, act, candidates, commit_s):
        count, decisions = len(candidates), v3._decisions()[0]
        seq = np.repeat(candidates[:, None, None, :], 2, axis=1)
        seq = np.repeat(seq, decisions, axis=2)
        seq[:, :, 1:, 0] = np.arange(count * 2).reshape(count, 2, 1) / 100.0
        captured["feedback_sequences"] = seq.copy()
        return _tagged_rollout(1000.0 + np.arange(count * 2)), seq, ("test_a", "test_b")

    monkeypatch.setattr(v3, "rollout_seq", rollout)
    monkeypatch.setattr(safety_feedback, "feedback_bank", feedback)
    monkeypatch.setattr(filter_, "_evaluate", lambda snap, ro: (np.zeros(ro.positions.shape[1]),
                                                               np.full(ro.positions.shape[1], -1.0)))

    def soft(snap, ro):
        old, extra = captured["old_sequences"], captured["feedback_sequences"]
        count, nrec = len(extra), (len(old) - 1) // len(extra)
        expected_sequences = np.concatenate((old[:-1].reshape(count, nrec, *old.shape[1:]), extra), axis=1)
        expected_sequences = np.concatenate((expected_sequences.reshape(-1, *old.shape[1:]), old[-1:]), axis=0)
        tags = np.concatenate((np.arange(len(old) - 1).reshape(count, nrec),
                               (1000.0 + np.arange(count * 2)).reshape(count, 2)), axis=1).ravel()
        tags = np.concatenate((tags, [len(old) - 1]))
        np.testing.assert_array_equal(ro.positions[:, :, 0], np.tile(tags, (2, 1)))
        np.testing.assert_array_equal(ro.positions[:, :, 1], np.tile(tags + 10000.0, (2, 1)))
        np.testing.assert_array_equal(ro.headings, np.tile(tags + 20000.0, (2, 1)))
        np.testing.assert_array_equal(ro.speeds, np.tile(tags + 30000.0, (2, 1)))
        selected = {"original": nrec + 2, "feedback": 2 * (nrec + 2) + nrec,
                    "continuation": len(tags) - 1}[winner]
        captured["selected"], captured["expected"] = selected, expected_sequences[selected]
        cost = np.ones(len(tags))
        cost[selected] = 0.1
        return safety_recovery.RecoveryMetrics(np.zeros(len(tags)), -tags, cost, cost)

    monkeypatch.setattr(safety_recovery, "evaluate", soft)
    out, _ = filter_.filter(env, np.array([0.2, 0.1]))
    np.testing.assert_array_equal(out, captured["expected"][0].astype(np.float32))
    assert filter_.last["soft_sequence_index"] == captured["selected"]
    assert filter_.last["feedback_backups_evaluated"]
    assert filter_.plan is None
    np.testing.assert_array_equal(captured["old_sequences"][-1], v3.continuation(previous))


def test_v5_rejects_unsupported_dual_response_ablation(monkeypatch):
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", True)
    with pytest.raises(ValueError, match="single-response predictor"):
        v5.SafetyFilterV5()


@pytest.mark.parametrize("only_infeasible", [False, True])
def test_hard_brake_reports_the_selected_feedback_template(filter_factory, monkeypatch, only_infeasible):
    import safety_course_backup
    filter_, env = filter_factory(traffic=True)
    monkeypatch.setattr(v5, "FEEDBACK_BACKUPS", True)
    monkeypatch.setattr(v5, "FEEDBACK_ONLY_INFEASIBLE", only_infeasible)
    monkeypatch.setattr(v5, "SIDESLIP_RESCUE", True)
    monkeypatch.setattr(safety_course_backup, "feedback_bank",
                        lambda *args: pytest.fail("An existing safe feedback backup must suppress rescue"))
    brake_row = 2 + len(v2.RUDDERS) * len(v2.THROTTLES)
    captured = {}
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, sequences:
                        _tagged_rollout(np.arange(len(sequences))))

    def feedback(snap, act, candidates, commit_s):
        sequences = np.repeat(candidates[:, None, None, :], 2, axis=1)
        sequences = np.repeat(sequences, v3._decisions()[0], axis=2)
        sequences[:, :, 1:, :] = (0.0, 0.0)
        sequences[brake_row, 1, 1:, 0] = 0.25
        captured["plan"] = sequences[brake_row, 1].copy()
        return _tagged_rollout(1000 + np.arange(len(candidates) * 2)), sequences, ("test_a", "test_b")

    def evaluate(snap, rollout):
        tags = rollout.positions[0, :, 0]
        chosen = tags == 1000 + 2 * brake_row + 1
        return np.where(chosen, np.inf, 0.0), np.where(chosen, 0.8, -1.0)

    monkeypatch.setattr(safety_feedback, "feedback_bank", feedback)
    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    out, changed = filter_.filter(env, np.array([-1.0, -1.0]))
    np.testing.assert_array_equal(out, [-1.0, -1.0])
    np.testing.assert_array_equal(filter_.plan, captured["plan"])
    assert filter_.last["why"] == "brake"
    assert filter_.last["recovery_template"] == "test_b"
    assert changed and env._v2_brake


@pytest.mark.parametrize("initial_mode", ["nominal", "recovery"])
@pytest.mark.parametrize("safe_option", ["nominal", "turn", "continue", "brake"])
def test_infeasible_only_feedback_preserves_original_safe_choices(
        filter_factory, monkeypatch, initial_mode, safe_option):
    import safety_course_backup
    stored, traffic = safe_option == "continue", safe_option == "brake"
    reference, reference_env = filter_factory(traffic=traffic, stored=stored)
    filter_, env = filter_factory(traffic=traffic, stored=stored)
    reference.mode = filter_.mode = initial_mode

    def evaluate(snap, sequences):
        first, clear = _hard_metrics(sequences, safe_option, stored, traffic)
        # A hard-safe policy plan with little room must also suppress the new
        # bank: the gate tests hard feasibility, not trigger/handback margins.
        clear[np.isinf(first)] = 0.5 * v2.TRIGGER_MARGIN_M
        return first, clear

    monkeypatch.setattr(reference, "_evaluate", evaluate)
    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    action = np.array([0.2, 0.1])
    expected, expected_changed = reference.filter(reference_env, action)
    monkeypatch.setattr(v5, "FEEDBACK_BACKUPS", True)
    monkeypatch.setattr(v5, "FEEDBACK_ONLY_INFEASIBLE", True)
    monkeypatch.setattr(v5, "SIDESLIP_RESCUE", True)

    def forbidden(*args):
        pytest.fail("Feedback generation must not replace an original hard-safe option")

    monkeypatch.setattr(safety_feedback, "feedback_bank", forbidden)
    monkeypatch.setattr(safety_course_backup, "feedback_bank", forbidden)
    out, changed = filter_.filter(env, action)
    np.testing.assert_array_equal(out, expected)
    assert changed == expected_changed
    assert env._v2_brake == reference_env._v2_brake
    assert filter_.mode == reference.mode
    assert filter_.last["why"] == reference.last["why"]
    assert filter_.last["feedback_only_infeasible"]
    assert not filter_.last["feedback_backups_evaluated"]
    assert filter_.last["any_safe"]
    if stored:
        assert filter_.last["continuation_ok"]
        assert filter_.last["n_safe_candidates"] == 0
    np.testing.assert_array_equal(filter_.plan, reference.plan)


def test_original_feedback_gate_still_runs_with_a_safe_alternative(filter_factory, monkeypatch):
    filter_, env = filter_factory()
    monkeypatch.setattr(v5, "FEEDBACK_BACKUPS", True)
    nrec = sum(throttle is not v2.BRAKE for _, throttle in v2.RECOVERY)
    chosen = 3 * nrec + 1
    captured = {}

    def rollout(snap, act, sequences):
        captured["original"] = sequences.copy()
        return _tagged_rollout(np.arange(len(sequences)))

    def feedback(snap, act, candidates, commit_s):
        captured["feedback_called"] = True
        sequences = np.repeat(candidates[:, None, None, :], 1, axis=1)
        sequences = np.repeat(sequences, v3._decisions()[0], axis=2)
        return _tagged_rollout(1000 + np.arange(len(candidates))), sequences, ("test",)

    def evaluate(snap, ro):
        safe = ro.positions[0, :, 0] == chosen
        return np.where(safe, np.inf, 0.0), np.where(safe, 0.8, -1.0)

    monkeypatch.setattr(v3, "rollout_seq", rollout)
    monkeypatch.setattr(safety_feedback, "feedback_bank", feedback)
    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    out, _ = filter_.filter(env, np.array([0.2, 0.1]))
    assert captured["feedback_called"]
    assert not filter_.last["feedback_only_infeasible"]
    assert filter_.last["feedback_backups_evaluated"]
    assert filter_.last["why"] == "turn"
    np.testing.assert_array_equal(out, captured["original"][chosen, 0].astype(np.float32))
    np.testing.assert_array_equal(filter_.plan, captured["original"][chosen])


@pytest.mark.parametrize("traffic", [False, True])
@pytest.mark.parametrize("legacy_feedback", [False, True])
def test_sideslip_rescue_admits_only_a_hard_safe_complete_plan(
        filter_factory, monkeypatch, traffic, legacy_feedback):
    import safety_course_backup
    filter_, env = filter_factory(traffic=traffic, stored=True)
    monkeypatch.setattr(v5, "SIDESLIP_RESCUE", True)
    monkeypatch.setattr(v5, "FEEDBACK_BACKUPS", legacy_feedback)
    monkeypatch.setattr(v5, "FEEDBACK_ONLY_INFEASIBLE", True)
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, sequences:
                        _tagged_rollout(np.arange(len(sequences))))
    captured = {}
    winner = 2 + len(v2.RUDDERS) * len(v2.THROTTLES) if traffic else 3

    def rescue(snap, act, candidates, commit_s):
        sequences = np.repeat(candidates[:, None, None, :], v3._decisions()[0], axis=2)
        sequences[:, :, 1:, :] = (0.37, 0.0)
        captured["plan"] = sequences[winner, 0].copy()
        return _tagged_rollout(10000 + np.arange(len(candidates))), sequences, ("edge_parallel_sideslip",)

    def old_feedback(snap, act, candidates, commit_s):
        sequences = np.repeat(candidates[:, None, None, :], 2, axis=1)
        sequences = np.repeat(sequences, v3._decisions()[0], axis=2)
        return _tagged_rollout(2000 + np.arange(len(candidates) * 2)), sequences, ("old_a", "old_b")

    def evaluate(snap, rollout):
        safe = rollout.positions[0, :, 0] == 10000 + winner
        return np.where(safe, np.inf, 0.0), np.where(safe, 0.8, -1.0)

    monkeypatch.setattr(safety_course_backup, "feedback_bank", rescue)
    monkeypatch.setattr(safety_feedback, "feedback_bank", old_feedback)
    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    out, changed = filter_.filter(env, np.array([0.2, 0.1]))
    expected = captured["plan"][0].copy()
    if traffic:
        expected[1] = -1.0
    np.testing.assert_array_equal(out, expected.astype(np.float32))
    np.testing.assert_array_equal(filter_.plan, captured["plan"])
    assert changed
    assert filter_.last["sideslip_rescue_evaluated"]
    assert filter_.last["sideslip_rescue_admitted"]
    assert filter_.last["recovery_template"] == "edge_parallel_sideslip"
    assert filter_.last["why"] == ("brake" if traffic else "turn")


@pytest.mark.parametrize("mode", ["no escape", "hold back", "last certificate"])
def test_rejected_sideslip_bank_cannot_change_existing_fallback(filter_factory, monkeypatch, mode):
    import safety_course_backup
    stored = mode == "last certificate"
    reference, reference_env = filter_factory(stored=stored)
    filter_, env = filter_factory(stored=stored)
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, sequences:
                        _tagged_rollout(np.arange(len(sequences))))

    def rescue(snap, act, candidates, commit_s):
        sequences = np.repeat(candidates[:, None, None, :], v3._decisions()[0], axis=2)
        return _tagged_rollout(10000 + np.arange(len(candidates))), sequences, ("edge_parallel_sideslip",)

    def evaluate(snap, rollout):
        tags = rollout.positions[0, :, 0]
        if tags[0] >= 10000:
            # Tempting but unsafe: including this bank would alter fallback
            # rankings even though no complete hard-safe plan was found.
            first = np.ones(len(tags))
            first[4] = 100.0
            return first, np.full(len(tags), -0.01)
        return _hard_metrics(np.empty((len(tags), 1, 2)), mode, stored)

    monkeypatch.setattr(safety_course_backup, "feedback_bank", rescue)
    monkeypatch.setattr(reference, "_evaluate", evaluate)
    monkeypatch.setattr(filter_, "_evaluate", evaluate)
    action = np.array([0.2, 0.1])
    expected, expected_changed = reference.filter(reference_env, action)
    monkeypatch.setattr(v5, "SIDESLIP_RESCUE", True)
    out, changed = filter_.filter(env, action)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(filter_.plan, reference.plan)
    assert changed == expected_changed
    assert filter_.last["why"] == reference.last["why"] == mode
    assert filter_.last["sideslip_rescue_evaluated"]
    assert not filter_.last["sideslip_rescue_admitted"]
