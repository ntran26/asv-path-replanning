"""Policy-priority selection tests; synthetic predictions, no episodes.

The first decision reproduces controls and clearance from the development
CRS-CV-04 step-13 trace. Future trajectories are synthetic feasibility fixtures.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v5 as v5


POLICY = np.array([-0.7543492317199707, -0.9987676739692688])
CONTINUATION = np.array([-0.3285370469093323, 0.7555042505264282 / 6.0 - 1.0])


@pytest.fixture
def priority_case(monkeypatch):
    for name in ("COUNTERSTEER_RECOVERY", "RETAIN_NOMINAL_PLAN", "DUAL_BRAKE_PREDICTION",
                 "FREE_SPACE_MEMORY", "MODEL_EGO_OBSERVER"):
        monkeypatch.setattr(v4, name, False)
    for name in ("ONE_DECISION_COMMIT", "SOFT_RECOVERY", "FEEDBACK_BACKUPS",
                 "FEEDBACK_ONLY_INFEASIBLE", "SIDESLIP_RESCUE"):
        monkeypatch.setattr(v5, name, False)
    monkeypatch.setattr(v5, "POLICY_FEEDBACK_PRESERVATION", False)
    monkeypatch.setattr(v5, "PREFER_CERTIFIED_POLICY", True)
    monkeypatch.setattr(cfg, "UPDATE_RATE", 0.5)
    monkeypatch.setattr(v2, "TRIGGER_MARGIN_M", 0.15)
    monkeypatch.setattr(v2, "ROOM_SLACK_M", 0.20)
    monkeypatch.setattr(v2, "W_THROTTLE", 2.0)
    monkeypatch.setattr(v3, "W_CONTINUE", 0.25)
    monkeypatch.setattr(v3, "RECOVERY_REFERENCE", "policy")
    monkeypatch.setattr(v3, "HANDBACK_SPEED", 0.30)
    monkeypatch.setattr(v3, "HANDBACK_MARGIN_M", 0.30)
    monkeypatch.setattr(v3, "MIN_RECOVERY_STEPS", 4)
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, sequences: sequences)

    def make(*, policy_safe=True, policy_margin=0.85, best_margin=0.86,
             continuation=CONTINUATION, continuation_safe=True,
             continuation_margin=0.86, speed=0.1251368159638691,
             mode="recovery"):
        filter_ = v5.SafetyFilterV5()
        snap = cc.Snapshot(
            x=5.0, y=5.0, heading=0.0, u=speed, v=0.0, r=0.0,
            tangent=np.array([0.0, 1.0]), right=np.array([1.0, 0.0]),
            centre=np.array([5.0, 5.0]), base_heading=0.0,
            lateral=0.0, remaining=20.0, points=np.empty((0, 2)),
            tracks=[SimpleNamespace(position=np.array([5.0, 7.0]))])
        monkeypatch.setattr(filter_.perception, "snapshot", lambda env: snap)
        monkeypatch.setattr(filter_, "_threat_in_reach", lambda snap: True)
        monkeypatch.setattr(filter_, "_rejoin", lambda snap: np.asarray(continuation).copy())
        issued = []
        monkeypatch.setattr(filter_.actuators, "issue",
                            lambda env, rudder: issued.append(rudder))
        previous = np.tile(np.asarray(continuation), (v3._decisions()[0], 1))
        filter_.plan = previous.copy()
        filter_.mode = mode
        filter_.recovery_steps = 7
        filter_.uncertified_steps = 2
        captured = {}
        nrec = len(v2.RECOVERY)  # The nearby tracked vessel enables brake templates.
        policy_recovery = 1

        def evaluate(snapshot, sequences):
            first = np.zeros(len(sequences))
            clear = np.full(len(sequences), -1.0)
            # A deliberately different future plan must replace the old plan
            # when the first policy command is selected.
            captured["policy_plan"] = sequences[policy_recovery].copy()
            if policy_safe:
                first[policy_recovery] = np.inf
                clear[policy_recovery] = policy_margin
            # Another safe grid candidate reproduces the trace's .86m best
            # margin and sets the same clearance floor as the existing rule.
            first[nrec], clear[nrec] = np.inf, best_margin
            if continuation_safe:
                first[-1], clear[-1] = np.inf, continuation_margin
            return first, clear

        monkeypatch.setattr(filter_, "_evaluate", evaluate)
        return SimpleNamespace(filter=filter_, env=SimpleNamespace(_v2_brake=True),
                               previous=previous, captured=captured, issued=issued)

    return make


@pytest.mark.parametrize("enabled", [False, True])
def test_recorded_feasible_policy_beats_continuation_bonus_only_when_enabled(
        priority_case, monkeypatch, enabled):
    monkeypatch.setattr(v5, "PREFER_CERTIFIED_POLICY", enabled)
    case = priority_case()
    distance = (CONTINUATION[0] - POLICY[0]) ** 2 + 2 * (CONTINUATION[1] - POLICY[1]) ** 2
    assert distance == pytest.approx(0.2124087396499057)
    assert distance - v3.W_CONTINUE < 0.0

    out, changed = case.filter.filter(case.env, POLICY)

    expected = POLICY if enabled else CONTINUATION
    np.testing.assert_array_equal(out, expected.astype(np.float32))
    np.testing.assert_array_equal(case.filter.plan, case.captured["policy_plan"]
                                  if enabled else case.previous[1:])
    assert case.filter.mode == "recovery"
    assert case.filter.recovery_steps == 8
    assert case.filter.uncertified_steps == 0
    assert changed is (not enabled)
    assert case.filter.last["changed"] is (not enabled)
    assert not case.env._v2_brake
    assert not case.filter.last["brake"]
    assert case.issued == [float(out[0])]
    assert case.filter.last["why"] == ("verified policy" if enabled else "continue")
    assert case.filter.last["verified_policy_priority_enabled"] is enabled
    assert case.filter.last["verified_policy_preserved"] is enabled
    assert case.filter.last["continuation_override_prevented"] is enabled
    assert case.filter.last["checked_clearance"] == (0.85 if enabled else 0.86)
    assert case.filter.last["plan_steps"] == len(case.filter.plan)


@pytest.mark.parametrize("policy_safe,margin", [(False, 0.85), (True, 0.14)])
def test_ineligible_policy_keeps_checked_continuation(priority_case, policy_safe, margin):
    case = priority_case(policy_safe=policy_safe, policy_margin=margin)
    out, changed = case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, CONTINUATION.astype(np.float32))
    np.testing.assert_array_equal(case.filter.plan, case.previous[1:])
    assert changed
    assert case.filter.last["why"] == "continue"
    assert case.filter.last["policy_safe"] is policy_safe
    assert not case.filter.last["verified_policy_preserved"]
    assert not case.filter.last["continuation_override_prevented"]
    assert case.filter.recovery_steps == 8
    assert case.filter.uncertified_steps == 0


def test_priority_does_not_change_rejoin_reference(priority_case, monkeypatch):
    monkeypatch.setattr(v3, "RECOVERY_REFERENCE", "rejoin")
    case = priority_case()
    out, changed = case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, CONTINUATION.astype(np.float32))
    np.testing.assert_array_equal(case.filter.plan, case.previous[1:])
    assert changed
    assert case.filter.last["policy_safe"]
    assert case.filter.last["why"] == "continue"
    assert not case.filter.last["verified_policy_preserved"]
    assert not case.filter.last["continuation_override_prevented"]


@pytest.mark.parametrize("continuation,safe", [(POLICY, True),
                                              (np.array([1.0, 1.0]), True),
                                              (CONTINUATION, False)])
def test_prevented_counter_requires_a_different_winning_continuation(
        priority_case, continuation, safe):
    case = priority_case(continuation=continuation, continuation_safe=safe)
    out, changed = case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY.astype(np.float32))
    np.testing.assert_array_equal(case.filter.plan, case.captured["policy_plan"])
    assert not changed
    assert case.filter.last["verified_policy_preserved"]
    assert not case.filter.last["continuation_override_prevented"]
    assert case.filter.last["why"] == "verified policy"


@pytest.mark.parametrize("mode,speed,why", [("nominal", 0.125, "nominal"),
                                            ("recovery", 0.5, "handback")])
def test_existing_nominal_and_handback_release_remain_unchanged(
        priority_case, mode, speed, why):
    case = priority_case(mode=mode, speed=speed)
    out, changed = case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY.astype(np.float32))
    assert not changed
    assert case.filter.mode == "nominal"
    assert case.filter.plan is None
    assert case.filter.recovery_steps == case.filter.uncertified_steps == 0
    assert case.filter.last["why"] == why
    assert not case.filter.last["verified_policy_preserved"]
    assert not case.filter.last["continuation_override_prevented"]


def test_priority_uses_existing_pool_floor_without_adding_a_trigger_threshold(priority_case):
    case = priority_case(policy_margin=0.05, best_margin=0.10, continuation_margin=0.10)
    out, changed = case.filter.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY.astype(np.float32))
    assert not changed
    assert case.filter.last["verified_policy_preserved"]
    assert case.filter.last["continuation_override_prevented"]
    assert case.filter.last["checked_clearance"] == 0.05
    assert case.filter.mode == "recovery"
