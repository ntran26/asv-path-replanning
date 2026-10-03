"""V10 paired-plan rules with synthetic predictions only; no environments."""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v8 as v8
import safety_v10 as v10


POLICY = np.array([.3, -.2], dtype=np.float32)
OVERRIDE = np.array([-.8, -1.], dtype=np.float32)


class Commands:
    def __init__(self):
        self.values = [.1]

    def issue(self, env, rudder):
        self.values.append(float(rudder))


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", False)
    f = v10.SafetyFilterV10()
    f.actuators = Commands()
    f.observer = None
    env = SimpleNamespace(command_rate_limit=False)
    snap = SimpleNamespace(tracks=[])
    plan = v3.plan_for([-.8, np.nan], [1., 0.])[:-1]
    calls = []

    def parent(self, env, action):
        self._observer_snapshot = snap
        self.mode, self.plan = "recovery", plan.copy()
        self.recovery_steps, self.uncertified_steps = 3, 2
        self.last = {"why": "last certificate", "changed": True, "brake": True,
                     "policy_margin": -.9, "checked_clearance": -.5}
        self.actuators.issue(env, float(OVERRIDE[0]))
        env._v2_brake = True
        return OVERRIDE.copy(), True

    monkeypatch.setattr(v8.SafetyFilterV8, "_filter", parent)

    def rollout(snapshot, actuators, sequences):
        assert snapshot is snap
        assert actuators.values == [.1]
        calls.append(sequences.copy())
        return sequences

    monkeypatch.setattr(f, "_rollout", rollout)
    monkeypatch.setattr(f, "_evaluate", lambda s, seq: (
        np.array([6.25, 4.75]), np.array([-.121877918, -.308146151])))
    return SimpleNamespace(f=f, env=env, plan=plan, snap=snap, calls=calls)


def predictions(case, monkeypatch, first, clear):
    monkeypatch.setattr(case.f, "_evaluate", lambda s, seq: (
        np.asarray(first, dtype=float), np.asarray(clear, dtype=float)))


def assert_parent_retained(case, result):
    out, changed = result
    np.testing.assert_array_equal(out, OVERRIDE)
    assert changed and case.env._v2_brake
    assert case.f.mode == "recovery"
    assert case.f.recovery_steps == 3 and case.f.uncertified_steps == 2
    assert case.f.last["why"] == "last certificate"
    assert case.f.last["checked_clearance"] == -.5
    assert not case.f.last["v10_policy_preserved"]
    assert case.f.actuators.values == [.1, float(OVERRIDE[0])]
    np.testing.assert_array_equal(case.f.plan, case.plan)


@pytest.mark.parametrize("first,clear", [
    # Saved first-divergence metrics, without any outcome-based runtime rule.
    ([6.25, 4.75], [-.121877918, -.308146151]),  # HO-FIX-05.
    ([3.125, 1.75], [-.097589506, -.462215522]),  # CH-CR-CV-073.
    ([4.875, 5.], [-.656338648, -.627214633]),  # Relative risk alone is insufficient.
    ([2., 1.75], [-.670574977, -.915540039]),
    ([np.nan, np.inf], [.1, np.nan]),
    ([np.inf, np.inf], [-.01, -.02]),
])
def test_both_failed_or_unknown_keeps_parent_fallback(case, monkeypatch, first, clear):
    predictions(case, monkeypatch, first, clear)
    assert_parent_retained(case, case.f.filter(case.env, POLICY))
    assert case.f.last["v10_pair_checked"]


@pytest.mark.parametrize("proposal_first,proposal_clear", [
    (.5, -.1), (np.inf, -.01), (np.nan, .2), (-np.inf, .2), (np.inf, np.nan),
])
def test_only_passing_policy_backup_preserves_sac(case, monkeypatch, proposal_first, proposal_clear):
    predictions(case, monkeypatch, [proposal_first, np.inf], [proposal_clear, .04])
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not case.env._v2_brake
    assert case.f.mode == "nominal" and case.f.recovery_steps == case.f.uncertified_steps == 0
    assert case.f.actuators.values == [.1, float(POLICY[0])]
    assert case.f.last["why"] == "feasible policy backup"
    assert case.f.last["checked_clearance"] == .04 < v2.TRIGGER_MARGIN_M
    assert case.f.last["v10_policy_preserved"]
    np.testing.assert_array_equal(case.f.plan, case.calls[0][1])


@pytest.mark.parametrize("proposal_clear,policy_clear", [(.0148103924, .0401245309), (.1, .1), (0., 0.)])
def test_both_pass_and_policy_margin_is_at_least_as_good(case, monkeypatch, proposal_clear, policy_clear):
    predictions(case, monkeypatch, [np.inf, np.inf], [proposal_clear, policy_clear])
    out, changed = case.f.filter(case.env, POLICY)
    assert not changed and case.f.last["why"] == "policy margin dominates"
    np.testing.assert_array_equal(out, POLICY)
    assert case.f.last["policy_margin"] == -.9  # Ignore unrelated rounded diagnostic.


@pytest.mark.parametrize("policy_first,policy_clear", [(np.inf, .08), (.5, .5), (np.nan, .5), (np.inf, np.nan)])
def test_policy_without_required_evidence_keeps_parent(case, monkeypatch, policy_first, policy_clear):
    predictions(case, monkeypatch, [np.inf, policy_first], [.1, policy_clear])
    assert_parent_retained(case, case.f.filter(case.env, POLICY))


def test_pair_uses_exact_first_commands_identical_padded_tail_and_brake_marker(case):
    saved = case.plan.copy()
    case.f.filter(case.env, POLICY)
    pair = case.calls[0]
    assert pair.shape == (2, int(np.ceil(v2.HORIZON_S / cfg.UPDATE_RATE)), 2)
    np.testing.assert_array_equal(pair[0, 1:], pair[1, 1:])
    np.testing.assert_array_equal(pair[1, 0], POLICY)
    assert pair[0, 0, 0] == OVERRIDE[0] and np.isnan(pair[0, 0, 1])
    np.testing.assert_array_equal(case.plan, saved)
    np.testing.assert_array_equal(pair[:, -1, :], [[0., 0.], [0., 0.]])


@pytest.mark.parametrize("prefer_feasible,prefer_margin,proposal_safe,expected_pass", [
    (False, False, False, False), (False, False, True, False),
    (True, False, False, True), (True, False, True, False),
    (False, True, False, False), (False, True, True, True),
    (True, True, False, True), (True, True, True, True),
])
def test_independent_ablation_switches(case, monkeypatch, prefer_feasible, prefer_margin,
                                    proposal_safe, expected_pass):
    case.f.prefer_feasible_policy, case.f.prefer_policy_margin = prefer_feasible, prefer_margin
    predictions(case, monkeypatch, [np.inf if proposal_safe else .5, np.inf],
                [.1 if proposal_safe else -.1, .12])
    result = case.f.filter(case.env, POLICY)
    assert result[1] != expected_pass
    if not expected_pass:
        assert_parent_retained(case, result)
    assert bool(case.calls) == (prefer_feasible or prefer_margin)


@pytest.mark.parametrize("skip", ["no_backup", "rate_limit", "parent_policy"])
def test_missing_or_unmodelled_backup_never_authorizes_extra_policy_pass(case, monkeypatch, skip):
    parent = v8.SafetyFilterV8._filter

    def adjusted(self, env, action):
        out, changed = parent(self, env, action)
        if skip == "no_backup":
            self.plan = None
        if skip == "parent_policy":
            return POLICY.copy(), False
        return out, changed

    monkeypatch.setattr(v8.SafetyFilterV8, "_filter", adjusted)
    case.env.command_rate_limit = skip == "rate_limit"
    out, changed = case.f.filter(case.env, POLICY)
    assert changed == (skip != "parent_policy")
    assert not case.calls and not case.f.last["v10_pair_checked"]
    assert not case.f.last["v10_policy_preserved"]
    assert case.f.uncertified_steps == 2


def test_dual_response_evaluator_can_reject_policy_in_other_envelope(case, monkeypatch):
    import safety_prediction
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", True)
    predictions(case, monkeypatch, [.5, np.inf], [-.1, .2])
    monkeypatch.setattr(safety_prediction, "evaluate_sequences", lambda *a: SimpleNamespace(
        first=np.array([.5, .2]), clear=np.array([-.1, -.2])))
    assert_parent_retained(case, case.f.filter(case.env, POLICY))
    assert not case.calls


def test_observer_receives_original_history_and_actual_dispatch_after_policy_restoration(case, monkeypatch):
    seen = []
    case.f.observer = SimpleNamespace(predict=lambda *args: seen.append(args))
    predictions(case, monkeypatch, [.5, np.inf], [-.1, .1])
    case.f.filter(case.env, POLICY)
    case.f.observe_issued_command(float(POLICY[0]), 4.8)
    assert len(seen) == 1 and seen[0][0] is case.snap
    assert seen[0][1].values == [.1]
    assert seen[0][2:] == (float(POLICY[0]), 4.8)


def test_constructor_snapshots_v10_flags_independently_of_v9(monkeypatch):
    import safety_v9 as v9
    monkeypatch.setattr(v10, "PREFER_FEASIBLE_POLICY", False)
    monkeypatch.setattr(v10, "PREFER_POLICY_MARGIN", False)
    monkeypatch.setattr(v10, "PREFER_ANY_FEASIBLE_POLICY", True)
    monkeypatch.setattr(v9, "REQUIRE_CURRENT_PLAN", True)
    monkeypatch.setattr(v9, "TARGET_TURN_RATE_DEG_S", 5.)
    f = v10.SafetyFilterV10()
    assert not f.require_current_plan and not f.prefer_feasible_policy and not f.prefer_policy_margin
    assert f.prefer_any_feasible_policy
    assert f.target_turn_rate_rad_s == 0.
    explicit = v10.SafetyFilterV10(prefer_feasible_policy=True, prefer_policy_margin=True,
                                  prefer_any_feasible_policy=False)
    assert explicit.prefer_feasible_policy and explicit.prefer_policy_margin
    assert not explicit.prefer_any_feasible_policy
    assert not f.prefer_feasible_policy and not f.prefer_policy_margin


def test_target_envelope_cache_resets_before_parent_checks(case, monkeypatch):
    case.f._target_snapshot = case.f._target_envelope = object()
    parent = v8.SafetyFilterV8._filter

    def adjusted(self, env, action):
        assert self._target_snapshot is self._target_envelope is None
        return parent(self, env, action)

    monkeypatch.setattr(v8.SafetyFilterV8, "_filter", adjusted)
    case.f.filter(case.env, POLICY)


@pytest.mark.parametrize("bad", [[np.nan, 0.], [0., np.nan]])
def test_nan_policy_is_rejected_before_parent_history_mutation(case, bad):
    with pytest.raises(ValueError, match="finite"):
        case.f.filter(case.env, bad)
    assert case.f.actuators.values == [.1] and not case.calls


def test_any_feasible_preference_is_disabled_by_default(case):
    assert v10.PREFER_ANY_FEASIBLE_POLICY is False
    assert not case.f.prefer_any_feasible_policy


@pytest.mark.parametrize("proposal_first,proposal_clear", [(np.inf, .5), (.5, -.1)])
def test_any_feasible_preference_can_act_without_other_switches(case, monkeypatch,
                                                             proposal_first, proposal_clear):
    case.f.prefer_feasible_policy = case.f.prefer_policy_margin = False
    case.f.prefer_any_feasible_policy = True
    predictions(case, monkeypatch, [proposal_first, np.inf], [proposal_clear, 0.])
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not case.env._v2_brake
    assert case.f.last["why"] == "feasible policy preference"
    assert case.f.last["v10_prefer_any_feasible_policy"]
    assert case.f.last["v10_policy_preserved"]
    assert case.f.last["checked_clearance"] == 0.
    np.testing.assert_array_equal(case.f.plan, case.calls[0][1])
    assert case.f.actuators.values == [.1, float(POLICY[0])]


@pytest.mark.parametrize("policy_first,policy_clear", [(.5, .5), (np.inf, -.01),
                                                       (np.nan, .5), (np.inf, np.nan)])
def test_any_feasible_preference_still_requires_passing_policy(case, monkeypatch,
                                                              policy_first, policy_clear):
    case.f.prefer_any_feasible_policy = True
    predictions(case, monkeypatch, [.5, policy_first], [-.1, policy_clear])
    assert_parent_retained(case, case.f.filter(case.env, POLICY))


@pytest.mark.parametrize("skip", ["no_backup", "rate_limit"])
def test_any_feasible_preference_requires_supported_plan(case, monkeypatch, skip):
    case.f.prefer_any_feasible_policy = True
    parent = v8.SafetyFilterV8._filter

    def adjusted(self, env, action):
        out, changed = parent(self, env, action)
        if skip == "no_backup":
            self.plan = None
        return out, changed

    monkeypatch.setattr(v8.SafetyFilterV8, "_filter", adjusted)
    case.env.command_rate_limit = skip == "rate_limit"
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, OVERRIDE)
    assert changed and not case.calls
    assert not case.f.last["v10_policy_preserved"]
    assert not case.f.last["v10_pair_checked"]
    assert case.f.recovery_steps == 3 and case.f.uncertified_steps == 2
