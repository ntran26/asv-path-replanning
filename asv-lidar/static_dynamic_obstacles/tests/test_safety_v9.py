"""V9 control contracts with synthetic predictions only; no episodes."""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v8 as v8
import safety_v9 as v9


POLICY = np.array([.3, -.2], dtype=np.float32)


class Commands:
    def __init__(self):
        self.values = [.1]

    def issue(self, env, rudder):
        self.values.append(float(rudder))


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", False)
    f = v9.SafetyFilterV9()
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
        self.actuators.issue(env, -.8)
        env._v2_brake = True
        return np.array([-.8, -1.], dtype=np.float32), True

    monkeypatch.setattr(v8.SafetyFilterV8, "_filter", parent)

    def rollout(snapshot, actuators, seq):
        assert snapshot is snap
        assert actuators.values == [.1]
        calls.append(seq.copy())
        return seq

    monkeypatch.setattr(f, "_rollout", rollout)
    monkeypatch.setattr(f, "_evaluate", lambda s, seq: (
        np.array([np.inf, .5]), np.array([.1, -.4])))
    return SimpleNamespace(f=f, env=env, plan=plan, snap=snap, calls=calls)


def predictions(case, monkeypatch, first, clear):
    monkeypatch.setattr(case.f, "_evaluate", lambda s, seq: (
        np.asarray(first, dtype=float), np.asarray(clear, dtype=float)))


def test_hard_safe_sub_trigger_intervention_remains_eligible(case):
    out, changed = case.f.filter(case.env, POLICY)
    assert changed and case.env._v2_brake
    assert case.f.last["checked_clearance"] == .1 < v2.TRIGGER_MARGIN_M
    assert case.f.uncertified_steps == 0
    assert case.f.actuators.values == [.1, -.8]
    np.testing.assert_array_equal(case.f.plan, case.calls[0][0])
    assert np.isnan(case.f.plan[0, 1])  # Full astern retained, not policy-space -1.


@pytest.mark.parametrize("first,clear", [([.5, .125], [-.1, -.8]),
                                       ([np.inf, np.inf], [-.01, .05]),
                                       ([np.nan, np.inf], [.1, .05]),
                                       ([-np.inf, np.inf], [.1, .05]),
                                       ([np.inf, .5], [np.nan, -.4])])
def test_unsafe_or_unknown_override_is_suppressed_even_when_less_bad(case, monkeypatch, first, clear):
    predictions(case, monkeypatch, first, clear)
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not case.env._v2_brake
    assert case.f.last["why"] == "unchecked override suppressed"
    assert case.f.mode == "nominal" and case.f.uncertified_steps == 0
    assert case.f.actuators.values == [.1, float(POLICY[0])]
    policy_passing = np.isposinf(first[1]) and clear[1] >= 0
    assert (case.f.plan is not None) == policy_passing
    if not policy_passing:
        assert case.f.last["checked_clearance"] is None


@pytest.mark.parametrize("policy_clear", [.1, .12])
def test_equal_or_better_policy_margin_preserves_sac_with_identical_tail(case, monkeypatch, policy_clear):
    predictions(case, monkeypatch, [np.inf, np.inf], [.1, policy_clear])
    out, changed = case.f.filter(case.env, POLICY)
    assert not changed and case.f.last["why"] == "policy margin dominates"
    assert case.f.last["v9_policy_tail_clearance"] == policy_clear
    # Comparison did not use the unrelated rounded policy-margin field.
    assert case.f.last["policy_margin"] == -.9
    pair = case.calls[0]
    np.testing.assert_array_equal(pair[0, 1:], pair[1, 1:])
    np.testing.assert_array_equal(pair[1, 0], POLICY)
    np.testing.assert_array_equal(case.f.plan, pair[1])
    assert len(case.f.plan) * cfg.UPDATE_RATE >= v2.HORIZON_S


def test_lower_policy_margin_keeps_currently_passing_proposal(case, monkeypatch):
    predictions(case, monkeypatch, [np.inf, np.inf], [.1, .08])
    assert case.f.filter(case.env, POLICY)[1]
    assert not case.f.last["v9_override_suppressed"]


def test_finite_policy_violation_is_not_accepted_despite_positive_margin(case, monkeypatch):
    predictions(case, monkeypatch, [np.inf, .5], [.1, .5])
    assert case.f.filter(case.env, POLICY)[1]


@pytest.mark.parametrize("reason", ["pass", "no_plan", "rate_limit"])
def test_unchecked_paths_do_not_claim_a_trajectory_check(case, monkeypatch, reason):
    parent = v8.SafetyFilterV8._filter

    def adjusted(self, env, action):
        out, changed = parent(self, env, action)
        if reason == "no_plan":
            self.plan = None
        if reason == "pass":
            return POLICY.copy(), False
        return out, changed

    monkeypatch.setattr(v8.SafetyFilterV8, "_filter", adjusted)
    case.env.command_rate_limit = reason == "rate_limit"
    out, changed = case.f.filter(case.env, POLICY)
    assert not changed and not case.calls
    assert not case.f.last["v9_current_plan_checked"]
    if reason != "pass":
        assert case.f.plan is None and not case.env._v2_brake


def test_dual_response_braking_check_is_preserved(case, monkeypatch):
    import safety_prediction
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", True)
    monkeypatch.setattr(safety_prediction, "evaluate_sequences", lambda *a: SimpleNamespace(
        first=np.array([.5, .2]), clear=np.array([-.1, -.2])))
    assert not case.f.filter(case.env, POLICY)[1]
    assert case.f.last["why"] == "unchecked override suppressed"


def test_disabled_guards_preserve_parent_plan_and_uncertified_counter(case):
    case.f.require_current_plan = case.f.prefer_policy_margin = False
    assert case.f.filter(case.env, POLICY)[1]
    assert not case.calls and case.f.uncertified_steps == 2
    np.testing.assert_array_equal(case.f.plan, case.plan)


def test_current_plan_only_ablation_does_not_apply_margin_preference(case, monkeypatch):
    case.f.prefer_policy_margin = False
    predictions(case, monkeypatch, [np.inf, np.inf], [.1, .12])
    assert case.f.filter(case.env, POLICY)[1]
    assert not case.f.last["v9_override_suppressed"]


def test_margin_only_ablation_does_not_remove_unsafe_parent_fallback(case, monkeypatch):
    case.f.require_current_plan = False
    predictions(case, monkeypatch, [.5, np.inf], [-.1, .12])
    assert case.f.filter(case.env, POLICY)[1]
    assert case.f.uncertified_steps == 2
    np.testing.assert_array_equal(case.f.plan, case.plan)


def test_constructor_snapshots_ablation_switches(monkeypatch):
    monkeypatch.setattr(v9, "REQUIRE_CURRENT_PLAN", False)
    monkeypatch.setattr(v9, "PREFER_POLICY_MARGIN", False)
    f = v9.SafetyFilterV9()
    assert not f.require_current_plan and not f.prefer_policy_margin
    explicit = v9.SafetyFilterV9(require_current_plan=True, prefer_policy_margin=True)
    assert explicit.require_current_plan and explicit.prefer_policy_margin


@pytest.mark.parametrize("rate", [-1, np.inf, np.nan])
def test_invalid_target_hypothesis_rate_is_rejected(rate):
    with pytest.raises(ValueError):
        v9.SafetyFilterV9(target_turn_rate_deg_s=rate)


def test_optional_target_check_is_cached_and_never_relaxes_original_check(monkeypatch):
    import safety_target_prediction as target
    f = v9.SafetyFilterV9(target_turn_rate_deg_s=5.)
    snap = SimpleNamespace(tracks=[object()])
    calls = []
    class Envelope:
        def evaluate(self, ro):
            return np.array([1., np.inf]), np.array([-.2, .3])
    def build(snapshot, *, turn_rate_rad_s):
        calls.append(snapshot)
        assert turn_rate_rad_s == pytest.approx(np.radians(5.))
        return Envelope()
    monkeypatch.setattr(target.TargetPredictionEnvelope, "from_snapshot", build)
    monkeypatch.setattr(v8.SafetyFilterV8, "_evaluate", lambda *a: (
        np.array([np.inf, .5]), np.array([.5, -.1])))
    for _ in range(3):
        first, clear = f._evaluate(snap, object())
        np.testing.assert_array_equal(first, [1., .5])
        np.testing.assert_array_equal(clear, [-.2, -.1])
    assert calls == [snap]
    other = SimpleNamespace(tracks=[object()])
    f._evaluate(other, object())
    assert calls == [snap, other]


def test_default_target_mode_is_unchanged_cv_without_envelope(monkeypatch):
    import safety_target_prediction as target
    f = v9.SafetyFilterV9()
    first, clear = np.array([np.inf]), np.array([.2])
    monkeypatch.setattr(v8.SafetyFilterV8, "_evaluate", lambda *a: (first, clear))
    monkeypatch.setattr(target.TargetPredictionEnvelope, "from_snapshot",
                        lambda *a, **kw: pytest.fail("CV default must not create extra hypotheses"))
    got = f._evaluate(SimpleNamespace(tracks=[object()]), object())
    assert got[0] is first and got[1] is clear


@pytest.mark.parametrize("version", [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
def test_native_dispatch_stops_before_physics(monkeypatch, version):
    from env import ASVLidarEnv
    calls = []
    class BeforePlant(Exception):
        pass
    class Sentinel:
        def filter(self, env, action):
            raise BeforePlant()
    def factory():
        calls.append(version)
        return Sentinel()
    monkeypatch.setitem(sys.modules, f"safety_v{version}",
                        SimpleNamespace(**{f"SafetyFilterV{version}": factory}))
    monkeypatch.setattr(cfg, "SAFETY_VERSION", version, raising=False)
    env = ASVLidarEnv.__new__(ASVLidarEnv)
    env.elapsed_time, env.estop_enabled = 0., True
    for _ in range(2):
        with pytest.raises(BeforePlant):
            env.step(object())
    assert calls == [version]
