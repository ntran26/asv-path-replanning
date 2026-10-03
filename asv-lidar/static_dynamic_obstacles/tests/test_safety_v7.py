"""Policy-preservation contracts, synthetic predictions only; no episodes."""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v6 as v6
import safety_v7 as v7


POLICY = np.array([.35, -.25], dtype=np.float32)


def test_replacement_changes_only_first_command_and_never_input():
    plan = v3.plan_for([-.8, 0.], [1., .2])
    original = plan.copy()
    replacement = v7.replace_first_action(plan, POLICY)
    np.testing.assert_array_equal(replacement[0], POLICY)
    np.testing.assert_array_equal(replacement[1:], original[1:])
    np.testing.assert_array_equal(plan, original)
    assert len(replacement) * cfg.UPDATE_RATE >= v2.HORIZON_S


@pytest.mark.parametrize("last_throttle,padded_throttle", [(np.nan, 0.), (-.5, -.5)])
def test_spent_plan_is_padded_before_full_horizon_check(last_throttle, padded_throttle):
    plan = np.array([[.8, 0.], [-.8, last_throttle]])
    replacement = v7.replace_first_action(plan, POLICY)
    np.testing.assert_array_equal(replacement[1], plan[1])
    np.testing.assert_array_equal(replacement[2:],
                                  np.tile([0., padded_throttle], (len(replacement)-2, 1)))


@pytest.mark.parametrize("plan", [[], [[np.nan, 0.]], [[2., 0.]], [[0., np.inf]], [[0., 2.]]])
def test_malformed_backups_fail_explicitly(plan):
    with pytest.raises(ValueError):
        v7.replace_first_action(plan, POLICY)


class Commands:
    def __init__(self):
        self.values = [0.1]

    def issue(self, env, rudder):
        self.values.append(float(rudder))


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(v7, "RISK_MONITOR_ENABLED", False)
    monkeypatch.setattr(v7, "POLICY_PREFIX_REPAIR", True)
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", False)
    f = v7.SafetyFilterV7()
    f.actuators = Commands()
    f.observer = None
    snap = object()
    plan = v3.plan_for([-.8, np.nan], [1., .2])
    calls = []

    def parent(self, env, action):
        self._observer_snapshot = snap
        self.plan = plan.copy()
        self.mode = "recovery"
        self.recovery_steps, self.uncertified_steps = 7, 2
        self.last = {"why": "brake", "changed": True, "brake": True}
        env._v2_brake = True
        self.actuators.issue(env, -.8)
        return np.array([-.8, -1.], dtype=np.float32), True

    monkeypatch.setattr(v6.SafetyFilterV6, "_filter", parent)

    def rollout(snapshot, actuators, sequences):
        assert snapshot is snap
        assert actuators.values == [.1]  # No proposed command leaked into check.
        calls.append(sequences.copy())
        return sequences

    monkeypatch.setattr(f, "_rollout", rollout)
    monkeypatch.setattr(f, "_evaluate", lambda s, seq: (np.array([np.inf]), np.array([.4])))
    return SimpleNamespace(f=f, env=SimpleNamespace(command_rate_limit=False),
                           calls=calls, plan=plan, snap=snap)


def test_valid_policy_substitution_resets_recovery_keeps_exact_checked_tail(case):
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    np.testing.assert_array_equal(case.f.plan, case.calls[0][0])
    np.testing.assert_array_equal(case.f.plan[1:], case.plan[1:])
    assert not changed and not case.env._v2_brake
    assert case.f.mode == "nominal"
    assert case.f.recovery_steps == case.f.uncertified_steps == 0
    assert case.f.actuators.values == [.1, float(POLICY[0])]
    assert case.f.last["v6_proposal_brake"]
    assert case.f.last["v6_proposed_change"]
    assert case.f.last["v7_policy_preserved"]
    assert case.f.last["nominal_plan_retained"]


@pytest.mark.parametrize("first,clearance", [(1., 1.), (-np.inf, 1.), (np.nan, 1.),
                                           (np.inf, .149), (np.inf, np.nan)])
def test_failed_full_check_preserves_v6_action_brake_plan_and_history(case, monkeypatch,
                                                                    first, clearance):
    monkeypatch.setattr(case.f, "_evaluate", lambda s, seq: (
        np.array([first]), np.array([clearance])))
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, np.array([-.8, -1.], dtype=np.float32))
    np.testing.assert_array_equal(case.f.plan, case.plan)
    assert changed and case.env._v2_brake
    assert case.f.actuators.values == [.1, -.8]
    assert case.f.mode == "recovery" and case.f.recovery_steps == 7
    assert case.f.last["v7_repair_checked"] and not case.f.last["v7_policy_preserved"]


def test_exact_existing_margin_is_accepted(case, monkeypatch):
    monkeypatch.setattr(case.f, "_evaluate", lambda s, seq: (
        np.array([np.inf]), np.array([v2.TRIGGER_MARGIN_M])))
    assert case.f.filter(case.env, POLICY)[1] is False


@pytest.mark.parametrize("reason", ["disabled", "rate_limit", "no_plan", "already_passed"])
def test_ineligible_branch_never_changes_parent_output(case, monkeypatch, reason):
    parent = v6.SafetyFilterV6._filter
    if reason == "disabled":
        monkeypatch.setattr(v7, "POLICY_PREFIX_REPAIR", False)
    elif reason == "rate_limit":
        case.env.command_rate_limit = True
    else:
        def altered(self, env, action):
            out, changed = parent(self, env, action)
            if reason == "no_plan":
                self.plan = None
            else:
                out, changed = POLICY.copy(), False
                env._v2_brake = False
            return out, changed
        monkeypatch.setattr(v6.SafetyFilterV6, "_filter", altered)
    out, changed = case.f.filter(case.env, POLICY)
    assert not case.calls and not case.f.last["v7_repair_checked"]
    assert changed == (reason != "already_passed")


@pytest.mark.parametrize("recommendation", [False, True])
def test_shadow_recommendation_never_bypasses_failed_check(case, monkeypatch, recommendation):
    monkeypatch.setattr(v7, "RISK_MONITOR_ENABLED", True)
    monkeypatch.setattr(case.f, "_evaluate", lambda s, seq: (np.array([.5]), np.array([-.1])))
    observed = []

    def monitor(snap, policy, actuators, rollout, *, fresh):
        observed.append((copy.deepcopy(actuators.values), fresh))
        return SimpleNamespace(as_dict=lambda: {"recommend_rescue": recommendation})

    monkeypatch.setattr(case.f.risk_monitor, "update", monitor)
    out, changed = case.f.filter(case.env, POLICY)
    assert changed and case.env._v2_brake
    assert observed == [([.1], True)]
    assert case.f.last["risk_monitor"]["recommend_rescue"] == recommendation


def test_dual_brake_envelope_remains_required(case, monkeypatch):
    import safety_prediction
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", True)
    monkeypatch.setattr(safety_prediction, "evaluate_sequences", lambda *a: SimpleNamespace(
        first=np.array([.5]), clear=np.array([-.1])))
    assert case.f.filter(case.env, POLICY)[1] is True
    assert not case.f.last["v7_policy_preserved"]


@pytest.mark.parametrize("version", [2, 3, 4, 5, 6, 7])
def test_native_dispatch_without_constructor_reset_or_plant(monkeypatch, version):
    from env import ASVLidarEnv
    factories = []

    class BeforePlant(Exception):
        pass

    class Sentinel:
        def filter(self, env, action):
            raise BeforePlant()

    for selected in range(2, 8):
        def factory(selected=selected):
            factories.append(selected)
            return Sentinel()
        monkeypatch.setitem(sys.modules, f"safety_v{selected}",
                            SimpleNamespace(**{f"SafetyFilterV{selected}": factory}))
    monkeypatch.setattr(cfg, "SAFETY_VERSION", version, raising=False)
    env = ASVLidarEnv.__new__(ASVLidarEnv)
    env.elapsed_time, env.estop_enabled = 0., True
    for _ in range(2):
        with pytest.raises(BeforePlant):
            env.step(object())
    assert factories == [version]
