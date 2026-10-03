"""Synthetic fallback/isolation contracts; no environments, resets or episodes."""
from concurrent.futures import ThreadPoolExecutor
import math
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

import safety_v2 as v2
import safety_v4 as v4
import safety_v6 as v6
import safety_v7 as v7
import safety_v8 as v8


POLICY = np.array([.123, .456], dtype=np.float32)


@pytest.fixture(autouse=True)
def isolate_flags(monkeypatch):
    monkeypatch.setattr(v2, "HOLD_BACK_GAIN_S", 1.)
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", False)
    monkeypatch.setattr(v6, "TRAJECTORY_SEARCH", False)
    monkeypatch.setattr(v7, "RISK_MONITOR_ENABLED", False)
    monkeypatch.setattr(v7, "POLICY_PREFIX_REPAIR", False)


def test_default_accessor_reads_existing_constant_and_v8_is_instance_local(monkeypatch):
    legacy = [object.__new__(cls) for cls in (v6.SafetyFilterV6, v7.SafetyFilterV7)]
    isolated = object.__new__(v8.SafetyFilterV8)
    for value in (1., 2.5):
        monkeypatch.setattr(v2, "HOLD_BACK_GAIN_S", value)
        assert [f._hold_back_gain_s() for f in legacy] == [value, value]
        assert math.isinf(isolated._hold_back_gain_s())
        assert v2.HOLD_BACK_GAIN_S == value


def test_nested_v7_decision_sees_finite_gain_during_v8_call(monkeypatch):
    outside = object.__new__(v8.SafetyFilterV8)
    nested = object.__new__(v7.SafetyFilterV7)
    observations = []

    def parent(self, env, action):
        observations.append((type(self).__name__, self._hold_back_gain_s(), v2.HOLD_BACK_GAIN_S))
        if self is outside:
            nested._filter(env, action)
        return action, False

    monkeypatch.setattr(v7.SafetyFilterV7, "_filter", parent)
    outside._filter(None, POLICY)
    assert observations == [("SafetyFilterV8", math.inf, 1.), ("SafetyFilterV7", 1., 1.)]


def test_concurrent_v7_observation_sees_no_temporary_global_write(monkeypatch):
    inside = object.__new__(v8.SafetyFilterV8)
    legacy = object.__new__(v7.SafetyFilterV7)
    rendezvous = Barrier(2)

    def parent(self, env, action):
        rendezvous.wait(timeout=5.)
        rendezvous.wait(timeout=5.)
        return action, False

    def observe():
        rendezvous.wait(timeout=5.)
        observed = legacy._hold_back_gain_s(), v2.HOLD_BACK_GAIN_S
        rendezvous.wait(timeout=5.)
        return observed

    monkeypatch.setattr(v7.SafetyFilterV7, "_filter", parent)
    with ThreadPoolExecutor(max_workers=2) as workers:
        active = workers.submit(inside._filter, None, POLICY)
        assert workers.submit(observe).result(timeout=10.) == (1., 1.)
        assert active.result(timeout=10.)[1] is False


def test_exception_path_does_not_write_shared_gain(monkeypatch):
    def failed(self, env, action):
        assert v2.HOLD_BACK_GAIN_S == 1.
        assert self._hold_back_gain_s() == math.inf
        raise RuntimeError("synthetic interrupted decision")

    monkeypatch.setattr(v7.SafetyFilterV7, "_filter", failed)
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        object.__new__(v8.SafetyFilterV8)._filter(None, POLICY)
    assert v2.HOLD_BACK_GAIN_S == 1.


class Commands:
    def __init__(self):
        self.issued = []

    def issue(self, env, rudder):
        self.issued.append(float(rudder))


def synthetic_filter(cls, monkeypatch, *, safe=False):
    filt = cls()
    filt.observer = None
    snap = SimpleNamespace(u=.6, v=0., r=0., tracks=[])
    filt.perception = SimpleNamespace(snapshot=lambda env: snap)
    filt.actuators = Commands()
    monkeypatch.setattr(filt, "_threat_in_reach", lambda snap: True)
    monkeypatch.setattr(filt, "_rejoin", lambda snap: np.array([0., 0.]))
    monkeypatch.setattr(filt, "_rollout", lambda snap, act, plans: plans)

    def evaluate(snap, plans):
        policy = np.all(np.isclose(plans[:, 0], POLICY), axis=1)
        if safe:
            return np.full(len(plans), np.inf), np.full(len(plans), .4)
        # Every sequence fails. A non-policy command delays the predicted
        # first contact by 2 s: sufficient for the unchanged legacy 1 s gate.
        return np.where(policy, 1., 3.), np.where(policy, -.2, -.1)

    monkeypatch.setattr(filt, "_evaluate", evaluate)
    return filt, SimpleNamespace(pose_stale=False, command_rate_limit=False)


@pytest.mark.parametrize("cls", [v6.SafetyFilterV6, v7.SafetyFilterV7, v8.SafetyFilterV8])
def test_hard_safe_policy_branch_is_unchanged(cls, monkeypatch):
    filt, env = synthetic_filter(cls, monkeypatch, safe=True)
    out, changed = filt.filter(env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not env._v2_brake
    assert filt.last["why"] == "nominal"
    assert filt.last["checked_clearance"] == .4
    assert filt.actuators.issued == [float(POLICY[0])]
    assert v2.HOLD_BACK_GAIN_S == 1.


@pytest.mark.parametrize("cls", [v6.SafetyFilterV6, v7.SafetyFilterV7])
@pytest.mark.parametrize("gain,expected", [(1., "hold back"), (3., "no escape")])
def test_v6_v7_retain_existing_all_unsafe_policy_cases(cls, gain, expected, monkeypatch):
    monkeypatch.setattr(v2, "HOLD_BACK_GAIN_S", gain)
    filt, env = synthetic_filter(cls, monkeypatch)
    out, changed = filt.filter(env, POLICY)
    assert filt.last["why"] == expected
    assert changed == (expected == "hold back")
    assert len(filt.actuators.issued) == 1
    if not changed:
        np.testing.assert_array_equal(out, POLICY)
    assert v2.HOLD_BACK_GAIN_S == gain


def test_v8_all_unsafe_no_plan_passes_policy_without_mutating_legacy_setting(monkeypatch):
    filt, env = synthetic_filter(v8.SafetyFilterV8, monkeypatch)
    out, changed = filt.filter(env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not env._v2_brake
    assert filt.last["why"] == "no escape"
    assert filt.plan is None and filt.mode == "nominal"
    assert filt.actuators.issued == [float(POLICY[0])]
    assert v2.HOLD_BACK_GAIN_S == 1.


def test_v8_still_has_currently_rejected_last_certificate_fallback(monkeypatch):
    filt, env = synthetic_filter(v8.SafetyFilterV8, monkeypatch)
    filt.plan = np.tile([-.8, 0.], (16, 1))
    out, changed = filt.filter(env, POLICY)
    assert changed and filt.last["why"] == "last certificate"
    assert not filt.last["continuation_ok"]
    assert filt.last["checked_clearance"] < 0.
    np.testing.assert_array_equal(out, np.array([-.8, 0.], dtype=np.float32))
    assert filt.uncertified_steps == 1
