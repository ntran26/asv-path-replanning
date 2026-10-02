"""Observer dispatch regressions; stop the environment before advancing physics."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
from env import ASVLidarEnv
import safety_observer
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v5 as v5


class _BeforePhysics(Exception):
    pass


@pytest.mark.parametrize("version,expected_class", [
    (2, v2.SafetyFilterV2), (3, v3.SafetyFilterV3), (4, v4.SafetyFilterV4),
    (5, v5.SafetyFilterV5),
])
def test_environment_selects_exact_safety_version_before_physics(monkeypatch, version, expected_class):
    monkeypatch.setattr(cfg, "SAFETY_VERSION", version, raising=False)
    monkeypatch.setattr(v4, "MODEL_EGO_OBSERVER", False)
    monkeypatch.setattr(v4, "FREE_SPACE_MEMORY", False)
    env = ASVLidarEnv.__new__(ASVLidarEnv)
    env.elapsed_time, env.rudder = 0.0, 0.0
    env.estop_enabled, env.command_rate_limit = True, False
    env._safety_v2, env._v2_brake = None, False
    env._observed_surge, env.u_body = 0.5, 0.5
    env.asv_x, env.asv_y = 5.0, 5.0
    env._emergency_stop_override = lambda: None

    def select(self, environment, action):
        assert type(self) is expected_class
        return action, False

    monkeypatch.setattr(expected_class, "filter", select)

    def first_physics_call(*args):
        assert type(env._safety_v2) is expected_class
        raise _BeforePhysics

    env.model = SimpleNamespace(update=first_physics_call)
    with pytest.raises(_BeforePhysics):
        env.step(np.zeros(2))
    assert type(env._safety_v2) is expected_class


@pytest.mark.parametrize("surge,expected_rpm", [(0.5, -24.0), (0.0, 0.0)])
@pytest.mark.parametrize("limited,expected_rudder", [(False, 1.0), (True, 0.25)])
def test_environment_dispatches_signed_rpm_and_limited_rudder_before_physics(
        monkeypatch, surge, expected_rpm, limited, expected_rudder):
    monkeypatch.setattr(cfg, "SAFETY_VERSION", 4, raising=False)
    monkeypatch.setattr(cfg, "FIXED_RPM", False)
    monkeypatch.setattr(cfg, "CRUISE_RPM", 6.0)
    monkeypatch.setattr(cfg, "RPM_DELTA", 6.0)
    monkeypatch.setattr(cfg, "RPM_FLOOR", 0.0)
    monkeypatch.setattr(cfg, "RPM_CEIL", 12.0)
    env = ASVLidarEnv.__new__(ASVLidarEnv)
    env.elapsed_time, env.rudder = 0.0, 0.0
    env.estop_enabled, env.command_rate_limit = True, limited
    env._observed_surge, env.u_body = surge, surge
    env.asv_x, env.asv_y = 5.0, 5.0
    env._emergency_stop_override = lambda: None
    observed = []

    def select(environment, action):
        environment._v2_brake = True
        return action, True

    def issued(rudder, rpm):
        observed.append((rudder, rpm))
        assert env._executed_action[1] == -1.0  # Identical for zero and astern.

    env._safety_v2 = SimpleNamespace(filter=select, observe_issued_command=issued)

    def first_physics_call(rpm, rudder_percent, dt):
        assert observed == [(expected_rudder, expected_rpm)]
        assert rpm == expected_rpm
        assert rudder_percent == expected_rudder * 100.0
        raise _BeforePhysics

    env.model = SimpleNamespace(update=first_physics_call)
    with pytest.raises(_BeforePhysics):
        env.step(np.array([1.0, -1.0]))
    assert observed == [(expected_rudder, expected_rpm)]


def _idle_filter(monkeypatch, enabled=True):
    monkeypatch.setattr(v4, "MODEL_EGO_OBSERVER", enabled)
    monkeypatch.setattr(v4, "FREE_SPACE_MEMORY", False)
    filter_ = v4.SafetyFilterV4()
    monkeypatch.setattr(filter_, "_threat_in_reach", lambda snap: False)
    monkeypatch.setattr(filter_.perception, "snapshot", lambda env: SimpleNamespace(
        x=5.0, y=5.0, heading=0.0, u=0.8, v=0.0, r=0.0))
    return filter_


def test_v4_observer_uses_actuator_history_before_issue(monkeypatch):
    filter_ = _idle_filter(monkeypatch)
    filter_.actuators.servo = 0.12
    filter_.actuators.buffer = [0.05] * cc.DELAY_STEPS
    before = copy.deepcopy(vars(filter_.actuators))
    env = SimpleNamespace(command_rate_limit=True, pose_stale=False)
    filter_.filter(env, np.array([1.0, 0.0]))
    assert filter_.actuators.executed == 0.25
    assert vars(filter_._observer_actuators) == before
    assert filter_._observer_actuators is not filter_.actuators
    observed = []

    def predict(snap, act, rudder, rpm):
        observed.append((snap.u, vars(act).copy(), rudder, rpm))
        return np.array([0.3, 0.02, 0.01])

    monkeypatch.setattr(safety_observer, "advance_ego", predict)
    filter_.observe_issued_command(0.25, -24.0)
    assert observed == [(0.8, before, 0.25, -24.0)]
    np.testing.assert_array_equal(filter_.observer.prior, [0.3, 0.02, 0.01])


def test_v4_stale_measurement_uses_prior_without_repeated_correction(monkeypatch):
    filter_ = _idle_filter(monkeypatch)
    env = SimpleNamespace(command_rate_limit=False, pose_stale=False)
    filter_.filter(env, np.zeros(2))
    prior = np.array([0.3, 0.02, 0.01])
    monkeypatch.setattr(safety_observer, "advance_ego", lambda *args: prior.copy())
    filter_.observe_issued_command(0.0, -24.0)
    env.pose_stale = True
    filter_.filter(env, np.zeros(2))
    np.testing.assert_array_equal(filter_.ego, prior)
    np.testing.assert_array_equal(filter_._raw_ego, [0.8, 0.0, 0.0])
    assert filter_._observer_snapshot.u == prior[0]
    # The same held reading remains skipped across the next issued command.
    prior[:] = [0.1, 0.01, 0.008]
    filter_.observe_issued_command(0.0, -24.0)
    filter_.filter(env, np.zeros(2))
    np.testing.assert_array_equal(filter_.ego, prior)


def test_observer_disabled_preserves_existing_ema_and_dispatch_is_noop(monkeypatch):
    filter_ = _idle_filter(monkeypatch, enabled=False)
    filter_.ego = np.array([0.4, 0.1, 0.01])
    expected = filter_.ego + v2.EGO_SMOOTHING * (np.array([0.8, 0.0, 0.0]) - filter_.ego)
    env = SimpleNamespace(command_rate_limit=False, pose_stale=False)
    filter_.filter(env, np.zeros(2))
    np.testing.assert_allclose(filter_.ego, expected)
    assert filter_.observer is None
    filter_.observe_issued_command(0.0, -24.0)
    np.testing.assert_allclose(filter_.ego, expected)
