"""Synthetic observer/controller checks only; no environments or episodes."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
import safety_observer as original
import safety_v4
import safety_v16
import safety_v17
from safety_v17 import SafetyFilterV17
from safety_yaw_observer import FreshYawObserver


def snapshot(ego):
    return SimpleNamespace(u=float(ego[0]), v=float(ego[1]), r=float(ego[2]),
                           heading=.3, x=2., y=4.)


@pytest.mark.parametrize("fresh", [True, False])
def test_initial_measurement_and_return_are_independent(fresh):
    obs = FreshYawObserver()
    raw = np.array([.6, .02, -.13])
    result = obs.update(raw, fresh=fresh)
    np.testing.assert_array_equal(result, raw)
    assert obs.last["yaw_source"] == "initial_measurement"
    assert obs.last["prior_yaw_rate_rad_s"] is None
    result[:] = 42.
    raw[:] = 21.
    np.testing.assert_array_equal(obs.estimate, [.6, .02, -.13])


@pytest.mark.parametrize("has_prediction", [False, True])
def test_only_fresh_yaw_correction_changes_for_same_prior(has_prediction):
    obs, parent = FreshYawObserver(), original.EgoObserver()
    for item in (obs, parent):
        item.update([.4, -.03, .2])
        if has_prediction:
            item.prior = np.array([.3, -.01, .4])
    raw = np.array([.7, .08, -.19])
    result, reference = obs.update(raw), parent.update(raw)
    np.testing.assert_array_equal(result[:2], reference[:2])
    assert result[2] == raw[2] and result[2] != reference[2]
    assert obs.prior is None
    assert obs.last["yaw_source"] == "fresh_measurement"
    np.testing.assert_array_equal(raw, [.7, .08, -.19])


def test_stale_measurement_uses_and_consumes_prediction_without_reassimilation():
    obs = FreshYawObserver()
    obs.update([.5, .02, .1])
    obs.prior = np.array([.4, .03, .2])
    expected = obs.prior.copy()
    np.testing.assert_array_equal(obs.update([9., 8., -7.], fresh=False), expected)
    assert obs.prior is None and obs.last["yaw_source"] == "model_or_held_prior"
    np.testing.assert_array_equal(obs.update([3., 2., 1.], fresh=False), expected)
    assert obs.last["prior_yaw_rate_rad_s"] == .2


def test_prediction_keeps_parent_method_and_uses_corrected_snapshot(monkeypatch):
    assert FreshYawObserver.predict is original.EgoObserver.predict
    obs = FreshYawObserver()
    obs.update([.5, 0., .1])
    obs.prior = np.array([.3, -.02, .7])
    corrected = obs.update([.6, .03, -.2])
    snap, act = snapshot(corrected), cc.Actuators()
    calls = []
    def advance(s, a, rudder, rpm):
        calls.append((s, a, rudder, rpm))
        # Coupling is intentional: changed yaw may change later u/v priors.
        return np.array([s.u + s.r, s.v - s.r, s.r + .05])
    monkeypatch.setattr(original, "advance_ego", advance)
    prediction = obs.predict(snap, act, -.6, -24.)
    assert calls == [(snap, act, -.6, -24.)]
    assert snap.r == -.2
    expected = np.array([snap.u-.2, snap.v+.2, -.15])
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-15)
    prediction[:] = 99.
    np.testing.assert_allclose(obs.update([.6, .03, -.2], fresh=False), expected, rtol=0, atol=1e-15)
    new = obs.update([.2, .01, -.5], fresh=True)
    assert new[2] == -.5


def test_reset_discards_prior_estimate_and_diagnostics():
    obs = FreshYawObserver()
    obs.update([.5, 0., .1])
    obs.prior = np.array([.1, .02, .3])
    obs.reset()
    assert obs.estimate is None and obs.prior is None and obs.last == {}
    np.testing.assert_array_equal(obs.update([.7, .05, -.2], fresh=False), [.7, .05, -.2])


@pytest.mark.parametrize("bad", [[np.nan, 0., 0.], [0., np.inf, 0.], [0., 0., -np.inf], [1., 2.]])
@pytest.mark.parametrize("fresh", [False, True])
def test_invalid_measurement_preserves_existing_state(bad, fresh):
    obs = FreshYawObserver()
    obs.update([.5, .02, .1])
    obs.prior = np.array([.4, .01, .2])
    before = copy.deepcopy(vars(obs))
    with pytest.raises(ValueError):
        obs.update(bad, fresh=fresh)
    np.testing.assert_array_equal(obs.estimate, before["estimate"])
    np.testing.assert_array_equal(obs.prior, before["prior"])
    assert obs.last == before["last"]


def test_predict_before_initialization_retains_parent_error():
    with pytest.raises(RuntimeError, match="call update"):
        FreshYawObserver().predict(snapshot([0., 0., 0.]), cc.Actuators(), 0., 0.)


def test_constructor_and_disabled_option_preserve_parent_selection():
    enabled = SafetyFilterV17(motion_axis_geometry=False, conditional_prefix=False,
                              policy_prefix_search=False, max_coast_s=4.)
    assert isinstance(enabled.observer, FreshYawObserver)
    assert enabled.track_persistence.max_coast_s == 4.
    assert not enabled.motion_axis_geometry and not enabled.conditional_prefix
    assert not enabled.policy_prefix_search
    disabled = SafetyFilterV17(fresh_yaw_measurement=False)
    assert type(disabled.observer) is original.EgoObserver
    assert disabled.motion_axis_geometry and disabled.conditional_prefix


def test_disabled_option_keeps_exact_parent_observer_object(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(safety_v16.SafetyFilterV16, "__init__", lambda self, **kwargs: setattr(self, "observer", sentinel))
    assert SafetyFilterV17(fresh_yaw_measurement=False).observer is sentinel


def test_module_switch_is_per_construction_and_explicit_option_wins(monkeypatch):
    monkeypatch.setattr(safety_v17, "FRESH_YAW_MEASUREMENT", False)
    assert type(SafetyFilterV17().observer) is original.EgoObserver
    assert isinstance(SafetyFilterV17(fresh_yaw_measurement=True).observer, FreshYawObserver)


def test_ablation_does_not_silently_enable_disabled_model_observer(monkeypatch):
    monkeypatch.setattr(safety_v4, "MODEL_EGO_OBSERVER", False)
    with pytest.raises(ValueError, match="requires the inherited model ego observer"):
        SafetyFilterV17()
    assert SafetyFilterV17(fresh_yaw_measurement=False).observer is None


@pytest.mark.parametrize("enabled", [False, True])
def test_parent_result_is_unchanged_and_diagnostics_are_detached(monkeypatch, enabled):
    action = np.array([.2, -.4])
    def parent(self, env, policy):
        self.last = {"why": "parent"}
        return action, True
    monkeypatch.setattr(safety_v16.SafetyFilterV16, "_filter", parent)
    filt = SafetyFilterV17(fresh_yaw_measurement=enabled)
    filt.observer.update([.5, .02, -.1])
    out, changed = filt._filter(object(), action)
    assert out is action and changed is True and filt.last["why"] == "parent"
    assert filt.last["v17_fresh_yaw_measurement"] is enabled
    if enabled:
        filt.observer.last["yaw_source"] = "changed"
        assert filt.last["yaw_observer"]["yaw_source"] == "initial_measurement"
    else:
        assert filt.last["yaw_observer"] == {}


class OnboardOnly:
    pose_stale = False
    command_rate_limit = False

    def __getattr__(self, name):
        if name in ("targets", "asv_x", "asv_y", "asv_w", "u_body", "v_body", "raw_ranges", "_s"):
            raise AssertionError("unexpected truth or sensor access: " + name)
        raise AttributeError(name)


def test_actual_idle_filter_and_issued_command_hook_share_one_correction(monkeypatch):
    filt, env = SafetyFilterV17(), OnboardOnly()
    raw = [.5, .02, .1]
    def sense(_):
        return cc.Snapshot(0., 0., 0., *raw, np.array([0., 1.]),
                           np.array([1., 0.]), np.zeros(2), 0., 0., 20.,
                           np.empty((0, 2)), [])
    filt.perception = SimpleNamespace(snapshot=sense)
    monkeypatch.setattr(filt, "_threat_in_reach", lambda snap: False)
    monkeypatch.setattr(original, "advance_ego", lambda *args: np.array([.4, .03, .7]))
    policy = np.array([.2, -.1], dtype=np.float32)
    out, changed = filt.filter(env, policy)
    np.testing.assert_array_equal(out, policy)
    assert not changed and filt._observer_snapshot.r == .1
    filt.observe_issued_command(.2, 12.)
    raw[:] = [.6, .04, -.2]
    filt.filter(env, policy)
    expected_uv = np.array([.4, .03]) + filt.observer.weight * (np.array(raw[:2])-[.4, .03])
    np.testing.assert_array_equal(filt.ego[:2], expected_uv)
    assert filt.ego[2] == -.2 and filt._observer_snapshot.r == -.2
    assert filt.last["yaw_observer"]["yaw_source"] == "fresh_measurement"
    filt.observe_issued_command(.2, 12.)
    env.pose_stale = True
    filt.filter(env, policy)
    np.testing.assert_array_equal(filt.ego, [.4, .03, .7])
    assert filt.last["yaw_observer"]["yaw_source"] == "model_or_held_prior"
