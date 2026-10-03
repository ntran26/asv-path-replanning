"""V11 integration contracts with synthetic snapshots/checks, no episodes."""
from collections import deque
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v10 as v10
import safety_v11 as v11
import safety_policy_prefix as prefix


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
    f = v11.SafetyFilterV11(policy_prefix_search=True)
    f.actuators, f.observer = Commands(), None
    env = SimpleNamespace(command_rate_limit=False)
    snap = SimpleNamespace(position=np.zeros(2), tracks=[])
    plan = v3.plan_for([-.8, np.nan], [1., 0.])[:-1]
    calls = []

    def parent(self, env, action):
        self._observer_snapshot = snap
        self.mode, self.plan = "recovery", plan.copy()
        self.recovery_steps, self.uncertified_steps = 3, 2
        self.last = {"why": "last certificate", "changed": True, "brake": True,
                     "checked_clearance": -.5, "v10_policy_preserved": False}
        self.actuators.issue(env, float(OVERRIDE[0]))
        env._v2_brake = True
        if self.track_persistence is not None:
            self.track_persistence.last_track_persistence_stats = {"added_hypotheses": 1,
                                                                   "hypotheses": [{"id": -5}]}
        return OVERRIDE.copy(), True

    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", parent)

    def rollout(snapshot, actuators, sequences):
        assert snapshot is snap and actuators.values == [.1]
        calls.append(sequences.copy())
        return sequences

    monkeypatch.setattr(f, "_rollout", rollout)
    monkeypatch.setattr(f, "_evaluate", lambda s, p: (
        np.full(len(p), np.inf), np.full(len(p), .2)))
    return SimpleNamespace(f=f, env=env, snap=snap, plan=plan, calls=calls)


def assert_parent(case, result):
    out, changed = result
    np.testing.assert_array_equal(out, OVERRIDE)
    assert changed and case.env._v2_brake and case.f.mode == "recovery"
    assert case.f.recovery_steps == 3 and case.f.uncertified_steps == 2
    assert case.f.last["why"] == "last certificate"
    assert case.f.last["checked_clearance"] == -.5
    assert case.f.actuators.values == [.1, float(OVERRIDE[0])]
    np.testing.assert_array_equal(case.f.plan, case.plan)


def test_defaults_wrap_v10_perception_with_independent_admission_and_persistence():
    f = v11.SafetyFilterV11()
    assert not f.policy_prefix_search
    assert f.perception is f.track_persistence
    assert f.track_persistence.enable_admission and f.track_persistence.enable_persistence
    assert f.track_persistence.max_coast_s == 8.
    assert f._prefix_rng is not f._search_rng


def test_constructor_allows_independent_switches_and_preserves_v10_options():
    admission = v11.SafetyFilterV11(enable_persistence=False, prefer_policy_margin=False)
    assert admission.track_persistence.enable_admission
    assert not admission.track_persistence.enable_persistence and not admission.prefer_policy_margin
    persistence = v11.SafetyFilterV11(enable_admission=False)
    assert not persistence.track_persistence.enable_admission
    assert persistence.track_persistence.enable_persistence
    neither = v11.SafetyFilterV11(enable_admission=False, enable_persistence=False)
    assert neither.track_persistence is None
    assert not isinstance(neither.perception, v11.TrackPersistencePerception)


def test_disabled_search_keeps_parent_action_and_deepcopies_track_diagnostics(case):
    case.f.policy_prefix_search = False
    assert_parent(case, case.f.filter(case.env, POLICY))
    assert not case.calls and not case.f.last["v11_policy_prefix_search_checked"]
    stats = case.f.last["track_persistence"]
    assert stats["added_hypotheses"] == 1
    case.f.track_persistence.last_track_persistence_stats["hypotheses"][0]["id"] = -6
    assert stats["hypotheses"][0]["id"] == -5


def test_all_disabled_keeps_parent_control_and_has_no_track_wrapper(case):
    case.f = v11.SafetyFilterV11(enable_admission=False, enable_persistence=False,
                                policy_prefix_search=False)
    case.f.actuators, case.f.observer = Commands(), None
    assert_parent(case, case.f.filter(case.env, POLICY))
    assert case.f.track_persistence is None
    assert case.f.last["track_persistence"] == {}
    assert not case.f.last["v11_policy_prefix_search_checked"]


def test_successful_real_prefix_utility_restores_policy_history_and_retains_exact_plan(case):
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not case.env._v2_brake and case.f.mode == "nominal"
    assert case.f.recovery_steps == case.f.uncertified_steps == 0
    assert case.f.actuators.values == [.1, float(POLICY[0])]
    assert sum(map(len, case.calls)) == prefix.MAX_PLANS
    assert any(np.array_equal(case.f.plan, plan, equal_nan=True)
               for batch in case.calls for plan in batch)
    np.testing.assert_array_equal(case.f.plan[0], POLICY)
    assert case.f.last["why"] == "policy prefix searched"
    assert case.f.last["v11_policy_prefix_preserved"]
    assert not case.f.last["v10_policy_preserved"]
    assert case.f.last["checked_clearance"] == .2


@pytest.mark.parametrize("first,clear", [(.5, -.1), (.5, .8), (np.inf, .149), (np.nan, .8)])
def test_failed_prefix_preserves_parent_plan_and_fallback_counters(case, monkeypatch, first, clear):
    monkeypatch.setattr(case.f, "_evaluate", lambda s, p: (
        np.full(len(p), first), np.full(len(p), clear)))
    assert_parent(case, case.f.filter(case.env, POLICY))
    assert case.f.last["v11_policy_prefix_search_checked"]
    assert not case.f.last["v11_policy_prefix_preserved"]


@pytest.mark.parametrize("kind", ["rate_limit", "no_snapshot", "already_policy"])
def test_search_skips_unsupported_or_unnecessary_paths(case, monkeypatch, kind):
    parent = v10.SafetyFilterV10._filter

    def adjusted(self, env, action):
        out, changed = parent(self, env, action)
        if kind == "no_snapshot":
            self._observer_snapshot = None
        if kind == "already_policy":
            self.last["why"] = "policy margin dominates"
            return POLICY.copy(), False
        return out, changed

    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", adjusted)
    case.env.command_rate_limit = kind == "rate_limit"
    out, changed = case.f.filter(case.env, POLICY)
    assert changed == (kind != "already_policy")
    assert not case.calls and not case.f.last["v11_policy_prefix_search_checked"]


@pytest.mark.parametrize("passing", [False, True])
def test_no_escape_can_find_backup_without_changing_current_action(case, monkeypatch, passing):
    def parent(self, env, action):
        self._observer_snapshot = case.snap
        self.mode, self.plan = "nominal", None
        self.recovery_steps = self.uncertified_steps = 0
        self.last = {"why": "no escape", "checked_clearance": None}
        self.actuators.issue(env, float(action[0]))
        env._v2_brake = False
        return action.copy(), False

    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", parent)
    monkeypatch.setattr(case.f, "_evaluate", lambda s, p: (
        np.full(len(p), np.inf if passing else .5), np.full(len(p), .2 if passing else -.1)))
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not case.env._v2_brake
    assert case.f.actuators.values == [.1, float(POLICY[0])]
    assert (case.f.plan is not None) == passing
    assert case.f.last["why"] == ("policy prefix searched" if passing else "no escape")


def test_dual_braking_wraps_every_prefix_batch_and_cannot_use_weak_check_alone(case, monkeypatch):
    import safety_prediction
    monkeypatch.setattr(v4, "DUAL_BRAKE_PREDICTION", True)
    dual_calls = []

    def dual(snap, act, sequences, evaluate):
        assert snap is case.snap and act.values == [.1]
        assert evaluate == case.f._evaluate
        dual_calls.append(sequences.copy())
        return SimpleNamespace(first=np.full(len(sequences), .5), clear=np.full(len(sequences), -.1))

    monkeypatch.setattr(safety_prediction, "evaluate_sequences", dual)
    assert_parent(case, case.f.filter(case.env, POLICY))
    assert len(dual_calls) == 4 and not case.calls
    assert sum(map(len, dual_calls)) == prefix.MAX_PLANS


def test_traffic_gate_uses_inherited_range_and_future_brake_stays_in_saved_tail(case, monkeypatch):
    case.snap.tracks = [SimpleNamespace(position=np.array([v2.ENGAGE_RANGE_M - .1, 0.]))]

    def evaluate(snap, plans):
        brake = np.isnan(plans[:, 1:, 1]).any(axis=1)
        return np.where(brake, np.inf, .5), np.where(brake, .3, -.1)

    monkeypatch.setattr(case.f, "_evaluate", evaluate)
    out, changed = case.f.filter(case.env, POLICY)
    assert not changed and not case.env._v2_brake
    assert np.isnan(case.f.plan[1:, 1]).any()
    assert np.isfinite(case.f.plan[0]).all()
    np.testing.assert_array_equal(out, POLICY)


def test_observer_prediction_uses_original_history_and_actual_dispatch(case):
    calls = []
    case.f.observer = SimpleNamespace(predict=lambda *args: calls.append(args))
    case.f.filter(case.env, POLICY)
    case.f.observe_issued_command(float(POLICY[0]), 4.8)
    assert len(calls) == 1 and calls[0][0] is case.snap
    assert calls[0][1].values == [.1]
    assert calls[0][2:] == (float(POLICY[0]), 4.8)


def test_prefix_rng_is_deterministic_and_does_not_advance_parent_or_global_rng(case, monkeypatch):
    global_before = copy.deepcopy(np.random.get_state())
    parent_before = copy.deepcopy(case.f._search_rng.bit_generator.state)
    prefix_before = copy.deepcopy(case.f._prefix_rng.bit_generator.state)
    case.f.filter(case.env, POLICY)
    assert case.f._search_rng.bit_generator.state == parent_before
    assert case.f._prefix_rng.bit_generator.state != prefix_before
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0] and global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(global_before[1], global_after[1])


def test_real_wrapper_adds_motion_and_persists_through_raw_id_loss_before_parent_selection(monkeypatch):
    base = SimpleNamespace(calls=0)
    original = cc.Snapshot(0., 0., 0., .5, 0., 0., np.array([0., 1.]),
                           np.array([1., 0.]), np.zeros(2), 0., 0., 20.,
                           np.array([[9., 12.]]), [])

    def snapshot(env):
        base.calls += 1
        return original

    base.snapshot = snapshot

    def initialize(self, **kwargs):
        self.perception, self.actuators, self.observer = base, Commands(), None

    seen = []

    def parent(self, env, action):
        snap = self.perception.snapshot(env)
        seen.append(snap)
        self._observer_snapshot = snap
        self.last = {"why": "nominal"}
        self.actuators.issue(env, float(action[0]))
        return action.copy(), False

    monkeypatch.setattr(v10.SafetyFilterV10, "__init__", initialize)
    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", parent)
    f = v11.SafetyFilterV11()
    track = SimpleNamespace(id=7, hits=cfg.TRACK_MIN_HITS, misses=0,
                            velocity=np.array([.4, 0.]), last_fit_heading_deg=90.,
                            last_evidence=None, is_dynamic=False, history=deque())
    env = SimpleNamespace(tracker=SimpleNamespace(tracks=[track]), pose_stale=False)
    axis = np.linspace(-cfg.LOA / 2., cfg.LOA / 2., 10)
    points = np.concatenate([np.column_stack((axis, np.full(10, side * cfg.BREADTH / 2.)))
                             for side in (-1, 1)])
    for serial in range(1, 4):
        track.position = track.last_fit_centre = np.array([4. + .2 * (serial - 1), 8.])
        track.history.append((serial, points + track.position))
        f.filter(env, POLICY)
    assert len(seen[0].tracks) == len(seen[1].tracks) == 0
    assert len(seen[2].tracks) == 1 and seen[2].tracks[0].id < 0
    assert f.last["track_persistence"]["added_hypotheses"] == 1
    env.tracker.tracks = []
    f.filter(env, POLICY)
    np.testing.assert_allclose(seen[-1].tracks[0].position, [4.6, 8.])
    assert f.last["track_persistence"]["hypotheses"][0]["age_s"] == cfg.UPDATE_RATE
    assert base.calls == 4 and not original.tracks and not track.is_dynamic
    assert all(snap.points is original.points for snap in seen)


@pytest.mark.parametrize("coast", [-1., np.nan, np.inf])
def test_invalid_coast_setting_rejected(coast):
    with pytest.raises(ValueError, match="max_coast_s"):
        v11.SafetyFilterV11(max_coast_s=coast)
