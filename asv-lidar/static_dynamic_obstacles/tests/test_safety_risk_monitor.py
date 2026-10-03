"""Synthetic observable-monitor checks; no environment or policy episodes."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_risk_monitor as monitor
import safety_v2 as v2
import safety_v3 as v3


@pytest.fixture(autouse=True)
def decision_times(monkeypatch):
    monkeypatch.setattr(cfg, "UPDATE_RATE", .5)
    monkeypatch.setattr(v2, "COMMIT_S", 1.)


def snapshot(*, x=0., y=0., points=None, tracks=(), edges=None):
    return cc.Snapshot(x, y, 0., .7, .0, .0, np.array([0., 1.]),
        np.array([1., 0.]), np.array([0., 0.]), 0., x, 20.,
        np.empty((0, 2)) if points is None else np.asarray(points, dtype=float),
        list(tracks), None if edges is None else np.asarray(edges[0], dtype=float),
        None if edges is None else np.asarray(edges[1], dtype=float))


class PathRollout:
    def __init__(self, offsets=None, times=None, headings=None):
        self.offsets = np.asarray(offsets if offsets is not None else
                                  [[0., .15], [0., .3], [0., .45], [0., .7]])
        self.times = np.asarray(times if times is not None else [.25, .5, .75, 1.])
        self.headings = np.asarray(headings if headings is not None else np.zeros(len(self.times)))
        self.calls = []

    def __call__(self, snap, act, sequences):
        self.calls.append((snap, act, sequences.copy()))
        return cc.Rollout((snap.position + self.offsets)[:, None, :],
                          (snap.heading + self.headings)[:, None],
                          np.full((len(self.times), 1), snap.u), self.times.copy())


def static_ahead(gap=.6):
    return [[0., cc.HALF_L + v2.GAP_STATIC_M + gap]]


def run(subject, snap, predictor=None, fresh=True):
    return subject.update(snap, np.array([.3, -.2]), SimpleNamespace(buffer=[.7]),
                          predictor or PathRollout(), fresh=fresh)


def test_shadow_records_threat_then_persistence_without_modifying_any_action():
    subject, predictor = monitor.ObservableRiskMonitor(), PathRollout()
    snap = snapshot(points=static_ahead())
    action = np.array([.3, -.2])
    act = SimpleNamespace(buffer=[.7], servo=.2)
    saved = copy.deepcopy((snap, action, act))
    first = subject.update(snap, action, act, predictor)
    assert first.shadow_only and monitor.SHADOW_ONLY
    assert not first.urgent and not first.persistent and not first.recommend_rescue
    hazard = first.hazards[0]
    assert hazard.kind == "static_memory"
    assert hazard.current_clearance_m == pytest.approx(.6)
    assert hazard.minimum_predicted_clearance_m == pytest.approx(-.1)
    assert hazard.first_violation_s == 1.
    assert hazard.measured_closing_rate_mps is None
    assert hazard.consecutive_fresh_threats == 1
    second = subject.update(snapshot(y=.1, points=static_ahead()), action, act, predictor)
    assert second.persistent and second.recommend_rescue
    assert second.hazards[0].measured_closing_rate_mps == pytest.approx(.2)
    np.testing.assert_array_equal(action, saved[1])
    np.testing.assert_array_equal(snap.points, saved[0].points)
    assert snap.y == saved[0].y and act.__dict__ == saved[2].__dict__
    np.testing.assert_array_equal(predictor.calls[0][2], [[[.3, -.2], [.3, -.2]]])
    assert predictor.calls[0][0] is snap and predictor.calls[0][1] is act


def test_next_decision_violation_is_urgent_without_waiting_for_persistence():
    evidence = run(monitor.ObservableRiskMonitor(), snapshot(points=static_ahead(.2)))
    assert evidence.urgent and evidence.recommend_rescue and not evidence.persistent
    assert evidence.hazards[0].first_violation_s == .5


def test_already_overlapping_but_opening_is_still_reported_urgent():
    away = PathRollout([[0., -.1], [0., -.2], [0., -.3], [0., -.4]])
    evidence = run(monitor.ObservableRiskMonitor(), snapshot(points=static_ahead(-.05)), away)
    assert evidence.urgent and evidence.hazards[0].first_violation_s == 0.


def test_safe_policy_recovery_and_clearance_opening_do_not_request_rescue():
    away = PathRollout([[0., -.1], [0., -.2], [0., -.3], [0., -.4]])
    subject = monitor.ObservableRiskMonitor()
    for _ in range(3):
        evidence = run(subject, snapshot(points=static_ahead(.01)), away)
        assert not evidence.recommend_rescue
        assert not evidence.hazards[0].predicted_closing
        assert evidence.hazards[0].consecutive_fresh_threats == 0


def test_safe_fresh_frame_breaks_threat_streak():
    subject = monitor.ObservableRiskMonitor()
    run(subject, snapshot(points=static_ahead()))
    run(subject, snapshot(points=static_ahead(5.)))
    evidence = run(subject, snapshot(points=static_ahead()))
    assert evidence.hazards[0].consecutive_fresh_threats == 1
    assert not evidence.persistent


def test_stale_frames_never_confirm_persistence_and_unknown_is_not_safe():
    subject = monitor.ObservableRiskMonitor()
    snap = snapshot(points=static_ahead())
    run(subject, snap)
    for _ in range(3):
        evidence = run(subject, snap, fresh=False)
        assert not evidence.fresh and not evidence.persistent and not evidence.recommend_rescue
        assert evidence.hazards[0].measured_closing_rate_mps is None
    resumed = run(subject, snap)
    assert resumed.hazards[0].consecutive_fresh_threats == 1
    assert resumed.hazards[0].measured_closing_rate_mps is None
    stale_urgent = run(subject, snapshot(points=static_ahead(.2)), fresh=False)
    assert stale_urgent.urgent and not stale_urgent.recommend_rescue


def test_missing_channel_and_reset_clear_history():
    subject = monitor.ObservableRiskMonitor()
    snap = snapshot(points=static_ahead())
    run(subject, snap)
    assert not run(subject, snapshot()).hazards
    assert not run(subject, snap).persistent
    subject.reset()
    assert not run(subject, snap).persistent


def test_separate_target_id_cannot_borrow_another_targets_streak():
    separation = 2 * cc.HALF_L + v2.GAP_TARGET_M + .6
    def target(identifier):
        return cc.TrackView(identifier, np.array([0., separation]), np.array([0., 0.]), 0.)
    subject = monitor.ObservableRiskMonitor()
    one = run(subject, snapshot(tracks=[target(1)]))
    two = run(subject, snapshot(tracks=[target(2)]))
    assert one.hazards[0].threatening and two.hazards[0].threatening
    assert two.hazards[0].key == "track:2" and not two.persistent
    assert run(subject, snapshot(tracks=[target(2)])).persistent


def test_curved_prediction_checks_intermediate_samples_and_delayed_actuators():
    wall_x = cc.HALF_W + v2.GAP_BOUNDARY_M + .4
    snap = snapshot(edges=([[wall_x, -10.]], [[wall_x, 10.]]))
    snap.v, snap.r = .2, .1
    curve = PathRollout([[.1, .1], [.5, .2], [.1, .3], [0., .4]])
    evidence = run(monitor.ObservableRiskMonitor(), snap, curve)
    assert evidence.urgent and evidence.hazards[0].key == "boundary:0"
    assert evidence.hazards[0].first_violation_s == .5
    assert evidence.world_velocity_mps == (.2, .7)
    assert evidence.yaw_rate_rad_s == .1
    assert curve.calls[0][1].buffer == [.7]


def test_far_returns_are_finite_and_do_not_trigger_a_prefilter_infinity_error():
    evidence = run(monitor.ObservableRiskMonitor(), snapshot(points=[[100., 100.]]))
    assert not evidence.recommend_rescue
    assert evidence.hazards[0].current_clearance_m > 100.
    json.dumps(evidence.as_dict(), allow_nan=False)


def test_only_existing_commit_horizon_is_inspected_even_if_callback_returns_more():
    predictor = PathRollout([[0., .1], [0., .2], [0., .3], [0., .4], [0., 2.]],
                            [.25, .5, .75, 1., 2.])
    evidence = run(monitor.ObservableRiskMonitor(), snapshot(points=static_ahead()), predictor)
    assert not evidence.recommend_rescue and evidence.horizon_s == v2.COMMIT_S


def test_real_onboard_rollout_is_read_only_and_serializes_without_truth():
    snap = snapshot(points=[[100., 100.]])
    act = cc.Actuators()
    act.servo, act.buffer = .04, [.02] * cc.DELAY_STEPS
    before = copy.deepcopy((snap, act))
    evidence = monitor.ObservableRiskMonitor().update(snap, [.2, 0.], act, v3.rollout_seq)
    assert evidence.horizon_s == 1. and not evidence.recommend_rescue
    assert act.servo == before[1].servo and act.buffer == before[1].buffer
    np.testing.assert_array_equal(snap.points, before[0].points)
    assert json.loads(json.dumps(evidence.as_dict(), allow_nan=False))["shadow_only"] is True


@pytest.mark.parametrize("action", [[np.nan, 0.], [0., np.inf], [1.1, 0.], [0.]])
def test_invalid_policy_action_rejected_without_callback(action):
    predictor = PathRollout()
    with pytest.raises(ValueError, match="Policy action"):
        monitor.ObservableRiskMonitor().update(snapshot(), action, None, predictor)
    assert not predictor.calls


@pytest.mark.parametrize("times", [[.25, .5], [.25, .25, .75, 1.], [0., .5, .75, 1.],
                                  [.25, .5, .75, 1.25]])
def test_incomplete_or_repeated_prediction_time_rejected(times):
    predictor = PathRollout(np.zeros((len(times), 2)), times)
    with pytest.raises(ValueError, match="Rollout times"):
        run(monitor.ObservableRiskMonitor(), snapshot(), predictor)
