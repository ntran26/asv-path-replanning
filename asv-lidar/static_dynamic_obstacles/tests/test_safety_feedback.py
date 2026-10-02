"""Feedback-bank model equivalence and steering checks, without episodes."""
import copy
import math

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_feedback as feedback
import safety_v2 as v2
import safety_v3 as v3


def snapshot():
    boundary = np.array([[0., 0.], [10., 0.], [10., 30.], [0., 30.]])
    return cc.Snapshot(1.0, 10.0, 0.2, 0.6, -0.02, 0.08,
                       np.array([0., 1.]), np.array([1., 0.]), np.array([5., 10.]),
                       0., -4., 20., np.empty((0, 2)), edges_a=boundary,
                       edges_b=np.roll(boundary, -1, axis=0))


@pytest.mark.parametrize("initialized", [False, True])
def test_recorded_feedback_sequence_exactly_replays_v3_dynamics(initialized):
    snap, act = snapshot(), cc.Actuators()
    if initialized:
        act.buffer = np.linspace(-0.2, 0.1, cc.DELAY_STEPS).tolist()
        act.servo, act.executed = 0.07, -0.4
    candidates = np.array([[0.4, 0.2], [-1.0, -1.0], [0.0, np.nan]])
    rollout, sequences, names = feedback.feedback_bank(snap, act, candidates, commit_s=2.0)
    assert sequences.shape[:2] == (3, 4)
    expected = v3.rollout_seq(snap, act, sequences.reshape(-1, sequences.shape[2], 2))
    np.testing.assert_allclose(rollout.positions, expected.positions, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rollout.headings, expected.headings, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rollout.speeds, expected.speeds, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(rollout.times, expected.times)
    assert names == feedback.COURSE_NAMES
    commit = int(round(2.0 / cfg.UPDATE_RATE))
    for i, candidate in enumerate(candidates):
        np.testing.assert_allclose(sequences[i, :, :commit],
                                   np.broadcast_to(candidate, (4, commit, 2)), equal_nan=True)
    np.testing.assert_array_equal(sequences[:, :, commit:, 1], 0.0)


def test_edge_heading_is_parallel_and_oriented_along_path_forward_direction():
    snap = snapshot()
    # The nearest (left) edge is stored southbound; desired course must be north.
    courses = feedback.desired_courses(snap)
    assert courses[0] == pytest.approx(0.0)
    np.testing.assert_allclose(courses[1:], [snap.heading, snap.heading - math.pi / 6,
                                            snap.heading + math.pi / 6])
    snap.base_heading = math.pi
    assert abs(feedback.desired_courses(snap)[0]) == pytest.approx(math.pi)


def test_nearest_edge_distance_uses_segment_interior_not_vertices():
    snap = snapshot()
    snap.edges_a = np.array([[0., -100.], [2., 12.]])
    snap.edges_b = np.array([[0., 100.], [3., 12.]])
    # Long vertical edge has distant endpoints but is closer than short panel.
    assert feedback.desired_courses(snap)[0] == pytest.approx(0.0)


def test_missing_or_degenerate_edges_fall_back_to_forward_course():
    snap = snapshot()
    snap.base_heading = 0.7
    snap.edges_a = snap.edges_b = None
    assert feedback.desired_courses(snap)[0] == pytest.approx(0.7)
    snap.edges_a = snap.edges_b = np.array([[1., 2.]])
    assert feedback.desired_courses(snap)[0] == pytest.approx(0.7)


def test_live_feedback_damps_yaw_even_on_current_heading(monkeypatch):
    snap, act = snapshot(), cc.Actuators()
    calls = []
    original = cc.course_rudder

    def capture(error, yaw):
        calls.append((np.asarray(error).copy(), np.asarray(yaw).copy()))
        return original(error, yaw)

    monkeypatch.setattr(cc, "course_rudder", capture)
    _, sequences, _ = feedback.feedback_bank(snap, act, np.array([[0.0, 0.0]]), commit_s=0.0)
    assert len(calls) == sequences.shape[2]
    assert calls[0][0][1] == pytest.approx(0.0)  # Hold present heading.
    assert sequences[0, 1, 0, 0] < 0.0        # Counter existing positive yaw.
    assert not np.allclose(calls[0][1], calls[-1][1])
    assert not np.allclose(sequences[0, 1, :, 0], sequences[0, 1, 0, 0])


def test_generation_does_not_modify_snapshot_actuators_or_candidates():
    snap, act = snapshot(), cc.Actuators()
    act.buffer = [0.1] * cc.DELAY_STEPS
    act.servo = -0.04
    candidates = np.array([[0.7, np.nan], [-0.3, 0.2]])
    before_snap, before_act = copy.deepcopy(vars(snap)), copy.deepcopy(vars(act))
    before_candidates = candidates.copy()
    feedback.feedback_bank(snap, act, candidates, commit_s=v2.COMMIT_S)
    for key, value in before_snap.items():
        np.testing.assert_equal(getattr(snap, key), value)
    assert vars(act) == before_act
    np.testing.assert_equal(candidates, before_candidates)


def test_empty_candidate_bank_has_consistent_shapes():
    rollout, sequences, names = feedback.feedback_bank(snapshot(), cc.Actuators(),
                                                       np.empty((0, 2)), commit_s=1.0)
    decisions = int(math.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))
    assert sequences.shape == (0, 4, decisions, 2)
    assert rollout.positions.shape == (decisions * cc.SUBSTEPS, 0, 2)
    assert rollout.headings.shape == rollout.speeds.shape == (decisions * cc.SUBSTEPS, 0)
    assert len(names) == 4
