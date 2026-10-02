"""Sideslip course compensation and model replay, without environment episodes."""
import copy

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_course_backup as backup
import safety_feedback
import safety_v2 as v2
import safety_v3 as v3


def snapshot():
    boundary = np.array([[0., 0.], [10., 0.], [10., 30.], [0., 30.]])
    return cc.Snapshot(8.5, 10.0, 0.2, 0.6, 0.04, 0.02,
                       np.array([0., 1.]), np.array([1., 0.]), np.array([5., 10.]),
                       0., 3.5, 20., np.empty((0, 2)), edges_a=boundary,
                       edges_b=np.roll(boundary, -1, axis=0))


def test_compensation_opposes_sway_and_vanishes_at_the_correct_course():
    state = np.zeros((7, 3))
    state[0] = 0.4
    state[1] = [0.02, -0.02, 0.0]
    target = np.zeros(3)
    command = backup._sideslip_rudder(target, state)
    assert command[0] < 0.0 < command[1]
    assert command[0] == pytest.approx(-command[1])
    assert command[2] == pytest.approx(0.0)
    # Pointing into the drift cancels cross-wall velocity in the actual model
    # convention, and zero course/yaw error requests no further turning.
    state[3] = -np.arctan2(state[1], state[0])
    cross_wall_velocity = state[0] * np.sin(state[3]) + state[1] * np.cos(state[3])
    np.testing.assert_allclose(cross_wall_velocity, 0.0, atol=1e-15)
    np.testing.assert_allclose(backup._sideslip_rudder(target, state), 0.0, atol=1e-15)


def test_zero_velocity_including_signed_zeros_has_zero_sideslip():
    state = np.zeros((7, 4))
    state[0], state[1] = [0.0, -0.0, 0.0, -0.0], [0.0, 0.0, -0.0, -0.0]
    state[3], state[2] = 0.1, 0.02
    target = np.full(4, 0.2)
    command = backup._sideslip_rudder(target, state)
    assert np.isfinite(command).all()
    np.testing.assert_allclose(command, cc.course_rudder(target - state[3], state[2]),
                               rtol=0, atol=1e-15)


@pytest.mark.parametrize("initialized", [False, True])
def test_extra_backup_replays_exactly_and_preserves_candidate_prefix(initialized):
    snap, act = snapshot(), cc.Actuators()
    if initialized:
        act.buffer = np.linspace(-0.2, 0.1, cc.DELAY_STEPS).tolist()
        act.servo, act.executed = 0.07, -0.4
    candidates = np.array([[0.4, 0.2], [-1.0, -1.0], [0.0, np.nan]])
    rollout, sequences, names = backup.feedback_bank(snap, act, candidates, commit_s=1.0)
    assert sequences.shape[:2] == (3, 1)
    assert names == ("edge_parallel_sideslip",)
    expected = v3.rollout_seq(snap, act, sequences.reshape(-1, sequences.shape[2], 2))
    for field in ("positions", "headings", "speeds", "times"):
        np.testing.assert_allclose(getattr(rollout, field), getattr(expected, field), rtol=0, atol=1e-12)
    commit = int(round(1.0 / cfg.UPDATE_RATE))
    np.testing.assert_allclose(sequences[:, 0, :commit],
                               np.repeat(candidates[:, None, :], commit, axis=1), equal_nan=True)
    np.testing.assert_array_equal(sequences[:, :, commit:, 1], 0.0)


def test_predicted_sideslip_is_recomputed_instead_of_frozen(monkeypatch):
    snap, act = snapshot(), cc.Actuators()
    snap.u, snap.v = 0.03, 0.08
    seen = []
    original = backup._sideslip_rudder

    def capture(target, state):
        seen.append((target.copy(), state.copy()))
        return original(target, state)

    monkeypatch.setattr(backup, "_sideslip_rudder", capture)
    _, sequences, _ = backup.feedback_bank(snap, act, np.array([[0., 0.]]), commit_s=0.0)
    assert len(seen) == sequences.shape[2]
    beta = [float(np.arctan2(state[1, 0], state[0, 0])) for _, state in seen]
    assert not np.allclose(beta, beta[0])
    parallel = safety_feedback.desired_courses(snap)[0]
    for decision, (target, state) in enumerate(seen):
        assert target[0] == pytest.approx(parallel)
        np.testing.assert_array_equal(sequences[:, 0, decision, 0], original(target, state))


def test_extra_backup_does_not_mutate_inputs():
    snap, act = snapshot(), cc.Actuators()
    act.buffer = [0.1] * cc.DELAY_STEPS
    act.servo = -0.04
    candidates = np.array([[0.7, np.nan], [-0.3, 0.2]])
    before_snap, before_act = copy.deepcopy(vars(snap)), copy.deepcopy(vars(act))
    before_candidates = candidates.copy()
    backup.feedback_bank(snap, act, candidates, commit_s=v2.COMMIT_S)
    for key, value in before_snap.items():
        np.testing.assert_equal(getattr(snap, key), value)
    assert vars(act) == before_act
    np.testing.assert_equal(candidates, before_candidates)


def test_empty_extra_bank_keeps_single_course_axis():
    rollout, sequences, names = backup.feedback_bank(snapshot(), cc.Actuators(),
                                                      np.empty((0, 2)), commit_s=1.0)
    assert sequences.shape == (0, 1, v3._decisions()[0], 2)
    assert rollout.positions.shape == (v3._decisions()[0] * cc.SUBSTEPS, 0, 2)
    assert names == backup.COURSE_NAMES
