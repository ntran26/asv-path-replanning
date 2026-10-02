"""Hard-check parity and complete-sequence soft recovery scoring."""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_recovery as recovery
import safety_v2 as v2


def snapshot(points=(), tracks=()):
    boundary = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]])
    return SimpleNamespace(points=np.asarray(points, dtype=float).reshape(-1, 2), tracks=list(tracks),
                           edges_a=boundary, edges_b=np.roll(boundary, -1, axis=0))


@pytest.mark.parametrize("terminal_boundary", [False, True])
def test_hard_scores_match_v2_with_targets_and_terminal_checks(monkeypatch, terminal_boundary):
    monkeypatch.setattr(v2, "TERMINAL_BOUNDARY", terminal_boundary)
    positions = np.array([[[5.0, 3.0], [8.8, 10.0], [2.0, 4.0]],
                          [[5.0, 3.5], [9.0, 10.0], [2.0, 4.5]],
                          [[5.0, 4.0], [9.2, 10.0], [2.0, 5.0]],
                          [[5.0, 4.5], [9.4, 10.0], [2.0, 5.5]]])
    headings = np.zeros((4, 3))
    headings[:, 1] = np.pi / 2.0
    rollout = cc.Rollout(positions, headings, np.full((4, 3), 0.6), np.arange(1, 5) * .125)
    target = cc.TrackView(1, np.array([2.0, 7.0]), np.array([0.0, -0.3]), 0.0)
    snap = snapshot([(5.0, 6.0)], [target])
    expected_first, expected_clear = v2.SafetyFilterV2()._evaluate(snap, rollout)
    result = recovery.evaluate(snap, rollout)
    np.testing.assert_array_equal(result.first, expected_first)
    np.testing.assert_array_equal(result.clear, expected_clear)
    assert (result.violation >= 0.0).all()


def test_terminal_deficit_has_one_decision_weight_and_boundary_toggle(monkeypatch):
    monkeypatch.setattr(v2, "TERMINAL_BOUNDARY", False)
    snap = snapshot([(5.0, 7.0)])
    rollout = cc.Rollout(np.array([[[5.0, 4.0]], [[5.0, 4.5]]]), np.zeros((2, 1)),
                        np.ones((2, 1)), np.array([.25, .5]))
    result = recovery.evaluate(snap, rollout)
    assert result.immediate_violation[0] == 0.0
    assert result.clear[0] < 0.0
    assert result.violation[0] == pytest.approx(-result.clear[0] * cfg.UPDATE_RATE)


def test_signed_boundary_penalty_grows_far_outside_and_accepts_both_windings():
    snap = snapshot()
    positions = np.array([[[9.0, 10.0], [11.0, 10.0], [14.0, 10.0]]])
    headings = np.zeros((1, 3))
    margin = recovery.signed_boundary_clearance(positions, headings, snap.edges_a, snap.edges_b)
    assert margin[0, 0] > 0.0
    assert margin[0, 2] < margin[0, 1] < 0.0
    reverse = snap.edges_a[::-1]
    np.testing.assert_allclose(margin, recovery.signed_boundary_clearance(
        positions, headings, reverse, np.roll(reverse, -1, axis=0)))
    rollout = cc.Rollout(positions, headings, np.zeros((1, 3)), np.array([.5]))
    result = recovery.evaluate(snap, rollout)
    # Preserve legacy hard semantics separately from the corrected soft metric.
    assert result.clear[1] > 0.0 and result.clear[2] > 0.0
    assert result.violation[2] > result.violation[1] > 0.0


def test_static_boundary_and_each_target_deficits_are_added(monkeypatch):
    monkeypatch.setattr(cc, "point_clearance", lambda *args, **kwargs: np.array([[-.2], [-.4]]))
    monkeypatch.setattr(cc, "boundary_clearance", lambda *args: np.array([[-.1], [-.2]]))
    monkeypatch.setattr(recovery, "signed_boundary_clearance", lambda *args: np.array([[-.1], [-.2]]))
    monkeypatch.setattr(cc, "target_gap", lambda *args: np.array([[-.3], [-.1]]))
    rollout = cc.Rollout(np.zeros((2, 1, 2)), np.zeros((2, 1)), np.zeros((2, 1)), np.array([.25, .75]))
    snap = snapshot(tracks=[object(), object()])
    result = recovery.evaluate(snap, rollout)
    deficit = np.array([.2, .4]) + v2.GAP_STATIC_M + np.array([.1, .2]) + v2.GAP_BOUNDARY_M
    deficit += 2.0 * (np.array([.3, .1]) + v2.GAP_TARGET_M)
    assert result.violation[0] == pytest.approx(deficit @ [.25, .5])
    assert result.immediate_violation[0] == pytest.approx(deficit @ [.25, .25])


def test_choose_keeps_metrics_on_complete_sequences_and_uses_policy_distance_last():
    sequences = np.array([[[1.0, 0.0]], [[0.0, 0.0]], [[-1.0, 0.0]]])
    metrics = recovery.RecoveryMetrics(np.array([8.0, 6.0, 4.0]), np.array([-1., -.1, -.2]),
                                        np.array([.3, .2, .1]), np.array([0., 0., .1]))
    assert recovery.choose(sequences, metrics, np.zeros(2)) == 2
    assert recovery.choose(sequences, metrics, np.zeros(2), immediate_first=True) == 1
    metrics.violation[:] = .1
    assert recovery.choose(sequences, metrics, np.zeros(2)) == 1
