"""Synthetic target-model checks; no environment, reset or policy episode."""
import copy
from dataclasses import FrozenInstanceError
import math
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
import safety_target_prediction as prediction
import safety_v2 as v2


def track(identifier=1, position=(3., 4.), velocity=(.2, -.4), heading=.4):
    return cc.TrackView(identifier, np.array(position, dtype=float), np.array(velocity, dtype=float), heading)


def rollout(times=(.1, .5, 1., 2.), n=3):
    times = np.asarray(times)
    positions = np.zeros((len(times), n, 2))
    positions[:, :, 0] = np.arange(n)
    positions[:, :, 1] = times[:, None] * .3
    return cc.Rollout(positions, np.zeros((len(times), n)), np.full((len(times), n), .3), times)


def envelope(tracks=(), rate=.1, **kwargs):
    return prediction.TargetPredictionEnvelope.from_snapshot(SimpleNamespace(tracks=list(tracks)),
                                                              turn_rate_rad_s=rate, **kwargs)


def test_zero_rate_exactly_matches_existing_target_gap_for_multiple_tracks():
    targets, ro = [track(), track(2, (-2., 1.))], rollout()
    subject = envelope(targets, rate=0.)
    expected = np.minimum.reduce([cc.target_gap(ro.positions, ro.headings, ro.times, t) - v2.GAP_TARGET_M
                                  for t in targets])
    np.testing.assert_array_equal(subject.clearance(ro), expected)
    assert subject.targets_count == 2 and subject.hypotheses_count == 2
    first, clear = subject.evaluate(ro)
    bad = expected < 0.
    np.testing.assert_array_equal(first, np.where(bad.any(axis=0), ro.times[bad.argmax(axis=0)], np.inf))
    np.testing.assert_array_equal(clear, expected.min(axis=0))


def test_empty_targets_are_neutral_to_all_existing_checks():
    subject, ro = envelope(), rollout()
    first, clear = subject.evaluate(ro)
    assert first.shape == clear.shape == (3,)
    assert np.isposinf(first).all() and np.isposinf(clear).all()
    assert subject.targets_count == subject.hypotheses_count == 0


def test_clockwise_and_counterclockwise_quarter_circle_compass_convention():
    times = [0., 1.]
    right, headings = prediction.constant_turn([0., 0.], [0., 2.], .3, times, math.pi / 2)
    left, left_headings = prediction.constant_turn([0., 0.], [0., 2.], .3, times, -math.pi / 2)
    np.testing.assert_allclose(right[-1], [4/math.pi, 4/math.pi])
    np.testing.assert_allclose(left[-1], [-4/math.pi, 4/math.pi])
    assert headings[-1] == pytest.approx(.3 + math.pi / 2)
    assert left_headings[-1] == pytest.approx(.3 - math.pi / 2)
    np.testing.assert_array_equal(right[0], [0., 0.])


def test_constant_turn_velocity_preserves_speed_and_rotates_with_hull():
    times = np.array([2., 2.000001])
    points, headings = prediction.constant_turn([2., -1.], [.3, .4], .7, times, .2)
    derivative = np.diff(points, axis=0)[0] / np.diff(times)[0]
    expected = [.3 * math.cos(.4) + .4 * math.sin(.4), .4 * math.cos(.4) - .3 * math.sin(.4)]
    np.testing.assert_allclose(derivative, expected, atol=1e-7)
    assert np.linalg.norm(derivative) == pytest.approx(.5, abs=1e-8)
    assert headings[0] == pytest.approx(1.1)


def test_near_zero_turn_uses_stable_cv_limit_without_cancellation():
    times = [.1, 1., 8.]
    expected, heading = prediction.constant_turn([2., -1.], [.3, .4], .7, times, 0.)
    for rate in (-1e-12, 1e-12):
        actual, _ = prediction.constant_turn([2., -1.], [.3, .4], .7, times, rate)
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0.)
    np.testing.assert_array_equal(heading, [.7] * 3)


def test_varying_heading_sat_matches_existing_scalar_formula_at_every_time():
    ro = rollout(n=4)
    target_positions = np.array([[3., 4.], [1., 0.], [-2., 4.], [-1., -2.]])
    target_headings = np.array([0., .3, math.pi/2, -2.])
    ro.headings[:] = np.arange(16).reshape(4, 4) * .2
    result = prediction.turning_hull_gap(ro.positions, ro.headings, target_positions, target_headings)
    for index in range(len(ro.times)):
        expected = cc.hull_separation(ro.positions[index], ro.headings[index], target_positions[index],
                                     target_headings[index], margin=cc.HULL_MARGIN)
        np.testing.assert_allclose(result[index], expected, atol=2e-15)


def test_turning_hypothesis_can_reject_a_cv_safe_rollout():
    t = track(position=(-4., 0.), velocity=(0., 1.), heading=0.)
    times = np.array([4., 8.])
    target, headings = prediction.constant_turn(t.position, t.velocity, t.heading, times, math.pi/8)
    ro = cc.Rollout(target[:, None].copy(), headings[:, None].copy(), np.zeros((2, 1)), times)
    assert np.isinf(envelope([t], rate=0.).evaluate(ro)[0][0])
    first, clearance = envelope([t], rate=math.pi/8).evaluate(ro)
    assert first[0] == 4. and clearance[0] < 0.


def test_ensemble_never_relaxes_cv_clearance():
    targets, ro = [track(), track(2, (-2., 1.))], rollout()
    cv = envelope(targets, rate=0.).clearance(ro)
    enriched = envelope(targets, rate=.2).clearance(ro)
    assert np.all(enriched <= cv)


def test_stationary_centre_has_no_fabricated_translation():
    points, headings = prediction.constant_turn([2., 3.], [0., 0.], 0., [0., 1., 3.], .2)
    np.testing.assert_array_equal(points, [[2., 3.]] * 3)
    np.testing.assert_allclose(headings, [0., .2, .6])


def test_envelope_is_immutable_and_independent_of_later_track_changes():
    original = track()
    subject, ro = envelope([original]), rollout()
    expected = subject.clearance(ro)
    original.position[:] = 99.
    original.velocity[:] = -99.
    original.heading = 9.
    np.testing.assert_array_equal(subject.clearance(ro), expected)
    np.testing.assert_array_equal(copy.deepcopy(subject).clearance(ro), expected)
    with pytest.raises(FrozenInstanceError):
        subject.turn_rate_rad_s = .9
    assert isinstance(subject._targets[0].position, tuple)


def test_optional_measured_rate_is_clipped_deduplicated_and_bounded():
    subject = envelope([track()], rate=.2, measured_turn_rates={1: .1})
    assert subject.hypotheses_count == 4
    assert subject._targets[0].turn_rates == (0., -.2, .2, .1)
    assert envelope([track()], rate=.2, measured_turn_rates={1: 9.}).hypotheses_count == 3
    assert envelope([track()], rate=0., measured_turn_rates={1: 9.}).hypotheses_count == 1


def test_only_onboard_track_fields_are_read():
    class OnboardOnly:
        tracks = [track()]
        def __getattr__(self, name):
            raise AssertionError(f"Unexpected environment/truth/context access: {name}")
    OnboardOnly.tracks[0].ctx = object()
    subject = prediction.TargetPredictionEnvelope.from_snapshot(OnboardOnly(), turn_rate_rad_s=.1)
    subject.evaluate(rollout())


@pytest.mark.parametrize("rate", [-.1, math.inf, math.nan])
def test_invalid_turn_bound_rejected(rate):
    with pytest.raises(ValueError, match="Turn-rate"):
        envelope([track()], rate=rate)


@pytest.mark.parametrize("times", [[.5, .5], [-.1, .5], [math.nan], []])
def test_invalid_prediction_times_rejected(times):
    with pytest.raises(ValueError, match="Prediction times"):
        envelope().evaluate(rollout(times=times))


def test_invalid_estimate_or_duplicate_association_is_not_silently_ignored():
    with pytest.raises(ValueError, match="Track velocity"):
        envelope([track(velocity=(math.nan, 0.))])
    with pytest.raises(ValueError, match="Duplicate"):
        envelope([track(), track()])
    with pytest.raises(ValueError, match="measured turn"):
        envelope([track()], measured_turn_rates={1: math.inf})


def test_intermediate_collision_is_not_replaced_by_final_clearance():
    t = track(position=(0., 0.), velocity=(0., 0.), heading=0.)
    ro = cc.Rollout(np.array([[[0., 0.]], [[20., 20.]]]), np.zeros((2, 1)), np.zeros((2, 1)),
                    np.array([.5, 1.]))
    first, clearance = envelope([t]).evaluate(ro)
    assert first[0] == .5 and clearance[0] < 0.
