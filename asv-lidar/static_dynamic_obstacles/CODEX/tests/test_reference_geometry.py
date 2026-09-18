"""Physical geometric fixtures for predicted target-hull clearance."""

import math

import numpy as np
import pytest

from reference_controller import hull_separation
from ship import HULL_MARGIN, VESSEL_LENGTH, VESSEL_WIDTH


HALF_LENGTH = 0.5 * VESSEL_LENGTH + HULL_MARGIN
HALF_WIDTH = 0.5 * VESSEL_WIDTH + HULL_MARGIN


@pytest.mark.parametrize("target,heading", [
    ((2.0 * HALF_WIDTH + 0.4, 0.0), 0.0),
    ((0.0, 2.0 * HALF_LENGTH + 0.4), 0.0),
    ((0.0, HALF_LENGTH + HALF_WIDTH + 0.4), math.pi / 2.0),
])
def test_axis_aligned_and_crossing_clearances_are_in_metres(target, heading):
    assert hull_separation((0.0, 0.0), 0.0, target, heading) == pytest.approx(0.4)


def test_overlap_touch_and_diagonal_gap_are_distinguished():
    assert hull_separation((0, 0), 0.0, (0, 0), 0.0) < 0.0
    assert hull_separation((0, 0), 0.0, (2 * HALF_WIDTH, 0), 0.0) == pytest.approx(0.0)
    # Corner-to-corner Euclidean separation is sqrt(0.4**2 + 0.4**2).
    # The SAT gap correctly remains a conservative lower bound, not distance.
    target = (2 * HALF_WIDTH + 0.4, 2 * HALF_LENGTH + 0.4)
    gap = hull_separation((0, 0), 0.0, target, 0.0)
    assert gap == pytest.approx(0.4)
    assert gap < math.hypot(0.4, 0.4)


def test_gap_is_unchanged_under_compass_rotation_and_translation():
    own = np.array([0.0, 0.0])
    target = np.array([2.0, 3.0])
    own_heading, target_heading = 0.35, -0.72
    expected = hull_separation(own, own_heading, target, target_heading)
    angle = 1.12
    rotation = np.array([[math.cos(angle), math.sin(angle)],
                         [-math.sin(angle), math.cos(angle)]])
    shift = np.array([8.0, -2.0])
    actual = hull_separation(rotation @ own + shift, own_heading + angle,
                             rotation @ target + shift, target_heading + angle)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_rollout_and_candidate_axes_broadcast_independently():
    own = np.zeros((2, 3, 2))
    own[1, :, 0] = 0.1
    target = np.array([[[2.0, 0.0]], [[3.0, 0.0]]])
    result = hull_separation(own, np.zeros((2, 3)), target, 0.0)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], 2.0 - 2.0 * HALF_WIDTH)
    np.testing.assert_allclose(result[1], 2.9 - 2.0 * HALF_WIDTH)


def test_extra_inflation_cannot_increase_certified_clearance():
    normal = hull_separation((0, 0), 0.3, (2, 3), -0.8)
    inflated = hull_separation((0, 0), 0.3, (2, 3), -0.8, margin=HULL_MARGIN + 0.2)
    assert inflated < normal
