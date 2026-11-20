"""Servo-image arithmetic checks only; no environment or episode execution."""

import math
from itertools import product

import numpy as np
import pytest

import ship
from safety_reachability import ConditionalPlantTube, IntervalBounds


def tube(tau=(0.6, 1.2), rate=(1000.0, 3000.0), history=None):
    params = {key: IntervalBounds(value, value) for key, value in ship.IDENTIFIED.items()}
    params["rud_tau"] = IntervalBounds(*tau)
    params["rud_rate"] = IntervalBounds(*rate)
    state = {key: IntervalBounds(0.0, 0.0)
             for key in ("x", "y", "heading", "u", "v", "r")}
    return ConditionalPlantTube(params, state, delay_steps=0, command_history=history)


def endpoints(interval):
    return float(interval.a), float(interval.b)


def test_uncapped_servo_image_matches_analytic_extrema_without_dependency_growth():
    plant = tube()
    result = plant._servo(plant._ctx.mpf([0.2, 0.4]), plant._ctx.mpf([0.6, 0.6]))
    low = 0.2 + (0.6 - 0.2) * (1.0 - math.exp(-0.05 / 1.2))
    high = 0.4 + (0.6 - 0.4) * (1.0 - math.exp(-0.05 / 0.6))
    assert endpoints(result) == pytest.approx((low, high), abs=2e-14)


def test_held_command_history_has_analytic_servo_range_not_full_deflection_range():
    plant = tube(history=[(80.0, 6.0)] * 10)
    box = plant.bounds()["servo"]
    command = -math.radians(40.0) * 0.8
    ends = [command * (1.0 - math.exp(-5.0 / tau)) for tau in (0.6, 1.2)]
    assert (box.lo, box.hi) == pytest.approx((min(ends), max(ends)), abs=2e-13)
    assert box.hi - box.lo < 0.01


@pytest.mark.parametrize("name,servo,command,tau,rate", [
    ("positive_up", (0.1, 0.2), (0.4, 0.55), (0.3, 1.2), (1.0, 12.0)),
    ("positive_down", (0.35, 0.45), (0.05, 0.15), (0.01, 0.3), (0.5, 5.0)),
    ("negative_up", (-0.5, -0.4), (-0.2, -0.05), (0.1, 0.9), (1.0, 8.0)),
    ("negative_down", (-0.15, -0.05), (-0.55, -0.4), (0.02, 0.8), (0.5, 10.0)),
    ("cross_clipped", (0.66, 0.78), (-0.9, -0.65), (0.03, 0.25), (100.0, 1000.0)),
])
def test_servo_enclosure_contains_20000_interiors_and_corners(name, servo, command, tau, rate):
    # Compare against the actual plant primitive, not another copy of the
    # corner implementation. Sampling can falsify, but cannot prove, containment.
    generator = np.random.default_rng(61427)
    plant = tube(tau=tau, rate=rate)
    enclosure = plant._servo(plant._ctx.mpf(servo), plant._ctx.mpf(command))
    sample = np.column_stack([generator.uniform(*bounds, 20000)
                              for bounds in (servo, command, tau, rate)])
    sample = np.vstack([sample, np.array(list(product(servo, command, tau, rate)))])
    actual = ship.dyn.advance_rudder(
        sample[:, 0], sample[:, 1],
        {"rud_tau": sample[:, 2], "rud_rate": sample[:, 3]}, 0.05)
    low, high = endpoints(enclosure)
    assert float(actual.min()) >= low - 2e-14
    assert float(actual.max()) <= high + 2e-14
    if name != "cross_clipped":
        # These four checks must certify a narrow image, rather than succeeding
        # trivially because the enclosure spans the full +/-40-degree range.
        assert high - low < 0.2 < 2.0 * math.radians(40.0)


@pytest.mark.parametrize("servo,command", [(0.1, 0.6), (-0.2, -0.6), (0.2, 0.2)])
def test_zero_rate_limit_keeps_servo_for_either_command_direction(servo, command):
    plant = tube(rate=(0.0, 0.0))
    enclosure = plant._servo(plant._ctx.mpf(servo), plant._ctx.mpf(command))
    assert endpoints(enclosure) == pytest.approx((servo, servo), abs=2e-14)
