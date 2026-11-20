"""Real-map algebra enclosure tests; no environment or episode execution.

Floating source comparisons use a stated numerical tolerance.  The analytic
identities, rather than samples, justify the enclosure simplifications.
"""

import numpy as np
import pytest

import ship
from safety_reachability import ConditionalPlantTube, IntervalBounds


def zero_force_tube(state_changes=None, parameter_changes=None):
    parameters = dict(ship.IDENTIFIED)
    parameters.update(T12=0.0, X_uu=0.0, X_vv=0.0, X_rr=0.0, X_delta=0.0,
                      Y_v=0.0, Y_vv=0.0, k_R=0.0, N_r=0.0, N_rr=0.0,
                      N_uv=ship.dyn.M11-ship.dyn.M22)
    parameters.update(parameter_changes or {})
    pbox = {key: IntervalBounds(value, value) for key, value in parameters.items()}
    state = {key: IntervalBounds(0.0, 0.0) for key in ("x", "y", "heading", "u", "v", "r")}
    state.update(state_changes or {})
    return ConditionalPlantTube(pbox, state, delay_steps=0)


def ends(value):
    return float(value.a), float(value.b)


@pytest.mark.parametrize("lo,hi", [(-0.2, 0.4), (-0.4, 0.2), (-0.3, -0.1),
                                  (0.1, 0.3), (0.0, 0.0), (0.2, 0.2),
                                  (-0.5, 0.0), (0.0, 0.5)])
def test_signed_square_is_endpoint_image_even_across_zero(lo, hi):
    tube = zero_force_tube()
    result = tube._signed_square(tube._ctx.mpf([lo, hi]))
    expected = (lo * abs(lo), hi * abs(hi))
    assert ends(result) == pytest.approx(expected, abs=2e-14)
    samples = np.linspace(lo, hi, 2001)
    actual = samples * np.abs(samples)
    lower, upper = ends(result)
    assert actual.min() >= lower - 2e-14
    assert actual.max() <= upper + 2e-14


def test_drift_and_munk_moments_cancel_for_all_admitted_u_and_v():
    tube = zero_force_tube({"u": IntervalBounds(0.1, 1.0),
                            "v": IntervalBounds(-0.6, 0.6)})
    yaw_derivative = tube._rhs(tube._state, tube._point(0.0))[2]
    # Both terms separately vary widely; their sum is exactly zero for this
    # declared parameter, for every shared u*v value rather than just its mean.
    assert ends(yaw_derivative) == (0.0, 0.0)


def test_crossflow_square_cannot_become_spurious_positive_surge_force():
    tube = zero_force_tube({"v": IntervalBounds(-0.2, 0.4)}, {"X_vv": 2.0})
    surge_derivative = tube._rhs(tube._state, tube._point(0.0))[0]
    assert ends(surge_derivative) == pytest.approx((-2.0 * 0.4**2 / ship.dyn.M11, 0.0), abs=2e-14)


@pytest.mark.parametrize("name,index,mass,coefficient", [
    ("v", 1, ship.dyn.M22, "Y_vv"), ("r", 2, ship.dyn.M33, "N_rr"),
])
@pytest.mark.parametrize("low,high", [(-0.3, -0.1), (0.1, 0.3), (-0.2, 0.4)])
def test_signed_damping_preserves_restoring_sign_and_asymmetric_crossing(name, index, mass, coefficient, low, high):
    tube = zero_force_tube({name: IntervalBounds(low, high)}, {coefficient: 2.0})
    derivative = tube._rhs(tube._state, tube._point(0.0))[index]
    expected = (-2.0 * high * abs(high) / mass, -2.0 * low * abs(low) / mass)
    assert ends(derivative) == pytest.approx(expected, abs=2e-14)
    if low >= 0.0:
        assert float(derivative.b) <= 0.0
    if high <= 0.0:
        assert float(derivative.a) >= 0.0


def test_full_rhs_contains_1000_independent_one_percent_state_parameter_samples():
    # Compare against the unmodified source RHS, which still contains the
    # original separated moments and products. No rollout/plant is invoked.
    generator = np.random.default_rng(61853)
    center = dict(u=0.55, v=-0.04, r=0.07, heading=0.35, x=2.0, y=4.0)
    interval = lambda value: IntervalBounds(min(0.99*value, 1.01*value),
                                           max(0.99*value, 1.01*value))
    pbox = {key: interval(value) for key, value in ship.IDENTIFIED.items()}
    state_box = {key: interval(value) for key, value in center.items()}
    servo_box = interval(-0.12)
    tube = ConditionalPlantTube(pbox, state_box, delay_steps=15, servo=servo_box)
    result = tube._rhs(tube._state, tube._point(6.0))
    count = 1000
    sample_p = {key: generator.uniform(box.lo, box.hi, count) for key, box in pbox.items()}
    sample_state = np.array([generator.uniform(box.lo, box.hi, count)
                             for box in (state_box["u"], state_box["v"], state_box["r"],
                                         state_box["heading"], servo_box,
                                         state_box["x"], state_box["y"])])
    actual = ship.dyn.derivatives(sample_state, np.full(count, 6.0), np.zeros(count), sample_p)
    for component, box in enumerate(result):
        lower, upper = ends(box)
        assert actual[component].min() >= lower - 2e-13
        assert actual[component].max() <= upper + 2e-13
