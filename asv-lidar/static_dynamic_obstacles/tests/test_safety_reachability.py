"""Synthetic transition checks; no environments, policy inference or episodes.

These numerical checks can falsify the interval implementation. They do not
prove binary64 roundoff containment or the assumptions of a safety theorem.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

import ship
from safety_reachability import ConditionalPlantTube, EnclosureFailure, IntervalBounds


def point_parameters(overrides=None):
    values = dict(ship.IDENTIFIED)
    values.update(overrides or {})
    return values, {name: IntervalBounds(value, value) for name, value in values.items()}


def make_state(**overrides):
    values = dict(x=2.0, y=4.0, heading=0.8, u=0.7, v=-0.04, r=0.1)
    values.update(overrides)
    return values


def interval_state(values, radius=0.0):
    return {name: IntervalBounds(value - radius, value + radius)
            for name, value in values.items()}


def set_body(plant, values):
    for i, name in ((0, "u"), (1, "v"), (2, "r"), (3, "heading"), (5, "x"), (6, "y")):
        plant._s[i, 0] = values[name]


def concrete_state(plant):
    return {name: float(plant._s[i, 0]) for i, name in enumerate(
        ("u", "v", "r", "heading", "servo", "x", "y"))}


def assert_near_point(box, concrete, tolerance=2e-11):
    # A point real-arithmetic image and NumPy binary64 can differ by rounding.
    # This is a parity check with an explicit tolerance, not a containment proof.
    for name, actual in concrete.items():
        bound = box[name]
        assert bound.lo - tolerance <= actual <= bound.hi + tolerance, name
        assert bound.hi - bound.lo < tolerance, name


@pytest.mark.parametrize("rudder,rpm,delay", [(70.0, 12.0, 15), (-85.0, 24.0, 0),
                                               (100.0, -24.0, 3), (0.0, 0.0, 15)])
def test_point_transition_matches_independent_plant(rudder, rpm, delay):
    p, parameter_box = point_parameters({"rud_delay": delay * 0.05})
    initial = make_state()
    tube = ConditionalPlantTube(parameter_box, interval_state(initial), delay_steps=delay)
    plant = ship.ShipModel(p)
    set_body(plant, initial)
    box = tube.advance(rudder, rpm)
    # Match env's five collision-check intervals, rather than cc's predictor.
    for _ in range(5):
        plant.update(rpm, rudder, 0.1)
    assert_near_point(box, concrete_state(plant))


@pytest.mark.parametrize("delay", [0, 3, 15])
def test_history_reconstructs_actuator_without_propagating_body(delay):
    history = [(65.0, 12.0), (-80.0, 12.0), (25.0, -24.0)]
    p, parameter_box = point_parameters({"rud_delay": delay * 0.05})
    initial = make_state(heading=6.2)
    saved = deepcopy((parameter_box, initial, history))
    plant = ship.ShipModel(p)
    for rudder, rpm in history:
        for _ in range(5):
            plant.update(rpm, rudder, 0.1)
    set_body(plant, initial)
    tube = ConditionalPlantTube(parameter_box, interval_state(initial), delay_steps=delay,
                                command_history=history)
    before = tube.bounds()
    for name, value in initial.items():
        assert before[name].lo <= value <= before[name].hi
        assert before[name].hi - before[name].lo < 1e-12
    assert before["servo"].lo - 1e-12 <= plant._s[4, 0] <= before["servo"].hi + 1e-12
    after = tube.advance(-40.0, 7.0)
    for _ in range(5):
        plant.update(7.0, -40.0, 0.1)
    assert_near_point(after, concrete_state(plant), tolerance=1e-10)
    assert (parameter_box, initial, history) == saved


@pytest.mark.parametrize("rud_rate,rud_tau", [(2.0, 0.001), (2984.75, 0.9)])
def test_rate_limit_and_fast_servo_match_plant(rud_rate, rud_tau):
    p, parameter_box = point_parameters({"rud_rate": rud_rate, "rud_tau": rud_tau,
                                         "rud_delay": 0.0})
    initial = make_state()
    tube = ConditionalPlantTube(parameter_box, interval_state(initial), delay_steps=0)
    plant = ship.ShipModel(p)
    set_body(plant, initial)
    for rudder in (100.0, -100.0, 0.0):
        box = tube.advance(rudder, 12.0, dt=0.1)
        plant.update(12.0, rudder, 0.1)
        assert_near_point(box, concrete_state(plant))


def test_uncertain_box_contains_independent_fixed_parameter_realizations():
    rng = np.random.default_rng(63091)
    p, _ = point_parameters({"rud_delay": 0.15})
    parameters = {name: IntervalBounds(min(value * 0.99, value * 1.01),
                                      max(value * 0.99, value * 1.01))
                  for name, value in p.items()}
    parameters["rud_delay"] = IntervalBounds(0.15, 0.15)
    initial = make_state()
    state_box = interval_state(initial, radius=0.002)
    tube = ConditionalPlantTube(parameters, state_box, delay_steps=3,
                                servo=IntervalBounds(-0.003, 0.003))
    commands = [(45.0, 12.0), (-20.0, -24.0)]
    enclosures = [tube.advance(rud, rpm, dt=0.1) for rud, rpm in commands]
    for _ in range(12):
        # The sampled parameter vector is held FIXED over both transitions.
        sample_p = {name: rng.uniform(bound.lo, bound.hi) for name, bound in parameters.items()}
        plant = ship.ShipModel(sample_p)
        set_body(plant, {name: rng.uniform(bound.lo, bound.hi) for name, bound in state_box.items()})
        plant._s[4, 0] = rng.uniform(-0.003, 0.003)
        for (rudder, rpm), box in zip(commands, enclosures):
            plant.update(rpm, rudder, 0.1)
            for name, actual in concrete_state(plant).items():
                assert box[name].lo <= actual <= box[name].hi, (name, box[name], actual)


@pytest.mark.parametrize("lo,hi", [(1.0, -1.0), (-math.inf, 1.0), (0.0, math.inf),
                                   (math.nan, 1.0), (0.0, math.nan)])
def test_unbounded_or_invalid_inputs_are_rejected(lo, hi):
    with pytest.raises(ValueError):
        IntervalBounds(lo, hi)


def test_missing_state_and_parameter_cannot_produce_enclosure():
    _, parameters = point_parameters()
    state = interval_state(make_state())
    incomplete = dict(state)
    incomplete.pop("r")
    with pytest.raises((ValueError, KeyError)):
        ConditionalPlantTube(parameters, incomplete, delay_steps=15)
    parameters.pop("N_uv")
    with pytest.raises((ValueError, KeyError)):
        ConditionalPlantTube(parameters, state, delay_steps=15)


@pytest.mark.parametrize("dt", [0.0, -0.1, 0.03, math.inf, math.nan])
def test_unsupported_timing_rejected(dt):
    _, parameters = point_parameters()
    tube = ConditionalPlantTube(parameters, interval_state(make_state()), delay_steps=15)
    with pytest.raises((ValueError, EnclosureFailure)):
        tube.advance(0.0, 12.0, dt=dt)


def test_first_command_reaches_servo_even_with_nonzero_fifo_length():
    _, parameters = point_parameters()
    tube = ConditionalPlantTube(parameters, interval_state(make_state()), delay_steps=15)
    after = tube.advance(100.0, 12.0, dt=0.05)
    # ShipModel initial-fill behavior is unusual and easy to implement wrongly.
    assert after["servo"].hi < 0.0


def test_astern_removes_surge_and_never_creates_reverse_motion():
    _, parameters = point_parameters({"rud_delay": 0.0})
    tube = ConditionalPlantTube(parameters, interval_state(make_state(
        u=0.01, v=0.0, r=0.0, heading=0.0)), delay_steps=0)
    box = tube.advance(0.0, -24.0)
    assert abs(box["u"].lo) < 1e-14 and abs(box["u"].hi) < 1e-14
    assert box["y"].lo >= 4.0 - 1e-12


def test_failure_keeps_prior_state_and_actuator_history():
    _, parameters = point_parameters()
    initial = interval_state(make_state())
    history = [(50.0, 12.0), (-30.0, 12.0)]
    tube = ConditionalPlantTube(parameters, initial, delay_steps=15, command_history=history)
    reference = ConditionalPlantTube(parameters, initial, delay_steps=15, command_history=history)
    before = tube.bounds()
    with pytest.raises(EnclosureFailure):
        tube.advance(95.0, 1e20)
    assert tube.bounds() == before
    assert tube.advance(-15.0, 10.0) == reference.advance(-15.0, 10.0)


def test_interval_precision_is_private():
    from mpmath import iv, mp
    precisions = (iv.dps, mp.dps)
    _, parameters = point_parameters()
    tube = ConditionalPlantTube(parameters, interval_state(make_state()), delay_steps=15)
    tube.advance(0.0, 12.0, dt=0.05)
    assert (iv.dps, mp.dps) == precisions
    assert tube.diagnostics["mission_safety_certificate"] is False
    assert tube.diagnostics["binary64_roundoff_validated"] is False
