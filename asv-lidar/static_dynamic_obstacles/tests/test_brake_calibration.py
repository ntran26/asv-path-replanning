"""Offline calibration math and data separation; no environment episodes."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/brake_calibration.py"
SPEC = importlib.util.spec_from_file_location("brake_calibration_test_module", PATH)
cal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cal
SPEC.loader.exec_module(cal)


def pulse(decisions=1, u=1.0):
    measured = np.tile([u, .03, .04], (decisions + 1, 1))
    return cal.Pulse("synthetic", "v4", "development", 0, measured,
                     measured.copy(), np.tile([.2, -24.], (decisions, 1)),
                     .1, np.linspace(-.1, .1, cal.cc.DELAY_STEPS))


def no_flow(state, *_):
    return state.copy()


@pytest.mark.parametrize("delay,expected", [(0., .5), (.125, .375), (.2, .3), (.5, 0.), (.75, 0.)])
def test_active_duration_integrates_partial_substep(delay, expected):
    duration = sum(cal.active_duration(i * .125, delay, .125) for i in range(4))
    assert duration == pytest.approx(expected)


def test_brake_onset_age_is_carried_across_decisions(monkeypatch):
    monkeypatch.setattr(cal.cc.dyn, "rk4_step", no_flow)
    result = cal.predict([pulse(3)], .4, delay=.75)[0]
    np.testing.assert_allclose(result[:, 0], [1., .9, .7])
    np.testing.assert_allclose(result[:, 1:], [[.03, .04]] * 3)


def test_reverse_never_creates_negative_physical_surge(monkeypatch):
    monkeypatch.setattr(cal.cc.dyn, "rk4_step", no_flow)
    measured = pulse(3, u=.1)
    result = cal.predict([measured], .5)[0]
    np.testing.assert_array_equal(result[:, 0], 0.)
    # Noisy negative response measurements are still valid residual targets.
    measured.measured[1:, 0] = -.02
    residual = cal.residuals([measured], .5).reshape(3, 3)
    np.testing.assert_allclose(residual[:, 0], .02 / cal.SIGMA[0])


def test_prediction_keeps_measurement_and_actuator_inputs_unchanged():
    item = pulse(2)
    before = {name: getattr(item, name).copy()
              for name in ("measured", "truth", "commands", "pending")}
    initial = np.array([[.3, -.03, .1]])
    saved_initial = initial.copy()
    result = cal.predict([item], .46, initial=initial)
    assert np.isfinite(result[0]).all()
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(item, name), expected)
    np.testing.assert_array_equal(initial, saved_initial)


def test_ground_truth_cannot_change_fitting_residuals_or_estimate(monkeypatch):
    monkeypatch.setattr(cal.cc.dyn, "rk4_step", no_flow)
    item = pulse(2)
    item.measured[1:, 0] = [.8, .6]
    changed = replace(item, truth=np.full_like(item.truth, 123456.))
    np.testing.assert_array_equal(cal.residuals([item], .3), cal.residuals([changed], .3))
    a, _, _ = cal.fit([item])
    b, _, _ = cal.fit([changed])
    assert a == pytest.approx(.4, abs=1e-5)
    assert a == b


def test_legacy_delay_starts_after_first_prediction_substep(monkeypatch):
    monkeypatch.setattr(cal.cc.dyn, "rk4_step", no_flow)
    # Existing v3 timestamp starts at t=.125; .75s delay first acts at .875.
    result = cal.predict([pulse(2)], .4, delay=.75, legacy=True)[0]
    np.testing.assert_allclose(result[:, 0], [1., .9])
