"""Observer unit checks, with no policy or scenario episode execution."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
import safety_observer as observer
import safety_v2 as v2
import ship


def snapshot(ego):
    return SimpleNamespace(u=float(ego[0]), v=float(ego[1]), r=float(ego[2]),
                           heading=0.3, x=2.0, y=4.0)


def test_prediction_does_not_modify_snapshot_or_actuator_history():
    snap = snapshot([0.5, -0.04, math.radians(12.0)])
    act = cc.Actuators()
    act.buffer = [0.1] * cc.DELAY_STEPS
    act.servo, act.executed = 0.02, 0.4
    before_snap, before_act = copy.deepcopy(vars(snap)), copy.deepcopy(vars(act))
    result = observer.advance_ego(snap, act, -0.8, -24.0)
    assert result.shape == (3,)
    assert np.isfinite(result).all()
    assert vars(snap) == before_snap
    assert vars(act) == before_act


@pytest.mark.parametrize("initialised_buffer", [False, True])
def test_prediction_matches_nominal_plant_without_measurement_noise(monkeypatch, initialised_buffer):
    monkeypatch.setattr(cc, "PRED_DT", 0.05)
    monkeypatch.setattr(cc, "SUBSTEPS", 10)
    monkeypatch.setattr(cc, "DELAY_STEPS", int(round(cc.IDENTIFIED["rud_delay"] / 0.05)))
    plant, act = ship.ShipModel(), cc.Actuators()
    plant._s[0, 0] = 0.6
    if initialised_buffer:
        plant._delayed_command(0.0, 0.05)
        act.buffer = [0.0] * cc.DELAY_STEPS
    commands = [(1.0, 12.0)] * 6 + [(-1.0, 12.0)] * 4 + [(0.0, -24.0)] * 2 + [(0.0, 0.0)] * 4
    estimate, stationary = observer.EgoObserver(), None
    estimated_errors, ema_errors = [], []
    for rudder, rpm in commands:
        truth = plant._s[:3, 0].copy()
        updated = estimate.update(truth)
        stationary = truth.copy() if stationary is None else stationary + v2.EGO_SMOOTHING * (truth - stationary)
        estimated_errors.append(np.abs(updated - truth))
        ema_errors.append(np.abs(stationary - truth))
        estimate.predict(snapshot(updated), act, rudder, rpm)
        act.issue(SimpleNamespace(command_rate_limit=False), rudder)
        for _ in range(5):
            plant.update(rpm, rudder * 100.0, 0.1)
    np.testing.assert_allclose(estimated_errors, 0.0, atol=1e-12)
    assert np.max(np.asarray(ema_errors)[:, 0]) > 0.1
    assert np.max(np.asarray(ema_errors)[:, 2]) > math.radians(2.0)


def test_held_measurement_is_not_assimilated_again(monkeypatch):
    estimate = observer.EgoObserver()
    raw = np.array([0.5, 0.0, 0.0])
    np.testing.assert_array_equal(estimate.update(raw), raw)
    predicted = np.array([0.2, 0.02, 0.1])
    monkeypatch.setattr(observer, "advance_ego", lambda *args: predicted.copy())
    estimate.predict(snapshot(raw), cc.Actuators(), 0.0, -24.0)
    np.testing.assert_array_equal(estimate.update(raw, fresh=False), predicted)
    # With neither a new command nor a fresh sample, repeated reads are stable.
    np.testing.assert_array_equal(estimate.update(raw, fresh=False), predicted)
    next_prediction = np.array([0.1, 0.01, 0.09])
    monkeypatch.setattr(observer, "advance_ego", lambda *args: next_prediction.copy())
    estimate.predict(snapshot(predicted), cc.Actuators(), 0.0, -24.0)
    np.testing.assert_array_equal(estimate.update(raw, fresh=False), next_prediction)


def test_fresh_correction_uses_existing_gain_and_returns_independent_arrays():
    estimate = observer.EgoObserver()
    raw = np.array([0.5, 0.0, 0.0])
    estimate.update(raw)
    estimate.prior = np.array([0.2, 0.1, 0.03])
    expected = estimate.prior + v2.EGO_SMOOTHING * (raw - estimate.prior)
    output = estimate.update(raw)
    np.testing.assert_allclose(output, expected)
    output[:] = 42.0
    raw[:] = 21.0
    np.testing.assert_allclose(estimate.estimate, expected)
    estimate.reset()
    assert estimate.estimate is None and estimate.prior is None


def test_missing_or_invalid_initialisation_is_reported():
    estimate = observer.EgoObserver()
    with pytest.raises(RuntimeError, match="call update"):
        estimate.predict(snapshot([0.0, 0.0, 0.0]), cc.Actuators(), 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        estimate.update([np.nan, 0.0, 0.0])
