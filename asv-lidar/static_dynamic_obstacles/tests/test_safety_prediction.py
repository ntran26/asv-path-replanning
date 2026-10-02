"""Braking-envelope unit tests; no policies or evaluation episodes are loaded."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_prediction as prediction
import safety_v3 as v3
import ship


def snapshot():
    return cc.Snapshot(5.0, 7.0, 0.0, 0.525, -0.05, math.radians(17.0),
                       np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([5.0, 7.0]),
                       0.0, 0.0, 20.0, np.empty((0, 2)))


def actuators():
    act = cc.Actuators()
    act.buffer = [0.0] * cc.DELAY_STEPS
    return act


def test_default_rollout_matches_v3_including_brake_release_and_restart():
    sequences = np.array([
        [(0.7, 0.3), (-0.4, 0.0), (0.1, -1.0), (0.5, 0.2)],
        [(0.0, np.nan), (0.0, np.nan), (0.5, 0.0), (-0.5, np.nan)],
        [(1.0, 0.0), (-1.0, np.nan), (0.5, np.nan), (0.0, np.nan)],
    ])
    snap, act = snapshot(), actuators()
    original = v3.rollout_seq(snap, act, sequences)
    actual = prediction.rollout_seq(snap, act, sequences)
    for name in ("positions", "headings", "speeds", "times"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))


def test_fast_brake_matches_nominal_plant_and_retains_spin(monkeypatch):
    # Exact plant integration resolution isolates the braking-model assumption.
    monkeypatch.setattr(cc, "PRED_DT", 0.05)
    monkeypatch.setattr(cc, "SUBSTEPS", 10)
    monkeypatch.setattr(cc, "DELAY_STEPS", int(round(cc.IDENTIFIED["rud_delay"] / 0.05)))
    monkeypatch.setattr(cfg, "FIXED_RPM", False)
    monkeypatch.setattr(cfg, "CRUISE_RPM", 6.0)
    monkeypatch.setattr(cfg, "RPM_DELTA", 6.0)
    monkeypatch.setattr(cfg, "RPM_FLOOR", 0.0)
    monkeypatch.setattr(cfg, "RPM_CEIL", 12.0)
    snap, act = snapshot(), actuators()
    sequences = np.array([[(0.0, np.nan)] * 2 + [(0.0, -1.0)] * 14])
    weak = prediction.rollout_seq(snap, act, sequences)
    fast = prediction.rollout_seq(snap, act, sequences,
                                  brake_efficiency=ship.REVERSE_THRUST_EFFICIENCY,
                                  brake_delay_s=0.0)
    plant = ship.ShipModel()
    plant._s[:, 0] = [snap.u, snap.v, snap.r, snap.heading, 0.0, snap.x, snap.y]
    plant._delayed_command(0.0, 0.05)
    for decision in range(16):
        rpm = -24.0 if decision < 2 else 0.0
        for _ in range(5):
            plant.update(rpm, 0.0, 0.1)
    np.testing.assert_allclose(fast.positions[-1, 0], plant._s[5:7, 0], atol=1e-12)
    assert fast.headings[-1, 0] == pytest.approx(plant._s[3, 0], abs=1e-12)
    # Fast braking can retain MORE heading change: shorter stopping distance
    # does not imply that the complete swept hull is conservatively bounded.
    assert fast.headings[-1, 0] - weak.headings[-1, 0] > math.radians(10.0)
    assert np.linalg.norm(fast.positions[-1, 0] - weak.positions[-1, 0]) > 0.2


def test_envelope_requires_both_complete_paths_and_only_duplicates_braking(monkeypatch):
    sequences = np.zeros((4, 4, 2))
    sequences[1:, 2, 1] = np.nan
    calls = []

    def rollout(snap, act, seq, **kwargs):
        calls.append((seq.copy(), kwargs))
        return SimpleNamespace(count=len(seq), fast=bool(kwargs))

    def evaluate(snap, rollout):
        if rollout.fast:
            assert rollout.count == 3
            return np.array([1.0, np.inf, np.inf]), np.array([-0.1, 0.8, 0.3])
        return np.array([np.inf, np.inf, 2.0, np.inf]), np.array([0.4, 0.5, -0.2, 0.6])

    monkeypatch.setattr(prediction, "rollout_seq", rollout)
    result = prediction.evaluate_sequences(None, None, sequences, evaluate)
    np.testing.assert_array_equal(result.first, [np.inf, 1.0, 2.0, np.inf])
    np.testing.assert_allclose(result.clear, [0.4, -0.1, -0.2, 0.3])
    np.testing.assert_array_equal(result.braking_mask, [False, True, True, True])
    assert result.dual_count == 3
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0], sequences[1:])
    assert calls[1][1] == {"brake_efficiency": ship.REVERSE_THRUST_EFFICIENCY,
                           "brake_delay_s": 0.0}


def test_no_braking_uses_one_prediction_and_preserves_original_results(monkeypatch):
    calls = []
    monkeypatch.setattr(prediction, "rollout_seq", lambda *args, **kwargs: calls.append(args))
    expected_first, expected_clear = np.array([np.inf, 2.0]), np.array([0.7, -0.3])
    result = prediction.evaluate_sequences(None, None, np.zeros((2, 4, 2)),
                                           lambda snap, ro: (expected_first, expected_clear))
    assert len(calls) == 1
    assert result.dual_count == 0
    np.testing.assert_array_equal(result.first, expected_first)
    np.testing.assert_array_equal(result.clear, expected_clear)
