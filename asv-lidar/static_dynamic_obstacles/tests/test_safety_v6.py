"""V6 experiment isolation and explicit braking prediction; no policy episodes."""
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
import safety_prediction
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v6 as v6


@pytest.fixture
def make_filter(monkeypatch):
    for name in ("COUNTERSTEER_RECOVERY", "RETAIN_NOMINAL_PLAN",
                 "DUAL_BRAKE_PREDICTION", "FREE_SPACE_MEMORY", "MODEL_EGO_OBSERVER"):
        monkeypatch.setattr(v4, name, False)
    monkeypatch.setattr(v6, "CALIBRATED_BRAKING", False)
    monkeypatch.setattr(v6, "FITTED_HULL_PERCEPTION", False)
    for flag in ("PROVISIONAL_TRACKS", "PROVISIONAL_MOTION_EVIDENCE",
                 "TRAJECTORY_SEARCH", "TRACK_HISTORY"):
        monkeypatch.setattr(v6, flag, False)
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, seq: seq)

    def make(cls, stored, traffic):
        f = cls()
        snap = cc.Snapshot(5., 5., 0., 1., 0., 0., np.array([0., 1.]),
                           np.array([1., 0.]), np.array([5., 5.]), 0., 0., 20.,
                           np.empty((0, 2)))
        if traffic:
            snap.tracks = [SimpleNamespace(position=np.array([5., 7.]))]
        monkeypatch.setattr(f.perception, "snapshot", lambda env: snap)
        monkeypatch.setattr(f, "_threat_in_reach", lambda snap: True)
        monkeypatch.setattr(f, "_rejoin", lambda snap: np.zeros(2))
        monkeypatch.setattr(f.actuators, "issue", lambda env, rudder: None)
        if stored:
            f.plan = v3.plan_for((0.75, 0.1), (-1., 0.))
        return f, SimpleNamespace(_v2_brake=False)
    return make


@pytest.mark.parametrize("why", ["nominal", "turn", "continue", "brake",
                                  "no escape", "hold back", "last certificate"])
def test_disabled_experiments_preserve_v4(make_filter, monkeypatch, why):
    stored, traffic = why in ("continue", "last certificate"), why == "brake"
    reference, ref_env = make_filter(v4.SafetyFilterV4, stored, traffic)
    candidate, env = make_filter(v6.SafetyFilterV6, stored, traffic)
    nrec = sum(traffic or t is not v2.BRAKE for _, t in v2.RECOVERY)

    def evaluate(snap, seq):
        first, clear = np.zeros(len(seq)), np.full(len(seq), -1.)
        index = None
        if why == "nominal":
            index = 1
        elif why == "turn":
            index = 3 * nrec + 1
        elif why == "continue":
            index = len(seq) - 1
        elif why == "brake":
            index = (2 + len(v2.RUDDERS) * len(v2.THROTTLES)) * nrec
        elif why == "hold back":
            first[3 * nrec] = 2. + v2.HOLD_BACK_GAIN_S
        if index is not None:
            first[index], clear[index] = np.inf, .8
        return first, clear

    for f in (reference, candidate):
        monkeypatch.setattr(f, "_evaluate", evaluate)
    action = np.array([.2, .1])
    out, changed = candidate.filter(env, action)
    expected, ref_changed = reference.filter(ref_env, action)
    np.testing.assert_array_equal(out, expected)
    assert changed == ref_changed
    assert env._v2_brake == ref_env._v2_brake
    assert candidate.mode == reference.mode
    assert candidate.last["why"] == reference.last["why"] == why
    assert candidate.uncertified_steps == reference.uncertified_steps
    if reference.plan is None:
        assert candidate.plan is None
    else:
        np.testing.assert_array_equal(candidate.plan, reference.plan)


def test_calibrated_backend_passes_explicit_fit_without_mutating_v2(monkeypatch):
    monkeypatch.setattr(v6, "CALIBRATED_BRAKING", True)
    monkeypatch.setattr(v6, "BRAKE_EFFICIENCY", .37)
    monkeypatch.setattr(v6, "BRAKE_DELAY_S", .125)
    original = (v2.BRAKE_EFFICIENCY, v2.BRAKE_DELAY_S)
    observed = {}
    sentinel = object()

    def predict(snap, act, seq, **kwargs):
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(safety_prediction, "rollout_seq", predict)
    f = v6.SafetyFilterV6()
    assert f._rollout(None, None, None) is sentinel
    assert observed == {"brake_efficiency": .37, "brake_delay_s": .125}
    assert original == (v2.BRAKE_EFFICIENCY, v2.BRAKE_DELAY_S)


@pytest.mark.parametrize("efficiency", [0., -1., 1.01, np.nan, np.inf])
def test_invalid_calibration_cannot_be_enabled(monkeypatch, efficiency):
    monkeypatch.setattr(v6, "CALIBRATED_BRAKING", True)
    monkeypatch.setattr(v6, "BRAKE_EFFICIENCY", efficiency)
    with pytest.raises(ValueError, match="development-fitted"):
        v6.SafetyFilterV6()


@pytest.mark.parametrize("delay", [-.1, np.nan, np.inf])
def test_invalid_calibrated_delay_rejected(monkeypatch, delay):
    monkeypatch.setattr(v6, "CALIBRATED_BRAKING", True)
    monkeypatch.setattr(v6, "BRAKE_EFFICIENCY", .4)
    monkeypatch.setattr(v6, "BRAKE_DELAY_S", delay)
    with pytest.raises(ValueError, match="delay"):
        v6.SafetyFilterV6()
