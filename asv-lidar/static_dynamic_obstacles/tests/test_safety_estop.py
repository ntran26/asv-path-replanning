"""Emergency-stop detector, stop commitment and evaluation-protocol rules.

Synthetic snapshots and stubbed filter decisions; no episode or policy call.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
import safety_estop as se
import safety_v16 as v16
import safety_v20_contingency as sc
import targets as tgtmod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics/safety"))
import estop_protocol as ep  # noqa: E402


def box(w=10.0, h=25.0):
    a = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    return a, np.roll(a, -1, axis=0)


def make_snap(x=5.0, y=5.0, heading=0.0, u=0.6, points=(), tracks=(), poly=None):
    a, b = box() if poly is None else poly
    pts = np.asarray(points, float).reshape(-1, 2)
    return cc.Snapshot(x, y, heading, u, 0.0, 0.0, np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                       np.array([x, y]), 0.0, 0.0, 15.0, pts, list(tracks), a, b)


def actuators():
    return SimpleNamespace(servo=0.0, buffer=[0.0] * cc.DELAY_STEPS)


GRID = np.array([(r, t) for r in (-1.0, 0.0, 1.0) for t in (-1.0, 1.0)] + [(0.0, np.nan)])


def test_sequences_hold_the_first_command_and_use_the_tail_banks():
    seqs = se.sequences([[0.5, 1.0], [0.0, np.nan]], decisions=8)
    bank = se.continuation_bank(8)
    assert seqs.shape == (2 * len(bank), 8, 2)
    assert np.allclose(seqs[0], [0.5, 1.0])                      # hold
    assert np.isnan(seqs[len(bank), :, 1]).all() and np.allclose(seqs[len(bank), :, 0], 0.0)
    assert np.allclose(seqs[1, 1:], sc.TAILS[0, :7], equal_nan=True)


def test_true_hull_clearance_exceeds_the_inflated_one_by_the_margin():
    pos = np.array([[[5.0, 5.0]]])
    hdg = np.zeros((1, 1))
    pts = np.array([[5.0, 6.5]])            # within the 1 m reach used for contact checks
    true = se.true_point_clearance(pos, hdg, pts)[0, 0]
    inflated = cc.point_clearance(pos, hdg, pts)[0, 0]
    assert true == pytest.approx(inflated + cc.HALF_L - se.TRUE_HALF_L)


def test_open_water_has_an_escape():
    t_star, _ = se.escape_time(make_snap(u=0.6), actuators(), GRID, window_s=3.0)
    assert math.isinf(t_star)


def test_enclosing_wall_close_ahead_makes_contact_certain():
    # A tight U of points around the bow and both sides: every continuation touches.
    pts = [(x, 6.4) for x in np.linspace(3.5, 6.5, 40)]
    pts += [(4.45, y) for y in np.linspace(3.0, 6.4, 30)] + [(5.55, y) for y in np.linspace(3.0, 6.4, 30)]
    t_star, details = se.escape_time(make_snap(u=0.9, points=pts), actuators(), GRID, window_s=3.0)
    assert t_star <= 3.0 and details["sequences"] == len(GRID) * len(se.continuation_bank(6))


def test_target_contact_uses_true_hulls():
    near = cc.TrackView(1, np.array([5.0, 7.3]), np.array([0.0, 0.0]), 0.0)
    snap = make_snap(u=0.0, tracks=[near])
    contact = se.first_contact_times(snap, sc.rollout_states(snap, actuators(), se.sequences([[0.0, np.nan]], 2)[:1],
                                                             *sc.STRONG))
    assert np.isinf(contact).all()          # true hulls 2.3 m apart along the axis: no contact at rest


def test_constructor_requires_a_window():
    with pytest.raises(ValueError):
        se.SafetyFilterV16E()
    with pytest.raises(ValueError):
        se.SafetyFilterV16E(estop_window_s=0.0)
    assert se.SafetyFilterV16E(estop_window_s=2.0).estop_window_s == 2.0


class Onboard(SimpleNamespace):
    @property
    def targets(self):
        raise AssertionError("truth target access")


def stub_v16(monkeypatch, why, out=(0.4, 0.8)):
    def fake(self, env, action):
        self._observer_snapshot = make_snap()
        self.actuators.issue(env, float(out[0]))
        env._v2_brake = False
        self.last = {"why": why}
        return np.asarray(out, np.float32), True
    monkeypatch.setattr(v16.SafetyFilterV16, "_filter", fake)


def make_env():
    return Onboard(command_rate_limit=False, _v2_brake=False, pose_stale=False)


def test_stop_commits_only_at_unchecked_decisions_and_is_permanent(monkeypatch):
    calls = []
    monkeypatch.setattr(se, "escape_time", lambda *a, **k: (calls.append(1) or 1.0, {}))
    filt, env = se.SafetyFilterV16E(estop_window_s=2.0), make_env()
    stub_v16(monkeypatch, "turn")
    out, _ = filt._filter(env, np.array([0.4, 0.8]))
    assert not calls and filt.estop_committed_decision is None and np.allclose(out, [0.4, 0.8])
    stub_v16(monkeypatch, "no escape")
    out, changed = filt._filter(env, np.array([0.4, 0.8]))
    assert calls and filt.estop_committed_decision == 2 and changed
    assert np.allclose(out, [0.0, -1.0]) and env._v2_brake and env.emergency_stop_committed
    assert filt.actuators.buffer[-1] == pytest.approx(0.0)
    stub_v16(monkeypatch, "turn")
    out, _ = filt._filter(env, np.array([0.9, 0.9]))
    assert np.allclose(out, [0.0, -1.0]) and filt.last["why"] == "emergency stop"


def test_no_stop_when_an_escape_exists(monkeypatch):
    monkeypatch.setattr(se, "escape_time", lambda *a, **k: (math.inf, {}))
    filt, env = se.SafetyFilterV17E(estop_window_s=2.0), make_env()
    stub_v16(monkeypatch, "last certificate")
    out, _ = filt._filter(env, np.array([0.4, 0.8]))
    assert filt.estop_committed_decision is None and np.allclose(out, [0.4, 0.8])
    assert math.isinf(filt.last["estop_t_star"])


def protocol_env(target):
    return SimpleNamespace(targets=[target], u_body=0.0, v_body=0.0, asv_w=0.0, speed_mps=0.0)


def test_targets_brake_only_after_the_commitment():
    target = tgtmod.Target(x=0.0, y=0.0, heading=0.0, speed=1.0)
    env = protocol_env(target)
    ep.install(env)
    target.step(0.5)
    assert target.y == pytest.approx(0.5) and target.speed == pytest.approx(1.0)
    env.emergency_stop_committed = True
    for _ in range(20):
        target.step(0.25)
    assert target.speed == 0.0
    travelled = target.y - 0.5
    assert travelled == pytest.approx(1.0 / (2 * ep.TARGET_DECEL_MPS2), abs=0.15)


def test_outcomes_after_commitment():
    target = tgtmod.Target(x=0.0, y=0.0, heading=0.0, speed=1.0)
    env = protocol_env(target)
    ep.install(env)
    info = {"collided": False, "reached_goal": False, "collision_kind": None}
    assert ep.after_step(env, (None, 0.0, False, False, info), 3)[2] is False   # not committed
    env.emergency_stop_committed = True
    env.u_body = 0.3
    assert ep.after_step(env, (None, 0.0, False, False, info), 4)[2] is False
    assert env.estop_protocol["commit_step"] == 4
    env.u_body, env.asv_w = 0.01, 3.0                       # still spinning: not still
    assert ep.after_step(env, (None, 0.0, False, False, info), 5)[2] is False
    env.asv_w = 0.5                                          # own ship still, target still braking
    assert ep.after_step(env, (None, 0.0, False, False, info), 6)[2] is False
    target.speed = 0.0
    result = ep.after_step(env, (None, 0.0, False, False, info), 7)
    assert result[2] is True and env.estop_protocol["outcome"] == "stopped"
    env2 = protocol_env(tgtmod.Target(x=0.0, y=0.0, heading=0.0, speed=1.0))
    ep.install(env2)
    env2.emergency_stop_committed = True
    ep.after_step(env2, (None, 0.0, True, False, {"collided": True, "collision_kind": "target",
                                                   "reached_goal": False}), 2)
    assert env2.estop_protocol["outcome"] == "collision:by_target"
