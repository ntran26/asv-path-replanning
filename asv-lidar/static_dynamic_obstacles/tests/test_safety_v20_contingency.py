"""V20 contingency certificate: model facts, rollout parity, geometry and checks.

Synthetic inputs only; no simulator reset, episode or policy call.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
import safety_prediction as sp
import safety_v2 as v2
import safety_v20_contingency as sc
import safety_v20_tubes as tubes
import ship
import targets as tgtmod

DYN = ship.dyn


def params(scale=None, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for key, value in ship.IDENTIFIED.items():
        factor = 1.0 if scale is None else float(rng.uniform(1 - scale, 1 + scale))
        out[key] = np.array([value * factor])
    return out


@pytest.mark.parametrize("scale", [None, 0.3, 0.6])
def test_rest_with_centred_rudder_is_an_equilibrium(scale):
    for seed in range(5):
        state = np.zeros((7, 1))
        d = DYN.derivatives(state, np.array([0.0]), np.array([0.0]), params(scale, seed))
        assert np.all(d == 0.0)


def test_rudder_held_at_rest_still_turns_the_hull():
    state = np.zeros((7, 1))
    state[4] = math.radians(35.0)
    d = DYN.derivatives(state, np.array([0.0]), np.array([math.radians(35.0)]), params())
    assert abs(d[2, 0]) > 1e-4


def test_braked_turn_keeps_spinning_at_zero_surge():
    # The identified model has almost purely quadratic yaw damping at zero
    # surge, so a braked turn is not at rest quickly. The certificate must
    # therefore simulate the window explicitly.
    snap = SimpleNamespace(x=0.0, y=0.0, heading=0.0, u=0.56, v=0.0, r=0.0)
    act = SimpleNamespace(servo=0.0, buffer=None)
    seq = np.tile((0.0, np.nan), (sc.DECISIONS, 1))
    seq[:4] = (1.0, 0.0)
    ro = sc.rollout_states(snap, act, seq[None], *sc.STRONG)
    late = ro.times >= 12.0
    assert np.all(ro.u[late] <= sc.STOP_SURGE)
    assert np.max(np.abs(ro.r[late])) > math.radians(1.0)


def test_tail_bank_encoding():
    tails = sc.TAILS
    assert tails.shape == (43, sc.DECISIONS - 1, 2)
    assert np.all(np.isnan(tails[:, -1, 1])) and np.all(tails[:, -1, 0] == 0.0)
    stops, turns = tails[:3], tails[3:]
    assert np.all(np.isnan(stops[:, :, 1]))
    assert set(stops[:, 0, 0]) == set(sc.STOP_RUDDERS)
    assert np.all(stops[:, 1:, 0] == 0.0)
    for tail in turns:
        finite = np.isfinite(tail[:, 1])
        n = int(finite.sum())
        assert n in sc.TURN_DECISIONS and finite[:n].all()
        assert np.all(tail[n:, 0] == 0.0)


def test_sequences_and_shift():
    first = np.array([[0.3, 0.5], [-0.2, np.nan]])
    seqs = sc.sequences_for(first)
    assert seqs.shape == (2 * len(sc.TAILS), sc.DECISIONS, 2)
    assert np.allclose(seqs[0, 0], first[0]) and np.isnan(seqs[len(sc.TAILS), 0, 1])
    shifted = sc.shift_sequence(seqs[5])
    assert shifted.shape == seqs[5].shape
    assert np.allclose(shifted[:-1], seqs[5][1:], equal_nan=True)
    assert shifted[-1, 0] == 0.0 and np.isnan(shifted[-1, 1])


@pytest.mark.parametrize("model", ["weak", "strong"])
def test_state_rollout_matches_inherited_predictor(model):
    snap = SimpleNamespace(x=1.0, y=2.0, heading=0.3, u=0.7, v=-0.05, r=0.08)
    act = cc.Actuators()
    env = SimpleNamespace(command_rate_limit=False)
    for rudder in (0.2, -0.4, 0.4):
        act.issue(env, rudder)
    seqs = sc.sequences_for(np.array([[0.5, 0.3], [-1.0, np.nan]]))[::7]
    eff, delay = sc.WEAK if model == "weak" else sc.STRONG
    ours = sc.rollout_states(snap, act, seqs, eff, delay)
    ref = sp.rollout_seq(snap, act, seqs, brake_efficiency=eff, brake_delay_s=delay)
    assert np.allclose(ours.positions[1:], ref.positions)
    assert np.allclose(ours.headings[1:], ref.headings)
    assert np.allclose(np.hypot(ours.u, ours.v)[1:], ref.speeds)
    assert np.allclose(ours.positions[0], [1.0, 2.0])


def test_lookup_is_monotone_next_horizon_and_extrapolates():
    table = ((0.5, 0.1), (1.0, 0.3), (1.5, 0.2), (2.0, 0.5))
    t = np.array([0.0, 0.5, 0.6, 1.5, 2.0, 3.0])
    out = sc._lookup(table, t)
    assert np.allclose(out[:5], [0.1, 0.1, 0.3, 0.3, 0.5])
    assert out[5] == pytest.approx(0.5 + (0.5 - 0.3) / 0.5 * 1.0)
    assert np.all(sc._lookup(None, t) == 0.0)
    two_d = sc._lookup(table, np.tile(t[:, None], (1, 3)))
    assert two_d.shape == (6, 3) and np.allclose(two_d[:, 1], out)


def box(w=10.0, h=25.0):
    a = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    return a, np.roll(a, -1, axis=0)


def test_signed_corners_catch_a_hull_outside_the_wall():
    a, b = box()
    outside = np.array([[[-1.5, 10.0]]])
    hdg = np.zeros((1, 1))
    unsigned = cc.boundary_clearance(outside, hdg, a, b)
    signed = sc.signed_corner_clearance(outside, hdg, a, b)
    assert unsigned[0, 0] > 0.0 > signed[0, 0]
    inside = np.array([[[5.0, 10.0]]])
    assert sc.signed_corner_clearance(inside, hdg, a, b)[0, 0] == pytest.approx(5.0 - cc.HALF_W)


def test_signed_containment_is_at_least_as_strict_as_the_border_test():
    a, b = box()
    rng = np.random.default_rng(3)
    pos = rng.uniform([-1, -1], [11, 26], size=(4000, 1, 2)).transpose(1, 0, 2)
    hdg = rng.uniform(-np.pi, np.pi, size=(1, 4000))
    signed = sc.signed_corner_clearance(pos, hdg, a, b)[0]
    for i in np.flatnonzero(signed >= 0.0):
        hull = np.asarray(tgtmod.hull_polygon(pos[0, i, 0], pos[0, i, 1], math.degrees(hdg[0, i])))
        border = min(hull[:, 0].min(), 10.0 - hull[:, 0].max(), hull[:, 1].min(), 25.0 - hull[:, 1].max())
        assert border >= 0.0


def make_snap(x=5.0, y=5.0, heading=0.0, u=0.6, points=(), tracks=(), poly=None):
    a, b = box() if poly is None else poly
    pts = np.asarray(points, float).reshape(-1, 2)
    return cc.Snapshot(x, y, heading, u, 0.0, 0.0, np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                       np.array([x, y]), 0.0, 0.0, 15.0, pts, list(tracks), a, b)


def actuators():
    return SimpleNamespace(servo=0.0, buffer=[0.0] * cc.DELAY_STEPS)


@pytest.mark.parametrize("tables", [(None, None), (tubes.OWN_TABLE, tubes.TARGET_TABLE)])
def test_open_water_is_certified_and_qualifies(tables):
    snap = make_snap(u=0.5)
    cert = sc.ContingencyChecker(snap, actuators(), *tables).certify([0.0, 0.0])
    assert cert.certified and cert.slack >= 0.0
    assert cert.sequence.shape == (sc.DECISIONS, 2)
    assert cert.rest_time_s + sc.HOLD_HORIZON_S <= sc.DECISIONS * cfg.UPDATE_RATE


def test_wall_of_points_ahead_is_not_certified():
    wall = [(x, 6.6) for x in np.linspace(3.0, 7.0, 25)]
    snap = make_snap(u=0.9, points=wall)
    cert = sc.ContingencyChecker(snap, actuators()).certify([0.0, 1.0])
    assert not cert.certified and cert.diagnostics["static"] < 0.0


def test_target_on_collision_course_blocks_a_stop_but_not_when_far():
    head_on = cc.TrackView(1, np.array([5.0, 12.0]), np.array([0.0, -0.6]), math.pi)
    blocked = sc.ContingencyChecker(make_snap(u=0.5, tracks=[head_on]), actuators()).certify([0.0, 0.0])
    assert not blocked.certified and blocked.diagnostics["target"] < 0.0
    clear = cc.TrackView(1, np.array([9.0, 20.0]), np.array([0.0, 0.6]), 0.0)
    free = sc.ContingencyChecker(make_snap(u=0.5, tracks=[clear]), actuators()).certify([0.0, 0.0])
    assert free.certified


def test_allowances_only_make_the_certificate_stricter():
    wall = [(x, 9.5) for x in np.linspace(3.0, 7.0, 25)]
    snap = make_snap(u=0.6, points=wall)
    none = sc.ContingencyChecker(snap, actuators()).certify([0.0, 0.0])
    cal = sc.ContingencyChecker(snap, actuators(), tubes.OWN_TABLE, tubes.TARGET_TABLE).certify([0.0, 0.0])
    assert cal.slack <= none.slack + 1e-12


def test_certify_rejects_unnormalised_commands():
    checker = sc.ContingencyChecker(make_snap(), actuators())
    for bad in ([1.5, 0.0], [0.0, 2.0], [np.nan, 0.0], [0.0, np.inf]):
        with pytest.raises(ValueError):
            checker.certify(bad)
    with pytest.raises(ValueError):
        sc.ContingencyChecker(make_snap(), actuators(), hold_horizon_s=-1.0)


def test_projection_candidates_start_nearest_to_reference():
    grid = sc.projection_candidates([0.4, 0.9])
    assert len(grid) == len(v2.RUDDERS) * len(v2.THROTTLES) + len(v2.BRAKE_RUDDERS)
    assert np.allclose(grid[0], [0.5, 1.0])
    d = sc.distance_to([0.4, 0.9], grid)
    assert np.all(np.diff(d) >= 0.0)


def test_observed_free_fraction_classifies_rays():
    snap = SimpleNamespace(x=0.0, y=0.0, heading=0.0, u=0.0, v=0.0, r=0.0)
    ro = sc.rollout_states(snap, actuators(), np.tile((0.0, np.nan), (1, 4, 1)), *sc.WEAK)
    bearings = np.linspace(-180, 179, 360)
    far = sc.observed_free_fraction(ro, 0, (0.0, -3.0), 0.0, bearings, np.full(360, 16.0), 16.0, 1.0)
    near = sc.observed_free_fraction(ro, 0, (0.0, -3.0), 0.0, bearings, np.full(360, 1.5), 16.0, 1.0)
    assert far["free"] == 1.0 and near["free"] == 0.0
    assert sum(far[k] for k in ("free", "dead_zone", "unobserved")) == pytest.approx(1.0)


def test_tables_are_nondecreasing_and_cover_the_window():
    for table in (tubes.OWN_TABLE, tubes.TARGET_TABLE):
        values = [q for _, q in table]
        assert all(b >= a for a, b in zip(values, values[1:]))
    assert tubes.OWN_TABLE[-1][0] >= 12.0 and tubes.TARGET_TABLE[-1][0] >= 20.0


def separate_models(checker, seqs):
    weak = sc.rollout_states(checker.snap, checker.actuators, seqs, *sc.WEAK)
    strong = sc.rollout_states(checker.snap, checker.actuators, seqs, *sc.STRONG)
    sw, rw, _ = checker._slack(weak)
    ss, rs, _ = checker._slack(strong)
    braking = np.isnan(seqs[:, :, 1]).any(axis=1)
    return np.where(braking, np.minimum(sw, ss), sw), np.where(braking, np.maximum(rw, rs), rw)


@pytest.mark.parametrize("case", range(4))
def test_vectorised_models_and_staged_search_match_full_evaluation(case):
    rng = np.random.default_rng(case)
    wall = [(x, 5.0 + rng.uniform(1.0, 4.0)) for x in np.linspace(2.0, 8.0, 20)]
    target = cc.TrackView(1, np.array([rng.uniform(2, 8), 14.0]), np.array([0.0, -0.4]), math.pi)
    snap = make_snap(u=float(rng.uniform(0.2, 0.9)), points=wall, tracks=[target] if case % 2 else [])
    checker = sc.ContingencyChecker(snap, actuators())
    first = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
    seqs = sc.sequences_for(first[None])
    slack, rest, _ = checker.evaluate(seqs)
    ref_slack, ref_rest = separate_models(checker, seqs)
    assert np.allclose(slack, ref_slack) and np.allclose(rest, ref_rest)
    full_ok = bool(np.any(np.isfinite(slack) & (slack >= 0.0)))
    cert = checker.certify(first)
    assert cert.certified == full_ok
    assert np.allclose(cert.sequence[1:], sc.TAILS[cert.tail_index], equal_nan=True)


def test_batched_certification_matches_single_commands():
    wall = [(x, 7.5) for x in np.linspace(2.0, 8.0, 20)]
    snap = make_snap(u=0.7, points=wall)
    checker = sc.ContingencyChecker(snap, actuators())
    firsts = sc.projection_candidates([0.2, 0.5])[:6]
    batch = checker.certify_many(firsts)
    for first, cert in zip(firsts, batch):
        single = checker.certify(first)
        assert single.certified == cert.certified and single.tail_index == cert.tail_index
        assert single.slack == pytest.approx(cert.slack)


def test_extended_tail_family_contains_stops_and_cruise_segments():
    ext = sc.EXTENDED_TAILS
    assert ext.shape == (71, sc.DECISIONS - 1, 2)
    assert np.allclose(ext[:3], sc.TAILS[:3], equal_nan=True)
    assert np.all(np.isnan(ext[:, -1, 1])) and np.all(ext[:, -1, 0] == 0.0)
    cruising = [(np.isfinite(t[:, 1]) & (t[:, 0] == 0.0)).sum() for t in ext[3:]]
    assert max(cruising) >= 12
    snap = make_snap(u=0.5)
    cert = sc.ContingencyChecker(snap, actuators(), tails=ext).certify([0.0, 0.0])
    assert cert.certified and np.allclose(cert.sequence[1:], ext[cert.tail_index], equal_nan=True)


def test_chunked_clearance_evaluation_is_identical(monkeypatch):
    wall = [(x, 7.0) for x in np.linspace(2.0, 8.0, 60)]
    target = cc.TrackView(1, np.array([4.0, 14.0]), np.array([0.0, -0.4]), math.pi)
    snap = make_snap(u=0.7, points=wall, tracks=[target])
    checker = sc.ContingencyChecker(snap, actuators())
    seqs = sc.sequences_for(sc.projection_candidates([0.1, 0.2])[:5])
    monkeypatch.setattr(sc, "SLACK_CHUNK_COLUMNS", 10_000)
    whole = checker.evaluate(seqs)
    monkeypatch.setattr(sc, "SLACK_CHUNK_COLUMNS", 7)
    chunked = checker.evaluate(seqs)
    assert np.array_equal(whole[0], chunked[0]) and np.array_equal(whole[1], chunked[1])
    for key in whole[2]:
        assert np.array_equal(whole[2][key], chunked[2][key])
