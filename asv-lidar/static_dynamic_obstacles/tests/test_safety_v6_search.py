"""V6 search/adapter integration with mocked checks; no policy episodes."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
import safety_trajectory_search as searcher
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v6 as v6


@pytest.fixture
def make_filter(monkeypatch):
    for name in ("COUNTERSTEER_RECOVERY", "RETAIN_NOMINAL_PLAN", "DUAL_BRAKE_PREDICTION",
                 "FREE_SPACE_MEMORY", "MODEL_EGO_OBSERVER"):
        monkeypatch.setattr(v4, name, False)
    for name in ("CALIBRATED_BRAKING", "FITTED_HULL_PERCEPTION", "PROVISIONAL_TRACKS"):
        monkeypatch.setattr(v6, name, False)
    monkeypatch.setattr(v6, "TRAJECTORY_SEARCH", True)
    monkeypatch.setattr(v3, "rollout_seq", lambda snap, act, seq: seq)

    def make(cls=v6.SafetyFilterV6, *, stored=False, traffic=False):
        f = cls()
        snap = cc.Snapshot(5., 5., 0., 1., 0., 0., np.array([0., 1.]),
            np.array([1., 0.]), np.array([5., 5.]), 0., 0., 20., np.empty((0, 2)))
        if traffic:
            snap.tracks = [SimpleNamespace(position=np.array([5., 7.]))]
        monkeypatch.setattr(f.perception, "snapshot", lambda env: snap)
        monkeypatch.setattr(f, "_threat_in_reach", lambda snap: True)
        monkeypatch.setattr(f, "_rejoin", lambda snap: np.zeros(2))
        issued = []
        monkeypatch.setattr(f.actuators, "issue", lambda env, rudder: issued.append(rudder))
        if stored:
            f.plan = v3.plan_for((.75, .1), (-1., 0.))
        return f, SimpleNamespace(_v2_brake=False), issued
    return make


def failed(fixed):
    return searcher.SearchResult(None, -np.inf, {"fixed_policy": fixed, "accepted": False})


def accepted(plan, fixed):
    return searcher.SearchResult(plan, .4, {"fixed_policy": fixed, "accepted": True})


def unsafe(snap, seq):
    return np.zeros(len(seq)), np.full(len(seq), -1.)


def test_adequate_nominal_policy_skips_search_and_issues_once(make_filter, monkeypatch):
    f, env, issued = make_filter()
    monkeypatch.setattr(f, "_evaluate", lambda snap, seq: (np.full(len(seq), np.inf),
                        np.full(len(seq), v2.TRIGGER_MARGIN_M)))
    monkeypatch.setattr(searcher, "search", lambda *a, **kw: pytest.fail("adequate policy searched"))
    action = np.array([.2, .1])
    out, changed = f.filter(env, action)
    np.testing.assert_array_equal(out, action.astype(np.float32))
    assert not changed and f.last["why"] == "nominal" and len(issued) == 1


@pytest.mark.parametrize("why", ["turn", "continue", "brake", "no escape", "hold back", "last certificate", "narrow policy"])
def test_failed_search_preserves_original_output_and_plan(make_filter, monkeypatch, why):
    stored, traffic = why in ("continue", "last certificate"), why == "brake"
    f, env, issued = make_filter(stored=stored, traffic=traffic)
    reference, ref_env, ref_issued = make_filter(v4.SafetyFilterV4, stored=stored, traffic=traffic)
    nrec = sum(traffic or t is not v2.BRAKE for _, t in v2.RECOVERY)
    calls = []

    def search(*args, fixed_policy, **kwargs):
        calls.append(fixed_policy)
        return failed(fixed_policy)

    def evaluate(snap, seq):
        first, clear = unsafe(snap, seq)
        index = None
        if why == "turn":
            index = 3 * nrec + 1
        elif why == "continue":
            index = len(seq) - 1
        elif why == "brake":
            index = (2 + len(v2.RUDDERS) * len(v2.THROTTLES)) * nrec
        elif why == "hold back":
            first[3 * nrec] = 2. + v2.HOLD_BACK_GAIN_S
        elif why == "narrow policy":
            first[0], clear[0] = np.inf, .05
        if index is not None:
            first[index], clear[index] = np.inf, .8
        return first, clear

    monkeypatch.setattr(searcher, "search", search)
    for filter_ in (f, reference):
        monkeypatch.setattr(filter_, "_evaluate", evaluate)
    action = np.array([.2, .1])
    out, changed = f.filter(env, action)
    expected, ref_changed = reference.filter(ref_env, action)
    np.testing.assert_array_equal(out, expected)
    assert changed == ref_changed and env._v2_brake == ref_env._v2_brake
    assert f.mode == reference.mode and f.last["why"] == reference.last["why"]
    assert f.uncertified_steps == reference.uncertified_steps
    assert f.recovery_steps == reference.recovery_steps
    if reference.plan is None:
        assert f.plan is None
    else:
        np.testing.assert_array_equal(f.plan, reference.plan)
    assert calls[0] is True and len(issued) == len(ref_issued) == 1


@pytest.mark.parametrize("initial_mode", ["nominal", "recovery"])
def test_searched_policy_keeps_exact_checked_plan_and_first_command(make_filter, monkeypatch, initial_mode):
    f, env, issued = make_filter(stored=True)
    f.mode = initial_mode
    f.recovery_steps = 7
    f.uncertified_steps = 2
    monkeypatch.setattr(f, "_evaluate", unsafe)
    action = np.array([.2, .1])
    plan = v3.plan_for(action, (-.7, .35))
    plan[-3:] = [.63, -.24]  # Preserve an arbitrary exact checked tail.
    original_plan = plan.copy()
    calls = []

    def search(snap, act, policy, seeds, rollout, evaluate, rng, *, fixed_policy, traffic):
        calls.append(fixed_policy)
        np.testing.assert_array_equal(policy, action)
        assert rollout.__self__ is f and rollout.__func__ is v6.SafetyFilterV6._rollout
        assert rng is f._search_rng
        return accepted(plan, True)

    monkeypatch.setattr(searcher, "search", search)
    out, changed = f.filter(env, action)
    np.testing.assert_array_equal(out, action.astype(np.float32))
    np.testing.assert_array_equal(f.plan, original_plan)
    np.testing.assert_array_equal(plan, original_plan)
    assert not np.shares_memory(f.plan, plan)
    assert not changed and not env._v2_brake
    assert f.mode == initial_mode and f.last["why"] == "searched policy"
    assert f.uncertified_steps == 0 and f.recovery_steps == 7 + (initial_mode == "recovery")
    assert calls == [True] and len(issued) == 1


@pytest.mark.parametrize("original_clearance", [.001, .10, v2.TRIGGER_MARGIN_M - 1e-6])
def test_free_search_runs_for_hard_safe_but_narrow_original(make_filter, monkeypatch, original_clearance):
    f, env, issued = make_filter()
    nrec = sum(t is not v2.BRAKE for _, t in v2.RECOVERY)
    def evaluate(snap, seq):
        first, clear = unsafe(snap, seq)
        first[3 * nrec], clear[3 * nrec] = np.inf, original_clearance
        return first, clear
    monkeypatch.setattr(f, "_evaluate", evaluate)
    plan = v3.plan_for((.7, .3), (-.8, .2))
    calls = []
    def search(*args, fixed_policy, **kwargs):
        calls.append(fixed_policy)
        return failed(True) if fixed_policy else accepted(plan, False)
    monkeypatch.setattr(searcher, "search", search)
    out, changed = f.filter(env, np.array([.2, .1]))
    assert calls == [True, False] and changed and len(issued) == 1
    assert f.last["any_safe"] and f.last["why"] == "searched escape"
    np.testing.assert_array_equal(out, plan[0].astype(np.float32))
    np.testing.assert_array_equal(f.plan, plan)


@pytest.mark.parametrize("original_clearance", [v2.TRIGGER_MARGIN_M, .8])
@pytest.mark.parametrize("continuation", [False, True])
def test_free_search_skipped_when_original_or_continuation_has_room(make_filter, monkeypatch, original_clearance, continuation):
    f, env, issued = make_filter(stored=continuation)
    nrec = sum(t is not v2.BRAKE for _, t in v2.RECOVERY)
    def evaluate(snap, seq):
        first, clear = unsafe(snap, seq)
        index = len(seq)-1 if continuation else 3*nrec
        first[index], clear[index] = np.inf, original_clearance
        return first, clear
    monkeypatch.setattr(f, "_evaluate", evaluate)
    calls = []
    def search(*args, fixed_policy, **kwargs):
        calls.append(fixed_policy)
        assert fixed_policy, "Free search must not replace a sufficiently clear original"
        return failed(True)
    monkeypatch.setattr(searcher, "search", search)
    f.filter(env, np.array([.2, .1]))
    assert calls == [True] and len(issued) == 1


def test_nan_brake_stays_in_stored_plan_but_uses_transport_throttle_once(make_filter, monkeypatch):
    f, env, issued = make_filter(traffic=True)
    monkeypatch.setattr(f, "_evaluate", unsafe)
    action = np.array([-1., -1.])
    plan = v3.plan_for((-1., np.nan), (.5, .2))
    original = plan.copy()
    monkeypatch.setattr(searcher, "search", lambda *a, fixed_policy, **kw:
                        failed(True) if fixed_policy else accepted(plan, False))
    out, changed = f.filter(env, action)
    np.testing.assert_array_equal(out, action.astype(np.float32))
    np.testing.assert_array_equal(f.plan, original)
    np.testing.assert_array_equal(plan, original)
    assert env._v2_brake and changed and f.last["brake"]
    assert issued == [-1.]


def test_search_uses_independent_per_filter_generator_and_no_global_rng(make_filter, monkeypatch):
    global_before = copy.deepcopy(np.random.get_state())
    filters = [make_filter() for _ in range(2)]
    draws = []
    def search(*args, fixed_policy, **kwargs):
        rng = args[6]
        assert isinstance(rng, np.random.Generator)
        draws.append((rng, rng.normal(size=4)))
        return failed(fixed_policy)
    monkeypatch.setattr(searcher, "search", search)
    for f, env, issued in filters:
        monkeypatch.setattr(f, "_evaluate", unsafe)
        f.filter(env, np.array([.2, .1]))
        assert len(issued) == 1
    assert draws[0][0] is draws[1][0] is filters[0][0]._search_rng
    assert draws[2][0] is draws[3][0] is filters[1][0]._search_rng
    assert draws[0][0] is not draws[2][0]
    np.testing.assert_array_equal(draws[0][1], draws[2][1])
    np.testing.assert_array_equal(draws[1][1], draws[3][1])
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0] and global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(global_before[1], global_after[1])


@pytest.mark.parametrize("hull_enabled", [False, True])
@pytest.mark.parametrize("evidence_enabled", [False, True])
def test_provisional_hook_wraps_selected_base_perception(make_filter, monkeypatch, hull_enabled, evidence_enabled):
    import safety_provisional_tracks as provisional
    import safety_hull_perception as hull
    created = []
    class Hull:
        def __init__(self, **kwargs):
            self.options = kwargs
            created.append(self)
    class Provisional:
        def __init__(self, base, *, require_motion_evidence):
            self.base = base
            self.require_motion_evidence = require_motion_evidence
    monkeypatch.setattr(v6, "FITTED_HULL_PERCEPTION", hull_enabled)
    monkeypatch.setattr(v6, "PROVISIONAL_TRACKS", True)
    monkeypatch.setattr(v6, "PROVISIONAL_MOTION_EVIDENCE", evidence_enabled)
    monkeypatch.setattr(hull, "HullSafetyPerception", Hull)
    monkeypatch.setattr(provisional, "ProvisionalTrackPerception", Provisional)
    f = v6.SafetyFilterV6()
    assert isinstance(f.perception, Provisional)
    assert f.perception.require_motion_evidence is evidence_enabled
    if hull_enabled:
        assert f.perception.base is created[0]
        assert created[0].options["fitted_track_centre"] == v6.FITTED_TRACK_CENTRE
    else:
        assert isinstance(f.perception.base, cc.Perception) and not created
