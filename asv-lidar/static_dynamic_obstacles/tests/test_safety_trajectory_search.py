"""Synthetic hard-check/search tests; no policy or environment episodes."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
import safety_v2 as v2
import safety_trajectory_search as searcher


DECISIONS = int(round(v2.HORIZON_S / cfg.UPDATE_RATE))
COMMIT = int(round(v2.COMMIT_S / cfg.UPDATE_RATE))


def seeds(*commands):
    return np.array([np.tile(c, (DECISIONS, 1)) for c in commands], dtype=float)


class Callbacks:
    def __init__(self, metric):
        self.metric = metric
        self.checked = []

    def rollout(self, snap, act, plans):
        self.checked.append(plans.copy())
        return plans

    def evaluate(self, snap, plans):
        return self.metric(plans)


def run(callbacks, initial=None, action=(.25, .4), fixed=True, traffic=False, seed=123):
    return searcher.search(SimpleNamespace(x=1.), SimpleNamespace(buffer=[.1]),
                           action, np.empty((0, DECISIONS, 2)) if initial is None else initial,
                           callbacks.rollout, callbacks.evaluate, np.random.default_rng(seed),
                           fixed_policy=fixed, traffic=traffic)


def test_fixed_policy_exact_commit_bounds_blocks_and_finite_budget():
    cb = Callbacks(lambda p: (np.full(len(p), np.inf), np.full(len(p), .3)))
    original = seeds((-.9, .8), (.7, -.5))
    saved = original.copy()
    result = run(cb, original)
    assert result.plan is not None
    for batch in cb.checked:
        np.testing.assert_array_equal(batch[:, :COMMIT],
                                      np.broadcast_to([.25, .4], batch[:, :COMMIT].shape))
        assert np.isfinite(batch).all()
        assert (np.abs(batch) <= 1).all()
        np.testing.assert_array_equal(batch[:, ::2], batch[:, 1::2])
    assert [len(p) for p in cb.checked] == [2, 64, 64, 64]
    assert result.diagnostics["evaluated_plans"] == 194
    assert any(np.array_equal(result.plan, plan) for batch in cb.checked for plan in batch)
    np.testing.assert_array_equal(original, saved)


@pytest.mark.parametrize("first,clear", [(1., 10.), (np.inf, .149999),
                                        (-np.inf, .9), (np.nan, .9), (np.inf, np.nan)])
def test_unsafe_or_insufficient_margin_never_returns_plan(first, clear):
    cb = Callbacks(lambda p: (np.full(len(p), first), np.full(len(p), clear)))
    result = run(cb)
    assert result.plan is None
    assert result.clearance == -np.inf
    assert not result.diagnostics["accepted"]
    assert result.diagnostics["accepted_plans"] == 0
    assert result.diagnostics["evaluated_plans"] == 192


def test_exact_existing_margin_is_accepted():
    cb = Callbacks(lambda p: (np.full(len(p), np.inf),
                              np.full(len(p), v2.TRIGGER_MARGIN_M)))
    result = run(cb)
    assert result.plan is not None
    assert result.clearance == v2.TRIGGER_MARGIN_M


def test_free_commit_and_warm_tail_are_preserved_exactly_when_checked_best():
    initial = seeds((.25, .4))
    # A shifted warm plan can change on odd decisions beyond the commit.
    initial[0, 1] = [.9, .8]
    initial[0, 5:8] = [-.7, -.3]
    expected = initial[0].copy()
    expected[:COMMIT] = expected[0]
    cb = Callbacks(lambda p: (np.full(len(p), np.inf), np.full(len(p), .25)))
    result = run(cb, initial, fixed=False)
    np.testing.assert_array_equal(result.plan, expected)
    assert result.diagnostics["policy_distance"] == 0.
    for batch in cb.checked:
        np.testing.assert_array_equal(batch[:, :COMMIT],
                                      np.repeat(batch[:, :1], COMMIT, axis=1))
    assert not np.array_equal(initial[0], expected)


def test_feasible_policy_deviation_precedes_extra_clearance():
    initial = seeds((.25, .4), (-1., -1.))
    cb = Callbacks(lambda p: (np.full(len(p), np.inf),
                              .2 + 5 * np.abs(p[:, 0, 0] - .25)))
    result = run(cb, initial, fixed=False)
    np.testing.assert_array_equal(result.plan, initial[0])
    assert result.clearance == .2


def test_brake_marker_is_not_averaged_into_a_real_out_of_bounds_command():
    initial = seeds((0., np.nan), (.5, .3))

    def metric(p):
        brake = np.isnan(p[:, :, 1]).any(axis=1)
        return np.where(brake, np.inf, 1.), np.where(brake, .4, -.2)

    cb = Callbacks(metric)
    result = run(cb, initial, fixed=False, traffic=True)
    assert result.plan is not None
    assert np.isnan(result.plan[:, 1]).any()
    for batch in cb.checked:
        assert np.isfinite(batch[:, :, 0]).all()
        assert (np.abs(batch[np.isfinite(batch)]) <= 1.).all()
        np.testing.assert_array_equal(np.isnan(batch[:, :COMMIT, 1]),
                                      np.repeat(np.isnan(batch[:, :1, 1]), COMMIT, axis=1))


def test_no_traffic_excludes_brake_seeds_and_never_samples_astern():
    cb = Callbacks(lambda p: (np.full(len(p), np.inf), np.full(len(p), .2)))
    result = run(cb, seeds((0., np.nan), (.5, .3)), traffic=False)
    assert result.diagnostics["dropped_brake_seeds"] == 1
    assert result.diagnostics["seed_plans"] == 1
    assert all(np.isfinite(p).all() for p in cb.checked)


def test_local_rng_replay_global_rng_and_inputs_unchanged():
    cb1 = Callbacks(lambda p: (np.full(len(p), np.inf), p[:, 4, 0]))
    cb2 = Callbacks(cb1.metric)
    global_before = copy.deepcopy(np.random.get_state())
    first, second = run(cb1), run(cb2)
    np.testing.assert_array_equal(first.plan, second.plan)
    for left, right in zip(cb1.checked, cb2.checked):
        np.testing.assert_array_equal(left, right)
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]


def test_search_can_find_powered_turn_then_counterturn_beyond_seed_bank():
    # A synthetic hard constraint requires opposite later turns at positive
    # throttle; all straight seed plans fail. This tests trajectory freedom,
    # not vessel dynamics or a collision-avoidance guarantee.
    def metric(p):
        clear = np.minimum.reduce([p[:, 4, 0], -p[:, 12, 0],
                                   p[:, 4, 1], p[:, 12, 1]])
        return np.where(clear >= 0., np.inf, 1.), clear

    cb = Callbacks(metric)
    result = run(cb, seeds((0., .5)), action=(0., .5))
    assert result.plan is not None
    assert result.clearance >= v2.TRIGGER_MARGIN_M
    assert result.plan[4, 0] > 0. and result.plan[12, 0] < 0.
    assert result.plan[4, 1] > 0. and result.plan[12, 1] > 0.


def test_invalid_rng_and_bad_seed_layout_rejected_before_callbacks():
    cb = Callbacks(lambda p: None)
    with pytest.raises(TypeError, match="Generator"):
        searcher.search(None, None, [0., 0.], [], cb.rollout, cb.evaluate,
                        np.random, fixed_policy=True, traffic=False)
    with pytest.raises(ValueError, match="shape"):
        run(cb, np.zeros((2, DECISIONS - 1, 2)))
    assert not cb.checked
