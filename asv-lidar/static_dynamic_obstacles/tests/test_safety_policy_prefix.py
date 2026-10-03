"""Standalone prefix-search contracts using synthetic checker callbacks only."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
import safety_v2 as v2
import safety_policy_prefix as prefix
import safety_trajectory_search as existing


DECISIONS = int(np.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))
ACTION = np.array([.25, .4], dtype=np.float32)


class Callbacks:
    def __init__(self, metric):
        self.metric, self.checked = metric, []
        self.snap = SimpleNamespace(x=1., tracks=[])
        self.act = SimpleNamespace(buffer=[.1])

    def rollout(self, snap, act, plans):
        assert snap is self.snap and act is self.act
        self.checked.append(plans.copy())
        return plans

    def evaluate(self, snap, plans):
        assert snap is self.snap
        return self.metric(plans)


def run(cb, backup=None, action=ACTION, traffic=False, seed=123, **kwargs):
    return prefix.search(cb.snap, cb.act, action, backup, cb.rollout, cb.evaluate,
                         np.random.default_rng(seed), traffic=traffic, **kwargs)


def test_exact_one_decision_prefix_and_full_horizon_with_bounded_total_work():
    cb = Callbacks(lambda p: (np.full(len(p), np.inf), np.full(len(p), .3)))
    result = run(cb)
    assert result.plan is not None
    assert result.diagnostics["evaluated_plans"] == prefix.MAX_PLANS == 192
    assert sum(map(len, cb.checked)) == 192 and len(cb.checked) == 4
    for batch in cb.checked:
        assert batch.shape[1:] == (DECISIONS, 2)
        np.testing.assert_array_equal(batch[:, 0], np.tile(ACTION, (len(batch), 1)))
        assert np.isfinite(batch).all() and (np.abs(batch) <= 1).all()
    for batch in cb.checked[1:]:
        np.testing.assert_array_equal(batch[:, 1:-1:2], batch[:, 2::2])
        assert np.any(batch[:, 1] != ACTION)
    assert result.diagnostics["fixed_policy_decisions"] == 1
    assert result.diagnostics["fixed_policy_seconds"] == cfg.UPDATE_RATE
    assert any(np.array_equal(result.plan, plan) for batch in cb.checked for plan in batch)


def test_existing_commit_restriction_can_hide_a_feasible_one_decision_backup():
    assert v2.COMMIT_S == 2 * cfg.UPDATE_RATE

    def metric(plans):
        # Synthetic constraint requires an immediate next-decision turn.
        room = -plans[:, 1, 0] - .5
        return np.where(room >= 0., np.inf, .75), room

    old, new = Callbacks(metric), Callbacks(metric)
    initial = np.tile((-1., 1.), (1, DECISIONS, 1))
    rejected = existing.search(old.snap, old.act, ACTION, initial, old.rollout, old.evaluate,
                               np.random.default_rng(123), fixed_policy=True, traffic=False)
    found = run(new, initial[0])
    assert rejected.plan is None
    assert found.plan is not None and found.clearance >= v2.TRIGGER_MARGIN_M
    np.testing.assert_array_equal(found.plan[0], ACTION)
    assert found.plan[1, 0] <= -.65


def test_warm_seed_is_checked_exactly_without_rounding_odd_tail_changes():
    warm = np.tile((.13, .37), (DECISIONS - 2, 1))
    warm[1:5] = [[-.81, .72], [-.34, .62], [.91, .11], [.44, -.22]]
    expected = prefix.replace_first_action(warm, ACTION)
    saved = warm.copy()

    def metric(plans):
        exact = np.all(plans == expected, axis=(1, 2))
        return np.where(exact, np.inf, .5), np.where(exact, .3, -.1)

    cb = Callbacks(metric)
    result = run(cb, warm)
    np.testing.assert_array_equal(result.plan, expected)
    np.testing.assert_array_equal(cb.checked[0][0], expected)
    np.testing.assert_array_equal(warm, saved)
    np.testing.assert_array_equal(result.plan[-2:], [[0., .37], [0., .37]])


@pytest.mark.parametrize("first,clear", [(1., 10.), (np.inf, .149999),
                                        (-np.inf, .9), (np.nan, .9), (np.inf, np.nan)])
def test_failed_unknown_or_below_margin_never_returns_plan(first, clear):
    cb = Callbacks(lambda p: (np.full(len(p), first), np.full(len(p), clear)))
    result = run(cb)
    assert result.plan is None and result.clearance == -np.inf
    assert not result.diagnostics["accepted"]
    assert result.diagnostics["evaluated_plans"] == 192


def test_default_margin_boundary_passes_and_full_terminal_constraint_is_checked():
    def metric(plans):
        room = np.where(plans[:, -1, 1] > .9, v2.TRIGGER_MARGIN_M, -.1)
        return np.where(room >= 0., np.inf, v2.HORIZON_S), room

    cb = Callbacks(metric)
    result = run(cb)
    assert result.plan is not None and result.clearance == v2.TRIGGER_MARGIN_M
    assert result.plan[-1, 1] > .9


def test_traffic_brakes_remain_nan_in_future_only_and_preserve_input_backup():
    warm = np.tile((-.2, np.nan), (DECISIONS, 1))
    saved = warm.copy()

    def metric(plans):
        brake = np.isnan(plans[:, 1:, 1]).any(axis=1)
        return np.where(brake, np.inf, .5), np.where(brake, .4, -.1)

    cb = Callbacks(metric)
    result = run(cb, warm, traffic=True)
    assert result.plan is not None and np.isnan(result.plan[1:, 1]).any()
    for batch in cb.checked:
        assert np.isfinite(batch[:, 0]).all() and np.isfinite(batch[:, :, 0]).all()
        assert (np.abs(batch[np.isfinite(batch)]) <= 1.).all()
    np.testing.assert_array_equal(warm, saved)
    assert result.diagnostics["evaluated_plans"] == 192


def test_no_traffic_drops_remaining_astern_backup_and_never_samples_braking():
    cb = Callbacks(lambda p: (np.full(len(p), np.inf), np.full(len(p), .2)))
    warm = np.tile((0., np.nan), (DECISIONS, 1))
    result = run(cb, warm)
    assert result.diagnostics["dropped_brake_backup"]
    assert not result.diagnostics["backup_seed_checked"]
    assert all(np.isfinite(batch).all() for batch in cb.checked)


def test_traffic_gate_applies_after_replacing_only_current_brake_command():
    cb = Callbacks(lambda p: (np.full(len(p), np.inf), np.full(len(p), .2)))
    warm = np.tile((0., .7), (DECISIONS, 1))
    warm[0, 1] = np.nan
    result = run(cb, warm)
    assert not result.diagnostics["dropped_brake_backup"]
    assert result.diagnostics["backup_seed_checked"]
    assert np.isfinite(cb.checked[0]).all()
    np.testing.assert_array_equal(cb.checked[0][0, 1:], warm[1:])


def test_deterministic_local_rng_leaves_global_rng_state_and_inputs_unchanged():
    first = Callbacks(lambda p: (np.full(len(p), np.inf), p[:, -1, 0]))
    second = Callbacks(first.metric)
    global_before = copy.deepcopy(np.random.get_state())
    snap_before, act_before = copy.deepcopy(vars(first.snap)), copy.deepcopy(vars(first.act))
    saved_action = ACTION.copy()
    a, b = run(first), run(second)
    np.testing.assert_array_equal(a.plan, b.plan)
    for left, right in zip(first.checked, second.checked):
        np.testing.assert_array_equal(left, right)
    after = np.random.get_state()
    assert global_before[0] == after[0] and global_before[2:] == after[2:]
    np.testing.assert_array_equal(global_before[1], after[1])
    assert vars(first.snap) == snap_before and vars(first.act) == act_before
    np.testing.assert_array_equal(ACTION, saved_action)


def test_random_tail_can_find_powered_counterturn_beyond_primitive_family():
    def metric(plans):
        room = np.minimum.reduce([plans[:, 3, 0], -plans[:, 11, 0],
                                  plans[:, 3, 1], plans[:, 11, 1]])
        return np.where(room >= 0., np.inf, 1.), room

    cb = Callbacks(metric)
    result = run(cb, action=[0., .5])
    assert result.plan is not None and result.clearance >= v2.TRIGGER_MARGIN_M
    first, clear = metric(cb.checked[0])
    assert not np.any(np.isposinf(first) & (clear >= v2.TRIGGER_MARGIN_M))


def test_diagnostics_separate_hard_passing_clearance_from_unusable_positive_metric():
    cb = Callbacks(lambda p: (np.full(len(p), .5), np.full(len(p), .9)))
    result = run(cb)
    assert result.diagnostics["maximum_clearance"] == .9
    assert result.diagnostics["maximum_hard_passing_clearance"] == -np.inf
    assert result.diagnostics["hard_passing_plans"] == 0


@pytest.mark.parametrize("margin", [-.1, np.inf, np.nan])
def test_invalid_requested_margin_rejected_before_callbacks(margin):
    cb = Callbacks(lambda p: None)
    with pytest.raises(ValueError, match="clearance"):
        run(cb, minimum_clearance=margin)
    assert not cb.checked


def test_explicit_zero_margin_requires_all_other_hard_checks():
    cb = Callbacks(lambda p: (np.full(len(p), np.inf), np.zeros(len(p))))
    result = run(cb, minimum_clearance=0.)
    assert result.plan is not None and result.clearance == 0.
    assert result.diagnostics["minimum_clearance"] == 0.


def test_bad_rng_action_backup_or_evaluator_shape_is_rejected():
    cb = Callbacks(lambda p: None)
    with pytest.raises(TypeError, match="Generator"):
        prefix.search(None, None, ACTION, None, cb.rollout, cb.evaluate, np.random, traffic=False)
    with pytest.raises(ValueError, match="Policy action"):
        run(cb, action=[0., np.nan])
    with pytest.raises(ValueError, match="Backup"):
        run(cb, backup=np.zeros((2, 3)))
    assert not cb.checked
    cb.metric = lambda p: (np.array([np.inf]), np.array([.5]))
    with pytest.raises(ValueError, match="per plan"):
        run(cb)
