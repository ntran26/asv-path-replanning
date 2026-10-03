"""Synthetic request/delegation checks; no environments or episodes."""
from types import SimpleNamespace

import numpy as np
import pytest

import safety_v2 as v2
import safety_v10 as v10
import safety_v14 as v14
import safety_v15 as v15
import safety_policy_prefix


def paired(clear=.1, first=np.inf, **updates):
    best = .25
    details = dict(v10_pair_checked=True, v10_proposed_first_violation=first,
                   v10_proposed_clearance=clear,
                   v10_proposed_currently_passing=bool(np.isposinf(first) and clear >= 0.),
                   selection_branch="ordinary", selection_best_clearance=best,
                   selection_floor=min(v2.TRIGGER_MARGIN_M, best-v2.ROOM_SLACK_M))
    details.update(updates)
    return details


def request(details, **options):
    filt = v15.SafetyFilterV15(**options)
    filt.last = details
    return filt._prefix_search_request()


def test_failing_current_pair_allows_zero_margin_without_ordinary_metadata():
    info = paired(-.1, first=.5)
    for key in ("selection_branch", "selection_best_clearance", "selection_floor"):
        info.pop(key)
    assert request(info) == (True, 0., "currently_failing_parent")


def test_terminal_or_negative_clearance_failure_uses_complete_paired_status():
    assert request(paired(-.01))[0:2] == (True, 0.)


@pytest.mark.parametrize("clear", [v2.TRIGGER_MARGIN_M, .24, .289740, 2.])
def test_adequate_parent_kept_even_if_prefix_could_find_larger_margin(clear):
    assert request(paired(clear)) == (False, None, "adequate_parent_clearance")


def test_ordinary_low_margin_requests_exact_existing_floor():
    info = paired(.10)
    assert request(info) == (True, info["selection_floor"], "ordinary_parent_below_trigger_margin")


def test_negative_existing_floor_is_clipped_only_at_hard_zero():
    info = paired(.02, selection_best_clearance=.03,
                  selection_floor=min(v2.TRIGGER_MARGIN_M, .03-v2.ROOM_SLACK_M))
    assert request(info)[0:2] == (True, 0.)


def test_high_best_pool_keeps_existing_trigger_floor():
    info = paired(.10, selection_best_clearance=.8, selection_floor=v2.TRIGGER_MARGIN_M)
    assert request(info)[0:2] == (True, v2.TRIGGER_MARGIN_M)


@pytest.mark.parametrize("updates", [
    {"v10_pair_checked": False}, {"v10_proposed_first_violation": None},
    {"v10_proposed_first_violation": np.nan}, {"v10_proposed_first_violation": -np.inf},
    {"v10_proposed_first_violation": -.1}, {"v10_proposed_first_violation": "inf"},
    {"v10_proposed_clearance": None}, {"v10_proposed_clearance": np.nan},
    {"v10_proposed_clearance": np.inf}, {"v10_proposed_clearance": True},
    {"v10_proposed_currently_passing": False}, {"v10_proposed_currently_passing": None},
    {"selection_branch": None}, {"selection_branch": "escape_search"},
    {"selection_floor": None}, {"selection_floor": np.nan},
    {"selection_best_clearance": np.inf}, {"selection_best_clearance": -.01},
    {"selection_floor": .06}, {"v10_policy_preserved": True},
])
def test_missing_or_inconsistent_evidence_cannot_authorize_search(updates):
    assert request(paired(**updates))[0] is False


def test_opt_out_uses_original_request_even_without_diagnostics():
    assert request({}, conditional_prefix=False) == (True, None, "default_trigger_margin")


def test_constructor_preserves_independent_options():
    filt = v15.SafetyFilterV15(target_turn_rate_deg_s=5., max_coast_s=4.,
                             geometry_prior=False, source_ownership=False,
                             prefer_persistent_estimates=False, prefix_rng_seed=42)
    assert filt.policy_prefix_search and filt.conditional_prefix
    assert np.isclose(filt.target_turn_rate_rad_s, np.radians(5.))
    assert not filt.perception.geometry_prior and not filt.perception.source_ownership
    assert not filt.prefer_persistent_estimates and filt.perception.max_coast_s == 4.
    assert not v15.SafetyFilterV15(policy_prefix_search=False).policy_prefix_search


def stub_parent(monkeypatch, info):
    output = np.array([.5, 1.], dtype=np.float32)
    plan = np.tile(output, (16, 1))
    def parent(self, env, action):
        self.last = dict(info, why="turn", checked_clearance=info.get("v10_proposed_clearance"))
        self.plan = plan.copy()
        self._observer_snapshot = SimpleNamespace(tracks=[])
        self.recovery_steps, self.uncertified_steps = 7, 2
        return output, True
    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", parent)
    return output, plan


def test_disallowed_hook_never_searches_or_changes_parent_state(monkeypatch):
    expected, plan = stub_parent(monkeypatch, paired(.3))
    def forbidden(*args, **kwargs):
        raise AssertionError("Search must not run")
    monkeypatch.setattr(safety_policy_prefix, "search", forbidden)
    filt = v15.SafetyFilterV15()
    out, changed = filt._filter(SimpleNamespace(), np.zeros(2))
    assert out is expected and changed
    np.testing.assert_array_equal(filt.plan, plan)
    assert (filt.recovery_steps, filt.uncertified_steps) == (7, 2)
    assert filt.last["v11_prefix_skip_reason"] == "adequate_parent_clearance"
    assert filt.last["v15_conditional_prefix"] is True


@pytest.mark.parametrize("conditional,expected_margin", [(True, 0.), (False, None)])
def test_hook_margin_forwarded_and_failed_search_keeps_parent(monkeypatch, conditional, expected_margin):
    expected, plan = stub_parent(monkeypatch, paired(-.1, first=.5))
    calls = []
    def search(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(plan=None, diagnostics={"accepted": False})
    monkeypatch.setattr(safety_policy_prefix, "search", search)
    filt = v15.SafetyFilterV15(conditional_prefix=conditional)
    out, changed = filt._filter(SimpleNamespace(), np.zeros(2))
    assert out is expected and changed
    assert len(calls) == 1 and calls[0]["minimum_clearance"] == expected_margin
    np.testing.assert_array_equal(filt.plan, plan)
    assert (filt.recovery_steps, filt.uncertified_steps) == (7, 2)


def test_search_disabled_matches_v14_parent_behavior(monkeypatch):
    expected, plan = stub_parent(monkeypatch, paired(.1))
    def forbidden(*args, **kwargs):
        raise AssertionError("Search/request must not run")
    monkeypatch.setattr(safety_policy_prefix, "search", forbidden)
    monkeypatch.setattr(v15.SafetyFilterV15, "_prefix_search_request", forbidden)
    action = np.zeros(2)
    baseline = v14.SafetyFilterV14(policy_prefix_search=False)
    candidate = v15.SafetyFilterV15(policy_prefix_search=False)
    a, ac = baseline._filter(SimpleNamespace(), action)
    b, bc = candidate._filter(SimpleNamespace(), action)
    assert a is b is expected and ac is bc is True
    np.testing.assert_array_equal(baseline.plan, candidate.plan)
    assert baseline.last == {k: value for k, value in candidate.last.items()
                             if k != "v15_conditional_prefix"}
