"""Synthetic prefix-request and exact-selection diagnostics; no episodes."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import safety_policy_prefix as prefix
import safety_trajectory_search as searcher
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v10 as v10
import safety_v11 as v11
from test_safety_v11 import case, POLICY, OVERRIDE, Commands, assert_parent
from test_safety_v6_search import make_filter, failed, accepted, unsafe


def test_default_hook_keeps_searcher_default_without_copying_threshold():
    filt = v11.SafetyFilterV11(policy_prefix_search=True)
    assert filt._prefix_search_request() == (True, None, "default_trigger_margin")


@pytest.mark.parametrize("traffic", [False, True])
def test_default_forwards_none_exact_policy_backup_rng_and_precommand_history(case, monkeypatch, traffic):
    if traffic:
        case.snap.tracks = [SimpleNamespace(position=np.array([0., 1.]))]
    seen = []

    def search(snap, actuators, policy, backup, rollout, evaluate, rng, **options):
        assert snap is case.snap and actuators is not case.f.actuators
        assert actuators.values == [.1]
        assert case.f.actuators.values == [.1, float(OVERRIDE[0])]
        np.testing.assert_array_equal(policy, POLICY)
        assert policy.dtype == np.float32 and backup is case.f.plan
        assert rollout == case.f._rollout and evaluate == case.f._evaluate
        assert rng is case.f._prefix_rng
        assert options == {"traffic": traffic, "minimum_clearance": None}
        seen.append(True)
        return prefix.PrefixSearchResult(None, -np.inf, {"accepted": False})

    monkeypatch.setattr(prefix, "search", search)
    assert_parent(case, case.f.filter(case.env, POLICY))
    assert seen == [True]
    assert case.f.last["v11_prefix_request_checked"] is True
    assert case.f.last["v11_prefix_minimum_clearance"] is None
    assert case.f.last["v11_prefix_request_reason"] == "default_trigger_margin"


def test_default_hook_preserves_real_cem_samples_plan_and_rng_state(case):
    direct_rng = np.random.default_rng()
    direct_rng.bit_generator.state = copy.deepcopy(case.f._prefix_rng.bit_generator.state)
    direct_batches = []

    def direct_rollout(snap, actuators, sequences):
        assert snap is case.snap and actuators.values == [.1]
        direct_batches.append(sequences.copy())
        return sequences

    # This is the pre-hook call signature, with minimum_clearance omitted.
    expected = prefix.search(case.snap, Commands(), POLICY, case.plan,
                             direct_rollout, case.f._evaluate, direct_rng, traffic=False)
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and len(case.calls) == len(direct_batches)
    for actual_batch, old_batch in zip(case.calls, direct_batches):
        np.testing.assert_array_equal(actual_batch, old_batch)
    np.testing.assert_array_equal(case.f.plan, expected.plan)
    assert case.f.last["checked_clearance"] == expected.clearance
    assert case.f._prefix_rng.bit_generator.state == direct_rng.bit_generator.state
    assert case.f.last["policy_prefix_search"] == expected.diagnostics


@pytest.mark.parametrize("minimum", [0., .06476591174532181, .15])
def test_custom_minimum_is_forwarded_without_rounding_or_new_actuator_issue(case, monkeypatch, minimum):
    monkeypatch.setattr(case.f, "_prefix_search_request", lambda: (True, minimum, "current_floor"))

    def search(*args, minimum_clearance, **kwargs):
        assert minimum_clearance == minimum
        return prefix.PrefixSearchResult(None, -np.inf, {"accepted": False})

    monkeypatch.setattr(prefix, "search", search)
    assert_parent(case, case.f.filter(case.env, POLICY))
    assert case.f.last["v11_prefix_minimum_clearance"] == minimum
    assert case.f.last["v11_prefix_request_reason"] == "current_floor"


@pytest.mark.parametrize("minimum,preserved", [(None, False), (.06476591174532181, True), (0., True)])
def test_requested_minimum_controls_real_acceptance_with_hard_checks_intact(case, monkeypatch, minimum, preserved):
    monkeypatch.setattr(case.f, "_prefix_search_request", lambda: (True, minimum, "test_floor"))
    monkeypatch.setattr(case.f, "_evaluate", lambda snap, plans: (
        np.full(len(plans), np.inf), np.full(len(plans), .10339112412476607)))
    result = case.f.filter(case.env, POLICY)
    if preserved:
        np.testing.assert_array_equal(result[0], POLICY)
        assert result[1] is False and not case.env._v2_brake
        np.testing.assert_array_equal(case.f.plan[0], POLICY)
        assert case.f.actuators.values == [.1, float(POLICY[0])]
        assert case.f.recovery_steps == case.f.uncertified_steps == 0
        assert case.f.last["v11_policy_prefix_preserved"]
    else:
        assert_parent(case, result)


@pytest.mark.parametrize("first,clear", [(.5, .1), (np.nan, .1), (np.inf, -.001)])
def test_zero_minimum_still_refuses_a_failed_whole_plan(case, monkeypatch, first, clear):
    monkeypatch.setattr(case.f, "_prefix_search_request", lambda: (True, 0., "failed_pair"))
    monkeypatch.setattr(case.f, "_evaluate", lambda snap, plans: (
        np.full(len(plans), first), np.full(len(plans), clear)))
    assert_parent(case, case.f.filter(case.env, POLICY))
    assert case.f.last["v11_policy_prefix_search_checked"]
    assert not case.f.last["v11_policy_prefix_preserved"]


def test_refused_request_keeps_exact_parent_objects_counters_brake_and_rng(case, monkeypatch):
    captured = {}
    original_parent = v10.SafetyFilterV10._filter

    def parent(self, env, action):
        result = original_parent(self, env, action)
        captured.update(out=result[0], plan=self.plan, actuators=self.actuators)
        return result

    def request():
        assert case.f.last["why"] == "last certificate"
        assert case.f.actuators.values == [.1, float(OVERRIDE[0])]
        return False, None, "uncertain_current_check"

    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", parent)
    monkeypatch.setattr(case.f, "_prefix_search_request", request)
    monkeypatch.setattr(prefix, "search", lambda *a, **k: pytest.fail("refused request ran search"))
    before_rng = copy.deepcopy(case.f._prefix_rng.bit_generator.state)
    result = case.f.filter(case.env, POLICY)
    assert_parent(case, result)
    assert result[0] is captured["out"] and case.f.plan is captured["plan"]
    assert case.f.actuators is captured["actuators"]
    assert case.f._prefix_rng.bit_generator.state == before_rng
    assert not case.calls and not case.f.last["v11_policy_prefix_search_checked"]
    assert case.f.last["v11_prefix_request_checked"]
    assert case.f.last["v11_prefix_skip_reason"] == "uncertain_current_check"


def test_refused_no_escape_request_leaves_unmodified_policy_and_no_plan(case, monkeypatch):
    def parent(self, env, action):
        self._observer_snapshot = case.snap
        self.plan, self.mode = None, "nominal"
        self.recovery_steps, self.uncertified_steps = 0, 0
        self.last = {"why": "no escape", "checked_clearance": -np.inf}
        self.actuators.issue(env, float(action[0]))
        env._v2_brake = False
        return action, False

    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", parent)
    monkeypatch.setattr(case.f, "_prefix_search_request", lambda: (False, None, "no_paired_check"))
    monkeypatch.setattr(prefix, "search", lambda *a, **k: pytest.fail("refused request ran search"))
    out, changed = case.f.filter(case.env, POLICY)
    np.testing.assert_array_equal(out, POLICY)
    assert not changed and not case.env._v2_brake and case.f.plan is None
    assert case.f.mode == "nominal" and case.f.recovery_steps == case.f.uncertified_steps == 0
    assert case.f.actuators.values == [.1, float(POLICY[0])]
    assert case.f.last["why"] == "no escape" and not case.calls


@pytest.mark.parametrize("skip", ["disabled", "rate_limit", "missing_snapshot", "policy_already_preserved"])
def test_request_hook_cannot_bypass_existing_search_preconditions(case, monkeypatch, skip):
    original_parent = v10.SafetyFilterV10._filter

    def parent(self, env, action):
        out, changed = original_parent(self, env, action)
        if skip == "missing_snapshot":
            self._observer_snapshot = None
        if skip == "policy_already_preserved":
            self.last["why"] = "policy margin dominates"
            return action, False
        return out, changed

    monkeypatch.setattr(v10.SafetyFilterV10, "_filter", parent)
    case.f.policy_prefix_search = skip != "disabled"
    case.env.command_rate_limit = skip == "rate_limit"
    monkeypatch.setattr(case.f, "_prefix_search_request", lambda: pytest.fail("premature request"))
    case.f.filter(case.env, POLICY)
    assert not case.f.last["v11_prefix_request_checked"]
    assert not case.f.last["v11_policy_prefix_search_checked"]


def test_v6_exposes_exact_current_pool_floor_including_checked_continuation(make_filter, monkeypatch):
    f, env, _ = make_filter(stored=True)
    nrec = sum(t is not v2.BRAKE for _, t in v2.RECOVERY)

    def evaluate(snap, plans):
        first, clear = unsafe(snap, plans)
        first[3*nrec], clear[3*nrec] = np.inf, .18182786571257586
        first[-1], clear[-1] = np.inf, .2647659117453218
        return first, clear

    monkeypatch.setattr(f, "_evaluate", evaluate)
    monkeypatch.setattr(searcher, "search", lambda *a, fixed_policy, **k: failed(fixed_policy))
    f.filter(env, np.array([.2, .1]))
    assert f.last["selection_branch"] == "ordinary"
    assert f.last["selection_best_clearance"] == .2647659117453218
    assert f.last["selection_floor"] == .06476591174532181
    assert f.last["selection_best_clearance"] != f.last["best_margin"]


@pytest.mark.parametrize("next_branch", ["idle", "nominal", "searched policy", "searched escape",
                                        "brake", "last certificate", "no escape"])
def test_v6_ordinary_selection_diagnostics_do_not_leak_into_later_branches(make_filter, monkeypatch, next_branch):
    traffic = next_branch == "brake"
    f, env, _ = make_filter(traffic=traffic)
    nrec = sum(traffic or t is not v2.BRAKE for _, t in v2.RECOVERY)

    def ordinary(snap, plans):
        first, clear = unsafe(snap, plans)
        first[3*nrec], clear[3*nrec] = np.inf, .2647659117453218
        return first, clear

    monkeypatch.setattr(f, "_evaluate", ordinary)
    monkeypatch.setattr(searcher, "search", lambda *a, fixed_policy, **k: failed(fixed_policy))
    action = np.array([.2, .1])
    f.filter(env, action)
    assert f.last["selection_branch"] == "ordinary"
    assert f.last["selection_floor"] == .06476591174532181
    f.mode = "nominal"
    if next_branch != "last certificate":
        f.plan = None
    monkeypatch.setattr(f, "_evaluate", unsafe)
    if next_branch == "idle":
        monkeypatch.setattr(f, "_threat_in_reach", lambda snap: False)
    elif next_branch == "nominal":
        monkeypatch.setattr(f, "_evaluate", lambda snap, plans: (
            np.full(len(plans), np.inf), np.full(len(plans), .8)))
    elif next_branch.startswith("searched"):
        plan = v3.plan_for(action if next_branch == "searched policy" else [.7, .2], [-1., 0.])
        monkeypatch.setattr(searcher, "search", lambda *a, fixed_policy, **k:
            accepted(plan, fixed_policy) if next_branch == "searched policy" or not fixed_policy
            else failed(fixed_policy))
    elif next_branch == "brake":
        def braking(snap, plans):
            first, clear = unsafe(snap, plans)
            index = (2+len(v2.RUDDERS)*len(v2.THROTTLES))*nrec
            first[index], clear[index] = np.inf, .8
            return first, clear
        monkeypatch.setattr(f, "_evaluate", braking)
    f.filter(env, action)
    assert f.last.get("why", f.last["mode"]) == next_branch
    assert f.last["selection_branch"] is None
    assert f.last["selection_floor"] is None
    assert f.last["selection_best_clearance"] is None
