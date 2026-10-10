"""V20 decision logic with a stubbed V16 decision and stubbed certificates.

No simulator, episode or policy. V16's `_filter` is replaced per test so each
branch is exercised deterministically; certificate physics is covered by
test_safety_v20_contingency.py.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
import safety_v16 as v16
import safety_v20 as v20
import safety_v20_contingency as sc


class Onboard(SimpleNamespace):
    """Environment stand-in that refuses truth and scenario access."""

    @property
    def targets(self):
        raise AssertionError("truth target access")

    @property
    def asv_x(self):
        raise AssertionError("truth pose access")

    @property
    def scenario(self):
        raise AssertionError("scenario access")


def make_env():
    lidar = SimpleNamespace(bearings=np.linspace(-180, 179, 360), ranges=np.full(360, 16.0))
    return Onboard(command_rate_limit=False, _v2_brake=False, lidar=lidar,
                   gated_ranges=np.full(360, 16.0), pose_stale=False)


def snap():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 25.0], [0.0, 25.0]])
    return cc.Snapshot(5.0, 5.0, 0.0, 0.5, 0.0, 0.0, np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                       np.array([5.0, 5.0]), 0.0, 0.0, 15.0, np.empty((0, 2)), [], a, np.roll(a, -1, 0))


def stub_v16(monkeypatch, why, out, brake=False, plan=None):
    def fake(self, env, action):
        self._observer_snapshot = snap()
        self.actuators.issue(env, float(out[0]))
        env._v2_brake = brake
        self.plan = None if plan is None else np.asarray(plan, float)
        self.mode = "recovery" if why == "last certificate" else "nominal"
        self.uncertified_steps = 2 if why == "last certificate" else 0
        self.last = {"why": why}
        changed = brake or not np.allclose(out, action)
        return np.asarray(out, np.float32), bool(changed)
    monkeypatch.setattr(v16.SafetyFilterV16, "_filter", fake)


class FakeChecker:
    """Certifies a command iff its rudder is in `allowed` (NaN throttle = brake)."""
    allowed = set()
    sequence_ok = True

    def __init__(self, *args, **kwargs):
        pass

    def certify(self, first):
        first = np.asarray(first, float)
        ok = round(float(first[0]), 6) in self.allowed
        seq = np.vstack([first[None], np.tile((0.0, np.nan), (sc.DECISIONS - 1, 1))])
        return sc.Certificate(ok, 0.1 if ok else -0.1, 0, seq, 3.0, {"static": 0.1})

    def certify_many(self, firsts):
        return [self.certify(f) for f in np.asarray(firsts, float).reshape(-1, 2)]

    def certify_sequence(self, seq):
        return sc.Certificate(self.sequence_ok, 0.05 if self.sequence_ok else -0.05, -1,
                              np.asarray(seq, float), 3.0, {})


@pytest.fixture
def fake_checker(monkeypatch):
    FakeChecker.allowed, FakeChecker.sequence_ok = set(), True
    monkeypatch.setattr(sc, "ContingencyChecker", FakeChecker)
    return FakeChecker


def make_filter(**options):
    return v20.SafetyFilterV20(**options)


def run(filt, env, action):
    filt._observer_actuators = copy.deepcopy(filt.actuators)
    return filt._filter(env, np.asarray(action, np.float32))


def test_constructor_validates_options():
    for bad in (dict(allowance_tables="x"), dict(out_of_contract="x"), dict(hold_horizon_s=-1.0),
                dict(tail_family="x"), dict(enforcement="x")):
        with pytest.raises(ValueError):
            make_filter(**bad)
    assert make_filter(allowance_tables="none").own_table is None


def test_checked_branch_is_never_changed_even_if_uncertified(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "turn", [-0.5, 0.0])
    filt, env = make_filter(), make_env()
    out, changed = run(filt, env, [0.8, 0.9])
    assert np.allclose(out, [-0.5, 0.0]) and changed
    assert filt.last["v20_level"] == "v16_unchanged_finite_only" and filt.v20_committed is None
    fake_checker.allowed = {-0.5}
    out, _ = run(filt, env, [0.8, 0.9])
    assert filt.last["v20_level"] == "v16_unchanged_certified" and filt.v20_committed is not None


def test_parity_mode_returns_v16_decision_and_state(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "no escape", [0.9, 0.9], plan=[[0.1, 0.1]])
    filt, env = make_filter(certified_fallback=False), make_env()
    out, changed = run(filt, env, [0.9, 0.9])
    assert np.allclose(out, [0.9, 0.9]) and not changed
    assert filt.last["v20_level"] == "v16_unchanged_finite_only"
    assert np.allclose(filt.plan, [[0.1, 0.1]]) and filt.mode == "nominal"


def test_no_escape_prefers_certified_sac_and_replaces_failed_plan(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "no escape", [0.9, 0.9], plan=[[0.1, 0.1]])
    fake_checker.allowed = {0.9}
    filt, env = make_filter(), make_env()
    out, changed = run(filt, env, [0.9, 0.9])
    assert np.allclose(out, [0.9, 0.9]) and not changed
    assert filt.last["v20_level"] == "replaced_by_sac"
    assert filt.plan.shape == (v20.V16_PLAN_DECISIONS, 2) and filt.uncertified_steps == 0


def test_last_certificate_uses_certified_projection_and_rewinds_actuators(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "last certificate", [1.0, np.float32(-1.0)], brake=True)
    fake_checker.allowed = {0.5}
    filt, env = make_filter(), make_env()
    for rudder in (0.2, 0.3):
        filt.actuators.issue(env, rudder)
    before = copy.deepcopy(filt.actuators)
    out, changed = run(filt, env, [0.4, 0.9])
    assert filt.last["v20_level"] == "replaced_by_projection" and filt.last["why"] == "v20 projection"
    assert np.allclose(out, [0.5, 1.0]) and changed and env._v2_brake is False
    expected = copy.deepcopy(before)
    expected.issue(env, 0.5)
    assert filt.actuators.servo == pytest.approx(expected.servo)
    assert np.allclose(filt.actuators.buffer, expected.buffer)
    assert filt.mode == "recovery" and filt.uncertified_steps == 0


def test_committed_contingency_then_out_of_contract_stop(monkeypatch, fake_checker):
    filt, env = make_filter(), make_env()
    # Decision 1: checked and certified; commit a brake contingency.
    stub_v16(monkeypatch, "nominal", [0.0, 0.0])
    fake_checker.allowed = {0.0}
    run(filt, env, [0.0, 0.0])
    assert filt.v20_committed is not None
    # Decision 2: unchecked, nothing new certified; the committed tail still is.
    stub_v16(monkeypatch, "no escape", [0.7, 0.7])
    fake_checker.allowed = set()
    out, _ = run(filt, env, [0.7, 0.7])
    assert filt.last["v20_level"] == "committed_continuation" and env._v2_brake is True
    assert out[1] == pytest.approx(-1.0)
    # Decision 3: the committed tail fails its recheck too: follow it, out of contract.
    fake_checker.sequence_ok = False
    run(filt, env, [0.7, 0.7])
    assert filt.last["v20_level"] == "out_of_contract_committed"
    # Decision 4: a checked V16 decision without a certificate clears the commitment.
    stub_v16(monkeypatch, "turn", [0.5, 0.0])
    run(filt, env, [0.7, 0.7])
    assert filt.v20_committed is None
    # Decision 5: unchecked with nothing at all: centred full astern.
    stub_v16(monkeypatch, "no escape", [0.7, 0.7])
    out, changed = run(filt, env, [0.7, 0.7])
    assert filt.last["v20_level"] == "out_of_contract_stop"
    assert np.allclose(out, [0.0, -1.0]) and changed and env._v2_brake is True and filt.plan is None


def test_out_of_contract_v16_option_keeps_v16_command(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "no escape", [0.7, 0.7])
    filt, env = make_filter(out_of_contract="v16"), make_env()
    out, _ = run(filt, env, [0.7, 0.7])
    assert filt.last["v20_level"] == "out_of_contract_v16" and np.allclose(out, [0.7, 0.7])


def test_unknown_branch_label_is_treated_as_unchecked(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "some future label", [0.7, 0.7])
    fake_checker.allowed = {0.7}
    filt, env = make_filter(), make_env()
    run(filt, env, [0.7, 0.7])
    assert filt.last["v20_level"] == "replaced_by_sac"


def test_command_limit_disables_v20_without_changing_v16(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "no escape", [0.7, 0.7])
    filt, env = make_filter(), make_env()
    env.command_rate_limit = True
    out, _ = run(filt, env, [0.7, 0.7])
    assert filt.last["v20_level"] == "not_evaluated" and np.allclose(out, [0.7, 0.7])


def test_real_certificate_path_runs_without_truth(monkeypatch):
    stub_v16(monkeypatch, "no escape", [0.2, 0.3])
    filt, env = make_filter(), make_env()
    out, _ = run(filt, env, [0.2, 0.3])
    assert filt.last["v20_level"] in ("replaced_by_sac", "replaced_by_v16_certified")
    assert filt.last["v20_observed_free"]["free"] >= 0.0


def test_gatekeeper_replaces_an_uncertified_checked_command_near_v16(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "turn", [-0.5, 0.0])
    fake_checker.allowed = {-1.0}
    filt, env = make_filter(enforcement="gatekeeper"), make_env()
    out, changed = run(filt, env, [0.8, 0.9])
    assert filt.last["v20_level"] == "gatekeeper_replaced" and filt.last["why"] == "v20 gatekeeper"
    assert out[0] == pytest.approx(-1.0) and changed and filt.v20_committed is not None


def test_gatekeeper_keeps_v16_when_nothing_certifies(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "turn", [-0.5, 0.0])
    filt, env = make_filter(enforcement="gatekeeper"), make_env()
    out, _ = run(filt, env, [0.8, 0.9])
    assert filt.last["v20_level"] == "gatekeeper_uncertified_v16" and np.allclose(out, [-0.5, 0.0])
    assert filt.v20_committed is None


def test_gatekeeper_leaves_certified_v16_commands_alone(monkeypatch, fake_checker):
    stub_v16(monkeypatch, "nominal", [0.3, 0.4])
    fake_checker.allowed = {0.3, 1.0}
    filt, env = make_filter(enforcement="gatekeeper"), make_env()
    out, changed = run(filt, env, [0.3, 0.4])
    assert filt.last["v20_level"] == "v16_unchanged_certified" and np.allclose(out, [0.3, 0.4]) and not changed
