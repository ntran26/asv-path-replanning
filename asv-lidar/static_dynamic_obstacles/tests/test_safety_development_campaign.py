"""Budget, selection and injection checks only; no simulator/model invocation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/development_campaign.py"
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("development_campaign_under_test", PATH)
dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dev)


def canonical():
    return [dict(suite="dev_field", case="DV3-A", seed=123, scenario_sha256="abc", obstacles=None, test_id=""),
            dict(suite="dev_field", case="DV3-B", seed=124, scenario_sha256="def", obstacles=None, test_id=""),
            dict(suite="frozen_b", case="B-00-001", seed=999, scenario_sha256="heldout", obstacles=None, test_id="")]


def test_selected_cases_keep_exact_canonical_fields_and_order():
    selected = dev.select_cases(["DV3-B", "DV3-A"], canonical())
    assert selected == list(reversed(canonical()[:2]))
    selected[0]["seed"] = 0
    assert canonical()[1]["seed"] == 124


@pytest.mark.parametrize("requested", [["DV3-A", "DV3-A"], ["B-00-001"], [],
    [{"suite": "frozen_b", "case": "DV3-A"}], [{"case": "DV3-A", "seed": 0}],
    [{"case": "DV3-A", "scenario_sha256": "tampered"}], [{"case": "DV3-A", "obstacles": 7}]])
def test_unknown_duplicate_heldout_or_tampered_selection_rejected(requested):
    with pytest.raises(ValueError):
        dev.select_cases(requested, canonical())


def test_failed_or_empty_tokens_consume_budget_and_stop_does_not(tmp_path):
    ledger = tmp_path / "new150" / "attempts"
    assert dev.reserve_attempt(ledger, {"case": "one"}, limit=3) == 1
    (ledger / "002.json").touch()
    (ledger.parent / "STOP").touch()
    with pytest.raises(dev.StopRequested):
        dev.reserve_attempt(ledger, {"case": "three"}, limit=3)
    assert not (ledger / "003.json").exists()
    (ledger.parent / "STOP").unlink()
    assert dev.reserve_attempt(ledger, {"case": "three"}, limit=3) == 3
    with pytest.raises(RuntimeError, match="EXHAUSTED"):
        dev.reserve_attempt(ledger, {"case": "retry"}, limit=3)


def test_new_budget_is_independent_of_old_quick_ledger(tmp_path):
    old = tmp_path / "quick_v5_budget100/attempts"
    old.mkdir(parents=True)
    (old / "001.json").write_bytes(b"old ledger must not change")
    new = tmp_path / "development_v6_budget150/attempts"
    assert dev.reserve_attempt(new, {"tag": "pilot"}) == 1
    assert (old / "001.json").read_bytes() == b"old ledger must not change"


def test_atomic_tokens_are_unique_across_concurrent_tags(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as executor:
        numbers = list(executor.map(lambda i: dev.reserve_attempt(tmp_path / "attempts", {"tag": str(i)}, limit=8), range(8)))
    assert sorted(numbers) == list(range(1, 9))


@pytest.mark.parametrize("limit", [0, 151, True])
def test_cap_cannot_be_raised(limit, tmp_path):
    with pytest.raises(ValueError):
        dev.reserve_attempt(tmp_path / "attempts", {}, limit=limit)


def test_v6_class_injection_restores_original_on_failure(monkeypatch):
    class V5: pass
    class V6(V5): pass
    module = SimpleNamespace(SafetyFilterV5=V5)
    monkeypatch.setitem(sys.modules, "safety_v5", module)
    with pytest.raises(RuntimeError):
        with dev.injected_filter("v6", SimpleNamespace(SafetyFilterV6=V6)):
            assert module.SafetyFilterV5 is V6
            raise RuntimeError("simulated failure")
    assert module.SafetyFilterV5 is V5
    with dev.injected_filter("v4", None):
        assert module.SafetyFilterV5 is V5


def test_observer_calls_step_once_without_changing_action_or_result():
    Filter = type("SafetyFilterV6", (), {"__module__": "safety_v6"})
    marker, action = object(), [0.2, 0.1]
    class Base:
        def reset(self):
            self.calls = 0
            self.rudder, self.rpm, self.propulsion_s2 = 25.0, -600.0, -100.0
            self._ego_hold = (1.0, 0.2, 90.0)
            self.tracker = SimpleNamespace(tracks=[SimpleNamespace(id=9, position=[1.,2.], velocity=[0.1,0.2],
                hits=3, misses=0, is_dynamic=False, last_fit_centre=[2.,3.], last_fit_heading_deg=45.,
                last_evidence=SimpleNamespace(appear=2,vacate=1,compared=20,violations=3,moving=True))])
            self.obstacles = [[[0.,0.],[1.,0.],[1.,1.]]]
            self._safety_v2 = Filter()
            self._safety_v2.last = {"mode": "recovery", "new_candidate_accepted": True}
            self._safety_v2._observer_snapshot = SimpleNamespace(x=3.0, y=4.0, heading=0.3,
                u=1.1, v=0.1, r=0.2, points=[1,2], tracks=[SimpleNamespace(
                    id=7, position=[9.0,10.0], velocity=[0.1,0.2], heading=0.4)])
            self._safety_v2.perception = SimpleNamespace(last_memory_stats={"cleared_points": 5},
                last_hull_stats={"current_hull_fits":1}, last_provisional_stats={"admitted":2})
            return marker
        def step(self, supplied):
            self.calls += 1
            self.tracker.tracks[0].position[0] += 10.0
            assert supplied is action
            return marker
    trace = io.StringIO()
    env = dev.observed_environment(Base, "v6", trace)()
    assert env.reset() is marker
    assert env.step(action) is marker and env.calls == 1
    assert action == [0.2, 0.1]
    assert env.development_counts["actual_filter_class"] == "safety_v6.SafetyFilterV6"
    assert env.development_counts["diagnostic_true_steps"] == {"new_candidate_accepted": 1}
    captured = json.loads(trace.getvalue())
    assert captured["policy_action"] == action
    assert captured["issued_rudder_normalized"] == 0.25
    assert captured["issued_rpm_signed"] == -600.0
    assert captured["measured_ego_before"]["r_radps"] == pytest.approx(1.5707963267948966)
    assert captured["observer_snapshot"]["tracks"][0]["id"] == 7
    assert captured["observer_snapshot"]["static_point_count"] == 2
    assert captured["perception_memory"]["cleared_points"] == 5
    assert captured["tracker_tracks_before"][0]["position"] == [1.,2.]
    assert captured["tracker_tracks_before"][0]["last_evidence"]["moving"] is True
    assert captured["perception_hull"] == {"current_hull_fits":1}
    assert captured["perception_provisional"] == {"admitted":2}
    assert captured["static_geometry_at_first_decision"]["obstacle_polygons_m"] == env.obstacles
    env.step(action)
    second = json.loads(trace.getvalue().splitlines()[1])
    assert "static_geometry_at_first_decision" not in second
    assert second["tracker_tracks_before"][0]["position"] == [11.,2.]


def test_duplicate_or_unknown_modes_rejected():
    for modes in ("v6,v6", "oracle", ""):
        with pytest.raises(ValueError):
            dev.parse_modes(modes)
    assert dev.parse_modes("off,v4,v5,v6") == ["off", "v4", "v5", "v6"]


OLD_ENV_SHA = "3664ee0c505fa18f26d7c0250d1ec0fff40f45d83f80c663d22cdda0ede6fb9b"
NEW_ENV_SHA = "608d2dce1f3e17efa6ff0eadee0831cbb023482c7422f02ac73bdda4bd280ccb"


@pytest.mark.parametrize("name", ["src/env.py", "src/scenario.py", "src/safety_v4.py"])
def test_reference_source_accepts_unchanged_bytes(name):
    assert dev.reference_source_matches(name, "a" * 64, "a" * 64)


def test_reference_source_accepts_only_audited_native_env_transition():
    assert dev.reference_source_matches("src/env.py", OLD_ENV_SHA, NEW_ENV_SHA)


@pytest.mark.parametrize("name,expected,actual", [
    ("src/env.py", OLD_ENV_SHA, "a" * 64),
    ("src/env.py", "a" * 64, NEW_ENV_SHA),
    ("src/env.py", NEW_ENV_SHA, OLD_ENV_SHA),
    ("src/scenario.py", OLD_ENV_SHA, NEW_ENV_SHA),
    ("src/safety_v4.py", OLD_ENV_SHA, NEW_ENV_SHA),
    ("src/scenario.py", "6aaa6df5878c786fee94e02b8b5ab762c1806dd53338531fe5061391de8fdf15",
     "a97659d9979b3dd6f4a1d4c131368bfc8d290514994ad188ce83470a9cfd97ba"),
])
def test_reference_source_rejects_unaudited_changes(name, expected, actual):
    assert not dev.reference_source_matches(name, expected, actual)


def test_runtime_rejects_scenario_drift_after_accepting_audited_env(monkeypatch):
    """Exercise the real preflight loop with synthetic sources, no model or env."""
    baseline = {"source_sha256": {"src/env.py": OLD_ENV_SHA, "src/scenario.py": "old-scenario"}}
    reference_bytes = json.dumps({"settings": baseline}).encode()
    files = {dev.REFERENCE: reference_bytes, dev.ROOT / "src/env.py": b"new-env",
             dev.ROOT / "src/scenario.py": b"changed-scenario"}
    monkeypatch.setattr(dev, "read_shared_bytes", files.__getitem__)
    monkeypatch.setattr(dev, "sha", lambda raw: {b"new-env": NEW_ENV_SHA,
                                               b"changed-scenario": "new-scenario"}[raw])
    for name, module in {
        "torch": SimpleNamespace(set_num_threads=lambda count: None),
        "constants": SimpleNamespace(), "curriculum": SimpleNamespace(),
        "train_formulation": SimpleNamespace(),
        "prediction_audit": SimpleNamespace(runtime_overrides=lambda: None),
        "suite_eval": SimpleNamespace(settings_snapshot=lambda *args: None),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match=r"Frozen reference source changed: src/scenario\.py"):
        dev.prepare_runtime(["v6"], {})
