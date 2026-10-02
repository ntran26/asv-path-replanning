"""Fixed sample and strict attempt limits; no model or episode is instantiated."""
import importlib.util
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/quick_campaign.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("quick_campaign_under_test", _PATH)
quick = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quick)


def inventory():
    rows = []
    def add(component, name, test_id=""):
        rows.append({"suite": component, "case": name, "test_id": test_id,
                     "seed": len(rows), "scenario_sha256": "scene-" + name, "obstacles": None})
    for fam in ["BAS-HO-CV", "BAS-CR-CV", "BAS-OT-CV", "BAS-BO-CV", "BAS-NU-CV", "CH-HO-CV", "CH-CR-CV", "CH-OT-CV"]:
        for i in range(3):
            add("frozen_b", fam + str(i), fam + f"-{i:03d}")
    for fam in ["BAS-HO-NC", "CH-HO-RE", "BAS-CR-RE", "CH-OT-RE", "BAS-BO-RE"]:
        for i in range(3):
            add("frozen_r", fam + str(i), fam + f"-{i:03d}")
    for fam in ["HO", "CRS", "OT"]:
        for width in ["W", "I", "N"]:
            add("frozen_a", f"A-{fam}-{width}")
    for fam in ["NT", "HO", "CRP", "BO", "CRS", "OT"]:
        for i in range(3):
            add("field_validation", f"FV-{fam}-{i:02d}")
    for lay in ["L1", "L2", "L3"]:
        for speed in ["FIX", "VAR"]:
            for fam in ["HO", "CRP", "OT"]:
                add("field_deployment", f"P2-{lay}-{fam}-{speed}-01")
    return rows


def test_sample_is_fixed_24_distinct_cases_independent_of_input_order():
    rows = inventory()
    selected = quick.select_cases(rows)
    assert len(selected) == 24
    assert {(c["suite"], c["case"]) for c in selected} == {(c["suite"], c["case"]) for c in quick.select_cases(list(reversed(rows)))}
    assert {name: sum(c["suite"] == name for c in selected) for name in {c["suite"] for c in selected}} == {
        "frozen_b": 8, "frozen_r": 4, "frozen_a": 2, "field_validation": 4, "field_deployment": 6}
    for case in selected:
        original = next(c for c in rows if (c["suite"], c["case"]) == (case["suite"], case["case"]))
        assert all(case[name] == original[name] for name in ("seed", "scenario_sha256", "obstacles"))
        assert case["inverse_probability_weight"] is None and case["design_probability"] is None
    assert {c["sampling_stratum"] for c in selected if c["suite"] == "field_deployment"} == {
        f"field_deployment/{lay}/{speed}" for lay in ("L1", "L2", "L3") for speed in ("FIX", "VAR")}
    assert {c["case"] for c in selected if c["suite"] == "frozen_a"} == {"A-HO-N", "A-CRS-I"}


def test_omitted_family_does_not_change_selection_of_predeclared_strata():
    rows = inventory()
    with_outcomes = [dict(row, irrelevant_outcome="goal" if i % 2 else "collision:boundary") for i, row in enumerate(rows)]
    assert [(r["suite"], r["case"]) for r in quick.select_cases(rows)] == [(r["suite"], r["case"]) for r in quick.select_cases(with_outcomes)]


def test_empty_required_stratum_fails_instead_of_silently_replacing_case():
    rows = [c for c in inventory() if not c["case"].startswith("FV-NT-")]
    with pytest.raises(ValueError, match="Empty predeclared"):
        quick.select_cases(rows)


@pytest.mark.parametrize("field,value", [("seed", -1), ("scenario_sha256", "changed"),
                                        ("obstacles", 99), ("test_id", "different")])
def test_selected_manifest_cannot_drift_from_canonical_seed_or_geometry(field, value):
    canonical = inventory()
    selected = quick.select_cases(canonical)
    quick.validate_sample_identity(selected, canonical)
    selected[0][field] = value
    with pytest.raises(ValueError, match=f"canonical {field} mismatch"):
        quick.validate_sample_identity(selected, canonical)


def test_attempt_tokens_are_consumed_before_execution_and_never_reused(tmp_path):
    attempts = tmp_path / "attempts"
    assert quick.reserve_attempt(attempts, {"mode": "v4", "case": "A"}, limit=3) == 1
    # A crash during token writing consumes the empty slot conservatively.
    (attempts / "002.json").write_bytes(b"")
    assert quick.reserve_attempt(attempts, {"mode": "v5", "case": "A"}, limit=3) == 3
    with pytest.raises(RuntimeError, match="HARD NEW EPISODE BUDGET EXHAUSTED"):
        quick.reserve_attempt(attempts, {"mode": "v5", "case": "retry"}, limit=3)
    assert json.loads((attempts / "001.json").read_text())["attempt"] == 1
    assert (attempts / "002.json").read_bytes() == b""


def test_tokens_are_unique_across_concurrent_tags(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as workers:
        numbers = list(workers.map(lambda i: quick.reserve_attempt(tmp_path / "attempts", {"tag": str(i)}, limit=8), range(8)))
    assert sorted(numbers) == list(range(1, 9))
    with pytest.raises(RuntimeError):
        quick.reserve_attempt(tmp_path / "attempts", {"tag": "later"}, limit=8)


@pytest.mark.parametrize("limit", [0, 101, 1000])
def test_cap_cannot_be_raised_above_100(tmp_path, limit):
    with pytest.raises(ValueError):
        quick.reserve_attempt(tmp_path / "attempts", {}, limit=limit)


def test_step_observer_preserves_actions_and_return_value():
    sentinel = object()
    class Base:
        def reset(self):
            self.obstacles = [1, 2, 3]
            self._safety_v2 = SimpleNamespace(last={})
            return sentinel
        def step(self, action):
            self.received = action
            self._safety_v2.last = {"sideslip_rescue_evaluated": True, "sideslip_rescue_admitted": True,
                                    "verified_policy_priority_enabled": True,
                                    "verified_policy_preserved": True, "continuation_override_prevented": True,
                                    "policy_feedback_evaluated": True, "policy_feedback_preserved": True,
                                    "recovery_template": "sideslip feedback"}
            return sentinel
    env = quick.audited_environment(Base)()
    assert env.reset() is sentinel
    action = object()
    assert env.step(action) is sentinel and env.received is action
    assert env.quick_counts["sideslip_rescue_evaluated_steps"] == 1
    assert env.quick_counts["sideslip_rescue_admitted_steps"] == 1
    assert env.quick_counts["verified_policy_preserved_steps"] == 1
    assert env.quick_counts["continuation_override_prevented_steps"] == 1
    assert env.quick_counts["policy_feedback_evaluated_steps"] == 1
    assert env.quick_counts["policy_feedback_preserved_steps"] == 1
    assert env.quick_counts["recovery_template_counts"] == {"sideslip feedback": 1}
    assert env.quick_counts["realized_obstacle_count"] == 3
    env.reset()
    assert env.quick_counts["sideslip_rescue_evaluated_steps"] == 0
    assert env.quick_counts["verified_policy_preserved_steps"] == 0
    assert env.quick_counts["policy_feedback_evaluated_steps"] == 0
    assert env.quick_counts["policy_feedback_preserved_steps"] == 0


def test_summary_exposes_unexercised_method_and_paired_losses():
    rows = [{"suite": "frozen_b", "case": "one", "mode": "v4", "outcome": "goal"},
            {"suite": "frozen_b", "case": "one", "mode": "v5", "outcome": "collision:boundary"}]
    summary = quick.summarize_results(rows)
    assert summary["paired_cases"] == 1 and summary["lost_goals_vs_v4"] == 1
    assert summary["mechanism_counts"]["v5"]["episodes_evaluated"] == 0
    assert not summary["population_performance_estimate"]


def test_stop_between_attempts_preserves_prior_tokens_without_consuming_next(tmp_path):
    attempts = tmp_path / "attempts"
    assert quick.reserve_attempt(attempts, {"mode": "off", "case": "first"}) == 1
    prior = (attempts / "001.json").read_bytes()
    marker = tmp_path / "STOP"
    marker.write_text("User requested cancellation")
    with pytest.raises(quick.StopRequested, match="STOP marker"):
        quick.reserve_attempt(attempts, {"mode": "v4", "case": "second"})
    assert (attempts / "001.json").read_bytes() == prior
    assert not (attempts / "002.json").exists()
    assert marker.exists()


def test_stop_before_first_attempt_does_not_create_budget_directory(tmp_path):
    (tmp_path / "STOP").touch()
    with pytest.raises(quick.StopRequested):
        quick.reserve_attempt(tmp_path / "attempts", {})
    assert not (tmp_path / "attempts").exists()


def test_unfiltered_mode_disables_filter_and_filtered_modes_select_their_versions():
    assert quick.mode_settings("off") == (1, False)
    assert quick.mode_settings("v4") == (4, True)
    assert quick.mode_settings("v5") == (5, True)
    with pytest.raises(ValueError):
        quick.mode_settings("v3")


def test_three_mode_summary_uses_matched_intersections_and_mechanism_episode_counts():
    rows = [
        {"suite": "frozen_b", "case": "one", "mode": "off", "outcome": "collision:boundary"},
        {"suite": "frozen_b", "case": "one", "mode": "v4", "outcome": "goal"},
        {"suite": "frozen_b", "case": "one", "mode": "v5", "outcome": "goal",
         "verified_policy_preserved_steps": 3, "continuation_override_prevented_steps": 2,
         "policy_feedback_evaluated_steps": 5, "policy_feedback_preserved_steps": 3},
        {"suite": "frozen_b", "case": "two", "mode": "v4", "outcome": "goal"},
        {"suite": "frozen_b", "case": "two", "mode": "v5", "outcome": "collision:target"},
        {"suite": "frozen_r", "case": "one", "mode": "off", "outcome": "goal"},
        {"suite": "frozen_r", "case": "one", "mode": "v5", "outcome": "timeout"},
    ]
    result = quick.summarize_results(rows)
    assert result["comparisons"]["v5_vs_v4"]["paired_cases"] == 2
    assert result["lost_goals_vs_v4"] == 1
    assert result["comparisons"]["v5_vs_off"]["paired_cases"] == 2
    assert result["gained_goals_vs_off"] == result["lost_goals_vs_off"] == 1
    assert result["comparisons"]["v4_vs_off"]["paired_cases"] == 1
    assert result["comparisons"]["v5_vs_off"]["candidate_unpaired_cases"] == 1
    assert result["mechanism_counts"]["v5"]["verified_policy_preserved_steps"] == 3
    assert result["mechanism_counts"]["v5"]["verified_policy_preserved_episodes"] == 1
    assert result["mechanism_counts"]["v5"]["continuation_override_prevented_steps"] == 2
    assert result["mechanism_counts"]["v5"]["continuation_override_prevented_episodes"] == 1
    assert result["mechanism_counts"]["off"]["verified_policy_preserved_episodes"] == 0
    assert result["mechanism_counts"]["v5"]["policy_feedback_evaluated_steps"] == 5
    assert result["mechanism_counts"]["v5"]["policy_feedback_evaluated_episodes"] == 1
    assert result["mechanism_counts"]["v5"]["policy_feedback_preserved_steps"] == 3
    assert result["mechanism_counts"]["v5"]["policy_feedback_preserved_episodes"] == 1
    assert result["mechanism_counts"]["off"]["policy_feedback_preserved_steps"] == 0
