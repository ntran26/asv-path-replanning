"""Synthetic trace reporting checks; no scenario or environment execution."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/classify_development_failures.py"
SPEC = importlib.util.spec_from_file_location("failure_classifier_test", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(decision, reason, *, changed=False, clearance=.3, tracks=None):
    return {"decision": decision, "actual_filter_class": "safety_v6.SafetyFilterV6",
            "filter": {"why": reason, "mode": "recovery", "changed": changed,
                       "checked_clearance": clearance, "any_safe": False},
            "observer_snapshot": {"tracks": [] if tracks is None else tracks},
            "targets_before": [{"id": 0, "position_m": [2., 3.]}]}


def episode(rows, outcome="collision:target"):
    return {"case": "DV3-CRP-CV-02", "mode": "v6", "outcome": outcome,
            "steps": len(rows), "observed_decisions": len(rows), "seed": 123,
            "scenario_sha256": "scene", "attempt": 1}


def test_failed_continuation_not_reported_as_last_checked_plan():
    rows = [row(1, "no escape", clearance=float("-inf")),
            row(2, "turn", changed=True, clearance=.02),
            row(3, "searched escape", changed=True, clearance=.2),
            row(4, "last certificate", changed=True, clearance=.3)]
    summary = audit.summarize_trace(episode(rows), rows, .15)
    assert summary["trace_classification"] == "final_unchecked_continuation"
    assert summary["last_hard_accepted"]["decision"] == 3
    assert summary["last_margin_qualified"]["decision"] == 3
    assert summary["first_changed_decision"] == 2
    assert summary["no_escape_before_first_override"]
    assert summary["no_escape_prefix_decisions"] == [1]
    assert summary["target_collision_with_empty_final_snapshot"]


def test_successful_search_is_checked_even_when_original_bank_all_unsafe():
    rows = [row(1, "searched policy", clearance=.15)]
    rows[0]["filter"]["policy_search"] = {"accepted": True, "maximum_clearance": float("inf")}
    summary = audit.summarize_trace(episode(rows), rows, .15)
    assert summary["trace_classification"] == "final_checked_margin_qualified"
    assert summary["first_changed_decision"] is None
    assert summary["final"]["original_bank_any_safe"] is False
    assert summary["target_collision_with_empty_final_snapshot"]
    json.dumps(audit.json_safe(summary), allow_nan=False)


def test_history_anchors_and_sources_are_not_vessel_counts():
    rows = [row(1, "turn", tracks=[{"id": -1, "position_m": [2.1, 3.]}])]
    rows[0]["perception_statistics"] = {"last_track_history_stats": {
        "added_hypotheses": 2, "stored_anchor_count": 3,
        "hypotheses": [{"id": -1, "source_id": 7}, {"id": -2, "source_id": 7}]}}
    summary = audit.summarize_trace(episode(rows), rows, .15)
    assert summary["final"]["history_added_hypotheses"] == 2
    assert summary["final"]["history_distinct_source_count"] == 1
    assert summary["final"]["history_stored_anchor_count"] == 3
    assert summary["history_distinct_source_ids_used"] == [7]
    assert summary["provisional_added_tracks_sum"] is None
    assert summary["provisional_added_tracks_logged_decisions"] == 0
    assert summary["final"]["nearest_snapshot_track_to_any_truth_target_m"] == pytest.approx(.1)
    assert not summary["target_collision_with_empty_final_snapshot"]


@pytest.mark.parametrize("corruption", ["gap", "class", "length"])
def test_bad_trace_identity_or_missing_decisions_rejected(corruption):
    rows = [row(1, "nominal"), row(2, "turn")]
    ep = episode(rows)
    if corruption == "gap":
        rows[1]["decision"] = 3
    elif corruption == "class":
        rows[0]["actual_filter_class"] = "safety_v4.SafetyFilterV4"
    else:
        ep["steps"] = 3
    with pytest.raises(ValueError):
        audit.summarize_trace(ep, rows, .15)


def test_continue_requires_current_successful_recheck():
    rows = [row(1, "continue", clearance=.5)]
    rows[0]["filter"]["continuation_ok"] = False
    assert audit.summarize_trace(episode(rows), rows, .15)["last_hard_accepted"] is None
    rows[0]["filter"]["continuation_ok"] = True
    assert audit.summarize_trace(episode(rows), rows, .15)["last_hard_accepted"]["decision"] == 1


def paired_fixture(tmp_path):
    directory = tmp_path / "campaign/runs/paired"
    directory.mkdir(parents=True)
    attempts = tmp_path / "campaign/attempts"
    attempts.mkdir()
    archive = b"synthetic source archive fixture"
    (directory / "evaluated_sources.zip").write_bytes(archive)
    manifest = {"archive_sha256": audit.sha(archive), "configuration": {
        "cases": [{"case": "DV3-CRP-CV-02", "suite": "dev_field", "seed": 123, "scenario_sha256": "scene"}],
        "modes": ["v4", "v6"], "settings": {"effective_safety_constants": {
            "v2": {"TRIGGER_MARGIN_M": "0.15"}}}}}
    raw = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(raw)
    episodes = []
    for number, mode in enumerate(("v4", "v6"), 1):
        rows = [row(1, "nominal")]
        rows[0]["actual_filter_class"] = f"safety_{mode}.SafetyFilter{mode.upper()}"
        ep = episode(rows, "goal")
        ep.update(mode=mode, attempt=number, run_manifest_sha256=audit.sha(raw), suite="dev_field", tag="paired")
        encoded = json.dumps(ep)
        (directory / f"episode_{number:03d}.json").write_text(encoded)
        (attempts / f"{number:03d}.json").write_text(encoded)
        (directory / f"trace_{number:03d}.jsonl").write_text(json.dumps(rows[0]) + "\n")
        episodes.append(ep)
    (directory / "complete.json").write_text(json.dumps({"completed": 2, "expected": 2, "rows": episodes}))
    return directory


def test_fresh_pair_manifest_tokens_and_traces_are_read_together(tmp_path):
    episodes, provenance = audit.load_run(paired_fixture(tmp_path))
    assert {e["mode"] for e in episodes} == {"v4", "v6"}
    assert provenance["paired_cases"] == 1
    assert provenance["trigger_margin_m"] == .15


@pytest.mark.parametrize("corruption", ["seed", "historical_manifest", "missing_episode"])
def test_pair_identity_or_missing_durable_record_cannot_be_substituted(tmp_path, corruption):
    directory = paired_fixture(tmp_path)
    path = directory / "episode_001.json"
    if corruption == "missing_episode":
        path.unlink()
    else:
        ep = json.loads(path.read_text())
        if corruption == "seed":
            ep["seed"] = 999
        else:
            ep["run_manifest_sha256"] = "historical"
        path.write_text(json.dumps(ep))
    with pytest.raises(ValueError):
        audit.load_run(directory)
