"""Synthetic files only: verification budget accounting without any simulator."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/development_report.py"
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("development_report_verification_under_test", PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _verification(attempt=1, status="passed"):
    return dict(attempt=attempt, tag="legacy_test_verification", category="verification",
                test_id="tests/test_legacy.py::test_constant_command", test_source_sha256="a" * 64,
                status=status, suite="legacy_L1_no_target", mode="v3",
                reservation_timing="Retrospective conservative charge after expanded tests.")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    root = tmp_path / "results/safety_dev/development_v6_budget150"
    monkeypatch.setattr(report, "CAMPAIGN", root)
    (root / "runs").mkdir(parents=True)
    (root / "attempts").mkdir()
    _write(root / "reference_audit.json", {"conclusion": "Synthetic reference; not an evaluation."})
    history = tmp_path / "results/safety_dev/dev_v4_observer_memory.csv"
    history.write_text("mode,case,outcome\n" + "".join(f"v4,DV3-{i:03},goal\n" for i in range(150)))
    return root


def _add_verifications(root, records):
    _write(root / "verification_runs.json", dict(kind="verification_runs", records=records,
           pytest_result=dict(passed=208, failed=1), failure="Existing legacy assertion failed."))
    for row in records:
        _write(root / "attempts" / f"{row['attempt']:03}.json", row)


def _add_policy_pair(root, attempts=(1, 2)):
    tag = "synthetic_pair"
    directory = root / "runs" / tag
    directory.mkdir()
    archive = b"synthetic source archive bytes, never imported"
    (directory / "evaluated_sources.zip").write_bytes(archive)
    case = dict(case="DV3-SYNTHETIC", suite="dev_field", seed=123, scenario_sha256="scenario")
    config = dict(cases=[case], modes=["v4", "v6"], planned_episodes=2, shared_attempt_cap=150,
                  source_sha256={}, settings={"actual_filter_classes": {"v4": "V4", "v6": "V6"}})
    manifest = dict(configuration=config, archive_sha256=report.sha(archive))
    _write(directory / "manifest.json", manifest)
    manifest_sha = report.sha((directory / "manifest.json").read_bytes())
    rows = []
    for attempt, mode in zip(attempts, ["v4", "v6"]):
        row = dict(case, attempt=attempt, tag=tag, mode=mode, run_manifest_sha256=manifest_sha,
                   outcome="goal", seconds=1.0, actual_filter_class=mode.upper())
        rows.append(row)
        _write(directory / f"episode_{attempt:03}.json", row)
        _write(root / "attempts" / f"{attempt:03}.json", row)
    _write(directory / "complete.json", dict(completed=2, expected=2, rows=rows,
           outcomes={"v4/goal": 1, "v6/goal": 1}))


def test_completed_verification_pass_and_failure_resolve_slots_without_policy_outcomes(campaign):
    _add_verifications(campaign, [_verification(1), _verification(2, "failed")])
    value = report.collect()
    assert value["budget"]["attempted"] == 2
    assert value["budget"]["remaining"] == 148
    assert value["budget"]["completed"] == value["budget"]["errors"] == 0
    assert value["budget"]["verification_completed"] == 2
    assert value["budget"]["verification_passed"] == value["budget"]["verification_failed"] == 1
    assert value["budget"]["unresolved_or_inflight_attempts"] == 0
    assert value["tags"] == [] and value["planned_repeated_identities"] == []
    assert value["verification_runs"]["records"][0]["source_file"] == "tests/test_legacy.py"
    text = report.markdown(value)
    assert "2 completed verification test simulations" in text
    assert "0 unresolved/in-flight" in text
    assert "no goal or collision outcome is inferred" in text
    assert "Existing legacy assertion failed" in text


def test_mixed_policy_verification_and_empty_reservation_accounted_separately(campaign):
    _add_policy_pair(campaign)
    _add_verifications(campaign, [_verification(3, "failed")])
    (campaign / "attempts/004.json").write_bytes(b"")
    value = report.collect()
    assert value["budget"]["attempted"] == 4
    assert value["budget"]["completed"] == 2
    assert value["budget"]["verification_completed"] == 1
    assert value["budget"]["unresolved_or_inflight_attempts"] == 1
    assert value["tags"][0]["fresh_v6_vs_v4"]["paired_cases"] == 1
    assert sum(row["goal"] for row in value["tags"][0]["outcomes"]) == 2


def test_missing_verification_document_leaves_its_token_unresolved(campaign):
    _write(campaign / "attempts/001.json", _verification())
    value = report.collect()
    assert value["budget"]["verification_completed"] == 0
    assert value["budget"]["unresolved_or_inflight_attempts"] == 1


@pytest.mark.parametrize("field,value", [("status", "passed"), ("test_source_sha256", "b" * 64),
                                       ("test_id", "tests/test_other.py::different")])
def test_verification_record_must_match_consumed_token(campaign, field, value):
    row = _verification(status="failed")
    _add_verifications(campaign, [row])
    token = dict(row, **{field: value})
    _write(campaign / "attempts/001.json", token)
    with pytest.raises(ValueError, match="disagrees with verification"):
        report.collect()


def test_verification_record_without_consumed_slot_is_rejected(campaign):
    _add_verifications(campaign, [_verification()])
    (campaign / "attempts/001.json").unlink()
    with pytest.raises(ValueError, match="disagrees with verification"):
        report.collect()


def test_duplicate_verification_attempt_is_rejected(campaign):
    _add_verifications(campaign, [_verification(), _verification()])
    with pytest.raises(ValueError, match="Duplicate or overlapping"):
        report.collect()


@pytest.mark.parametrize("change", [{"status": "running"}, {"outcome": "goal"},
                                     {"test_source_sha256": "not-a-hash"},
                                     {"test_id": "../outside.py::test_bad"}])
def test_incomplete_or_policy_like_verification_is_rejected(change):
    row = dict(_verification(), **change)
    with pytest.raises(ValueError, match="Invalid"):
        report.validate_verification(row, row)


def _prepared_aborted_probe(root):
    directory = root / "runs" / "aborted_probe"
    directory.mkdir()
    archive = b"synthetic preflight archive, never imported"
    (directory / "evaluated_sources.zip").write_bytes(archive)
    config = dict(cases=[dict(case="DV3-PROBE", suite="dev_field", seed=1, scenario_sha256="scene")],
                  modes=["v6"], planned_episodes=1, shared_attempt_cap=150, source_sha256={},
                  settings={"actual_filter_classes": {"v6": "V6"}})
    _write(directory / "manifest.json", dict(configuration=config, archive_sha256=report.sha(archive)))
    _write(directory / "preflight_aborted.json", dict(status="aborted_before_reservation",
           attempts_reserved=0, episodes_started=0, reason="Frozen source changed: src/scenario.py"))
    return directory


def test_preflight_abort_is_complete_nonexecution_without_budget_or_error_charge(campaign):
    _prepared_aborted_probe(campaign)
    (campaign / "finite_suite_bound.md").write_text("Synthetic finite-suite note")
    _write(campaign / "integration_audit.json", {"synthetic": True})
    offline = campaign / "offline/paired_classification/REPORT.md"
    offline.parent.mkdir(parents=True)
    offline.write_text("Synthetic paired report")
    value = report.collect()
    assert value["tags"][0]["status"] == "aborted before execution"
    assert value["budget"]["attempted"] == value["budget"]["errors"] == 0
    assert value["budget"]["unresolved_or_inflight_attempts"] == 0
    assert value["tags"][0]["outcomes"] is None
    text = report.markdown(value)
    assert "aborted before execution" in text and "Zero attempts reserved" in text
    assert "[Deterministic full-suite upper bound](finite_suite_bound.md)" in text
    assert "offline/paired_classification/REPORT.md" in text and "integration_audit.json" in text


@pytest.mark.parametrize("evidence", ["token", "started", "episode", "trace"])
def test_preflight_abort_rejects_any_contradictory_execution_evidence(campaign, evidence):
    directory = _prepared_aborted_probe(campaign)
    if evidence == "token":
        _write(campaign / "attempts/001.json", {"tag": directory.name})
    elif evidence == "started":
        _write(directory / "started.json", {"started": True})
    elif evidence == "episode":
        (directory / "episode_001.json").write_bytes(b"")
    else:
        (directory / "trace_001.jsonl").write_bytes(b"")
    with pytest.raises(ValueError, match="Preflight abort"):
        report.collect()
