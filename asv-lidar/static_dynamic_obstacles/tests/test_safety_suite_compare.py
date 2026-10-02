"""Paired report correctness and provenance guards without evaluation episodes."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety"
sys.path.insert(0, str(DIRECTORY))


def _module(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", DIRECTORY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _module("suite_compare")


def _case(component, name="same-id", seed=7):
    return {"suite": component, "case": name, "test_id": "", "seed": seed,
            "scenario_sha256": "scene-" + str(seed)}


def _run(path, cases, outcomes, modes=("off", "v4", "v5"), **settings_updates):
    path.mkdir()
    settings = {key: {} for key in report.PROVENANCE_KEYS}
    settings.update(schema=1, modes=list(modes), checkpoint_sha256="model",
                    effective_safety_constants={"v4": {"OBSERVER": "True"}, "v5": {"SOFT": "True"}},
                    **settings_updates)
    manifest = {"settings": settings, "cases": cases, "expected_episodes": len(cases) * len(modes)}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [dict(case, mode=mode, outcome=outcome)
            for case in cases for mode in modes
            if (outcome := outcomes.get((mode, case["suite"], case["case"]))) is not None]
    (path / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _fixture(tmp_path):
    cases = [_case("frozen_b"), _case("frozen_r", seed=8), _case("field_validation", seed=9)]
    labels = {"off": ["collision:obstacle", "goal", "timeout"],
              "v4": ["goal", "goal", "collision:target"],
              "v5": ["goal", "collision:boundary", "timeout"]}
    outcomes = {(mode, case["suite"], case["case"]): labels[mode][i]
                for mode in labels for i, case in enumerate(cases)}
    return cases, outcomes


def test_repeated_ids_stay_distinct_and_aggregate_rescues_are_paired(tmp_path):
    cases, outcomes = _fixture(tmp_path)
    directories = [_run(tmp_path / "frozen", cases[:2], outcomes),
                   _run(tmp_path / "field", cases[2:], outcomes)]
    expected, completed, _ = report.load_runs(directories)
    totals, pairs, comparisons = report.aggregate(expected, completed)
    assert len(completed) == 9
    selected = next(row for row in totals if row["level"] == "all" and row["mode"] == "v5")
    assert (selected["goals"], selected["collisions"], selected["timeouts"]) == (1, 1, 1)
    assert selected["goal_rate"] == pytest.approx(1 / 3)
    assert len([row for row in pairs if row["candidate"] == "v5" and row["reference"] == "off"]) == 3
    off = next(row for row in comparisons if row["level"] == "all" and row["candidate"] == "v5" and row["reference"] == "off")
    assert (off["gained_goal"], off["lost_goal"], off["net_goals"]) == (1, 1, 0)
    assert (off["rescued_collision"], off["introduced_collision"]) == (1, 1)
    v4 = next(row for row in comparisons if row["level"] == "all" and row["candidate"] == "v5" and row["reference"] == "v4")
    # Target collision -> timeout counts as avoiding collision but not gaining a goal.
    assert (v4["gained_goal"], v4["lost_goal"], v4["rescued_collision"]) == (0, 1, 1)
    assert (v4["collision_to_goal"], v4["collision_to_timeout"]) == (0, 1)
    assert (v4["candidate_collisions"], v4["reference_collisions"]) == (1, 1)
    assert (v4["candidate_timeouts"], v4["reference_timeouts"]) == (1, 0)
    assert v4["paired_cases"] == 3 and v4["complete_pairs"]


def test_partial_journal_is_read_only_and_exposes_unmatched_cases(tmp_path):
    cases, outcomes = _fixture(tmp_path)
    del outcomes["v5", "frozen_r", "same-id"]
    path = _run(tmp_path / "run", cases, outcomes)
    journal = path / "episodes.jsonl"
    data = journal.read_bytes() + b'{"mode":"v5"'
    journal.write_bytes(data)
    with pytest.raises(ValueError, match="Uncommitted journal tail"):
        report.load_runs([path])
    expected, completed, inputs = report.load_runs([path], allow_partial=True)
    assert journal.read_bytes() == data
    assert inputs[0]["ignored_uncommitted_bytes"] == len(b'{"mode":"v5"')
    _, pairs, comparisons = report.aggregate(expected, completed)
    summary = next(row for row in comparisons if row["level"] == "all" and row["candidate"] == "v5" and row["reference"] == "off")
    assert (summary["paired_cases"], summary["missing_candidate"], summary["missing_reference"]) == (2, 1, 0)
    assert not summary["complete_pairs"]
    assert len([row for row in pairs if row["candidate"] == "v5" and row["reference"] == "off"]) == 2


@pytest.mark.parametrize("failure", ["duplicate", "seed", "invalid_outcome", "unknown"])
def test_rejects_invalid_completed_records(tmp_path, failure):
    case = _case("dev_field")
    path = _run(tmp_path / "run", [case], {("v5", "dev_field", "same-id"): "goal"}, modes=("v5",))
    journal = path / "episodes.jsonl"
    row = json.loads(journal.read_text())
    if failure == "duplicate":
        journal.write_text(journal.read_text() * 2)
    else:
        row.update({"seed": 8} if failure == "seed" else
                   {"outcome": "diagnostic_limit"} if failure == "invalid_outcome" else {"case": "absent"})
        journal.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        report.load_runs([path])


def test_rejects_changed_provenance_or_duplicate_input(tmp_path):
    case = _case("dev_field")
    rows = {("v5", "dev_field", "same-id"): "goal"}
    a = _run(tmp_path / "a", [case], rows, modes=("v5",))
    with pytest.raises(ValueError, match="Duplicate declared"):
        report.load_runs([a, a])
    b = _run(tmp_path / "b", [_case("dev_width")], {}, modes=("off",), source_sha256={"env.py": "changed"})
    with pytest.raises(ValueError, match="Incompatible evaluation provenance"):
        report.load_runs([a, b], allow_partial=True)


def test_cross_mode_scene_seed_mismatch_is_rejected(tmp_path):
    a = _run(tmp_path / "a", [_case("dev_field")], {}, modes=("off",))
    b = _run(tmp_path / "b", [_case("dev_field", seed=99)], {}, modes=("v5",))
    with pytest.raises(ValueError, match="Cross-mode seed/digest mismatch"):
        report.load_runs([a, b], allow_partial=True)


def test_cli_does_not_overwrite_reports(tmp_path, monkeypatch):
    cases, outcomes = _fixture(tmp_path)
    path = _run(tmp_path / "run", cases, outcomes)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["suite_compare.py", str(path), "--tag", "test"])
    report.main()
    result = tmp_path / "results/safety_dev/test_outcomes.csv"
    before = result.read_bytes()
    with pytest.raises(SystemExit):
        report.main()
    assert result.read_bytes() == before


def test_aggregate_rates_pool_episode_counts_instead_of_component_means():
    expected, completed = {}, {}
    for component, count in (("frozen_a", 1), ("frozen_b", 9)):
        for i in range(count):
            case = _case(component, str(i), i)
            key = ("off", component, str(i))
            expected[key] = case
            completed[key] = dict(case, mode="off", outcome="goal" if component == "frozen_a" else "timeout")
    totals, _, _ = report.aggregate(expected, completed)
    overall = next(row for row in totals if row["level"] == "all")
    assert (overall["completed_cases"], overall["goals"], overall["timeouts"]) == (10, 1, 9)
    assert overall["goal_rate"] == pytest.approx(0.1)  # not (1.0 + 0.0) / 2
    assert overall["collision_rate"] == 0.0


def test_full_benchmark_requires_all_8670_keys_and_each_component_mode():
    expected = {(mode, component, str(i)): {} for mode in report.FULL_BENCHMARK_MODES
                for component, count in report.FULL_BENCHMARK_COUNTS.items() for i in range(count)}
    assert len(expected) == 8670
    assert report.benchmark_coverage(expected, expected)["complete"]
    completed = dict(expected)
    completed.pop(("off", "frozen_b", "640"))
    coverage = report.benchmark_coverage(expected, completed)
    assert not coverage["complete"]
    assert coverage["coverage_mismatches"] == [{"mode": "off", "component": "frozen_b",
        "required": 800, "declared": 800, "completed": 799}]
    subset = {key: value for key, value in expected.items() if key[1] != "field_validation"}
    assert not report.benchmark_coverage(subset, subset)["complete"]


def test_complete_subset_cannot_be_presented_as_final_full_benchmark(tmp_path, monkeypatch):
    cases, outcomes = _fixture(tmp_path)
    path = _run(tmp_path / "run", cases, outcomes)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["suite_compare.py", str(path), "--tag", "final", "--require-full-benchmark"])
    with pytest.raises(SystemExit):
        report.main()
    assert not (tmp_path / "results/safety_dev/final_outcomes.csv").exists()


def test_missing_interior_committed_record_is_rejected_without_any_partial_tail(tmp_path):
    cases, outcomes = _fixture(tmp_path)
    del outcomes["off", "frozen_r", "same-id"]
    path = _run(tmp_path / "run", cases, outcomes)
    with pytest.raises(ValueError, match="Incomplete run"):
        report.load_runs([path])


def test_report_inputs_use_shared_reader(tmp_path, monkeypatch):
    cases, outcomes = _fixture(tmp_path)
    path = _run(tmp_path / "run", cases, outcomes)
    calls = []
    original = report.read_shared_bytes
    def shared(file):
        calls.append(Path(file).name)
        return original(file)
    monkeypatch.setattr(report, "read_shared_bytes", shared)
    report.load_runs([path])
    assert calls == ["manifest.json", "episodes.jsonl"]


def test_development_compare_includes_selected_v4_without_changing_prior_refs(tmp_path, monkeypatch):
    compare = _module("compare")
    root = tmp_path / "results"
    (root / "safety_dev").mkdir(parents=True)
    (root / "safety_v2_dev_iter2.csv").write_text("mode,case,outcome\noff,A,collision:boundary\nv2,A,timeout\n")
    (root / "safety_v2_dev_v3_iter2_refpolicy.csv").write_text("mode,case,outcome\nv3,A,goal\n")
    (root / "safety_dev/dev_v4_observer_memory.csv").write_text("mode,case,outcome\nv4,A,collision:target\n")
    candidate = root / "candidate.csv"
    candidate.write_text("mode,case,outcome\nv5,A,goal\n")
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare.py", str(candidate), "--mode", "v5"])
    compare.main()
    with (root / "candidate_comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["reference"] for row in rows] == ["off", "v2_iter2", "v3", "v4_selected"]
    assert [row["gained"] for row in rows] == ["1", "1", "0", "1"]
