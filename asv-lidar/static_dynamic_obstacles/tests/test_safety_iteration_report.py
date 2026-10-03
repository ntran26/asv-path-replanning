"""Synthetic saved-artifact validation; no environment or policy imports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/report_safety_iteration.py"
spec = importlib.util.spec_from_file_location("iteration_report_test", SCRIPT)
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def change(path, **values):
    data = json.loads(path.read_text())
    data.update(values)
    write(path, data)


def fixture(tmp_path, tag="candidate", mode="v10", outcome="goal", case_name="TS2:A"):
    directory = tmp_path / tag
    directory.mkdir()
    case = dict(case=case_name, dataset="ts2", seed=42, scenario_sha256="a" * 64,
                selection_stratum="active", historical={"policy_outcome": "goal", "v8_outcome": "collision:target"})
    selection = tmp_path / (tag + "_selection.json")
    write(selection, {"cases": [case]})
    source = b"# frozen controller\n"
    with zipfile.ZipFile(directory / "evaluated_sources.zip", "w") as archive:
        archive.writestr("src/safety_v10.py", source)
    manifest = dict(selection_source=str(selection), selection_sha256=report.sha(selection.read_bytes()),
                    cases=[case], modes=[mode], planned_runs=1, checkpoint_sha256="b" * 64,
                    config_sha256="c" * 64, source_sha256={"src/safety_v10.py": report.sha(source)},
                    constants={"DT": "0.05"}, low_speed_start_frac=0.0,
                    filter_classes={mode: "safety_v10.SafetyFilterV10"}, constructor_options={mode: {}})
    write(directory / "manifest.json", manifest)
    identity = dict(attempt=1, case=case_name, mode=mode, seed=42)
    write(directory / "attempts/001.json", identity)
    result = dict(identity, dataset="ts2", stratum="active", elapsed_s=1.2, outcome=outcome,
                  collided=outcome.startswith("collision:"), collided_target=outcome == "collision:target",
                  steps=2, safety_v2_steps=1, safety_v2_brake_steps=0,
                  why_counts={"turn": 1, "idle": 1})
    write(directory / "attempts/001_result.json", result)
    trace = [dict(step=i + 1, changed=i == 0, brake=False, rudder_command=0.2,
                  signed_rpm_command=6.0, pre_state=[0., 0., 0., 0.5], post_state=[0., 0.1, 0., 0.5],
                  policy_action=[0.2, 0.5], filter={"why": "turn", "v10_pair_checked": True} if i == 0 else {}) for i in range(2)]
    (directory / "traces").mkdir()
    (directory / "traces" / f"001_{mode}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    write(directory / "completion.json", dict(completed_runs=1, elapsed_s=2.0, source_drift=[],
                                               checkpoint_unchanged=True, selection_unchanged=True))
    return directory


def load(path, interim=False):
    return report.load_run(path, report.Reader(), interim)


def test_complete_artifacts_and_mechanism_counts(tmp_path):
    run = load(fixture(tmp_path))
    assert run["complete"] and run["attempted"] == 1
    assert run["statistics"][("TS2:A", "v10")]["mechanism_steps"]["v10_pair_checked"] == 1


def test_missing_completion_blocks_final_but_interim_is_read_only(tmp_path):
    path = fixture(tmp_path)
    (path / "completion.json").unlink()
    with pytest.raises(ValueError, match="requires completion"):
        load(path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    value = report.report([path], interim=True)
    assert value["iterations"][0]["completed"] == 1
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("target,changes,match", [
    ("attempts/001.json", {"seed": 43}, "seed"),
    ("attempts/001_result.json", {"seed": 43}, "identity"),
    ("attempts/001_result.json", {"safety_v2_steps": 0}, "counter"),
    ("attempts/001_result.json", {"safety_v2_brake_steps": 1}, "counter"),
    ("attempts/001_result.json", {"why_counts": {"nominal": 2}}, "reason"),
    ("attempts/001_result.json", {"outcome": "crash"}, "Unknown outcome"),
    ("attempts/001_result.json", {"collided": True}, "Collision flag"),
    ("completion.json", {"source_drift": ["src/env.py"]}, "drift"),
    ("completion.json", {"completed_runs": 2}, "count"),
])
def test_rejects_inconsistent_durable_evidence(tmp_path, target, changes, match):
    path = fixture(tmp_path)
    change(path / target, **changes)
    with pytest.raises(ValueError, match=match):
        load(path)


@pytest.mark.parametrize("filename", ["attempts/001_error.json", "traces/999_v10.jsonl"])
def test_orphan_artifacts_rejected(tmp_path, filename):
    path = fixture(tmp_path)
    (path / filename).write_text("{}\n")
    with pytest.raises(ValueError, match="artifact"):
        load(path)


def test_archive_tamper_rejected(tmp_path):
    path = fixture(tmp_path)
    with zipfile.ZipFile(path / "evaluated_sources.zip", "w") as archive:
        archive.writestr("src/safety_v10.py", "changed source")
    with pytest.raises(ValueError, match="source hash"):
        load(path)


def test_selection_tamper_rejected(tmp_path):
    path = fixture(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    Path(manifest["selection_source"]).write_text("{}")
    with pytest.raises(ValueError, match="Selection hash"):
        load(path)


@pytest.mark.parametrize("mutation,match", [("order", "step order"), ("tail", "tail"), ("nan", "Nonfinite")])
def test_invalid_completed_trace_rejected(tmp_path, mutation, match):
    path = fixture(tmp_path)
    trace = path / "traces/001_v10.jsonl"
    raw = trace.read_text()
    if mutation == "order":
        trace.write_text("\n".join(reversed(raw.splitlines())) + "\n")
    elif mutation == "tail":
        trace.write_text(raw.rstrip())
    else:
        trace.write_text(raw.replace('"rudder_command": 0.2', '"rudder_command": NaN'))
    with pytest.raises(ValueError, match=match):
        load(path)


def test_duplicate_case_mode_tokens_rejected(tmp_path):
    path = fixture(tmp_path)
    token = json.loads((path / "attempts/001.json").read_text())
    token["attempt"] = 2
    write(path / "attempts/002.json", token)
    with pytest.raises(ValueError, match="excessive"):
        load(path)


def test_fresh_reference_identity_and_checkpoint_must_match(tmp_path):
    run = load(fixture(tmp_path, "candidate"))
    pilot = load(fixture(tmp_path, "pilot", mode="v8"))
    assert report.compatibility(run, pilot, "TS2:A") == {}
    for section, key, wrong, message in [
        ("manifest", "checkpoint_sha256", "d" * 64, "checkpoint"),
        ("manifest", "config_sha256", "d" * 64, "config"),
        ("case", "scenario_sha256", "d" * 64, "scene/seed"),
        ("case", "seed", 50, "scene/seed"),
    ]:
        other = copy.deepcopy(pilot)
        target = other["manifest"] if section == "manifest" else other["cases"]["TS2:A"]
        target[key] = wrong
        with pytest.raises(ValueError, match=message):
            report.compatibility(run, other, "TS2:A")


def test_source_differences_disclosed_not_called_equal(tmp_path):
    run = load(fixture(tmp_path, "candidate"))
    pilot = copy.deepcopy(run)
    pilot["manifest"]["source_sha256"]["src/field_training.py"] = "d" * 64
    assert "src/field_training.py" in report.compatibility(run, pilot, "TS2:A")


def test_historical_and_fresh_references_remain_distinct(tmp_path):
    run = load(fixture(tmp_path, "candidate"))
    pilot = load(fixture(tmp_path, "pilot", mode="v8", outcome="collision:obstacle"))
    history = {"TS2:A": dict(seed="42", policy_outcome="goal", v8_outcome="collision:target")}
    outcome, evidence, _ = report.reference_for(run, pilot, history, "TS2:A", "v8", "v10")
    assert (outcome, evidence) == ("collision:obstacle", "fresh_prior_pilot")
    outcome, evidence, _ = report.reference_for(run, None, history, "TS2:A", "v8", "v10")
    assert (outcome, evidence) == ("collision:target", "historical_saved_branch")
    assert report.reference_for(run, None, history, "TS2:A", "v9", "v10")[1] == "unavailable"
    history["TS2:A"]["v8_outcome"] = "goal"
    with pytest.raises(ValueError, match="selection/audit"):
        report.reference_for(run, None, history, "TS2:A", "v8", "v10")


def test_collision_to_timeout_is_not_a_rescued_goal():
    rows = [dict(case="A", outcome="timeout", off_outcome="collision:target"),
            dict(case="B", outcome="goal", off_outcome="collision:obstacle"),
            dict(case="C", outcome="collision:boundary", off_outcome="goal")]
    stats = report.paired_counts(rows, "off")
    assert stats["collision_to_timeout"] == 1 and stats["collision_to_goal"] == 1
    assert stats["gained_cases"] == ["B"] and stats["lost_cases"] == ["C"]
    assert stats['reference_goals'] == 1 and stats['preserved_reference_goals'] == 0
    assert stats['reference_failures'] == 2 and stats['unresolved_reference_failures'] == 1


def test_strata_keep_denominators_and_reference_provenance(tmp_path):
    run = load(fixture(tmp_path))
    history = {"TS2:A": dict(seed="42", policy_outcome="goal", v8_outcome="collision:target")}
    paired, summary = report.summarize(run, None, history)
    assert paired[0]["v8_gained_goal"] and paired[0]["v8_evidence"] == "historical_saved_branch"
    assert len(summary["groups"]) == 3
    assert all(g["n"] == 1 for g in summary["groups"])
    assert all(p["evidence"] == "historical_saved_branch" for g in summary["groups"] for p in g["comparisons"])


def test_canonical_seed_and_scene_guards_even_without_fresh_overlap(tmp_path):
    run = load(fixture(tmp_path))
    canonical = {"TS2:A": {k: run["cases"]["TS2:A"][k] for k in report.IDENTITY}}
    report.validate_canonical_cases(run, canonical)
    canonical["TS2:A"]["scenario_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="Canonical scene/seed"):
        report.validate_canonical_cases(run, canonical)


def test_final_report_writes_only_new_artifacts_and_refuses_overwrite(tmp_path, monkeypatch):
    path = fixture(tmp_path)
    run = load(path)
    canonical = {"TS2:A": {k: run["cases"]["TS2:A"][k] for k in report.IDENTITY}}
    monkeypatch.setattr(report, "canonical_inventory", lambda reader: canonical)
    monkeypatch.setattr(report, "BASELINE_AUDIT", tmp_path / "nonexistent_audit.json")
    history = tmp_path / "history.csv"
    history.write_text("case,seed,policy_outcome,v8_outcome\nTS2:A,42,goal,collision:target\n")
    output = tmp_path / "new_report"
    before = (path / "manifest.json").read_bytes()
    summary = report.report([path], output, pilot_directory=None, audit_path=history)
    assert summary["iterations"][0]["completed"] == 1
    assert {p.name for p in output.iterdir()} == {"report.md", "paired.csv", "summary.json", "provenance.json"}
    assert (path / "manifest.json").read_bytes() == before
    assert b"\r\n" not in (output / "report.md").read_bytes()
    with pytest.raises(ValueError, match="refusing overwrite"):
        report.report([path], output, pilot_directory=None, audit_path=history)


def update_trace(path, details):
    target = next((path / "traces").glob("*.jsonl"))
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    rows[0]["filter"].update(details)
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_missing_diagnostics_are_distinct_from_explicit_false(tmp_path):
    path = fixture(tmp_path, mode="v11_prefix")
    update_trace(path, {"v11_policy_prefix_search_checked": False, "v11_policy_prefix_preserved": False})
    run = load(path)
    history = {"TS2:A": dict(seed="42", policy_outcome="goal", v8_outcome="collision:target")}
    _, summary = report.summarize(run, None, history)
    events = summary["mechanisms"]["v11_prefix"]["events"]
    assert events["v11_policy_prefix_preserved"]["steps"] == 0
    assert events["v11_policy_prefix_preserved"]["observed_steps"] == 1
    assert events["v9_override_suppressed"]["steps"] is None
    assert summary["mechanisms"]["v11_prefix"]["track_persistence"]["admission_events"] is None


def test_v11_prefix_and_persistence_counts_are_hypothesis_exposure(tmp_path):
    path = fixture(tmp_path, mode="v11_combined")
    update_trace(path, {
        "v11_policy_prefix_search_checked": True, "v11_policy_prefix_preserved": True,
        "policy_prefix_search": {"evaluated_plans": 192, "accepted_plans": 3, "hard_passing_plans": 8},
        "track_persistence": {"added_hypotheses": 2,
            "hypotheses": [{"source_id": 9, "age_s": 0.}, {"source_id": 10, "age_s": 1.}],
            "updates": [{"kind": "admission"}, {"kind": "measured_refresh"}],
            "expired_source_ids": [2], "contradicted_source_ids": []}})
    run = load(path)
    history = {"TS2:A": dict(seed="42", policy_outcome="goal", v8_outcome="collision:target")}
    _, summary = report.summarize(run, None, history)
    value = summary["mechanisms"]["v11_combined"]
    assert value["events"]["v11_policy_prefix_preserved"]["steps"] == 1
    assert value["track_persistence"]["admission_events"] == 1
    assert value["track_persistence"]["added_hypothesis_decisions"] == 2
    assert value["track_persistence"]["coasted_hypothesis_decisions"] == 1
    assert value["policy_prefix_search"]["evaluated_plans"] == 192


@pytest.mark.parametrize("details", [{"error": "failed"}, {"filter_error": "failed"}, {"mode": "error"}])
def test_silent_error_fallback_cannot_earn_a_goal(tmp_path, details):
    path = fixture(tmp_path)
    update_trace(path, details)
    with pytest.raises(ValueError, match="Filter error"):
        load(path)


def test_repeated_cases_across_tags_are_counted_separately_from_coverage(tmp_path, monkeypatch):
    one = fixture(tmp_path, "one")
    two = fixture(tmp_path, "two")
    run = load(one)
    canonical = {"TS2:A": {k: run["cases"]["TS2:A"][k] for k in report.IDENTITY}}
    monkeypatch.setattr(report, "canonical_inventory", lambda reader: canonical)
    monkeypatch.setattr(report, "BASELINE_AUDIT", tmp_path / "nonexistent_audit.json")
    history = tmp_path / "history.csv"
    history.write_text("case,seed,policy_outcome,v8_outcome\nTS2:A,42,goal,collision:target\n")
    result = report.report([one, two], tmp_path / "report", pilot_directory=None, audit_path=history)
    assert result["completed_iteration_records"] == 2
    assert result["distinct_cases"] == 1
    assert result["repeated_case_record_counts"] == {"TS2:A": 2}


def disjoint_runs(tmp_path):
    one = load(fixture(tmp_path, "one", case_name="TS2:A"))
    two = load(fixture(tmp_path, "two", case_name="TS2:B"))
    return one, two


def test_disjoint_cohort_allows_only_optional_snapshot_difference(tmp_path):
    one, two = disjoint_runs(tmp_path)
    one["manifest"]["snapshots"] = True
    two["manifest"]["snapshots"] = False
    one["manifest"]["source_sha256"]["tools/diagnostics/safety/trace_snapshot.py"] = "a" * 64
    joined = report.combine_cohort([one, two], "joined")
    assert len(joined["cases"]) == len(joined["results"]) == 2
    assert joined["attempted"] == joined["manifest"]["planned_runs"] == 2
    assert joined["origin_by_key"][("TS2:B", "v10")]["tag"] == "two"
    assert joined["cohort_components"][1]["source_inventory_difference_from_first"] == ["tools/diagnostics/safety/trace_snapshot.py"]


@pytest.mark.parametrize("field,value", [
    ("filter_classes", {"v10": "different.Filter"}),
    ("constructor_options", {"v10": {"prefer_any_feasible_policy": True}}),
    ("checkpoint_sha256", "d" * 64),
    ("config_sha256", "d" * 64),
    ("constants", {"DT": "0.1"}),
    ("torch_threads", 2),
])
def test_cohort_rejects_configuration_drift(tmp_path, field, value):
    one, two = disjoint_runs(tmp_path)
    two["manifest"][field] = value
    with pytest.raises(ValueError, match="Cohort"):
        report.combine_cohort([one, two], "joined")


def test_cohort_rejects_shared_source_drift_and_unknown_extra_source(tmp_path):
    one, two = disjoint_runs(tmp_path)
    two["manifest"]["source_sha256"]["src/safety_v10.py"] = "d" * 64
    with pytest.raises(ValueError, match="shared source"):
        report.combine_cohort([one, two], "joined")
    two["manifest"]["source_sha256"] = dict(one["manifest"]["source_sha256"], **{"src/unknown.py": "e" * 64})
    with pytest.raises(ValueError, match="beyond optional"):
        report.combine_cohort([one, two], "joined")


def test_cohort_rejects_duplicate_scenarios_and_incomplete_parts(tmp_path):
    one, two = disjoint_runs(tmp_path)
    with pytest.raises(ValueError, match="Duplicate scenario"):
        report.combine_cohort([one, copy.deepcopy(one)], "joined")
    two["complete"] = False
    with pytest.raises(ValueError, match="complete component"):
        report.combine_cohort([one, two], "joined")


def test_extra_fresh_v10_reference_is_labeled_and_checked(tmp_path):
    candidate = load(fixture(tmp_path, "candidate", mode="v11", outcome="goal"))
    reference = load(fixture(tmp_path, "v10_reference", mode="v10", outcome="collision:target"))
    history = {"TS2:A": dict(seed="42", policy_outcome="goal", v8_outcome="collision:target")}
    pairs, summary = report.summarize(candidate, None, history, [reference])
    assert pairs[0]["v10_outcome"] == "collision:target"
    assert pairs[0]["v10_evidence"] == "fresh_reference_iteration"
    assert pairs[0]["v10_gained_goal"]
    assert summary["fresh_reference_provenance"]["v10_reference"]["overlapping_cases"] == 1
    with pytest.raises(ValueError, match="Ambiguous"):
        report.reference_for(candidate, None, history, "TS2:A", "v10", "v11", [reference, reference])


def test_complete_disjoint_cohort_report_retains_origins(tmp_path, monkeypatch):
    one_path = fixture(tmp_path, "one", case_name="TS2:A")
    two_path = fixture(tmp_path, "two", case_name="TS2:B")
    runs = [load(one_path), load(two_path)]
    canonical = {case: {k: row[k] for k in report.IDENTITY} for run in runs for case, row in run["cases"].items()}
    monkeypatch.setattr(report, "canonical_inventory", lambda reader: canonical)
    monkeypatch.setattr(report, "BASELINE_AUDIT", tmp_path / "no_audit.json")
    history = tmp_path / "history.csv"
    history.write_text("case,seed,policy_outcome,v8_outcome\nTS2:A,42,goal,collision:target\nTS2:B,42,goal,collision:target\n")
    output = tmp_path / "joined_report"
    summary = report.report([one_path, two_path], output, pilot_directory=None, audit_path=history, cohort="joined")
    assert len(summary["iterations"]) == 1
    assert summary["completed_iteration_records"] == summary["distinct_cases"] == 2
    assert summary["iterations"][0]["cohort_components"][1]["tag"] == "two"
    assert 'sum of component runner durations' in (output / "report.md").read_text()


@pytest.mark.parametrize("env_changed", [False, True])
def test_native_dispatch_audit_linked_only_for_relevant_source_difference(tmp_path, monkeypatch, env_changed):
    audit = tmp_path / "native_audit.json"
    write(audit, {"before": "a" * 64, "after": "b" * 64})
    monkeypatch.setattr(report, "NATIVE_DISPATCH_AUDIT", audit)
    monkeypatch.setattr(report, "BASELINE_AUDIT", tmp_path / "no_baseline.json")
    summary = {"iterations": [{"pilot_source_differences": {"src/env.py": {}} if env_changed else {},
                                "fresh_reference_provenance": {}}]}
    reader = report.Reader()
    report.attach_source_audits(summary, reader)
    assert ("native_dispatch_source_audit" in summary) == env_changed
    if env_changed:
        assert summary["native_dispatch_source_audit"]["sha256"] == report.sha(audit.read_bytes())


def multi_fixture(tmp_path, tag="multimode", case_name="TS2:A"):
    path = fixture(tmp_path, tag, mode="v12", case_name=case_name)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["modes"] = ["v12", "v13"]
    manifest["planned_runs"] = 2
    manifest["filter_classes"]["v13"] = "safety_v13.SafetyFilterV13"
    manifest["constructor_options"]["v13"] = {}
    write(path / "manifest.json", manifest)
    token = json.loads((path / "attempts/001.json").read_text())
    token.update(attempt=2, mode="v13")
    write(path / "attempts/002.json", token)
    result = json.loads((path / "attempts/001_result.json").read_text())
    result.update(attempt=2, mode="v13")
    write(path / "attempts/002_result.json", result)
    (path / "traces/002_v13.jsonl").write_bytes((path / "traces/001_v12.jsonl").read_bytes())
    change(path / "completion.json", completed_runs=2)
    return path


def test_mode_selection_preserves_source_counts_and_original_attempts(tmp_path):
    run = load(multi_fixture(tmp_path))
    view = report.select_run_mode(run, "v13")
    assert view["manifest"]["modes"] == ["v13"]
    assert view["results"][("TS2:A", "v13")]["attempt"] == 2
    assert view["attempted"] == 1 and view["mode_selection"]["source_attempted"] == 2
    assert view["mode_selection"]["excluded_modes"] == ["v12"]
    assert run["manifest"]["modes"] == ["v12", "v13"]
    assert run["attempted"] == 2


def test_multimode_component_can_join_matching_single_mode(tmp_path):
    one = report.select_run_mode(load(multi_fixture(tmp_path)), "v12")
    two = report.select_run_mode(load(fixture(tmp_path, "remaining", mode="v12", case_name="TS2:B")), "v12")
    joined = report.combine_cohort([one, two], "v12_complete")
    assert len(joined["results"]) == 2
    assert joined["cohort_components"][0]["mode_selection"]["source_completed_runs"] == 2
    assert joined["cohort_components"][0]["completed"] == 1


def test_reference_mode_spec_validates_excluded_mode_before_slicing(tmp_path):
    path = multi_fixture(tmp_path)
    reference = report.load_reference("v12=" + str(path), report.Reader())
    assert reference["manifest"]["modes"] == ["v12"]
    change(path / "attempts/002_result.json", safety_v2_steps=99)
    with pytest.raises(ValueError, match="counter"):
        report.load_reference("v12=" + str(path), report.Reader())


def test_unknown_reference_mode_and_incomplete_mode_selection_rejected(tmp_path):
    path = multi_fixture(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        report.load_reference("v99=" + str(path), report.Reader())
    run = load(path)
    run["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        report.select_run_mode(run, "v12")


def add_archived_source(run, name, raw):
    run["manifest"]["source_sha256"][name] = report.sha(raw)
    run["archived_sources"][name] = raw


def test_inactive_higher_version_needs_explicit_opt_in_and_records_audit(tmp_path):
    one, two = disjoint_runs(tmp_path)
    add_archived_source(two, "src/safety_v14.py", b"# future unused candidate\n")
    with pytest.raises(ValueError, match="beyond optional"):
        report.combine_cohort([one, two], "joined")
    joined = report.combine_cohort([one, two], "joined", allow_inactive_version_additions=True)
    audit = joined["cohort_components"][1]["inactive_version_source_audit"][0]
    assert audit["added_version"] == 14 and audit["selected_filter_versions"] == [10]
    assert audit["literal_module_or_class_references"] == []
    assert audit["sha256"] == report.sha(b"# future unused candidate\n")


@pytest.mark.parametrize("extra,reference", [
    ("src/helper.py", b"# no reference\n"),
    ("src/safety_v9.py", b"# no reference\n"),
    ("src/safety_v14.py", b"import safety_v14\n"),
    ("src/safety_v14.py", b"cls = SafetyFilterV14\n"),
])
def test_inactive_allowance_rejects_unknown_older_or_referenced_module(tmp_path, extra, reference):
    one, two = disjoint_runs(tmp_path)
    for run in (one, two):
        add_archived_source(run, "src/shared.py", reference)
    add_archived_source(two, extra, b"# added\n")
    with pytest.raises(ValueError, match="Inactive"):
        report.combine_cohort([one, two], "joined", allow_inactive_version_additions=True)


def test_later_added_source_hashes_must_also_match_each_other(tmp_path):
    one, two = disjoint_runs(tmp_path)
    three = load(fixture(tmp_path, "three", case_name="TS2:C"))
    add_archived_source(two, "src/safety_v14.py", b"# first added version\n")
    add_archived_source(three, "src/safety_v14.py", b"# changed added version\n")
    with pytest.raises(ValueError, match="shared source"):
        report.combine_cohort([one, two, three], "joined", allow_inactive_version_additions=True)


def test_prefix_requests_are_version_agnostic_and_missing_is_unrecorded(tmp_path):
    path = fixture(tmp_path)
    update_trace(path, {'v15_prefix_request_checked': True,
                        'v15_prefix_request_reason': 'ordinary_parent_below_trigger_margin',
                        'v15_prefix_skip_reason': 'missing_snapshot'})
    run = load(path)
    history = {'TS2:A': dict(seed='42', policy_outcome='goal', v8_outcome='collision:target')}
    _, summary = report.summarize(run, None, history)
    stats = summary['mechanisms']['v10']
    assert stats['prefix_requests']['observed_decisions'] == 1
    assert stats['prefix_requests']['checked_decisions'] == 1
    assert stats['prefix_requests']['request_reason_counts'] == {'ordinary_parent_below_trigger_margin': 1}
    assert stats['prefix_requests']['skip_reason_counts'] == {'missing_snapshot': 1}
    assert stats['motion_axis_geometry']['replacement_decisions'] is None
    unrecorded = load(fixture(tmp_path, 'unrecorded'))
    _, summary = report.summarize(unrecorded, None, history)
    assert summary['mechanisms']['v10']['prefix_requests']['request_reason_counts'] is None


def test_motion_axis_counts_repeated_id_observations_not_distinct_vessels(tmp_path):
    path = fixture(tmp_path)
    trace = path/'traces/001_v10.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    for row in rows:
        row['filter']['motion_axis_geometry'] = {'replaced_source_ids': [7],
            'updates': [{'source_id': 7}], 'rejections': {'low_speed': 2}}
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    run = load(path)
    history = {'TS2:A': dict(seed='42', policy_outcome='goal', v8_outcome='collision:target')}
    _, summary = report.summarize(run, None, history)
    stats = summary['mechanisms']['v10']['motion_axis_geometry']
    assert stats['observed_decisions'] == stats['replacement_decisions'] == stats['replacement_id_events'] == 2
    assert stats['replacement_episodes'] == 1
    assert stats['rejection_reason_counts'] == {'low_speed': 4}


def test_recorded_skip_reason_does_not_invent_unrecorded_request_flag(tmp_path):
    path = fixture(tmp_path)
    update_trace(path, {'v11_prefix_skip_reason': 'unmodelled command limit'})
    history = {'TS2:A': dict(seed='42', policy_outcome='goal', v8_outcome='collision:target')}
    _, summary = report.summarize(load(path), None, history)
    stats = summary['mechanisms']['v10']['prefix_requests']
    assert stats['skip_reason_counts'] == {'unmodelled command limit': 1}
    assert stats['checked_observed_decisions'] == 0 and stats['checked_decisions'] is None


@pytest.mark.parametrize('details,match', [
    ({'v11_prefix_request_checked': 'false'}, 'flag'),
    ({'v11_prefix_request_reason': 0}, 'reason'),
    ({'v11_prefix_request_checked': False, 'v16_prefix_request_checked': True}, 'aliases'),
    ({'motion_axis_geometry': {'replaced_source_ids': [7, 7], 'updates': [], 'rejections': {}}}, 'IDs'),
    ({'motion_axis_geometry': {'replaced_source_ids': [7], 'updates': [{'source_id': 8}], 'rejections': {}}}, 'mismatch'),
])
def test_invalid_new_diagnostic_counters_rejected(tmp_path, details, match):
    path = fixture(tmp_path)
    update_trace(path, details)
    with pytest.raises(ValueError, match=match): load(path)


def test_reference_label_keeps_original_mode_and_separates_configurations(tmp_path):
    one = fixture(tmp_path, 'default', mode='v14', outcome='goal')
    two = fixture(tmp_path, 'prefix', mode='v14', outcome='collision:target')
    reference = report.load_reference('v14_prefix::'+str(two), report.Reader())
    assert reference['reference_label'] == {'label': 'v14_prefix', 'original_mode': 'v14'}
    assert reference['results'][('TS2:A', 'v14_prefix')]['mode'] == 'v14'
    assert json.loads((two/'manifest.json').read_text())['modes'] == ['v14']
    candidate = load(fixture(tmp_path, 'candidate', mode='v15'))
    history = {'TS2:A': dict(seed='42', policy_outcome='goal', v8_outcome='collision:target')}
    paired, _ = report.summarize(candidate, None, history, [load(one), reference])
    assert paired[0]['v14_outcome'] == 'goal'
    assert paired[0]['v14_prefix_outcome'] == 'collision:target'
    change(two/'attempts/001_result.json', safety_v2_steps=20)
    with pytest.raises(ValueError, match='counter'):
        report.load_reference('v14_prefix::'+str(two), report.Reader())


def test_label_requires_selected_single_mode_and_cannot_impersonate_off(tmp_path):
    path = multi_fixture(tmp_path)
    with pytest.raises(ValueError, match='one selected mode'):
        report.load_reference('alias::'+str(path), report.Reader())
    assert report.load_reference('alias::v12='+str(path), report.Reader())['reference_label']['original_mode'] == 'v12'
    with pytest.raises(ValueError, match='reserved'):
        report.load_reference('off::v12='+str(path), report.Reader())


@pytest.mark.parametrize('after_matches', [False, True])
def test_v15_audit_links_require_relevant_source_hash(tmp_path, monkeypatch, after_matches):
    source = 'src/safety_v6.py'
    integration, diagnostic = tmp_path/'integration.json', tmp_path/'diagnostic.json'
    write(integration, {'files': {source: {'before_sha256': 'old', 'after_sha256': 'new'}}})
    write(diagnostic, {'source': source, 'before_sha256': 'old', 'after_sha256': 'new'})
    monkeypatch.setattr(report, 'V15_INTEGRATION_AUDIT', integration)
    monkeypatch.setattr(report, 'V15_DIAGNOSTIC_AUDIT', diagnostic)
    summary = {'iterations': [{'pilot_source_differences': {source: {
        'candidate': 'new' if after_matches else 'different', 'reference': 'old'}}, 'fresh_reference_provenance': {}}]}
    report.attach_source_audits(summary, report.Reader())
    for key, path in [('v15_integration_source_audit', integration), ('v15_v6_diagnostic_ast_audit', diagnostic)]:
        assert (key in summary) == after_matches
        if after_matches:
            assert summary[key]['sha256'] == report.sha(path.read_bytes())
            assert summary[key]['applicable_changed_sources'] == [source]


@pytest.mark.parametrize('source,candidate,reference,expected', [
    ('src/env.py', 'new', 'old', True),
    ('src/env.py', 'older', 'new', True),
    ('src/env.py', 'unknown', 'old', False),
    ('src/unrelated.py', 'new', 'old', False),
])
def test_v16_dispatch_changes_schema_requires_relevant_archived_hash(
        tmp_path, monkeypatch, source, candidate, reference, expected):
    audit = tmp_path/'v16_native.json'
    write(audit, {'changes': [{'path': 'src/env.py', 'before_sha256': 'old',
                              'after_sha256': 'new', 'diff': 'dispatch branch only'}]})
    monkeypatch.setattr(report, 'V16_NATIVE_DISPATCH_AUDIT', audit)
    summary = {'iterations': [{'pilot_source_differences': {},
        'fresh_reference_provenance': {'reference': {'source_differences': {
            source: {'candidate': candidate, 'reference': reference}}}}}]}
    reader = report.Reader()
    report.attach_source_audits(summary, reader)
    key = 'v16_native_dispatch_source_audit'
    assert (key in summary) == expected
    if expected:
        assert summary[key]['sha256'] == report.sha(audit.read_bytes())
        assert summary[key]['applicable_changed_sources'] == ['src/env.py']


def fresh_off_paired_fixture(tmp_path):
    path = multi_fixture(tmp_path, 'new_paired', case_name='TS2:NEW')
    manifest = json.loads((path/'manifest.json').read_text())
    manifest.update(modes=['off', 'v16'], filter_classes={'off': None, 'v16': 'safety_v16.SafetyFilterV16'},
                    constructor_options={'off': {}, 'v16': {}})
    write(path/'manifest.json', manifest)
    for attempt, old_mode, mode in [(1, 'v12', 'off'), (2, 'v13', 'v16')]:
        for suffix in ['', '_result']:
            filename = path/f'attempts/{attempt:03d}{suffix}.json'
            values = {'mode': mode}
            if attempt == 1 and suffix:
                values.update(outcome='collision:obstacle', collided=True, collided_target=False,
                              safety_v2_steps=0, safety_v2_brake_steps=0, why_counts={'idle': 2})
            change(filename, **values)
        old = path/f'traces/{attempt:03d}_{old_mode}.jsonl'
        rows = [json.loads(line) for line in old.read_text().splitlines()]
        if mode == 'off':
            for row in rows: row.update(filter={}, changed=False, brake=False)
        (path/f'traces/{attempt:03d}_{mode}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        old.unlink()
    return path


def test_nonpilot_candidate_uses_fresh_off_and_separate_historical_v8(tmp_path):
    run = load(fresh_off_paired_fixture(tmp_path))
    pilot = load(fixture(tmp_path, 'old_pilot', mode='v8', case_name='TS2:OLD'))
    history = {'TS2:NEW': dict(seed='42', policy_outcome='goal', v8_outcome='collision:target')}
    paired, summary = report.summarize(run, pilot, history)
    candidate = next(r for r in paired if r['mode'] == 'v16')
    assert candidate['off_outcome'] == 'collision:obstacle'
    assert candidate['off_evidence'] == 'fresh_same_iteration' and candidate['off_gained_goal']
    assert candidate['v8_evidence'] == 'historical_saved_branch' and candidate['v8_outcome'] == 'collision:target'
    off = next(r for r in paired if r['mode'] == 'off')
    assert off['off_outcome'] == 'goal' and off['off_evidence'] == 'historical_saved_branch'
    assert off['off_lost_goal']  # Independent fresh SAC reproduction loss stays visible.
    overall = next(g for g in summary['groups'] if g['mode'] == 'v16' and g['dimension'] == 'overall')
    comparison = next(g for g in overall['comparisons'] if g['reference'] == 'off')
    assert comparison['matched_cases'] == 1 and comparison['gained_goals'] == 1
    assert comparison['reference_goals'] == comparison['preserved_reference_goals'] == 0
    assert comparison['reference_failures'] == 1 and comparison['unresolved_reference_failures'] == 0
    assert summary['broken_sac_success_inventory'] == []  # Fresh-off reproduction loss is not a safety loss.


def test_named_broken_sac_inventory_retains_stratum_and_evidence(tmp_path):
    run = load(fixture(tmp_path, outcome='collision:boundary'))
    history = {'TS2:A': dict(seed='42', policy_outcome='goal', v8_outcome='collision:target')}
    _, summary = report.summarize(run, None, history)
    broken = summary['broken_sac_success_inventory']
    assert len(broken) == 1 and broken[0]['case'] == 'TS2:A'
    assert broken[0]['stratum'] == 'active' and broken[0]['off_evidence'] == 'historical_saved_branch'


@pytest.mark.parametrize('bad', ['intervention', 'filter_class', 'missing_pair'])
def test_fresh_off_pair_guards(tmp_path, bad):
    path = fresh_off_paired_fixture(tmp_path)
    if bad == 'intervention':
        change(path/'attempts/001_result.json', safety_v2_steps=1)
        match = 'off control'
    elif bad == 'filter_class':
        manifest = json.loads((path/'manifest.json').read_text())
        manifest['filter_classes']['off'] = 'safety_v16.SafetyFilterV16'
        write(path/'manifest.json', manifest)
        match = 'off control'
    else:
        (path/'attempts/001_result.json').unlink()
        match = 'Missing completed'
    with pytest.raises(ValueError, match=match): load(path)


def test_ts3_canonical_inventory_keeps_same_bare_id_separate(tmp_path, monkeypatch):
    for version, seed, digest in [(2, 22, 'a' * 64), (3, 33, 'b' * 64)]:
        path = tmp_path/f'results/test_set/v{version}/definition.csv'
        path.parent.mkdir(parents=True)
        path.write_text(f'test_id,episode_seed,digest\nSHARED,{seed},{digest}\n', encoding='utf-8')
    write(tmp_path/'results/safety_dev/suites/inventory_v5/manifest.json', {'cases': []})
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    inventory = report.canonical_inventory(report.Reader())
    assert set(inventory) == {'TS2:SHARED', 'TS3:SHARED'}
    assert inventory['TS3:SHARED'] == dict(case='TS3:SHARED', dataset='ts3', seed=33,
                                        scenario_sha256='b' * 64)
    assert inventory['TS2:SHARED']['seed'] == 22
    with pytest.raises(ValueError, match='Canonical'):
        report.validate_canonical_cases({'cases': {'TS3:SHARED': dict(
            inventory['TS3:SHARED'], scenario_sha256='a' * 64)}}, inventory)


def test_ts3_fresh_off_pairs_without_assigning_v2_history(tmp_path):
    path = fresh_off_paired_fixture(tmp_path)
    manifest = json.loads((path/'manifest.json').read_text())
    case = dict(manifest['cases'][0], case='TS3:NEW', dataset='ts3')
    case.pop('historical')
    selection = Path(manifest['selection_source'])
    write(selection, {'cases': [case]})
    manifest.update(cases=[case], selection_sha256=report.sha(selection.read_bytes()))
    write(path/'manifest.json', manifest)
    for attempt in (1, 2):
        change(path/f'attempts/{attempt:03d}.json', case='TS3:NEW')
        change(path/f'attempts/{attempt:03d}_result.json', case='TS3:NEW', dataset='ts3')
    run = load(path)
    pilot = load(fixture(tmp_path, 'old_pilot', mode='v8', case_name='TS2:NEW'))
    history = {'TS2:NEW': dict(seed='42', policy_outcome='goal', v8_outcome='collision:target')}
    paired, summary = report.summarize(run, pilot, history)
    candidate = next(row for row in paired if row['mode'] == 'v16')
    assert candidate['off_evidence'] == 'fresh_same_iteration'
    assert candidate['off_outcome'] == 'collision:obstacle' and candidate['off_gained_goal']
    assert candidate['v8_evidence'] == 'unavailable' and candidate['v8_outcome'] == ''
    off = next(row for row in paired if row['mode'] == 'off')
    assert off['off_evidence'] == 'unavailable' and off['off_outcome'] == ''
    assert summary['broken_sac_success_inventory'] == []
    run['results'].pop(('TS3:NEW', 'off'))
    assert report.reference_for(run, pilot, history, 'TS3:NEW', 'off', 'v16') == ('', 'unavailable', '')


def test_ts3_scope_distinguishes_full_set_from_selected_subset():
    canonical = {'TS3:A': {}, 'TS3:B': {}, 'TS2:A': {}}
    runs = [{'tag': 'all_v3', 'cases': {'TS3:A': {}, 'TS3:B': {}}},
            {'tag': 'probe', 'cases': {'TS3:B': {}}}]
    scope = report.primary_benchmark_scope(runs, canonical)
    assert scope['defined_scenarios'] == scope['distinct_selected_scenarios'] == 2
    assert scope['iterations'] == [dict(tag='all_v3', selected_scenarios=2, coverage='full set'),
                                   dict(tag='probe', selected_scenarios=1, coverage='subset')]
    assert report.primary_benchmark_scope([{'tag': 'legacy', 'cases': {'TS2:A': {}}}], canonical) is None
