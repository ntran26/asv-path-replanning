"""Suite coverage, seed pairing and safe continuation without running episodes."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/suite_eval.py"
_SPEC = importlib.util.spec_from_file_location("suite_eval_under_test", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_all_suites_includes_all_development_field_and_explicit_frozen_extension():
    assert runner.expand_suites(["all"], True) == [
        "dev_field", "dev_legacy", "dev_width", "frozen_b", "frozen_r",
        "field_deployment", "field_validation", "frozen_a"]
    assert runner.expand_suites(["dev", "dev_field"]) == ["dev_field", "dev_legacy", "dev_width"]
    with pytest.raises(ValueError):
        runner.expand_suites(["typo"])


def test_suite_job_seed_conventions_and_width_obstacle_override(monkeypatch):
    def scene(name):
        return SimpleNamespace(case_id=name)
    b0, b1 = scene("B-00-000"), scene("B-00-001")
    monkeypatch.setitem(sys.modules, "suite", SimpleNamespace(
        build_tier_b=lambda: ([b0, b1], []), build_tier_a=lambda: [scene("A1")],
        tier_a=lambda: ["A1", "missing"], tier_a_shortfall=lambda _: ["missing"],
        robustness_variants=lambda _: [(scene("B-00-001-RE"), 1, "re")], test_id=lambda x: "T-" + x))
    monkeypatch.setitem(sys.modules, "field_training", SimpleNamespace(validation_set=lambda: [scene("FV-NT-01")]))
    monkeypatch.setitem(sys.modules, "paper2_set", SimpleNamespace(build=lambda env: ([
        {"built": scene("P2-scenario"), "episode_seed": 470031, "test_id": "P2-L1-HO-01"}], [])))
    monkeypatch.setitem(sys.modules, "common", SimpleNamespace(
        development_set=lambda n: [scene("legacy0"), scene("legacy1")],
        head_on_width_set=lambda: [scene("width0"), scene("width1")]))
    monkeypatch.setitem(sys.modules, "env", SimpleNamespace(ASVLidarEnv=lambda **kw: SimpleNamespace(close=lambda: None)))
    monkeypatch.setitem(sys.modules, "prediction_audit", SimpleNamespace(load_development_cases=lambda:
        [(0, scene("DV3-first")), (149, scene("DV3-last"))]))
    monkeypatch.setattr(runner, "scenario_cache_provenance", lambda: {})
    monkeypatch.setattr(runner, "cached_component", lambda name, provenance, builder: builder())
    jobs, details = runner.build_jobs(runner.expand_suites(["all"], True))
    by_key = {(j["suite"], j["case"]): j for j in jobs}
    assert by_key["dev_field", "DV3-first"]["seed"] == 900120
    assert by_key["dev_field", "DV3-last"]["seed"] == 900269
    assert by_key["dev_legacy", "DEV-001"]["seed"] == 900001
    assert by_key["dev_width", "HW-001"]["seed"] == 700001
    assert by_key["dev_width", "HW-001"]["obstacles"] == 0
    assert by_key["frozen_b", b1.case_id]["seed"] == by_key["frozen_r", "B-00-001-RE"]["seed"] == 400001
    assert by_key["frozen_a", "A1"]["seed"] == 300000
    assert by_key["field_deployment", "P2-L1-HO-01"]["seed"] == 470031
    assert by_key["field_validation", "FV-NT-01"]["seed"] == 950000
    assert all(j["obstacles"] is None for j in jobs if j["suite"] != "dev_width")
    assert details["frozen_a_shortfall"] == ["missing"]


def test_manifest_resume_requires_identical_settings_and_scenarios(tmp_path):
    manifest = {"settings": {"model": "hash1"}, "cases": [{"case": "A", "seed": 7}]}
    runner.create_or_check_manifest(tmp_path, manifest, False)
    runner.create_or_check_manifest(tmp_path, manifest, True)
    with pytest.raises(FileExistsError):
        runner.create_or_check_manifest(tmp_path, manifest, False)
    with pytest.raises(ValueError, match="differs"):
        runner.create_or_check_manifest(tmp_path, dict(manifest, settings={"model": "hash2"}), True)


def test_suite_counts_cannot_silently_shrink_but_named_shortfalls_are_explicit():
    runner.check_suite_coverage({"counts": {"dev_field": 150, "frozen_b": 800, "frozen_a": 35},
                                "frozen_a_shortfall": ["A", "B", "C"], "frozen_a_defined": 38})
    with pytest.raises(ValueError, match="Incomplete suite"):
        runner.check_suite_coverage({"counts": {"dev_field": 149}})
    with pytest.raises(ValueError, match="shortfalls"):
        runner.check_suite_coverage({"counts": {"frozen_a": 35},
                                    "frozen_a_shortfall": [], "frozen_a_defined": 38})


def test_historical_manifest_checks_exact_case_digests(tmp_path):
    path = tmp_path / "historical.json"
    path.write_text(json.dumps({"manifest_digest": "old", "case_digests": {"B-1": "abc", "B-2": "def"}}))
    cases = [{"suite": "frozen_b", "case": "B-1", "scenario_sha256": "abc"}]
    assert runner.check_reference_cases(cases, {"frozen": path})["frozen"]["matched_cases"] == 1
    cases[0]["scenario_sha256"] = "changed"
    with pytest.raises(ValueError, match="digest mismatch"):
        runner.check_reference_cases(cases, {"frozen": path})


def _record():
    return {"mode": "v5", "suite": "dev_field", "case": "DV3-one", "seed": 900120,
            "scenario_sha256": "scene-hash", "outcome": "goal"}


def test_resume_retains_completed_records_and_archives_partial_tail(tmp_path):
    row = _record()
    expected = {("v5", "dev_field", "DV3-one"): row}
    path = tmp_path / "episodes.jsonl"
    valid = (json.dumps(row) + "\n").encode()
    path.write_bytes(valid + b'{"mode": "off"')
    assert runner.load_completed(path, expected) == [row]
    assert path.read_bytes() == valid
    archives = list(tmp_path.glob("incomplete_tail_*.bin"))
    assert len(archives) == 1 and archives[0].read_bytes() == b'{"mode": "off"'


@pytest.mark.parametrize("failure", ["duplicate", "seed", "outcome"])
def test_resume_rejects_duplicate_or_mismatched_completed_rows(tmp_path, failure):
    original = _record()
    expected = {("v5", "dev_field", "DV3-one"): original}
    row = dict(original)
    if failure == "seed":
        row["seed"] += 1
    elif failure == "outcome":
        row["outcome"] = "diagnostic_limit"
    text = json.dumps(row) + "\n"
    if failure == "duplicate":
        text *= 2
    path = tmp_path / "episodes.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        runner.load_completed(path, expected)


def test_resume_lock_never_replaces_live_run(tmp_path, monkeypatch):
    path = tmp_path / "running.lock"
    path.write_text("1234")
    monkeypatch.setattr(runner, "pid_is_alive", lambda pid: True)
    with pytest.raises(RuntimeError, match="still active"):
        runner.acquire_lock(path, True)
    assert path.read_text() == "1234"
    monkeypatch.setattr(runner, "pid_is_alive", lambda pid: False)
    runner.acquire_lock(path, True)
    assert path.read_text() == str(runner.os.getpid())
