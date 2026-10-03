"""Synthetic filesystem/command guards; never execute compact or simulator."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/compact_safety_artifacts.py"
spec = importlib.util.spec_from_file_location("safety_compact_test", SCRIPT)
compact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compact
spec.loader.exec_module(compact)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(scope):
    run = scope / "run"
    identity = dict(attempt=1, case="TS3:A", mode="v16", seed=10)
    write(run / "manifest.json", dict(planned_runs=1, cases=[dict(case="TS3:A", seed=10)], modes=["v16"]))
    write(run / "completion.json", dict(completed_runs=1, source_drift=[], selection_unchanged=True, checkpoint_unchanged=True))
    write(run / "attempts/001.json", identity)
    write(run / "attempts/001_result.json", dict(identity, outcome="goal"))
    trace = run / "traces/001_v16.jsonl"
    write(trace, dict(step=1))
    return run, trace


def test_preflight_only_selects_complete_resident_traces(tmp_path, monkeypatch):
    run, trace = fixture(tmp_path)
    write(tmp_path / "loose.jsonl", dict(unowned=True))
    write(tmp_path / "incomplete/traces/001_v16.jsonl", dict(step=1))
    monkeypatch.setattr(compact, "allocated_bytes", lambda p: p.stat().st_size)
    result = compact.preflight(tmp_path)
    assert [row["path"] for row in result["candidates"]] == ["run/traces/001_v16.jsonl"]
    assert result["logical_bytes"] == trace.stat().st_size
    assert result["completed_runs"]["run"]["completed_runs"] == 1


@pytest.mark.parametrize("field,value", [("completed_runs", 0), ("source_drift", ["src/env.py"]),
                                         ("checkpoint_unchanged", False), ("selection_unchanged", False)])
def test_incomplete_or_drifted_runs_rejected(tmp_path, field, value):
    run, _ = fixture(tmp_path)
    path = run / "completion.json"
    data = json.loads(path.read_text())
    data[field] = value
    write(path, data)
    with pytest.raises(ValueError, match="Incomplete"):
        compact.eligible_run(run, tmp_path)


@pytest.mark.parametrize("change", ["missing_token", "extra_trace", "wrong_seed"])
def test_completion_is_not_enough_without_exact_journals(tmp_path, change):
    run, _ = fixture(tmp_path)
    if change == "missing_token":
        (run / "attempts/001.json").unlink()
    elif change == "extra_trace":
        write(run / "traces/002_v16.jsonl", dict(step=1))
    else:
        data = json.loads((run / "attempts/001.json").read_text())
        write(run / "attempts/001.json", dict(data, seed=99))
    with pytest.raises((ValueError, FileNotFoundError)):
        compact.eligible_run(run, tmp_path)


def test_scope_escape_rejected_before_read(tmp_path):
    with pytest.raises(ValueError, match="Outside"):
        compact.resident_regular(tmp_path.parent / "outside.jsonl", tmp_path)


@pytest.mark.parametrize("attributes", [0x400, 0x1000, 0x40000, 0x400000])
def test_reparse_offline_recall_rejected(tmp_path, monkeypatch, attributes):
    _, trace = fixture(tmp_path)
    original = Path.lstat
    def fake(path):
        result = original(path)
        if path == trace:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=attributes,
                                   st_nlink=1)
        return result
    monkeypatch.setattr(Path, "lstat", fake)
    with pytest.raises(ValueError, match="Nonresident"):
        compact.resident_regular(trace, tmp_path)


@pytest.mark.parametrize("changed", [False, True])
def test_compression_hash_guard_and_single_native_command(tmp_path, monkeypatch, changed):
    _, trace = fixture(tmp_path)
    info = trace.stat()
    row = dict(path=trace.relative_to(tmp_path).as_posix(), logical_bytes=info.st_size,
               allocated_before=info.st_size, mtime_ns=info.st_mtime_ns)
    calls, recorded = [], []
    monkeypatch.setattr(compact, "resident_regular", lambda p, scope: SimpleNamespace(
        st_size=info.st_size, st_mtime_ns=info.st_mtime_ns, st_file_attributes=0x800 if calls else 0))
    monkeypatch.setattr(compact, "allocated_bytes", lambda p: 1)
    def command(argv, **kwargs):
        assert recorded and recorded[0]["stage"] == "before"
        calls.append(argv)
        if changed:
            trace.write_bytes(b"x" * info.st_size)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(compact.subprocess, "run", command)
    if changed:
        with pytest.raises(ValueError, match="Content changed"):
            compact.compress_file(row, tmp_path, Path("compact.exe"), recorded.append)
    else:
        result = compact.compress_file(row, tmp_path, Path("compact.exe"), recorded.append)
        assert result["sha256_before"] == result["sha256_after"] == compact.digest(trace)
        assert result["reclaimed_bytes"] == info.st_size - 1
    assert calls == [["compact.exe", "/C", "/Q", str(trace)]]
