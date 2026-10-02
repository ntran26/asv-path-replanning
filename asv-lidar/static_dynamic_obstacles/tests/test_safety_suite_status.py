"""Journal-based monitoring remains read-only and rejects misleading progress."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/suite_status.py"
_SPEC = importlib.util.spec_from_file_location("suite_status_under_test", _PATH)
status = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(status)


def _case(component="frozen_b"):
    return {"suite": component, "case": "repeated-id", "seed": 8, "scenario_sha256": "scene"}


def _run(path, rows, tail=b""):
    cases = [_case(), _case("frozen_r")]
    manifest = {"settings": {"modes": ["off", "v4"]}, "cases": cases, "expected_episodes": 4}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    journal = ("".join(json.dumps(row) + "\r\n" for row in rows)).encode() + tail
    (path / "episodes.jsonl").write_bytes(journal)
    # This stale projection must never govern the status result.
    (path / "progress.json").write_text('{"completed":99999,"complete":true}', encoding="utf-8")
    return journal


def test_committed_journal_lines_and_component_keys_are_authoritative(tmp_path):
    rows = [dict(_case(), mode="off", outcome="goal"),
            dict(_case("frozen_r"), mode="off", outcome="collision:boundary"),
            dict(_case(), mode="v4", outcome="goal")]
    _run(tmp_path, rows)
    result = status.run_status(tmp_path)
    assert (result["completed"], result["expected"], result["complete"]) == (3, 4, False)
    assert result["modes"] == {"off": {"completed": 2, "expected": 2}, "v4": {"completed": 1, "expected": 2}}
    assert result["outcomes"] == {"goal": 2, "collision:boundary": 1}


def test_partial_utf8_final_tail_is_ignored_and_no_files_change(tmp_path):
    row = dict(_case(), mode="off", outcome="goal")
    tail = b'{"unfinished": "\xe2\x82'
    original = _run(tmp_path, [row], tail)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = status.run_status(tmp_path)
    assert result["completed"] == 1 and result["ignored_tail_bytes"] == len(tail)
    assert status.read_shared_bytes(tmp_path / "episodes.jsonl") == original
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("failure", ["duplicate", "invalid_outcome", "unknown_identity", "seed_mismatch"])
def test_invalid_records_do_not_inflate_progress(tmp_path, failure):
    row = dict(_case(), mode="off", outcome="goal")
    rows = [row]
    if failure == "duplicate":
        rows.append(dict(row))
    elif failure == "invalid_outcome":
        row["outcome"] = "diagnostic_limit"
    elif failure == "unknown_identity":
        row["case"] = "absent"
    else:
        row["seed"] += 1
    _run(tmp_path, rows)
    with pytest.raises(ValueError):
        status.run_status(tmp_path)


def test_malformed_committed_json_is_not_treated_as_an_incomplete_tail():
    with pytest.raises(ValueError, match="Invalid committed JSON"):
        status.committed_records(b'{"bad"\n')
    assert status.committed_records(b'{"uncommitted"') == ([], len(b'{"uncommitted"'))
    assert status.committed_records(b"") == ([], 0)


def test_manifest_without_journal_has_zero_committed_episodes(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"settings": {"modes": ["v4"]},
        "cases": [_case()], "expected_episodes": 1}), encoding="utf-8")
    result = status.run_status(tmp_path)
    assert result["completed"] == 0 and result["expected"] == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_windows_reader_permits_simultaneous_write_and_delete_access(tmp_path, monkeypatch):
    import ctypes

    path = tmp_path / "shared.txt"
    path.write_bytes(b"committed content\n")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    original_read = kernel.ReadFile
    checked = []

    def read_hook(handle, buffer, length, received, overlapped):
        # Opening these handles performs no writes or deletions. It proves the
        # status reader does not deny either access while its own handle is live.
        for access in (0x40000000, 0x00010000):  # GENERIC_WRITE, DELETE
            concurrent = kernel.CreateFileW(str(path), access, 0x7, None, 3, 0x80, None)
            assert concurrent != ctypes.c_void_p(-1).value
            kernel.CloseHandle(concurrent)
            checked.append(access)
        original_read.argtypes = read_hook.argtypes
        original_read.restype = read_hook.restype
        return original_read(handle, buffer, length, received, overlapped)

    proxy = SimpleNamespace(CreateFileW=kernel.CreateFileW, GetFileSizeEx=kernel.GetFileSizeEx,
                            ReadFile=read_hook, CloseHandle=kernel.CloseHandle)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: proxy)
    assert status.read_shared_bytes(path) == b"committed content\n"
    assert checked == [0x40000000, 0x00010000]
    assert path.read_bytes() == b"committed content\n"
