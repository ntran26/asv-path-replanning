"""Read committed suite progress without locking writers or changing files.

python -B tools/diagnostics/safety/suite_status.py
python -B tools/diagnostics/safety/suite_status.py results/safety_dev/suites/R_final --json

With no paths, inspect all manifest-bearing directories under safety_dev/suites.
Windows handles explicitly share reads, writes, and deletion. Completed JSONL
records are authoritative; progress.json and process state are never consulted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import errno
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VALID_OUTCOMES = {"goal", "timeout", "collision:boundary", "collision:obstacle", "collision:target"}


def read_shared_bytes(path):
    """Read the file's initial extent, with FILE_SHARE_READ|WRITE|DELETE on Windows."""
    path = Path(path)
    if os.name != "nt":
        with path.open("rb") as handle:
            return handle.read(os.fstat(handle.fileno()).st_size)
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    kernel.GetFileSizeEx.restype = wintypes.BOOL
    kernel.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    def fail():
        code = ctypes.get_last_error()
        if code in (2, 3):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        error = ctypes.WinError(code)
        error.filename = str(path)
        raise error

    # GENERIC_READ, share READ|WRITE|DELETE, OPEN_EXISTING, normal attributes.
    handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 0x1 | 0x2 | 0x4,
                                None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        fail()
    try:
        length = ctypes.c_longlong()
        if not kernel.GetFileSizeEx(handle, ctypes.byref(length)):
            fail()
        remaining, chunks = length.value, []
        buffer = ctypes.create_string_buffer(65536)
        while remaining:
            received = wintypes.DWORD()
            if not kernel.ReadFile(handle, buffer, min(remaining, len(buffer)), ctypes.byref(received), None):
                fail()
            if received.value == 0:
                break  # The journal may have been truncated to its last committed line on resume.
            chunks.append(buffer.raw[:received.value])
            remaining -= received.value
        return b"".join(chunks)
    finally:
        kernel.CloseHandle(handle)


def committed_records(data):
    """Ignore only an unterminated final record, without decoding or repairing it."""
    lines = data.split(b"\n")
    tail_bytes = len(lines.pop())
    rows = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid committed JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Committed record at line {line_number} is not an object")
        rows.append(row)
    return rows, tail_bytes


def run_status(directory):
    directory = Path(directory)
    manifest = json.loads(read_shared_bytes(directory / "manifest.json"))
    modes = manifest["settings"]["modes"]
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Manifest has missing or duplicate modes")
    expected = {}
    for case in manifest["cases"]:
        for mode in modes:
            key = (mode, case["suite"], case["case"])
            if key in expected:
                raise ValueError(f"Duplicate manifest identity: {key}")
            expected[key] = case
    if len(expected) != manifest["expected_episodes"]:
        raise ValueError("Manifest episode count does not match its mode/case identities")
    try:
        data = read_shared_bytes(directory / "episodes.jsonl")
    except FileNotFoundError:
        data = b""
    records, tail_bytes = committed_records(data)
    done, mode_counts, outcomes = set(), Counter(), Counter()
    for row in records:
        key = (row["mode"], row["suite"], row["case"])
        if key in done:
            raise ValueError(f"Duplicate completed identity: {key}")
        if key not in expected:
            raise ValueError(f"Unknown completed identity: {key}")
        if row.get("outcome") not in VALID_OUTCOMES:
            raise ValueError(f"Invalid completed outcome: {key}")
        if any(row.get(name) != expected[key][name] for name in ("seed", "scenario_sha256")):
            raise ValueError(f"Completed seed/digest mismatch: {key}")
        done.add(key)
        mode_counts[row["mode"]] += 1
        outcomes[row["outcome"]] += 1
    declared = Counter(key[0] for key in expected)
    return {"run": directory.name, "directory": str(directory.resolve()), "status": "complete" if len(done) == len(expected) else "incomplete",
            "completed": len(done), "expected": len(expected), "complete": len(done) == len(expected),
            "modes": {mode: {"completed": mode_counts[mode], "expected": declared[mode]} for mode in modes},
            "outcomes": dict(outcomes), "journal_snapshot_bytes": len(data), "ignored_tail_bytes": tail_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="*")
    parser.add_argument("--json", action="store_true", help="emit only a JSON object to stdout")
    args = parser.parse_args()
    directories = args.runs
    if not directories:
        parent = ROOT / "results/safety_dev/suites"
        directories = sorted(path for path in parent.iterdir() if path.is_dir() and (path / "manifest.json").exists()) if parent.exists() else []
    results = []
    for directory in directories:
        try:
            results.append(run_status(directory))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append({"run": directory.name, "directory": str(directory.resolve()),
                            "status": "error", "error": str(exc)})
    if args.json:
        print(json.dumps({"runs": results}, indent=2))
    elif not results:
        print("No suite manifests found.")
    else:
        for result in results:
            if result["status"] == "error":
                print(f"{result['run']}: ERROR {result['error']}")
                continue
            modes = ", ".join(f"{mode} {counts['completed']}/{counts['expected']}" for mode, counts in result["modes"].items())
            tail = f"; ignored uncommitted tail {result['ignored_tail_bytes']} bytes" if result["ignored_tail_bytes"] else ""
            print(f"{result['run']}: {result['completed']}/{result['expected']} [{result['status']}] | {modes}{tail}")
    return int(any(result["status"] == "error" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
