"""Lossless NTFS compression of resident traces in completed safety runs.

No traces, results, source archives or cache files are deleted or renamed.
Default is a read-only preflight. --apply creates an audit then invokes compact.exe
on one verified file at a time, checking identical SHA-256 after each operation.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
SCOPE = ROOT / "results/safety_dev"
UNSAFE_ATTRIBUTES = 0x400 | 0x1000 | 0x40000 | 0x400000  # reparse/offline/recall
COMPRESSED = 0x800


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resident_regular(path, scope=SCOPE):
    path, scope = Path(path).absolute(), Path(scope).absolute()
    require(path.is_relative_to(scope) and path != scope, f"Outside safety scope: {path}")
    for parent in [scope, *reversed(path.relative_to(scope).parents[:-1])]:
        # Relative ancestors are explicitly rooted, never interpreted at cwd.
        parent = parent if parent.is_absolute() else scope / parent
        info = parent.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & UNSAFE_ATTRIBUTES,
                f"Unsafe ancestor: {parent}")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"Not a regular file: {path}")
    require(not getattr(info, "st_file_attributes", 0) & UNSAFE_ATTRIBUTES, f"Nonresident or reparse file: {path}")
    require(info.st_nlink == 1, f"Multiple hard links: {path}")
    require(path.resolve().is_relative_to(scope.resolve()), f"Resolved path outside scope: {path}")
    return info


def allocated_bytes(path):
    require(os.name == "nt", "NTFS compression requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetCompressedFileSizeW
    query.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.DWORD
    high = wintypes.DWORD()
    ctypes.set_last_error(0)
    low = query(str(path), ctypes.byref(high))
    error = ctypes.get_last_error()
    if low == 0xFFFFFFFF and error:
        raise ctypes.WinError(error)
    return (high.value << 32) | low


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path, scope):
    resident_regular(path, scope)
    return json.loads(path.read_bytes())


def eligible_run(directory, scope):
    """Recognize only the exact completed pilot/iteration journal schema."""
    completion_path, manifest_path = directory / "completion.json", directory / "manifest.json"
    completion, manifest = read_json(completion_path, scope), read_json(manifest_path, scope)
    expected = manifest["planned_runs"]
    require(isinstance(expected, int) and expected > 0, "Invalid planned count")
    require(completion.get("completed_runs") == expected and completion.get("source_drift") == []
            and completion.get("selection_unchanged") is True
            and completion.get("checkpoint_unchanged") is True, f"Incomplete or drifted run: {directory}")
    attempts = directory / "attempts"
    results = sorted(attempts.glob("*_result.json"))
    tokens = {p.name for p in attempts.glob("*.json") if re.fullmatch(r"\d+\.json", p.name)}
    traces = {p.name for p in (directory / "traces").glob("*.jsonl")}
    require(len(results) == expected, f"Result count mismatch: {directory}")
    expected_tokens, expected_traces, keys = set(), set(), set()
    cases = {row["case"]: row for row in manifest["cases"]}
    require(len(cases) == len(manifest["cases"]), "Duplicate manifest identity")
    for path in results:
        result = read_json(path, scope)
        number, mode, case = result["attempt"], result["mode"], result["case"]
        require(case in cases and mode in manifest["modes"] and result["seed"] == cases[case]["seed"], "Result identity mismatch")
        key = (case, mode)
        require(key not in keys, "Duplicate result identity")
        keys.add(key)
        expected_tokens.add(f"{number:03d}.json")
        expected_traces.add(f"{number:03d}_{mode}.jsonl")
        require(path.name == f"{number:03d}_result.json", "Result attempt filename mismatch")
        token = read_json(attempts / f"{number:03d}.json", scope)
        require(all(token[k] == result[k] for k in ("attempt", "case", "mode", "seed")), "Token identity mismatch")
    require(keys == {(case, mode) for case in cases for mode in manifest["modes"]}, "Incomplete case/mode product")
    require(tokens == expected_tokens and traces == expected_traces, f"Trace/token set mismatch: {directory}")
    return {"manifest_sha256": digest(manifest_path), "completion_sha256": digest(completion_path),
            "completed_runs": expected}


def preflight(scope=SCOPE):
    scope = Path(scope).absolute()
    candidates, skipped, runs = [], [], {}
    for folder, directories, files in os.walk(scope, followlinks=False):
        safe_directories = []
        for name in directories:
            path = Path(folder) / name
            attrs = getattr(path.lstat(), "st_file_attributes", 0)
            if path.is_symlink() or attrs & UNSAFE_ATTRIBUTES:
                skipped.append({"path": str(path.relative_to(scope)), "reason": "unsafe directory"})
            else:
                safe_directories.append(name)
        directories[:] = safe_directories
        if Path(folder).name != "traces" or "completion.json" not in {p.name for p in Path(folder).parent.iterdir()}:
            continue
        run = Path(folder).parent
        try:
            proof = eligible_run(run, scope)
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append({"path": str(run.relative_to(scope)), "reason": str(error)})
            continue
        runs[str(run.relative_to(scope))] = proof
        for name in sorted(files):
            if not name.endswith(".jsonl"):
                continue
            path = Path(folder) / name
            try:
                info = resident_regular(path, scope)
                if getattr(info, "st_file_attributes", 0) & COMPRESSED:
                    skipped.append({"path": str(path.relative_to(scope)), "reason": "already compressed"})
                    continue
                candidates.append({"path": path.relative_to(scope).as_posix(), "logical_bytes": info.st_size,
                                   "allocated_before": allocated_bytes(path), "mtime_ns": info.st_mtime_ns})
            except (OSError, ValueError) as error:
                skipped.append({"path": str(path.relative_to(scope)), "reason": str(error)})
    return {"scope": str(scope), "candidates": candidates, "completed_runs": runs, "skipped": skipped,
            "logical_bytes": sum(row["logical_bytes"] for row in candidates),
            "allocated_before": sum(row["allocated_before"] for row in candidates)}


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def compress_file(row, scope, compact, before_apply=None):
    path = scope / row["path"]
    info = resident_regular(path, scope)
    require(info.st_size == row["logical_bytes"] and info.st_mtime_ns == row["mtime_ns"], f"File changed after preflight: {path}")
    before_hash = digest(path)
    if before_apply is not None:
        before_apply(dict(row, stage="before", sha256_before=before_hash))
    completed = subprocess.run([str(compact), "/C", "/Q", str(path)], capture_output=True, text=True,
                               errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    after = resident_regular(path, scope)
    after_hash = digest(path)
    require(after_hash == before_hash and after.st_size == info.st_size, f"Content changed during compression: {path}")
    require(completed.returncode == 0, f"compact failed for {path}: {completed.stdout} {completed.stderr}")
    require(bool(getattr(after, "st_file_attributes", 0) & COMPRESSED), f"File was not marked compressed: {path}")
    size = allocated_bytes(path)
    return dict(row, sha256_before=before_hash, sha256_after=after_hash, allocated_after=size,
                reclaimed_bytes=row["allocated_before"] - size, compact_returncode=completed.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Compress after validated preflight; never delete anything")
    args = parser.parse_args()
    plan = preflight()
    print(json.dumps({"preflight": True, "files": len(plan["candidates"]), "completed_runs": len(plan["completed_runs"]),
                      "logical_bytes": plan["logical_bytes"], "skipped": plan["skipped"]}), flush=True)
    if not args.apply:
        return
    compact = Path(os.environ["SystemRoot"]) / "System32/compact.exe"
    require(compact.is_file(), "Native compact.exe missing")
    directory = SCOPE / "maintenance" / ("ntfs_compact_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    directory.mkdir(parents=True, exist_ok=False)
    plan.update(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                tool_sha256=digest(Path(__file__)), compact_path=str(compact),
                free_bytes_before=shutil.disk_usage(SCOPE).free)
    write_json(directory / "inventory.json", plan)
    started, reclaimed, completed_count = time.perf_counter(), 0, 0
    try:
        with (directory / "progress.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            def journal(record):
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            for row in plan["candidates"]:
                result = compress_file(row, SCOPE, compact, journal)
                journal(dict(result, stage="verified"))
                completed_count += 1
                reclaimed += result["reclaimed_bytes"]
                print(json.dumps({"completed": completed_count, "total": len(plan["candidates"]),
                                  "reclaimed_bytes": reclaimed, "path": row["path"]}), flush=True)
    except Exception as error:
        write_json(directory / "error.json", {"completed": completed_count, "reclaimed_bytes": reclaimed,
                   "error": repr(error), "no_files_deleted": True})
        raise
    result = {"completed_files": completed_count, "reclaimed_bytes": reclaimed,
              "logical_bytes_unchanged": plan["logical_bytes"], "all_content_hashes_equal": True,
              "no_files_deleted": True, "elapsed_s": time.perf_counter() - started,
              "free_bytes_before": plan["free_bytes_before"], "free_bytes_after": shutil.disk_usage(SCOPE).free,
              "inventory_sha256": digest(directory / "inventory.json"), "progress_sha256": digest(directory / "progress.jsonl")}
    write_json(directory / "summary.json", result)
    print(json.dumps({"audit_directory": str(directory), **result}), flush=True)


if __name__ == "__main__":
    main()
