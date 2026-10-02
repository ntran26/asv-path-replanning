"""Launch unchanged frozen evaluators; verify journals before queue advancement.

Run at most one instance per lane, and TWO evaluation workers in total:
    python -B tools/diagnostics/safety/run_frozen_queue.py --lane A
    python -B tools/diagnostics/safety/run_frozen_queue.py --lane B

This wrapper changes no controller/evaluator code, settings or manifests. The
original evaluator validates --resume and recovers its durable journal. Each
attempt gets a unique log containing this wrapper's hash and exact argv.
Only progress-file access failures and valid but incomplete journals are
retried. Invalid journals, other exceptions, and episode errors stop the queue.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from suite_status import run_status

ROOT = Path(__file__).resolve().parents[3]
WRAPPER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
STATUS_SHA256 = hashlib.sha256(Path(__file__).with_name("suite_status.py").read_bytes()).hexdigest()
EXCEPTION_LINE = re.compile(r"(?:[\w.]*\.)?(?:\w*(?:Error|Exception)|KeyboardInterrupt|SystemExit|StopIteration)(?::.*)?")
LANES = {
    "A": [
        ("frozen_r", "v5_frozen_r", ["--reference-frozen", "results/frozen_suite/sacs0_bl3/manifest.json"]),
        ("dev,field_validation,frozen_a", "v5_dev_fv_a", []),
    ],
    "B": [
        ("frozen_b", "v5_frozen_b", ["--reference-frozen", "results/frozen_suite/sacs0_bl3/manifest.json"]),
        ("field_deployment", "v5_field_deployment", ["--reference-field", "results/paper2_set/sacs0_bl3/manifest.json"]),
    ],
}


def progress_permission_line(line):
    return line.startswith("PermissionError:") and re.search(r"\bprogress\.json['\"]?\s*$", line) is not None


def fatal_exception_line(line):
    line = line.strip()
    return "Resume rejected:" in line or bool(EXCEPTION_LINE.fullmatch(line) and not progress_permission_line(line))


def retryable_progress_error(output, episode_error_exists):
    """Never retry an episode failure or a rejected manifest/journal."""
    lines = output.splitlines()
    return (not episode_error_exists and any(progress_permission_line(line) for line in lines)
            and not any(fatal_exception_line(line) for line in lines))


def run_component(suites, tag, references, *, max_attempts=5):
    directory = ROOT / "results" / "safety_dev" / "suites" / tag
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for attempt in range(1, max_attempts + 1):
        command = [sys.executable, "-B", "tools/diagnostics/safety/suite_eval.py",
                   "--suites", suites, "--modes", "off,v4,v5", "--tag", tag, *references]
        if (directory / "manifest.json").exists():
            command.append("--resume")
        log_path = ROOT / "results" / "safety_dev" / f"queue_{tag}_{attempt}_{uuid.uuid4().hex}.log"
        tail = deque(maxlen=200)
        fatal_seen = False
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps({"wrapper_sha256": WRAPPER_SHA256, "status_source_sha256": STATUS_SHA256, "argv": command,
                                  "attempt": attempt, "tag": tag}) + "\n")
            log.flush()
            print(f"{tag}: attempt {attempt}; log {log_path.name}", flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line)
                fatal_seen |= fatal_exception_line(line)
                print(line, end="", flush=True)
            returncode = process.wait()
        errors = directory / "errors.jsonl"
        episode_error = errors.exists() and errors.stat().st_size > 0
        if episode_error:
            raise RuntimeError(f"Episode error journal is nonempty in {tag}; see {errors} and {log_path}")
        if returncode == 0:
            # Child in-memory counters may differ from the persisted journal.
            # Verify complete identities using the independent shared reader.
            # Invalid records are fatal; never reconstruct them from the CSV.
            try:
                status = run_status(directory)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"Journal validation failed in {tag}; see {log_path}: {exc}") from exc
            if status["complete"]:
                print(f"{tag}: verified {status['completed']}/{status['expected']} committed journal records.", flush=True)
                return
            if attempt == max_attempts:
                raise RuntimeError(f"Journal still incomplete in {tag}: {status['completed']}/{status['expected']}; see {log_path}")
            print(f"{tag}: child exited successfully but journal has {status['completed']}/{status['expected']}; resuming missing cases.", flush=True)
        elif attempt == max_attempts or fatal_seen or not retryable_progress_error("".join(tail), episode_error):
            raise RuntimeError(f"Evaluation failed in {tag} (exit {returncode}); see {log_path}")
        else:
            print(f"{tag}: transient progress-file denial; resuming unchanged manifest/journal.", flush=True)
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANES), required=True)
    args = parser.parse_args()
    for suites, tag, references in LANES[args.lane]:
        run_component(suites, tag, references)


if __name__ == "__main__":
    main()
