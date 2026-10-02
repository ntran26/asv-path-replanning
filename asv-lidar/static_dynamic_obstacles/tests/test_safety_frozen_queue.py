"""Retry and queue behavior with mocked subprocesses; no evaluations launched."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/run_frozen_queue.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("frozen_queue_under_test", _PATH)
queue = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(queue)
DENIED = "PermissionError: [Errno 13] Permission denied: 'results/safety_dev/suites/run/progress.json'\n"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "results/safety_dev").mkdir(parents=True)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(queue, "run_status", lambda directory: {"complete": True, "completed": 4, "expected": 4})
    return tmp_path


def mock_processes(monkeypatch, responses, on_start=None):
    commands, events, state = [], [], {"active": False}
    pending = iter(responses)

    def start(command, **kwargs):
        assert not state["active"], "queue attempted overlapping evaluators"
        state["active"] = True
        commands.append((command, kwargs))
        events.append("start")
        if on_start:
            on_start(command)
        code, output = next(pending)

        class FakeProcess:
            stdout = io.StringIO(output)

            def wait(self):
                state["active"] = False
                events.append("wait")
                return code

        return FakeProcess()

    monkeypatch.setattr(queue.subprocess, "Popen", start)
    return commands, events


def test_resume_argv_and_metadata_survive_specific_progress_error(sandbox, monkeypatch):
    def create_manifest(command):
        path = sandbox / "results/safety_dev/suites/run"
        path.mkdir(exist_ok=True, parents=True)
        (path / "manifest.json").write_text("{}")

    commands, events = mock_processes(monkeypatch, [(1, DENIED), (0, "finished\n")], create_manifest)
    refs = ["--reference-frozen", "reference with spaces/manifest.json"]
    queue.run_component("frozen_r", "run", refs)
    assert len(commands) == 2 and events == ["start", "wait", "start", "wait"]
    first, options = commands[0]
    second, _ = commands[1]
    assert first[:3] == [sys.executable, "-B", "tools/diagnostics/safety/suite_eval.py"]
    assert first[-2:] == refs and "--resume" not in first
    assert second == first + ["--resume"]
    assert options["cwd"] == sandbox
    assert options["env"]["OMP_NUM_THREADS"] == options["env"]["MKL_NUM_THREADS"] == "1"
    logs = list((sandbox / "results/safety_dev").glob("queue_run_*.log"))
    assert len(logs) == 2
    metadata = [json.loads(path.read_text().splitlines()[0]) for path in logs]
    assert {row["attempt"] for row in metadata} == {1, 2}
    assert all(len(row["wrapper_sha256"]) == 64 for row in metadata)
    assert refs == ["--reference-frozen", "reference with spaces/manifest.json"]


@pytest.mark.parametrize("fatal", ["ValueError: invalid manifest\n", "RuntimeError: live lock\n",
                                    "Resume rejected: changed settings\n", "KeyError: 'outcome'\n",
                                    "TypeError: malformed journal\n", "KeyboardInterrupt\n"])
def test_fatal_error_masked_by_final_progress_denial_is_not_retried(sandbox, monkeypatch, fatal):
    commands, _ = mock_processes(monkeypatch, [(1, fatal + DENIED), (0, "must not reach\n")])
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        queue.run_component("frozen_r", "run", [])
    assert len(commands) == 1


@pytest.mark.parametrize("returncode", [0, 1])
def test_episode_error_journal_blocks_progress_retry(sandbox, monkeypatch, returncode):
    directory = sandbox / "results/safety_dev/suites/run"
    directory.mkdir(parents=True)
    (directory / "errors.jsonl").write_text('{"error":"policy failed"}\n')
    commands, _ = mock_processes(monkeypatch, [(returncode, DENIED), (0, "must not reach\n")])
    with pytest.raises(RuntimeError, match="Episode error journal"):
        queue.run_component("frozen_r", "run", [])
    assert len(commands) == 1


def test_non_progress_permission_failure_is_fatal(sandbox, monkeypatch):
    commands, _ = mock_processes(monkeypatch, [(1, DENIED.replace("progress.json", "episodes.jsonl"))])
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        queue.run_component("frozen_r", "run", [])
    assert len(commands) == 1


def test_retry_budget_is_bounded(sandbox, monkeypatch):
    commands, _ = mock_processes(monkeypatch, [(1, DENIED), (1, DENIED)])
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        queue.run_component("frozen_r", "run", [], max_attempts=2)
    assert len(commands) == 2


def test_lane_advances_only_after_serial_success(sandbox, monkeypatch):
    commands, events = mock_processes(monkeypatch, [(0, "first done\n"), (0, "second done\n")])
    monkeypatch.setattr(sys, "argv", ["run_frozen_queue.py", "--lane", "A"])
    queue.main()
    assert events == ["start", "wait", "start", "wait"]
    assert [command[command.index("--tag") + 1] for command, _ in commands] == ["v5_frozen_r", "v5_dev_fv_a"]


def test_fatal_exception_outside_retained_tail_still_blocks_retry(sandbox, monkeypatch):
    output = "TypeError: invalid durable data\n" + "ordinary progress\n" * 250 + DENIED
    commands, _ = mock_processes(monkeypatch, [(1, output)])
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        queue.run_component("frozen_r", "run", [])
    assert len(commands) == 1


def test_successful_child_with_missing_interior_case_is_resumed(sandbox, monkeypatch):
    from suite_status import run_status
    monkeypatch.setattr(queue, "run_status", run_status)
    directory = sandbox / "results/safety_dev/suites/run"
    cases = [{"suite": "frozen_b", "case": name, "seed": seed, "scenario_sha256": str(seed)}
             for seed, name in enumerate(("B-06-039", "B-06-040", "B-06-041"))]
    rows = [dict(case, mode="off", outcome="goal") for case in cases]

    def simulate_child(command):
        directory.mkdir(parents=True, exist_ok=True)
        if "--resume" not in command:
            (directory / "manifest.json").write_text(json.dumps({"settings": {"modes": ["off"]},
                "cases": cases, "expected_episodes": 3}))
            (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (rows[0], rows[2])))
        else:
            with (directory / "episodes.jsonl").open("a") as handle:
                handle.write(json.dumps(rows[1]) + "\n")

    commands, events = mock_processes(monkeypatch, [(0, "3/3 in memory\n"), (0, "filled missing case\n")], simulate_child)
    queue.run_component("frozen_b", "run", [])
    assert len(commands) == 2 and "--resume" in commands[-1][0]
    assert events == ["start", "wait", "start", "wait"]
    assert run_status(directory)["completed"] == 3


def test_invalid_journal_on_success_is_fatal(sandbox, monkeypatch):
    def reject(directory):
        raise ValueError("Duplicate completed identity")
    monkeypatch.setattr(queue, "run_status", reject)
    commands, _ = mock_processes(monkeypatch, [(0, "child success\n")])
    with pytest.raises(RuntimeError, match="Journal validation failed"):
        queue.run_component("frozen_r", "run", [])
    assert len(commands) == 1


def test_incomplete_success_reconciliation_obeys_attempt_budget(sandbox, monkeypatch):
    monkeypatch.setattr(queue, "run_status", lambda directory: {"complete": False, "completed": 2, "expected": 3})
    commands, _ = mock_processes(monkeypatch, [(0, "child success\n"), (0, "child success\n")])
    with pytest.raises(RuntimeError, match="Journal still incomplete"):
        queue.run_component("frozen_r", "run", [], max_attempts=2)
    assert len(commands) == 2
