"""Synthetic report fixtures only; no environment/model imports or episodes."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/v9_pilot_report.py"
SPEC = importlib.util.spec_from_file_location("v9_report_under_test", PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def core():
    cases = [{"case": "TS2:A", "seed": 7, "dataset": "ts2", "selection_stratum": "fixture"}]
    selection = {"cases": cases}
    manifest = {"cases": cases, "modes": list(report.MODES), "planned_runs": 3}
    completion = {"completed_runs": 3, "source_drift": [], "selection_unchanged": True, "checkpoint_unchanged": True}
    rows, tokens = [], []
    for i, mode in enumerate(report.MODES, 1):
        token = {"attempt": i, "case": "TS2:A", "mode": mode, "seed": 7}
        tokens.append(token)
        rows.append(dict(token, dataset="ts2", stratum="fixture", outcome="goal", collided=False,
                         collided_target=False, elapsed_s=1.0, steps=1))
    return selection, manifest, completion, rows, tokens


def test_complete_exact_pairs_validated():
    assert len(report.validate_core(*core())) == 3


@pytest.mark.parametrize("fault", ["missing_result", "duplicate_pair", "token_seed", "completion", "source_drift", "selection_seed"])
def test_incomplete_or_mismatched_campaign_rejected(fault):
    args = copy.deepcopy(core())
    if fault == "missing_result": args[3].pop()
    if fault == "duplicate_pair": args[3][1]["mode"] = "off"
    if fault == "token_seed": args[4][0]["seed"] = 8
    if fault == "completion": args[2]["completed_runs"] = 2
    if fault == "source_drift": args[2]["source_drift"] = ["src/env.py"]
    if fault == "selection_seed": args[0]["cases"][0]["seed"] = 8
    with pytest.raises(ValueError): report.validate_core(*args)


@pytest.mark.parametrize("gains,losses,expected", [(0, 0, 1.0), (5, 0, 0.0625), (0, 5, 0.0625),
                                                 (1, 4, 0.375), (3, 3, 1.0)])
def test_exact_paired_mcnemar(gains, losses, expected):
    assert report.exact_mcnemar(gains, losses) == expected


def step(number, rudder=.2, rpm=12., changed=False, **details):
    return {"step": number, "pre_state": [1., 2., 3., .4], "policy_action": [.2, .3],
            "rudder_command": rudder, "signed_rpm_command": rpm, "changed": changed, "filter": details}


def test_first_divergence_uses_executed_commands_not_flags_or_reason():
    left = [step(1, changed=True, why="turn"), step(2, rpm=-24., why="last certificate")]
    right = [step(1, why="nominal"), step(2, why="unchecked override suppressed", v9_parent_why="last certificate",
                                             v9_proposed_clearance=-.2, v9_policy_tail_clearance=-.1)]
    found = report.first_divergence(left, right)
    assert found["first_command_divergence_step"] == 2
    assert found["pre_state_equal"] and found["policy_actions_equal"]
    assert found["divergence_v9_v9_parent_why"] == "last certificate"
    assert found["divergence_v9_v9_policy_tail_clearance"] == -.1


def test_equal_prefix_does_not_hide_different_trace_lengths():
    found = report.first_divergence([step(1)], [step(1), step(2)])
    assert found["first_command_divergence_step"] == ""
    assert found["trace_lengths_equal"] is False


def test_trace_partial_tail_or_diagnostic_counter_mismatch_rejected():
    raw = json.dumps(step(1)).encode()
    result = dict(steps=1, checked_steps=0, suppressed_steps=0, safety_v2_steps=0, why_counts={"off": 1})
    with pytest.raises(ValueError, match="tail"): report.decode_trace(raw, result)
    assert len(report.decode_trace(raw + b"\n", result)) == 1
    result["suppressed_steps"] = 1
    with pytest.raises(ValueError, match="counters"): report.decode_trace(raw + b"\n", result)


def test_trace_reason_counts_are_verified():
    raw = (json.dumps(step(1, why="nominal")) + "\n").encode()
    result = dict(steps=1, checked_steps=0, suppressed_steps=0, safety_v2_steps=0, why_counts={"off": 1})
    with pytest.raises(ValueError, match="reasons"): report.decode_trace(raw, result)


def test_goal_gain_loss_and_collision_to_timeout_are_distinct():
    rows = [{"case": "gain", "v8_outcome": "collision:target", "v9_outcome": "goal"},
            {"case": "loss", "v8_outcome": "goal", "v9_outcome": "collision:boundary"},
            {"case": "timeout", "v8_outcome": "collision:obstacle", "v9_outcome": "timeout"}]
    stats = report.pair_stats(rows, "v9", "v8")
    assert stats["gained_goals"] == stats["lost_goals"] == stats["collision_to_goal"] == stats["collision_to_timeout"] == 1
    assert stats["both_fail"] == 1 and stats["matched_cases"] == 3
