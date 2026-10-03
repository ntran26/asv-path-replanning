"""Synthetic CSV reconstruction only; no simulator/model imports or episodes."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/v8_followup_audit.py"
SPEC = importlib.util.spec_from_file_location("v8_saved_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def inputs():
    base = []
    for case, outcome, fire in [("TS2:A", "goal", False), ("TS2:B", "collision:target", True),
                                ("DV3:C", "goal", True)]:
        base.append(dict(case=case, set="fixture", seed="1", outcome=outcome, steps="1",
                         v4_fires="0", v4_first_fire="", v7_fires=str(int(fire)), v7_first_fire="0" if fire else ""))
    variant = [dict(base[1], v7_fires="0", v7_first_fire=""), dict(base[2])]
    def steps(episodes):
        return [dict(case=r["case"], step="0", v4_fire="False", v7_fire=str(r["v7_fires"] == "1"),
                     v7_why="last certificate", v7_checked_clearance="-0.2", v7_v7_repair_clearance="-0.1")
                for r in episodes]
    def branch(case, policy, result):
        return dict(case=case, version="v7", fire_index="1", step="0", main_outcome=policy,
                    branch_outcome=result, branch_steps="2", branch_interventions="1")
    return [base, steps(base), [branch("TS2:B", "collision:target", "goal"), branch("DV3:C", "goal", "goal")],
            variant, steps(variant), [branch("DV3:C", "goal", "collision:boundary")]]


def test_reconstruction_distinguishes_three_valid_sources_and_clearance_opportunity():
    rows = audit.reconstruct(*inputs())
    by_id = {r["case"]: r for r in rows}
    assert by_id["TS2:A"]["reconstruction"] == "base_v7_never_fired"
    assert by_id["TS2:B"]["reconstruction"] == "nohold_never_fired"
    assert by_id["TS2:B"]["v8_outcome"] == "collision:target"
    assert by_id["DV3:C"]["reconstruction"] == "first_fire_branch"
    assert by_id["DV3:C"]["v8_category"] == "broken"
    assert by_id["DV3:C"]["repair_clearance_not_worse"] is True
    summary = audit.summarize(rows)
    assert summary["groups"]["all"]["n"] == 3
    assert summary["groups"]["all"]["v8_categories"] == {"broken": 1, "both_goal": 1, "both_fail": 1}


def test_missing_variant_episode_cannot_be_assumed_never_fired():
    args = inputs(); args[3] = args[3][1:]
    with pytest.raises(ValueError, match="subset"):
        audit.reconstruct(*args)


def test_missing_first_fire_branch_cannot_be_assumed_policy_outcome():
    args = inputs(); args[5] = []
    with pytest.raises(ValueError, match="Missing first-fire branch"):
        audit.reconstruct(*args)


def test_missing_step_rejected():
    args = inputs(); args[4] = args[4][1:]
    with pytest.raises(ValueError, match="Missing/non-contiguous"):
        audit.reconstruct(*args)


@pytest.mark.parametrize("field,value", [("seed", "2"), ("outcome", "goal"), ("steps", "2")])
def test_variant_identity_or_policy_history_change_rejected(field, value):
    args = inputs(); args[3][0][field] = value
    with pytest.raises(ValueError, match="history differs"):
        audit.reconstruct(*args)


def test_duplicate_branch_identity_rejected():
    args = inputs(); args[5] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        audit.reconstruct(*args)


def test_later_replay_is_not_a_first_fire_counterfactual():
    args = inputs(); args[5][0]["fire_index"] = "5"
    with pytest.raises(ValueError, match="later branches"):
        audit.reconstruct(*args)
