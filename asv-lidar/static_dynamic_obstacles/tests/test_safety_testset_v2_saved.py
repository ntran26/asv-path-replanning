"""Synthetic provenance checks; no simulator imports or episodes."""
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/analyze_testset_v2_saved.py"
SPEC = importlib.util.spec_from_file_location("testset_v2_saved", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def definition(identifier="A", digest="scene", seed="12"):
    return dict(test_id=identifier, digest=digest, episode_seed=seed, cell="basin", source="frozen")


def record(mode="off", outcome="goal", **changes):
    value = dict(test_id="A", scenario_sha256="scene", seed=12, suite="frozen_b", case="B-00-001",
                 mode=mode, outcome=outcome, steps=10, safety_v2_steps=2, safety_v2_brake_steps=1)
    value.update(changes)
    return value


def test_exact_pair_and_episode_trade_counts():
    pairs, coverage, _, excluded = audit.pair_rows([definition()], [record(), record("v4", "collision:target")])
    summary = audit.aggregate(pairs)
    assert coverage[0]["saved_status"] == "paired"
    assert summary["transitions"] == {"lost_goal": 1}
    assert summary["v4_contacts"] == 1
    assert summary["interventions_by_transition"]["lost_goal"]["recorded_intervention_steps"] == 2
    assert excluded == {}


@pytest.mark.parametrize("change", [{"seed": 13}, {"scenario_sha256": "replacement"}])
def test_same_id_different_seed_or_scene_never_pairs(change):
    pairs, coverage, _, excluded = audit.pair_rows([definition()], [record(), record("v4", **change)])
    assert pairs == []
    assert coverage[0]["saved_status"] == "off_only"
    assert excluded == {"scene_or_seed_mismatch": 1}


def test_replacement_id_never_joins_removed_scene():
    pairs, coverage, _, excluded = audit.pair_rows([definition("P2v2-L1")], [record(test_id="P2-L1")])
    assert pairs == []
    assert coverage[0]["saved_status"] == "unrecorded"
    assert excluded == {"not_in_current_v2": 1}


def test_duplicate_committed_result_is_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate committed"):
        audit.pair_rows([definition()], [record(), record()])


def test_incomplete_final_record_is_not_a_result():
    rows, tail = audit.committed_records(b'{"mode":"off"}\n{"mode":')
    assert rows == [{"mode": "off"}]
    assert tail == len(b'{"mode":')


def test_settings_allow_only_suite_difference():
    first = dict(suites=["B"], checkpoint_sha256="a", source_sha256={"env": "original"})
    second = dict(first, suites=["R"])
    assert audit.settings_identity(first) == audit.settings_identity(second)
    second["source_sha256"] = {"env": "changed"}
    assert audit.settings_identity(first) != audit.settings_identity(second)


def test_successful_intervention_is_not_counted_as_lost_goal():
    pairs, _, _, _ = audit.pair_rows([definition()], [record(), record("v4")])
    result = audit.aggregate(pairs)
    assert result["transitions"] == {"preserved_goal": 1}
    assert result["interventions_by_transition"]["preserved_goal"]["episodes_with_recorded_intervention"] == 1
    assert result["interventions_by_transition"]["lost_goal"]["episodes"] == 0
