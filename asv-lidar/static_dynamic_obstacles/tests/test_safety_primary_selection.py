"""Primary benchmark routing without simulator or policy execution."""
import importlib.util
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("primary_safety_runner", DIRECTORY / "safety_candidate_iteration.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_default_is_frozen_entire_v3_selection():
    assert runner.DEFAULT_SELECTION == runner.ROOT / "results/safety_dev/testset_v3_main/selection.json"


def test_primary_cases_use_separate_output_directory():
    assert runner.campaign_directory([{"case": "TS3:A"}, {"case": "TS3:B"}]) == runner.PRIMARY_CAMPAIGN


def test_explicit_historical_selection_keeps_its_output_directory():
    assert runner.campaign_directory([{"case": "TS2:A"}, {"case": "DV3:B"}]) == runner.CAMPAIGN


def test_primary_and_historical_cases_cannot_be_silently_pooled():
    with pytest.raises(ValueError, match="Do not mix"):
        runner.campaign_directory([{"case": "TS3:A"}, {"case": "TS2:A"}])


def test_primary_subset_is_not_reported_as_complete_suite():
    selection = {"evaluation_role": "primary", "cases": list(range(1000))}
    assert "Explicit primary-benchmark subset: 40/1000" in runner.evaluation_scope(selection, list(range(40)))
    assert "Full primary benchmark: 1000/1000" in runner.evaluation_scope(selection, selection["cases"])


def test_legacy_scope_is_preserved():
    assert runner.evaluation_scope({"scope": "original development selection"}, []) == "original development selection"
