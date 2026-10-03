"""Saved-data checks only: no simulator, policy, or scenario generation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/testset_v2_analysis.py"
SPEC = importlib.util.spec_from_file_location("testset_v2_saved_analysis", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixture():
    definitions = [dict(test_id="one", origin_id="one", source="frozen", cell="basin-null",
                        stratum="basin", variant="CV", **{"class": "null"}, digest="abc", episode_seed="42")]
    rows = [dict(definitions[0], **{"class": ""}, policy="model", safety="off", outcome="goal",
                 collided="False", collided_target="False", estops="0", safety_v2_steps="0", safety_v2_brake_steps="0")]
    digest = analysis.sha(json.dumps({"one": "abc"}, sort_keys=True, separators=(",", ":")).encode())
    return rows, definitions, {"episodes": 1, "manifest_digest": digest}


def test_blank_null_class_is_valid_but_id_and_digest_are_pinned():
    rows, definitions, metadata = fixture()
    assert analysis.validate_rows(rows, definitions, metadata)["one"]["class"] == "null"
    metadata["manifest_digest"] = "changed"
    with pytest.raises(ValueError, match="digest"):
        analysis.validate_rows(rows, definitions, metadata)


@pytest.mark.parametrize("change", [dict(test_id="unknown"), dict(safety="on"), dict(safety_v2_steps="1"),
                                    dict(outcome="invalid"), dict(collided="True"), dict(cell="other")])
def test_invalid_result_rejected(change):
    rows, definitions, metadata = fixture()
    rows[0].update(change)
    with pytest.raises(ValueError):
        analysis.validate_rows(rows, definitions, metadata)


def test_duplicate_result_identity_rejected():
    rows, definitions, metadata = fixture()
    metadata["episodes"] = 2
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.validate_rows(rows * 2, definitions * 2, metadata)


def test_fixed_panel_count_takes_precedence_over_unused_requested_count():
    built = SimpleNamespace(flags={"fixed_obstacles": [[], [], []]}, n_obstacles=0)
    features = analysis.scenario_features(built)
    assert features["requested_obstacle_count"] == 0
    assert features["fixed_obstacle_count"] == features["obstacle_count_for_matching"] == 3


def test_cached_seed_and_digest_guard_without_environment():
    rows, definitions, metadata = fixture()
    by_id = analysis.validate_rows(rows, definitions, metadata)
    built = SimpleNamespace(flags={}, n_obstacles=0, digest=lambda: "abc")
    item = dict(test_id="one", episode_seed=42, built=built)
    assert analysis.attach_scenarios(rows, by_id, [item], metadata, metadata)[0]["episode_seed"] == 42
    item["episode_seed"] = 43
    with pytest.raises(ValueError, match="seed/scenario digest"):
        analysis.attach_scenarios(rows, by_id, [item], metadata, metadata)


def row(name, width, outcome="goal", variant="CV", cell="cell"):
    return dict(test_id=name, source="frozen", cell=cell, variant=variant, scenario_class="crossing",
                stratum="basin", nominal_width=width, outcome=outcome, episode_seed=1,
                scenario_sha256=name, steps=123, rms_cte=999)


def test_controls_use_same_cell_variant_static_distance_and_deterministic_ties():
    rows = [row("failure", 5., "collision:target"), row("z-tie", 6.), row("a-tie", 4.),
            row("wrong-variant", 5., variant="RE"), row("wrong-cell", 5., cell="other")]
    matches, scales = analysis.match_successes(rows)
    assert [r["control_id"] for r in matches] == ["a-tie", "z-tie"]
    assert list(scales["frozen/cell"]) == ["nominal_width"]
    rows[1]["steps"], rows[2]["rms_cte"] = 999999, -1
    assert analysis.match_successes(rows)[0] == matches


def test_unmatched_failure_explicit_and_group_denominator_includes_collisions():
    rows = [row("f", 5., "collision:boundary"), row("g", 6., variant="RE")]
    matches, _ = analysis.match_successes(rows)
    assert len(matches) == 1 and matches[0]["control_id"] == ""
    assert matches[0]["unmatched_reason"] == "no successful same-cell/variant case"
    overall = analysis.group_counts(rows)[0]
    assert overall["n"] == 2 and overall["goal_rate"] == 0.5 and overall["boundary"] == 1
