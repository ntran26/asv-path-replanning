"""Regression checks for actual scene coverage, replay, and learning gates."""
import json

import numpy as np
import pytest

import constants as cfg
from env import ASVLidarEnv
from scene_sampling import (FINAL_WEIGHTS, STRATA, SceneTrainingEnv, canonical_hash,
                            case_record, checkpoint_rank, draw_stratum, mastery_passes,
                            restore_case, sample_scene, write_manifest)


def row(stratum="empty", cls="no_target", **changes):
    value = dict(stratum=stratum, **{"class": cls}, outcome="goal", mean_abs_cte=0.1,
                 mean_speed_error=0.05 * cfg.U_NOM, violation_fraction=0.0,
                 wrong_turn_fraction=0.0, estops=0)
    value.update(changes)
    return value


def test_final_sampling_keeps_separate_empty_static_dynamic_and_combined():
    rng = np.random.default_rng(331)
    counts = {s: 0 for s in STRATA}
    for _ in range(10000):
        counts[draw_stratum(rng, 5)] += 1
    assert all(abs(counts[s] / 10000 - FINAL_WEIGHTS[s]) < 0.015 for s in STRATA)


@pytest.mark.parametrize("stratum", STRATA)
def test_scene_wrapper_realizes_requested_stratum(stratum):
    wrapper = SceneTrainingEnv(ASVLidarEnv(render_mode=None), level=4,
                               recovery_probability=0.0)
    try:
        observation, info = wrapper.reset(seed=311, options={"scene_stratum": stratum,
            "encounter_class": "no_target" if stratum in ("empty", "static") else "head_on"})
        assert wrapper.observation_space.contains(observation)
        assert bool(wrapper.unwrapped.targets) == (stratum in ("dynamic", "combined"))
        assert bool(wrapper.unwrapped.obstacles) == (stratum in ("static", "combined"))
        assert info["scene_stratum"] == stratum
        _, _, _, _, info = wrapper.step(np.zeros(2, dtype=np.float32))
        assert info["scene_stratum"] == stratum
    finally:
        wrapper.close()


def test_mastery_requires_retained_skills_and_a_complete_sample():
    good = [row()] * 4 + [row("static")] * 4
    assert mastery_passes(good, 2)
    assert not mastery_passes(good[:4], 2)
    assert not mastery_passes(good[:-1], 2)
    assert not mastery_passes([row(outcome="collision:boundary")] + good[1:], 2)
    assert not mastery_passes([row(mean_abs_cte=1)] * 4 + good[4:], 2)
    assert not mastery_passes([row(mean_speed_error=cfg.U_NOM)] * 4 + good[4:], 2)


def test_mastery_cannot_hide_failed_target_class_or_wrong_turn():
    rows = [row()] * 4 + [row("static")] * 4
    targets = [row("dynamic", cls) for cls in ("head_on", "null")] * 2
    assert mastery_passes(rows + targets, 3)
    assert not mastery_passes(rows + [row("dynamic", "head_on")] * 4, 3)
    assert not mastery_passes(rows + [dict(r, wrong_turn_fraction=0.20) for r in targets], 3)


def test_checkpoint_selection_considers_safety_worst_stratum_and_turning():
    safe = [row(), row("static")]
    assert checkpoint_rank(safe) > checkpoint_rank([row(outcome="collision:target"), row("static")])
    assert checkpoint_rank(safe) > checkpoint_rank([row(), row("static", outcome="timeout")])
    assert checkpoint_rank(safe) > checkpoint_rank([row(wrong_turn_fraction=0.25), row("static")])
    assert checkpoint_rank(safe) > checkpoint_rank([row(mean_abs_cte=1.0), row("static")])
    assert checkpoint_rank(safe) > checkpoint_rank([row(min_hull_clearance=0.02), row("static")])
    assert not mastery_passes([row(min_boundary_clearance=0.01)] * 4, 1)


def test_realized_manifest_round_trips_static_geometry_and_observation(tmp_path):
    env = ASVLidarEnv(render_mode=None)
    try:
        seed = 204132
        built = sample_scene(seed, "static", 2, namespace="development")
        options = {"generated": built, "recovery_fraction": 0.04, "recovery_heading_deg": 3.0}
        before, _ = env.reset(seed=seed, options=options)
        saved = case_record(built, env, seed, options)
        manifest = write_manifest(tmp_path / "manifest.json", [saved], {"version": "test"})
        restored = json.loads((tmp_path / "manifest.json").read_text())
        assert restored["cases_sha256"] == canonical_hash(restored["cases"])
        assert manifest["source_sha256"]["scene_sampling.py"]
        _, replay = restore_case(restored["cases"][0])
        after, _ = env.reset(seed=seed, options=replay)
        assert env.obstacles == [[tuple(point) for point in poly] for poly in saved["scenario"]["obstacles"]]
        assert np.allclose([env.asv_x, env.asv_y, env.asv_h], saved["initial_pose"])
        for key in before:
            np.testing.assert_allclose(after[key], before[key], atol=1e-7)
    finally:
        env.close()


def test_schema_guard_rejects_legacy_checkpoint_and_output_escape(tmp_path):
    from train_formulation import output_directory, validate_resume_schema
    with pytest.raises(ValueError, match="no config.json"):
        validate_resume_schema(tmp_path / "legacy.zip", {"version": "new"})
    (tmp_path / "config.json").write_text(json.dumps({"observation_schema": {"version": "old"}}))
    with pytest.raises(ValueError, match="differs"):
        validate_resume_schema(tmp_path / "legacy.zip", {"version": "new"})
    with pytest.raises(ValueError, match="must stay under"):
        output_directory(tmp_path, "unscoped")


def test_trajectory_turn_check_uses_true_heading_latch_and_exempts_emergency():
    from types import SimpleNamespace
    from train_formulation import trajectory_wrong_turn
    context = SimpleNamespace(t_engage=10, psi_engage=45.0, engaged=True,
                              in_extremis=False, turn_admissible=True, compliant_turn_sense=1)
    latches = {}
    assert not trajectory_wrong_turn(0.0, {1: context}, latches)
    assert not trajectory_wrong_turn(15.0, {1: context}, latches)
    assert trajectory_wrong_turn(345.0, {1: context}, latches)
    context.in_extremis = True
    assert not trajectory_wrong_turn(345.0, {1: context}, latches)
    context.in_extremis = False
    context.t_engage = 20
    assert not trajectory_wrong_turn(345.0, {1: context}, latches)
