"""The field-layout fine-tune scenarios (2026-09-27, F106).

Pinned: training layouts come from the Paper 2 family but never reproduce a
deployment (test) layout; they keep their panels (no CPA guard) and the field
rules; the environment hook is off in every baseline run.
"""

import numpy as np
import pytest

import constant_temp as ct
import field_training as ft
import paper2_set as p2
import targets as tgt


def test_layouts_are_paper2_family_and_feasible():
    rng = np.random.default_rng(7)
    got = [lay for lay in (ft.sample_layout(rng) for _ in range(200)) if lay is not None]
    assert len(got) > 60
    for lay in got:
        assert ft.LEG_X_RANGE[0] <= lay["start"][0] <= ft.LEG_X_RANGE[1]
        assert lay["start"][1] == 2.0 and lay["goal"][1] == 22.0
        assert len(lay["panels"]) == 3
        assert not ft._near_deployment_layout(lay["start"], lay["goal"], lay["centres"])


def test_deployment_layouts_are_recognised_as_excluded():
    for lay in p2.LAYOUTS.values():
        assert ft._near_deployment_layout(lay["start"], lay["goal"], lay["panels"])
        jittered = [(x + 0.3, y - 0.3) for x, y in lay["panels"]]
        assert ft._near_deployment_layout(lay["start"], lay["goal"], jittered)


def test_training_scenarios_keep_panels_and_field_rules():
    rng = np.random.default_rng(11)
    for _ in range(8):
        b = ft.sample(rng)
        assert len(b.flags["fixed_obstacles"]) == 3
        if b.encounter_class != "no_target":
            lo, hi = ct.FIELD_TARGET_SPEED_RANGE
            assert lo <= b.target_speed <= hi
            assert b.flags["target_stop_box"] == p2.stop_box()
            assert b.target_behaviour in (tgt.T_CV, tgt.T_VS)


def test_field_mix_is_off_by_default():
    from env import ASVLidarEnv
    env = ASVLidarEnv(render_mode=None, scenario_stage=5)
    assert env.field_mix == 0.0
    env.reset(seed=5)
    assert not (env.scenario and (env.scenario.flags or {}).get("set") == "field_training")
    env.set_field_mix(1.0)
    env.reset(seed=6)
    assert env.scenario.flags.get("set") == "field_training"
    assert len(env.obstacles) == 3
