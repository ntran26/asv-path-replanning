"""Opt-in straight legs for dense draws (baseline-v4 revision, 2026-10-03)."""
import copy

import numpy as np
import pytest

import constants as cfg
import field_training as ft
import scenario as scn


@pytest.fixture()
def dense_stage():
    base = copy.deepcopy(cfg.CURRICULUM_STAGES[5])
    cfg.CURRICULUM_STAGES[99] = dict(base, clutter=(3, 3), dense_straight_share=1.0, dense_panels=3)
    cfg.CURRICULUM_STAGES[98] = dict(base, clutter=(3, 3))
    yield
    del cfg.CURRICULUM_STAGES[99], cfg.CURRICULUM_STAGES[98]


def _draw(stage, seed):
    return scn.ScenarioGenerator(stage=stage, seed_namespace="training").sample(
        seed, encounter_class="head_on", geometry_mode="basin")


def test_dense_draws_get_straight_legs_only_when_the_stage_asks(dense_stage):
    straight = [_draw(99, s) for s in range(20)]
    straight = [b for b in straight if b is not None]
    assert straight and all(abs(b.slant_realised_deg) < 1e-9 and b.n_obstacles == 3 for b in straight)
    plain = [b for b in (_draw(98, s) for s in range(20)) if b is not None]
    assert sum(abs(b.slant_realised_deg) > 2.0 for b in plain) >= len(plain) // 2


def test_a_stage_without_the_key_draws_exactly_as_before(dense_stage):
    # stage 5 has no key: two generators with the same seed agree, and the key-less
    # path never pre-draws the panel count.
    a, b = _draw(5, 1234), _draw(5, 1234)
    assert a.digest() == b.digest()
    g = scn.ScenarioGenerator(stage=5, seed_namespace="training")
    g.sample(1234, encounter_class="head_on", geometry_mode="basin")
    assert g._pre_obstacles is None and not g._force_straight


def test_field_family_straight_share():
    built = ft.sample(np.random.default_rng(7), encounter="NT", straight_share=1.0)
    start, goal = built.flags["basin_leg"]
    assert abs(start[0] - goal[0]) < 1e-9
