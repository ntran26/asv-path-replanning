"""Hand-back starts (planning/HANDBACK_STARTS_PLAN.md, 2026-10-01)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

import constants as cfg
import curriculum
import paper2_set as p2
import train_formulation as tf
from env import ASVLidarEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "diagnostics"))
import harvest_handback_starts as hb


@pytest.fixture()
def v2():
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    old = getattr(cfg, "SAFETY_VERSION", None)
    cfg.SAFETY_VERSION = 2
    yield
    if old is None:
        del cfg.SAFETY_VERSION
    else:
        cfg.SAFETY_VERSION = old


def _harvest():
    env = ASVLidarEnv(render_mode=None, emergency_stop=True)
    recs, _ = p2.build(env, layouts=["L1"], encounters=[], include_no_target=True)
    rec = recs[0]                                   # straight at the L1 obstacle: the filter steps in
    records, _, after = hb.collect(env, rec["built"], rec["episode_seed"],
                                   lambda obs: np.array([0.0, 0.0], dtype=np.float32), "test")
    return records, after


def test_a_hand_back_start_resumes_where_the_filter_handed_back(v2, tmp_path):
    records, after = _harvest()
    assert records, "the filter never handed back"
    r = records[0]
    assert len(r["states"]) == len(r["actions"]) == len(r["brakes"]) == r["k"]
    path = tmp_path / "pool.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"meta": {}, "records": [r]}, fh)

    env = ASVLidarEnv(render_mode=None, scenario_stage=5, emergency_stop=False)
    env.set_start_pool(str(path), share=1.0, min_stage=5, jitter=False)
    obs, info = env.reset(seed=3)
    assert info.get("handback_start") and env.handback_starts == 1
    assert env.step_count == r["k"]
    x, y, h = after[r["k"] - 1][:3]                # where the harvested episode was then
    assert np.hypot(env.asv_x - x, env.asv_y - y) < 0.05
    assert abs((env.asv_h - h + 180.0) % 360.0 - 180.0) < 2.0
    assert set(obs) == set(env.observation_space.spaces)


def test_no_pool_or_an_early_stage_draws_nothing(v2, tmp_path):
    records, _ = _harvest()
    path = tmp_path / "pool.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"meta": {}, "records": records[:1]}, fh)
    env = ASVLidarEnv(render_mode=None, scenario_stage=5, emergency_stop=False)
    _, info = env.reset(seed=3)
    assert not info.get("handback_start") and env.step_count == 0
    env.set_start_pool(str(path), share=1.0, min_stage=6, jitter=False)   # stage 5 < 6
    _, info = env.reset(seed=3)
    assert not info.get("handback_start") and env.step_count == 0
