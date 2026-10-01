"""Safety layer v2 (2026-10-01): the predictive filter around the policy."""
import numpy as np
import pytest

import constants as cfg
import curriculum
import paper2_set as p2
import train_formulation as tf
from env import ASVLidarEnv


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


def _l1_no_target(env):
    recs, _ = p2.build(env, layouts=["L1"], encounters=[], include_no_target=True)
    return recs[0]


def test_filter_steers_off_an_obstacle_dead_ahead_and_leaves_open_water_alone(v2):
    env = ASVLidarEnv(render_mode=None, emergency_stop=True)
    rec = _l1_no_target(env)                       # L1: a static obstacle on the leg at (5, 8)
    env.reset(seed=rec["episode_seed"], options={"generated": rec["built"]})
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not env.safety_v2_changed               # 5.7 m off, room to spare: untouched
    changed, executed, quiet_after = 0, [], False
    for _ in range(60):                            # hold straight at cruise toward it
        _, _, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        changed += int(env.safety_v2_changed)
        executed.append((env._safety_v2.last.get("chosen"), env.u_body, env.asv_y))
        if env.asv_y > 10.0 and not env.safety_v2_changed:
            quiet_after = True                     # past it: the policy has the helm again
        if term or trunc:
            break
    assert not info.get("collided")
    assert env.asv_y > 10.0                        # got past the obstacle ...
    assert env.safety_v2_steps == changed
    assert any(c is not None and abs(c[0]) > 0.0 for c, _, _ in executed)   # ... by turning,
    assert min(u for _, u, y in executed if y < 10.0) > 0.3   # not by stopping (keeps steerage)
    assert quiet_after


def test_filter_is_off_with_the_safety_layer_off(v2):
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    rec = _l1_no_target(env)
    env.reset(seed=rec["episode_seed"], options={"generated": rec["built"]})
    for _ in range(40):
        _, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert not env.safety_v2_changed
        if term or trunc:
            break
    assert env.safety_v2_steps == 0
