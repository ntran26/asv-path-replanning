"""Safety layer v8 (2026-10-03): v7 without the hold-back fallback."""
import math

import numpy as np
import pytest

import constants as cfg
import curriculum
import paper2_set as p2
import safety_v2
import train_formulation as tf
from env import ASVLidarEnv


@pytest.fixture()
def v8():
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    old = getattr(cfg, "SAFETY_VERSION", None)
    cfg.SAFETY_VERSION = 8
    yield
    if old is None:
        del cfg.SAFETY_VERSION
    else:
        cfg.SAFETY_VERSION = old


def test_v8_is_selected_never_holds_back_and_restores_the_shared_constant(v8):
    import safety_v8
    env = ASVLidarEnv(render_mode=None, emergency_stop=True)
    rec = p2.build(env, layouts=["L2"], encounters=["HO"], include_no_target=False)[0][0]
    env.reset(seed=rec["episode_seed"], options={"generated": rec["built"]})
    gain = safety_v2.HOLD_BACK_GAIN_S
    whys = []
    for _ in range(40):                       # hold straight at the L2 gate: the filter has work to do
        _, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert isinstance(env._safety_v2, safety_v8.SafetyFilterV8)
        whys.append((env._safety_v2.last or {}).get("why"))
        assert safety_v2.HOLD_BACK_GAIN_S == gain          # restored after every decision
        if term or trunc:
            break
    assert "hold back" not in whys
    assert any(w not in (None, "idle", "nominal") for w in whys)   # it did intervene
    assert math.isfinite(gain)
