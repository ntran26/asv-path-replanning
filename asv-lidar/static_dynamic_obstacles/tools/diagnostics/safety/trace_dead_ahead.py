import sys
sys.path[:0] = ["src", "tools/tiers"]
import numpy as np
import constants as cfg, curriculum, train_formulation as tf, paper2_set as p2, safety_v2 as sv
curriculum.apply_stage(tf.PROPULSION_STAGE)
cfg.SAFETY_VERSION = 3
from env import ASVLidarEnv
env = ASVLidarEnv(render_mode=None, emergency_stop=True)
recs, _ = p2.build(env, layouts=["L1"], encounters=[], include_no_target=True)
r = recs[0]
env.reset(seed=r["episode_seed"], options={"generated": r["built"]})
orig = sv.SafetyFilterV2.filter
for k in range(60):
    _, _, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    f = env._safety_v2
    print(k, f"x={env.asv_x:5.2f} y={env.asv_y:5.2f} u={env.u_body:4.2f}", {kk: f.last.get(kk) for kk in ("mode","why","policy_margin","best_margin","chosen")}, flush=True)
    if term or trunc:
        print("end", info.get("collided"), info.get("collision_kind")); break
