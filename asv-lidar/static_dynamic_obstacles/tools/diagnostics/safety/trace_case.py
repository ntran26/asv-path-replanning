import sys
sys.path[:0] = ["src", "tools/tiers"]
import numpy as np
import constants as cfg, curriculum, train_formulation as tf, formulation_v3 as fv
from common import load_model
curriculum.apply_stage(tf.PROPULSION_STAGE)
cfg.SAFETY_VERSION = 3
from env import ASVLidarEnv
import os, safety_v3 as _sv
for kv in filter(None, os.environ.get('V3_SET', '').split(',')):
    k, v = kv.split('='); setattr(_sv, k, type(getattr(_sv, k))(eval(v)))
print("building set", flush=True); scen = fv.field_development_set(); print("built", flush=True)
case = sys.argv[1]
i = [k for k, b in enumerate(scen) if b.case_id == case][0]
model = load_model("runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip")
env = ASVLidarEnv(render_mode=None, emergency_stop=True)
obs, _ = env.reset(seed=900_000 + 120 + i, options={"generated": scen[i]})
actor = tf.EpisodeActor(model)
print("obstacles", [np.asarray(o).mean(axis=0).round(1).tolist() for o in env.obstacles] if hasattr(env, "obstacles") else "")
log = []
for k in range(400):
    a = actor(obs)
    obs, r, term, trunc, info = env.step(a)
    f = env._safety_v2.last
    log.append(f"{k:3d} x={env.asv_x:5.2f} y={env.asv_y:5.2f} h={env.asv_h:6.1f} u={env.u_body:4.2f} pol={np.round(a,2).tolist()} {f.get('mode')} {f.get('why')} pm={f.get('policy_margin')} bm={f.get('best_margin')} cont={f.get('continuation_ok')} ch={f.get('chosen')} br={f.get('brake')} ch?={f.get('changed')}")
    if term or trunc:
        log.append(f"end {info.get('collided')} {info.get('collision_kind')} {info.get('outcome')}"); break
print("\n".join(log[-45:]))
