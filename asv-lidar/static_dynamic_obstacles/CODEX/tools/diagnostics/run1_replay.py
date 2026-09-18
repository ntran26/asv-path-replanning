import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import constants as cfg, curriculum
import train_formulation as tf
from env import ASVLidarEnv
from stable_baselines3 import PPO
model = PPO.load(ROOT / "runs/ppo_formulation_seed0/ppo_1750000_steps.zip", device="cpu")
curriculum.apply_stage(tf.PROPULSION_STAGE)
env = ASVLidarEnv(render_mode=None)
scen = tf.development_set(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
rows = []
for i, built in enumerate(scen):
    obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
    sp, terms, trace = [], defaultdict(float), []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        sp.append(info["speed_mps"])
        for k, v in info.items():
            if k.startswith("reward/weighted/") or k.startswith("reward/terminal") or k == "reward/intervention":
                terms[k.split("/")[-1]] += float(v)
        rng = min([np.hypot(t.x-env.asv_x, t.y-env.asv_y) for t in env.targets] or [np.inf])
        trace.append((round(info["speed_mps"],2), round(float(a[0]),2), round(float(a[1]),2), round(rng,2)))
        if term or trunc: break
    rows.append(dict(cls=built.encounter_class, dcpa=round(built.dcpa_m,2), tcpa=round(getattr(built,'tcpa_s',float('nan')),1),
        goal=info["reached_goal"], coll=info["collision_kind"], both=bool(info["reached_goal"] and info["collided"]),
        steps=len(sp), mean_u=round(float(np.mean(sp)),2), max_u=round(float(np.max(sp)),2), ret=round(sum(terms.values()),1),
        terms={k: round(v,1) for k, v in terms.items()}, tail=trace[-6:], head=trace[:4]))
for r in rows:
    print(json.dumps(r))
print("info keys sample:", sorted(k for k in info if k.startswith("reward")))
