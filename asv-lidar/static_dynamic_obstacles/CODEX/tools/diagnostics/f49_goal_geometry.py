import sys
from pathlib import Path
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
scen = tf.development_set(6)
for i in (19, 21, 22, 23, 30):
    built = scen[i]
    obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
    bp = np.asarray(env.boundary_polygon.exterior.coords) if hasattr(env.boundary_polygon, "exterior") else None
    print(i, built.encounter_class, "start", round(env.asv_x,2), round(env.asv_y,2), "goal", round(env.goal_x,2), round(env.goal_y,2),
          "path_end", np.round(env.path.points[-1],2), "path_len", round(env.path.length,2),
          "channel y-extent", None if bp is None else (round(bp[:,1].min(),2), round(bp[:,1].max(),2)))
    while True:
        a, _ = model.predict(obs, deterministic=True)
        prev = (env.asv_x, env.asv_y, env.asv_h)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc: break
    hull = env.hull_polygon()
    print("   end: pos", round(env.asv_x,2), round(env.asv_y,2), "h", round(env.asv_h,1), "prev", np.round(prev,2),
          "d_goal", round(env.distance_to_goal,2), "cte", round(env.cross_track_error,2), "goal", info["reached_goal"], "coll", info["collision_kind"],
          "hull y-max", round(max(p[1] for p in np.asarray(hull.exterior.coords)),2) if hasattr(hull,'exterior') else None,
          "targets", [(round(t.x,2), round(t.y,2)) for t in env.targets])
