"""Tune loop for safety layer v2: the v3 three-panel development set (150), kept 3 M policy."""
import sys, time, json, os
_unknown_modes = set(sys.argv[1].split(",")) - {"off", "v2", "v3"} if len(sys.argv) > 1 else set()
if _unknown_modes:
    raise SystemExit("Unsupported legacy safety mode(s): " + ", ".join(sorted(_unknown_modes))
                     + ". This runner accepts off,v2,v3. For v4, run: "
                     + "python tools/diagnostics/safety/paired_eval.py v4 <tag>")
sys.path[:0] = ["src", "tools/tiers"]
import pandas as pd
import constants as cfg, curriculum, train_formulation as tf, formulation_v3 as fv
from common import load_model, run_episode
curriculum.apply_stage(tf.PROPULSION_STAGE)
from env import ASVLidarEnv
import safety_v2 as _sv
import safety_v3 as _sv3
for kv in filter(None, os.environ.get('V3_SET', '').split(',')):
    k, v = kv.split('='); setattr(_sv3, k, type(getattr(_sv3, k))(eval(v))); print('set v3', k, getattr(_sv3, k), flush=True)
for kv in filter(None, os.environ.get('V2_SET', '').split(',')):
    k, v = kv.split('='); setattr(_sv, k, type(getattr(_sv, k))(eval(v))); print('set', k, getattr(_sv, k), flush=True)
scen = fv.field_development_set()
import os
only = os.environ.get('V2_CASES')
idx = list(range(len(scen)))
if only:
    keep = set(only.split(','))
    idx = [i for i in idx if scen[i].case_id in keep]
model = load_model("runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip")
env = ASVLidarEnv(render_mode=None, emergency_stop=True)
modes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["off", "v2"]
rows = []
for mode in modes:
    env.estop_enabled = mode != "off"
    cfg.SAFETY_VERSION = {"v2": 2, "v3": 3}.get(mode, 1)
    t0 = time.time()
    for i in idx:
        b = scen[i]
        r = run_episode(env, b, 900_000 + 120 + i, "model", model)
        rows.append(dict(mode=mode, case=b.case_id, enc=b.case_id.split("-")[1], outcome=r["outcome"],
                         v2=r["safety_v2_steps"], estops=r.get("safety_v2_brake_steps", 0)))
    print(mode, f"{(time.time() - t0) / len(idx):.1f} s/episode", flush=True)
d = pd.DataFrame(rows)
tag = sys.argv[2] if len(sys.argv) > 2 else "iter2"
d.to_csv(f"results/safety_v2_dev_{tag}.csv", index=False)
pd.set_option("display.width", 200)
print(d.groupby(["mode"]).outcome.value_counts().unstack().fillna(0).astype(int))
print(d.groupby(["enc", "mode"]).outcome.apply(lambda s: round((s == "goal").mean(), 2)).unstack())
print("interventions/episode:", d[d["mode"] == "v2"].v2.mean().round(1), "brake steps:", d[d["mode"] == "v2"].estops.mean().round(2))
