import sys, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import constants as cfg, curriculum
import train_formulation as tf
from env import ASVLidarEnv
from stable_baselines3 import PPO
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
RUN = ROOT / "runs" / "ppo_formulation_seed0_v2"
scen = tf.development_set(20)
def rel_bearing(b):
    if b.encounter_class == "no_target": return np.nan
    dx, dy = b.target_spawn[0]-b.own_spawn[0], b.target_spawn[1]-b.own_spawn[1]
    return (math.degrees(math.atan2(dx, dy)) - b.own_heading) % 360
geo = pd.DataFrame([dict(idx=i, cls=b.encounter_class, dcpa=b.dcpa_m, ct=b.ct_deg, k=b.speed_ratio,
                         tcpa=b.tcpa_s, brg=rel_bearing(b), w=b.nominal_width) for i, b in enumerate(scen)])
geo['side'] = np.where(geo.brg < 180, 'stbd', 'port')
e = pd.read_csv(RUN / "eval_episodes.csv", keep_default_na=False, na_values=[""])
e['idx'] = e.groupby('timesteps').cumcount()
assert (e.groupby('timesteps').size() == len(scen)).all()
late = e[e.timesteps >= 1_600_000].merge(geo, on='idx')
assert (late['class'] == late.cls).all()
late['coll'] = late.outcome.str.startswith('collision')
late['dbin'] = pd.cut(late.dcpa, [-0.01, 0.7, 1.4, 2.6, 99])
print("== collision rate by class x DCPA bin (late evals, 3 x 20/class)")
print(late[late.cls != 'no_target'].pivot_table(index='cls', columns='dbin', values='coll', aggfunc=['mean', 'size'], observed=True).round(2))
print("== crossing by target side (stbd = own gives way)")
cr = late[late.cls == 'crossing']
print(cr.pivot_table(index='side', columns='dbin', values='coll', aggfunc=['mean', 'size'], observed=True).round(2))
print(pd.crosstab(cr.side, cr.outcome))
print("== head-on by width"); ho = late[late.cls == 'head_on']
print(ho.pivot_table(index=pd.cut(ho.w, [0, 7, 11]), columns='dbin', values='coll', aggfunc=['mean', 'size'], observed=True).round(2))
print("== being overtaken: outcome by k and dcpa"); bo = late[late.cls == 'being_overtaken']
print(bo.pivot_table(index=pd.cut(bo.k, [1.4, 1.8, 2.3]), columns='dbin', values='coll', aggfunc=['mean', 'size'], observed=True).round(2))
print(pd.crosstab(pd.cut(bo.w, [0, 7, 8.5, 11]), bo.outcome))

# replay the final model: does it alter before CPA?
curriculum.apply_stage(tf.PROPULSION_STAGE)
model = PPO.load(RUN / "final_model.zip", device="cpu")
env = ASVLidarEnv(render_mode=None)
rows = []
for i, b in enumerate(scen):
    if b.encounter_class not in ("crossing", "being_overtaken", "head_on"): continue
    obs, _ = env.reset(seed=900_000 + i, options={"generated": b})
    h0, t, hdg, spd, thr, rud, rng = env.asv_h, 0, [], [], [], [], []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a); t += cfg.UPDATE_RATE
        hdg.append(((env.asv_h - h0 + 180) % 360) - 180); spd.append(info["speed_mps"]); thr.append(float(a[1])); rud.append(float(a[0]))
        rng.append(min([math.hypot(x.x-env.asv_x, x.y-env.asv_y) for x in env.targets] or [np.inf]))
        if term or trunc: break
    n_cpa = max(1, min(len(hdg), int(b.tcpa_s / cfg.UPDATE_RATE)))
    pre = slice(0, n_cpa)
    rows.append(dict(idx=i, cls=b.encounter_class, side='stbd' if rel_bearing(b) < 180 else 'port', dcpa=round(b.dcpa_m, 2), w=round(b.nominal_width, 1),
                     out=(f"coll:{info['collision_kind']}" if info['collided'] else 'goal' if info['reached_goal'] else 'timeout'),
                     t_end=t, tcpa=round(b.tcpa_s, 1), hdg_min_pre=round(min(hdg[pre]), 1), hdg_max_pre=round(max(hdg[pre]), 1),
                     u_min_pre=round(min(spd[pre]), 2), u_max_pre=round(max(spd[pre]), 2), thr_mean_pre=round(np.mean(thr[pre]), 2),
                     min_rng=round(min(rng), 2)))
R = pd.DataFrame(rows)
R['altered'] = (R.hdg_max_pre.abs().clip(lower=0).combine(R.hdg_min_pre.abs(), max) >= 15) | (R.u_min_pre <= 0.6 * cfg.U_REF)
print("== replay final model (nominal hull): acted before CPA? (|dpsi|>=15 deg or u<=0.6 U_REF)")
print(R.groupby(['cls', 'side', 'altered']).out.value_counts().unstack(fill_value=0))
print(R[R.cls == 'crossing'].sort_values(['side', 'dcpa']).to_string(index=False))
print(R[R.cls == 'being_overtaken'].sort_values('dcpa').to_string(index=False))
