"""C14: what the run 2 model perceives and does before CPA in head-on encounters.

Logs, over pre-CPA frames with the target inside 8 m: perceived and true class,
heading-intersection deviation from 180 deg, how often the starboard alteration
is inadmissible, and the minimum speed.  Found (F54): the ship stops before CPA
in 74 % of target collisions, and the starboard turn is inadmissible in 66 % of
frames at 5 m.

    python tools/diagnostics/c14_pre_cpa.py
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

from c14_common import OUT, WIDTH_BINS, head_on_scenarios, load_run2_model
from env import ASVLidarEnv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scen = head_on_scenarios()
    model = load_run2_model()
    env = ASVLidarEnv(render_mode=None)
    rows = []
    for i, b in enumerate(scen):
        env.forced_num_obs = 0
        obs, _ = env.reset(seed=700_000 + i, options={"generated": b})
        cls, cls_true, ct, blocked, speeds, pre = Counter(), Counter(), [], 0, [], 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            t = env.targets[0]
            rng = math.hypot(t.x - env.asv_x, t.y - env.asv_y)
            for ctx in env.encounter_contexts.values():
                if rng < 8.0 and ctx.tcpa > 0.0:
                    pre += 1
                    cls[str(ctx.cls)] += 1
                    cls_true[str(ctx.cls_true)] += 1
                    ct.append(abs(((ctx.ct - 180.0 + 180.0) % 360.0) - 180.0))
                    blocked += int(not ctx.a_stbd)
                    speeds.append(info["speed_mps"])
            if term or trunc:
                break
        n = max(pre, 1)
        rows.append(dict(idx=i, width=b.nominal_width, bend=b.bend_deg, dcpa=b.dcpa_m,
                         coll=info["collision_kind"] == "target",
                         outcome=info["collision_kind"] or ("goal" if info["reached_goal"] else "timeout"),
                         pre_frames=pre, p_head_on=cls.get("head_on", 0) / n,
                         p_crossing=cls.get("crossing", 0) / n, p_none=cls.get("none", 0) / n,
                         p_true_head_on=cls_true.get("head_on", 0) / n,
                         ct_dev_med=float(np.median(ct)) if ct else np.nan,
                         p_stbd_blocked=blocked / n, u_min=min(speeds) if speeds else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "c14_preCPA.csv", index=False)
    d["wbin"] = pd.cut(d.width, *WIDTH_BINS)
    d["stopped"] = d.u_min < 0.2
    pd.set_option("display.width", 250)
    print(d.groupby("wbin", observed=True).agg(
        n=("idx", "size"), coll=("coll", "mean"), p_head_on=("p_head_on", "mean"),
        p_crossing=("p_crossing", "mean"), p_true_head_on=("p_true_head_on", "mean"),
        ct_dev=("ct_dev_med", "median"), stbd_blocked=("p_stbd_blocked", "mean"),
        u_min=("u_min", "median")).round(2))
    print(d.pivot_table(index="wbin", columns="stopped", values="coll",
                        aggfunc=["mean", "size"], observed=True).round(2))
    print("share of target collisions stopped before CPA:", round(d[d.coll].stopped.mean(), 2))


if __name__ == "__main__":
    main()
