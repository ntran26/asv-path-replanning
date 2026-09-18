"""Are generated crossings avoidable at all?

Tier 0's scripted compliant alteration (30 deg, compliant sense) still hit the
target in 10 of 12 crossings, and Tier 2's crossing collision stayed near 0.5.
This replays the development crossings (20 per side-agnostic draw, obstacles
off) under oracle-ish scripted responses, taken the moment the scenario starts
rather than on engagement, so late engagement cannot be the excuse:

* `follower`   — hold path and speed;
* `stop`       — full astern from t = 0 (the Rule 8(e) extreme);
* `slow`       — half the cruise speed from t = 0;
* `turn30/60`  — an immediate committed alteration in the compliant sense
                 (starboard for a target from starboard, port from port);
* `best`       — the least-colliding of the above, per scenario.

If `best` still hits the target in a large share, the draw is infeasible for
the own ship and the class needs a generator fix, not more training.

    python tools/diagnostics/crossing_feasibility.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import common  # noqa: E402
import curriculum  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT = ROOT / "results" / "crossing_feasibility"
POLICIES = ("follower", "stop", "slow", "turn30", "turn60")


def action(env, policy, h0, sense):
    if policy == "follower":
        return common.follower_action(env)
    if policy == "stop":
        return np.array([common.follower_action(env)[0], -1.0], dtype=np.float32)
    if policy == "slow":
        # throttle -0.5 maps to half the RPM delta below cruise
        return np.array([common.follower_action(env)[0], -0.5], dtype=np.float32)
    alteration = 30.0 if policy == "turn30" else 60.0
    want = h0 + sense * alteration
    rudder = float(np.clip(((want - env.asv_h + 180.0) % 360.0 - 180.0) / 20.0, -1.0, 1.0))
    return np.array([rudder, 0.0], dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None)
    env.estop_enabled = False
    scen = common.development_set(20, classes=("crossing",))
    rows = []
    for i, b in enumerate(scen):
        sense = -1 if b.ct_deg < 180.0 else +1
        for policy in POLICIES:
            env.forced_num_obs = 0
            env.reset(seed=900_000 + i, options={"generated": b})
            env.estop_enabled = False
            h0, min_rng, steps = env.asv_h, float("inf"), 0
            while True:
                _, _, term, trunc, info = env.step(action(env, policy, h0, sense))
                steps += 1
                t = env.targets[0]
                min_rng = min(min_rng, math.hypot(t.x - env.asv_x, t.y - env.asv_y))
                if term or trunc or steps * 0.5 > b.tcpa_s + 8.0:
                    break
            rows.append(dict(idx=i, policy=policy, side="port" if b.ct_deg < 180.0 else "starboard",
                             width=b.nominal_width, dcpa=b.dcpa_m, tcpa=b.tcpa_s, k=b.speed_ratio,
                             ct=b.ct_deg, range0=b.spawn_range_m, hit=info["collision_kind"] == "target",
                             label_escapable=getattr(b, "crossing_escapable", None),
                             kind=info["collision_kind"] or "", min_range=min_rng))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "episodes.csv", index=False)
    best = d.groupby("idx").hit.min().rename("best_hit")
    geo = d[d.policy == "follower"].set_index("idx")[["side", "width", "dcpa", "tcpa", "k", "ct", "range0",
                                                      "label_escapable"]].join(best)
    pd.set_option("display.width", 250)
    lines = ["== target-collision rate by scripted response (development crossings, obstacles off, supervisor off)",
             d.pivot_table(index="policy", columns="side", values="hit", aggfunc="mean").round(2).to_string(),
             "", f"best of all responses still hits the target: {geo.best_hit.mean():.2f} "
             f"({int(geo.best_hit.sum())}/{len(geo)})", "",
             "== unavoidable draws (best response still hits)",
             geo[geo.best_hit].round(2).to_string(), "",
             "== drawn geometry, avoidable vs unavoidable (medians)",
             geo.groupby("best_hit")[["dcpa", "tcpa", "k", "range0", "width"]].median().round(2).to_string(), "",
             "== A22 label against this replay's best response (label from the generator's escape check)",
             pd.crosstab(geo.label_escapable.astype(str), geo.best_hit.map({True: "best still hits", False: "some response clears"})).to_string()]
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
