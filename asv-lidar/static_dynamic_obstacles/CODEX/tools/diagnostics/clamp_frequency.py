"""How often does `targets.clamp_to_corridor` move a confined target before CPA?

The clamp teleports a target that breaches the channel to the nearest
centreline station and turns it onto the channel tangent.  In a being-overtaken
Tier 0 trace that put a 1.30 m DCPA overtaker straight onto the own ship's
track, 0.19 m off, and it hit the stern.  This counts, per class, over the
development set with a path follower and obstacles off:

* episodes where the clamp fires at all, and before the drawn TCPA;
* the lateral jump of the first clamp;
* the target-collision rate with and without a pre-CPA clamp.

    python tools/diagnostics/clamp_frequency.py
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
import targets as tgt  # noqa: E402
import train_formulation as tf  # noqa: E402
import env as envmod  # noqa: E402

OUT = ROOT / "results" / "clamp_frequency"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    events = []
    original = tgt.clamp_to_corridor

    def spy(target, corridor, polygon=None):
        before = (target.x, target.y, target.heading)
        original(target, corridor, polygon)
        after = (target.x, target.y, target.heading)
        if before != after:
            events.append((math.hypot(after[0] - before[0], after[1] - before[1]),
                           abs(((after[2] - before[2]) + 180.0) % 360.0 - 180.0)))

    envmod.tgtmod.clamp_to_corridor = spy
    env = envmod.ASVLidarEnv(render_mode=None)
    rows = []
    scen = common.development_set(20, classes=("head_on", "overtaking", "being_overtaken", "null"))
    for i, b in enumerate(scen):
        env.forced_num_obs = 0
        env.reset(seed=900_000 + i, options={"generated": b})
        events.clear()
        first_t, first_jump, first_turn, step = None, 0.0, 0.0, 0
        while True:
            _, _, term, trunc, info = env.step(common.follower_action(env))
            step += 1
            if events and first_t is None:
                first_t, (first_jump, first_turn) = step * 0.5, events[0]
            if term or trunc:
                break
        tcpa = float(b.tcpa_s) if b.tcpa_s > 0 else float("inf")
        rows.append(dict(cls=b.encounter_class, width=b.nominal_width, dcpa=b.dcpa_m, tcpa=tcpa,
                         clamped=first_t is not None,
                         clamped_before_cpa=first_t is not None and first_t <= tcpa,
                         first_clamp_s=first_t, jump_m=first_jump, turn_deg=first_turn,
                         target_collision=info["collision_kind"] == "target"))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "clamp_frequency.csv", index=False)
    pd.set_option("display.width", 250)
    s = d.groupby("cls").agg(n=("dcpa", "size"), clamped=("clamped", "mean"),
                             clamped_before_cpa=("clamped_before_cpa", "mean"),
                             median_first_clamp_s=("first_clamp_s", "median"),
                             median_jump_m=("jump_m", lambda x: float(np.median(x[x > 0])) if (x > 0).any() else 0.0),
                             median_turn_deg=("turn_deg", lambda x: float(np.median(x[x > 0])) if (x > 0).any() else 0.0),
                             target_collision=("target_collision", "mean")).round(2)
    c = d.pivot_table(index="cls", columns="clamped_before_cpa", values="target_collision",
                      aggfunc=["mean", "size"]).round(2)
    text = "== clamp frequency, path follower, obstacles off\n" + s.to_string() + \
           "\n\n== target collision by pre-CPA clamp\n" + c.to_string()
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
