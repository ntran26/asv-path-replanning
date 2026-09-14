"""F55: does A17's side-dependent crossing sense get latched on head-on encounters?

Replays the C14 head-on set with a non-avoiding path follower and the run 2
model, and logs the latched class, crossing side and compliant turn sense on
engaged pre-CPA frames.  Found: the model latched a port sense in 67 % of
episodes (43 % of frames), because its own starboard alteration re-classifies
the head-on as a crossing from port.

    python tools/diagnostics/c14_turn_sense.py
"""
from collections import Counter

import numpy as np
import pandas as pd

from c14_common import OUT, WIDTH_BINS, head_on_scenarios, load_run2_model
from env import ASVLidarEnv


def follower(env):
    rudder = float(np.clip(env.lookahead_course_error / 25.0 + 0.4 * env.cross_track_error, -1.0, 1.0))
    return np.array([rudder, 0.0], dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scen = head_on_scenarios()
    model = load_run2_model()
    env = ASVLidarEnv(render_mode=None)
    rows = []
    for policy in ("follower", "model"):
        for i, b in enumerate(scen):
            env.forced_num_obs = 0
            obs, _ = env.reset(seed=700_000 + i, options={"generated": b})
            eng, first, n_eng, port_sense = Counter(), None, 0, False
            while True:
                action = follower(env) if policy == "follower" else model.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, info = env.step(action)
                for ctx in env.encounter_contexts.values():
                    if ctx.engaged and ctx.tcpa > 0.0:
                        n_eng += 1
                        key = (str(ctx.cls), str(ctx.crossing_side), int(ctx.compliant_turn_sense))
                        eng[key] += 1
                        first = first or key
                        port_sense |= (ctx.compliant_turn_sense < 0 and str(ctx.cls) == "crossing")
                if term or trunc:
                    break
            n = max(n_eng, 1)
            rows.append(dict(policy=policy, idx=i, width=b.nominal_width, bend=b.bend_deg,
                             coll=info["collision_kind"] == "target", engaged_frames=n_eng,
                             first_cls=first[0] if first else "never",
                             p_head_on=sum(v for k, v in eng.items() if k[0] == "head_on") / n,
                             p_cross_port_sense=sum(v for k, v in eng.items()
                                                    if k[0] == "crossing" and k[2] < 0) / n,
                             ever_port_sense_crossing=port_sense))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "c14_sense.csv", index=False)
    d["wbin"] = pd.cut(d.width, *WIDTH_BINS)
    pd.set_option("display.width", 250)
    print(d.groupby(["policy", "wbin"], observed=True).agg(
        n=("idx", "size"), coll=("coll", "mean"),
        first_head_on=("first_cls", lambda s: (s == "head_on").mean()),
        p_head_on=("p_head_on", "mean"), p_cross_port_sense=("p_cross_port_sense", "mean"),
        ever_port_sense=("ever_port_sense_crossing", "mean")).round(2))


if __name__ == "__main__":
    main()
