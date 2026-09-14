"""Formulation run 3 (A15-A17 in) against run 2 (F49-F51 in).

Both runs: PPO, seed 0, 2 M steps, 20 development scenarios per class per
evaluation.  Run 3 predates A18-A20, so its head-on results do not test them.

    python tools/diagnostics/run3_vs_run2.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RUN = {"run2": RUNS / "ppo_formulation_seed0_v2", "run3": RUNS / "ppo_formulation_seed0_v3"}
LOG = {"run2": RUNS / "ppo_formulation_seed0_v2.log", "run3": RUNS / "ppo_formulation_seed0_v3.log"}
CLASSES = ("head_on", "crossing", "overtaking", "being_overtaken", "null", "no_target")
LATE = 1_600_000

sys.path.insert(0, str(ROOT / "src"))


def eval_table(run):
    import json
    rows = json.loads((RUN[run] / "eval_summary.json").read_text())
    out = []
    for r in rows:
        out.append({"t": r["timesteps"], "goal": r["goal_rate"], "coll": r["collision_rate"],
                    **{c: r.get(f"{c}/collision") for c in CLASSES}})
    return pd.DataFrame(out).set_index("t")


def episodes(run):
    e = pd.read_csv(RUN[run] / "eval_episodes.csv", keep_default_na=False, na_values=[""])
    e["coll"] = e.outcome.str.startswith("collision")
    return e


def geometry_for_run2(e):
    """Run 2's CSV predates `dcpa_m` / `ct_deg`; regenerate the development set
    (row order matches, `run2_geometry_join.py`).

    **Being-overtaken geometry is dropped.**  A15 changed that class's DCPA draw
    after run 2, so today's generator does not reproduce run 2's being-overtaken
    scenarios; `PROJECT_STATE.md` F52 has the valid run 2 split (0.69 / 0.39 /
    0.50).  Every other class draws exactly as it did.
    """
    import train_formulation as tf
    scen = tf.development_set(20)
    e = e.copy()
    e["idx"] = e.groupby("timesteps").cumcount()
    geo = pd.DataFrame([{"idx": i,
                         "dcpa_m": np.nan if b.encounter_class == "being_overtaken" else b.dcpa_m,
                         "ct_deg": b.ct_deg, "dcpa_below_floor": np.nan} for i, b in enumerate(scen)])
    return e.merge(geo, on="idx")


def monitor(run):
    m = pd.read_csv(RUN[run] / "monitor.csv", skiprows=1)
    m["T"] = m.l.cumsum()
    return m[m["T"] > 1_500_000]


def ppo_trend(run):
    text = LOG[run].read_text(encoding="utf-8", errors="replace")
    grab = lambda key: [float(x) for x in re.findall(rf"\|\s+{key}\s+\|\s+([-0-9.e]+)", text)]
    kl, clip, fps = grab("approx_kl"), grab("clip_fraction"), grab("fps")
    tail = lambda xs: round(float(np.mean(xs[-10:])), 3) if xs else float("nan")
    return {"approx_kl_last10": tail(kl), "clip_fraction_last10": tail(clip), "fps_last": fps[-1] if fps else None,
            "done": "done in" in text}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for run in RUN:
        print(f"\n== {run}: evaluation (goal, collision, per-class collision)")
        print(eval_table(run).round(2))

    late = {}
    for run in RUN:
        e = episodes(run)
        if "ct_deg" not in e.columns:
            e = geometry_for_run2(e)
        late[run] = e[e.timesteps >= LATE].copy()

    print(f"\n== late evaluations (>= {LATE:,}): per-class collision rate and mean speed")
    rows = []
    for run, e in late.items():
        g = e.groupby("class").agg(coll=("coll", "mean"), n=("coll", "size"), speed=("mean_speed", "mean"))
        g.columns = [f"{run}_{c}" for c in g.columns]
        rows.append(g)
    print(pd.concat(rows, axis=1).round(2))

    print("\n== crossing by target side (ct < 180: from port; > 180: from starboard), late evaluations")
    for run, e in late.items():
        c = e[e["class"] == "crossing"].copy()
        c["side"] = np.where(c.ct_deg < 180.0, "from port", "from starboard")
        c["dbin"] = pd.cut(c.dcpa_m, [-0.01, 0.7, 1.4, 2.6])
        print(f"-- {run}")
        print(c.pivot_table(index="side", columns="dbin", values="coll", aggfunc=["mean", "size"], observed=True).round(2))

    print("\n== being overtaken, late evaluations")
    for run, e in late.items():
        b = e[e["class"] == "being_overtaken"].copy()
        b["dbin"] = pd.cut(b.dcpa_m, [-0.01, 0.7, 1.0, 1.4, 2.1])
        print(f"-- {run}: collision by drawn DCPA")
        print(b.groupby("dbin", observed=True).agg(coll=("coll", "mean"), n=("coll", "size"),
                                                   speed=("mean_speed", "mean"), hold_frames=("frames_v_hold", "mean")).round(2))
        print(pd.crosstab(b["class"], b.outcome))

    print("\n== training episodes after 1.5 M steps: goal and collision per class")
    rows = []
    for run in RUN:
        m = monitor(run)
        g = m.groupby("scenario_class").agg(goal=("reached_goal", "mean"), coll=("collided", "mean"), n=("r", "size"))
        g.columns = [f"{run}_{c}" for c in g.columns]
        rows.append(g)
    print(pd.concat(rows, axis=1).round(2))

    print("\n== PPO diagnostics")
    for run in RUN:
        print(run, ppo_trend(run))


if __name__ == "__main__":
    main()
