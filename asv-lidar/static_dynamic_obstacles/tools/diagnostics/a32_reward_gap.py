"""A32: does the reward pay for turning the compliant way in a crossing?

Six policies across four runs each pick one opening direction and apply it to
both sides (F92). Observability and exposure are ruled out (F91, F92). This
asks the reward directly: from the **same state** -- the step the encounter
engages -- the episode is branched into scripted responses, and the return from
that step on is split by term. Same seed, same follower up to the branch, so the
branches differ only in what was done at engagement:

* `compliant_30/60` -- a 30 / 60 deg alteration in the latched compliant sense
                       (the context's `compliant_turn_sense`, which the reward reads);
* `wrong_30/60`     -- the same alteration the other way;
* `stand_on`        -- hold the path at cruise.

Each alteration is held until the latch leaves ENGAGED, then the path resumes
at cruise (the release rule of `port_crossing_responses.py` v2). Returns are
reported undiscounted and discounted at PPO's gamma from the branch step: the
discounted gap is what the advantage of the opening decision sees.

Crossings: the 20 development crossings plus 60 per side from the training
namespace with the side forced; basin default, obstacles off, supervisor off.

    python tools/diagnostics/a32_reward_gap.py --processes 10
"""
import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
OUT = ROOT / "results" / "a32_reward_gap"
RESPONSES = ("stand_on", "compliant_30", "wrong_30", "compliant_60", "wrong_60")
TERMS = ("pf", "prog", "exist", "smooth", "obs", "bnd", "dom", "col")
_W = {}


def _cases():
    import scenario as scn
    import train_formulation as tf
    cases = [("dev", i, b) for i, b in enumerate(tf.development_set(20))
             if b.encounter_class == "crossing"]
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="training")
    for side in ("port", "starboard"):
        got, j = 0, 0
        while got < 60 and j < 400:
            b = generator.sample(scn.seed_for("training", 70_000 + (0 if side == "port" else 5_000) + j),
                                 encounter_class="crossing", flags={"side": side})
            j += 1
            if b is not None:
                cases.append(("extra", 70_000 + j, b))
                got += 1
    return cases


def _init():
    import curriculum
    import train_formulation as tf
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)
    _W["cases"] = _cases()
    _W["gamma"] = float(tf.PPO_HYPERPARAMS["gamma"])


def _episode(job):
    from common import follower_action
    k, response = job
    src, seed, built = _W["cases"][k]
    env, gamma = _W["env"], _W["gamma"]
    env.forced_num_obs = 0
    env.reset(seed=900_000 + seed, options={"generated": built})
    side = "port" if built.ct_deg < 180.0 else "starboard"
    trigger_h, trigger_step, sense, released = None, None, 0, False
    step, min_rng, vport_frames = 0, float("inf"), 0
    undisc = {t: 0.0 for t in TERMS}
    disc = {t: 0.0 for t in TERMS}
    undisc_terminal = disc_terminal = pre_return = 0.0
    first_heading_dev = None
    while True:
        engaged = [c for c in env.encounter_contexts.values() if c.engaged]
        if trigger_h is None and engaged:
            trigger_h, trigger_step = float(env.asv_h), step
            sense = int(engaged[0].compliant_turn_sense)
        if trigger_h is not None and not engaged:
            released = True
        rudder, throttle = follower_action(env)
        if trigger_h is not None and not released and response != "stand_on" and sense != 0:
            s = sense if response.startswith("compliant") else -sense
            want = trigger_h + s * (30.0 if response.endswith("30") else 60.0)
            rudder = float(np.clip(((want - env.asv_h + 180.0) % 360.0 - 180.0) / 20.0, -1.0, 1.0))
        _, reward, term, trunc, info = env.step(np.array([rudder, throttle], dtype=np.float32))
        b = env.last_reward
        if trigger_step is None:
            pre_return += float(reward)
        else:
            w = gamma ** (step - trigger_step)
            for t in TERMS:
                v = float(b.weighted.get(t, 0.0))
                undisc[t] += v
                disc[t] += w * v
            tv = float(b.terminal) + float(b.intervention)
            undisc_terminal += tv
            disc_terminal += w * tv
            vport_frames += int(float(b.colregs.get("v_port", 0.0)) > 0.0)
            if first_heading_dev is None and step - trigger_step == 6:     # 3 s after the branch
                first_heading_dev = (env.asv_h - trigger_h + 180.0) % 360.0 - 180.0
        step += 1
        t0 = env.targets[0]
        min_rng = min(min_rng, math.hypot(t0.x - env.asv_x, t0.y - env.asv_y))
        if term or trunc:
            break
    row = {"src": src, "case": k, "side": side, "response": response,
           "engaged": trigger_step is not None, "sense": sense,
           "trigger_step": trigger_step, "steps_after": (step - trigger_step) if trigger_step is not None else 0,
           "outcome": ("target" if info["collision_kind"] == "target" else
                       "other" if info["collided"] else
                       "goal" if info["reached_goal"] else "timeout"),
           "min_range": min_rng, "vport_frames": vport_frames,
           "heading_dev_3s": first_heading_dev, "pre_return": pre_return,
           "dcpa": float(built.dcpa_m), "tcpa": float(built.tcpa_s), "ct": float(built.ct_deg)}
    for t in TERMS:
        row[f"u_{t}"] = undisc[t]
        row[f"d_{t}"] = disc[t]
    row["u_terminal"], row["d_terminal"] = undisc_terminal, disc_terminal
    row["u_total"] = sum(undisc.values()) + undisc_terminal
    row["d_total"] = sum(disc.values()) + disc_terminal
    return row


def _md(frame: pd.DataFrame) -> str:
    return frame.to_string()


def analyse(d: pd.DataFrame) -> str:
    lines = []
    d = d[d.engaged & (d.sense != 0)]
    n_cases = d.case.nunique()
    lines.append(f"{n_cases} crossings engaged with a compliant sense "
                 f"({dict(d[d.response == 'stand_on'].side.value_counts())})")
    bad = d[(d.side == "port") & (d.sense != -1) | (d.side == "starboard") & (d.sense != +1)]
    lines.append(f"sense disagreeing with the drawn side (A17): {bad.case.nunique()} crossings\n")

    lines.append("== outcome by response and side")
    lines.append(_md(pd.crosstab([d.response, d.side], d.outcome, normalize="index").round(2)))

    for size in ("30", "60"):
        c = d[d.response == f"compliant_{size}"].set_index("case")
        w = d[d.response == f"wrong_{size}"].set_index("case")
        both = c.index.intersection(w.index)
        c, w = c.loc[both], w.loc[both]
        pair = pd.DataFrame({"side": c.side, "out_c": c.outcome, "out_w": w.outcome})
        for pre in ("u", "d"):
            for t in TERMS + ("terminal", "total"):
                pair[f"{pre}_{t}"] = c[f"{pre}_{t}"] - w[f"{pre}_{t}"]
        clean = pair[(pair.out_c == "goal") & (pair.out_w == "goal")]
        lines.append(f"\n==== {size} deg: compliant minus wrong, paired from the same engagement "
                     f"(positive = the reward prefers compliance)")
        lines.append(f"pairs {len(pair)}; both reach the goal {len(clean)}; "
                     f"outcome differs {int((pair.out_c != pair.out_w).sum())}")
        cols = ["d_col", "d_pf", "d_prog", "d_bnd", "d_obs", "d_smooth", "d_exist", "d_terminal", "d_total"]
        lines.append("\n-- discounted from the branch (what the opening decision's advantage sees), all pairs, mean")
        lines.append(_md(pair.groupby("side")[cols].mean().round(2)))
        lines.append("\n-- discounted, pairs where both reach the goal (the pure preference), mean")
        if len(clean):
            lines.append(_md(clean.groupby("side")[cols].mean().round(2)))
            lines.append("\n-- share of clean pairs where the reward prefers compliance (d_total > 0)")
            lines.append(_md(clean.groupby("side").d_total.apply(lambda s: (s > 0).mean()).round(2)))
        ucols = ["u_col", "u_pf", "u_prog", "u_terminal", "u_total"]
        lines.append("\n-- undiscounted to episode end, all pairs, mean")
        lines.append(_md(pair.groupby("side")[ucols].mean().round(2)))
        # Signal against noise: the gap against the spread of the discounted
        # return across crossings for one fixed response.
        spread = d[d.response == f"compliant_{size}"].groupby("side").d_total.std()
        gap = pair.groupby("side").d_total.mean()
        lines.append("\n-- discounted gap vs the across-crossing std of the discounted return")
        lines.append(_md(pd.DataFrame({"gap": gap, "std": spread, "gap/std": gap / spread}).round(2)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=10)
    ap.add_argument("--reanalyse", action="store_true", help="re-read episodes.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.reanalyse:
        d = pd.read_csv(OUT / "episodes.csv")
    else:
        n = len(_cases())
        jobs = [(k, r) for k in range(n) for r in RESPONSES]
        with ProcessPoolExecutor(args.processes, initializer=_init) as pool:
            d = pd.DataFrame(list(pool.map(_episode, jobs, chunksize=4)))
        d.to_csv(OUT / "episodes.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    text = (f"A32 reward gap: scripted responses branched at engagement, obstacles off, "
            f"supervisor off ({time.time() - started:.0f} s)\n\n" + analyse(d) + "\n")
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
