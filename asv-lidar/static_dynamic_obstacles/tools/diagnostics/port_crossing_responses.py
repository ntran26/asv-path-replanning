"""Which scripted response clears a crossing, by side?  (A27)

Run 6 turned starboard first in 8 of 12 port crossings, where A17's compliant
sense is a port turn (pass astern); the reference controller reached the goal in
only 6 of 12 port crossings either way. Before choosing a fix, this separates
the learner from the geometry: every crossing is replayed under fixed responses,
each triggered **when the encounter engages** (as a policy sees it), held until
the target is past and opening, then the path resumed at cruise; supervisor off:

* `stand_on`   -- hold path and speed (Rule 17(a)(i) for a target from port);
* `a17_30/60`  -- a 30 / 60 deg alteration in A17's sense (port for a target from
                  port, starboard from starboard);
* `other_30/60`-- the same alteration the other way (starboard for a target from
                  port: the Rule 17(c) direction);
* `slow`       -- coast to the RPM floor, on the path;
* `a17_60_slow`, `other_60_slow` -- the alteration and the slowdown together.

Crossings: the 20 development crossings plus 60 per side drawn from the
training namespace with the side forced, basin default, obstacles off so the
encounter alone decides.

    python tools/diagnostics/port_crossing_responses.py --processes 3
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
OUT = ROOT / "results" / "port_crossing_responses"
RESPONSES = ("stand_on", "a17_30", "a17_60", "other_30", "other_60", "slow",
             "a17_60_slow", "other_60_slow")
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


def _episode(job):
    from common import follower_action
    k, response = job
    src, seed, built = _W["cases"][k]
    env = _W["env"]
    env.forced_num_obs = 0
    env.reset(seed=900_000 + seed, options={"generated": built})
    side = "port" if built.ct_deg < 180.0 else "starboard"
    a17 = -1.0 if side == "port" else +1.0
    trigger_h, released, steps, min_rng = None, False, 0, float("inf")
    while True:
        engaged = any(c.engaged for c in env.encounter_contexts.values())
        if trigger_h is None and engaged:
            trigger_h = float(env.asv_h)
        # A manoeuvre, not a new heading for ever: hold the response until the
        # target is past and opening (the latch leaves ENGAGED), then resume the
        # path at cruise.  Holding the turn to the end put 40-50 % of turned
        # episodes into a wall and hid whether the target was cleared.
        if trigger_h is not None and not engaged:
            released = True
        rudder, throttle = follower_action(env)
        if trigger_h is not None and not released and response != "stand_on":
            if response.startswith("a17") or response.startswith("other"):
                sense = a17 if response.startswith("a17") else -a17
                alteration = 30.0 if "_30" in response else 60.0
                want = trigger_h + sense * alteration
                rudder = float(np.clip(((want - env.asv_h + 180.0) % 360.0 - 180.0) / 20.0, -1.0, 1.0))
            if response.endswith("slow"):
                throttle = -1.0
        _, _, term, trunc, info = env.step(np.array([rudder, throttle], dtype=np.float32))
        steps += 1
        t = env.targets[0]
        min_rng = min(min_rng, math.hypot(t.x - env.asv_x, t.y - env.asv_y))
        if term or trunc:
            break
    return {"src": src, "case": k, "side": side, "mode": built.geometry_mode,
            "response": response, "engaged": trigger_h is not None,
            "target_hit": info["collision_kind"] == "target",
            "other_hit": info["collided"] and info["collision_kind"] != "target",
            "goal": bool(info["reached_goal"]) and not info["collided"],
            "min_range": min_rng, "dcpa": float(built.dcpa_m), "tcpa": float(built.tcpa_s),
            "ct": float(built.ct_deg), "k": float(built.speed_ratio)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n = len(_cases())
    jobs = [(k, r) for k in range(n) for r in RESPONSES]
    with ProcessPoolExecutor(args.processes, initializer=_init) as pool:
        d = pd.DataFrame(list(pool.map(_episode, jobs, chunksize=4)))
    d.to_csv(OUT / "episodes.csv", index=False)
    pd.set_option("display.width", 220)
    hit = d.pivot_table(index="response", columns="side", values="target_hit", aggfunc="mean").round(2)
    goal = d.pivot_table(index="response", columns="side", values="goal", aggfunc="mean").round(2)
    wall = d.pivot_table(index="response", columns="side", values="other_hit", aggfunc="mean").round(2)
    best = d.groupby(["case", "side"]).goal.max().groupby("side").mean().round(2)
    counts = d[d.response == "stand_on"].side.value_counts()
    text = (f"Scripted crossing responses, triggered at engagement, obstacles off, supervisor off\n"
            f"{dict(counts)} crossings, {len(d)} episodes, {time.time() - started:.0f} s\n\n"
            f"== goal rate\n{goal}\n\n== target collision rate\n{hit}\n\n"
            f"== wall / other collision rate\n{wall}\n\n"
            f"== share of crossings some scripted response solves\n{best}\n")
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
