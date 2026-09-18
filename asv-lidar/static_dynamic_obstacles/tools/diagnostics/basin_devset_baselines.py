"""Classical baselines on the basin-default development set (F74).

06 and your brief: layouts must be feasible, and still hard enough that a
classical method fails on some of them. This replays the formulation
development set (now 75 % basin for the width-governed classes, basin for the
rest) under

* `follower` -- hold the path at cruise, avoid nothing: what the scene costs a
  policy that does not react;
* `reference` -- the CODEX predictive LOS controller (`src/reference_controller.py`),
  perception-only: the classical comparator;

with the supervisor off and nominal noise, seeds as Tier 1. Every layout
passed the A* feasibility filter at reset, so a reference failure is a
feasible case the classical method does not solve.

    python tools/diagnostics/basin_devset_baselines.py
"""
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
OUT = ROOT / "results" / "basin_devset_baselines"
_W = {}


def _init():
    import curriculum
    import train_formulation as tf
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)
    _W["dev"] = tf.development_set(20)


def _episode(job):
    import feasibility as feas
    from common import follower_action
    from reference_controller import ReferenceController
    i, policy = job
    env, built = _W["env"], _W["dev"][i]
    started = time.time()
    obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
    feasible = feas.layout_feasible((env.start_x, env.start_y), (env.goal_x, env.goal_y),
                                    env.boundary_polygon, env.obstacles)
    controller = ReferenceController() if policy == "reference" else None
    steps, min_range = 0, float("inf")
    while True:
        action = controller.action(env, obs) if controller else follower_action(env)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        for t in env.targets:
            min_range = min(min_range, math.hypot(t.x - env.asv_x, t.y - env.asv_y))
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return {"scenario": i, "policy": policy, "class": built.encounter_class,
            "mode": built.geometry_mode, "slant": float(built.slant_realised_deg),
            "w_eff_at_cpa": float(built.w_eff_at_cpa), "n_obstacles": len(env.obstacles),
            "layout_redraws": int(getattr(env, "layout_redraws", 0)),
            "layout_thinned": int(getattr(env, "layout_thinned", 0)),
            "feasible": bool(feasible), "outcome": outcome, "steps": steps,
            "min_target_range": min_range, "wall_s": time.time() - started}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = [(i, p) for p in ("follower", "reference") for i in range(120)]
    with ProcessPoolExecutor(10, initializer=_init) as pool:
        d = pd.DataFrame(list(pool.map(_episode, jobs)))
    d.to_csv(OUT / "episodes.csv", index=False)
    pd.set_option("display.width", 220)
    g = lambda s: (s == "goal").mean()
    per = d.pivot_table(index=["class", "mode"], columns="policy", values="outcome",
                        aggfunc=g).round(2)
    per["n"] = d[d.policy == "reference"].groupby(["class", "mode"]).size()
    overall = d.groupby("policy").outcome.agg(g).round(3)
    kinds = d.groupby("policy").outcome.value_counts(normalize=True).round(3)
    ref = d[d.policy == "reference"]
    text = (f"Development set, basin default (F74): {len(ref)} scenarios, "
            f"{time.time() - started:.0f} s\n"
            f"geometry: {ref['mode'].value_counts().to_dict()}; all layouts A*-feasible: "
            f"{bool(ref.feasible.all())}; redrawn {int((ref.layout_redraws > 0).sum())}, "
            f"thinned {int((ref.layout_thinned > 0).sum())}\n\n"
            f"== goal rate\n{overall}\n\n== outcomes\n{kinds}\n\n== goal by class and mode\n{per}\n")
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
