"""Codex's reference controller on the formulation development set.

`CODEX/` (a separate working copy) built a predictive LOS reference controller
and reports 20/20 on ten hand-built scenes in a 10 m channel. This replays it on
the development set every PPO run is scored on (20 per class, generated
corridors 5-10 m, obstacles, nominal noise, supervisor off, stage-4 propulsion,
seeds 900 000 + i), using CODEX's own environment and sources. Nothing in
`CODEX/` is modified.

CODEX tightened `GOAL_CTE_RADIUS` to 0.60 m; `--goal-cte` restores the 1.60 m the
PPO runs were scored with (default), so the goal rates are comparable.

    python tools/diagnostics/codex_reference_devset.py
"""
import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODEX = ROOT / "CODEX"
OUT = ROOT / "results" / "codex_reference_devset"
EVAL_CLASSES = ("head_on", "crossing", "overtaking", "being_overtaken", "null", "no_target")
_W = {}


def _paths():
    sys.path.insert(0, str(CODEX / "src"))


def development_set(per_class: int):
    """The parent's `train_formulation.development_set`; `scenario.py` is identical in CODEX."""
    import scenario as scn
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    out = []
    for cls in EVAL_CLASSES:
        index = found = 0
        while found < per_class and index < 50 * per_class:
            built = generator.sample(scn.seed_for("development", 10_000 * (EVAL_CLASSES.index(cls) + 1) + index),
                                     encounter_class=cls)
            index += 1
            if built is not None:
                out.append(built)
                found += 1
    return out


def _init(goal_cte: float):
    _paths()
    import constants as cfg
    import curriculum
    cfg.GOAL_CTE_RADIUS = float(goal_cte)
    curriculum.apply_stage(4)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)
    _W["dev"] = development_set(20)


def _episode(i: int):
    from reference_controller import ReferenceController
    env, built = _W["env"], _W["dev"][i]
    started = time.time()
    obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
    controller = ReferenceController()
    steps, fallbacks, relaxed, speeds, min_range = 0, 0, 0, [], float("inf")
    while True:
        action = controller.action(env, obs)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        diag = controller.last_diagnostics
        fallbacks += int(bool(diag.get("fallback", False)))
        relaxed += int(bool(diag.get("rule_relaxed", False)))
        speeds.append(float(info["speed_mps"]))
        for t in env.targets:
            min_range = min(min_range, math.hypot(t.x - env.asv_x, t.y - env.asv_y))
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    ct = float(getattr(built, "ct_deg", 0.0))
    return {"scenario": i, "class": built.encounter_class, "width": float(built.nominal_width),
            "crossing_side": ("port" if ct < 180.0 else "starboard") if built.encounter_class == "crossing" else "",
            "outcome": outcome, "steps": steps, "mean_speed": sum(speeds) / len(speeds),
            "min_target_range": min_range, "fallback_steps": fallbacks, "rule_relaxed_steps": relaxed,
            "wall_s": time.time() - started}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-cte", type=float, default=1.60)
    ap.add_argument("--processes", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with ProcessPoolExecutor(args.processes, initializer=_init, initargs=(args.goal_cte,)) as pool:
        rows = list(pool.map(_episode, range(len(EVAL_CLASSES) * 20)))
    d = pd.DataFrame(rows)
    tag = f"goalcte{args.goal_cte:.2f}"
    d.to_csv(OUT / f"episodes_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    per_class = d.groupby("class").agg(
        goal=("outcome", lambda s: (s == "goal").mean()),
        target=("outcome", lambda s: (s == "collision:target").mean()),
        obstacle=("outcome", lambda s: (s == "collision:obstacle").mean()),
        boundary=("outcome", lambda s: (s == "collision:boundary").mean()),
        timeout=("outcome", lambda s: (s == "timeout").mean()),
        speed=("mean_speed", "mean"), fallback_frac=("fallback_steps", "sum"),
        steps=("steps", "sum"), wall_s=("wall_s", "mean")).round(2)
    per_class["fallback_frac"] = (per_class.fallback_frac / per_class.steps).round(2)
    text = (f"CODEX reference controller, development set, GOAL_CTE_RADIUS {args.goal_cte} m "
            f"({len(d)} episodes, {time.time() - started:.0f} s)\n"
            f"overall goal {(d.outcome == 'goal').mean():.3f}, collision "
            f"{d.outcome.str.startswith('collision').mean():.3f}, timeout {(d.outcome == 'timeout').mean():.3f}\n\n"
            f"{per_class}\n")
    (OUT / f"summary_{tag}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
