"""Tier 1 — what does a trained policy do under today's code?  (~15 min)

Replays a saved model, deterministic, on two fixed sets:

* the formulation development set (20 per class, generated obstacles), and
* the C14 head-on width set (20 each at 5, 6, 7, 8, 10 m, obstacles off).

It measures the *mechanism* of a code change on a policy that was trained
before it, which is exactly what a debugging question usually needs.  It
cannot show what retraining would learn; that is Tier 2.

Reported: per-class outcomes against the model's own last in-training
evaluation; crossing by side; being overtaken above and below the A15 floor;
head-on by width; supervisor stops, their reasons, and how often a target
collision follows one (the F58 5 m stop check); port-sense latching.

    python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v3/final_model.zip --tag run3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, development_set, head_on_width_set, run_pool, width_bin


def _own_eval(model_path: Path) -> pd.DataFrame:
    csv = model_path.parent / "eval_episodes.csv"
    if not csv.exists():
        return pd.DataFrame()
    e = pd.read_csv(csv, keep_default_na=False, na_values=[""])
    e = e[e.timesteps == e.timesteps.max()]
    e["collided"] = e.outcome.str.startswith("collision")
    return e.groupby("class").agg(own_goal=("outcome", lambda s: (s == "goal").mean()),
                                  own_collision=("collided", "mean")).round(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "runs" / "ppo_formulation_seed0_v3" / "final_model.zip")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--tag", default="run3")
    ap.add_argument("--processes", type=int, default=None, help="worker processes (default: cores - 2)")
    ap.add_argument("--stop-test-fit", choices=("on", "off"), default=None,
                    help="override STOP_TEST_USES_HULL_FIT in the workers (F66 A/B)")
    ap.add_argument("--supervisor", choices=("on", "off"), default=None,
                    help="F68: run the replay with the stop latch on or off")
    args = ap.parse_args()
    model = args.model if args.model.is_absolute() else ROOT / args.model

    out = RESULTS / f"tier1_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dev = development_set(args.per_class)
    ho = head_on_width_set()
    jobs = [(b, 900_000 + i, "model", None, {"set": "development", "scenario": i}) for i, b in enumerate(dev)]
    jobs += [(b, 700_000 + i, "model", 0, {"set": "head_on_width", "scenario": i}) for i, b in enumerate(ho)]
    overrides = {}
    if args.stop_test_fit is not None:
        overrides["STOP_TEST_USES_HULL_FIT"] = args.stop_test_fit == "on"
    if args.supervisor is not None:
        overrides["EMERGENCY_STOP_ENABLED"] = args.supervisor == "on"
    overrides = overrides or None
    d = pd.DataFrame(run_pool(jobs, model_path=model, processes=args.processes, overrides=overrides))
    d.to_csv(out / "episodes.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    dv = d[d.set == "development"]
    per_class = dv.groupby("class").agg(
        n=("steps", "size"), goal=("outcome", lambda s: (s == "goal").mean()),
        collision=("collided", "mean"), target=("collided_target", "mean"),
        speed=("mean_speed", "mean"), colregs=("colregs_integral", "mean"),
        estops=("estops", "mean"), intervention=("estops", lambda x: float((x > 0).mean())),
        port_sense=("ever_port_sense_non_overtaking", "mean")).round(2)
    per_class = per_class.join(_own_eval(model))

    cr = dv[dv["class"] == "crossing"].copy()
    cr["dcpa_bin"] = pd.cut(cr.dcpa_m, [-0.01, 0.7, 1.4, 2.6])
    crossing = cr.pivot_table(index="crossing_side", columns="dcpa_bin", values="collided",
                              aggfunc=["mean", "size"], observed=True).round(2)
    if "crossing_escapable" in cr and cr.crossing_escapable.notna().any():
        crossing = crossing.to_string() + "\n\n-- by A22 label (escapable by a lawful response)\n" + \
            cr.groupby(cr.crossing_escapable.astype(str)).agg(
                n=("steps", "size"), collision=("collided", "mean"),
                target=("collided_target", "mean")).round(2).to_string()

    bo = dv[dv["class"] == "being_overtaken"].copy()
    bo["floor"] = np.where(bo.dcpa_below_floor.astype(str) == "True", "below floor", "at/above floor")
    overtaken = bo.groupby("floor").agg(n=("steps", "size"), collision=("collided", "mean"),
                                        target=("collided_target", "mean"), speed=("mean_speed", "mean"),
                                        v_hold=("frames_v_hold", "mean")).round(2)

    hw = d[d.set == "head_on_width"].copy()
    hw["width_bin"] = hw.width.map(width_bin)
    head_on = hw.groupby("width_bin").agg(
        n=("steps", "size"), target_collision=("collided_target", "mean"),
        estops=("estops", "mean"), stopped_then_hit=("estop_then_target_collision", "sum"),
        min_speed_mean=("mean_speed", "mean"), port_sense=("ever_port_sense_non_overtaking", "mean"))
    head_on = head_on.reindex(["5", "6", "7", "8", "10"]).round(2)

    stopped = d[d.estops > 0]
    reasons = (stopped.estop_reasons.str.split(" | ", regex=False).explode()
               .str.extract(r"^(8\(e\) in extremis: \w+)")[0].value_counts())

    lines = [f"Tier 1 ({args.tag}) — model {model.relative_to(ROOT) if model.is_relative_to(ROOT) else model}",
             f"{len(d)} episodes, {time.time() - started:.0f} s", "",
             "== development set, per class (own_* = the model's last in-training evaluation, before today's code)",
             per_class.to_string(), "",
             "== crossing collision by side x drawn DCPA", crossing if isinstance(crossing, str) else crossing.to_string(), "",
             "== being overtaken, A15 floor", overtaken.to_string(), "",
             "== head-on width set (obstacles off)", head_on.to_string(), "",
             f"== supervisor stops: {int(d.estops.sum())} in {len(stopped)} episodes; "
             f"followed by a target collision in {int(stopped.estop_then_target_collision.sum())}",
             reasons.to_string() if len(reasons) else "(none)"]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
