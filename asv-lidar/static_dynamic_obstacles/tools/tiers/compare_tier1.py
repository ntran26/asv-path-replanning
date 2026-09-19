"""Tier 1 results side by side: learned policies against the classical comparators.

Every Tier 1 replay runs the same scenarios with the same seeds, so the rows
pair on `(set, scenario)` and the success comparison is an exact McNemar test
on the discordant pairs (`src/compare.py`), against the first tag given.

    python tools/tiers/compare_tier1.py --tags run7_supervisor_off los_dwa_supervisor_off \
        encounter_vo_supervisor_off --out results/classical_comparison/supervisor_off.txt

`--reference` adds the CODEX reference controller's development-set replay
(`results/basin_devset_baselines`, supervisor off; goal and collisions only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "tiers"
sys.path.insert(0, str(ROOT / "src"))
from compare import mcnemar_exact  # noqa: E402

CLASS_ORDER = ["head_on", "crossing", "overtaking", "being_overtaken", "null", "no_target"]


def load(tag: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / f"tier1_{tag}" / "episodes.csv", keep_default_na=False, na_values=[""])
    d["method"] = tag
    d["goal"] = d.outcome == "goal"
    for kind in ("target", "obstacle", "boundary"):
        d[kind] = d.outcome == f"collision:{kind}"
    return d


def load_reference() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results" / "basin_devset_baselines" / "episodes.csv",
                    keep_default_na=False, na_values=[""])      # the class "null" is not NaN
    d = d[d.policy == "reference"].copy()
    d["method"], d["set"] = "reference (CODEX)", "development"
    d["goal"] = d.outcome == "goal"
    for kind in ("target", "obstacle", "boundary"):
        d[kind] = d.outcome == f"collision:{kind}"
    return d


def per_class(d: pd.DataFrame) -> pd.DataFrame:
    cols = {"n": ("goal", "size"), "goal": ("goal", "mean"), "target": ("target", "mean"),
            "obstacle": ("obstacle", "mean"), "boundary": ("boundary", "mean")}
    for name, col, fn in (("speed", "mean_speed", "mean"), ("colregs", "colregs_integral", "mean"),
                          ("estops", "estops", "mean"), ("rms_cte", "rms_cte", "mean")):
        if col in d:
            cols[name] = (col, fn)
    return d.groupby(["method", "class"]).agg(**cols).round(2)


def paired(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    j = base.merge(other, on=["set", "scenario"], suffixes=("_a", "_b"))
    return mcnemar_exact(j.goal_a.to_numpy(), j.goal_b.to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="tier1_<tag> directories; the first is the baseline")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = [load(t) for t in args.tags]
    if args.reference:
        runs.append(load_reference())
    allrows = pd.concat(runs, ignore_index=True)

    lines = [f"Tier 1: {', '.join(args.tags)}{' + reference' if args.reference else ''}", ""]
    for set_name in ("development", "head_on_width"):
        s = allrows[allrows.set == set_name]
        if s.empty:
            continue
        overall = s.groupby("method", sort=False).agg(
            n=("goal", "size"), goal=("goal", "mean"), target=("target", "mean"),
            obstacle=("obstacle", "mean"), boundary=("boundary", "mean")).round(3)
        base = runs[0][runs[0].set == set_name]
        tests = {}
        for r in runs[1:]:
            rs = r[r.set == set_name]
            if len(rs):
                m = paired(base, rs)
                tests[r.method.iloc[0]] = (f"{m['only_a_success']}/{m['only_b_success']}", round(m["p_value"], 4))
        overall["discordant (base only / this only)"] = [tests.get(m, ("—", None))[0] for m in overall.index]
        overall["McNemar p"] = [tests.get(m, ("—", None))[1] for m in overall.index]
        lines += [f"== {set_name} set, overall (base: {args.tags[0]})", "", overall.to_string(), ""]

        pc = per_class(s).reset_index()
        order = [c for c in CLASS_ORDER if c in set(pc["class"])]
        goal = pc.pivot(index="class", columns="method", values="goal").reindex(order)
        goal = goal[[m for m in overall.index]]
        lines += [f"-- {set_name}, goal rate by class", "", goal.to_string(), ""]
        lines += [f"-- {set_name}, detail by class", "", pc.set_index(["method", "class"]).to_string(), ""]
        if set_name == "development" and "crossing_side" in s:
            cr = s[(s["class"] == "crossing") & s.crossing_side.astype(str).isin(["port", "starboard"])]
            side = cr.pivot_table(index="crossing_side", columns="method", values="goal", aggfunc="mean").round(2)
            lines += ["-- crossing goal rate by side", "", side[[m for m in overall.index if m in side]].to_string(), ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
