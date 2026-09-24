"""The frozen suite on one policy — the evaluation the paper reports (04a §9).

Unlike Tier 0-3, which replay the *development* namespace and exist to debug the
formulation, this runs the **frozen namespace**: cases no model has been
selected on.

* **Tier B is the default frozen suite** (your call, 2026-09-23): the stratified
  holdout, 39 cells x 20 = 780 episodes in suite 3.1. The headline, tables
  R1-R2. Every evaluation runs this and only this unless asked otherwise.
* **Tier A is the extended set** — the named cases (38 defined, 35 realised; the
  three that cannot be built are reported, not hidden), table R8. It runs
  **only when explicitly requested**, with `--tiers a` or `--tiers a,b`.

Both supervisor modes: compliance is reported with it **off**, and with it
**on** the intervention rate is its own column (claim C-7).

    python tools/tiers/frozen_suite.py --model runs/ppo_formulation_seed0_bl2/best_model.zip \
        --tag ppos0_bl2 --supervisor both

Writes `results/frozen_suite/<tag>/`: `episodes.csv`, `summary.txt`, and
`manifest.json` — the suite manifest digest the run was scored against, so a
table can be traced to the exact cases that produced it.

**Around the Clock (R8's second half) is not here**: its 24 constellations are
defined in `suite.around_the_clock()` but no scenario builder realises them yet
(C3-C6). It has to be added before R8 is complete.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import suite  # noqa: E402
from common import run_pool  # noqa: E402

# `common.RESULTS` is `results/tiers`, which is where the *tier* replays belong;
# the frozen suite is the paper's evaluation, so it sits at `results/frozen_suite`
# -- the path the campaign script and TRAINING_GUIDE.md name.
OUT = ROOT / "results" / "frozen_suite"
TIER_A_SEED = 300_000
TIER_B_SEED = 400_000


def _md(frame: pd.DataFrame) -> str:
    return frame.to_string()


def _summarise(d: pd.DataFrame, group) -> pd.DataFrame:
    g = d.groupby(group)
    out = pd.DataFrame({
        "n": g.size(),
        "success": g.apply(lambda x: (x.outcome == "goal").mean(), include_groups=False),
        "coll_target": g.collided_target.mean(),
        "coll_other": g.apply(lambda x: (x.collided & ~x.collided_target).mean(), include_groups=False),
        "rms_cte": g.rms_cte.mean(),
        "min_range": g.min_target_range.median(),
        "colregs": g.colregs_integral.mean(),
        "estops": g.estops.mean(),
    })
    return out.round(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--supervisor", choices=("off", "on", "both"), default="both")
    ap.add_argument("--tiers", default="b",
                    help="b (default, the frozen suite), a (the extended named cases), or a,b")
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()

    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model = args.model if args.model.is_absolute() else ROOT / args.model
    tiers = {t.strip() for t in args.tiers.split(",")}
    started = time.time()

    scenarios, jobs, shortfall_b = [], [], []
    if "a" in tiers:
        tier_a = suite.build_tier_a()
        missing = suite.tier_a_shortfall(tier_a)
        scenarios += tier_a
        jobs += [(b, TIER_A_SEED + i, "model", None,
                  {"set": "tier_a", "case_id": b.case_id, "scenario": i})
                 for i, b in enumerate(tier_a)]
    else:
        tier_a, missing = [], []
    if "b" in tiers:
        tier_b, shortfall_b = suite.build_tier_b()
        scenarios += tier_b
        # The stratum belongs to the cell, not to the built scenario:
        # `getattr(b, "stratum", "")` was blank on every row, which left the
        # headline unable to split by geometry at all.
        cells = suite.tier_b_cells()
        by_cell = [cells[int(b.case_id.split("-")[1])] for b in tier_b]
        jobs += [(b, TIER_B_SEED + i, "model", None,
                  {"set": "tier_b", "case_id": b.case_id, "scenario": i,
                   "stratum": cell["stratum"], "behaviour": cell["behaviour"]})
                 for i, (b, cell) in enumerate(zip(tier_b, by_cell))]

    with open(out / "manifest.json", "w") as fh:
        json.dump(suite.manifest(scenarios), fh, indent=1, default=str)
    digest = suite.manifest(scenarios)["manifest_digest"][:16]

    frames = []
    for mode in (("off", "on") if args.supervisor == "both" else (args.supervisor,)):
        rows = run_pool(jobs, model_path=model, processes=args.processes,
                        overrides={"EMERGENCY_STOP_ENABLED": mode == "on"})
        frame = pd.DataFrame(rows)
        frame["supervisor"] = mode
        frames.append(frame)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(out / "episodes.csv", index=False)

    pd.set_option("display.width", 220)
    text = [f"Frozen suite ({args.tag}) — model {model.relative_to(ROOT)}",
            f"{len(d)} episodes, suite manifest {digest}, {time.time() - started:.0f} s",
            ""]
    if "a" in tiers:
        text += [f"== Tier A: {len(tier_a)} of {len(suite.tier_a())} named cases realised"
                 + (f"; not realisable: {', '.join(missing)}" if missing else ""), ""]
        a = d[d.set == "tier_a"]
        text += ["-- by supervisor", _md(_summarise(a, "supervisor")), "",
                 "-- by class (supervisor off)",
                 _md(_summarise(a[a.supervisor == "off"], "class")), "",
                 "-- per case (supervisor off)",
                 _md(a[a.supervisor == "off"].set_index("case_id")
                     [["class", "width", "outcome", "min_target_range", "rms_cte"]].round(2)), ""]
    if "b" in tiers:
        b = d[d.set == "tier_b"]
        if shortfall_b:
            text += [f"== Tier B: {len(shortfall_b)} cell(s) short of their episode count", ""]
        text += ["== Tier B (headline, R1)",
                 "-- by supervisor", _md(_summarise(b, "supervisor")), "",
                 "-- by class (supervisor off)",
                 _md(_summarise(b[b.supervisor == "off"], "class")), "",
                 "-- by stratum (supervisor off)",
                 _md(_summarise(b[b.supervisor == "off"], "stratum")), "",
                 "-- by target behaviour (supervisor off)",
                 _md(_summarise(b[b.supervisor == "off"], "behaviour")), "",
                 "-- crossings by side (supervisor off), the A32 split",
                 _md(_summarise(b[(b.supervisor == "off") & (b["class"] == "crossing")],
                                "crossing_side")), ""]
    body = "\n".join(str(t) for t in text) + "\n"
    (out / "summary.txt").write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
