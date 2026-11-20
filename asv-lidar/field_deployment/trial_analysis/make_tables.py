"""Section 5 tables F1, F2, F3 and F5, and the field-claim verdicts (C-15 water part, C-16,
C-17, C-18 closed loop, C-20), from one or more trial sessions.

Registered empty in `PAPER3_CLAIMS_AND_TABLES.md` (Part 2); F4 is the calibration's coverage
report (`coverage.py`).  Run `twin.py` first for F3's twin columns; without twins F3 carries the
open-loop columns only.

    python make_tables.py <session_dir> [<session_dir> ...] [--calibrated params.json] [--out DIR]
Output (default <first session>/analysis/): per_run_metrics.csv, f1_runs_by_scenario.csv,
f2_family_b.csv, f3_sim_to_field.csv, f3_outcome_agreement.csv, f5_perception.csv,
verdicts.json, tables.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import trial_io as tio
import trial_metrics as tm

TEST_SET_RESULTS = tio.SIM_ROOT / "results" / "test_set" / "v4" / "sacs0_bl3_metrics" / "episodes.csv"
COMMAND_PATH_LIMIT_S = 0.5          # T-14
C17_LOWER_QUANTILE = 0.05           # one-sided 95 % lower bound


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------
def c15_water(inputs: List[tio.TrialInputs]) -> Dict:
    """C-15, water part: every decision's observation inside its space; command path p99 < 0.5 s."""
    from env import ASVLidarEnv
    space = ASVLidarEnv(render_mode=None).observation_space
    outside, decisions = [], 0
    for inp in inputs:
        for k, obs in enumerate(inp.run.observation):
            decisions += 1
            for key, box in space.spaces.items():
                v = np.asarray(obs.get(key, []), dtype=float)
                if v.shape != box.shape or not np.all(np.isfinite(v)) or np.any(v < box.low - 1e-6) \
                        or np.any(v > box.high + 1e-6):
                    outside.append(f"{inp.run.run_id}:{k}:{key}")
    lat = np.concatenate([i.run.command_latency for i in inputs]) if inputs else np.array([])
    lat = lat[np.isfinite(lat)]
    p99 = float(np.percentile(lat, 99)) if len(lat) else float("nan")
    return {"decisions": decisions, "observations_outside_space": len(outside), "first_outside": outside[:10],
            "command_path_p99_s": p99, "met": (not outside) and math.isfinite(p99) and p99 < COMMAND_PATH_LIMIT_S}


def c17(rows: List[Dict], results_csv: Path, variants=("FIX", "VAR")) -> Dict:
    """C-17: pooled family C field successes not below the one-sided 95 % binomial lower bound of
    the pooled simulated rate of the same cells, for the same number of runs."""
    runs = [r for r in rows if r["family"] == "C" and r["valid"]]
    if not runs:
        return {"met": None, "reason": "no valid family C runs"}
    e = pd.read_csv(results_csv)
    e = e[(e.safety == "off") & e.variant.isin(variants)] if "safety" in e else e[e.variant.isin(variants)]
    per_cell = []
    for cell in sorted({r["sim_cell"] for r in runs}):
        f = [r for r in runs if r["sim_cell"] == cell]
        s = e[e.cell == cell]
        per_cell.append({"cell": cell, "field_goals": sum(r["outcome"] == "goal" for r in f), "field_runs": len(f),
                         "sim_goals": int((s.outcome == "goal").sum()), "sim_episodes": int(len(s))})
    sim_goals = sum(c["sim_goals"] for c in per_cell)
    sim_eps = sum(c["sim_episodes"] for c in per_cell)
    if sim_eps == 0:
        return {"met": None, "reason": "no simulated episodes in these cells", "per_cell": per_cell}
    p = sim_goals / sim_eps
    n = len(runs)
    k = sum(r["outcome"] == "goal" for r in runs)
    bound = int(stats.binom.ppf(C17_LOWER_QUANTILE, n, p))
    weighted = (sum(c["field_runs"] * c["sim_goals"] / max(c["sim_episodes"], 1) for c in per_cell) / n)
    return {"field_goals": k, "field_runs": n, "pooled_sim_rate": p, "lower_bound": bound,
            "met": k >= bound, "variants": list(variants), "results": str(results_csv),
            "run_weighted_sim_rate": weighted, "per_cell": per_cell,
            "rule": "field goals >= binom.ppf(0.05, runs, pooled simulated rate)"}


def c20(rows: List[Dict]) -> Dict:
    valid = [r for r in rows if r["valid"]]
    return {"runs": len(valid),
            "would_change_rate_median": float(np.nanmedian([r["shadow_would_change_rate"] for r in valid])) if valid else float("nan"),
            "runs_with_stop_commitment": sum(bool(r["shadow_stop_committed"]) for r in valid),
            "descriptive": True}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def f1(rows: List[Dict], results_csv: Path) -> pd.DataFrame:
    e = pd.read_csv(results_csv)
    if "safety" in e:
        e = e[e.safety == "off"]
    out = []
    for sid in sorted({r["scenario_id"] for r in rows}):
        g = [r for r in rows if r["scenario_id"] == sid]
        v = [r for r in g if r["valid"]]
        ranges = [r["min_target_range"] for r in v if np.isfinite(r["min_target_range"])]
        cell = g[0]["sim_cell"]
        s = e[e.cell == cell] if cell else e.iloc[0:0]
        out.append({"scenario": sid, "family": g[0]["family"], "encounter": g[0]["encounter"],
                    "layout": g[0]["layout"], "valid / runs": f"{len(v)} / {len(g)}",
                    "goals": sum(r["outcome"] == "goal" for r in v),
                    "contact panel": sum(r["outcome"] == "contact_panel" for r in v),
                    "contact boundary": sum(r["outcome"] == "contact_boundary" for r in v),
                    "contact target": sum(r["outcome"] == "contact_target" for r in v),
                    "operator stops": sum(r["outcome"] == "operator_stop" for r in v),
                    "would-be layer stops": sum(bool(r["shadow_stop_committed"]) for r in v),
                    "min target range m (median, range)": (f"{np.median(ranges):.2f} ({min(ranges):.2f}-{max(ranges):.2f})"
                                                          if ranges else ""),
                    "simulated twin cell": (f"{cell}: {int((s.outcome == 'goal').sum())} / {len(s)}" if cell else "")})
    return pd.DataFrame(out)


F2_COLUMNS = ("run_id", "encounter", "valid", "outcome", "reached_cpa_without_stop", "first_manoeuvre",
              "max_lateral_m", "max_course_alteration_deg", "first_alteration_after_engagement_s",
              "target_side_at_cpa", "crossed_ahead", "cpa_m", "speed_reduction_mps", "rule", "rule_compliant")


def f3(rows: List[Dict], twins_dir: Path, open_loop_rows: List[Dict]) -> (pd.DataFrame, pd.DataFrame):
    import twin as twinmod   # the comparison helpers; importing it loads the simulator
    by_run = {r["run_id"]: r for r in open_loop_rows}
    out, agreement = [], []
    for r in rows:
        if not r["valid"]:
            continue
        row = {"run_id": r["run_id"], "family": r["family"], "encounter": r["encounter"], "outcome": r["outcome"]}
        for name in ("training", "calibrated"):
            path = twins_dir / r["run_id"] / f"twins_{name}.csv"
            if not path.exists():
                continue
            cmp_ = twinmod.compare(r, pd.read_csv(path).to_dict("records"))
            row.update({f"{name}_{k}": v for k, v in cmp_.items()})
            agreement.append({"model": name, "field_success": cmp_["field_success"], "twin_success": cmp_["twin_success"]})
        row.update({k: v for k, v in by_run.get(r["run_id"], {}).items() if k not in ("run_id",)})
        out.append(row)
    table = pd.DataFrame(agreement)
    if len(table):
        table = (table.groupby(["model", "field_success", "twin_success"]).size()
                 .rename("runs").reset_index())
    return pd.DataFrame(out), table


def f5(perc: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(perc)


def markdown(df: pd.DataFrame, title: str) -> str:
    if df is None or not len(df):
        return f"### {title}\n\n(no rows)\n"
    cols = list(df.columns)

    def cell(v):
        if isinstance(v, float):
            return "" if not math.isfinite(v) else f"{v:.3g}"
        return str(v)
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in rec) + " |" for rec in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--calibrated", help="parameter JSON of the session-1 fit (open-loop columns)")
    ap.add_argument("--twins", type=Path, help="twin output folder (default <session>/analysis/twins)")
    ap.add_argument("--results", type=Path, default=TEST_SET_RESULTS, help="test set v4 episodes.csv of the deployed policy")
    ap.add_argument("--variants", default="FIX,VAR", help="test set target variants pooled for C-17 and F1")
    ap.add_argument("--skip-c15", action="store_true", help="skip the observation-space check (loads the simulator)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    inputs = [i for s in args.sessions for i in tio.load_session(s)]
    out = args.out or (args.sessions[0] / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for inp in inputs:
        r = tm.run_metrics(inp)
        r["sim_cell"] = tio.sim_cell(inp.sheet)
        rows.append(r)
    behaviour = [b for b in (tm.encounter_behaviour(i) for i in inputs) if b]
    perc = [tm.perception(i) for i in inputs]

    import open_loop as ol
    models = ol.load_models(args.calibrated)
    runs = [r for r in (ol.bluefin_run(i) for i in inputs if i.sheet["valid"]) if r]
    ol_rows = ol.per_run_table(runs, models) if runs else []

    twins_dir = args.twins or (args.sessions[0] / "analysis" / "twins")
    t3, agreement = f3(rows, twins_dir, ol_rows)
    variants = tuple(v for v in args.variants.split(",") if v)
    verdicts = {"C-16": tm.c16_verdict(behaviour), "C-17": c17(rows, args.results, variants), "C-20": c20(rows)}
    if not args.skip_c15:
        verdicts["C-15 water"] = c15_water(inputs)
    if len(agreement):
        verdicts["C-18 closed loop"] = {"outcome_agreement": agreement.to_dict("records"), "descriptive": True}

    tables = {"f1_runs_by_scenario": f1(rows, args.results),
              "f2_family_b": pd.DataFrame([{k: b.get(k) for k in F2_COLUMNS} for b in behaviour if b["family"] == "B"]),
              "f3_sim_to_field": t3, "f3_outcome_agreement": agreement, "f5_perception": f5(perc),
              "per_run_metrics": pd.DataFrame(rows)}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2, default=str))
    md = ["# Section 5 tables (generated)", ""]
    md += [markdown(tables[k], k) for k in ("f1_runs_by_scenario", "f2_family_b", "f3_outcome_agreement", "f5_perception")]
    md += ["### Verdicts", "", "```json", json.dumps(verdicts, indent=2, default=str), "```", ""]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(inputs)} runs -> {out}")
    print(json.dumps({k: v.get("met") for k, v in verdicts.items()}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
