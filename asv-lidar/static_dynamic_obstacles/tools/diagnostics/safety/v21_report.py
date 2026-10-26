"""Paired report for V21-phase runs (results/safety_dev/v21_development/runs/<tag>/).

Reads every completed attempt's result JSON (the per-tag episodes.csv can mix row
layouts when protocol modes add columns), pairs modes by case, episode seed and
scenario digest, and writes outcome counts, rescues and losses against OFF and
against each base, and outcome transitions.

    python -B tools/diagnostics/safety/v21_report.py --tags val_off_check,val_v16_bt_c1,... \
        --name validation_v16_bt

Scoring (planning/SAFETY_ESTOP_PROTOCOL.md section 1, decision 2026-10-10): a goal is a
success; `stopped` and `timeout` rank between a goal and a contact; own-ship contacts
(obstacle, boundary, target) and target strikes on the stopped ship (`collision:by_target`)
are contacts.  Writes results/safety_dev/v21_development/reports/<name>/{report.md,paired.csv}.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "results/safety_dev/v21_development/runs"
REPORTS = ROOT / "results/safety_dev/v21_development/reports"
SELECTION = ROOT / "results/safety_dev/v21_development/selection_validation.json"
# Own-vessel collisions (decision 2026-10-25: the layer's purpose): contacts made by the own
# vessel; a target striking the stopped ship (`collision:by_target`) is not one.
OWN = ("collision:obstacle", "collision:boundary", "collision:target")
COLUMNS = (("goal", "goal"), ("obstacle", "collision:obstacle"), ("boundary", "collision:boundary"),
           ("target", "collision:target"), ("by_target", "collision:by_target"), ("stopped", "stopped"),
           ("timeout", "timeout"))
CONTACTS = ("collision:obstacle", "collision:boundary", "collision:target", "collision:by_target")


def load(tags, runs=RUNS) -> pd.DataFrame:
    rows = []
    for tag in tags:
        for path in sorted((runs / tag / "attempts").glob("*_result.json")):
            r = json.loads(path.read_text())
            rows.append({"tag": tag, "case": r["case"], "mode": r["mode"], "seed": int(r["seed"]),
                         "digest": r["scenario_sha256"], "outcome": r["outcome"], "steps": r["steps"],
                         "interventions": r.get("safety_v2_steps", 0), "elapsed_s": r.get("elapsed_s"),
                         "why_counts": json.dumps(r.get("why_counts", {}), sort_keys=True)})
    d = pd.DataFrame(rows)
    dup = d.duplicated(["case", "mode"])
    if dup.any():
        raise SystemExit(f"duplicate case/mode records: {d[dup][['case', 'mode']].values.tolist()[:5]}")
    ident = d.groupby("case")[["seed", "digest"]].nunique()
    if (ident > 1).any().any():
        raise SystemExit("a case has more than one seed or digest across modes")
    return d


def counts(d: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for mode, x in d.groupby("mode", sort=False):
        c = Counter(x.outcome)
        out[mode] = {"episodes": len(x), **{k: c.get(v, 0) for k, v in COLUMNS},
                     "contacts": sum(c.get(v, 0) for v in CONTACTS),
                     "own_vessel": sum(c.get(v, 0) for v in OWN)}
    return pd.DataFrame(out).T


def markdown(t: pd.DataFrame) -> str:
    """A plain markdown table (no optional dependency)."""
    head = "| Mode | " + " | ".join(map(str, t.columns)) + " |"
    rule = "|---|" + "---:|" * len(t.columns)
    body = ["| " + str(i) + " | " + " | ".join(str(v) for v in row) + " |" for i, row in zip(t.index, t.values)]
    return "\n".join([head, rule] + body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", default="off")
    ap.add_argument("--bases", default="v16,v16_estop_bt", help="modes every other mode is also paired with")
    ap.add_argument("--runs", type=Path, default=RUNS, help="run directory (test set v4: results/safety_dev/testset_v4_main/runs)")
    ap.add_argument("--reports", type=Path, default=REPORTS)
    args = ap.parse_args()
    d = load(args.tags.split(","), args.runs)
    try:
        source = {c["case"]: c.get("source", "") for c in json.loads(SELECTION.read_text())["cases"]}
    except FileNotFoundError:
        source = {}
    d["source"] = d.case.map(source).fillna("")
    modes = list(dict.fromkeys(d["mode"]))
    p = d.pivot(index="case", columns="mode", values="outcome")
    complete = p.dropna()
    lines = [f"# V21 paired report: {args.name}", "",
             f"Tags: {', '.join(args.tags.split(','))}. {len(complete)} cases with every mode "
             f"({', '.join(modes)}); paired by case, episode seed and scenario digest.", "",
             "## Outcomes", "", markdown(counts(d[d.case.isin(complete.index)])), ""]
    for src in sorted(set(d.source) - {""}):
        x = d[(d.source == src) & d.case.isin(complete.index)]
        lines += [f"### {src}", "", markdown(counts(x)), ""]
    refs = [args.reference] + [b for b in args.bases.split(",") if b in modes]
    lines += ["## Paired changes", "", "| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |",
              "|---|---:|---:|---:|---:|---:|"]
    transitions = []
    for ref in refs:
        for m in modes:
            if m == ref or ref not in complete:
                continue
            a, b = complete[ref], complete[m]
            gained, lost = int(((a != "goal") & (b == "goal")).sum()), int(((a == "goal") & (b != "goal")).sum())
            lines.append(f"| {m} vs {ref} | {len(a)} | {(a == 'goal').sum()} | {(b == 'goal').sum()} | {gained} | {lost} |")
            ch = Counter(f"{x} -> {y}" for x, y in zip(a, b) if x != y)
            transitions.append(f"- {m} vs {ref}: " + "; ".join(f"{k} x{v}" for k, v in sorted(ch.items())))
    lines += ["", "Outcome transitions (reference -> candidate):", ""] + transitions + [""]
    out = args.reports / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    complete.join(d.groupby("case").source.first()).to_csv(out / "paired.csv")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
