"""Saved-trace audit of V16 decision branches before contact (V20 phase 1).

Read-only. No environment, policy, controller, reset or episode is executed.
Inputs are the validated V16 cohorts already on disk:

* 32-case cohort: ``reports/motion_axis32/paired.csv`` and the V16 traces it names;
* 40-case cohort: ``reports/v16_broader40_paired/paired.csv`` and its V16 traces;
* optional context: V16 with ``prefer_any_feasible_policy=True`` in
  ``v16_feasible_probe9`` (three cases that option lost).

Each V16 decision is labelled by the filter's recorded ``why``. Two branches
issue a command that the filter's own checker did not pass at that decision:
``last certificate`` (a stored plan whose current recheck fails) and
``no escape`` (no passing plan, the policy command stands). Every other
non-idle branch issued a command whose plan hard-passed the inherited checker
at that decision. ``idle`` means no hazard was within the engagement test.

The recorded ``pre_state`` surge is simulator truth. It is used here only to
score when the vessel was at rest; it never enters any controller.

Usage (from the project root):
    python -B tools/diagnostics/safety/audit_v20_saved_cascades.py            # print only
    python -B tools/diagnostics/safety/audit_v20_saved_cascades.py --tag NEW  # also write results
Existing output tags are refused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "results/safety_dev/v10_iterations"
OUT_ROOT = ROOT / "results/safety_dev/v20_development"
COHORTS = (("32", CAMPAIGN / "reports/motion_axis32/paired.csv", None),
           ("40", CAMPAIGN / "reports/v16_broader40_paired/paired.csv", "v16_broader40_paired"))
ANY_FEASIBLE = ("v16_feasible_probe9",
                ("DV3:DV3-HO-CV-03", "TS2:P2-L1-CRP-VAR-12", "TS2:P2-L2-HO-FIX-19"))
UNCHECKED = ("last certificate", "no escape", "hold back")
REST_SPEED = 0.05   # m/s, equals constants.ESTOP_STOP_SPEED; truth-scored only


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_trace(tag: str, case: str, mode: str):
    directory = CAMPAIGN / tag
    for result in sorted((directory / "attempts").glob("[0-9][0-9][0-9]_result.json")):
        row = json.loads(result.read_text())
        if row["case"] == case and row["mode"] == mode:
            return directory / "traces" / f"{row['attempt']:03d}_{mode}.jsonl", row
    raise FileNotFoundError(f"{tag}: no {mode} record for {case}")


def category(off: str, filtered: str) -> str:
    if off == "goal":
        return "both_goal" if filtered == "goal" else "lost_sac_success"
    return "rescue" if filtered == "goal" else "both_fail"


def finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else ("inf" if number > 0 else "-inf")


def summarise(trace: Path, run: dict, cohort: str, off: str, inputs: dict) -> dict:
    inputs[trace.relative_to(ROOT).as_posix()] = sha(trace)
    records = [json.loads(line) for line in trace.open(encoding="utf-8")]
    whys = [r["filter"].get("why", "idle") for r in records]
    changed = [r for r in records if r["changed"]]
    first = changed[0] if changed else None
    rest = None
    if first is not None:
        rest = next((r["step"] for r in records[first["step"] - 1:]
                     if r["pre_state"][3] < REST_SPEED), None)
    contact = run["outcome"].startswith("collision")
    tail_unchecked = 0
    for why in reversed(whys):
        if why not in UNCHECKED:
            break
        tail_unchecked += 1
    last_checked = next((r for r in reversed(records)
                         if r["filter"].get("why", "idle") not in UNCHECKED), None)
    first_filter = first["filter"] if first else {}
    return {
        "cohort": cohort, "case": run["case"], "seed": run["seed"],
        "category": category(off, run["outcome"]), "off_outcome": off,
        "v16_outcome": run["outcome"], "decisions": len(records),
        "changed_decisions": len(changed),
        "first_change_step": first["step"] if first else None,
        "first_change_why": first_filter.get("why") if first else None,
        "first_change_sac_same_tail_first_violation_s": finite_or_none(
            first_filter.get("v10_policy_first_violation")) if first else None,
        "first_change_sac_same_tail_hard_pass": (
            finite_or_none(first_filter.get("v10_policy_first_violation")) == "inf") if first else None,
        "brake_decisions": sum(bool(r["brake"]) for r in records),
        "first_rest_step_after_first_change_truth_scored": rest,
        "last_certificate_decisions": whys.count("last certificate"),
        "no_escape_decisions": whys.count("no escape"),
        "final_why": whys[-1],
        "contact": contact,
        "consecutive_unchecked_decisions_before_end": tail_unchecked,
        "last_checked_step": last_checked["step"] if last_checked else None,
        "last_checked_why": last_checked["filter"].get("why", "idle") if last_checked else None,
        "branch_counts": dict(Counter(whys)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Write results under v20_development/<tag>")
    args = parser.parse_args()
    out = None
    if args.tag:
        if Path(args.tag).name != args.tag:
            parser.error("Use a simple tag")
        out = OUT_ROOT / args.tag
        if out.exists():
            parser.error("Existing tag; results are never overwritten")

    inputs, rows = {}, []
    for cohort, paired_path, fixed_tag in COHORTS:
        inputs[paired_path.relative_to(ROOT).as_posix()] = sha(paired_path)
        for row in csv.DictReader(paired_path.open(encoding="utf-8")):
            if row["mode"] != "v16":
                continue
            tag = fixed_tag or row["source_iteration"]
            trace, run = find_trace(tag, row["case"], "v16")
            if run["outcome"] != row["outcome"] or int(run["seed"]) != int(row["seed"]):
                raise ValueError(f"Paired row does not match its run record: {row['case']}")
            rows.append(summarise(trace, run, cohort, row["off_outcome"], inputs))

    feasible = []
    tag, cases = ANY_FEASIBLE
    for case in cases:
        trace, run = find_trace(tag, case, "v16")
        feasible.append(summarise(trace, run, "any_feasible_probe", "goal", inputs))

    groups = {}
    for name in ("lost_sac_success", "rescue", "both_fail", "both_goal"):
        group = [r for r in rows if r["category"] == name]
        groups[name] = {
            "cases": len(group),
            "intervened": sum(r["changed_decisions"] > 0 for r in group),
            "braked": sum(r["brake_decisions"] > 0 for r in group),
            "at_rest_after_first_change_truth_scored": sum(
                r["first_rest_step_after_first_change_truth_scored"] is not None for r in group),
            "with_last_certificate": sum(r["last_certificate_decisions"] > 0 for r in group),
            "with_no_escape": sum(r["no_escape_decisions"] > 0 for r in group),
            "first_change_with_sac_same_tail_hard_pass": sum(
                r["first_change_sac_same_tail_hard_pass"] is True for r in group),
        }
    contacts = [r for r in rows if r["contact"]]
    final = Counter(r["final_why"] for r in contacts)
    branch_totals = Counter()
    for r in rows:
        branch_totals.update(r["branch_counts"])
    summary = {
        "schema": 1,
        "scope": ("Saved V16 traces of the disjoint 32- and 40-case development cohorts; "
                  "read-only; no episodes; truth used only to score rest"),
        "cases": len(rows), "decisions": sum(r["decisions"] for r in rows),
        "groups": groups,
        "contact_episodes": len(contacts),
        "contact_final_branch": dict(final),
        "contact_final_branch_unchecked": sum(final[w] for w in UNCHECKED),
        "branch_totals": dict(branch_totals),
        "any_feasible_losses": feasible,
        "input_sha256": inputs,
        "script_sha256": sha(Path(__file__)),
    }
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("input_sha256", "any_feasible_losses")}, indent=1))
    for r in contacts:
        print(f"{r['cohort']} {r['case']:28s} {r['category']:16s} {r['v16_outcome']:18s} "
              f"final={r['final_why']:16s} unchecked_tail={r['consecutive_unchecked_decisions_before_end']:2d} "
              f"last_checked={r['last_checked_step']}/{r['last_checked_why']}")
    if out is None:
        return
    out.mkdir(parents=True)
    (out / "audit.json").write_text(json.dumps(dict(summary, rows=rows), indent=1) + "\n",
                                    encoding="utf-8")
    fields = [k for k in rows[0] if k != "branch_counts"]
    with (out / "cases.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
