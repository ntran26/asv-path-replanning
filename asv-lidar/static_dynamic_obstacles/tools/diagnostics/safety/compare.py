"""Compare development outcomes by original case ID, retaining every gain/loss.

python -B tools/diagnostics/safety/compare.py results/safety_dev/dev_v4_initial.csv --mode v4
Historical off, best v2 iteration 2, v3, and selected v4 are the fixed references.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--mode", default="v4")
    args = parser.parse_args()
    candidate = pd.read_csv(args.candidate)
    candidate = candidate[candidate["mode"].eq(args.mode)]
    if candidate.empty or candidate["case"].duplicated().any():
        parser.error("candidate must contain exactly one row per case for the selected mode")
    if "diagnostic_limit" in set(candidate.outcome):
        parser.error("unfinished diagnostic episodes cannot be performance results")
    reference_files = [("off", "safety_v2_dev_iter2.csv", "off"),
                       ("v2_iter2", "safety_v2_dev_iter2.csv", "v2"),
                       ("v3", "safety_v2_dev_v3_iter2_refpolicy.csv", "v3"),
                       ("v4_selected", "safety_dev/dev_v4_observer_memory.csv", "v4")]
    summaries, pairs = [], []
    for label, filename, mode in reference_files:
        reference = pd.read_csv(ROOT / "results" / filename)
        reference = reference[reference["mode"].eq(mode)]
        if reference["case"].duplicated().any():
            raise ValueError(f"duplicate reference cases: {filename}")
        merged = candidate[["case", "outcome"]].merge(reference[["case", "outcome"]],
                   on="case", how="left", suffixes=("_candidate", "_reference"), validate="one_to_one")
        if merged.outcome_reference.isna().any():
            raise ValueError("candidate includes cases outside the development reference")
        merged["reference"] = label
        merged["gained_goal"] = merged.outcome_candidate.eq("goal") & ~merged.outcome_reference.eq("goal")
        merged["lost_goal"] = ~merged.outcome_candidate.eq("goal") & merged.outcome_reference.eq("goal")
        merged["changed_outcome"] = merged.outcome_candidate.ne(merged.outcome_reference)
        pairs.append(merged)
        summaries.append(dict(reference=label, paired_cases=len(merged),
                              candidate_goals=int(merged.outcome_candidate.eq("goal").sum()),
                              reference_goals=int(merged.outcome_reference.eq("goal").sum()),
                              gained=int(merged.gained_goal.sum()), lost=int(merged.lost_goal.sum()),
                              net=int(merged.gained_goal.sum() - merged.lost_goal.sum())))
    prefix = args.candidate.with_suffix("")
    pd.concat(pairs).to_csv(str(prefix) + "_paired.csv", index=False)
    pd.DataFrame(summaries).to_csv(str(prefix) + "_comparison.csv", index=False)
    print(candidate.outcome.value_counts().to_string())
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
