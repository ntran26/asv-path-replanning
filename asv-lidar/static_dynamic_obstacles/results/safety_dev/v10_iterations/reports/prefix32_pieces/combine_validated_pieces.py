"""Explicitly sum two validated prefix pieces without a cohort-source waiver."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools/diagnostics/safety"))
from suite_status import read_shared_bytes

HASHES = {}
OUTCOMES = ("goal", "collision:target", "collision:obstacle", "collision:boundary", "timeout")
REFERENCES = ("off", "v8", "v9", "v10", "v11")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = read_shared_bytes(path)
    HASHES[path.relative_to(ROOT).as_posix()] = sha(raw)
    return raw


def paired(rows, reference):
    gain = [r["case"] for r in rows if r["outcome"] == "goal" and r[reference + "_outcome"] != "goal"]
    loss = [r["case"] for r in rows if r["outcome"] != "goal" and r[reference + "_outcome"] == "goal"]
    return {"reference": reference, "matched_cases": len(rows), "gained_goals": len(gain), "lost_goals": len(loss),
            "net_goals": len(gain) - len(loss), "gained_cases": gain, "lost_cases": loss,
            "reference_evidence": sorted({r[reference + "_evidence"] for r in rows}),
            "collision_to_goal": sum(r[reference + "_outcome"].startswith("collision:") and r["outcome"] == "goal" for r in rows),
            "collision_to_timeout": sum(r[reference + "_outcome"].startswith("collision:") and r["outcome"] == "timeout" for r in rows)}


def main():
    summary = json.loads(read(OUT / "summary.json"))
    provenance = json.loads(read(OUT / "provenance.json"))
    rows = list(csv.DictReader(io.StringIO(read(OUT / "paired.csv").decode("utf-8"))))
    audit_path = OUT.parent / "prefix32_inventory_audit/audit.json"
    audit = json.loads(read(audit_path))
    expected = json.loads(read(ROOT / "results/safety_dev/v9_paired_pilot/selection.json"))
    canonical = {r["case"]: r for r in expected["cases"]}
    assert len(rows) == len({r["case"] for r in rows}) == 32 and {r["case"] for r in rows} == set(canonical)
    assert {r["tag"] for r in rows} == {"prefix_success16", "prefix_remaining16"}
    assert all(sum(r["tag"] == tag for r in rows) == 16 for tag in ("prefix_success16", "prefix_remaining16"))
    assert all(r["mode"] == "v11_prefix" and r["outcome"] in OUTCOMES for r in rows)
    for row in rows:
        case = canonical[row["case"]]
        assert int(row["seed"]) == case["seed"] and row["scenario_sha256"] == case["scenario_sha256"]
        assert row["dataset"] == case["dataset"]
        for reference in REFERENCES:
            assert row[reference + "_outcome"] in OUTCOMES
            assert row[reference + "_evidence"] == ("fresh_prior_pilot" if reference in ("off", "v8", "v9") else "fresh_reference_iteration")
    pieces = {r["tag"]: r for r in summary["iterations"]}
    assert set(pieces) == {"prefix_success16", "prefix_remaining16"}
    assert all(r["complete"] and r["completed"] == r["planned_runs"] == 16 and not r["filter_error_steps"] for r in pieces.values())
    assert audit["combined_distinct_cases"] == 32 and audit["all_case_seeds_and_scenario_digests_match_canonical_saved_inventory"]
    assert not audit["changed_common_sources"] and not audit["removed_sources"]
    assert set(audit["added_sources"]) == {"src/dev_set_v4.py", "src/safety_v14.py"}
    assert not any(audit["literal_references_in_common_archived_sources"].values())
    # Link the exact immutable manifests/archives inspected by both validators.
    for tag in pieces:
        for name in ("manifest.json", "evaluated_sources.zip"):
            path = ROOT / "results/safety_dev/v10_iterations" / tag / name
            relative = path.relative_to(ROOT).as_posix()
            assert audit["input_sha256"][relative] == provenance["input_sha256"][str(path.resolve())]
    outcomes = {value: sum(r["outcome"] == value for r in rows) for value in OUTCOMES}
    comparisons = [paired(rows, reference) for reference in REFERENCES]
    result = {"schema": 1, "combination": "Explicit descriptive sum of two separately validated16-case pieces with unequal source inventories; strict cohort exception not used.",
              "completed_records": 32, "distinct_scenarios": 32, "outcomes": outcomes, "comparisons": comparisons,
              "piece_outcomes": {tag: {value: sum(r["tag"] == tag and r["outcome"] == value for r in rows) for value in OUTCOMES} for tag in pieces},
              "episode_seconds": sum(float(r["elapsed_s"]) for r in rows),
              "dataset_counts": {name: sum(r["dataset"] == name for r in rows) for name in sorted({r["dataset"] for r in rows})},
              "inventory_audit": {"path": audit_path.relative_to(ROOT).as_posix(), "common_identical_sources": audit["common_sources_identical"],
                                  "added_sources": audit["added_sources"]},
              "input_sha256": HASHES, "builder_sha256": sha(Path(__file__).read_bytes()), "new_episode_runs": 0}
    with (OUT / "combined_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    table_rows = []
    for comparison in comparisons:
        table_rows.append({"candidate": "V11 policy-prefix search, tracks disabled", "N": 32,
                           **outcomes, **{k: comparison[k] for k in ("reference", "matched_cases", "gained_goals", "lost_goals", "net_goals")},
                           "source_scope": "sum of separately validated pieces; unequal inventories disclosed"})
    with (OUT / "combined_table.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    lines = ["# Policy-prefix search: explicit32-case sum", "",
             f"The two completed16-case pieces total **{outcomes['goal']}/32 goals**, with {outcomes['collision:target']} target, "
             f"{outcomes['collision:obstacle']} obstacle and {outcomes['collision:boundary']} boundary collisions, and {outcomes['timeout']} timeouts.", "",
             "This table sums separately validated pieces with differing source inventories. All99 shared sources, controller options, checkpoint/config and scenario identities match; "
             "the later archive additionally contains `src/dev_set_v4.py` and `src/safety_v14.py`. The cached loader bypasses the new generator. "
             "The [source-inventory audit](../prefix32_inventory_audit/audit.md) records these checks and their dynamic-import limitation. The strict cohort API was not overridden.", "",
             "| Reference | Matched | Gained goals | Lost goals | Net |", "|---|---:|---:|---:|---:|"]
    for comparison in comparisons:
        lines.append(f"| {comparison['reference']} |32|{comparison['gained_goals']}|{comparison['lost_goals']}|{comparison['net_goals']:+d}|")
    v10 = next(p for p in comparisons if p["reference"] == "v10")
    lines += ["", "Versus default V10, gained: " + (", ".join(v10["gained_cases"]) or "none") + ".",
              "Versus default V10, lost: " + (", ".join(v10["lost_cases"]) or "none") + ".", "",
              "These are the same32 outcome-enriched development scenarios, not a population sample or a formal safety guarantee. "
              "All reference outcomes are fresh recorded evaluations; no historical-only outcomes are substituted. No new episodes were run for this report.", "",
              "[Separate validated pieces](report.md), [all per-case results](paired.csv), [combined counts and hashes](combined_summary.json), [table CSV](combined_table.csv).", ""]
    text = "\n".join(lines).replace("explicit32", "explicit 32").replace("completed16", "completed 16").replace("All99", "All 99").replace("same32", "same 32")
    with (OUT / "combined_report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(json.dumps({"outcomes": outcomes, "comparisons": [{k: p[k] for k in ("reference", "gained_goals", "lost_goals")} for p in comparisons]}))


if __name__ == "__main__":
    main()
