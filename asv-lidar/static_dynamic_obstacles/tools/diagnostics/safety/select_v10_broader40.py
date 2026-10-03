"""Freeze an enriched saved-data challenge cohort; never import a simulator.

Run from any directory. Writes one new selection JSON, refusing overwrite.
Historical outcomes define diagnostic strata, not controller inputs or a
population sampling design. Within declared strata the ordering is SHA-256.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/safety_dev/v10_iterations/selection_broader40.json"
SALT = "v10-broader40-2026-10-03-before-expanded-outcomes-v1"
INPUTS = {
    "history": ROOT / "results/safety_dev/v8_followup_offline/audit/all_1150.csv",
    "excluded": ROOT / "results/safety_dev/v9_paired_pilot/selection.json",
    "ts2": ROOT / "results/test_set/v2/definition.csv",
    "dv3": ROOT / "results/safety_dev/suites/inventory_v5/manifest.json",
    "geometry": ROOT / "results/safety_dev/testset_v2_offline/analysis/scenario_features.csv",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rank(row):
    return sha((SALT + "|" + row["case"]).encode("utf-8"))


def build():
    history = read_csv(INPUTS["history"])
    excluded = {row["case"] for row in json.loads(INPUTS["excluded"].read_text())["cases"]}
    definitions = {"TS2:" + row["test_id"]: row for row in read_csv(INPUTS["ts2"])}
    dv3 = {"DV3:" + row["case"]: row for row in json.loads(INPUTS["dv3"].read_text())["cases"]
           if row["suite"] == "dev_field"}
    geometry = {"TS2:" + row["test_id"]: row for row in read_csv(INPUTS["geometry"])}
    assert len(history) == len({row["case"] for row in history}) == 1150
    assert len(excluded) == 32 and len(definitions) == 1000 and len(dv3) == 150
    pool = [row for row in history if row["case"] not in excluded]
    assert len(pool) == 1118
    chosen, used, strata = [], set(), []

    def family(row):
        if row["dataset"] == "ts2":
            entry = definitions[row["case"]]
            return entry["cell"], entry["variant"], entry["class"]
        parts = row["case"].split(":", 1)[1].split("-")
        return parts[1], parts[2], parts[1]

    def add(row, stratum, rationale, matching=None):
        case = row["case"]
        assert case not in used and case not in excluded
        if row["dataset"] == "ts2":
            canonical = definitions[case]
            seed, digest = int(canonical["episode_seed"]), canonical["digest"]
            feature = geometry[case]
            assert int(feature["episode_seed"]) == seed and feature["scenario_sha256"] == digest
            static = {k: feature[k] for k in (
                "nominal_width", "width_ratio", "bend_deg", "path_offset_frac",
                "target_speed", "speed_ratio", "dcpa_m", "tcpa_s",
                "obstacle_count_for_matching", "fixed_obstacle_count")}
        else:
            canonical = dv3[case]
            seed, digest = int(canonical["seed"]), canonical["scenario_sha256"]
            static = None  # Cache digest pins geometry; no simulator/cache import needed here.
        assert seed == int(row["seed"]) and len(digest) == 64
        item = {
            "case": case, "scenario_id": case.split(":", 1)[1],
            "dataset": row["dataset"], "seed": seed, "scenario_sha256": digest,
            "selection_stratum": stratum, "rationale": rationale,
            "selection_rank_sha256": rank(row), "family": family(row),
            "static_geometry_metadata": static,
            "historical": {k: row[k] for k in (
                "policy_outcome", "v4_outcome", "v7_outcome", "v8_outcome",
                "v8_category", "first_reason", "first_fire_step",
                "selected_clearance", "policy_prefix_repair_clearance")},
        }
        if matching is not None:
            item["rescue_matching"] = matching
        chosen.append(item)
        used.add(case)

    broken = sorted((r for r in pool if r["v8_category"] == "broken"), key=rank)
    assert len(broken) == 11
    for row in broken:
        add(row, "remaining_known_sac_regression", "Every remaining saved SAC success broken by V8.")
    strata.append({"name": "remaining_known_sac_regression", "eligible": 11, "selected": 11})

    for source in broken:
        candidates = [r for r in pool if r["v8_category"] == "rescued" and r["case"] not in used
                      and r["dataset"] == source["dataset"] and r["first_reason"] == source["first_reason"]]

        def priority(row):
            a, b = family(source), family(row)
            return 0 if a[:2] == b[:2] else 1 if a[0] == b[0] else 2 if a[2] == b[2] else 3

        assert candidates
        selected = min(candidates, key=lambda r: (priority(r), rank(r)))
        add(selected, "rescue_control", "Match a regression's dataset and initial intervention reason; prefer family/variant, then fixed hash.",
            {"regression_case": source["case"], "dataset_and_first_reason_exact": True,
             "priority": priority(selected), "priority_meaning":
             ["same cell and variant", "same cell", "same scenario class", "dataset and reason only"][priority(selected)],
             "eligible_remaining_same_dataset_reason": len(candidates)})
    strata.append({"name": "rescue_control", "eligible": 67, "selected": 11,
                   "selection": "Sequential mandatory-regression hash order; distinct controls; exact dataset/reason and categorical family preference."})

    quotas = [
        ("both_goal", "active", "dv3", 1), ("both_goal", "active", "ts2", 4),
        ("both_goal", "no_fire", "dv3", 1), ("both_goal", "no_fire", "ts2", 4),
        ("both_fail", "active", "dv3", 1), ("both_fail", "active", "ts2", 3),
        ("both_fail", "no_fire", "dv3", 1), ("both_fail", "no_fire", "ts2", 3),
    ]
    for category, activity, dataset, count in quotas:
        name = "_".join((category, activity, dataset))
        eligible = [r for r in pool if r["v8_category"] == category and r["dataset"] == dataset
                    and (r["first_reason"] == "no fire") == (activity == "no_fire")]
        assert len(eligible) >= count
        for row in sorted(eligible, key=rank)[:count]:
            add(row, name, "Fixed hash rank within predeclared dataset, historical outcome, and trigger-activity stratum.")
        strata.append({"name": name, "eligible": len(eligible), "selected": count})

    assert len(chosen) == len(used) == 40 and not (used & excluded)
    categories = Counter(r["historical"]["v8_category"] for r in chosen)
    assert categories == {"broken": 11, "rescued": 11, "both_goal": 10, "both_fail": 8}
    return {
        "schema": 1, "tag": "v10_broader40", "scenarios": 40,
        "scope": "Outcome-enriched development challenge, frozen before expanded-candidate results. Candidate-only runs are staged separately; this selection does not authorize or assume fresh baseline runs.",
        "salt": SALT, "ranking": "SHA256(UTF8(salt + '|' + canonical case)) ascending; categorical rescue matching takes priority, with hash tie-breaks.",
        "excluded_previous_cases": sorted(excluded), "excluded_count": 32,
        "input_sha256": {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in INPUTS.values()},
        "selection_builder": Path(__file__).relative_to(ROOT).as_posix(),
        "selection_builder_sha256": sha(Path(__file__).read_bytes()),
        "strata": strata, "cases": chosen,
        "dataset_counts": dict(Counter(r["dataset"] for r in chosen)),
        "historical_category_counts": dict(categories),
        "historical_mode_goal_counts": {mode: sum(r["historical"][mode + "_outcome"] == "goal" for r in chosen)
                                        for mode in ("policy", "v4", "v7", "v8")},
        "limitations": [
            "Historical outcomes intentionally define diagnostic strata; no representative population estimate or statistical significance claim is justified.",
            "All 11 previously unselected known SAC-to-V8 regressions are included; coverage does not imply all possible candidate regressions are known.",
            "Rescue matching is categorical diagnostic balancing, not causal matching, a learned classifier, or a runtime controller rule.",
            "Saved outcomes are historical references. Fresh versus historical comparisons require source/model/seed provenance qualification; no fresh baseline outcomes are fabricated.",
            "Numeric geometry metadata is available from saved TS2 analysis; DV3 geometry is pinned by canonical digest without importing or regenerating scenarios.",
            "No simulator, model inference, environment reset, or episode execution occurs in this selection script.",
        ],
    }


if __name__ == "__main__":
    manifest = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"path": OUT.relative_to(ROOT).as_posix(), "sha256": sha(OUT.read_bytes()),
                      "counts": manifest["historical_category_counts"], "datasets": manifest["dataset_counts"]}))
