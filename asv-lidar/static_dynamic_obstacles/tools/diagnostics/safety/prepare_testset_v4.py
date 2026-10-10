"""Freeze the complete test-set-v4 safety benchmark from existing files.

Read-only with respect to the suite, policy and simulator. No episode, reset,
scenario generation or controller evaluation occurs. Existing output files
must match exactly; a changed definition requires a separately named protocol.
Test set v4 is test set v3 with 67 near-impossible episodes replaced
(planning/BASELINE_V4_PLAN.md); overlaps with test sets v2/v3 are recorded.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results/safety_dev/testset_v4_main"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite a different frozen artifact: {path}")
    else:
        path.write_bytes(raw)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    suite = ROOT / "results/test_set/v4"
    paths = [suite / name for name in ("definition.json", "definition.csv", "set_v4.0.pkl")]
    source = {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in paths}
    definition = json.loads(paths[0].read_text())
    rows = read_rows(paths[1])
    kept, cache_report = pickle.loads(paths[2].read_bytes())
    by_id = {row["test_id"]: row for row in rows}
    cached = {row["test_id"]: row for row in kept}
    if (definition["version"] != "4.0" or len(rows) != 1000 or len(kept) != 1000
            or len(by_id) != 1000 or len(cached) != 1000 or by_id.keys() != cached.keys()):
        raise ValueError("V4 must contain exactly the same 1,000 unique IDs in CSV and cache")
    digests, cases = {}, []
    for row in rows:
        item = cached[row["test_id"]]
        digest = item["built"].digest()
        if digest != row["digest"] or int(item["episode_seed"]) != int(row["episode_seed"]):
            raise ValueError("Cached geometry/seed differs from definition: " + row["test_id"])
        digests[row["test_id"]] = digest
        cases.append(dict(case="TS4:" + row["test_id"], scenario_id=row["test_id"], dataset="ts4",
                          seed=int(row["episode_seed"]), scenario_sha256=digest,
                          selection_stratum="|".join(row[k] for k in ("source", "cell", "variant")),
                          source=row["source"], cell=row["cell"], encounter=row["class"],
                          variant=row["variant"], leg=row["leg"], panels=int(row["panels"])))
    digest = sha(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode())
    if digest != definition["manifest_digest"] or digest != cache_report["manifest_digest"]:
        raise ValueError("Actual cached geometry does not match the frozen manifest digest")
    selection = dict(
        schema=1, tag="testset_v4_main", suite_version="4.0", evaluation_role="primary_test_set_v4",
        scenarios=1000,
        scope=("Evaluation on the entire frozen test set v4; no safety tuning on these outcomes. "
               "Overlap with v2/v3 development exposure is reported separately; not a wholly unseen set."),
        authorization="User, 2026-10-10: testing authorized on development data and test set v4.",
        ranking="All 1,000 definition rows in their original order; no outcome-based selection.",
        definition_digest=digest, input_sha256=source, cases=cases)
    overlap = {}
    for name, path in (("v2", ROOT / "results/test_set/v2/definition.csv"),
                       ("v3", ROOT / "results/test_set/v3/definition.csv")):
        previous = {r["test_id"]: r for r in read_rows(path)}
        overlap[name] = sum(1 for r in rows if r["test_id"] in previous and all(
            r[k] == previous[r["test_id"]][k] for k in ("digest", "episode_seed")))
    baseline_path = suite / "sacs0_bl3/episodes.csv"
    baseline = read_rows(baseline_path)
    inventory = dict(
        schema=1, suite_version="4.0", definition_digest=digest, episodes=1000,
        source_counts=dict(Counter(r["source"] for r in rows)),
        variant_counts=dict(Counter(r["variant"] for r in rows)),
        shared_geometry_and_seed=overlap, replaced_from_v3=len(definition.get("replaced", [])),
        historical_sac=dict(path=baseline_path.relative_to(ROOT).as_posix(),
                            sha256=sha(baseline_path.read_bytes()),
                            outcomes=dict(Counter(r["outcome"] for r in baseline)),
                            evidence=("Historical saved CSV reusing v3 rows for 933 shared episodes; no frozen "
                                      "source/checkpoint provenance. Fresh matched OFF is the pairing reference.")),
        episodes_run_by_preparation=0, primary_selection="results/safety_dev/testset_v4_main/selection.json",
        input_sha256=source)
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / "selection.json", selection)
    freeze(OUT / "inventory.json", inventory)
    print(json.dumps({k: inventory[k] for k in ("episodes", "source_counts", "shared_geometry_and_seed",
                                                 "replaced_from_v3", "episodes_run_by_preparation")}, indent=2))
    print(json.dumps(inventory["historical_sac"]["outcomes"]))


if __name__ == "__main__":
    main()
