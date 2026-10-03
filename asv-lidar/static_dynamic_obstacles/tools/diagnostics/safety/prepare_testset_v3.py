"""Freeze the complete main safety benchmark from existing test-set-v3 files.

Read-only with respect to the suite, policy and simulator. No episode, reset,
scenario generation or controller evaluation occurs. Existing output files
must match exactly; a changed definition requires a separately named protocol.
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
OUT = ROOT / "results/safety_dev/testset_v3_main"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite a different frozen artifact: {path}")
    else:
        path.write_bytes(raw)


def main():
    suite = ROOT / "results/test_set/v3"
    paths = [suite / name for name in ("definition.json", "definition.csv", "set_v3.0.pkl")]
    source = {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in paths}
    definition = json.loads(paths[0].read_text())
    with paths[1].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Trusted, existing project cache only; never regenerate the benchmark.
    kept, cache_report = pickle.loads(paths[2].read_bytes())
    by_id = {row["test_id"]: row for row in rows}
    cached = {row["test_id"]: row for row in kept}
    if (definition["version"] != "3.0" or len(rows) != 1000 or len(kept) != 1000
            or len(by_id) != 1000 or len(cached) != 1000 or by_id.keys() != cached.keys()):
        raise ValueError("V3 must contain exactly the same 1,000 unique IDs in CSV and cache")
    digests, cases = {}, []
    for row in rows:
        item = cached[row["test_id"]]
        digest = item["built"].digest()
        if digest != row["digest"] or int(item["episode_seed"]) != int(row["episode_seed"]):
            raise ValueError("Cached geometry/seed differs from definition: " + row["test_id"])
        digests[row["test_id"]] = digest
        cases.append(dict(case="TS3:" + row["test_id"], scenario_id=row["test_id"], dataset="ts3",
                          seed=int(row["episode_seed"]), scenario_sha256=digest,
                          selection_stratum="|".join(row[k] for k in ("source", "cell", "variant")),
                          source=row["source"], cell=row["cell"], encounter=row["class"],
                          variant=row["variant"], leg=row["leg"], panels=int(row["panels"])))
    digest = sha(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode())
    if digest != definition["manifest_digest"] or digest != cache_report["manifest_digest"]:
        raise ValueError("Actual cached geometry does not match the frozen manifest digest")
    selection = dict(schema=1, tag="testset_v3_main", suite_version="3.0",
                     evaluation_role="primary", scenarios=1000,
                     scope="Primary evaluation on the entire frozen test set v3; no safety tuning on these outcomes. Overlap with prior v2 development is reported separately; this is not a wholly unseen set.",
                     authorization="User: Test set v3 is now the main set to be evaluated on.",
                     ranking="All 1,000 definition rows in their original order; no outcome-based selection.",
                     definition_digest=digest, input_sha256=source, cases=cases)
    baseline_path = suite / "sacs0_bl3/episodes.csv"
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    if (len(baseline) != 1000 or {r["test_id"] for r in baseline} != by_id.keys()
            or len({r["test_id"] for r in baseline}) != 1000
            or any(r["safety"] != "off" for r in baseline)):
        raise ValueError("Historical SAC rows do not cover the same 1,000 unique OFF cases")
    with (ROOT / "results/test_set/v2/definition.csv").open(newline="", encoding="utf-8") as handle:
        previous = {r["test_id"]: r for r in csv.DictReader(handle)}
    shared = [r["test_id"] for r in rows if r["test_id"] in previous
              and all(r[k] == previous[r["test_id"]][k] for k in ("digest", "episode_seed"))]
    inventory = dict(schema=1, suite_version="3.0", definition_digest=digest, episodes=1000,
                     source_counts=dict(Counter(r["source"] for r in rows)),
                     shared_v2_geometry_and_seed=len(shared), new_or_changed_since_v2=1000-len(shared),
                     historical_sac=dict(path=baseline_path.relative_to(ROOT).as_posix(),
                         sha256=sha(baseline_path.read_bytes()), outcomes=dict(Counter(r["outcome"] for r in baseline)),
                         evidence="Historical saved CSV; old run has no frozen checkpoint/config/source provenance. Fresh matched OFF is required for the next candidate comparison."),
                     recommended_modes=["off", "v16"], episodes_run_by_preparation=0,
                     primary_selection="results/safety_dev/testset_v3_main/selection.json",
                     input_sha256=source)
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / "selection.json", selection)
    freeze(OUT / "inventory.json", inventory)
    print(json.dumps({k: inventory[k] for k in ("episodes", "source_counts", "shared_v2_geometry_and_seed", "new_or_changed_since_v2", "episodes_run_by_preparation")}, indent=2))


if __name__ == "__main__":
    main()
