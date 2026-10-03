"""Saved-source audit only; no simulator import, reset, inference or episode."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tools/diagnostics/safety"))
from suite_status import read_shared_bytes

OUT = Path(__file__).resolve().parent
INPUTS = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = read_shared_bytes(path)
    INPUTS[path.relative_to(ROOT).as_posix()] = sha(raw)
    return raw


def read_json(path):
    return json.loads(read(path))


def archive(path, hashes):
    with zipfile.ZipFile(io.BytesIO(read(path))) as z:
        assert len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(hashes)
        sources = {name: z.read(name) for name in z.namelist()}
        assert all(sha(sources[name]) == expected for name, expected in hashes.items())
        return sources


def main():
    base = ROOT / "results/safety_dev/v10_iterations"
    tags = ("prefix_success16", "prefix_remaining16")
    manifests = [read_json(base / tag / "manifest.json") for tag in tags]
    sources = [archive(base / tag / "evaluated_sources.zip", m["source_sha256"]) for tag, m in zip(tags, manifests)]
    old, new = manifests
    common = set(sources[0]) & set(sources[1])
    changed = [name for name in sorted(common) if sources[0][name] != sources[1][name]]
    added = sorted(set(sources[1]) - set(sources[0]))
    removed = sorted(set(sources[0]) - set(sources[1]))
    assert not changed and not removed
    assert added == ["src/dev_set_v4.py", "src/safety_v14.py"]
    checked_settings = ("modes", "filter_classes", "constructor_options", "checkpoint_sha256", "config_sha256",
                        "constants", "low_speed_start_frac", "filter_installation", "torch_threads", "native_thread_environment")
    assert all(old[key] == new[key] for key in checked_settings)
    references = {term: [name for name in sorted(common) if term in sources[1][name].decode("utf-8-sig")]
                  for term in ("dev_set_v4", "safety_v14", "SafetyFilterV14")}
    assert not any(references.values())

    prior = read_json(ROOT / "results/safety_dev/v9_paired_pilot/selection.json")
    expected = {row["case"]: row for row in prior["cases"]}
    left, right = ({row["case"]: row for row in m["cases"]} for m in manifests)
    assert len(left) == len(right) == 16 and not set(left) & set(right)
    joined = dict(left, **right)
    assert set(joined) == set(expected) and len(joined) == 32
    canonical = {}
    for row in csv.DictReader(io.StringIO(read(ROOT / "results/test_set/v2/definition.csv").decode("utf-8"))):
        canonical["TS2:" + row["test_id"]] = (int(row["episode_seed"]), row["digest"])
    inventory = read_json(ROOT / "results/safety_dev/suites/inventory_v5/manifest.json")
    for row in inventory["cases"]:
        if row["suite"] == "dev_field":
            canonical["DV3:" + row["case"]] = (row["seed"], row["scenario_sha256"])
    for case, row in joined.items():
        assert all(row[key] == expected[case][key] for key in ("case", "dataset", "seed", "scenario_sha256"))
        assert (row["seed"], row["scenario_sha256"]) == canonical[case]
    for m in manifests:
        raw = read(Path(m["selection_source"]))
        assert sha(raw) == m["selection_sha256"]

    helper_name = "tools/diagnostics/safety/v9_paired_pilot.py"
    helper = sources[1][helper_name].decode("utf-8-sig")
    function = next(n for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef) and n.name == "load_scenes")
    loader_source = ast.get_source_segment(helper, function)
    assert "pickle.loads" in loader_source and "candidate.digest()" in loader_source
    assert "Selection seed/geometry mismatch" in loader_source and "refusing regeneration" in loader_source
    generator = sources[1]["src/dev_set_v4.py"].decode("utf-8-sig")
    parsed = ast.parse(generator)
    functions = [n.name for n in parsed.body if isinstance(n, ast.FunctionDef)]
    assert functions == ["extension"]
    top_level_calls = [ast.unparse(node) for statement in parsed.body if not isinstance(statement, ast.FunctionDef)
                       for node in ast.walk(statement) if isinstance(node, ast.Call)]
    assert not top_level_calls
    result = {
        "scope": "Source-inventory evidence for two disjoint prefix slices. This audit does not waive the report helper's strict cohort guard or create results.",
        "tags": list(tags), "case_counts": [16, 16], "combined_distinct_cases": 32,
        "all_case_seeds_and_scenario_digests_match_canonical_saved_inventory": True,
        "all_cases_exactly_cover_prior_fixed32": True,
        "identical_setting_keys": list(checked_settings), "common_sources_identical": len(common),
        "changed_common_sources": changed, "removed_sources": removed,
        "added_sources": {name: {"sha256": sha(sources[1][name])} for name in added},
        "literal_references_in_common_archived_sources": references,
        "cached_loader_source": {"file": helper_name, "sha256": sha(sources[1][helper_name]),
                                 "first_line": function.lineno, "source": loader_source},
        "generator_inspection": {"function": "extension", "top_level_calls": top_level_calls,
                                 "role": "Defines a separate60-case DV4X generator; generation occurs inside extension(), not while the cached loader reads source archives."},
        "reasoning": [
            "The archived runner resolves prebuilt TS2 records from set_v2.0.pkl and DV3 from digest-matching cache files, then checks reset seeds and built.digest before creating its manifest.",
            "The unchanged load_scenes function contains no fallback generation path; a missing DV3 cache raises refusing regeneration.",
            "All99 shared archived Python files are byte-identical and contain no literal module/class reference to either added module.",
            "The selected class/options remain SafetyFilterV11 with track admission/persistence disabled and policy-prefix search enabled.",
            "These checks support treating the extra generator as unused by these cached-scenario runs, while explicitly retaining the unequal source inventories.",
        ],
        "limitations": [
            "A literal reference scan is not a general dynamic-import proof; no runtime import tracing or new simulations were performed.",
            "This is not a completed-outcome report; each component still requires its own final completion/result/trace validation.",
            "The strict report cohort API continues to reject dev_set_v4.py; combined outcomes must be labeled as an explicitly disclosed sum of separate validated pieces.",
        ],
        "input_sha256": INPUTS, "audit_script_sha256": sha(Path(__file__).read_bytes()),
        "new_episode_runs": 0,
    }
    with (OUT / "audit.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    md = "\n".join([
        "# Prefix32 source-inventory audit", "",
        "The two16-case selections are disjoint and exactly cover the original32 pilot identities. Every saved seed and scenario digest matches the canonical inventory.", "",
        f"All{len(common)} shared archived source files are byte-identical. The later archive adds `src/safety_v14.py` and `src/dev_set_v4.py`; the inventories are explicitly different.", "",
        "No common archived source contains a literal reference to either added module or SafetyFilterV14. The added generator defines a separate60-case DV4X extension. The archived runner loads prebuilt TS2/DV3 cache entries and validates their seeds/digests; it refuses missing caches instead of regenerating scenarios. The selected class and constructor options are unchanged.", "",
        "This supports the added generator being unused in these cached-scenario runs. It is not a general dynamic-import proof. No episodes, resets, generation, or policy inference were run for this audit.", "",
        "The strict cohort API remains unchanged and rejects the extra generator file. Report both completed pieces separately, then label any combined32-case table as their explicitly disclosed sum. Source checks do not replace completion/result/trace checks.", "",
        "[Exact hashes, shared-reference scan and archived cache-loader source](audit.json). Reproduce into a new directory or preserve/remove only these generated outputs before rerunning `build_audit.py`; the builder refuses overwrite.", "",
    ])
    # Spaces are explicit so the concise report remains readable.
    for old_text, new_text in (("two16", "two 16"), ("original32", "original 32"), ("All99", "All 99"), ("separate60", "separate 60"), ("combined32", "combined 32")):
        md = md.replace(old_text, new_text)
    with (OUT / "audit.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(md)
    print(json.dumps({"directory": str(OUT), "common_sources": len(common), "added": added, "cases": 32, "new_episodes": 0}))


if __name__ == "__main__":
    main()
