"""Aggregate suite_eval journals without changing evaluation artifacts.

python -B tools/diagnostics/safety/suite_compare.py results/safety_dev/suites/frozen_final results/safety_dev/suites/other_final --tag final_suites

Complete declared runs are required unless --allow-partial is explicit. Pairing
uses (suite component, case), never a case ID alone. Reports show available
pairs and missing counterparts, including when a reference mode was not run.
Use --require-full-benchmark for the final 8,670-record simulation report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from suite_status import read_shared_bytes

ROOT = Path(__file__).resolve().parents[3]
OUTCOMES = ("goal", "collision:obstacle", "collision:boundary", "collision:target", "timeout")
GROUPS = {"dev_field": "dev", "dev_legacy": "dev", "dev_width": "dev",
          "frozen_a": "frozen", "frozen_b": "frozen", "frozen_r": "frozen",
          "field_deployment": "field", "field_validation": "field"}
FULL_BENCHMARK_MODES = ("off", "v4", "v5")
FULL_BENCHMARK_COUNTS = {"dev_field": 150, "dev_legacy": 120, "dev_width": 100,
                       "frozen_a": 35, "frozen_b": 800, "frozen_r": 900,
                       "field_deployment": 630, "field_validation": 155}
PROVENANCE_KEYS = ("checkpoint_sha256", "config_sha256", "source_sha256",
                   "effective_constants", "runtime_versions", "threads", "processes")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_runs(directories, allow_partial=False):
    """Read a consistent snapshot of committed lines; never repair live files."""
    expected, completed, inputs = {}, {}, []
    settings_reference, safety_constants = None, {}
    for directory in map(Path, directories):
        manifest_bytes = read_shared_bytes(directory / "manifest.json")
        manifest = json.loads(manifest_bytes)
        settings = manifest["settings"]
        if settings.get("schema") != 1:
            raise ValueError(f"Unsupported suite manifest schema: {directory}")
        if not settings["modes"] or len(set(settings["modes"])) != len(settings["modes"]):
            raise ValueError(f"Missing or duplicate manifest modes: {directory}")
        provenance = {key: settings[key] for key in PROVENANCE_KEYS}
        if settings_reference is None:
            settings_reference = provenance
        elif provenance != settings_reference:
            changed = [key for key in PROVENANCE_KEYS if provenance[key] != settings_reference[key]]
            raise ValueError(f"Incompatible evaluation provenance ({', '.join(changed)}): {directory}")
        for mode, constants in settings["effective_safety_constants"].items():
            if mode in safety_constants and safety_constants[mode] != constants:
                raise ValueError(f"Incompatible effective safety constants for {mode}: {directory}")
            safety_constants[mode] = constants
        local_expected = {}
        for case in manifest["cases"]:
            if case["suite"] not in GROUPS:
                raise ValueError(f"Unknown suite component {case['suite']}")
            for mode in settings["modes"]:
                key = (mode, case["suite"], case["case"])
                if key in expected or key in local_expected:
                    raise ValueError(f"Duplicate declared mode/component/case: {key}")
                local_expected[key] = case
        if len(local_expected) != manifest["expected_episodes"]:
            raise ValueError(f"Manifest episode count mismatch: {directory}")
        expected.update(local_expected)
        journal = directory / "episodes.jsonl"
        try:
            data = read_shared_bytes(journal)
        except FileNotFoundError:
            data = b""
        tail_bytes, local_completed = 0, 0
        for raw in data.splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                tail_bytes = len(raw)
                if not allow_partial:
                    raise ValueError(f"Uncommitted journal tail: {journal}; use --allow-partial for progress")
                continue
            row = json.loads(raw)
            key = (row["mode"], row["suite"], row["case"])
            if key not in local_expected or key in completed:
                raise ValueError(f"Unknown or duplicate completed mode/component/case: {key}")
            case = local_expected[key]
            if any(row.get(field) != case[field] for field in ("seed", "scenario_sha256")):
                raise ValueError(f"Completed-case seed/digest mismatch: {key}")
            if row.get("outcome") not in OUTCOMES:
                raise ValueError(f"Invalid or unfinished outcome: {key}")
            completed[key] = row
            local_completed += 1
        if not allow_partial and local_completed != len(local_expected):
            raise ValueError(f"Incomplete run {directory}: {local_completed}/{len(local_expected)}; use --allow-partial for progress")
        inputs.append({"directory": str(directory.resolve()), "manifest_sha256": digest(manifest_bytes),
                       "journal_snapshot_sha256": digest(data), "expected": len(local_expected),
                       "completed": local_completed, "ignored_uncommitted_bytes": tail_bytes,
                       "declared_suite_details": manifest.get("suite_details", {})})
    # A seed/scene disagreement must fail even if both outcomes happen to match.
    identity = {}
    for (_, component, case_id), case in expected.items():
        key = (component, case_id)
        value = (case["seed"], case["scenario_sha256"])
        if key in identity and identity[key] != value:
            raise ValueError(f"Cross-mode seed/digest mismatch: {key}")
        identity[key] = value
    return expected, completed, inputs


def benchmark_coverage(expected, completed):
    """Assert the frozen inventory, rather than only the supplied run subset."""
    required = {(mode, component): count for mode in FULL_BENCHMARK_MODES
                for component, count in FULL_BENCHMARK_COUNTS.items()}
    declared = Counter((mode, component) for mode, component, _ in expected)
    committed = Counter((mode, component) for mode, component, _ in completed)
    deficits = [{"mode": key[0], "component": key[1], "required": required.get(key, 0),
                 "declared": declared[key], "completed": committed[key]}
                for key in sorted(required.keys() | declared.keys() | committed.keys())
                if declared[key] != required.get(key, 0) or committed[key] != required.get(key, 0)]
    return {"complete": not deficits, "required_episodes": sum(required.values()),
            "required_modes": list(FULL_BENCHMARK_MODES), "required_components": FULL_BENCHMARK_COUNTS,
            "coverage_mismatches": deficits}


def scopes(expected):
    components = sorted({key[1] for key in expected})
    result = [("component", name, {name}) for name in components]
    result += [("group", group, {name for name in components if GROUPS[name] == group})
               for group in sorted({GROUPS[name] for name in components})]
    return result + [("all", "all", set(components))]


def aggregate(expected, completed):
    outcomes, pairs, comparisons = [], [], []
    modes = sorted({key[0] for key in expected})
    for level, scope, components in scopes(expected):
        for mode in modes:
            declared = {key for key in expected if key[0] == mode and key[1] in components}
            if not declared:
                continue
            available = declared & completed.keys()
            counts = Counter(completed[key]["outcome"] for key in available)
            collision_count = sum(counts[label] for label in OUTCOMES if label.startswith("collision:"))
            outcomes.append({"level": level, "scope": scope, "mode": mode,
                "expected_cases": len(declared), "completed_cases": len(available),
                "complete": len(declared) == len(available), "goals": counts["goal"],
                "collision_obstacle": counts["collision:obstacle"], "collision_boundary": counts["collision:boundary"],
                "collision_target": counts["collision:target"], "collisions": collision_count,
                "timeouts": counts["timeout"], "goal_rate": counts["goal"] / len(available) if available else None,
                "collision_rate": collision_count / len(available) if available else None})
    for candidate in (mode for mode in modes if mode != "off"):
        for reference in ("off", "v4"):
            if candidate == reference:
                continue
            identities = {(suite, case) for mode, suite, case in expected if mode in (candidate, reference)}
            candidate_pairs = []
            for component, case_id in sorted(identities):
                left = completed.get((candidate, component, case_id))
                right = completed.get((reference, component, case_id))
                if left is None or right is None:
                    continue
                a, b = left["outcome"], right["outcome"]
                row = {"component": component, "case": case_id, "test_id": left.get("test_id", ""),
                    "seed": left["seed"], "scenario_sha256": left["scenario_sha256"],
                    "candidate": candidate, "reference": reference,
                    "outcome_candidate": a, "outcome_reference": b,
                    "gained_goal": a == "goal" and b != "goal", "lost_goal": a != "goal" and b == "goal",
                    "rescued_collision": not a.startswith("collision:") and b.startswith("collision:"),
                    "introduced_collision": a.startswith("collision:") and not b.startswith("collision:"),
                    "collision_to_goal": a == "goal" and b.startswith("collision:"),
                    "collision_to_timeout": a == "timeout" and b.startswith("collision:"),
                    "changed_outcome": a != b}
                pairs.append(row)
                candidate_pairs.append(row)
            for level, scope, components in scopes(expected):
                ids = {key for key in identities if key[0] in components}
                if not ids:
                    continue
                available = [row for row in candidate_pairs if row["component"] in components]
                missing_candidate = sum((candidate, *key) not in completed for key in ids)
                missing_reference = sum((reference, *key) not in completed for key in ids)
                metrics = {name: sum(row[name] for row in available) for name in
                           ("gained_goal", "lost_goal", "rescued_collision", "introduced_collision",
                            "collision_to_goal", "collision_to_timeout", "changed_outcome")}
                comparisons.append({"level": level, "scope": scope, "candidate": candidate, "reference": reference,
                    "expected_union_cases": len(ids), "paired_cases": len(available),
                    "missing_candidate": missing_candidate, "missing_reference": missing_reference,
                    "complete_pairs": missing_candidate == missing_reference == 0,
                    "candidate_goals": sum(row["outcome_candidate"] == "goal" for row in available),
                    "reference_goals": sum(row["outcome_reference"] == "goal" for row in available),
                    "candidate_collisions": sum(row["outcome_candidate"].startswith("collision:") for row in available),
                    "reference_collisions": sum(row["outcome_reference"].startswith("collision:") for row in available),
                    "candidate_timeouts": sum(row["outcome_candidate"] == "timeout" for row in available),
                    "reference_timeouts": sum(row["outcome_reference"] == "timeout" for row in available),
                    **metrics, "net_goals": metrics["gained_goal"] - metrics["lost_goal"],
                    "net_collision_reduction": metrics["rescued_collision"] - metrics["introduced_collision"]})
    return outcomes, pairs, comparisons


def write_table(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--tag", required=True, help="new report filename prefix in results/safety_dev")
    parser.add_argument("--allow-partial", action="store_true", help="report committed progress with explicit pair coverage")
    parser.add_argument("--require-full-benchmark", action="store_true",
                        help="require all eight components and off/v4/v5: exactly 8,670 committed records")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in (".", "..") or "\\" in args.tag:
        parser.error("--tag must be a plain filename stem")
    if args.allow_partial and args.require_full_benchmark:
        parser.error("--require-full-benchmark cannot be combined with --allow-partial")
    directory = ROOT / "results" / "safety_dev"
    paths = {name: directory / f"{args.tag}_{name}.{extension}" for name, extension in
             (("outcomes", "csv"), ("paired", "csv"), ("comparison", "csv"), ("sources", "json"))}
    if any(path.exists() for path in paths.values()):
        parser.error("Report already exists; choose a new --tag")
    try:
        expected, completed, inputs = load_runs(args.runs, args.allow_partial)
        coverage = benchmark_coverage(expected, completed)
        if args.require_full_benchmark and not coverage["complete"]:
            raise ValueError("Full benchmark incomplete: require all eight components, off/v4/v5, and exactly 8,670 committed journal records")
        outcomes, pairs, comparisons = aggregate(expected, completed)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"inputs": inputs, "allow_partial": args.allow_partial,
                "completed": len(completed), "expected": len(expected), "complete": len(completed) == len(expected),
                "require_full_benchmark": args.require_full_benchmark, "benchmark_coverage": coverage,
                "definitions": {"rescue": "reference collision becomes candidate goal or timeout",
                    "regression": "reference goal or timeout becomes candidate collision",
                    "rates": "completed cases only; consult coverage before interpreting partial reports",
                    "aggregation": "episode-weighted pooled counts; each robustness variant is a separate paired case, not an independent statistical replicate",
                    "field_sets": "simulated field-layout/validation scenarios; these results are not new real-world trials",
                    "groups": GROUPS}, "report_source_sha256": digest(Path(__file__).read_bytes())}
    with paths["sources"].open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    for name, rows in (("outcomes", outcomes), ("paired", pairs), ("comparison", comparisons)):
        write_table(paths[name], rows)
    print(f"Committed episodes: {len(completed)}/{len(expected)}; supplied_runs_complete={metadata['complete']}; full_benchmark_complete={coverage['complete']}")
    for row in outcomes:
        if row["level"] == "all":
            print(f"{row['mode']}: {row['goals']} goals, {row['collisions']} collisions, {row['timeouts']} timeouts / {row['completed_cases']}")
    print(f"Reports: {directory / args.tag}_*.csv; provenance: {paths['sources']}")


if __name__ == "__main__":
    main()
