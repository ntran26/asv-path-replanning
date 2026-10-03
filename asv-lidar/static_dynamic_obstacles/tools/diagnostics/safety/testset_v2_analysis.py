"""Analyze saved test-set v2 SAC rows and retrieve descriptive success controls.

No environment is constructed or reset, no model is loaded, and no episode is
run. The trusted local pickle supplies scenario metadata only. All 1,000 saved
rows are analyzed; nearest successful controls are a diagnostic selection, not
a representative evaluation subset or a causal estimate.

python -B tools/diagnostics/safety/testset_v2_analysis.py
Use --output NEW_DIRECTORY to repeat without overwriting earlier artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "results/test_set/v2"
DEFAULT_OUTPUT = ROOT / "results/safety_dev/testset_v2_offline/analysis"
OUTCOMES = ("goal", "collision:target", "collision:obstacle", "collision:boundary", "timeout")
GEOMETRY_FIELDS = (
    "nominal_width", "width_ratio", "bend_deg", "path_offset_frac", "slant_realised_deg",
    "target_speed", "speed_ratio", "dcpa_m", "tcpa_s", "spawn_range_m", "ct_deg", "w_eff_at_cpa",
    "own_start_s_override", "obstacle_count_for_matching", "fixed_obstacle_count",
    "own_spawn_x", "own_spawn_y", "own_heading_sin", "own_heading_cos",
    "path_midpoint_x", "path_midpoint_y", "target_spawn_x", "target_spawn_y",
    "target_heading_sin", "target_heading_cos",
)
RETROSPECTIVE = ("steps", "mean_speed", "max_speed", "min_target_range", "rms_cte",
                 "engaged_frames", "colregs_integral", "p_port_sense")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def normalized_class(value):
    # pandas.read_csv treats the literal class "null" as NA when reusing v1.
    return "null" if value in (None, "", "null") else value


def validate_rows(rows, definitions, metadata):
    expected = int(metadata["episodes"])
    if len(rows) != expected or len(definitions) != expected:
        raise ValueError("Result/definition count differs from declared episode count")
    by_id = {row["test_id"]: row for row in definitions}
    if len(by_id) != expected or len({r["test_id"] for r in rows}) != expected:
        raise ValueError("Duplicate result or definition identity")
    if {r["test_id"] for r in rows} != set(by_id):
        raise ValueError("Result/definition identity mismatch")
    manifest = {r["test_id"]: r["digest"] for r in definitions}
    actual_digest = sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    if actual_digest != metadata["manifest_digest"]:
        raise ValueError("Definition digest mismatch")
    for row in rows:
        definition = by_id[row["test_id"]]
        for field in ("origin_id", "source", "cell", "stratum", "variant"):
            if row[field] != definition[field]:
                raise ValueError(f"Result {field} differs from definition: {row['test_id']}")
        if normalized_class(row["class"]) != normalized_class(definition["class"]):
            raise ValueError("Result encounter class differs from definition")
        if row["safety"] != "off" or row["policy"] != "model" or row["outcome"] not in OUTCOMES:
            raise ValueError("Expected valid unfiltered model outcomes only")
        for field in ("estops", "safety_v2_steps", "safety_v2_brake_steps"):
            if number(row[field]) != 0:
                raise ValueError("Unfiltered reference contains intervention counters")
        for field, expected_flag in (("collided", row["outcome"].startswith("collision:")),
                                     ("collided_target", row["outcome"] == "collision:target")):
            if row[field].lower() != str(expected_flag).lower():
                raise ValueError(f"Inconsistent outcome flag: {field}")
    return by_id


def scenario_features(built):
    flags = built.flags or {}
    features = {key: number(getattr(built, key, None)) for key in GEOMETRY_FIELDS}
    features["own_start_s_override"] = number(flags.get("own_start_s"))
    features["requested_obstacle_count"] = number(built.n_obstacles)
    fixed = flags.get("fixed_obstacles")
    features["fixed_obstacle_count"] = float(len(fixed)) if fixed is not None else None
    features["obstacle_count_for_matching"] = (float(len(fixed)) if fixed is not None else number(built.n_obstacles))
    for name in ("own_spawn", "target_spawn", "path_midpoint"):
        pair = getattr(built, name, None)
        for i, suffix in enumerate(("x", "y")):
            features[name + "_" + suffix] = number(pair[i]) if pair is not None else None
    for name in ("own_heading", "target_heading"):
        angle = number(getattr(built, name, None))
        features[name + "_sin"] = math.sin(math.radians(angle)) if angle is not None else None
        features[name + "_cos"] = math.cos(math.radians(angle)) if angle is not None else None
    return features


def attach_scenarios(rows, definitions, items, report, metadata):
    if report != metadata or len(items) != len(rows):
        raise ValueError("Cached definition report differs from saved manifest")
    cached = {item["test_id"]: item for item in items}
    if len(cached) != len(items) or set(cached) != set(definitions):
        raise ValueError("Cached scenario identities differ from definitions")
    enriched = []
    for row in rows:
        item = cached[row["test_id"]]
        built = item["built"]
        definition = definitions[row["test_id"]]
        if str(item["episode_seed"]) != definition["episode_seed"] or built.digest() != definition["digest"]:
            raise ValueError("Cached seed/scenario digest differs from definition")
        enriched.append(dict(row, **scenario_features(built), episode_seed=int(item["episode_seed"]),
                             scenario_sha256=definition["digest"], scenario_class=normalized_class(row["class"])))
    return enriched


def group_counts(rows):
    result = []
    groupings = [(), ("source",), ("stratum",), ("scenario_class",), ("source", "cell"),
                 ("source", "variant"), ("source", "scenario_class")]
    for fields in groupings:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[f] for f in fields)].append(row)
        for values, subset in sorted(groups.items()):
            counts = Counter(row["outcome"] for row in subset)
            n = len(subset)
            result.append({"group_by": "+".join(fields) or "overall", "group": " / ".join(values) or "all",
                "n": n, "goals": counts["goal"], "failures": n - counts["goal"],
                "target": counts["collision:target"], "obstacle": counts["collision:obstacle"],
                "boundary": counts["collision:boundary"], "timeouts": counts["timeout"],
                "goal_rate": counts["goal"] / n, "failure_rate": 1 - counts["goal"] / n})
    return result


def match_successes(rows, count=2):
    """Nearest controls within exact cell AND variant; scale from all same-cell rows.

    Local methodological precedent: tools/tiers/test_set.py::_features (line93)
    and ::_trim (line155), which standardize scenario geometry and compare
    Euclidean distances. Here the task is diagnostic retrieval, not set trimming.
    Only coordinates observed for every scenario in the cell are used. Constant
    coordinates are discarded. Distances are Euclidean in units of that cell's
    population SD. Outcome is used only to choose failures and eligible controls,
    never as a distance coordinate. Controls may be reused and are not twins.
    """
    cells = defaultdict(list)
    for row in rows:
        cells[(row["source"], row["cell"])].append(row)
    matches, scaling = [], {}
    for (source, cell), subset in sorted(cells.items()):
        scales = {}
        for name in GEOMETRY_FIELDS:
            values = [r.get(name) for r in subset]
            if all(v is not None and math.isfinite(v) for v in values):
                sd = statistics.pstdev(values)
                if sd > 1e-9:
                    scales[name] = sd
        scaling[source + "/" + cell] = scales
        for failed in sorted((r for r in subset if r["outcome"] != "goal"), key=lambda r: r["test_id"]):
            controls = [r for r in subset if r["outcome"] == "goal" and r["variant"] == failed["variant"]]
            candidates = [(math.sqrt(sum(((r[k] - failed[k]) / sd) ** 2 for k, sd in scales.items())),
                           r["test_id"], r) for r in controls] if scales else []
            for rank, (distance, _, control) in enumerate(sorted(candidates)[:count], 1):
                matches.append({"failure_id": failed["test_id"], "failure_outcome": failed["outcome"],
                    "source": source, "cell": cell, "variant": failed["variant"], "rank": rank,
                    "control_id": control["test_id"], "distance": distance, "features_used": len(scales),
                    "failure_seed": failed["episode_seed"], "control_seed": control["episode_seed"],
                    "failure_scenario_sha256": failed["scenario_sha256"],
                    "control_scenario_sha256": control["scenario_sha256"], "unmatched_reason": ""})
            if not candidates:
                matches.append({"failure_id": failed["test_id"], "failure_outcome": failed["outcome"],
                    "source": source, "cell": cell, "variant": failed["variant"], "rank": 0,
                    "control_id": "", "distance": None, "features_used": len(scales),
                    "failure_seed": failed["episode_seed"], "control_seed": None,
                    "failure_scenario_sha256": failed["scenario_sha256"], "control_scenario_sha256": "",
                    "unmatched_reason": "no successful same-cell/variant case" if not controls else "no varying shared features"})
    return matches, scaling


def retrospective_stats(rows):
    result = {}
    for outcome in OUTCOMES:
        subset = [r for r in rows if r["outcome"] == outcome]
        if subset:
            result[outcome] = {"n": len(subset), "median": {}}
            for key in RETROSPECTIVE:
                values = [x for r in subset if (x := number(r.get(key))) is not None]
                result[outcome]["median"][key] = {"value": statistics.median(values) if values else None,
                                                "finite_n": len(values)}
    return result


def prior_reuse(rows, input_root):
    path = input_root.parent / "sacs0_bl3/episodes.csv"
    if not path.exists():
        return {"available": False}
    previous = {r["test_id"]: r for r in read_csv(path)}
    shared, substantive = 0, Counter()
    # Compare the original episode fields, not the newly attached static features.
    for row in rows:
        if row["test_id"] not in previous:
            continue
        shared += 1
        for key, value in row.items():
            old = previous[row["test_id"]].get(key)
            if key == "class":
                equal = normalized_class(value) == normalized_class(old)
            else:
                a, b = number(value), number(old)
                equal = value == old or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))
            if not equal:
                substantive[key] += 1
    return {"available": True, "path": str(path), "sha256": sha(path.read_bytes()), "shared_ids": shared,
            "numerical_tolerance": 1e-12, "substantive_difference_counts": dict(substantive),
            "new_ids": [r["test_id"] for r in rows if r["test_id"] not in previous]}


def csv_write(path, rows, fields=None):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary):
    total = next(r for r in summary["groups"] if r["group_by"] == "overall")
    groups = {(r["group_by"], r["group"]): r for r in summary["groups"]}
    field, crossing = groups[("source", "paper2")], groups[("scenario_class", "crossing")]
    l2 = groups[("source+cell", "paper2 / L2-HO")]
    fixed, varying = groups[("source+variant", "paper2 / FIX")], groups[("source+variant", "paper2 / VAR")]
    reuse = summary["prior_v1_reuse"]
    reuse_text = (f"The {reuse['shared_ids']} shared v1 results agree numerically within 1e-12 after "
                  "normalizing the blank/null class." if reuse.get("available") and not reuse["substantive_difference_counts"]
                  else "Shared v1 result equivalence was not established; inspect summary.json.")
    lines = ["# Saved SAC test-set v2 analysis", "",
        f"SAC baseline 3 (3M checkpoint): **{total['goals']}/{total['n']} goals ({100 * total['goal_rate']:.1f}%)**, "
        f"with {total['target']} target, {total['obstacle']} obstacle and {total['boundary']} boundary collisions; "
        f"{total['timeouts']} timeouts. All saved rows have safety off and zero intervention counters.", "",
        "This report reads saved data only. No simulator reset, policy inference or new episode was performed. "
        "The user has authorized development on this set; any resulting tuning must be described as development, "
        "not an untouched test of the tuned filter.", "", "## Provenance and limits", "",
        "The definition, cached scenario digests, seeds and 1,000 unique result IDs agree. "
        f"Manifest SHA256: `{summary['manifest_digest']}`. The recorded checkpoint path is "
        "`runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip`. These results do not archive checkpoint, "
        "source or runtime hashes, so the path alone cannot establish exact current-runtime equivalence.", "",
        "The set combines simulated frozen-suite scenarios and simulated deployment layouts. It is not a "
        "real-world vessel trial. Variant twins were collapsed, then geometrically near-duplicate cases removed "
        "within cells: 2,330 source runs became 1,103 underlying scenarios, then 1,000. These are mixed, "
        "deliberately selected cases; no population weighting, confidence interval or causal effect is inferred.", "",
        "Version 2 replaced 15 L1 being-overtaken cases that began in obstacle contact, using a corrected start "
        "and a 0.30 m panel-clearance requirement for regenerated cases. " + reuse_text + " "
        "The evaluation script reuses matching v1 rows; its recorded runtime must not be interpreted as "
        "timing for 1,000 fresh simulations.", "", "## Outcome groups", "",
        "Every rate uses the stated row count, including all collision types and timeouts in its denominator.", "",
        "| Group | N | Goals | Target | Obstacle | Boundary | Timeout | Goal rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        if group["group_by"] in ("overall", "source", "scenario_class"):
            lines.append(f"| {group['group_by']}: {group['group']} | {group['n']} | {group['goals']} | "
                         f"{group['target']} | {group['obstacle']} | {group['boundary']} | {group['timeouts']} | {group['goal_rate']:.1%} |")
    lines += ["", f"The simulated layout subset has {field['failures']} of the {total['failures']} failures despite "
        f"containing {field['n']} of {total['n']} cases. Crossing cases account for {crossing['failures']} failures "
        f"and {crossing['target']} of the {total['target']} target contacts. L2 head-on has {l2['goals']}/{l2['n']} "
        f"goals: {l2['obstacle']} obstacle contacts and {l2['boundary']} boundary contacts. "
        "This is a geometry/encounter concentration worth investigating, "
        "not evidence that a target-directed turn caused each collision. See groups.csv for all cells and variants.", "",
        f"Paper2 FIX has {fixed['goals']}/{fixed['n']} goals; VAR has {varying['goals']}/{varying['n']}. "
        "These are different retained scenarios, not matched twins; "
        "the difference cannot be attributed to speed variation alone. Nominal width also does not measure the "
        "available gap between panels.", "", "## Available features and leakage", "",
        "`scenario_features.csv` contains all 1,000 cases, with nominal pose/path geometry, width, target spawn, "
        "speed, DCPA/TCPA, crossing angle and obstacle-count metadata. Scenario truth is suitable for offline "
        "stratification; it is not automatically available to an onboard filter. Own spawn is the scenario's "
        "nominal start; `own_start_s_override` records an explicit reset-path offset when present.", "",
        "Fixed deployment panels are cached in `flags.fixed_obstacles`, not `built.obstacles`. Frozen-suite "
        "obstacle placements are generated at reset and are absent from these saved rows/cache; "
        "`requested_obstacle_count` is not a reconstruction of their actual positions. Matching uses fixed "
        "panel count when present, otherwise the requested count. Thus these controls "
        "match recorded geometry, not exact obstacle layouts.", "",
        "Saved episode summaries include steps, mean/max speed, minimum target range, RMS cross-track error "
        "and COLREG/encounter totals. They are retrospective and depend on episode duration and termination. "
        "Do not use them, final outcomes or case IDs as online trigger features. Median summaries with finite "
        "sample counts are in summary.json; their differences are descriptive, not predictive validation.", "",
        "There are no saved action sequences, intermediate poses, LiDAR rays, track histories or safety candidate "
        "rollouts in this dataset directory. Therefore this report cannot identify the first harmful intervention, "
        "replay an unfiltered policy's future behavior, or prove an override unnecessary.", "", "## Future regression controls (not executed)", "",
        f"For each failure, up to two successful controls are retrieved within its exact cell and variant using "
        f"Euclidean distance on variable, fully observed static coordinates standardized by the same cell's "
        f"population SD. {summary['matching']['matched_failures']} failures have controls; "
        f"{summary['matching']['unmatched_failures']} have none; {summary['matching']['unique_controls']} unique successes "
        f"appear in {summary['matching']['pairs']} pairs. L2-HO has no same-cell successful case. Distance is a "
        "retrieval aid, not a propensity score or a causal match; outcomes choose the two groups but never "
        "enter distance calculations. Controls can recur across failures. Scales and coordinates are frozen "
        "in summary.json, and ties resolve by test ID. The local methodological precedent is "
        f"[the test-set feature construction](<{(ROOT / 'tools/tiers/test_set.py').as_posix()}:93>) "
        f"and [standardized geometric distance](<{(ROOT / 'tools/tiers/test_set.py').as_posix()}:155>); "
        "this report adapts them for success-control retrieval with exact cell/variant eligibility, "
        "rather than claiming a published causal matching method.", "",
        "`failures.csv` inventories all failures; `matched_controls.csv` lists every pair and unmatched reason; "
        "`successful_controls.csv` preserves the unique successful cases, seeds and scenario digests. "
        "This selected control list does not cover all 872 successes and must not replace all-set regression accounting.", "",
        "## Observable hypotheses to test before changing intervention rules", "",
        "1. Distinguish imminent collision under the currently issued policy action from failure to find an "
        "8 s backup. Candidate signals are short-horizon hull clearance and predicted contact time; the saved "
        "episode aggregates cannot establish an appropriate threshold.",
        "2. Check whether existing policy steering is already reducing risk: recent measured yaw/lateral motion, "
        "relative bearing/closing-rate trends, and local side clearance. A successful action may look unsafe "
        "when held open loop; a genuine near-term collision still needs intervention.",
        "3. Separate target-threat evidence from perception uncertainty: track hits/misses, fresh motion evidence, "
        "fit validity and innovation. Preserve static panel and boundary checks when target evidence is weak; "
        "neither missing detections nor increasing centroid range proves safety.",
        "4. Log whether recovery continuation wins solely through its preference bonus despite an admissible "
        "policy candidate. That event is observable inside the filter and directly tests unnecessary override "
        "logic without using case labels or future success.", "",
        "These are untested engineering hypotheses, not implemented methods or causal findings. This report "
        "uses only counts, rates, medians and an explicitly defined descriptive distance; it makes no paper-derived "
        "safety guarantee or statistical-significance claim.", ""]
    return "\n".join(lines)


def analyze(input_root, output):
    if output.exists():
        raise FileExistsError("Choose a new output directory; saved reports are never overwritten")
    names = ["definition.json", "definition.csv", "set_v2.0.pkl", "sacs0_bl3/episodes.csv", "sacs0_bl3/summary.txt"]
    raw = {name: (input_root / name).read_bytes() for name in names}
    metadata = json.loads(raw["definition.json"])
    if metadata["version"] != "2.0" or metadata["episodes"] != 1000:
        raise ValueError("This report expects the complete 1,000-case v2 definition")
    rows = read_csv(input_root / "sacs0_bl3/episodes.csv")
    definitions = validate_rows(rows, read_csv(input_root / "definition.csv"), metadata)
    sys.path.insert(0, str(ROOT / "src"))
    items, report = pickle.loads(raw["set_v2.0.pkl"])
    enriched = attach_scenarios(rows, definitions, items, report, metadata)
    matches, scaling = match_successes(enriched)
    controls = {m["control_id"] for m in matches if m["control_id"]}
    summary = {"schema": 1, "scope": "all saved test-set v2 SAC off rows; no new runs",
        "manifest_digest": metadata["manifest_digest"], "groups": group_counts(enriched),
        "retrospective_statistics_not_online_features": retrospective_stats(rows),
        "prior_v1_reuse": prior_reuse(rows, input_root),
        "matching": {"coordinates": list(GEOMETRY_FIELDS), "scales_by_source_cell": scaling,
                     "pairs": sum(bool(m["control_id"]) for m in matches), "unique_controls": len(controls),
                     "matched_failures": len({m["failure_id"] for m in matches if m["control_id"]}),
                     "unmatched_failures": sum(not m["control_id"] for m in matches),
                     "strict_same_cell_and_variant": True, "maximum_per_failure": 2,
                     "duplicate_controls_allowed": True, "causal_estimate": False},
        "trajectory_records_available": False}
    # Pin every input used before emitting derived output.
    for name, content in raw.items():
        if (input_root / name).read_bytes() != content:
            raise ValueError(f"Input changed while analyzing: {name}")
    output.mkdir(parents=True, exist_ok=False)
    fields = ["test_id", "origin_id", "source", "cell", "variant", "scenario_class", "outcome",
              "episode_seed", "scenario_sha256", "requested_obstacle_count"] + list(GEOMETRY_FIELDS)
    features = [{name: r[name] for name in fields} for r in enriched]
    csv_write(output / "scenario_features.csv", features)
    csv_write(output / "groups.csv", summary["groups"])
    csv_write(output / "failures.csv", [r for r in features if r["outcome"] != "goal"])
    csv_write(output / "matched_controls.csv", matches)
    csv_write(output / "successful_controls.csv", [r for r in features if r["test_id"] in controls])
    provenance = {"analysis_script_sha256": sha(Path(__file__).read_bytes()),
        "input_directory": str(input_root.resolve()), "input_sha256": {k: sha(v) for k, v in raw.items()},
        "new_episode_runs": 0, "environment_construction_or_reset": False,
        "recorded_checkpoint_path_only": "runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip",
        "historical_checkpoint_source_runtime_hashes_available": False,
        "classification_normalization": "75 frozen null classes serialized as blank by pandas; matched as null"}
    for name, value in (("summary.json", summary), ("provenance.json", provenance)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown(summary))
    print(json.dumps({"output": str(output), "rows": len(rows), "failures": sum(r['outcome'] != 'goal' for r in rows),
                      "matched_failures": summary["matching"]["matched_failures"],
                      "controls": len(controls), "new_runs": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
