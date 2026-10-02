"""Validate and report a completed 72-episode fixed quick comparison; never evaluate.

    python -B tools/diagnostics/safety/report_quick_campaign.py --tag policy_feedback_v1

Requires all 24 matched cases in off/v4/v5 and a matching complete_*.json.
Reads with Windows shared handles and refuses to overwrite reporting artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

from suite_status import read_shared_bytes

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "results/safety_dev/quick_v5_budget100"
MODES = ("off", "v4", "v5")
OUTCOMES = ("goal", "collision:obstacle", "collision:boundary", "collision:target", "timeout")
FLAGS = ("sideslip_rescue_evaluated", "sideslip_rescue_admitted",
         "verified_policy_priority_enabled", "verified_policy_preserved",
         "continuation_override_prevented", "policy_feedback_evaluated", "policy_feedback_preserved")
COMPARISONS = (("v5", "v4"), ("v5", "off"), ("v4", "off"))
COMPONENT_COUNTS = {"frozen_b": 8, "frozen_r": 4, "frozen_a": 2,
                    "field_validation": 4, "field_deployment": 6}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def integer(value):
    return type(value) is int and value >= 0


def validate_records(manifest, metadata, rows, tokens, completions, tag):
    """Independent identity/accounting validation; no controller imports."""
    cases = manifest["cases"]
    if (manifest.get("plan_revision") != 3 or manifest.get("modes") != list(MODES)
            or manifest.get("new_episodes_planned") != 72 or len(cases) != 24
            or Counter(c["suite"] for c in cases) != COMPONENT_COUNTS):
        raise ValueError("Expected the fixed revision 3 24-case/72-episode design")
    expected = {(mode, c["suite"], c["case"]): c for c in cases for mode in MODES}
    if len(expected) != 72 or len(rows) != 72:
        raise ValueError(f"Final report requires exactly 72 unique results; found {len(rows)}")
    if metadata.get("shared_attempt_cap") != 100 or manifest.get("shared_new_attempt_cap") != 100:
        raise ValueError("Shared attempt cap must remain100")
    if not tokens or len(tokens) > 100 or any(type(n) is not int or not 1 <= n <= 100 for n in tokens):
        raise ValueError("Invalid shared attempt ledger")
    metadata_sha = canonical_sha(metadata)
    indexed, used = {}, set()
    for row in rows:
        key = (row["mode"], row["suite"], row["case"])
        if key not in expected or key in indexed:
            raise ValueError(f"Unknown or duplicate quick result: {key}")
        if row.get("tag") != tag or row.get("run_metadata_sha256") != metadata_sha:
            raise ValueError(f"Result settings/tag mismatch: {key}")
        if any(row.get(name) != expected[key][name] for name in ("seed", "scenario_sha256")):
            raise ValueError(f"Result seed/scenario mismatch: {key}")
        if row.get("outcome") not in OUTCOMES:
            raise ValueError(f"Invalid completed outcome: {key}")
        number = row.get("attempt")
        if type(number) is not int or number in used or number not in tokens:
            raise ValueError(f"Missing/reused attempt token: {key}")
        token = tokens[number]
        fields = ("tag", "mode", "suite", "case", "seed", "scenario_sha256", "run_metadata_sha256", "attempt")
        if not isinstance(token, dict) or any(name not in token or token[name] != row.get(name) for name in fields):
            raise ValueError(f"Attempt token disagrees with result: {key}")
        if not isinstance(row.get("seconds"), (int, float)) or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
            raise ValueError(f"Invalid runtime: {key}")
        if not integer(row.get("steps")):
            raise ValueError(f"Invalid episode step count: {key}")
        for name in FLAGS:
            value = row.get(name + "_steps")
            if not integer(value) or value > row["steps"]:
                raise ValueError(f"Invalid passive counter {name}: {key}")
        templates = row.get("recovery_template_counts")
        if not isinstance(templates, dict) or any(not integer(n) for n in templates.values()) or sum(templates.values()) > row["steps"]:
            raise ValueError(f"Invalid recovery template counts: {key}")
        used.add(number)
        indexed[key] = row
    ordered = [indexed[(mode, c["suite"], c["case"])] for c in cases for mode in MODES]
    derived = aggregate(cases, ordered)
    if not completions:
        raise ValueError("No completed-run summary; final report is unavailable")
    for complete in completions:
        if (complete.get("completed") != 72 or complete.get("expected") != 72
                or complete.get("sample_only") is not True or complete.get("population_performance_estimate") is not False
                or complete.get("outcomes") != dict(Counter(r["mode"] + "/" + r["outcome"] for r in rows))):
            raise ValueError("Completed summary disagrees with durable episode results")
        for name, comparison in derived["comparisons"].items():
            saved = complete.get("comparisons", {}).get(name, {})
            if any(saved.get(k) != comparison[k] for k in ("paired_cases", "gained_goals", "lost_goals")):
                raise ValueError(f"Completed paired summary mismatch: {name}")
        for mode in MODES:
            saved = complete.get("mechanism_counts", {}).get(mode, {})
            for name in FLAGS:
                if any(saved.get(name + suffix) != derived["mechanisms"][mode][name + suffix]
                       for suffix in ("_steps", "_episodes")):
                    raise ValueError(f"Completed mechanism summary mismatch: {mode}/{name}")
    return ordered, derived


def aggregate(cases, rows):
    indexed = {(r["mode"], r["suite"], r["case"]): r for r in rows}
    outcomes = []
    for component in ("overall", *COMPONENT_COUNTS):
        for mode in MODES:
            selected = [r for r in rows if r["mode"] == mode and (component == "overall" or r["suite"] == component)]
            counts = Counter(r["outcome"] for r in selected)
            times = [r["seconds"] for r in selected]
            outcomes.append({"suite": component, "mode": mode, "episodes": len(selected),
                **{o: counts[o] for o in OUTCOMES}, "seconds_total": sum(times),
                "seconds_median": statistics.median(times), "seconds_max": max(times)})
    pairs, comparisons = [], {}
    for case in cases:
        matched = {m: indexed[m, case["suite"], case["case"]] for m in MODES}
        pair = {k: case[k] for k in ("suite", "case", "seed", "scenario_sha256", "family", "sampling_stratum")}
        pair.update({k: case.get("features", {}).get(k) for k in
                     ("nominal_width_m", "effective_width_at_cpa_m", "declared_obstacle_count", "target_motion")})
        for mode, row in matched.items():
            pair[mode + "_outcome"] = row["outcome"]
            pair[mode + "_seconds"] = row["seconds"]
        for candidate, reference in COMPARISONS:
            a, b = matched[candidate]["outcome"], matched[reference]["outcome"]
            name = candidate + "_vs_" + reference
            pair[name + "_gained_goal"] = a == "goal" and b != "goal"
            pair[name + "_lost_goal"] = a != "goal" and b == "goal"
        for name in FLAGS:
            pair["v5_" + name + "_steps"] = matched["v5"][name + "_steps"]
        pairs.append(pair)
    for candidate, reference in COMPARISONS:
        name = candidate + "_vs_" + reference
        transitions = Counter((r[reference + "_outcome"], r[candidate + "_outcome"]) for r in pairs)
        comparisons[name] = {"candidate": candidate, "reference": reference, "paired_cases": len(pairs),
            "gained_goals": sum(r[name + "_gained_goal"] for r in pairs),
            "lost_goals": sum(r[name + "_lost_goal"] for r in pairs),
            "collision_to_goal": sum(n for (a, b), n in transitions.items() if a.startswith("collision:") and b == "goal"),
            "collision_to_timeout": sum(n for (a, b), n in transitions.items() if a.startswith("collision:") and b == "timeout"),
            "transitions": [{"reference_outcome": a, "candidate_outcome": b, "cases": n}
                            for (a, b), n in sorted(transitions.items())]}
    mechanisms = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        counts = {}
        for name in FLAGS:
            counts[name + "_steps"] = sum(r[name + "_steps"] for r in selected)
            counts[name + "_episodes"] = sum(r[name + "_steps"] > 0 for r in selected)
        templates = Counter()
        for row in selected:
            templates.update(row["recovery_template_counts"])
        counts["recovery_template_counts"] = dict(templates)
        counts["changed_action_steps"] = sum(r.get("safety_v2_steps", 0) for r in selected)
        counts["changed_action_episodes"] = sum(r.get("safety_v2_steps", 0) > 0 for r in selected)
        mechanisms[mode] = counts
    return {"outcomes": outcomes, "pairs": pairs, "comparisons": comparisons, "mechanisms": mechanisms}


def load_run(tag):
    directory = CAMPAIGN / "runs" / tag
    if (CAMPAIGN / "active_run.lock").exists():
        raise ValueError("Quick campaign is still active; wait for normal completion before reporting")
    inputs = {}
    def read(path):
        raw = read_shared_bytes(path)
        inputs[str(path.relative_to(ROOT))] = sha(raw)
        return json.loads(raw)
    manifest_path = CAMPAIGN / "sample_manifest_v3.json"
    manifest, metadata = read(manifest_path), read(directory / "metadata.json")
    if metadata["sample_manifest_sha256"] != inputs[str(manifest_path.relative_to(ROOT))]:
        raise ValueError("Sample manifest hash changed")
    inventory_path = Path(manifest["inventory_path"])
    inventory = read(inventory_path)
    if inputs[str(inventory_path.relative_to(ROOT))] != manifest["inventory_sha256"]:
        raise ValueError("Canonical inventory hash changed")
    original = {(c["suite"], c["case"]): c for c in inventory["cases"]}
    previous_path = CAMPAIGN / "sample_manifest_v2.json"
    previous = read(previous_path)
    if inputs[str(previous_path.relative_to(ROOT))] != manifest["previous_sample_manifest_sha256"]:
        raise ValueError("Previous fixed sample manifest hash changed")
    if previous["cases"] != manifest["cases"]:
        raise ValueError("Revision3 cases drifted from the fixed revision2 selection")
    baseline_path = CAMPAIGN / "frozen_baseline.json"
    read(baseline_path)
    if inputs[str(baseline_path.relative_to(ROOT))] != manifest["frozen_baseline_sha256"]:
        raise ValueError("Frozen baseline provenance hash changed")
    for case in manifest["cases"]:
        canonical = original[case["suite"], case["case"]]
        if any(case[name] != canonical[name] for name in ("seed", "scenario_sha256", "obstacles", "test_id")):
            raise ValueError("Sample canonical scenario/seed mismatch")
    for name, expected in metadata["settings"]["source_sha256"].items():
        if sha(read_shared_bytes(ROOT / name)) != expected:
            raise ValueError(f"Frozen run source changed since evaluation: {name}")
    if sha(read_shared_bytes(Path(__file__).with_name("quick_campaign.py"))) != metadata["runner_sha256"]:
        raise ValueError("Quick runner changed since evaluation")
    checkpoint = Path(metadata["settings"]["checkpoint"])
    if sha(read_shared_bytes(checkpoint)) != metadata["settings"]["checkpoint_sha256"]:
        raise ValueError("Policy checkpoint changed")
    config = checkpoint.with_name("config.json")
    if (sha(read_shared_bytes(config)) if config.exists() else None) != metadata["settings"]["config_sha256"]:
        raise ValueError("Policy model config changed")
    tokens, token_sources = {}, {}
    for path in sorted((CAMPAIGN / "attempts").glob("*.json")):
        if not re.fullmatch(r"\d{3}\.json", path.name):
            raise ValueError(f"Unexpected budget token name: {path.name}")
        raw = read_shared_bytes(path)
        token_sources[path.name] = sha(raw)
        try:
            tokens[int(path.stem)] = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            tokens[int(path.stem)] = None  # Failed token writes still consume budget.
    rows = []
    for path in sorted(directory.glob("episode_*.json")):
        row = read(path)
        if path.name != f"episode_{row['attempt']:03d}.json":
            raise ValueError("Episode filename/attempt mismatch")
        rows.append(row)
    completions = [read(p) for p in sorted(directory.glob("complete_*.json"))]
    ordered, derived = validate_records(manifest, metadata, rows, tokens, completions, tag)
    result_numbers = {r["attempt"] for r in ordered}
    tag_tokens = [n for n, token in tokens.items() if isinstance(token, dict) and token.get("tag") == tag]
    budget = {"hard_cap": 100, "consumed_attempts": len(tokens), "remaining_attempts": 100 - len(tokens),
              "reported_completed_episodes": 72, "this_tag_attempts": len(tag_tokens),
              "this_tag_attempts_without_completed_result": sorted(set(tag_tokens) - result_numbers),
              "other_attempts_excluded_from_report": len(tokens) - len(tag_tokens),
              "unparseable_consumed_tokens": sum(token is None for token in tokens.values()),
              "attempts_by_tag": dict(Counter(token.get("tag", "unknown") if isinstance(token, dict) else "unparseable" for token in tokens.values()))}
    archive_path = directory / "evaluated_sources.zip"
    if archive_path.exists():
        archive_manifest = read(directory / "evaluated_sources_manifest.json")
        budget["evaluated_sources_archive_sha256"] = sha(read_shared_bytes(archive_path))
        if archive_manifest["archive_sha256"] != budget["evaluated_sources_archive_sha256"]:
            raise ValueError("Evaluated source archive hash mismatch")
        if archive_manifest["checkpoint_sha256"] != metadata["settings"]["checkpoint_sha256"]:
            raise ValueError("Evaluated source archive checkpoint mismatch")
        inputs[str(archive_path.relative_to(ROOT))] = budget["evaluated_sources_archive_sha256"]
    return directory, manifest, metadata, ordered, derived, budget, inputs, token_sources


def markdown(tag, manifest, metadata, derived, budget):
    goals = {r["mode"]: r["goal"] for r in derived["outcomes"] if r["suite"] == "overall"}
    headline = f"**Goals: policy alone {goals['off']}/24; v4 {goals['v4']}/24; v5 {goals['v5']}/24.** "
    if goals["v5"] == goals["v4"]:
        headline += "V5 achieved no additional goals over v4 in this sample."
    else:
        headline += f"V5 changed the goal count by {goals['v5'] - goals['v4']:+d} relative to v4 in this sample."
    lines = [f"# Fixed 24-case safety comparison: {tag}", "",
        headline, "",
        "Completed **72 fresh episodes: 24 identical scenario/seed triples × policy alone (off), selected v4, and candidate v5**. "
        "Development probes are excluded from these outcomes and remain charged to the shared attempt budget.", "",
        f"Shared budget: **{budget['consumed_attempts']}/100 attempts consumed**; {budget['remaining_attempts']} remain. "
        f"This tag used {budget['this_tag_attempts']} attempts for 72 completed results; "
        f"{budget['other_attempts_excluded_from_report']} other attempts (including the earlier development probes) are excluded.", "",
        "Cases were fixed from scenario-only strata and hash order before quick outcomes were observed. "
        "This mixed-suite diagnostic subset does **not** estimate performance over all 2,890 cases / 8,670 mode runs. "
        "No significance or confidence interval is claimed; robustness variants can share base scenarios. "
        "Field-layout and field-validation cases are simulations, not real-world trials.", "",
        "## Outcomes", "", "Counts use all episodes in each row as denominator; timeouts remain separate from collisions. "
        "Overall counts pool the 24 selected cases, rather than averaging suite percentages.", "",
        "| Component | Mode | n | Goals | Obstacles | Boundaries | Targets | Timeouts |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in derived["outcomes"]:
        lines.append(f"| {r['suite']} | {r['mode']} | {r['episodes']} | {r['goal']} | {r['collision:obstacle']} | {r['collision:boundary']} | {r['collision:target']} | {r['timeout']} |")
    lines += ["", "## Paired changes", "", "Each comparison has exactly 24 matched case IDs. A gained/lost goal compares goal against any non-goal outcome.", "",
              "| Candidate vs reference | Goal gains | Goal losses | Collision → goal | Collision → timeout |",
              "|---|---:|---:|---:|---:|"]
    for name, r in derived["comparisons"].items():
        lines.append(f"| {name} | {r['gained_goals']} | {r['lost_goals']} | {r['collision_to_goal']} | {r['collision_to_timeout']} |")
    lines += ["", "Changed outcome types:", "", "| Comparison | Reference | Candidate | Cases |", "|---|---|---|---:|"]
    for name, comp in derived["comparisons"].items():
        for r in comp["transitions"]:
            if r["reference_outcome"] != r["candidate_outcome"]:
                lines.append(f"| {name} | {r['reference_outcome']} | {r['candidate_outcome']} | {r['cases']} |")
    lines += ["", "## Mechanism use", "", "Cells show **steps / episodes with at least one such step**. "
              "`policy_feedback_preserved` counts accepted policy-feedback certification checks; it does not establish that a safety override was prevented or that the action changed. "
              "`continuation_override_prevented` is a separate diagnostic. Enabled-step counts describe configuration, not interventions.", "",
              "| Diagnostic | off | v4 | v5 |", "|---|---:|---:|---:|"]
    for name in FLAGS:
        values = [f"{derived['mechanisms'][m][name + '_steps']} / {derived['mechanisms'][m][name + '_episodes']}" for m in MODES]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    changed = derived["mechanisms"]
    lines += ["", f"Changed-action steps: off {changed['off']['changed_action_steps']}, "
              f"v4 {changed['v4']['changed_action_steps']}, v5 {changed['v5']['changed_action_steps']}. "
              "A smaller intervention count is not evidence of a safety improvement."]
    lines += ["", "Selected recovery-template step counts:", ""]
    for mode in MODES:
        lines.append(f"- {mode}: `{json.dumps(derived['mechanisms'][mode]['recovery_template_counts'], sort_keys=True)}`")
    lines += ["", "## Runtime", "", "Recorded per-episode elapsed runtime excludes model loading and runner setup; totals are not end-to-end campaign wall time.", "",
              "| Mode | Total seconds | Median seconds/episode | Maximum seconds/episode |", "|---|---:|---:|---:|"]
    for r in derived["outcomes"]:
        if r["suite"] == "overall":
            lines.append(f"| {r['mode']} | {r['seconds_total']:.2f} | {r['seconds_median']:.2f} | {r['seconds_max']:.2f} |")
    switches = metadata["settings"]["effective_safety_constants"]["v5"]
    lines += ["", "## Provenance and limits", "",
        f"Policy checkpoint SHA256: `{metadata['settings']['checkpoint_sha256']}`.", "",
        "Frozen candidate switches: " + "; ".join(f"`{key}={switches.get(key, 'missing')}`" for key in
        ("PREFER_CERTIFIED_POLICY", "POLICY_FEEDBACK_PRESERVATION", "FEEDBACK_BACKUPS", "SIDESLIP_RESCUE", "ONE_DECISION_COMMIT", "SOFT_RECOVERY")) + ".", "",
        "The report validates all 72 unique identities, original reset seeds/scenario digests, metadata and source hashes, immutable budget tokens, and completed-summary accounting. "
        "Per-episode JSON records are authoritative. Earlier partial benchmark outcomes and selected development counterfactuals are separate evidence. "
        "Omitted family coverage remains in `sample_family_coverage_v3.csv`; no inverse-probability weighting is applied.", ""]
    if "evaluated_sources_archive_sha256" in budget:
        lines += ["Frozen evaluated code: [source archive](evaluated_sources.zip) and "
                  "[archive manifest](evaluated_sources_manifest.json).", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    first = list(rows[0])
    fields = first + sorted(set().union(*(r.keys() for r in rows)) - set(first))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="policy_feedback_v1")
    parser.add_argument("--prefix", default="report", help="new output stem; existing artifacts are never overwritten")
    args = parser.parse_args()
    for value in (args.tag, args.prefix):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            parser.error("tag/prefix must contain letters, digits, underscores or hyphens")
    directory, manifest, metadata, rows, derived, budget, inputs, token_sources = load_run(args.tag)
    outputs = {"markdown": directory / f"{args.prefix}.md", "episodes": directory / f"{args.prefix}_episodes.csv",
               "pairs": directory / f"{args.prefix}_pairs.csv", "provenance": directory / f"{args.prefix}_provenance.json"}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Reporting artifacts already exist; choose a new --prefix")
    text = markdown(args.tag, manifest, metadata, derived, budget)
    write_csv(outputs["episodes"], rows)
    write_csv(outputs["pairs"], derived["pairs"])
    with outputs["markdown"].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    provenance = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": args.tag, "sample_only": True, "population_performance_estimate": False,
        "report_source_sha256": sha(read_shared_bytes(Path(__file__))), "settings": metadata["settings"],
        "metadata_canonical_sha256": canonical_sha(metadata), "input_sha256": inputs,
        "budget_token_sha256": token_sources, "budget": budget, "summary": derived,
        "output_sha256": {kind: sha(read_shared_bytes(path)) for kind, path in outputs.items() if kind != "provenance"}}
    with outputs["provenance"].open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    print(f"Validated 72 results and 24 triples; budget {budget['consumed_attempts']}/100. Wrote {outputs['markdown']}")


if __name__ == "__main__":
    main()
