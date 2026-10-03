"""Join saved OFF/V4 results to test-set v2; never import or run a simulator.

Only committed archived journal records with matching test ID, scene digest,
reset seed, checkpoint, and evaluator settings enter the paired comparison.
Outcome regressions are not labels for individual false-positive overrides.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "results/safety_dev/cancelled/v4_stopped_full_sweep_20261002T132311Z"
DATASET = ROOT / "results/test_set/v2"
OUTPUT = ROOT / "results/safety_dev/testset_v2_offline"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def unique(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def committed_records(data):
    """Incomplete trailing bytes are evidence of no committed result."""
    end = data.rfind(b"\n") + 1
    return [json.loads(line) for line in data[:end].splitlines() if line], len(data) - end


def same_identity(definition, row):
    return (definition["test_id"] == row["test_id"]
            and definition["digest"] == row["scenario_sha256"]
            and int(definition["episode_seed"]) == int(row["seed"]))


def settings_identity(settings):
    # B/R differ only in their requested component. All runtime and controller
    # settings, source hashes, checkpoint, and modes must otherwise match.
    return {key: value for key, value in settings.items() if key != "suites"}


def pair_rows(definitions, records):
    """Return exact paired cases plus full current-dataset coverage."""
    definitions = unique(definitions, "test_id")
    matched = defaultdict(dict)
    excluded = Counter()
    for row in records:
        identifier = row["test_id"]
        if identifier not in definitions:
            excluded["not_in_current_v2"] += 1
            continue
        if not same_identity(definitions[identifier], row):
            excluded["scene_or_seed_mismatch"] += 1
            continue
        if row["mode"] not in ("off", "v4"):
            excluded["other_mode"] += 1
            continue
        if row["mode"] in matched[identifier]:
            raise ValueError(f"Duplicate committed result: {identifier}/{row['mode']}")
        matched[identifier][row["mode"]] = row
    pairs, coverage = [], []
    for identifier, definition in definitions.items():
        modes = matched.get(identifier, {})
        status = "paired" if {"off", "v4"} <= modes.keys() else (
            "off_only" if "off" in modes else "v4_only" if "v4" in modes else "unrecorded")
        coverage.append({**definition, "saved_status": status})
        if status != "paired":
            continue
        off, candidate = modes["off"], modes["v4"]
        off_goal, candidate_goal = off["outcome"] == "goal", candidate["outcome"] == "goal"
        transition = ("rescue" if candidate_goal and not off_goal else
                      "lost_goal" if off_goal and not candidate_goal else
                      "preserved_goal" if off_goal else "both_failed")
        pairs.append({**definition, "suite": off["suite"], "case": off["case"],
                      "off_outcome": off["outcome"], "v4_outcome": candidate["outcome"],
                      "transition": transition,
                      "off_steps": off["steps"], "v4_steps": candidate["steps"],
                      "v4_intervention_steps": int(candidate["safety_v2_steps"]),
                      "v4_brake_steps": int(candidate["safety_v2_brake_steps"])})
    return pairs, coverage, matched, dict(excluded)


def aggregate(pairs):
    off = Counter(row["off_outcome"] for row in pairs)
    candidate = Counter(row["v4_outcome"] for row in pairs)
    trades = Counter(row["transition"] for row in pairs)
    interventions = {}
    for transition in ("preserved_goal", "lost_goal", "rescue", "both_failed"):
        rows = [row for row in pairs if row["transition"] == transition]
        interventions[transition] = {
            "episodes": len(rows),
            "episodes_with_recorded_intervention": sum(row["v4_intervention_steps"] > 0 for row in rows),
            "recorded_intervention_steps": sum(row["v4_intervention_steps"] for row in rows),
            "recorded_brake_steps": sum(row["v4_brake_steps"] for row in rows),
        }
    return {"paired": len(pairs), "off": dict(off), "v4": dict(candidate),
            "off_contacts": sum(count for key, count in off.items() if key.startswith("collision:")),
            "v4_contacts": sum(count for key, count in candidate.items() if key.startswith("collision:")),
            "transitions": dict(trades), "interventions_by_transition": interventions}


def collect(root=ROOT, archive=ARCHIVE, dataset=DATASET):
    inputs = {}

    def evidence(path):
        raw = path.read_bytes()
        inputs[str(path.relative_to(root)).replace("\\", "/")] = {
            "sha256": sha(raw), "bytes": len(raw)}
        return raw

    definitions_path, metadata_path = dataset / "definition.csv", dataset / "definition.json"
    evidence(definitions_path)
    definitions = read_csv(definitions_path)
    metadata = json.loads(evidence(metadata_path))
    definition_map = unique(definitions, "test_id")
    if len(definitions) != metadata["episodes"]:
        raise ValueError("Current definition row count does not match metadata")
    if canonical_sha({key: row["digest"] for key, row in definition_map.items()}) != metadata["manifest_digest"]:
        raise ValueError("Current definition scene manifest digest mismatch")
    # Record cache bytes for provenance; do not unpickle or call cache builders.
    evidence(dataset / "set_v2.0.pkl")
    old_path = dataset.parent / "definition.csv"
    evidence(old_path)
    old = unique(read_csv(old_path), "test_id")
    removed, added = sorted(old.keys() - definition_map.keys()), sorted(definition_map.keys() - old.keys())
    changed = sorted(key for key in old.keys() & definition_map.keys()
                     if (old[key]["digest"], old[key]["episode_seed"]) !=
                     (definition_map[key]["digest"], definition_map[key]["episode_seed"]))
    if removed != sorted(metadata["start_in_contact"]):
        raise ValueError("V1 removed IDs do not match documented v2 replacements")

    sources = json.loads(evidence(root / "results/safety_dev/v4_stopped_full_sweep_sources.json"))
    expected = {Path(item["directory"]).name: item for item in sources["inputs"]}
    all_records, manifests, journal_stats = [], [], []
    for tag in ("v5_frozen_b", "v5_frozen_r"):
        manifest_raw = evidence(archive / tag / "manifest.json")
        journal_raw = evidence(archive / tag / "episodes.jsonl")
        if sha(manifest_raw) != expected[tag]["manifest_sha256"]:
            raise ValueError(f"Archived manifest hash mismatch: {tag}")
        if sha(journal_raw) != expected[tag]["journal_snapshot_sha256"]:
            raise ValueError(f"Archived journal hash mismatch: {tag}")
        manifest = json.loads(manifest_raw)
        records, tail = committed_records(journal_raw)
        if len(records) != expected[tag]["completed"]:
            raise ValueError(f"Committed record count mismatch: {tag}")
        cases = unique(manifest["cases"], "test_id")
        for row in records:
            declared = cases.get(row["test_id"])
            fields = ("suite", "case", "test_id", "seed", "scenario_sha256")
            if declared is None or any(row[field] != declared[field] for field in fields):
                raise ValueError(f"Journal identity differs from manifest: {tag}/{row['test_id']}")
            if row["policy"] != "model":
                raise ValueError("Unexpected non-model journal result")
        manifests.append(manifest)
        all_records.extend(records)
        journal_stats.append({"tag": tag, "records": len(records), "ignored_tail_bytes": tail})
    settings = settings_identity(manifests[0]["settings"])
    if any(settings_identity(manifest["settings"]) != settings for manifest in manifests[1:]):
        raise ValueError("Archived B/R checkpoint, settings, runtime, or source mismatch")
    checkpoint = Path(settings["checkpoint"])
    if sha(evidence(checkpoint)) != settings["checkpoint_sha256"]:
        raise ValueError("Current baseline checkpoint differs from archived checkpoint")
    if sha(evidence(checkpoint.with_name("config.json"))) != settings["config_sha256"]:
        raise ValueError("Current baseline configuration differs from archived configuration")

    pairs, coverage, matched, excluded = pair_rows(definitions, all_records)
    baseline_path = dataset / "sacs0_bl3/episodes.csv"
    evidence(baseline_path)
    baseline = unique(read_csv(baseline_path), "test_id")
    disagreements, crosschecks = [], 0
    for identifier, modes in matched.items():
        if "off" not in modes or identifier not in baseline:
            continue
        crosschecks += 1
        if modes["off"]["outcome"] != baseline[identifier]["outcome"]:
            disagreements.append(identifier)
    summary = aggregate(pairs)
    summary.update({"dataset_episodes": len(definitions),
                    "dataset_sources": dict(Counter(row["source"] for row in definitions)),
                    "coverage": dict(Counter(row["saved_status"] for row in coverage)),
                    "coverage_by_source": {source: dict(Counter(row["saved_status"] for row in coverage
                                                                 if row["source"] == source))
                                           for source in sorted({row["source"] for row in coverage})},
                    "excluded_records": excluded,
                    "saved_v2_off_outcome_crosscheck": {"compared": crosschecks,
                                                       "disagreements": disagreements},
                    "v1_to_v2": {"removed_ids": removed, "added_ids": added,
                                 "shared_ids_with_changed_scene_or_seed": changed},
                    "by_cell": {cell: aggregate([row for row in pairs if row["cell"] == cell])
                                for cell in sorted({row["cell"] for row in pairs})}})
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "offline_only": True, "simulator_imports": False, "episodes_started": 0,
                  "analysis_source_sha256": sha(Path(__file__).read_bytes()),
                  "dataset_manifest_digest": metadata["manifest_digest"],
                  "archived_settings_sha256": canonical_sha(settings),
                  "archived_settings": settings, "journals": journal_stats, "inputs": inputs,
                  "limitations": [
                      "Archived OFF/V4 are paired with one another, then mapped to canonical v2 by ID, scene digest and seed.",
                      "Saved v2 SAC CSV has no run-level checkpoint/settings manifest; its outcome agreement is a crosscheck, not execution-provenance proof.",
                      "No assumption of equivalence between archived and current controller/simulator source is made.",
                      "Episode intervention totals cannot label a particular override as necessary, false-positive, or causal.",
                      "Partial sequential frozen-source coverage does not estimate complete v2 performance.",
                      "No saved V6/V7 results are inferred; no changed L1-BO record is substituted."]}
    return summary, pairs, coverage, provenance


def markdown(summary):
    lines = ["# Saved OFF/V4 outcomes on exact test-set-v2 matches", "",
             "Offline analysis only: no environment construction, reset, step, model loading, or new episode.", "",
             f"The archived stopped journals contain **{summary['paired']} exact paired cases** from the current "
             f"{summary['dataset_episodes']}-case definition. Pairing requires test ID, scene digest and reset seed; "
             "both archived modes share checkpoint, evaluator settings, source hashes and runtime settings.", "",
             "| Mode | Goals | Obstacle | Boundary | Target | Total contacts |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in ("off", "v4"):
        counts = summary[mode]
        lines.append(f"| {mode.upper()} | {counts.get('goal', 0)} | {counts.get('collision:obstacle', 0)} | "
                     f"{counts.get('collision:boundary', 0)} | {counts.get('collision:target', 0)} | {summary[mode + '_contacts']} |")
    trades = summary["transitions"]
    lines += ["", f"V4 rescues **{trades.get('rescue', 0)}** failed OFF episodes but loses **{trades.get('lost_goal', 0)}** "
              "OFF goals. These are observed episode trades, not ground-truth labels for individual false overrides.", "",
              "| Episode transition | Cases | Cases with recorded intervention | Intervention decisions | Brake decisions |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for label, values in summary["interventions_by_transition"].items():
        lines.append(f"| {label} | {values['episodes']} | {values['episodes_with_recorded_intervention']} | "
                     f"{values['recorded_intervention_steps']} | {values['recorded_brake_steps']} |")
    lines += ["", "The journal's `safety_v2_steps` and `safety_v2_brake_steps` are recorded episode totals. "
              "A successful OFF episode shows that its complete policy trajectory succeeded; it does not show "
              "that every nominal action after V4 has diverted the boat would remain safe. Interventions in "
              "preserved-goal episodes therefore cannot all be counted as false positives. The journals do not "
              "contain predecision sensors/actions or filter margins, so they cannot support an online trigger "
              "classifier or local causal replay without additional already-recorded decision traces.", "",
              "| Cell | Paired | OFF goals | V4 goals | Rescues | Lost goals |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for cell, values in summary["by_cell"].items():
        lines.append(f"| {cell} | {values['paired']} | {values['off'].get('goal', 0)} | "
                     f"{values['v4'].get('goal', 0)} | {values['transitions'].get('rescue', 0)} | "
                     f"{values['transitions'].get('lost_goal', 0)} |")
    check = summary["saved_v2_off_outcome_crosscheck"]
    lines += ["", f"All {check['compared']} available matching archived OFF outcomes were checked against "
              f"the saved v2 SAC CSV; disagreements: {len(check['disagreements'])}. The latter CSV lacks its own "
              "run-level source/settings manifest, so it is a consistency check and is not substituted for a "
              "missing committed OFF result.", "",
              f"Coverage: `{json.dumps(summary['coverage'], sort_keys=True)}`. All pairs come from the frozen-source "
              "portion. None of the simulated Paper 2 layout cases has a paired V4 result in these stopped journals. "
              "They are simulated cases, not new field trials. The 15 original L1-BO IDs were replaced by 15 "
              "regenerated IDs; the other 985 shared IDs have unchanged scene digests and reset seeds. No removed "
              "scene was matched to a replacement. Missing records, including B-06-040 OFF, remain missing.", "",
              "This partial sequential sample is not a full-v2 rate estimate and reports archived V4 only. "
              "It provides no V6/V7 success claim and no evidence for a 95% result. The baseline checkpoint and "
              "configuration hashes match the saved evaluator metadata; current source equivalence is not assumed.", "",
              "Artifacts: [paired cases](pairs.csv), [goal trades](goal_trades.csv), "
              "[all 1,000 coverage identities](coverage.csv), [summary](summary.json), "
              "[full input hashes and archived settings](provenance.json).", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="new output directory; must not already exist")
    args = parser.parse_args()
    summary, pairs, coverage, provenance = collect()
    output = args.output or OUTPUT / (datetime.now(timezone.utc).strftime("saved_v4_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary", summary), ("provenance", provenance)):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_csv(output / "pairs.csv", pairs)
    write_csv(output / "goal_trades.csv", [row for row in pairs if row["transition"] in ("rescue", "lost_goal")])
    write_csv(output / "coverage.csv", coverage)
    (output / "REPORT.md").write_text(markdown(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "pairs": summary["paired"], "transitions": summary["transitions"]}))


if __name__ == "__main__":
    main()
