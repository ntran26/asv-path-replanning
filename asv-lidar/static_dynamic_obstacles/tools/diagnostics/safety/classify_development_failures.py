"""Offline descriptive classification of a complete, fresh paired DV3 run.

Consumes development_campaign artifacts only. No environment, policy or
controller imports, no replay, and no evaluation-attempt writes. Failure labels
describe recorded decisions; they are not causal or recoverability proofs.
The failure-enriched sample cannot estimate population success or a 95% target.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite_status import read_shared_bytes

CHECKED_REASONS = {"nominal", "handback", "turn", "continue", "brake",
                   "searched policy", "searched escape", "policy feedback"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def count(value):
    return len(value) if isinstance(value, list) else None


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def summarize_trace(episode, rows, trigger):
    """Summarize observed reasons without assuming all tracks are real targets."""
    if [r.get("decision") for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Trace decisions must be complete and consecutive from one")
    if len(rows) != episode["steps"] or len(rows) != episode["observed_decisions"]:
        raise ValueError("Trace/episode decision counts differ")
    expected_class = f"safety_{episode['mode']}.SafetyFilter{episode['mode'].upper()}"
    if any(r.get("actual_filter_class") != expected_class for r in rows):
        raise ValueError("Recorded filter class differs from paired mode")
    details, accepted, qualified = [], [], []
    for row in rows:
        filt, snap = row.get("filter", {}), row.get("observer_snapshot") or {}
        provisional = row.get("perception_provisional") or {}
        history = (row.get("perception_statistics") or {}).get("last_track_history_stats") or {}
        tracks, targets = snap.get("tracks"), row.get("targets_before")
        why = filt.get("why", "idle" if filt.get("mode") == "idle" else "unreported")
        changed = bool(row.get("action_changed") or filt.get("changed") or filt.get("brake"))
        clearance = finite(filt.get("checked_clearance"))
        hard_accepted = (why in CHECKED_REASONS and clearance is not None and clearance >= 0.
                         and (why != "continue" or filt.get("continuation_ok") is True))
        hypotheses = history.get("hypotheses", [])
        nearest = None
        if tracks and targets:
            nearest = min(math.dist(t["position_m"], est["position_m"])
                          for t in targets for est in tracks)
        before = row.get("before", {})
        ego_error = {}
        if snap and all(before.get(k) is not None for k in ("asv_x", "asv_y", "u_body", "v_body", "asv_w")):
            ego_error = {"position_error_m": math.hypot(snap["x_m"] - before["asv_x"], snap["y_m"] - before["asv_y"]),
                         "u_error_mps": snap["u_mps"] - before["u_body"],
                         "v_error_mps": snap["v_mps"] - before["v_body"],
                         "yaw_rate_error_dps": math.degrees(snap["r_radps"]) - before["asv_w"]}
        details.append({"decision": row["decision"], "why": why, "mode": filt.get("mode"),
                        "changed": changed, "brake": bool(filt.get("brake")),
                        "checked_clearance_m": clearance, "hard_accepted_by_reason": hard_accepted,
                        "margin_qualified": hard_accepted and clearance >= trigger,
                        "original_bank_any_safe": filt.get("any_safe"),
                        "policy_safe_in_original_bank": filt.get("policy_safe"),
                        "truth_target_count": count(targets), "snapshot_track_count": count(tracks),
                        "raw_tracker_count": count(row.get("tracker_tracks_before")),
                        "published_dynamic_track_count": count(row.get("published_dynamic_track_ids_before")),
                        "nearest_snapshot_track_to_any_truth_target_m": nearest,
                        "provisional_added_tracks": provisional.get("added_track_count"),
                        "provisional_new_admissions": count(provisional.get("new_admissions")),
                        "history_added_hypotheses": history.get("added_hypotheses"),
                        "history_distinct_source_count": len({h["source_id"] for h in hypotheses}) if history else None,
                        "history_stored_anchor_count": history.get("stored_anchor_count"),
                        "history_anchor_ids": [h["id"] for h in hypotheses],
                        "history_source_ids": sorted({h["source_id"] for h in hypotheses}),
                        "truth_u_mps": before.get("u_body"), "truth_v_mps": before.get("v_body"),
                        "truth_yaw_rate_dps": before.get("asv_w"), "ego_estimation_error": ego_error,
                        "policy_search": filt.get("policy_search"), "escape_search": filt.get("escape_search"),
                        "provisional_stats": provisional, "history_stats": history})
        if hard_accepted:
            accepted.append(details[-1])
            if clearance >= trigger:
                qualified.append(details[-1])
    first_changed = next((d["decision"] for d in details if d["changed"]), None)
    no_escape = [d["decision"] for d in details if d["why"] == "no escape"]
    final = details[-1]
    if final["why"] in ("last certificate", "no escape", "hold back"):
        label = {"last certificate": "final_unchecked_continuation",
                 "no escape": "final_no_escape_policy_pass", "hold back": "final_delay_only_fallback"}[final["why"]]
    elif final["hard_accepted_by_reason"]:
        label = "final_checked_margin_qualified" if final["margin_qualified"] else "final_checked_below_trigger"
    else:
        label = "final_" + str(final["why"]).replace(" ", "_")
    result = {k: episode[k] for k in ("case", "mode", "outcome", "steps", "seed", "scenario_sha256", "attempt")}
    result.update({"trace_classification": label, "first_changed_decision": first_changed,
                   "changed_decisions": sum(d["changed"] for d in details),
                   "brake_decisions": sum(d["brake"] for d in details),
                   "first_no_escape_decision": no_escape[0] if no_escape else None,
                   "no_escape_before_first_override": any(k < (first_changed or math.inf) for k in no_escape),
                   "no_escape_prefix_decisions": [k for k in no_escape if k < (first_changed or math.inf)],
                   "reason_counts": dict(Counter(d["why"] for d in details)),
                   "final": final, "last_hard_accepted": accepted[-1] if accepted else None,
                   "last_margin_qualified": qualified[-1] if qualified else None,
                   "target_collision_with_empty_final_snapshot": episode["outcome"] == "collision:target"
                       and final["truth_target_count"] is not None and final["truth_target_count"] > 0
                       and final["snapshot_track_count"] == 0,
                   "history_distinct_anchor_ids_used": sorted({i for d in details for i in d["history_anchor_ids"]}),
                   "history_distinct_source_ids_used": sorted({i for d in details for i in d["history_source_ids"]})})
    for window in (5, 10):
        tail = details[-window:]
        result[f"last{window}"] = tail
        result[f"last{window}_reason_counts"] = dict(Counter(d["why"] for d in tail))
        result[f"last{window}_truth_present_snapshot_empty"] = sum(
            d["truth_target_count"] is not None and d["truth_target_count"] > 0
            and d["snapshot_track_count"] == 0 for d in tail)
    for key in ("provisional_added_tracks", "provisional_new_admissions", "history_added_hypotheses"):
        observed = [d[key] for d in details if d[key] is not None]
        result[key + "_sum"] = sum(observed) if observed else None
        result[key + "_logged_decisions"] = len(observed)
    return result


def load_run(directory):
    """Require full fresh paired records; never borrow historical V4 outcomes."""
    manifest_raw = read_shared_bytes(directory / "manifest.json")
    manifest, manifest_sha = json.loads(manifest_raw), sha(manifest_raw)
    config = manifest["configuration"]
    if set(config["modes"]) != {"v4", "v6"} or len(config["modes"]) != 2:
        raise ValueError("Require fresh V4 and V6 within the same paired run")
    cases = {c["case"]: c for c in config["cases"]}
    if len(cases) != len(config["cases"]) or any(not case.startswith("DV3-") for case in cases):
        raise ValueError("Unique DEVELOPMENT cases are required")
    if any(case.get("suite") != "dev_field" for case in cases.values()):
        raise ValueError("Only the development suite may enter this classifier")
    complete_raw = read_shared_bytes(directory / "complete.json")
    complete = json.loads(complete_raw)
    if complete["completed"] != complete["expected"] or complete["expected"] != len(cases) * 2:
        raise ValueError("Run is incomplete")
    archive_path = directory / "evaluated_sources.zip"
    if sha(read_shared_bytes(archive_path)) != manifest["archive_sha256"]:
        raise ValueError("Frozen source archive differs from manifest")
    trigger = float(config["settings"]["effective_safety_constants"]["v2"]["TRIGGER_MARGIN_M"])
    completed = {(r["case"], r["mode"]): r for r in complete["rows"]}
    if len(completed) != len(cases) * 2:
        raise ValueError("Completion rows are missing or duplicated")
    summaries, seen, provenance = [], set(), {}
    for path in sorted(directory.glob("episode_*.json")):
        raw = read_shared_bytes(path)
        episode = json.loads(raw)
        if episode.get("suite") != "dev_field" or episode.get("tag") != directory.name:
            raise ValueError("Episode suite/tag differs from the development run")
        if path.name != f"episode_{episode['attempt']:03d}.json":
            raise ValueError("Episode filename differs from attempt identity")
        if episode["outcome"] not in {"goal", "timeout", "collision:target", "collision:obstacle", "collision:boundary"}:
            raise ValueError("Unrecognized episode outcome")
        key = episode["case"], episode["mode"]
        if key in seen or key not in completed or episode["case"] not in cases:
            raise ValueError("Duplicate or unexpected paired episode")
        seen.add(key)
        case = cases[episode["case"]]
        for field in ("seed", "scenario_sha256"):
            if episode[field] != case[field]:
                raise ValueError(f"Paired episode {field} mismatch")
        if episode["run_manifest_sha256"] != manifest_sha:
            raise ValueError("Episode belongs to another frozen run")
        if json.dumps(episode, sort_keys=True) != json.dumps(completed[key], sort_keys=True):
            raise ValueError("Completion row differs from its durable episode")
        attempt_path = directory.parents[1] / "attempts" / f"{episode['attempt']:03d}.json"
        attempt_raw = read_shared_bytes(attempt_path)
        attempt = json.loads(attempt_raw)
        for field in ("case", "mode", "seed", "scenario_sha256", "run_manifest_sha256", "attempt", "suite", "tag"):
            if attempt[field] != episode[field]:
                raise ValueError("Budget token identity differs from episode")
        trace_path = path.with_name(path.stem.replace("episode_", "trace_") + ".jsonl")
        trace_raw = read_shared_bytes(trace_path)
        rows = [json.loads(line) for line in trace_raw.splitlines() if line.strip()]
        summaries.append(summarize_trace(episode, rows, trigger))
        provenance[path.name] = sha(raw)
        provenance[trace_path.name] = sha(trace_raw)
        provenance["attempts/" + attempt_path.name] = sha(attempt_raw)
    if seen != set(completed):
        raise ValueError("Missing durable episodes; no historical outcomes will be substituted")
    if sha(read_shared_bytes(directory / "manifest.json")) != manifest_sha:
        raise ValueError("Manifest changed during report read")
    return summaries, {"manifest_sha256": manifest_sha, "complete_sha256": sha(complete_raw),
                       "archive_sha256": manifest["archive_sha256"], "files_sha256": provenance,
                       "paired_cases": len(cases), "trigger_margin_m": trigger,
                       "checkpoint_sha256": config["settings"].get("checkpoint_sha256"),
                       "model_config_sha256": config["settings"].get("config_sha256"),
                       "candidate_effective_constants": config["settings"]["effective_safety_constants"].get("v6", {}),
                       "candidate_overrides": config["settings"].get("v6_overrides", {}),
                       "selection_origin": manifest.get("selection_origin")}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/safety_dev/development_v6_budget150/runs/broad_v6_50")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("Tag must be one directory name")
    episodes, provenance = load_run(args.run)
    by_key = {(e["case"], e["mode"]): e for e in episodes}
    pairs = []
    for case in sorted({e["case"] for e in episodes}):
        base, candidate = by_key[case, "v4"], by_key[case, "v6"]
        gain = candidate["outcome"] == "goal" and base["outcome"] != "goal"
        loss = base["outcome"] == "goal" and candidate["outcome"] != "goal"
        pairs.append({"case": case, "seed": base["seed"], "scenario_sha256": base["scenario_sha256"],
                      "v4_outcome": base["outcome"], "v6_outcome": candidate["outcome"],
                      "paired_change": "gain" if gain else "loss" if loss else "same_success_status",
                      "v4_final_trace_class": base["trace_classification"],
                      "v6_final_trace_class": candidate["trace_classification"],
                      "v4_changed_decisions": base["changed_decisions"], "v6_changed_decisions": candidate["changed_decisions"]})
    summary = {"utc": datetime.now(timezone.utc).isoformat(), "new_episodes": 0,
               "scope": "Complete fresh V4/V6 paired DEVELOPMENT traces; no held-out outcomes or historical V4 substitutions",
               "inference_limit": "Failure-enriched selection; descriptive paired results cannot estimate population success or support a 95% claim",
               "provenance": provenance, "script_sha256": sha(Path(__file__).read_bytes()),
               "outcomes": {mode: dict(Counter(e["outcome"] for e in episodes if e["mode"] == mode)) for mode in ("v4", "v6")},
               "gains": [p["case"] for p in pairs if p["paired_change"] == "gain"],
               "losses": [p["case"] for p in pairs if p["paired_change"] == "loss"],
               "failure_trace_classes": {mode: dict(Counter(e["trace_classification"] for e in episodes
                    if e["mode"] == mode and e["outcome"] != "goal")) for mode in ("v4", "v6")},
               "interpretation": ["Reason-based labels are observations, not causal diagnoses.",
                   "Snapshot and raw tracker fields describe pre-dynamics filter input.",
                   "Estimated tracks and synthetic history IDs do not establish true-target visibility or distinct vessel count.",
                   "History stored anchors include age zero; added hypotheses count older appended constraints.",
                   "Missing diagnostics remain null, not zero.",
                   "Original-bank any_safe may remain false when trajectory search successfully finds a safe plan.",
                   "Hard acceptance is inferred only from known checked-path reasons and nonnegative checked clearance; last certificate/hold back/no escape are excluded."]}
    output = ROOT / "results/safety_dev/development_v6_budget150/offline" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    (output / "episodes.json").write_text(json.dumps(json_safe(episodes), indent=2, allow_nan=False) + "\n")
    (output / "failures.json").write_text(json.dumps(json_safe([e for e in episodes if e["outcome"] != "goal"]), indent=2, allow_nan=False) + "\n")
    write_csv(output / "paired_cases.csv", pairs)
    failure_rows = []
    for episode in episodes:
        if episode["outcome"] == "goal":
            continue
        final = episode["final"]
        accepted = episode["last_hard_accepted"] or {}
        qualified = episode["last_margin_qualified"] or {}
        failure_rows.append({"case": episode["case"], "mode": episode["mode"],
                             "outcome": episode["outcome"], "steps": episode["steps"],
                             "classification": episode["trace_classification"],
                             "final_reason": final["why"],
                             "first_changed_decision": episode["first_changed_decision"],
                             "no_escape_before_first_override": episode["no_escape_before_first_override"],
                             "last_hard_accepted_decision": accepted.get("decision"),
                             "last_hard_accepted_clearance_m": accepted.get("checked_clearance_m"),
                             "last_margin_qualified_decision": qualified.get("decision"),
                             "last_margin_qualified_clearance_m": qualified.get("checked_clearance_m"),
                             "final_truth_targets": final["truth_target_count"],
                             "final_snapshot_tracks": final["snapshot_track_count"],
                             "final_raw_tracker_tracks": final["raw_tracker_count"],
                             "final_published_dynamic_tracks": final["published_dynamic_track_count"],
                             "final_provisional_added": final["provisional_added_tracks"],
                             "final_history_added": final["history_added_hypotheses"],
                             "last5_reason_counts": json.dumps(episode["last5_reason_counts"], sort_keys=True),
                             "last10_reason_counts": json.dumps(episode["last10_reason_counts"], sort_keys=True)})
    if failure_rows:
        write_csv(output / "failure_summary.csv", failure_rows)
    report = ["# Paired development failure classification", "", summary["scope"], "",
              "**" + summary["inference_limit"] + ".**", "",
              f"Matched cases: {provenance['paired_cases']}. Gains: {len(summary['gains'])}; losses: {len(summary['losses'])}.", "",
              "| Mode | Goals | Boundary | Obstacle | Target | Other |", "|---|---:|---:|---:|---:|---:|"]
    for mode, counts in summary["outcomes"].items():
        keys = ("goal", "collision:boundary", "collision:obstacle", "collision:target")
        values = [counts.get(key, 0) for key in keys]
        report.append(f"| {mode} | " + " | ".join(map(str, values + [sum(counts.values()) - sum(values)])) + " |")
    report += ["", "| Recorded final failure pattern | V4 | V6 |", "|---|---:|---:|"]
    classes = sorted(set(summary["failure_trace_classes"]["v4"]) | set(summary["failure_trace_classes"]["v6"]))
    for label in classes:
        report.append(f"| {label} | {summary['failure_trace_classes']['v4'].get(label, 0)} | {summary['failure_trace_classes']['v6'].get(label, 0)} |")
    report += ["", "Gained goals: " + (", ".join(summary["gains"]) or "none") + ".",
               "Lost goals: " + (", ".join(summary["losses"]) or "none") + ".", "",
               "Full last-five/last-ten reason sequences, first intervention, no-escape prefix, last checked clearance, "
               "ego error and target/provisional/history counts are in failures.json; failure_summary.csv provides a compact table. These are observed patterns, not causal proofs. "
               "History hypotheses are not distinct vessels; an estimated track is not proof the true target was observed. "
               "Counts missing from the trace remain null.", "", "No new episodes were run.", ""]
    (output / "REPORT.md").write_text("\n".join(report))
    print(json.dumps({"output": str(output), "outcomes": summary["outcomes"], "gains": summary["gains"], "losses": summary["losses"]}))


if __name__ == "__main__":
    main()
