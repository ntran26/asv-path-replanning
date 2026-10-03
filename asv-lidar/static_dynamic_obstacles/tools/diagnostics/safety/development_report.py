"""Offline reports for the development150 campaign; never import a controller.

  python -B tools/diagnostics/safety/development_report.py
  python -B tools/diagnostics/safety/development_report.py --refresh

Only fully completed tags enter outcome comparisons. Active/failed tags still
enter attempt accounting. Each refresh preserves an immutable report snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from suite_status import read_shared_bytes

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "results/safety_dev/development_v6_budget150"
OUTCOMES = ("goal", "collision:obstacle", "collision:boundary", "collision:target", "timeout")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def validate_row(row, cases, modes, tag, manifest_sha):
    identity = (row["mode"], row["case"])
    if row["mode"] not in modes or row["case"] not in cases or row.get("suite") != "dev_field":
        raise ValueError(f"Unknown development result identity: {tag}/{identity}")
    case = cases[row["case"]]
    if (row.get("tag") != tag or row.get("run_manifest_sha256") != manifest_sha
            or any(row.get(k) != case[k] for k in ("seed", "scenario_sha256"))):
        raise ValueError(f"Result provenance mismatch: {tag}/{identity}")
    if row.get("outcome") not in OUTCOMES:
        raise ValueError(f"Invalid completed outcome: {tag}/{identity}")
    if type(row.get("attempt")) is not int or not 1 <= row["attempt"] <= 150:
        raise ValueError("Invalid episode attempt number")
    if not isinstance(row.get("seconds"), (int, float)) or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
        raise ValueError("Invalid episode runtime")
    return identity


def validate_token(row, token):
    fields = ("attempt", "tag", "mode", "suite", "case", "seed", "scenario_sha256", "run_manifest_sha256")
    if not isinstance(token, dict) or any(k not in token or token[k] != row.get(k) for k in fields):
        raise ValueError(f"Consumed attempt token disagrees with episode: {row.get('attempt')}")


def validate_verification(row, token):
    """Completed simulator-based tests consume slots, never policy outcomes."""
    fields = ("attempt", "tag", "category", "test_id", "test_source_sha256",
              "status", "suite", "mode", "reservation_timing")
    if (not isinstance(row, dict) or any(key not in row for key in fields)
            or type(row["attempt"]) is not int or not 1 <= row["attempt"] <= 150
            or row["category"] != "verification" or row["status"] not in ("passed", "failed")
            or any(not isinstance(row[key], str) or not row[key] for key in fields if key != "attempt")
            or not re.fullmatch(r"[0-9a-f]{64}", row["test_source_sha256"])
            or "::" not in row["test_id"]
            or any(key in row for key in ("outcome", "case", "scenario_sha256"))):
        raise ValueError("Invalid completed verification record")
    source = Path(row["test_id"].split("::", 1)[0])
    if source.is_absolute() or ".." in source.parts or source.suffix != ".py":
        raise ValueError("Invalid verification source path")
    if not isinstance(token, dict) or any(token.get(key) != row[key] for key in fields):
        raise ValueError(f"Consumed attempt token disagrees with verification: {row['attempt']}")
    return dict(row, source_file=source.as_posix())


def paired(rows):
    indexed = {(r["mode"], r["case"]): r for r in rows}
    pairs = []
    for case in sorted({r["case"] for r in rows}):
        a, b = indexed.get(("v6", case)), indexed.get(("v4", case))
        if a is None or b is None:
            continue
        if any(a[k] != b[k] for k in ("seed", "scenario_sha256", "run_manifest_sha256")):
            raise ValueError("Fresh paired provenance mismatch")
        pairs.append({"case": case, "v4_outcome": b["outcome"], "v6_outcome": a["outcome"],
            "gained_goal": a["outcome"] == "goal" and b["outcome"] != "goal",
            "lost_goal": a["outcome"] != "goal" and b["outcome"] == "goal"})
    return {"scope": "fresh v4/v6 results within this one frozen tag only", "paired_cases": len(pairs),
            "gained_goals": sum(p["gained_goal"] for p in pairs),
            "lost_goals": sum(p["lost_goal"] for p in pairs), "pairs": pairs}


def mode_summary(rows, modes):
    output = []
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        counts = Counter(r["outcome"] for r in selected)
        output.append({"mode": mode, "completed": len(selected), **{o: counts[o] for o in OUTCOMES},
            "seconds_sum": sum(r["seconds"] for r in selected),
            "changed_action_steps": sum(r.get("safety_v2_steps", 0) for r in selected)})
    return output


def collect():
    input_hashes = {}
    def raw(path):
        data = read_shared_bytes(path)
        input_hashes[str(path.relative_to(ROOT))] = sha(data)
        return data
    def read(path):
        return json.loads(raw(path))
    tags, all_rows, errors_by_attempt = [], {}, {}
    for directory in sorted((CAMPAIGN / "runs").iterdir()):
        if not directory.is_dir() or not (directory / "manifest.json").exists():
            continue
        manifest_bytes = raw(directory / "manifest.json")
        manifest, manifest_sha = json.loads(manifest_bytes), sha(manifest_bytes)
        config = manifest["configuration"]
        cases = {c["case"]: c for c in config["cases"]}
        if (len(cases) != len(config["cases"]) or any(c["suite"] != "dev_field" for c in cases.values())
                or config["planned_episodes"] != len(cases)*len(config["modes"])
                or config["shared_attempt_cap"] != 150):
            raise ValueError(f"Invalid frozen campaign manifest: {directory.name}")
        if sha(raw(directory / "evaluated_sources.zip")) != manifest["archive_sha256"]:
            raise ValueError(f"Frozen source archive changed: {directory.name}")
        complete, partial_files = None, []
        if (directory / "complete.json").exists():
            try:
                complete = read(directory / "complete.json")
            except (ValueError, UnicodeDecodeError):
                partial_files.append("complete.json")
        rows, identities, attempts = [], set(), set()
        for path in sorted(directory.glob("episode_*.json")):
            try:
                row = read(path)
            except (ValueError, UnicodeDecodeError):
                if complete is not None:
                    raise ValueError(f"Incomplete episode file in completed tag: {path}")
                partial_files.append(path.name)
                continue
            key = validate_row(row, cases, config["modes"], directory.name, manifest_sha)
            if key in identities or row["attempt"] in all_rows or row["attempt"] in attempts:
                raise ValueError(f"Duplicate durable result: {directory.name}/{key}")
            if path.name != f"episode_{row['attempt']:03d}.json":
                raise ValueError("Episode filename/attempt mismatch")
            expected_class = config["settings"]["actual_filter_classes"][row["mode"]]
            if row.get("actual_filter_class") != expected_class:
                raise ValueError("Recorded actual filter class differs from manifest")
            identities.add(key)
            attempts.add(row["attempt"])
            rows.append(row)
            all_rows[row["attempt"]] = row
        errors = []
        for path in sorted(directory.glob("error_*.json")):
            try:
                error = read(path)
            except (ValueError, UnicodeDecodeError):
                partial_files.append(path.name)
                continue
            if error.get("run_manifest_sha256") != manifest_sha or error.get("tag") != directory.name:
                raise ValueError("Error-record provenance mismatch")
            case = cases.get(error.get("case"))
            if (case is None or error.get("mode") not in config["modes"] or error.get("suite") != "dev_field"
                    or any(error.get(k) != case[k] for k in ("seed", "scenario_sha256"))
                    or type(error.get("attempt")) is not int or not 1 <= error["attempt"] <= 150
                    or path.name != f"error_{error['attempt']:03d}.json"):
                raise ValueError("Invalid error identity/attempt")
            if error["attempt"] in errors_by_attempt:
                raise ValueError("Duplicate error attempt")
            errors_by_attempt[error["attempt"]] = error
            errors.append(error)
        if complete is not None:
            if complete.get("completed") != len(rows) or complete.get("expected") != config["planned_episodes"] or len(rows) != config["planned_episodes"]:
                raise ValueError(f"Completed tag has missing/extra results: {directory.name}")
            saved = {r["attempt"]: canonical_sha(r) for r in complete.get("rows", [])}
            if len(saved) != len(complete.get("rows", [])) or saved != {r["attempt"]: canonical_sha(r) for r in rows}:
                raise ValueError(f"Completed summary differs from episode files: {directory.name}")
            if complete.get("outcomes") != dict(Counter(r["mode"]+"/"+r["outcome"] for r in rows)):
                raise ValueError("Completed outcome summary mismatch")
        preflight_abort = None
        if (directory / "preflight_aborted.json").exists():
            preflight_abort = read(directory / "preflight_aborted.json")
            if (preflight_abort.get("status") != "aborted_before_reservation"
                    or type(preflight_abort.get("attempts_reserved")) is not int
                    or preflight_abort["attempts_reserved"] != 0
                    or type(preflight_abort.get("episodes_started")) is not int
                    or preflight_abort["episodes_started"] != 0
                    or not isinstance(preflight_abort.get("reason"), str)
                    or not preflight_abort["reason"]
                    or complete is not None or rows or errors
                    or (directory / "started.json").exists()
                    or any(directory.glob("episode_*.json"))
                    or any(directory.glob("trace_*.jsonl"))):
                raise ValueError(f"Preflight abort contradicts execution records: {directory.name}")
        status = "aborted before execution" if preflight_abort is not None else "complete" if complete is not None else "failed" if errors else "stopped" if (directory / "stopped.json").exists() else "started/incomplete" if (directory / "started.json").exists() else "prepared"
        tags.append({"tag": directory.name, "status": status, "planned": config["planned_episodes"],
            "completed": len(rows), "errors": len(errors), "attempts_with_results": sorted(attempts),
            "seconds_completed_sum": sum(r["seconds"] for r in rows), "modes": config["modes"],
            "case_ids": list(cases), "manifest_sha256": manifest_sha,
            "archive_sha256": manifest["archive_sha256"], "configuration_sha256": canonical_sha(config),
            "source_sha256": config["source_sha256"], "settings": config["settings"],
            "partial_files_ignored": partial_files,
            "preflight_abort": preflight_abort,
            "outcomes": mode_summary(rows, config["modes"]) if complete is not None else None,
            "fresh_v6_vs_v4": paired(rows) if complete is not None else None})
    tokens = {}
    for path in sorted((CAMPAIGN / "attempts").glob("*.json")):
        if not re.fullmatch(r"\d{3}\.json", path.name) or not 1 <= int(path.stem) <= 150:
            raise ValueError(f"Unknown budget ledger filename: {path.name}")
        try:
            tokens[int(path.stem)] = read(path)
        except (ValueError, UnicodeDecodeError):
            tokens[int(path.stem)] = None  # Empty/partial reservations count.
    for number, row in all_rows.items():
        validate_token(row, tokens.get(number))
    for number, row in errors_by_attempt.items():
        validate_token(row, tokens.get(number))
    verification_document, verifications = {}, {}
    verification_path = CAMPAIGN / "verification_runs.json"
    if verification_path.exists():
        verification_document = read(verification_path)
        if (not isinstance(verification_document, dict)
                or verification_document.get("kind") != "verification_runs"
                or not isinstance(verification_document.get("records"), list)):
            raise ValueError("Invalid verification runs document")
        for record in verification_document["records"]:
            number = record.get("attempt") if isinstance(record, dict) else None
            row = validate_verification(record, tokens.get(number))
            if number in verifications or number in all_rows or number in errors_by_attempt:
                raise ValueError(f"Duplicate or overlapping verification attempt: {number}")
            verifications[number] = row
    by_tag = Counter(token.get("tag") if isinstance(token, dict) else "unparseable" for token in tokens.values())
    for tag in tags:
        tag["attempted"] = by_tag[tag["tag"]]
        if tag["preflight_abort"] is not None and tag["attempted"]:
            raise ValueError(f"Preflight abort has a consumed attempt token: {tag['tag']}")
        tag_errors = {n for n, row in errors_by_attempt.items() if row["tag"] == tag["tag"]}
        tag["completed_with_error_record"] = len(set(tag["attempts_with_results"]) & tag_errors)
        tag_verifications = {n for n, row in verifications.items() if row["tag"] == tag["tag"]}
        tag["verification_completed"] = len(tag_verifications)
        tag["unresolved_attempts"] = tag["attempted"] - len(set(tag["attempts_with_results"]) | tag_errors | tag_verifications)
    repeats = defaultdict(list)
    for row in all_rows.values():
        repeats[row["mode"], row["case"], row["seed"], row["scenario_sha256"]].append({
            "tag": row["tag"], "attempt": row["attempt"], "manifest_sha256": row["run_manifest_sha256"]})
    repeats = [{"mode": key[0], "case": key[1], "seed": key[2], "scenario_sha256": key[3], "runs": value}
               for key, value in sorted(repeats.items()) if len(value) > 1]
    audit = read(CAMPAIGN / "reference_audit.json")
    history_path = ROOT / "results/safety_dev/dev_v4_observer_memory.csv"
    history = list(csv.DictReader(io.StringIO(raw(history_path).decode("utf-8-sig"))))
    if len(history) != 150 or len({r["case"] for r in history}) != 150 or any(r["outcome"] not in OUTCOMES or r["mode"] != "v4" for r in history):
        raise ValueError("Historical v4 file is not the expected150 unique completed DV3 cases")
    historical = {"scope": "historical150-case development reference; excluded from fresh paired comparisons",
        "episodes": len(history), "outcomes": dict(Counter(r["outcome"] for r in history)),
        "audit": audit["conclusion"], "source_audit": "reference_audit.json"}
    supporting_notes = []
    note_paths = [CAMPAIGN / "finite_suite_bound.md", CAMPAIGN / "integration_audit.json"]
    note_paths += sorted((CAMPAIGN / "offline").glob("*/REPORT.md"))
    for path in note_paths:
        if path.exists():
            supporting_notes.append({"path": path.relative_to(CAMPAIGN).as_posix(),
                                     "sha256": sha(raw(path))})
    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "DV3 development diagnostics; version-specific results, not independent replication or held-out performance",
        "budget": {"hard_cap": 150, "attempted": len(tokens), "remaining": 150-len(tokens),
            "completed": len(all_rows), "errors": len(errors_by_attempt),
            "verification_completed": len(verifications),
            "verification_passed": sum(row["status"] == "passed" for row in verifications.values()),
            "verification_failed": sum(row["status"] == "failed" for row in verifications.values()),
            "completed_with_error_record": len(set(all_rows) & set(errors_by_attempt)),
            "unresolved_or_inflight_attempts": len(tokens)-len(set(all_rows) | set(errors_by_attempt) | set(verifications)),
            "attempts_by_tag": dict(by_tag), "old_quick_campaign_excluded": True},
        "verification_runs": dict(verification_document, records=list(verifications.values()),
            scope="Completed test simulations charged to the shared cap; excluded from all policy outcome and pairing tables."),
        "tags": tags, "planned_repeated_identities": repeats,
        "repeat_note": "Identical case/mode seeds intentionally recur in different frozen experiment manifests. They remain separate; no pooled version success rate is computed.",
        "historical_reference": historical, "input_sha256": input_hashes,
        "supporting_notes": supporting_notes,
        "report_source_sha256": sha(read_shared_bytes(Path(__file__)))}


def markdown(report):
    budget = report["budget"]
    lines = ["# Development safety campaign", "", f"Snapshot: {report['created_utc']}.", "",
        f"**Budget: {budget['attempted']}/150 new attempts consumed; {budget['remaining']} remain.** "
        f"{budget['completed']} durable completed policy results, {budget['errors']} recorded policy-run errors, "
        f"{budget.get('verification_completed', 0)} completed verification test simulations, "
        f"{budget['unresolved_or_inflight_attempts']} unresolved/in-flight reservations. The previous quick76/100 campaign is excluded.", "",
        "Each tag is a separate frozen source/settings version. Only completed tags enter the outcome tables. "
        "Active or failed tags still consume budget. Episode JSON files are authoritative; no missing outcome is inferred. "
        "Runtimes sum completed episode seconds, excluding setup and failed/in-flight execution.", "",
        "| Tag | Status | Attempts | Completed / planned | Errors | Completed seconds |", "|---|---|---:|---:|---:|---:|"]
    for tag in report["tags"]:
        lines.append(f"| {tag['tag']} | {tag['status']} | {tag['attempted']} | {tag['completed']} / {tag['planned']} | {tag['errors']} | {tag['seconds_completed_sum']:.2f} |")
    for tag in report["tags"]:
        if tag.get("preflight_abort") is not None:
            lines += ["", f"`{tag['tag']}` was [aborted before execution](runs/{tag['tag']}/preflight_aborted.json): "
                      f"{tag['preflight_abort']['reason']}. Zero attempts reserved and zero episodes started; "
                      "this is not a policy-run error or a pending evaluation."]
    verification = report.get("verification_runs", {})
    if verification.get("records"):
        lines += ["", "## Verification simulations charged separately", "",
            "These completed test simulations consume budget but are not SAC/DV3 policy evaluations. "
            "Passed and failed assertions both complete their consumed slots; no goal or collision outcome is inferred. "
            "Recorded source hashes identify the test files at execution, without requiring later working copies to remain unchanged.", "",
            "| Attempt | Test | Status | Test source SHA-256 |", "|---:|---|---|---|"]
        for row in verification["records"]:
            lines.append(f"| {row['attempt']} | `{row['test_id']}` | {row['status']} | `{row['test_source_sha256']}` |")
        if verification.get("failure"):
            lines += ["", verification["failure"]]
        lines += ["", "[Verification records and accounting explanation](verification_runs.json)."]
    lines += ["", "## Completed outcomes by frozen tag", "",
        "| Tag | Mode | n | Goals | Obstacles | Boundaries | Targets | Timeouts | Seconds |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in report["tags"]:
        for row in tag["outcomes"] or []:
            lines.append(f"| {tag['tag']} | {row['mode']} | {row['completed']} | {row['goal']} | {row['collision:obstacle']} | {row['collision:boundary']} | {row['collision:target']} | {row['timeout']} | {row['seconds_sum']:.2f} |")
    lines += ["", "## Fresh v6 versus v4 pairs", "", "Only matching case/seed/geometry records within the same tag are paired. Goal gains/losses compare goal against any non-goal outcome. Historical v4 outcomes are excluded.", "",
        "| Tag | Fresh paired cases | V6 gained goals | V6 lost goals |", "|---|---:|---:|---:|"]
    for tag in report["tags"]:
        pair = tag["fresh_v6_vs_v4"]
        if pair is not None:
            lines.append(f"| {tag['tag']} | {pair['paired_cases']} | {pair['gained_goals']} | {pair['lost_goals']} |")
    lines += ["", "## Repeated identities and provenance", "",
        f"{len(report['planned_repeated_identities'])} scenario/mode identities recur across experiment tags. "
        "These planned reruns are listed with seed, scenario digest, tag, attempt and manifest hash in `summary.json`; "
        "they are not additional independent scenarios. Results from different source versions are not pooled.", ""]
    for tag in report["tags"]:
        lines.append(f"- {tag['tag']}: [manifest](runs/{tag['tag']}/manifest.json), [source archive](runs/{tag['tag']}/evaluated_sources.zip); manifest `{tag['manifest_sha256'][:16]}`.")
    history = report["historical_reference"]
    lines += ["", "## Historical reference, separately qualified", "",
        f"Historical selected v4 recorded **{history['outcomes'].get('goal',0)}/150 goals** on the original DV3 set. "
        "This is not a fresh within-tag comparator. The [reference audit](reference_audit.json) confirms the recorded policy/config, "
        "150 seeds/scenario digests and v2–v4 effective constants, but historical `env.py` and `safety_v4.py` source hashes differ. "
        "Exact historical all-source behavioral equivalence is unproven; wrapper differences are reported separately.", "",
        "These are selected development cases used to design and compare candidate methods. No held-out success estimate, "
        "95–100% performance claim, confidence interval, or statistical significance follows from these pilots.", ""]
    if report.get("supporting_notes"):
        lines += ["## Completed analysis and integration notes", ""]
        titles = {"finite_suite_bound.md": "Deterministic full-suite upper bound",
                  "integration_audit.json": "Native v6 integration and source-drift audit"}
        for note in report["supporting_notes"]:
            title = titles.get(note["path"]) or Path(note["path"]).parent.name.replace("_", " ") or Path(note["path"]).stem
            lines.append(f"- [{title}]({note['path']})")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="publish a new snapshot while preserving earlier report versions")
    args = parser.parse_args()
    targets = [CAMPAIGN / "report.md", CAMPAIGN / "summary.json"]
    if any(p.exists() for p in targets) and not args.refresh:
        raise FileExistsError("Report exists; use --refresh to preserve and publish a new snapshot")
    report = collect()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    archive = CAMPAIGN / "reports" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    contents = [markdown(report), json.dumps(report, indent=2, allow_nan=True) + "\n"]
    for target, content in zip(targets, contents):
        with (archive / target.name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, target)
    print(f"Reported {report['budget']['completed']} durable results across {len(report['tags'])} tags; "
          f"budget {report['budget']['attempted']}/150. Snapshot {archive.name}.")


if __name__ == "__main__":
    main()
