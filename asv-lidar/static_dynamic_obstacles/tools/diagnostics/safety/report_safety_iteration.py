"""Validate and report saved safety iterations; never imports a simulator.

python -B tools/diagnostics/safety/report_safety_iteration.py RUN [RUN ...] --output NEW_REPORT_DIR
Use --interim for a read-only JSON snapshot of committed records, without final
claims or files. Final reports require clean completion and exact artifact sets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite_status import read_shared_bytes

ROOT = Path(__file__).resolve().parents[3]
LOADED_REPORT_SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PILOT = ROOT / "results/safety_dev/v9_paired_pilot"
AUDIT = ROOT / "results/safety_dev/v8_followup_offline/audit/all_1150.csv"
BASELINE_AUDIT = ROOT / "results/safety_dev/v10_iterations/baseline_source_audit.json"
NATIVE_DISPATCH_AUDIT = ROOT / "results/safety_dev/v10_iterations/native_dispatch_source_audit.json"
V15_INTEGRATION_AUDIT = ROOT / "results/safety_dev/v10_iterations/v15_integration_source_audit.json"
V15_DIAGNOSTIC_AUDIT = ROOT / "results/safety_dev/v10_iterations/v15_v6_diagnostic_ast_audit.json"
V16_NATIVE_DISPATCH_AUDIT = ROOT / "results/safety_dev/v10_iterations/v16_native_dispatch_source_audit.json"
OUTCOMES = ("goal", "collision:target", "collision:obstacle", "collision:boundary", "timeout")
REFERENCES = ("off", "v8", "v9")
IDENTITY = ("case", "dataset", "seed", "scenario_sha256")
MECHANISMS = ("v9_current_plan_checked", "v9_override_suppressed", "v10_pair_checked", "v10_policy_preserved",
              "v11_policy_prefix_search_checked", "v11_policy_prefix_preserved")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Reader:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = read_shared_bytes(path)
        self.hashes[str(path)] = sha(raw)
        return raw

    def json(self, path):
        return json.loads(self.read(path))


def validate_archive(raw, hashes):
    require(bool(hashes), "Missing archived source inventory")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(hashes), "Source archive inventory mismatch")
        sources = {name: archive.read(name) for name in hashes}
        for name, expected in hashes.items():
            require(sha(sources[name]) == expected, f"Archived source hash mismatch: {name}")
    return sources


def canonical_inventory(reader):
    ts2 = ROOT / "results/test_set/v2/definition.csv"
    dv3 = ROOT / "results/safety_dev/suites/inventory_v5/manifest.json"
    cases = {}
    for row in csv.DictReader(io.StringIO(reader.read(ts2).decode("utf-8"))):
        case = "TS2:" + row["test_id"]
        require(case not in cases, "Duplicate canonical TS2 case")
        cases[case] = {"case": case, "dataset": "ts2", "seed": int(row["episode_seed"]), "scenario_sha256": row["digest"]}
    ts3 = ROOT / "results/test_set/v3/definition.csv"
    if ts3.exists():
        for row in csv.DictReader(io.StringIO(reader.read(ts3).decode("utf-8"))):
            case = "TS3:" + row["test_id"]
            require(case not in cases, "Duplicate canonical TS3 case")
            cases[case] = {"case": case, "dataset": "ts3", "seed": int(row["episode_seed"]), "scenario_sha256": row["digest"]}
    for row in reader.json(dv3)["cases"]:
        if row["suite"] == "dev_field":
            case = "DV3:" + row["case"]
            require(case not in cases, "Duplicate canonical DV3 case")
            cases[case] = {"case": case, "dataset": "dv3", "seed": int(row["seed"]), "scenario_sha256": row["scenario_sha256"]}
    return cases


def validate_canonical_cases(run, canonical):
    for case, row in run["cases"].items():
        require(case in canonical and all(row[k] == canonical[case][k] for k in IDENTITY), "Canonical scene/seed mismatch")


def validate_trace(raw, result, reason_default):
    require(raw.endswith(b"\n"), "Incomplete completed trace tail")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require([r["step"] for r in rows] == list(range(1, result["steps"] + 1)), "Trace step order/count mismatch")
    for row in rows:
        details = row["filter"]
        require(not any(details.get(k) for k in ("error", "exception", "filter_error", "safety_error")), "Filter error fallback recorded")
        reason = str(details.get("why", "")).lower()
        require(details.get("mode") != "error" and reason not in ("error", "filter error")
                and not reason.startswith(("error:", "filter error:")), "Filter error fallback reason recorded")
        require(all(math.isfinite(float(row[k])) for k in ("rudder_command", "signed_rpm_command")), "Nonfinite executed command")
        require(type(row["changed"]) is bool and type(row["brake"]) is bool, "Invalid trace event flag")
        require(all(type(row["filter"][k]) is bool for k in MECHANISMS if k in row["filter"]), "Invalid mechanism flag")
        for key, size in (("policy_action", 2), ("pre_state", 4), ("post_state", 4)):
            require(len(row[key]) == size and all(math.isfinite(float(x)) for x in row[key]), f"Invalid trace {key}")
    changed = sum(row["changed"] for row in rows)
    brakes = sum(row["brake"] for row in rows)
    require(changed == result["safety_v2_steps"], "Changed-action counter mismatch")
    require(brakes == result["safety_v2_brake_steps"], "Brake counter mismatch")
    reasons = dict(Counter(row["filter"].get("why", reason_default) for row in rows))
    require(reasons == result["why_counts"], "Decision-reason counter mismatch")
    mechanisms = {key: sum(bool(row["filter"].get(key, False)) for row in rows) for key in MECHANISMS}
    observed = {key: sum(key in row["filter"] for row in rows) for key in MECHANISMS}
    for result_key, trace_key in (("checked_steps", "v9_current_plan_checked"), ("suppressed_steps", "v9_override_suppressed")):
        if result_key in result:
            require(result[result_key] == mechanisms[trace_key], f"{result_key} counter mismatch")
    persistence = {"observed_decisions": 0, "admission_events": 0, "measured_refresh_events": 0,
                   "added_hypothesis_decisions": 0, "coasted_hypothesis_decisions": 0,
                   "expired_source_events": 0, "contradicted_source_events": 0}
    prefix = {"observed_decisions": 0, "evaluated_plans": 0, "accepted_plans": 0, "hard_passing_plans": 0}
    requests = {"observed_decisions": 0, "checked_observed_decisions": 0, "checked_decisions": 0, "request_reason_counts": Counter(), "skip_reason_counts": Counter()}
    geometry = {"observed_decisions": 0, "replacement_decisions": 0, "replacement_id_events": 0, "rejection_reason_counts": Counter()}
    for row in rows:
        details = row["filter"]
        def diagnostic(suffix):
            values = [value for key, value in details.items() if key == suffix or re.fullmatch(r"v\d+_" + suffix, key)]
            require(not values or all(value == values[0] for value in values), "Conflicting versioned diagnostic aliases")
            return (values[0], True) if values else (None, False)
        checked, checked_present = diagnostic("prefix_request_checked")
        reason, reason_present = diagnostic("prefix_request_reason")
        skipped, skipped_present = diagnostic("prefix_skip_reason")
        if checked_present or reason_present or skipped_present:
            requests["observed_decisions"] += 1
        if checked_present:
            require(type(checked) is bool, "Invalid prefix request flag")
            requests["checked_observed_decisions"] += 1
            requests["checked_decisions"] += checked
        for value, present, key in ((reason, reason_present, "request_reason_counts"), (skipped, skipped_present, "skip_reason_counts")):
            if present:
                require(isinstance(value, str) and bool(value), "Invalid prefix request/skip reason")
                requests[key][value] += 1
        motion = details.get("motion_axis_geometry")
        if motion:
            require(isinstance(motion, dict), "Invalid motion-axis diagnostics")
            ids, updates = motion["replaced_source_ids"], motion["updates"]
            require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids)
                    and len(ids) == len(set(ids)), "Invalid motion-axis replacement IDs")
            require(sorted(u["source_id"] for u in updates) == sorted(ids), "Motion-axis replacement/update mismatch")
            require(all(type(v) is int and v >= 0 for v in motion["rejections"].values()), "Invalid motion-axis rejection counts")
            geometry["observed_decisions"] += 1
            geometry["replacement_decisions"] += bool(ids)
            geometry["replacement_id_events"] += len(ids)
            geometry["rejection_reason_counts"].update(motion["rejections"])
        stats = row["filter"].get("track_persistence")
        if stats:
            require(type(stats["added_hypotheses"]) is int and stats["added_hypotheses"] >= 0, "Invalid persistence hypothesis count")
            hypotheses = stats["hypotheses"]
            require(len(hypotheses) == stats["added_hypotheses"], "Persistence hypothesis count mismatch")
            require(all(math.isfinite(float(h["age_s"])) and h["age_s"] >= 0 for h in hypotheses), "Invalid hypothesis age")
            require(all(u["kind"] in ("admission", "measured_refresh") for u in stats["updates"]), "Unknown persistence update kind")
            persistence["observed_decisions"] += 1
            persistence["admission_events"] += sum(u["kind"] == "admission" for u in stats["updates"])
            persistence["measured_refresh_events"] += sum(u["kind"] == "measured_refresh" for u in stats["updates"])
            persistence["added_hypothesis_decisions"] += len(hypotheses)
            persistence["coasted_hypothesis_decisions"] += sum(h["age_s"] > 0 for h in hypotheses)
            persistence["expired_source_events"] += len(stats["expired_source_ids"])
            persistence["contradicted_source_events"] += len(stats["contradicted_source_ids"])
        stats = row["filter"].get("policy_prefix_search")
        if stats:
            prefix["observed_decisions"] += 1
            for key in ("evaluated_plans", "accepted_plans", "hard_passing_plans"):
                require(type(stats[key]) is int and stats[key] >= 0, "Invalid prefix plan counter")
                prefix[key] += stats[key]
    return {"changed_action_steps": changed, "brake_steps": brakes, "decisions": len(rows),
            "mechanism_steps": mechanisms, "mechanism_observed_steps": observed,
            "track_persistence": persistence, "policy_prefix_search": prefix, "decision_reasons": reasons,
            "prefix_requests": requests, "motion_axis_geometry": geometry,
            "filter_error_steps": 0}


def load_run(directory, reader, interim=False):
    directory = Path(directory).resolve()
    manifest = reader.json(directory / "manifest.json")
    source = Path(manifest.get("selection_source", directory / "selection.json"))
    if not source.is_absolute():
        source = ROOT / source
    selection_raw = reader.read(source)
    require(sha(selection_raw) == manifest["selection_sha256"], "Selection hash mismatch")
    selection = json.loads(selection_raw)
    all_cases = {r["case"]: r for r in selection["cases"]}
    require(len(all_cases) == len(selection["cases"]), "Duplicate source selection case")
    cases = {r["case"]: r for r in manifest["cases"]}
    require(cases and len(cases) == len(manifest["cases"]), "Empty/duplicate manifest case")
    require(all(case in all_cases and row == all_cases[case] for case, row in cases.items()), "Manifest differs from canonical selection")
    modes = manifest["modes"]
    require(modes and len(set(modes)) == len(modes), "Empty/duplicate modes")
    expected = {(case, mode) for case in cases for mode in modes}
    require(manifest["planned_runs"] == len(expected), "Planned count mismatch")
    for key in ("checkpoint_sha256", "config_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", manifest[key]) is not None, f"Invalid {key}")
    archived_sources = validate_archive(reader.read(directory / "evaluated_sources.zip"), manifest["source_sha256"])

    completion_path = directory / "completion.json"
    completion = reader.json(completion_path) if completion_path.exists() else None
    require(interim or completion is not None, "Final report requires completion.json")
    if completion is not None:
        require(not completion["source_drift"] and completion["checkpoint_unchanged"] is True
                and completion["selection_unchanged"] is True, "Completion reports frozen-input drift")
        require(completion["completed_runs"] == len(expected), "Completion count mismatch")
        require(math.isfinite(completion["elapsed_s"]) and completion["elapsed_s"] >= 0, "Invalid completion duration")
    strict = completion is not None or not interim

    # Capture committed result names first: a live process may reserve a newer
    # token afterward. Partial traces are never interpreted as completed runs.
    attempt_dir = directory / "attempts"
    result_paths = sorted(attempt_dir.glob("*_result.json"))
    token_paths = sorted(p for p in attempt_dir.glob("*.json") if re.fullmatch(r"\d+\.json", p.name))
    tokens, results = {}, {}
    for path in token_paths:
        token = reader.json(path)
        attempt = token["attempt"]
        require(type(attempt) is int and path.name == f"{attempt:03d}.json" and attempt not in tokens, "Invalid/duplicate token filename")
        require((token["case"], token["mode"]) in expected, "Unknown token identity")
        require(token["seed"] == cases[token["case"]]["seed"], "Token seed mismatch")
        tokens[attempt] = token
    require(set(tokens) == set(range(1, len(tokens) + 1)) and len(tokens) <= len(expected), "Token sequence is incomplete or excessive")
    require(len({(r["case"], r["mode"]) for r in tokens.values()}) == len(tokens), "Duplicate case/mode token")
    statistics = {}
    default_reason = "idle" if "selection_source" in manifest else "off"
    for path in result_paths:
        result = reader.json(path)
        attempt, key = result["attempt"], (result["case"], result["mode"])
        require(path.name == f"{attempt:03d}_result.json" and attempt in tokens, "Result filename/token mismatch")
        require(key not in results and key in expected, "Unknown/duplicate result identity")
        require(all(result[k] == tokens[attempt][k] for k in ("attempt", "case", "mode", "seed")), "Token/result identity mismatch")
        case = cases[key[0]]
        require(result["dataset"] == case["dataset"] and result["stratum"] == case.get("selection_stratum", "explicit"), "Result selection metadata mismatch")
        require(result["outcome"] in OUTCOMES, "Unknown outcome")
        require(type(result["collided"]) is bool and result["collided"] == result["outcome"].startswith("collision:"), "Collision flag mismatch")
        require(type(result["collided_target"]) is bool and result["collided_target"] == (result["outcome"] == "collision:target"), "Target flag mismatch")
        require(type(result["steps"]) is int and result["steps"] > 0, "Invalid decision count")
        require(math.isfinite(result["elapsed_s"]) and result["elapsed_s"] >= 0, "Invalid episode duration")
        if result["mode"] == "off":
            require(result["safety_v2_steps"] == result["safety_v2_brake_steps"] == 0,
                    "Fresh off control recorded a safety intervention")
            if "filter_classes" in manifest:
                require(manifest["filter_classes"].get("off") is None,
                        "Fresh off control declares an enabled filter class")
        trace = directory / "traces" / f"{attempt:03d}_{result['mode']}.jsonl"
        statistics[key] = validate_trace(reader.read(trace), result, default_reason)
        results[key] = result
    if strict:
        require(set(results) == expected and len(tokens) == len(expected), "Missing completed result/token")
        expected_attempt_files = {f"{i:03d}{suffix}.json" for i in tokens for suffix in ("", "_result")}
        require({p.name for p in attempt_dir.iterdir() if p.is_file()} == expected_attempt_files, "Extra/missing attempt artifact")
        expected_traces = {f"{r['attempt']:03d}_{r['mode']}.jsonl" for r in results.values()}
        require({p.name for p in (directory / "traces").iterdir() if p.is_file()} == expected_traces, "Extra/missing trace artifact")
    return {"directory": str(directory), "tag": directory.name, "manifest": manifest, "selection": selection,
            "cases": cases, "results": results, "statistics": statistics, "completion": completion,
            "attempted": len(tokens), "complete": completion is not None, "archived_sources": archived_sources}


def compatibility(candidate, reference, case):
    a, b = candidate["manifest"], reference["manifest"]
    for key in ("checkpoint_sha256", "config_sha256", "low_speed_start_frac"):
        require(a[key] == b[key], f"Fresh reference {key} mismatch")
    require(all(candidate["cases"][case][k] == reference["cases"][case][k] for k in IDENTITY), "Fresh reference scene/seed identity mismatch")
    ca = a.get("constants", a.get("effective_constants"))
    cb = b.get("constants", b.get("effective_constants"))
    require(ca is not None and cb is not None and ca == cb, "Fresh reference effective constants mismatch")
    return {name: {"candidate": a["source_sha256"].get(name), "reference": b["source_sha256"].get(name)}
            for name in sorted(set(a["source_sha256"]) | set(b["source_sha256"]))
            if a["source_sha256"].get(name) != b["source_sha256"].get(name)}


def select_run_mode(run, mode):
    """View one mode only after every artifact in the complete run was checked.

    Preserve the full source-run exposure and duration in metadata. The source
    manifest/files are never rewritten; attempt IDs keep their original value.
    """
    require(run["complete"], "Mode selection requires a complete validated run")
    require(mode in run["manifest"]["modes"], "Selected mode is absent from run")
    view = copy.deepcopy(run)
    source_modes = list(run["manifest"]["modes"])
    view["results"] = {key: value for key, value in run["results"].items() if key[1] == mode}
    view["statistics"] = {key: value for key, value in run["statistics"].items() if key[1] == mode}
    require({key[0] for key in view["results"]} == set(run["cases"]), "Selected mode lacks a case")
    view["manifest"]["modes"] = [mode]
    for key in ("filter_classes", "constructor_options"):
        if run["manifest"].get(key) is not None:
            require(mode in run["manifest"][key], f"Selected mode lacks {key}")
            view["manifest"][key] = {mode: run["manifest"][key][mode]}
    view["manifest"]["planned_runs"] = len(view["results"])
    view["attempted"] = len(view["results"])
    view["completion"]["completed_runs"] = len(view["results"])
    view["mode_selection"] = {"selected_mode": mode, "source_modes": source_modes,
                              "excluded_modes": [m for m in source_modes if m != mode],
                              "source_planned_runs": run["manifest"]["planned_runs"],
                              "source_completed_runs": len(run["results"]), "source_attempted": run["attempted"],
                              "selected_completed_runs": len(view["results"]),
                              "duration_scope": "Full original source-run duration, including any omitted modes"}
    return view


def load_reference(spec, reader):
    """PATH, MODE=PATH or LABEL::PATH/MODE=PATH; validate before labeling."""
    text = str(spec)
    if "::" in text and not Path(text).exists():
        label, source = text.split("::", 1)
        require(re.fullmatch(r"[a-z][a-z0-9_]*", label) is not None and source, "Use LABEL::PATH for a reference label")
        run = load_reference(source, reader)
        require(len(run["manifest"]["modes"]) == 1, "Reference labels require one selected mode")
        mode = run["manifest"]["modes"][0]
        require(label not in REFERENCES, "Reference label conflicts with reserved pilot mode")
        run["reference_label"] = {"label": label, "original_mode": mode}
        run["manifest"]["modes"] = [label]
        for key in ("filter_classes", "constructor_options"):
            if key in run["manifest"]:
                run["manifest"][key] = {label: run["manifest"][key][mode]}
        run["results"] = {(case, label): result for (case, _), result in run["results"].items()}
        run["statistics"] = {(case, label): stats for (case, _), stats in run["statistics"].items()}
        return run
    if "=" in text and not Path(text).exists():
        mode, path = text.split("=", 1)
        require(re.fullmatch(r"[a-z][a-z0-9_]*", mode) is not None and path, "Use MODE=PATH for reference mode selection")
        return select_run_mode(load_run(Path(path), reader), mode)
    return load_run(Path(spec), reader)


def inactive_version_audit(path, source_hashes, shared_sources, classes):
    match = re.fullmatch(r"src/safety_v(\d+)\.py", path)
    require(match is not None, "Inactive-source allowance only covers conventional safety_vN.py modules")
    versions = []
    for cls in classes.values():
        if cls is None:
            continue
        selected = re.search(r"\.SafetyFilterV(\d+)$", cls)
        require(selected is not None, "Cannot establish selected filter version for inactive-source audit")
        versions.append(int(selected.group(1)))
    added = int(match.group(1))
    require(versions and added > max(versions), "Inactive source version must exceed every selected filter class")
    literal = re.compile(r"(?<![A-Za-z0-9_])(?:safety_v" + str(added) + r"|SafetyFilterV" + str(added) + r")(?![A-Za-z0-9_])")
    references = [name for name, raw in shared_sources.items() if name.endswith(".py") and literal.search(raw.decode("utf-8-sig"))]
    require(not references, "Inactive added module is referenced by shared archived sources: " + ", ".join(references))
    return {"path": path, "sha256": source_hashes[path], "added_version": added,
            "selected_filter_versions": sorted(set(versions)), "shared_python_sources_scanned": len(shared_sources),
            "literal_module_or_class_references": references,
            "scope": "Explicit opt-in: unchanged common code, higher conventional version, and no literal shared-source reference. Not a general proof about arbitrary dynamic imports."}


def combine_cohort(runs, name, allow_inactive_version_additions=False):
    """Join disjoint completed slices of one frozen controller configuration."""
    require(len(runs) >= 2 and name and Path(name).name == name, "Cohort needs at least two runs and a simple name")
    first = runs[0]
    manifest = copy.deepcopy(first["manifest"])
    require(manifest.get("filter_classes") is not None and manifest.get("constructor_options") is not None,
            "Cohort requires explicit filter classes/options")
    cases, results, statistics, origins, components = {}, {}, {}, {}, []
    source_bytes = dict(first["archived_sources"])
    allowed_extra = {"tools/diagnostics/safety/trace_snapshot.py"}
    for run in runs:
        require(run["complete"], "Cohort requires complete component runs")
        current = run["manifest"]
        for key in ("modes", "filter_classes", "constructor_options", "checkpoint_sha256", "config_sha256",
                    "low_speed_start_frac", "filter_installation", "torch_threads", "native_thread_environment", "threadpools"):
            require(current.get(key) == manifest.get(key), f"Cohort {key} mismatch")
        require(current.get("constants", current.get("effective_constants")) == manifest.get("constants", manifest.get("effective_constants")),
                "Cohort effective constants mismatch")
        a, b = manifest["source_sha256"], current["source_sha256"]
        common = set(a) & set(b)
        require(all(a[key] == b[key] for key in common), "Cohort shared source hash mismatch")
        extra = set(a) ^ set(b)
        exceptional = extra - allowed_extra
        require(not exceptional or allow_inactive_version_additions, "Cohort source inventory differs beyond optional snapshot helper")
        inactive = []
        for path in sorted(exceptional):
            shared = {key: source_bytes[key] for key in common}
            inactive.append(inactive_version_audit(path, dict(a, **b), shared, manifest["filter_classes"]))
        require(not (set(cases) & set(run["cases"])), "Duplicate scenario across cohort components")
        cases.update(run["cases"])
        results.update(run["results"])
        statistics.update(run["statistics"])
        origins.update({key: {"tag": run["tag"], "directory": run["directory"]} for key in run["results"]})
        components.append({"tag": run["tag"], "directory": run["directory"], "completed": len(run["results"]),
                           "distinct_cases": len(run["cases"]), "snapshots": current.get("snapshots"),
                           "mode_selection": run.get("mode_selection"),
                           "source_inventory_difference_from_first": sorted(set(first["manifest"]["source_sha256"]) ^ set(b)),
                           "inactive_version_source_audit": inactive,
                           "verified_common_source_hashes": len(common)})
        manifest["source_sha256"].update(current["source_sha256"])
        source_bytes.update(run["archived_sources"])
    manifest["cases"] = list(cases.values())
    manifest["planned_runs"] = sum(r["manifest"]["planned_runs"] for r in runs)
    return {"directory": "cohort:" + name, "tag": name, "manifest": manifest,
            "selection": {"cases": list(cases.values())}, "cases": cases, "results": results,
            "statistics": statistics, "origin_by_key": origins, "cohort_components": components,
            "allow_inactive_version_additions": allow_inactive_version_additions,
            "archived_sources": source_bytes,
            "completion": {"completed_runs": len(results), "elapsed_s": sum(r["completion"]["elapsed_s"] for r in runs),
                           "source_drift": [], "checkpoint_unchanged": True, "selection_unchanged": True},
            "attempted": sum(r["attempted"] for r in runs), "complete": True}


def reference_for(run, pilot, history, case, reference, candidate_mode, reference_runs=()):
    # A same-iteration control is freshest, except comparing a mode with itself
    # would hide baseline reproduction failures. Use the archived pilot there.
    key = (case, reference)
    if reference != candidate_mode and key in run["results"]:
        return run["results"][key]["outcome"], "fresh_same_iteration", run["directory"]
    matches = [r for r in reference_runs if key in r["results"]]
    require(len(matches) <= 1, "Ambiguous fresh reference case/mode; specify only one controller configuration")
    if matches:
        source = matches[0]
        compatibility(run, source, case)
        return source["results"][key]["outcome"], "fresh_reference_iteration", source["directory"]
    if pilot is not None and key in pilot["results"]:
        compatibility(run, pilot, case)
        return pilot["results"][key]["outcome"], "fresh_prior_pilot", pilot["directory"]
    if reference not in ("off", "v8"):
        return "", "unavailable", ""
    # v3 is a distinct benchmark, even where its test IDs match v2.
    # Only explicitly matched fresh records above can provide its references.
    if case.startswith("TS3:"):
        return "", "unavailable", ""
    row, archived = run["cases"][case], history.get(case)
    require(archived is not None and int(archived["seed"]) == int(row["seed"]), "Historical reference missing/seed mismatch")
    key = "policy_outcome" if reference == "off" else reference + "_outcome"
    value = row["historical"][key]
    require(value == archived[key] and value in OUTCOMES, "Historical selection/audit outcome mismatch")
    return value, "historical_saved_branch", str(AUDIT)


def paired_counts(rows, reference):
    usable = [r for r in rows if r[reference + "_outcome"]]
    gains = [r["case"] for r in usable if r["outcome"] == "goal" and r[reference + "_outcome"] != "goal"]
    losses = [r["case"] for r in usable if r["outcome"] != "goal" and r[reference + "_outcome"] == "goal"]
    return {"matched_cases": len(usable), "gained_goals": len(gains), "lost_goals": len(losses),
            "net_goals": len(gains) - len(losses), "gained_cases": gains, "lost_cases": losses,
            "reference_goals": sum(r[reference + "_outcome"] == "goal" for r in usable),
            "preserved_reference_goals": sum(r[reference + "_outcome"] == r["outcome"] == "goal" for r in usable),
            "reference_failures": sum(r[reference + "_outcome"] != "goal" for r in usable),
            "unresolved_reference_failures": sum(r[reference + "_outcome"] != "goal" and r["outcome"] != "goal" for r in usable),
            "collision_to_goal": sum(r[reference + "_outcome"].startswith("collision:") and r["outcome"] == "goal" for r in usable),
            "collision_to_timeout": sum(r[reference + "_outcome"].startswith("collision:") and r["outcome"] == "timeout" for r in usable),
            "transitions": dict(Counter(r[reference + "_outcome"] + " -> " + r["outcome"] for r in usable))}


def summarize(run, pilot, history, reference_runs=()):
    references = tuple(dict.fromkeys(REFERENCES + tuple(m for r in reference_runs for m in r["manifest"]["modes"])))
    paired = []
    for (case, mode), result in sorted(run["results"].items()):
        record = run["cases"][case]
        row = {"tag": run["tag"], "mode": mode, "case": case, "dataset": record["dataset"],
               "stratum": record.get("selection_stratum", "explicit"), "seed": record["seed"],
               "scenario_sha256": record["scenario_sha256"], "outcome": result["outcome"],
               "steps": result["steps"], "elapsed_s": result["elapsed_s"], "changed_action_steps": result["safety_v2_steps"]}
        origin = run.get("origin_by_key", {}).get((case, mode), {"tag": run["tag"], "directory": run["directory"]})
        row.update(source_iteration=origin["tag"], source_directory=origin["directory"])
        for reference in references:
            value, evidence, path = reference_for(run, pilot, history, case, reference, mode, reference_runs)
            row.update({reference + "_outcome": value, reference + "_evidence": evidence, reference + "_source": path,
                        reference + "_gained_goal": bool(value and result["outcome"] == "goal" and value != "goal"),
                        reference + "_lost_goal": bool(value == "goal" and result["outcome"] != "goal")})
        paired.append(row)
    groups = []
    for mode in run["manifest"]["modes"]:
        selected = [r for r in paired if r["mode"] == mode]
        for dimension in ("overall", "dataset", "stratum"):
            names = ["all"] if dimension == "overall" else sorted({r[dimension] for r in selected})
            for name in names:
                rows = selected if dimension == "overall" else [r for r in selected if r[dimension] == name]
                comparisons = []
                for reference in references:
                    for evidence in sorted({r[reference + "_evidence"] for r in rows} - {"unavailable"}):
                        subset = [r for r in rows if r[reference + "_evidence"] == evidence]
                        comparisons.append(dict(reference=reference, evidence=evidence, **paired_counts(subset, reference)))
                groups.append({"mode": mode, "dimension": dimension, "slice": name, "n": len(rows),
                               "outcomes": {v: sum(r["outcome"] == v for r in rows) for v in OUTCOMES}, "comparisons": comparisons})
    mechanisms = {}
    for mode in run["manifest"]["modes"]:
        stats = [s for (case, m), s in run["statistics"].items() if m == mode]
        reasons = Counter()
        for value in stats:
            reasons.update(value["decision_reasons"])
        def nested_counts(name):
            observed = sum(s[name]["observed_decisions"] for s in stats)
            keys = set().union(*(s[name] for s in stats))
            return {"observed_episodes": sum(s[name]["observed_decisions"] > 0 for s in stats),
                    **{key: sum(s[name][key] for s in stats) if observed or key == "observed_decisions" else None for key in sorted(keys)}}

        def diagnostic_counts(name):
            observed = sum(s[name]["observed_decisions"] for s in stats)
            result = {"observed_decisions": observed, "observed_episodes": sum(s[name]["observed_decisions"] > 0 for s in stats)}
            for key in set().union(*(s[name] for s in stats)) - {"observed_decisions"}:
                if key.endswith("_counts"):
                    counts = Counter()
                    for s in stats: counts.update(s[name][key])
                    result[key] = dict(counts) if observed else None
                else:
                    result[key] = sum(s[name][key] for s in stats) if observed else None
            if name == "motion_axis_geometry":
                result["replacement_episodes"] = sum(s[name]["replacement_decisions"] > 0 for s in stats) if observed else None
            if name == "prefix_requests" and not result["checked_observed_decisions"]:
                result["checked_decisions"] = None
            return result

        mechanisms[mode] = {"changed_action_steps": sum(s["changed_action_steps"] for s in stats),
                            "brake_steps": sum(s["brake_steps"] for s in stats),
                            "decisions": sum(s["decisions"] for s in stats), "decision_reasons": dict(reasons),
                            "events": {key: {"steps": sum(s["mechanism_steps"][key] for s in stats) if any(s["mechanism_observed_steps"][key] for s in stats) else None,
                                              "episodes": sum(s["mechanism_steps"][key] > 0 for s in stats) if any(s["mechanism_observed_steps"][key] for s in stats) else None,
                                              "observed_steps": sum(s["mechanism_observed_steps"][key] for s in stats),
                                              "observed_episodes": sum(s["mechanism_observed_steps"][key] > 0 for s in stats)} for key in MECHANISMS},
                            "track_persistence": nested_counts("track_persistence"),
                            "policy_prefix_search": nested_counts("policy_prefix_search"),
                            "prefix_requests": diagnostic_counts("prefix_requests"),
                            "motion_axis_geometry": diagnostic_counts("motion_axis_geometry")}
    differences = {}
    if pilot is not None:
        overlap = set(run["cases"]) & set(pilot["cases"])
        if overlap:
            differences = compatibility(run, pilot, sorted(overlap)[0])
    reference_provenance = {}
    for source in reference_runs:
        overlap = set(run["cases"]) & set(source["cases"])
        if overlap:
            label = source["tag"] + ("[" + source["mode_selection"]["selected_mode"] + "]" if source.get("mode_selection") else "")
            reference_provenance[label] = {
                "directory": source["directory"], "overlapping_cases": len(overlap),
                "source_differences": compatibility(run, source, sorted(overlap)[0]),
                "filter_classes": source["manifest"].get("filter_classes"),
                "constructor_options": source["manifest"].get("constructor_options")}
            reference_provenance[label]["mode_selection"] = source.get("mode_selection")
            reference_provenance[label]["reference_label"] = source.get("reference_label")
    return paired, {"tag": run["tag"], "directory": run["directory"], "complete": run["complete"],
                    "planned_runs": run["manifest"]["planned_runs"], "attempted": run["attempted"],
                    "completed": len(run["results"]), "distinct_cases": len({key[0] for key in run["results"]}),
                    "filter_error_steps": 0,
                    "groups": groups, "mechanisms": mechanisms,
                    "broken_sac_success_inventory": [{
                        "mode": row["mode"], "case": row["case"], "dataset": row["dataset"],
                        "stratum": row["stratum"], "outcome": row["outcome"],
                        "off_evidence": row["off_evidence"], "off_source": row["off_source"]}
                        for row in paired if row["mode"] != "off" and row["off_lost_goal"]],
                    "episode_seconds": sum(r["elapsed_s"] for r in paired),
                    "campaign_seconds": None if run["completion"] is None else run["completion"]["elapsed_s"],
                    "pilot_source_differences": differences,
                    "fresh_reference_provenance": reference_provenance,
                    "cohort_components": run.get("cohort_components"),
                    "mode_selection": run.get("mode_selection"),
                    "allow_inactive_version_additions": run.get("allow_inactive_version_additions", False),
                    "checkpoint_sha256": run["manifest"]["checkpoint_sha256"],
                    "config_sha256": run["manifest"]["config_sha256"],
                    "filter_classes": run["manifest"].get("filter_classes"),
                    "constructor_options": run["manifest"].get("constructor_options")}


def primary_benchmark_scope(runs, canonical):
    selected = set().union(*(set(run["cases"]) for run in runs))
    if not selected or not all(case.startswith("TS3:") for case in selected):
        return None
    full = {case for case in canonical if case.startswith("TS3:")}
    return {"name": "Primary test set v3", "defined_scenarios": len(full),
            "distinct_selected_scenarios": len(selected),
            "iterations": [{"tag": run["tag"], "selected_scenarios": len(run["cases"]),
                            "coverage": "full set" if set(run["cases"]) == full else "subset"}
                           for run in runs]}


def markdown(summary):
    benchmark = summary.get("primary_benchmark")
    description = "primary test set v3 simulation episodes" if benchmark else "deliberately selected development episodes"
    lines = ["# Saved safety-iteration report", "", "All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. "
             f"These finite {description} do not establish 100% safety, preservation of every SAC success, "
             "a population success rate, or real-world performance. No episodes were run to create this report.", "",
             "Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. "
             "Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; "
             "identity checks alone do not establish numerical equivalence of changed source or native-thread settings.", ""]
    if benchmark:
        lines += [f"Primary benchmark: **test set v3**, containing {benchmark['defined_scenarios']} frozen scenarios. "
                  "TS3 identities remain separate from test set v2; unavailable fresh references are left unreported.", ""]
        for item in benchmark["iterations"]:
            lines.append(f"- {item['tag']}: **{item['coverage']}**, {item['selected_scenarios']}/{benchmark['defined_scenarios']} scenarios per evaluated mode.")
        lines.append("")
    lines += ["| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in summary["iterations"]:
        for group in run["groups"]:
            if group["dimension"] == "overall":
                lines.append(f"| {run['tag']} | {group['mode']} | {group['n']} | " + " | ".join(str(group["outcomes"][v]) for v in OUTCOMES) + " |")
    lines += ["", "| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |",
              "|---|---|---|---:|---:|---:|---:|"]
    for run in summary["iterations"]:
        for group in run["groups"]:
            if group["dimension"] != "overall":
                continue
            for pair in group["comparisons"]:
                lines.append(f"| {run['tag']} / {group['mode']} | {pair['reference']} | {pair['evidence']} | {pair['matched_cases']} | {pair['gained_goals']} | {pair['lost_goals']} | {pair['net_goals']:+d} |")
    lines += ["", "A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. "
              "Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators "
              "are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, "
              "not independent replications or extra coverage; they are never pooled into a success-rate estimate.", ""]
    lines += ["| Iteration / mode | Selection stratum | SAC evidence | Matched | Preserved / SAC goals | Rescued / SAC failures | Broken SAC goals | Unresolved failures |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for run in summary["iterations"]:
        for group in run["groups"]:
            if group["dimension"] != "stratum" or group["mode"] == "off":
                continue
            for pair in group["comparisons"]:
                if pair["reference"] == "off":
                    lines.append(f"| {run['tag']} / {group['mode']} | {group['slice']} | {pair['evidence']} | "
                        f"{pair['matched_cases']} | {pair['preserved_reference_goals']}/{pair['reference_goals']} | "
                        f"{pair['gained_goals']}/{pair['reference_failures']} | {pair['lost_goals']} | {pair['unresolved_reference_failures']} |")
    lines += ["", "Named broken-SAC-success inventory (the comparison evidence is identified for every case):", ""]
    broken = [(run["tag"], row) for run in summary["iterations"] for row in run["broken_sac_success_inventory"]]
    lines += ([f"- {tag} / {row['mode']}: {row['case']} → {row['outcome']} ({row['stratum']}; {row['off_evidence']})."
               for tag, row in broken] if broken else ["- None among the matched cases."])
    lines += [""]
    lines += [f"Across the included tags there are {summary['completed_iteration_records']} completed iteration records "
              f"covering {summary['distinct_cases']} distinct canonical scenarios. Reference-pilot records are comparison evidence "
              "and are not added to the new iteration count.", ""]
    if summary.get("baseline_source_audit"):
        lines += ["The separate [baseline source audit](<" + Path(summary["baseline_source_audit"]["path"]).as_posix() + ">) records the rationale "
                  "and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.", ""]
    if summary.get("native_dispatch_source_audit"):
        lines += ["The [native-dispatch source audit](<" + Path(summary["native_dispatch_source_audit"]["path"]).as_posix()
                  + ">) records the later environment-selector integration. The evaluated archives remain authoritative; "
                  "this audit is linked because an environment source hash differs across the compared runs.", ""]
    for key, label in (("v15_integration_source_audit", "V15 integration source audit"),
                       ("v15_v6_diagnostic_ast_audit", "V6 diagnostic-only AST audit"),
                       ("v16_native_dispatch_source_audit", "V16 native-dispatch source audit")):
        if summary.get(key):
            lines += ["The [" + label + "](<" + Path(summary[key]["path"]).as_posix()
                      + ">) addresses recorded source differences across these runs. Its hash is preserved in provenance; "
                      "the evaluated source archives remain authoritative.", ""]
    for run in summary["iterations"]:
        clock_label = "sum of component runner durations" if run["cohort_components"] else "runner wall time"
        if run["mode_selection"] or any(c.get("mode_selection") for c in (run["cohort_components"] or [])):
            clock_label += " (full source runs, including omitted modes)"
        lines += [f"## {run['tag']}", "", f"{run['completed']}/{run['planned_runs']} completed, {run['attempted']} reserved attempts, {run['distinct_cases']} distinct scenarios. "
                  f"Timed episodes sum to {run['episode_seconds']:.2f} s; {clock_label} {run['campaign_seconds']:.2f} s. "
                  "Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.", "",
                  "| Mode | Decisions | Changed-action steps | Requested-brake steps |",
                  "|---|---:|---:|---:|"]
        for mode, stats in run["mechanisms"].items():
            lines.append(f"| {mode} | {stats['decisions']} | {stats['changed_action_steps']} | {stats['brake_steps']} |")
        lines += ["", "Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. "
                  "Requested braking does not imply a particular signed RPM; the trace records the executed command. "
                  "Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision "
                  "counts include repeated predictions of the same vessel; they are not counts of distinct vessels.", ""]
        for mode, stats in run["mechanisms"].items():
            requests, geometry = stats["prefix_requests"], stats["motion_axis_geometry"]
            lines += [f"- {mode} prefix-request diagnostics: {requests['observed_decisions']} recorded decisions; "
                      f"checked={requests['checked_decisions']}; request reasons={requests['request_reason_counts']}; "
                      f"skip reasons={requests['skip_reason_counts']}.",
                      f"- {mode} motion-axis geometry: {geometry['observed_decisions']} recorded decisions; "
                      f"replacement decisions={geometry['replacement_decisions']}, ID-events={geometry['replacement_id_events']}, "
                      f"episodes with replacements={geometry['replacement_episodes']}. ID-events sum repeated observations across "
                      "decisions and episodes; they are not distinct vessels."]
        lines += [""]
        if run["cohort_components"]:
            lines += ["This cohort combines disjoint completed slices after checking identical classes, constructor options, checkpoint/config, "
                      "effective constants, runtime thread settings and every common archived source hash. Optional snapshot-helper inclusion "
                      "and snapshot logging may differ. Any explicitly allowed inactive-version addition is audited below. Component identity is retained in each CSV row:"]
            lines += [f"- {item['tag']}: {item['completed']} records / {item['distinct_cases']} scenarios; snapshots={item['snapshots']}; "
                      f"optional source difference={item['source_inventory_difference_from_first']}." for item in run["cohort_components"]]
            for item in run["cohort_components"]:
                selected = item.get("mode_selection")
                if selected:
                    lines.append(f"  Selected {selected['selected_mode']}: {selected['selected_completed_runs']} included records from "
                                 f"{selected['source_completed_runs']} fully validated source-run records; omitted modes={selected['excluded_modes']}. "
                                 "Other modes are not counted as this candidate's episodes, and original attempt IDs remain unchanged.")
                for audit in item["inactive_version_source_audit"]:
                    lines.append(f"  Explicit inactive-version allowance: {audit['path']} ({audit['sha256']}); selected filter versions "
                                 f"{audit['selected_filter_versions']}, added version {audit['added_version']}, no literal module/class reference "
                                 f"in {audit['shared_python_sources_scanned']} common archived sources. This is not a general proof about arbitrary dynamic imports.")
            lines += [""]
        for group in run["groups"]:
            if group["dimension"] != "overall":
                continue
            for pair in group["comparisons"]:
                if pair["reference"] in ("off", "v8", "v10"):
                    lines.append(f"- {group['mode']} vs {pair['reference']} ({pair['evidence']}): gains "
                                 + (", ".join(pair["gained_cases"]) or "none") + "; losses "
                                 + (", ".join(pair["lost_cases"]) or "none") + ".")
        lines += ["", "Archived source differences relative to the prior fresh pilot:"]
        if run["pilot_source_differences"]:
            lines += ["- " + name for name in run["pilot_source_differences"]]
        else:
            lines += ["- None among compared archive inventories, or no pilot-case overlap."]
        for reference, info in run["fresh_reference_provenance"].items():
            lines += ["", f"Additional fresh reference {reference}: {info['overlapping_cases']} matching scenario identities; "
                      "its archived class/options and source differences are preserved in summary.json."]
        lines += ["", f"Checkpoint SHA-256: `{run['checkpoint_sha256']}`. The frozen manifest records filter classes/options; "
                  "[provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. "
                  "Result JSON and traces are authoritative; episodes.csv is not used as evidence.", ""]
    return "\n".join(lines)


def attach_source_audits(summary, reader):
    if BASELINE_AUDIT.exists():
        summary["baseline_source_audit"] = {"path": str(BASELINE_AUDIT), "sha256": sha(reader.read(BASELINE_AUDIT))}
    env_differs = any("src/env.py" in run["pilot_source_differences"] or any(
        "src/env.py" in item["source_differences"] for item in run["fresh_reference_provenance"].values())
        for run in summary["iterations"])
    if env_differs and NATIVE_DISPATCH_AUDIT.exists():
        summary["native_dispatch_source_audit"] = {
            "path": str(NATIVE_DISPATCH_AUDIT), "sha256": sha(reader.read(NATIVE_DISPATCH_AUDIT))}
    source_differences = []
    for run in summary["iterations"]:
        source_differences.append(run["pilot_source_differences"])
        for item in run["fresh_reference_provenance"].values():
            source_differences.append(item["source_differences"])
    changed_sources = set().union(*source_differences)
    for key, path, relevant in (
        ("v15_integration_source_audit", V15_INTEGRATION_AUDIT, {"src/env.py", "src/safety_v6.py", "src/safety_v11.py"}),
        ("v15_v6_diagnostic_ast_audit", V15_DIAGNOSTIC_AUDIT, {"src/safety_v6.py"}),
        ("v16_native_dispatch_source_audit", V16_NATIVE_DISPATCH_AUDIT, {"src/env.py"}),
    ):
        if changed_sources & relevant and path.exists():
            raw = reader.read(path)
            audit = json.loads(raw)
            files = ({item["path"]: item for item in audit["changes"]} if "changes" in audit
                     else audit.get("files", {audit.get("source"): audit}))
            applicable = {name for differences in source_differences for name, hashes in differences.items()
                          if name in relevant and name in files
                          and files[name].get("after_sha256") in (hashes.get("candidate"), hashes.get("reference"))
                          and files[name].get("after_sha256") is not None}
            if applicable:
                summary[key] = {"path": str(path), "sha256": sha(raw),
                                "applicable_changed_sources": sorted(applicable)}


def report(directories, output=None, pilot_directory=PILOT, interim=False, audit_path=AUDIT, cohort=None, reference_directories=(), cohort_mode=None, allow_inactive_version_additions=False):
    require(interim or output is not None, "Final report needs --output")
    require(not interim or output is None, "--interim is read-only; do not supply --output")
    require(not interim or not cohort, "An interim snapshot cannot declare a complete cohort")
    require(not cohort_mode or cohort, "--cohort-mode requires --cohort")
    require(not allow_inactive_version_additions or cohort, "Inactive-version allowance requires --cohort")
    reader = Reader()
    runs = [load_run(path, reader, interim=interim) for path in directories]
    require(len({r["tag"] for r in runs}) == len(runs), "Duplicate iteration tag")
    if interim:
        value = {"interim": True, "report_generation_new_episodes": 0, "iterations": [
            {"tag": r["tag"], "complete": r["complete"], "planned": r["manifest"]["planned_runs"],
             "attempted": r["attempted"], "completed": len(r["results"]),
             "outcomes": {m: dict(Counter(x["outcome"] for (_, mode), x in r["results"].items() if mode == m)) for m in r["manifest"]["modes"]}}
            for r in runs]}
        print(json.dumps(value))
        return value
    pilot = load_run(pilot_directory, reader) if pilot_directory is not None else None
    reference_runs = [load_reference(path, reader) for path in reference_directories]
    require(len({(r["directory"], tuple(r["manifest"]["modes"])) for r in reference_runs}) == len(reference_runs), "Duplicate fresh reference view")
    require(len({key for r in reference_runs for key in r["results"]}) == sum(len(r["results"]) for r in reference_runs),
            "Duplicate fresh reference case/mode; different configurations must not be merged")
    if pilot is not None:
        for run in runs + reference_runs:
            for key in ("checkpoint_sha256", "config_sha256", "low_speed_start_frac"):
                require(run["manifest"][key] == pilot["manifest"][key], f"Historical/pilot baseline {key} mismatch")
            current = run["manifest"].get("constants", run["manifest"].get("effective_constants"))
            reference = pilot["manifest"].get("constants", pilot["manifest"].get("effective_constants"))
            require(current is not None and current == reference, "Historical/pilot baseline effective constants mismatch")
    canonical = canonical_inventory(reader)
    for run in runs + reference_runs + ([pilot] if pilot is not None else []):
        validate_canonical_cases(run, canonical)
    if cohort:
        if cohort_mode:
            runs = [select_run_mode(run, cohort_mode) for run in runs]
        runs = [combine_cohort(runs, cohort, allow_inactive_version_additions)]
    history_rows = list(csv.DictReader(io.StringIO(reader.read(audit_path).decode("utf-8"))))
    history = {r["case"]: r for r in history_rows}
    require(len(history) == len(history_rows), "Duplicate historical audit identity")
    pairs, summaries = [], []
    for run in runs:
        p, s = summarize(run, pilot, history, reference_runs)
        pairs.extend(p)
        summaries.append(s)
    identities = Counter(row["case"] for row in pairs)
    summary = {"schema": 1, "iterations": summaries, "report_generation_new_episodes": 0,
               "completed_iteration_records": len(pairs), "distinct_cases": len(identities),
               "repeated_case_record_counts": {case: count for case, count in identities.items() if count > 1},
               "scope": "Finite, enriched development cohorts; fresh/historical references remain distinct; no formal safety or population claim."}
    primary = primary_benchmark_scope(runs, canonical)
    if primary:
        summary["primary_benchmark"] = primary
        summary["scope"] = "Primary test set v3 simulation benchmark; full-set/subset coverage is explicit per iteration; no formal safety or population claim."
    attach_source_audits(summary, reader)
    provenance = {"input_sha256": reader.hashes, "report_script_sha256": LOADED_REPORT_SCRIPT_SHA256,
                  "shared_reader_sha256": sha(read_shared_bytes(Path(__file__).with_name("suite_status.py"))),
                  "report_generation_new_episodes": 0, "authoritative_records": "attempt/result JSON and complete traces; never episodes.csv"}
    output = Path(output)
    names = ("report.md", "paired.csv", "summary.json", "provenance.json")
    require(not any((output / name).exists() for name in names), "Report output exists; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("provenance.json", provenance)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    with (output / "paired.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown(summary))
    print(json.dumps({"output": str(output), "completed": {r["tag"]: r["completed"] for r in summaries}}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--interim", action="store_true")
    parser.add_argument("--cohort", help="Combine only disjoint slices of an identical frozen controller configuration")
    parser.add_argument("--cohort-mode", help="Select one mode from every fully validated cohort component")
    parser.add_argument("--allow-inactive-version-additions", action="store_true", help="Explicitly audit higher unused safety_vN.py archive additions; shared hashes must still match")
    parser.add_argument("--reference-run", action="append", default=[], help="Complete fresh reference PATH, MODE=PATH or LABEL::MODE=PATH")
    args = parser.parse_args()
    report(args.directories, args.output, args.pilot, args.interim, cohort=args.cohort,
           reference_directories=args.reference_run, cohort_mode=args.cohort_mode,
           allow_inactive_version_additions=args.allow_inactive_version_additions)
