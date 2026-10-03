"""Report the completed fixed32 SAC/V8/V9 pilot; never runs an episode.

Requires all96 durable results, traces, tokens and a clean completion record.
python -B tools/diagnostics/safety/v9_pilot_report.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite_status import read_shared_bytes

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "results/safety_dev/v9_paired_pilot"
MODES = ("off", "v8", "v9")
OUTCOMES = ("goal", "collision:target", "collision:obstacle", "collision:boundary", "timeout")
RUDDER_ATOL = 1e-9
RPM_ATOL = 1e-6


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def exact_mcnemar(gained, lost):
    """Two-sided exact conditional binomial calculation on discordant pairs.

    Formula reference: Fagerland, Lydersen & Laake (2013),
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/ . Descriptive here: this is
    an outcome-enriched development cohort, not independent population sampling.
    """
    if any(type(x) is not int or x < 0 for x in (gained, lost)):
        raise ValueError("Discordant counts must be nonnegative integers")
    n = gained + lost
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n)


def validate_core(selection, manifest, completion, rows, tokens):
    cases = selection["cases"]
    expected = {(r["case"], m) for r in cases for m in MODES}
    if len({r["case"] for r in cases}) != len(cases):
        raise ValueError("Duplicate selection case")
    if manifest["cases"] != cases or tuple(manifest["modes"]) != MODES:
        raise ValueError("Manifest selection/modes differ")
    if manifest["planned_runs"] != len(expected) or completion["completed_runs"] != len(expected):
        raise ValueError("Campaign incomplete")
    if completion["source_drift"] or not completion["selection_unchanged"] or not completion["checkpoint_unchanged"]:
        raise ValueError("Completion reports changed frozen inputs")
    if len(rows) != len(expected) or len(tokens) != len(expected):
        raise ValueError("Missing or extra durable results/tokens")
    if len({r["attempt"] for r in rows}) != len(rows) or len({r["attempt"] for r in tokens}) != len(tokens):
        raise ValueError("Duplicate attempt")
    if {r["attempt"] for r in rows} != set(range(1, len(expected) + 1)):
        raise ValueError("Attempt sequence has holes or exceeds fixed budget")
    by_key = {(r["case"], r["mode"]): r for r in rows}
    if len(by_key) != len(rows) or set(by_key) != expected:
        raise ValueError("Duplicate or incomplete case/mode pair")
    canonical = {r["case"]: r for r in cases}
    token_by_attempt = {r["attempt"]: r for r in tokens}
    for row in rows:
        case = canonical[row["case"]]
        if any(row[k] != token_by_attempt[row["attempt"]][k] for k in ("attempt", "case", "mode", "seed")):
            raise ValueError("Token/result identity mismatch")
        if row["seed"] != case["seed"] or row["dataset"] != case["dataset"] or row["stratum"] != case["selection_stratum"]:
            raise ValueError("Result differs from selected scenario identity")
        if row["outcome"] not in OUTCOMES:
            raise ValueError("Unknown outcome")
        if bool(row["collided"]) != row["outcome"].startswith("collision:") or bool(row["collided_target"]) != (row["outcome"] == "collision:target"):
            raise ValueError("Inconsistent collision flags")
        if not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0 or row["steps"] < 1:
            raise ValueError("Invalid episode timing/steps")
    return by_key


def decode_trace(raw, result):
    if not raw.endswith(b"\n"):
        raise ValueError("Incomplete trace tail")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if [r["step"] for r in rows] != list(range(1, result["steps"] + 1)):
        raise ValueError("Trace step count/sequence differs from durable result")
    for r in rows:
        if not all(math.isfinite(float(r[k])) for k in ("rudder_command", "signed_rpm_command")):
            raise ValueError("Invalid executed command")
    checks = sum(bool(r["filter"].get("v9_current_plan_checked")) for r in rows)
    suppressed = sum(bool(r["filter"].get("v9_override_suppressed")) for r in rows)
    changes = sum(bool(r["changed"]) for r in rows)
    if (checks, suppressed, changes) != (result["checked_steps"], result["suppressed_steps"], result["safety_v2_steps"]):
        raise ValueError("Trace diagnostic counters differ from result")
    if dict(Counter(r["filter"].get("why", "off") for r in rows)) != result["why_counts"]:
        raise ValueError("Trace decision reasons differ from result")
    return rows


def vectors_close(a, b, atol=1e-9):
    return len(a) == len(b) and all(math.isclose(float(x), float(y), rel_tol=0, abs_tol=atol) for x, y in zip(a, b))


def first_divergence(v8, v9):
    base = {"first_command_divergence_step": "", "compared_command_steps": min(len(v8), len(v9)),
            "trace_lengths_equal": len(v8) == len(v9), "pre_state_equal": "", "policy_actions_equal": ""}
    detail_keys = ("why", "v9_parent_why", "policy_margin", "checked_clearance", "v7_repair_clearance",
                   "v9_proposed_clearance", "v9_policy_tail_clearance", "v9_proposed_first_violation",
                   "v9_policy_first_violation", "v9_proposed_currently_passing", "v9_policy_tail_passing")
    for mode in ("v8", "v9"):
        for name in ("rudder", "rpm") + detail_keys:
            base["divergence_" + mode + "_" + name] = ""
    for left, right in zip(v8, v9):
        equal = (math.isclose(left["rudder_command"], right["rudder_command"], rel_tol=0, abs_tol=RUDDER_ATOL)
                 and math.isclose(left["signed_rpm_command"], right["signed_rpm_command"], rel_tol=0, abs_tol=RPM_ATOL))
        if equal:
            continue
        base.update(first_command_divergence_step=left["step"],
                    pre_state_equal=vectors_close(left["pre_state"], right["pre_state"]),
                    policy_actions_equal=vectors_close(left["policy_action"], right["policy_action"], 1e-8))
        for mode, row in (("v8", left), ("v9", right)):
            base["divergence_" + mode + "_rudder"] = row["rudder_command"]
            base["divergence_" + mode + "_rpm"] = row["signed_rpm_command"]
            for name in detail_keys:
                base["divergence_" + mode + "_" + name] = row["filter"].get(name, "")
        break
    return base


def pair_stats(rows, candidate, reference):
    gain = [r["case"] for r in rows if r[candidate + "_outcome"] == "goal" and r[reference + "_outcome"] != "goal"]
    loss = [r["case"] for r in rows if r[candidate + "_outcome"] != "goal" and r[reference + "_outcome"] == "goal"]
    transitions = Counter(r[reference + "_outcome"] + " -> " + r[candidate + "_outcome"] for r in rows)
    return {"matched_cases": len(rows), "gained_goals": len(gain), "lost_goals": len(loss), "net_goals": len(gain)-len(loss),
        "gained_cases": gain, "lost_cases": loss, "exact_mcnemar_two_sided_descriptive_p": exact_mcnemar(len(gain), len(loss)),
        "both_goal": sum(r[candidate + "_outcome"] == r[reference + "_outcome"] == "goal" for r in rows),
        "both_fail": sum(r[candidate + "_outcome"] != "goal" and r[reference + "_outcome"] != "goal" for r in rows),
        "collision_to_goal": sum(r[reference + "_outcome"].startswith("collision:") and r[candidate + "_outcome"] == "goal" for r in rows),
        "collision_to_timeout": sum(r[reference + "_outcome"].startswith("collision:") and r[candidate + "_outcome"] == "timeout" for r in rows),
        "outcome_transitions": dict(transitions)}


def summarize(selection, results, traces, completion):
    paired = []
    reproduction = []
    for case in selection["cases"]:
        row = {"case": case["case"], "dataset": case["dataset"], "stratum": case["selection_stratum"],
               "seed": case["seed"], "scenario_sha256": case["scenario_sha256"]}
        for mode in MODES:
            result = results[(case["case"], mode)]
            for field in ("outcome", "steps", "elapsed_s", "safety_v2_steps", "checked_steps", "suppressed_steps"):
                row[mode + "_" + field] = result[field]
            if mode in ("off", "v8"):
                historical = case["historical"]["policy_outcome" if mode == "off" else "v8_outcome"]
                row[mode + "_historical_outcome"] = historical
                row[mode + "_reproduced_history"] = historical == result["outcome"]
                if historical != result["outcome"]:
                    reproduction.append({"case": case["case"], "mode": mode, "historical": historical, "fresh": result["outcome"]})
        row.update(first_divergence(traces[(case["case"], "v8")], traces[(case["case"], "v9")]))
        paired.append(row)
    groups = {}
    for dimension in ("overall", "dataset", "stratum"):
        slices = {"all": paired} if dimension == "overall" else {
            name: [r for r in paired if r[dimension] == name] for name in sorted({r[dimension] for r in paired})}
        groups[dimension] = {}
        for name, rows in slices.items():
            groups[dimension][name] = {"cases": len(rows), "mode_outcomes": {
                mode: {outcome: sum(r[mode + "_outcome"] == outcome for r in rows) for outcome in OUTCOMES} for mode in MODES},
                "pairs": {a + "_vs_" + b: pair_stats(rows, a, b) for a, b in (("v9", "v8"), ("v9", "off"), ("v8", "off"))}}
    mechanisms = {}
    for mode in MODES:
        r = [row for (_, m), row in results.items() if m == mode]
        reasons = Counter()
        for item in r:
            reasons.update(item["why_counts"])
        mechanisms[mode] = {"changed_action_steps": sum(x["safety_v2_steps"] for x in r),
            "checked_steps": sum(x["checked_steps"] for x in r), "checked_episodes": sum(x["checked_steps"] > 0 for x in r),
            "suppressed_steps": sum(x["suppressed_steps"] for x in r), "suppressed_episodes": sum(x["suppressed_steps"] > 0 for x in r),
            "episode_seconds": sum(x["elapsed_s"] for x in r), "decision_reasons": dict(reasons)}
    summary = {"schema": 1, "completed_fresh_runs": len(results), "fixed_scenarios": len(paired),
        "groups": groups, "mechanisms": mechanisms, "historical_mismatches": reproduction,
        "historical_reproduction": {mode: {"compared": len(paired), "matching": sum(r[mode + "_reproduced_history"] for r in paired)} for mode in ("off", "v8")},
        "first_command_divergence_cases": sum(r["first_command_divergence_step"] != "" for r in paired),
        "command_comparison_tolerances": {"rudder_absolute": RUDDER_ATOL, "rpm_absolute": RPM_ATOL},
        "campaign_elapsed_seconds": completion["elapsed_s"], "reported_new_episode_runs": 0,
        "scope": "All32 preselected diagnostic cases; not a population or all1000 estimate"}
    return paired, summary


def markdown(summary):
    whole = summary["groups"]["overall"]["all"]
    comparison = whole["pairs"]["v9_vs_v8"]
    goals = {m: whole["mode_outcomes"][m]["goal"] for m in MODES}
    direction = "more" if comparison["net_goals"] > 0 else "fewer" if comparison["net_goals"] < 0 else "the same number of"
    relation = "as" if comparison["net_goals"] == 0 else "than"
    lines = ["# Fresh SAC/V8/V9 paired development pilot", "",
        f"**V9 reached {goals['v9']}/32 goals, V8 {goals['v8']}/32 and SAC alone {goals['off']}/32.** "
        f"V9 achieved {direction} goals {relation} V8 in this fixed cohort: {comparison['gained_goals']} gained and "
        f"{comparison['lost_goals']} lost. This is a diagnostic pilot, not an estimate for all 1,000 test cases.", "",
        "The user's later request explicitly authorized fresh episodes to determine whether V9 is better or worse, "
        "superseding the earlier no-new-runs restriction for this pilot. Selection was fixed before evaluation: "
        "32 scenarios × SAC alone/V8/V9 = **96 new episode runs**, with no historical result reuse or automatic retries. "
        "This report performs no additional runs. Five cases are DV3 and 27 are test-set v2; all are development evidence.", "",
        "## Outcomes", "", "| Mode | N | Goal | Target | Obstacle | Boundary | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        c = whole["mode_outcomes"][mode]
        lines.append(f"| {mode} | 32 | " + " | ".join(str(c[k]) for k in OUTCOMES) + " |")
    lines += ["", "All rates use 32 cases per controller; errors or missing records are rejected before reporting.", "",
        "| Candidate vs reference | Matched | Gained goals | Lost goals | Net | Exact paired p (descriptive) |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, p in whole["pairs"].items():
        lines.append(f"| {name} | {p['matched_cases']} | {p['gained_goals']} | {p['lost_goals']} | {p['net_goals']:+d} | {p['exact_mcnemar_two_sided_descriptive_p']:.6g} |")
    lines += ["", "Exact two-sided McNemar values use twice the smaller Binomial(discordant pairs, 0.5) tail, "
        "capped at one ([Fagerland, Lydersen & Laake, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/)). "
        "They are descriptive here: outcome-enriched selection and shared scenario families do not justify a "
        "population significance claim. Collision→timeout is reported separately from collision→goal in summary.json.", "",
        "V9 gained over V8: " + (", ".join(comparison["gained_cases"]) or "none") + ".",
        "V9 lost versus V8: " + (", ".join(comparison["lost_cases"]) or "none") + ".", "",
        "## Dataset and selection strata", "", "| Slice | N | SAC goals | V8 goals | V9 goals | V9/V8 gains | V9/V8 losses |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for dimension in ("dataset", "stratum"):
        for name, group in summary["groups"][dimension].items():
            p = group["pairs"]["v9_vs_v8"]
            lines.append(f"| {dimension}: {name} | {group['cases']} | " + " | ".join(str(group["mode_outcomes"][m]["goal"]) for m in MODES)
                         + f" | {p['gained_goals']} | {p['lost_goals']} |")
    lines += ["", "Per-slice collision types and all three paired comparisons are retained in summary.json. "
        "The selected cohort deliberately contains known rescues and breaks; its aggregate percentage cannot "
        "be compared directly with V8's saved 912/1,000 rate.", "", "## Baseline reproduction", ""]
    for mode in ("off", "v8"):
        r = summary["historical_reproduction"][mode]
        lines.append(f"- Fresh {mode}: {r['matching']}/{r['compared']} outcomes match their recorded historical outcome.")
    for row in summary["historical_mismatches"]:
        lines.append(f"- {row['case']} ({row['mode']}): historical {row['historical']} → fresh {row['fresh']}.")
    lines += ["", "All paired claims use the fresh outcomes regardless of historical agreement.", "",
        "## Intervention mechanisms and first command divergence", "",
        "| Mode | Changed-action steps | Guard checks (steps/episodes) | Suppressed proposals (steps/episodes) | Episode seconds |",
        "|---|---:|---:|---:|---:|"]
    for mode, m in summary["mechanisms"].items():
        lines.append(f"| {mode} | {m['changed_action_steps']} | {m['checked_steps']}/{m['checked_episodes']} | "
                     f"{m['suppressed_steps']}/{m['suppressed_episodes']} | {m['episode_seconds']:.2f} |")
    lines += ["", f"V8 and V9 first issue different commands in {summary['first_command_divergence_cases']}/32 cases. "
        "paired.csv records the first differing executed rudder/RPM command, both reasons, V9's parent reason, "
        "and its actual selected-versus-policy-tail margins and first-violation times. Steps are one-based. "
        f"Command tolerances are {RUDDER_ATOL:g} normalized rudder and {RPM_ATOL:g} RPM. Pre-state comparison "
        "covers only recorded x/y/heading/surge; it is not a check of the entire hidden simulator state. "
        "Equal command prefixes with different trace lengths are marked explicitly. More suppressed steps "
        "or fewer changed actions alone is not evidence of improved safety. The first divergence locates "
        "where behavior separates; later interventions also differ, so it neither proves that one decision "
        "caused the final outcome nor isolates the effects of V9's two guards.", "",
        "Decision-reason totals retain the runner's raw labels. In enabled V8/V9 traces, `off` can mean "
        "an idle filter with no `why` field; it does not establish that safety was disabled.", "",
        f"Runner wall time: {summary['campaign_elapsed_seconds']:.2f} s. Per-mode seconds sum timed episodes; "
        "they exclude some setup/archive work and reflect one serial process with rotated controller order. "
        "Outcomes change episode duration: an early collision can reduce runtime, so these totals do not "
        "isolate filter computation speed.", "",
        "## Provenance", "",
        "[Selection](selection.json), [frozen manifest](manifest.json), [evaluated source archive](evaluated_sources.zip), "
        "[completion](completion.json), [paired cases](paired.csv) and [report provenance](report_provenance.json). "
        "The report validates all96 tokens/results/traces, selection identity and seeds, completion flags, "
        "and every archived source hash. Source settings belong to the archived pilot; no result is credited "
        "to later controller edits. Original artifacts are read with Windows shared handles and are unchanged.", ""]
    return "\n".join(lines)


def report(directory):
    names = ("report.md", "paired.csv", "summary.json", "report_provenance.json")
    if any((directory / name).exists() for name in names):
        raise FileExistsError("Report outputs already exist; preserve the completed report")
    observed = {}
    def read(path):
        raw = read_shared_bytes(path)
        observed[path.relative_to(directory).as_posix()] = sha(raw)
        return raw
    selection_raw = read(directory / "selection.json")
    selection = json.loads(selection_raw)
    manifest = json.loads(read(directory / "manifest.json"))
    completion = json.loads(read(directory / "completion.json"))
    if manifest["selection_sha256"] != sha(selection_raw) or len(selection["cases"]) != 32 or manifest["planned_runs"] != 96:
        raise ValueError("This report requires the fixed32/96-run frozen selection")
    token_paths = sorted((directory / "attempts").glob("[0-9][0-9][0-9].json"))
    result_paths = sorted((directory / "attempts").glob("[0-9][0-9][0-9]_result.json"))
    tokens = [json.loads(read(path)) for path in token_paths]
    rows = [json.loads(read(path)) for path in result_paths]
    for path, row in zip(token_paths, tokens):
        if int(path.stem) != row["attempt"]:
            raise ValueError("Token filename does not match attempt")
    for path, row in zip(result_paths, rows):
        if int(path.name.split("_")[0]) != row["attempt"]:
            raise ValueError("Result filename does not match attempt")
    results = validate_core(selection, manifest, completion, rows, tokens)
    traces = {}
    for key, row in results.items():
        path = directory / "traces" / f"{row['attempt']:03d}_{row['mode']}.jsonl"
        traces[key] = decode_trace(read(path), row)
    archive_raw = read(directory / "evaluated_sources.zip")
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        if set(archive.namelist()) != set(manifest["source_sha256"]) or len(archive.namelist()) != len(manifest["source_sha256"]):
            raise ValueError("Archive source inventory differs from manifest")
        for name, expected in manifest["source_sha256"].items():
            if sha(archive.read(name)) != expected:
                raise ValueError(f"Archived source hash mismatch: {name}")
    paired, summary = summarize(selection, results, traces, completion)
    provenance = {"input_sha256": observed, "report_script_sha256": sha(read_shared_bytes(Path(__file__))),
        "checkpoint_sha256_recorded": manifest["checkpoint_sha256"], "config_sha256_recorded": manifest["config_sha256"],
        "all_archived_sources_verified": len(manifest["source_sha256"]), "fresh_episode_count": len(results),
        "report_generation_new_episodes": 0, "new_authorization_recorded": manifest["new_authorization"]}
    for name, value in (("summary.json", summary), ("report_provenance.json", provenance)):
        with (directory / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    with (directory / "paired.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    with (directory / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown(summary))
    print(json.dumps({"directory": str(directory), "outcomes": summary["groups"]["overall"]["all"]["mode_outcomes"],
                      "v9_vs_v8": summary["groups"]["overall"]["all"]["pairs"]["v9_vs_v8"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    report(parser.parse_args().directory)
