"""Saved-only last-decision supplement for V8's never-fired TS2 collisions.

python -B tools/diagnostics/safety/v8_no_fire_diagnosis.py
Preserves the original audit and refuses to overwrite an existing supplement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import v8_followup_audit as audit


def diagnose(records, base_episodes, variant_episodes, base_steps, variant_steps):
    base = {r["case"]: r for r in base_episodes}
    variant = {r["case"]: r for r in variant_episodes}
    base_index = audit.unique(base_steps, ("case", "step"))
    variant_index = audit.unique(variant_steps, ("case", "step"))
    result = []
    for record in records:
        if record["dataset"] != "ts2" or record["v8_category"] != "both_fail" or record["first_reason"] != "no fire":
            continue
        case = record["case"]
        episode = variant.get(case, base[case])
        if int(episode["v7_fires"]) != 0 or episode["outcome"] != record["v8_outcome"]:
            raise ValueError("Selected episode does not share the unmodified policy trajectory")
        step = int(episode["steps"]) - 1
        last = (variant_index if case in variant else base_index)[(case, str(step))]
        if last["v7_fire"] != "False":
            raise ValueError("Last decision is not a policy pass")
        result.append({"case": case, "seed": int(episode["seed"]), "outcome": episode["outcome"],
            "last_decision_step_zero_based": step, "episode_steps": int(episode["steps"]),
            "source": "v7_nohold/steps.csv" if case in variant else "steps.csv",
            "last_reason": last["v7_why"], "policy_margin": last["v7_policy_margin"],
            "checked_clearance_m": last["v7_checked_clearance"],
            "risk_monitor_urgent": last["rm_urgent"] == "True",
            "risk_monitor_persistent": last["rm_persistent"] == "True"})
    return sorted(result, key=lambda r: r["case"])


def supplement(directory):
    targets = [directory / ("no_fire_last_decisions." + suffix) for suffix in ("json", "md")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Supplement already exists; earlier evidence is preserved")
    original_provenance_raw = (directory / "provenance.json").read_bytes()
    original_provenance = json.loads(original_provenance_raw)
    names = ("episodes", "steps", "branches")
    paths = {f"base_{name}": audit.DATA / (name + ".csv") for name in names}
    paths.update({f"variant_{name}": audit.DATA / "v7_nohold" / (name + ".csv") for name in names})
    raw = {name: path.read_bytes() for name, path in paths.items()}
    for name, content in raw.items():
        relative = paths[name].relative_to(audit.ROOT).as_posix()
        # Original provenance may use Windows path separators.
        expected = {key.replace("\\", "/"): value for key, value in original_provenance["inputs_sha256"].items()}
        if audit.sha(content) != expected.get(relative):
            raise ValueError("Saved input differs from original audit provenance")
    tables = {name: audit.read_rows(content) for name, content in raw.items()}
    records = audit.reconstruct(tables["base_episodes"], tables["base_steps"], tables["base_branches"],
        tables["variant_episodes"], tables["variant_steps"], tables["variant_branches"])
    rows = diagnose(records, tables["base_episodes"], tables["variant_episodes"], tables["base_steps"], tables["variant_steps"])
    counts = dict(Counter(r["last_reason"] for r in rows))
    if len(rows) != 17 or counts != {"no escape": 14, "nominal": 3}:
        raise ValueError("Saved diagnosis differs from the reviewed 17-case subset")
    positive = [r for r in rows if r["last_reason"] == "nominal"]
    if any(r["outcome"] != "collision:target" or float(r["checked_clearance_m"]) <= 0 or r["risk_monitor_urgent"] for r in positive):
        raise ValueError("Unexpected positive-margin terminal pattern")
    for name, content in raw.items():
        if paths[name].read_bytes() != content:
            raise ValueError("Input changed during diagnosis")
    result = {"schema": 1, "new_episode_runs": 0, "episodes": len(rows), "last_reason_counts": counts,
        "rows": rows, "original_audit_provenance_sha256": audit.sha(original_provenance_raw),
        "original_audit_tool_sha256_recorded": original_provenance["analysis_script_sha256"],
        "supplement_script_sha256": audit.sha(Path(__file__).read_bytes()),
        "reconstruction_dependency_sha256": audit.sha(Path(audit.__file__).read_bytes()),
        "input_sha256": {paths[k].relative_to(audit.ROOT).as_posix(): audit.sha(v) for k, v in raw.items()},
        "interpretation": "Last shadow decision is also V8's last decision only because these episodes never fire. "
            "No saved target tracks or full snapshots identify which perception/model error caused a positive-margin collision."}
    with targets[0].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    lines = ["# V8 never-fired collisions: last-decision supplement", "",
        "Of the 71 unrescued test-set-v2 failures, 17 never trigger V8. Their recorded policy and V8 trajectories "
        "therefore coincide. At the final decision before collision, **14 say `no escape` and 3 say `nominal`**. "
        "The 14 no-escape cases have policy margin `-inf`; this indicates failure to find a passing primitive, "
        "not a proof that physical escape is impossible.", "",
        "| Never-fired target collision | Final step (zero-based) | Checked clearance | Urgent |",
        "|---|---:|---:|---|"]
    for row in positive:
        lines.append(f"| {row['case']} | {row['last_decision_step_zero_based']} | {float(row['checked_clearance_m']):.8f} m | false |")
    lines += ["", "These three collisions occur after the filter passes the policy with positive predicted clearance. "
        "They motivate separate investigation of target perception and target/dynamics prediction, rather than "
        "treating every failure as a trigger threshold problem. The CSV contains neither target tracks nor full "
        "prediction snapshots, so it cannot distinguish those mechanisms or support an exact offline recheck.", "",
        "This inference about the final decision applies only to the 17 episodes that never fire. Shadow "
        "rows from an intervened episode are on SAC's trajectory and cannot describe the filtered branch's last decision.", "",
        "All 17 rows, seeds, final outcomes, reasons, margins and monitor flags are in "
        "[no_fire_last_decisions.json](no_fire_last_decisions.json). Inputs are the existing "
        "[base steps](../../trigger_counterfactual/steps.csv) and "
        "[nohold steps](../../trigger_counterfactual/v7_nohold/steps.csv), joined to their episode/branch tables. "
        "The script reuses the original audit's strict reconstruction and verifies all six input hashes against "
        "[its provenance](provenance.json). The supplement records its own script and dependency hashes; "
        "the original audit and raw data are unchanged. No episodes or prediction rollouts were run.", ""]
    with targets[1].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"output": str(directory), "last_reason_counts": counts, "new_episode_runs": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-directory", type=Path, default=audit.OUTPUT)
    supplement(parser.parse_args().audit_directory)
