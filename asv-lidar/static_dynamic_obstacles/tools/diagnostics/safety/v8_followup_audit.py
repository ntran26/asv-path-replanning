"""Saved-only V8 follow-up audit. Never constructs an environment or runs episodes.

python -B tools/diagnostics/safety/v8_followup_audit.py
Use --output NEW_DIRECTORY to preserve earlier audits when repeating.
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

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "results/safety_dev/trigger_counterfactual"
OUTPUT = ROOT / "results/safety_dev/v8_followup_offline/audit"
OUTCOMES = {"goal", "collision:target", "collision:obstacle", "collision:boundary", "timeout"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def unique(rows, fields):
    index = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        if key in index:
            raise ValueError(f"Duplicate identity {fields}: {key}")
        index[key] = row
    return index


def validate_steps(episodes, steps, versions):
    indexed = unique(steps, ("case", "step"))
    grouped = defaultdict(list)
    for row in steps:
        if row["case"] not in episodes:
            raise ValueError("Step belongs to an unknown episode")
        grouped[row["case"]].append(row)
    for case, episode in episodes.items():
        rows = grouped[case]
        if sorted(int(r["step"]) for r in rows) != list(range(int(episode["steps"]))):
            raise ValueError(f"Missing/non-contiguous steps: {case}")
        for version in versions:
            if any(r[version + "_fire"] not in ("True", "False") for r in rows):
                raise ValueError("Invalid fire boolean")
            fired = sorted(int(r["step"]) for r in rows if r[version + "_fire"] == "True")
            recorded = episode[version + "_first_fire"]
            if len(fired) != int(episode[version + "_fires"]) or recorded != (str(fired[0]) if fired else ""):
                raise ValueError(f"Episode fire summary disagrees with step records: {case}")
    return indexed


def validate_branches(episodes, branches, steps, versions, *, first_only=False):
    unique(branches, ("case", "version", "fire_index"))
    first = {}
    for row in branches:
        case, version = row["case"], row["version"]
        if case not in episodes or version not in versions:
            raise ValueError("Unknown branch case/version")
        if first_only and row["fire_index"] != "1":
            raise ValueError("Variant contains unexpected later branches")
        if row["main_outcome"] != episodes[case]["outcome"] or row["branch_outcome"] not in OUTCOMES:
            raise ValueError("Branch outcome inconsistency")
        key = (case, row["step"])
        if key not in steps or steps[key][version + "_fire"] != "True":
            raise ValueError("Branch has no corresponding fire")
        if int(row["branch_steps"]) < 1 or int(row["branch_interventions"]) < 1:
            raise ValueError("Invalid branch step/intervention count")
        if row["fire_index"] == "1":
            if row["step"] != episodes[case][version + "_first_fire"]:
                raise ValueError("First branch is not the first fire")
            first[(case, version)] = row
    for case, episode in episodes.items():
        for version in versions:
            if (int(episode[version + "_fires"]) > 0) != ((case, version) in first):
                raise ValueError(f"Missing first-fire branch or branch for never-fired case: {case}")
    return first


def numeric(value):
    return float(value) if value not in (None, "") else None


def category(policy, filtered):
    return ("rescued" if policy != "goal" and filtered == "goal" else
            "broken" if policy == "goal" and filtered != "goal" else
            "both_goal" if filtered == "goal" else "both_fail")


def reconstruct(base_rows, base_steps, base_branches, variant_rows, variant_steps, variant_branches):
    base = {k[0]: r for k, r in unique(base_rows, ("case",)).items()}
    variant = {k[0]: r for k, r in unique(variant_rows, ("case",)).items()}
    if any(r["outcome"] not in OUTCOMES for r in base_rows + variant_rows):
        raise ValueError("Unknown episode outcome")
    expected_subset = {case for case, r in base.items() if int(r["v7_fires"]) > 0}
    if set(variant) != expected_subset:
        raise ValueError("Nohold subset must equal every base V7-fired case exactly")
    for case, row in variant.items():
        if any(row[k] != base[case][k] for k in ("seed", "set", "outcome", "steps")):
            raise ValueError("Nohold policy history differs from base episode identity/outcome")
    base_st = validate_steps(base, base_steps, ("v4", "v7"))
    variant_st = validate_steps(variant, variant_steps, ("v7",))
    base_first = validate_branches(base, base_branches, base_st, ("v4", "v7"))
    variant_first = validate_branches(variant, variant_branches, variant_st, ("v7",), first_only=True)
    records = []
    for case, episode in sorted(base.items()):
        branch = variant_first.get((case, "v7"))
        v7branch = base_first.get((case, "v7"))
        v4branch = base_first.get((case, "v4"))
        v8 = branch["branch_outcome"] if branch else episode["outcome"]
        step = variant_st[(case, branch["step"])] if branch else {}
        reason = step.get("v7_why", "no fire")
        selected, repair = numeric(step.get("v7_checked_clearance")), numeric(step.get("v7_v7_repair_clearance"))
        row = {"case": case, "dataset": "ts2" if case.startswith("TS2:") else "dv3",
            "set": episode["set"], "seed": episode["seed"], "policy_outcome": episode["outcome"],
            "v4_outcome": v4branch["branch_outcome"] if v4branch else episode["outcome"],
            "v7_outcome": v7branch["branch_outcome"] if v7branch else episode["outcome"],
            "v8_outcome": v8, "v8_category": category(episode["outcome"], v8),
            "reconstruction": "first_fire_branch" if branch else "nohold_never_fired" if case in variant else "base_v7_never_fired",
            "first_fire_step": branch["step"] if branch else "", "first_reason": reason,
            "branch_interventions": branch["branch_interventions"] if branch else 0,
            "branch_steps": branch["branch_steps"] if branch else 0,
            "selected_clearance": step.get("v7_checked_clearance", ""),
            "policy_prefix_repair_clearance": step.get("v7_v7_repair_clearance", ""),
            "primitive_policy_margin_rounded": step.get("v7_policy_margin", ""),
            "primitive_best_nonbrake_margin_rounded": step.get("v7_best_margin", ""),
            "repair_clearance_not_worse": (repair >= selected if repair is not None and selected is not None else ""),
            "repair_minus_selected": repair - selected if repair is not None and selected is not None else "",
            "selected_below_zero": selected < 0 if selected is not None else "",
            "selected_below_trigger_015": selected < .15 if selected is not None else "",
            "policy_rudder": step.get("rudder_cmd", ""), "policy_throttle": step.get("throttle_cmd", ""),
            "q": step.get("q", ""), "speed_logged_from_env_u_body": step.get("speed", ""),
            "rm_urgent": step.get("rm_urgent", ""), "rm_persistent": step.get("rm_persistent", "")}
        records.append(row)
    return records


def summarize(records):
    groups = {}
    reasons = []
    for dataset in ("all", "ts2", "dv3"):
        rows = [r for r in records if dataset == "all" or r["dataset"] == dataset]
        groups[dataset] = {"n": len(rows), "outcomes": {mode: dict(Counter(r[mode + "_outcome"] for r in rows))
            for mode in ("policy", "v4", "v7", "v8")}, "v8_categories": dict(Counter(r["v8_category"] for r in rows)),
            "reconstruction": dict(Counter(r["reconstruction"] for r in rows)),
            "v7_to_v8_changed_outcomes": sum(r["v7_outcome"] != r["v8_outcome"] for r in rows)}
        for reason in sorted({r["first_reason"] for r in rows}):
            sub = [r for r in rows if r["first_reason"] == reason]
            counts = Counter(r["v8_category"] for r in sub)
            reasons.append(dict(dataset=dataset, reason=reason, n=len(sub), **{k: counts[k] for k in
                ("rescued", "broken", "both_fail", "both_goal")},
                selected_negative=sum(r["selected_below_zero"] is True for r in sub),
                repair_clearance_not_worse=sum(r["repair_clearance_not_worse"] is True for r in sub)))
    groups["ts2"]["first_fire_margin_groups"] = {cat: {"n": len(sub),
        "selected_below_trigger_015": sum(r["selected_below_trigger_015"] is True for r in sub),
        "selected_negative": sum(r["selected_below_zero"] is True for r in sub),
        "risk_monitor_urgent": sum(r["rm_urgent"] == "True" for r in sub)}
        for cat in ("rescued", "broken", "both_fail", "both_goal")
        for sub in [[r for r in records if r["dataset"] == "ts2" and r["v8_category"] == cat and r["first_reason"] != "no fire"]]}
    return {"groups": groups, "first_fire_reasons": reasons, "new_episode_runs": 0,
            "exact_prediction_recheck_possible_from_csv": False,
            "margin_comparison_is_not_full_dominance": True}


def write_csv(path, rows, fields=None):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(summary):
    groups = summary["groups"]
    lines = ["# V8 saved counterfactual follow-up audit", "",
        "All results below are reconstructed from existing saved runs. No simulator reset, policy inference, "
        "new episode or new rollout was performed. The nohold run used V7 with hold-back gain set to infinity; "
        "it is the evaluated behavior subsequently named V8.", "",
        "## Reconstruction", "",
        "The base contains 1,150 unique episodes (1,000 test-set v2 and 150 DV3). The nohold subset contains "
        "exactly the 549 cases where base V7 fired. Of these, 526 have a first-fire branch and 23 never fire "
        "under nohold. The other 601 cases never fired in base V7 and retain their recorded policy outcome. "
        "Missing branches are never treated as successful: a branch is required for every nonzero-fire episode. "
        "Contiguous step counts, fire counts, first-fire indices, unique identities, seeds and policy outcomes "
        "are checked against the CSVs. Reconstruction inherits the runner's first-fire shadow/replay equivalence "
        "claim; CSV integrity checks do not independently prove simulator-clone equivalence.", "",
        "| Set | N | Policy goals | V4 goals | V7 goals | V8 goals | V8 rescued | V8 broken |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("ts2", "dv3", "all"):
        g = groups[name]
        lines.append(f"| {name} | {g['n']} | " + " | ".join(str(g["outcomes"][m].get("goal", 0)) for m in ("policy", "v4", "v7", "v8"))
                     + f" | {g['v8_categories'].get('rescued', 0)} | {g['v8_categories'].get('broken', 0)} |")
    lines += ["", "V8 test-set outcomes: 912 goals, 31 target, 27 obstacle and 30 boundary contacts. "
        "Its 88 failures comprise 71 unrescued policy failures and 17 broken policy successes. The 71 have "
        "first reasons: no fire 17, turn 32, brake 9, searched escape 5, last certificate 2, continue 6. "
        "These are first interventions, not terminal failure causes.", "", "## Current-certification gap", "",
        "V8 disables hold-back but retains the inherited `last certificate` fallback. This follows a stored "
        "continuation despite failure of its current check. Therefore disabling hold-back alone does not establish "
        "that every intervention has a currently passing escape plan.", "",
        "| First-fire last certificate | Cases | Rescued | Broken | Both fail | Both goal |",
        "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("ts2", "dv3"):
        r = next(r for r in summary["first_fire_reasons"] if r["dataset"] == dataset and r["reason"] == "last certificate")
        lines.append(f"| {dataset} | {r['n']} | {r['rescued']} | {r['broken']} | {r['both_fail']} | {r['both_goal']} |")
    lines += ["", "All 27 test-set first-fire last-certificate clearances are negative. The three broken successes "
        "are CH-CR-RE-038 (step 16, selected −1.2419 m, repaired policy −1.0783 m), BAS-CR-RE-072 "
        "(step 6, −0.2721 m versus +0.0430 m), and BAS-HO-NC-059 (step 20, −0.6706 m versus −0.9155 m). "
        "The latter has a worse repaired-policy clearance, so a same-tail clearance comparison would not suppress all three.", "",
        "The counterexamples matter: DV3-HO-CV-03, DV3-HO-VS-01, DV3-CRP-CV-04 and DV3-CRP-CV-17 "
        "were rescued after first firing through last certificate. DV3-BO-CV-04 was broken. Two of those four "
        "rescues also have repaired-policy clearance greater than the selected continuation's clearance. "
        "A clearance-only dominance rule or blanket expiry removal therefore has no established net benefit.", "",
        "## A concrete experimental guard", "",
        "Recheck the actual selected backup and the same backup with only its first action replaced by the "
        "policy command, using identical snapshot, actuator history, horizon and constraints. A policy-preserving "
        "decision should depend on that actual comparison, including hard feasibility and first-violation time, "
        "rather than a rounded best-of-bank policy margin or a uniform new clearance threshold. A V9 guard built "
        "around this comparison remains experimental; these records cannot establish its closed-loop result, "
        "and it must not be promoted as improving 912/1,000.", "",
        "Any oracle total derived here is restricted to selecting between the already observed first-fire "
        "branch and the recorded SAC-only episode. It does not bound different trigger times, different "
        "future interventions or new rescue methods.", "",
        "Raising all intervention margins to 0.15 m is not supported here: 22 of 57 rescues and 9 of 17 breaks "
        "began below 0.15 m (the nine include three negative last certificates). All first-fire TS2 rows have "
        "risk-monitor urgent=false, including every rescue; urgency-only gating cannot be justified by a first-fire "
        "oracle calculation. This does not predict a later-triggered controller's outcome.", "",
        "## What can and cannot be checked offline", "",
        "`steps.csv` records pre-decision summaries along the policy-alone shadow trajectory. `branches.csv` "
        "records final branch outcomes and intervention counts, not branch decision traces. Only the first-fire "
        "branch corresponds to the complete filtered episode; later phase-1 branches are conditioned on a "
        "different policy-only history. They are never pooled as independent closed-loop episodes.", "",
        "`checked_clearance` and `v7_repair_clearance` compare selected versus repaired same-tail trajectories, "
        "but the CSV omits repair first-violation time, candidate first-violation time, full controls, actuator "
        "delay buffer, state snapshot, static points and target tracks. The original Python environment/filter "
        "copies were local runtime objects; they were not serialized to these outputs. Thus exact hard checks "
        "cannot be rerun from these CSVs without new state acquisition. `margin_opportunities.csv` only flags "
        "numerical clearance opportunities, not certified dominance or expected rescued episodes.", "",
        "`policy_margin` is rounded and takes the best passing primitive recovery, with −inf denoting no passing "
        "primitive; it is not the same trajectory as the prefix repair. `best_margin` excludes brake candidates. "
        "Neither should be directly substituted for the selected/repaired plan comparison. The recorded `speed` "
        "comes from `env.u_body`, so it is simulator state rather than a verified noisy onboard measurement. "
        "The saved gate fit uses critic Q and its trend, not speed; this logging caveat does not by itself "
        "invalidate those critic-gate results.", "",
        "## Sources and provenance", "",
        "The matching/data sources are [the V8 plan](<../../../../planning/SAFETY_LAYER_V8_PLAN.md>), "
        "[base episodes](../../trigger_counterfactual/episodes.csv), [base steps](../../trigger_counterfactual/steps.csv), "
        "[base branches](../../trigger_counterfactual/branches.csv), and the corresponding "
        "[nohold episodes](../../trigger_counterfactual/v7_nohold/episodes.csv), "
        "[steps](../../trigger_counterfactual/v7_nohold/steps.csv) and "
        "[branches](../../trigger_counterfactual/v7_nohold/branches.csv). "
        "The implementation inspected is [the shadow runner](../../../../tools/diagnostics/safety/trigger_counterfactual.py), "
        "[V6 selection](../../../../src/safety_v6.py), [V7 prefix repair](../../../../src/safety_v7.py), "
        "and [V8](../../../../src/safety_v8.py). Input bytes and current review-source SHA256 values are in "
        "provenance.json. Current source hashes are not historical run snapshots; the old configs record model "
        "path/settings but do not pin source/model/runtime hashes. This report makes no new method-performance "
        "or formal safety claim.", ""]
    return "\n".join(lines)


def analyze(output):
    if output.exists():
        raise FileExistsError("Use a new directory; previous audits are never overwritten")
    input_paths = {f"base_{name}": DATA / (name + ".csv") for name in ("episodes", "steps", "branches")}
    input_paths.update({f"variant_{name}": DATA / "v7_nohold" / (name + ".csv") for name in ("episodes", "steps", "branches")})
    raw = {name: path.read_bytes() for name, path in input_paths.items()}
    tables = {name: read_rows(content) for name, content in raw.items()}
    records = reconstruct(tables["base_episodes"], tables["base_steps"], tables["base_branches"],
                          tables["variant_episodes"], tables["variant_steps"], tables["variant_branches"])
    if Counter(r["dataset"] for r in records) != {"ts2": 1000, "dv3": 150}:
        raise ValueError("Expected all 1,000 TS2 and 150 DV3 saved episodes")
    summary = summarize(records)
    ts = [r for r in records if r["dataset"] == "ts2"]
    # Pin this report to the reviewed completed data, rather than silently
    # rendering dataset-specific statements for different future results.
    if (summary["groups"]["ts2"]["outcomes"]["v8"] != {"goal": 912, "collision:target": 31,
            "collision:obstacle": 27, "collision:boundary": 30}
            or summary["groups"]["dv3"]["outcomes"]["v8"].get("goal") != 127):
        raise ValueError("Saved outcomes differ from the reviewed V8 dataset")
    for name, content in raw.items():
        if input_paths[name].read_bytes() != content:
            raise ValueError("Input changed while reading")
    review_paths = [ROOT / name for name in ("planning/SAFETY_LAYER_V8_PLAN.md", "src/safety_v6.py",
        "src/safety_v7.py", "src/safety_v8.py", "tools/diagnostics/safety/trigger_counterfactual.py")]
    review_paths += [DATA / name for name in ("config.json", "config_phase1.json", "v7_nohold/config.json")]
    provenance = {"inputs_sha256": {str(input_paths[k].relative_to(ROOT)): sha(v) for k, v in raw.items()},
        "review_sources_current_sha256_not_run_snapshot": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in review_paths},
        "analysis_script_sha256": sha(Path(__file__).read_bytes()), "new_episode_runs": 0}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("all_1150.csv", records), ("remaining_71_ts2_failures.csv", [r for r in ts if r["v8_category"] == "both_fail"]),
        ("broken_17_ts2_successes.csv", [r for r in ts if r["v8_category"] == "broken"]),
        ("first_fire.csv", [r for r in records if r["first_reason"] != "no fire"]),
        ("margin_opportunities.csv", [r for r in records if r["repair_clearance_not_worse"] is True]),
        ("first_fire_reasons.csv", summary["first_fire_reasons"])):
        write_csv(output / name, rows)
    for name, value in (("summary.json", summary), ("provenance.json", provenance)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render(summary))
    print(json.dumps({"output": str(output), "episodes_reconstructed": len(records), "new_episode_runs": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    analyze(parser.parse_args().output)
