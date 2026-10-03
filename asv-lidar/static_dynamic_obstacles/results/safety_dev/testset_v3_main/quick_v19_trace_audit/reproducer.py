"""Completed primary-set quick audit only; no controller imports or episodes.

This saved analysis reports outcomes and command/activation evidence. It does
not choose controller changes or tune methods using primary-set results.
Existing report outputs are refused; this file is an immutable code artifact.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "results/safety_dev/testset_v3_main/runs"
PROVENANCE = {}


def read(path):
    data = path.read_bytes()
    PROVENANCE[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def close(a, b, tolerance):
    return len(a) == len(b) and all(abs(float(x)-float(y)) <= tolerance for x, y in zip(a, b))


def command(row):
    return [row["rudder_command"], row["signed_rpm_command"]]


def compare(reference, candidate):
    first = next(((a, b) for a, b in zip(reference, candidate)
                  if not close(command(a), command(b), 1e-7)), None)
    exact = next((a["step"] for a, b in zip(reference, candidate) if command(a) != command(b)), None)
    return dict(compared_decisions=min(len(reference), len(candidate)),
        same_trace_length=len(reference) == len(candidate), first_exact_command_difference=exact,
        first_command_divergence=None if first is None else dict(
            step=first[0]["step"], reference_command=command(first[0]), candidate_command=command(first[1]),
            same_pre_state=close(first[0]["pre_state"], first[1]["pre_state"], 1e-9),
            same_policy_action=close(first[0]["policy_action"], first[1]["policy_action"], 1e-7),
            reference_reason=(first[0].get("filter") or {}).get("why"),
            candidate_reason=(first[1].get("filter") or {}).get("why"),
            reference_checked_clearance=(first[0].get("filter") or {}).get("checked_clearance"),
            candidate_checked_clearance=(first[1].get("filter") or {}).get("checked_clearance")))


def activations(rows):
    events = []
    for row in rows:
        stats = row["filter"].get("source_points")
        if stats is None:
            raise ValueError("Expected V19 method diagnostics missing")
        if stats.get("removed_point_count", 0):
            events.append(dict(step=row["step"], why=row["filter"].get("why"), changed=row["changed"],
                               command=command(row), evidence=stats))
    return dict(active_decisions=len(events), events=events,
        transferred_point_decisions=sum(e["evidence"].get("removed_point_count", 0) for e in events))


def main():
    if (OUT / "audit.json").exists() or (OUT / "REPORT.md").exists():
        raise FileExistsError("Refusing to overwrite completed primary analysis")
    fresh, manifests, completions = {}, {}, {}
    for tag in ("v19_quick_a", "v19_quick_b"):
        directory = RUNS / tag
        manifest = json.loads(read(directory / "manifest.json"))
        completion = json.loads(read(directory / "completion.json"))
        assert manifest["planned_runs"] == completion["completed_runs"] == 27
        assert not completion["source_drift"] and completion["checkpoint_unchanged"] and completion["selection_unchanged"]
        cases = {c["case"]: c for c in manifest["cases"]}
        results = [json.loads(read(p)) for p in sorted((directory / "attempts").glob("*_result.json"))]
        assert len(results) == 27
        table = list(csv.DictReader(io.StringIO(read(directory / "episodes.csv").decode())))
        assert len(table) == 27
        for result in results:
            key = (result["case"], result["mode"])
            assert result["case"].startswith("TS3:") and key not in fresh
            assert result["seed"] == cases[result["case"]]["seed"]
            csv_row = next(r for r in table if int(r["attempt"]) == result["attempt"])
            assert all(str(result[k]) == csv_row[k] for k in ("case", "mode", "outcome", "steps", "seed"))
            trace = directory / "traces" / f"{int(result['attempt']):03d}_{result['mode']}.jsonl"
            rows = [json.loads(line) for line in read(trace).splitlines() if line]
            assert len(rows) == result["steps"]
            assert [r["step"] for r in rows] == list(range(1, len(rows)+1))
            fresh[key] = dict(tag=tag, result=result, case=cases[result["case"]], rows=rows)
        manifests[tag], completions[tag] = manifest, completion
    ma, mb = manifests.values()
    for key in ("checkpoint_sha256", "config_sha256", "source_sha256", "constants", "constructor_options"):
        assert ma[key] == mb[key], key
    selected = {case for case, mode in fresh}
    assert len(fresh) == 54 and len(selected) == 18
    assert all((case, mode) in fresh for case in selected for mode in ("off", "v16", "v19"))
    cases = []
    for case in sorted(selected):
        reference = fresh[(case, "off")]
        row = dict(case=case, seed=reference["case"]["seed"],
                   scenario_sha256=reference["case"]["scenario_sha256"], modes={})
        for mode in ("off", "v16", "v19"):
            item = fresh[(case, mode)]
            row["modes"][mode] = dict(outcome=item["result"]["outcome"], steps=item["result"]["steps"],
                interventions=item["result"]["safety_v2_steps"], brake_steps=item["result"]["safety_v2_brake_steps"],
                versus_off_commands=compare(reference["rows"], item["rows"]))
        row["v19_vs_v16_commands"] = compare(fresh[(case, "v16")]["rows"], fresh[(case, "v19")]["rows"])
        row["v19_activation"] = activations(fresh[(case, "v19")]["rows"])
        cases.append(row)
    outcomes = {mode: dict(Counter(fresh[(case, mode)]["result"]["outcome"] for case in selected))
                for mode in ("off", "v16", "v19")}
    pairs = {}
    for candidate, reference in (("v16", "off"), ("v19", "off"), ("v19", "v16")):
        count = Counter()
        gains, losses = [], []
        for case in sorted(selected):
            ref = fresh[(case, reference)]["result"]["outcome"] == "goal"
            cand = fresh[(case, candidate)]["result"]["outcome"] == "goal"
            count["both_goal" if ref and cand else "rescued_failure" if cand else "lost_goal" if ref else "both_failed"] += 1
            if cand and not ref: gains.append(case)
            if ref and not cand: losses.append(case)
        pairs[candidate+"_vs_"+reference] = dict(counts=dict(count), gains=gains, losses=losses,
            reference_goals=count["both_goal"]+count["lost_goal"], reference_failures=count["rescued_failure"]+count["both_failed"])
    method = dict(active_cases=sum(bool(c["v19_activation"]["active_decisions"]) for c in cases),
        active_decisions=sum(c["v19_activation"]["active_decisions"] for c in cases),
        transferred_point_decisions=sum(c["v19_activation"]["transferred_point_decisions"] for c in cases),
        command_divergent_cases=sum(c["v19_vs_v16_commands"]["first_command_divergence"] is not None for c in cases),
        complete_exact_command_sequences_equal=sum(c["v19_vs_v16_commands"]["same_trace_length"] and c["v19_vs_v16_commands"]["first_exact_command_difference"] is None for c in cases))
    PROVENANCE[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    artifact = dict(created_utc=datetime.now(timezone.utc).isoformat(), analysis_episodes=0,
        evaluation_role="primary evaluation only; no controller selection or tuning from this audit",
        completed_records=54, completed_cases=18, completions=completions, outcomes=outcomes,
        paired_goal_outcomes=pairs, v19_method=method, cases=cases, provenance_sha256=PROVENANCE,
        command_comparison_absolute_tolerance=1e-7, shared_pre_state_absolute_tolerance=1e-9,
        limitations=["18 preselected primary cases, not the complete 1000-case population.",
            "Activations change geometry and need not change actions; transferred counts are point-decision exposures, not unique returns.",
            "Light traces cannot support full prediction re-scoring or causal explanations for all outcomes.",
            "Root-owned strict validated report separately checks campaign-level protocol and archive integrity."])
    (OUT / "audit.json").write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Completed primary quick V19 trace audit", "", f"Created {artifact['created_utc']}. Both runs completed cleanly:54/54 records,18 exact scene/seed triples. This audit ran no episodes and makes no controller-tuning recommendation.", "",
        "| Mode | Goals | Obstacle | Target | Boundary |", "|---|---:|---:|---:|---:|"]
    for mode in ("off", "v16", "v19"):
        counts=outcomes[mode]
        lines.append(f"| {mode} | {counts.get('goal',0)} | {counts.get('collision:obstacle',0)} | {counts.get('collision:target',0)} | {counts.get('collision:boundary',0)} |")
    lines += ["", "| Candidate/reference | Preserved goals | Lost goals | Rescued failures | Both failed |", "|---|---:|---:|---:|---:|"]
    for name, pair in pairs.items():
        counts=pair["counts"]
        lines.append("| "+name+" | "+" | ".join(str(counts.get(k,0)) for k in ("both_goal","lost_goal","rescued_failure","both_failed"))+" |")
    lines += ["", f"V19 transfers current returns on {method['active_decisions']} decisions across {method['active_cases']} cases ({method['transferred_point_decisions']} point-decision exposures). Its complete issued command sequences exactly equal V16 in {method['complete_exact_command_sequences_equal']}/18 cases; {method['command_divergent_cases']} cases have a command difference above1e-7.", "",
        "| Case | OFF | V16 | V19 | V19 active steps | V19/V16 first command divergence |", "|---|---|---|---|---|---|"]
    for row in cases:
        divergence=row["v19_vs_v16_commands"]["first_command_divergence"]
        active=", ".join(str(e["step"]) for e in row["v19_activation"]["events"]) or "None"
        lines.append("| "+" | ".join([row["case"],*[row["modes"][m]["outcome"] for m in ("off","v16","v19")],active,str(divergence["step"]) if divergence else "None"])+" |")
    lines += ["", "The JSON records each activation and first command difference, including shared pre-state/policy-action checks, commands and filter reasons. It also preserves all input hashes. This fixed subset is an evaluation sample, not a full-population safety estimate. No controller or threshold was selected, fitted or changed from these primary outcomes."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"outcomes":outcomes,"paired":pairs,"method":method},indent=2))


if __name__ == "__main__":
    main()
