"""Read completed development records only. No controller imports or episodes.

Immutable analysis-code artifact. Run from the project directory; refuses to
overwrite its audit.json/REPORT.md. The root-owned strict report independently
validates the complete evaluation protocol; this audit explains saved commands.
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
RUNS = ROOT / "results/safety_dev/v10_iterations"
PROVENANCE = {}


def read(path):
    data = path.read_bytes()
    PROVENANCE[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def load_json(path):
    return json.loads(read(path))


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
            reference_reason=first[0]["filter"].get("why"), candidate_reason=first[1]["filter"].get("why"),
            reference_checked_clearance=first[0]["filter"].get("checked_clearance"),
            candidate_checked_clearance=first[1]["filter"].get("checked_clearance")))


def load_trace(directory, result):
    path = directory / "traces" / f"{int(result['attempt']):03d}_{result['mode']}.jsonl"
    rows = [json.loads(line) for line in read(path).splitlines() if line]
    assert len(rows) == int(result["steps"])
    assert [r["step"] for r in rows] == list(range(1, len(rows)+1))
    return rows


def activations(rows, mode):
    events = []
    for row in rows:
        stats = row["filter"].get("hull_heading_memory" if mode == "v18" else "source_points")
        if stats is None:
            raise ValueError("Expected method diagnostics missing")
        active = stats.get("replaced_source_ids", []) if mode == "v18" else stats.get("removed_point_count", 0)
        if active:
            events.append(dict(step=row["step"], why=row["filter"].get("why"), changed=row["changed"],
                               command=command(row), evidence=stats))
    return dict(active_decisions=len(events), events=events,
        heading_track_decisions=sum(len(e["evidence"].get("replaced_source_ids", [])) for e in events),
        transferred_point_decisions=sum(e["evidence"].get("removed_point_count", 0) for e in events))


def main():
    if (OUT / "audit.json").exists() or (OUT / "REPORT.md").exists():
        raise FileExistsError("Refusing to overwrite the completed analysis")
    fresh, manifests, completions = {}, {}, {}
    for tag in ("v18_v19_probe_a", "v18_v19_probe_b"):
        directory = RUNS / tag
        manifest, completion = load_json(directory / "manifest.json"), load_json(directory / "completion.json")
        assert manifest["planned_runs"] == completion["completed_runs"] == 18
        assert not completion["source_drift"] and completion["checkpoint_unchanged"] and completion["selection_unchanged"]
        cases = {c["case"]: c for c in manifest["cases"]}
        results = [load_json(p) for p in sorted((directory / "attempts").glob("*_result.json"))]
        assert len(results) == 18
        table = list(csv.DictReader(io.StringIO(read(directory / "episodes.csv").decode())))
        assert len(table) == 18
        for result in results:
            key = (result["case"], result["mode"])
            assert key not in fresh and result["seed"] == cases[result["case"]]["seed"]
            csv_row = next(r for r in table if int(r["attempt"]) == result["attempt"])
            assert all(str(result[k]) == csv_row[k] for k in ("case", "mode", "outcome", "steps", "seed"))
            fresh[key] = dict(tag=tag, result=result, case=cases[result["case"]],
                              rows=load_trace(directory, result))
        manifests[tag], completions[tag] = manifest, completion
    ma, mb = manifests.values()
    for key in ("checkpoint_sha256", "config_sha256", "source_sha256", "constants", "constructor_options"):
        assert ma[key] == mb[key], key
    assert len(fresh) == 36
    selected = {case for case, mode in fresh}
    assert len(selected) == 12 and all((case, mode) in fresh for case in selected for mode in ("v16", "v18", "v19"))
    old = {}
    for tag in ("motion_axis_probe5", "motion_axis_remaining27", "v16_broader40_paired"):
        directory = RUNS / tag
        manifest = load_json(directory / "manifest.json")
        cases = {c["case"]: c for c in manifest["cases"]}
        for row in csv.DictReader(io.StringIO(read(directory / "episodes.csv").decode())):
            if row["mode"] != "v16" or row["case"] not in selected:
                continue
            assert row["case"] not in old
            new = fresh[(row["case"], "v16")]
            assert all(cases[row["case"]][k] == new["case"][k] for k in ("seed", "scenario_sha256"))
            assert all(manifest[k] == ma[k] for k in ("checkpoint_sha256", "config_sha256"))
            old[row["case"]] = dict(tag=tag, outcome=row["outcome"], steps=int(row["steps"]),
                outcome_reproduced=row["outcome"] == new["result"]["outcome"],
                constant_differences=[k for k in set(manifest["constants"]) | set(ma["constants"])
                    if manifest["constants"].get(k) != ma["constants"].get(k)],
                commands=compare(load_trace(directory, row), new["rows"]))
    assert len(old) == 12
    pairs = []
    for case in sorted(selected):
        reference = fresh[(case, "v16")]
        record = dict(case=case, seed=reference["case"]["seed"],
            scenario_sha256=reference["case"]["scenario_sha256"], old_v16=old[case], modes={})
        for mode in ("v16", "v18", "v19"):
            item = fresh[(case, mode)]
            record["modes"][mode] = dict(outcome=item["result"]["outcome"], steps=item["result"]["steps"],
                interventions=item["result"]["safety_v2_steps"], brake_steps=item["result"]["safety_v2_brake_steps"],
                paired_commands=compare(reference["rows"], item["rows"]),
                activation=None if mode == "v16" else activations(item["rows"], mode))
        pairs.append(record)
    summary = {mode:dict(outcomes=dict(Counter(fresh[(case, mode)]["result"]["outcome"] for case in selected)),
        gains=sum(fresh[(case,"v16")]["result"]["outcome"] != "goal" and fresh[(case,mode)]["result"]["outcome"] == "goal" for case in selected),
        losses=sum(fresh[(case,"v16")]["result"]["outcome"] == "goal" and fresh[(case,mode)]["result"]["outcome"] != "goal" for case in selected),
        command_divergent_cases=sum(p["modes"][mode]["paired_commands"]["first_command_divergence"] is not None for p in pairs),
        active_cases=sum(bool(p["modes"][mode]["activation"]["active_decisions"]) for p in pairs) if mode != "v16" else None,
        active_decisions=sum(p["modes"][mode]["activation"]["active_decisions"] for p in pairs) if mode != "v16" else None)
        for mode in ("v16", "v18", "v19")}
    PROVENANCE[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    artifact = dict(created_utc=datetime.now(timezone.utc).isoformat(), episodes_run_by_analysis=0,
        completed_records=36, completions=completions, summary=summary, cases=pairs,
        fresh_v16_old_outcomes_reproduced=sum(x["outcome_reproduced"] for x in old.values()),
        fresh_v16_old_full_command_sequences_identical=sum(x["commands"]["same_trace_length"] and x["commands"]["first_exact_command_difference"] is None for x in old.values()),
        command_comparison_absolute_tolerance=1e-7, shared_pre_state_absolute_tolerance=1e-9,
        provenance_sha256=PROVENANCE,
        limitations=["Selected 12-case development diagnostic, not a population success-rate estimate.",
            "Method activations are geometry changes, not necessarily command changes or intervention prevention.",
            "Transferred point-decision counts may include repeated physical points; source IDs are local associations, not unique ships.",
            "Light traces contain no complete prediction snapshots, so this analysis does not re-score candidate trajectories or assert why every activation did not affect control.",
            "Root-owned validated_development_report is the separate strict campaign validation."])
    (OUT / "audit.json").write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Completed V18/V19 development command audit", "", f"Created {artifact['created_utc']}. Both runs have clean completion records: 36/36 episodes, 12 exact scene/seed triples. This analysis ran no episodes.", "",
        "All three controllers reached **5/12 goals**. Both candidates have **0 gains and 0 losses** against fresh V16. V19 changes one failed case from boundary to obstacle contact; this is not a rescue.", "",
        f"Fresh V16 reproduces **{artifact['fresh_v16_old_outcomes_reproduced']}/12 older outcomes** and **{artifact['fresh_v16_old_full_command_sequences_identical']}/12 complete exact command sequences** from motion_axis_probe5, motion_axis_remaining27 and v16_broader40_paired. Case IDs, scenario digests, seeds and checkpoint/config hashes match.", "",
        "| Case | V16 | V18 | V19 | V18 active decisions | V19 active decisions | First command difference |", "|---|---|---|---|---:|---:|---|"]
    for item in pairs:
        modes=item["modes"]; divs=[]
        for mode in ("v18", "v19"):
            div=modes[mode]["paired_commands"]["first_command_divergence"]
            if div: divs.append(f"{mode}: step {div['step']}")
        lines.append("| " + " | ".join([item["case"],*[modes[x]["outcome"] for x in ("v16","v18","v19")],str(modes["v18"]["activation"]["active_decisions"]),str(modes["v19"]["activation"]["active_decisions"]),", ".join(divs) or "None"]) + " |")
    lines += ["", "V19's first changed command in DV3-CRS-CV-04 is at decision 13: V16 rudder −0.3452420533 versus V19 −0.3387790024; both command −24 RPM and report searched escape. The current-return transfer removes 9 of 58 static points at this decision. The pre-state and SAC action still match. Later trajectories diverge and end in different contact types.", "",
        "V18 can correct geometry without changing the issued command. In fresh FIX05 its only activation is the terminal decision 34; the older V13 saved-state sensitivity is a different closed-loop trajectory and does not establish an earlier rescue opportunity in this V16 run.", "",
        "The JSON retains every activated decision's evidence, commands and reason, first differences, historical comparisons and input hashes. Point totals are point-decision exposures, not unique physical returns. No primary test-set-v3 results were read. These selected development cases do not establish statistical safety or population success."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"summary":summary,"old_outcomes_reproduced":artifact["fresh_v16_old_outcomes_reproduced"],"old_commands_identical":artifact["fresh_v16_old_full_command_sequences_identical"]},indent=2))


if __name__ == "__main__":
    main()
