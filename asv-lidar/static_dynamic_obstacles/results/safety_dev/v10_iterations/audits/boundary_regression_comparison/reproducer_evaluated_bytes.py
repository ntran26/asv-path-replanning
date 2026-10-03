"""Read saved matched boundary-regression traces; never run an environment.

This audit reports recorded decisions only. Missing perception/actuator snapshots
are not reconstructed from truth. Use --tag NEW_NAME to save; default is read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.diagnostics.safety.suite_status import read_shared_bytes

CASES = ("DV3:DV3-BO-CV-04", "TS2:CH-CR-CV-031", "TS2:BAS-NU-CV-070")
RUNS = ("v9_paired_pilot", "v10_iterations/conditional_policy_32",
        "v10_iterations/feasible_policy_32", "v10_iterations/prefix_success16")


def same_command(a, b):
    return all(math.isclose(float(a[key]), float(b[key]), rel_tol=0., abs_tol=1e-7)
               for key in ("rudder_command", "signed_rpm_command"))


def decision(row):
    f = row["filter"]
    risk = f.get("risk_monitor", {})
    hazards = risk.get("hazards", [])
    boundary = [h for h in hazards if h["kind"] == "boundary"]
    keys = ("mode", "why", "checked_clearance", "best_margin", "policy_margin",
            "policy_safe", "any_safe", "n_safe_candidates", "continuation_checked",
            "continuation_ok", "continuation_clearance", "v7_repair_clearance",
            "v7_repair_first_violation", "v10_proposed_clearance", "v10_policy_tail_clearance",
            "v10_proposed_first_violation", "v10_policy_first_violation",
            "v10_proposed_currently_passing", "v10_policy_tail_passing",
            "v9_proposed_clearance", "v9_policy_tail_clearance", "v9_proposed_first_violation",
            "v9_policy_first_violation", "v11_policy_prefix_search_checked",
            "v11_policy_prefix_preserved", "policy_search", "escape_search", "policy_prefix_search")
    return dict(step=row["step"], pre_state=row["pre_state"], policy_action=row["policy_action"],
        rudder=row["rudder_command"], rpm=row["signed_rpm_command"], changed=row["changed"],
        brake=row["brake"], checks={key: f[key] for key in keys if key in f},
        shadow_recommends_rescue=risk.get("recommend_rescue"),
        shadow_world_velocity=risk.get("world_velocity_mps"), shadow_yaw_rate=risk.get("yaw_rate_rad_s"),
        closest_boundary=min(boundary, key=lambda h: h["current_clearance_m"], default=None),
        target_hazards=[h for h in hazards if h["kind"] == "target"],
        has_full_prediction_snapshot=bool(row.get("diagnostic_decision", {}).get("snapshot")))


def divergence(rows, reference):
    for index, (row, other) in enumerate(zip(rows, reference)):
        if not same_command(row, other):
            delta = [abs(float(x)-float(y)) for x, y in zip(row["pre_state"], other["pre_state"])]
            return dict(step=index+1, maximum_pre_state_difference=max(delta),
                        same_pre_state=all(d <= 1e-9 for d in delta),
                        candidate=decision(row), reference=decision(other))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    args = parser.parse_args()
    output_dir = None
    if args.tag:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
            parser.error("--tag must be a plain directory name")
        output_dir = ROOT / "results/safety_dev/v10_iterations/audits" / args.tag
        output_dir.mkdir(parents=True, exist_ok=False)
    manifests, traces, episodes, hashes = {}, {}, {}, {}
    canonical, checkpoint, config = {}, None, None
    for run in RUNS:
        directory = ROOT / "results/safety_dev" / run
        if not (directory / "completion.json").exists():
            raise ValueError("Expected completed saved run: " + run)
        data = read_shared_bytes(directory / "manifest.json")
        manifest = json.loads(data)
        hashes[run + "/manifest.json"] = hashlib.sha256(data).hexdigest()
        manifests[run] = {key: manifest.get(key) for key in
            ("checkpoint_sha256", "config_sha256", "source_sha256", "constructor_options", "snapshots")}
        if checkpoint is None:
            checkpoint, config = manifest["checkpoint_sha256"], manifest["config_sha256"]
        if (manifest["checkpoint_sha256"], manifest["config_sha256"]) != (checkpoint, config):
            raise ValueError("Checkpoint/config mismatch")
        for case in manifest["cases"]:
            if case["case"] not in CASES:
                continue
            identity = {key: case[key] for key in ("case", "seed", "scenario_sha256")}
            if case["case"] in canonical and canonical[case["case"]] != identity:
                raise ValueError("Case, reset seed or scene digest mismatch")
            canonical[case["case"]] = identity
        csv_bytes = read_shared_bytes(directory / "episodes.csv")
        hashes[run + "/episodes.csv"] = hashlib.sha256(csv_bytes).hexdigest()
        for episode in csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))):
            case, mode = episode["case"], episode["mode"]
            if case not in CASES:
                continue
            if int(episode["seed"]) != canonical[case]["seed"]:
                raise ValueError("Episode seed mismatch")
            name = f"{int(episode['attempt']):03d}_{mode}.jsonl"
            trace_bytes = read_shared_bytes(directory / "traces" / name)
            rows = [json.loads(line) for line in trace_bytes.splitlines() if line]
            if len(rows) != int(episode["steps"]) or [r["step"] for r in rows] != list(range(1, len(rows)+1)):
                raise ValueError("Incomplete/nonsequential trace")
            key = (run, mode, case)
            if key in traces:
                raise ValueError("Duplicate episode")
            traces[key], episodes[key] = rows, episode
            hashes[run + "/traces/" + name] = hashlib.sha256(trace_bytes).hexdigest()
    records = []
    for (run, mode, case), rows in traces.items():
        off = traces[(RUNS[0], "off", case)]
        v8 = traces[(RUNS[0], "v8", case)]
        episode = episodes[(run, mode, case)]
        records.append(dict(run=run, mode=mode, case=case, outcome=episode["outcome"],
            steps=len(rows), intervention_steps=[r["step"] for r in rows if r["changed"]],
            first_vs_off=divergence(rows, off), first_vs_v8=divergence(rows, v8),
            all_commands_equal_off=len(rows)==len(off) and all(same_command(a,b) for a,b in zip(rows,off)),
            full_snapshot_count=sum(bool(r.get("diagnostic_decision", {}).get("snapshot")) for r in rows),
            decisions=[decision(r) for r in rows]))
    # Exact saved floors, without treating a rounded diagnostic as an exact max.
    floors = {}
    for case in CASES:
        row = next(r for r in records if r["run"] == RUNS[-1] and r["case"] == case)
        first = row["first_vs_off"]["candidate"]
        checks = first["checks"]
        rounded_best = float(checks["best_margin"])
        if checks["continuation_ok"] and float(checks["continuation_clearance"]) > rounded_best + .005:
            best = float(checks["continuation_clearance"])
            floor = min(.15, best - .20)
            proof = "Exact continuation exceeds the entire rounding interval of the best nonbraking margin."
        elif math.isfinite(rounded_best) and rounded_best - .005 - .20 >= .15:
            floor = .15
            proof = "Even the lower rounding bound saturates the unchanged trigger floor."
        elif not checks["any_safe"]:
            floor = None
            proof = "No hard-passing original plan: fallback branch has no certified candidate-pool floor."
        else:
            raise ValueError("Exact floor is not recoverable from saved scalar diagnostics")
        prefix = checks["policy_prefix_search"]
        room = prefix["maximum_hard_passing_clearance"]
        floors[case] = dict(step=first["step"], parent_reason=checks["why"],
            current_parent_clearance=checks["checked_clearance"], floor=floor, proof=proof,
            prefix_best_hard_passing_clearance=room, prefix_hard_passing_count=prefix["hard_passing_plans"],
            prefix_meets_existing_floor=None if floor is None else room >= max(0., floor),
            original_prefix_acceptance_margin=prefix["minimum_clearance"])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Saved logs only; zero new episodes; no environment or model rollout calls.",
        cases=canonical, manifests=manifests, source_data_sha256=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command_equality_absolute_tolerance=1e-7, source_behavior_identical_assumed=False,
        replay_limit="These traces lack full onboard snapshots and actuator history; no faithful prediction replay is claimed.",
        first_divergence_candidate_floors=floors, records=records)
    if output_dir:
        with (output_dir / "audit.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with (output_dir / "reproducer_evaluated_bytes.py").open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
        print("OUTPUT", output_dir / "audit.json")
    else:
        print("READ_ONLY: no files written")
    print(json.dumps(floors, indent=2))


if __name__ == "__main__":
    main()
