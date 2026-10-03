"""Score recorded successful SAC commands from an exact shared saved state.

No environment/reset/step or new policy inference. Future commands are read from
the matched completed OFF trace, not predicted SAC behavior. Truth poses score
prediction error only; the prediction starts from recorded onboard state.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.diagnostics.safety.audit_v11_prediction import audit_arguments, clean
from tools.diagnostics.safety.suite_status import read_shared_bytes


def main():
    args = audit_arguments(__doc__)
    run = args.run_dir
    manifest_bytes = read_shared_bytes(run / "manifest.json")
    manifest = json.loads(manifest_bytes)
    trace_bytes = read_shared_bytes(run / "traces" / args.trace)
    rows = [json.loads(line) for line in trace_bytes.splitlines() if line]
    result = json.loads(read_shared_bytes(run / "attempts" / f"{args.attempt:03d}_result.json"))
    if len(rows) != result["steps"]:
        raise ValueError("Current trace is not a completed committed episode")
    reference = ROOT / "results/safety_dev/v9_paired_pilot"
    old_manifest = json.loads(read_shared_bytes(reference / "manifest.json"))
    case = next(c for c in manifest["cases"] if c["case"] == result["case"])
    old_case = next(c for c in old_manifest["cases"] if c["case"] == result["case"])
    for key in ("seed", "scenario_sha256"):
        if case[key] != old_case[key]:
            raise ValueError("Matched scenario/seed mismatch")
    for key in ("checkpoint_sha256", "config_sha256"):
        if manifest[key] != old_manifest[key]:
            raise ValueError("Matched model/config mismatch")
    import csv
    import io
    episode = next(r for r in csv.DictReader(io.StringIO(read_shared_bytes(reference / "episodes.csv").decode()))
                   if r["case"] == result["case"] and r["mode"] == "off")
    if episode["outcome"] != "goal":
        raise ValueError("Reference must be the recorded successful policy")
    reference_bytes = read_shared_bytes(reference / "traces" / f"{int(episode['attempt']):03d}_off.jsonl")
    off = [json.loads(line) for line in reference_bytes.splitlines() if line]
    if len(off) != int(episode["steps"]):
        raise ValueError("Reference trace is incomplete")
    keys = ("rudder_command", "signed_rpm_command")
    first = next((i for i, (a, b) in enumerate(zip(rows, off))
                  if not np.allclose([a[k] for k in keys], [b[k] for k in keys], atol=1e-7, rtol=0)), None)
    if first is None:
        raise ValueError("No command divergence to score")
    for a, b in zip(rows[:first+1], off[:first+1]):
        np.testing.assert_allclose(a["pre_state"], b["pre_state"], atol=1e-9, rtol=0)
    row = rows[first]
    if row.get("diagnostic_decision", {}).get("snapshot") is None:
        raise ValueError("Missing onboard prediction snapshot")
    flags = row["filter"]
    if (flags.get("calibrated_braking") or flags.get("dual_brake_sequences", 0)
            or flags.get("v10_target_turn_rate_deg_s", 0)):
        raise ValueError("This saved audit supports the nominal weak-brake/CV preset only")
    dependencies = ["src/constants.py", "src/constant_temp.py", "src/classical/common.py",
                    "src/safety_v2.py", "src/safety_v3.py", "src/safety_v4.py",
                    "src/safety_v7.py", "src/ship.py", "src/reference_controller.py", "bluefin/dynamics.py"]
    source_hashes = {}
    for name in dependencies:
        digest = hashlib.sha256(read_shared_bytes(ROOT / name)).hexdigest()
        if digest != manifest["source_sha256"][name]:
            raise ValueError("Source differs from evaluated predictor: " + name)
        source_hashes[name] = digest
    import constants as cfg
    for key, text in manifest["constants"].items():
        try:
            setattr(cfg, key, ast.literal_eval(text))
        except (ValueError, SyntaxError):
            pass
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    import safety_v4 as v4
    from safety_v7 import replace_first_action
    values = copy.deepcopy(row["diagnostic_decision"]["snapshot"])
    values.pop("units", None)
    for key in ("tangent", "right", "centre", "points", "edges_a", "edges_b"):
        values[key] = np.asarray(values[key], dtype=float)
    values["tracks"] = [cc.TrackView(t["id"], np.asarray(t["position"]), np.asarray(t["velocity"]),
                                    t["heading"]) for t in values["tracks"]]
    snap = cc.Snapshot(**values)
    saved_act = row["diagnostic_decision"]["actuators_before_decision"]
    act = cc.Actuators()
    act.servo, act.executed, act.buffer = saved_act["servo"], saved_act["executed"], copy.deepcopy(saved_act["buffer"])
    count = int(np.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))
    commands = off[first:first+count]
    if len(commands) != count:
        raise ValueError("Successful reference ends before the full prediction horizon")
    sequences = np.array([[r["rudder_command"], np.nan if r["signed_rpm_command"] < 0 else
                           (r["signed_rpm_command"] - cfg.CRUISE_RPM) / cfg.RPM_DELTA] for r in commands])[None]
    ro = v3.rollout_seq(snap, act, sequences)

    def components(positions, headings, times):
        arrays = {"static_memory": cc.point_clearance(positions, headings, snap.points,
                    reach=cc.HALF_L+1.) - v2.GAP_STATIC_M,
                  "boundary": cc.boundary_clearance(positions, headings, snap.edges_a,
                    snap.edges_b) - v2.GAP_BOUNDARY_M}
        for t in snap.tracks:
            arrays[f"track:{t.id}"] = cc.target_gap(positions, headings, times, t) - v2.GAP_TARGET_M
        return {key: dict(clearance_m=float(value.min()), minimum_time_s=float(times[int(np.argmin(value))]),
                          first_violation_s=float(times[np.flatnonzero(value[:, 0] < 0)[0]])
                            if (value < 0).any() else None) for key, value in arrays.items()}

    t, clearance = v2.SafetyFilterV2._evaluate(None, snap, ro)
    errors = []
    for j, future in enumerate(commands):
        k = (j+1) * cc.SUBSTEPS - 1
        actual = future["post_state"]
        errors.append(dict(horizon_s=float((j+1)*cfg.UPDATE_RATE), predicted_position=ro.positions[k, 0],
            actual_position=actual[:2], position_error_m=float(np.linalg.norm(ro.positions[k, 0]-actual[:2])),
            heading_error_deg=float(np.degrees(cc.wrap_pi(ro.headings[k, 0]-np.radians(actual[2]))))))
    actual_positions = np.asarray([r["post_state"][:2] for r in commands])[:, None]
    actual_headings = np.radians([r["post_state"][2] for r in commands])[:, None]
    actual_times = cfg.UPDATE_RATE * np.arange(1, count+1)
    # Ordinary bank reconstruction uses the actual pre-command snapshot/history.
    traffic = any(np.linalg.norm(track.position-snap.position) < v2.ENGAGE_RANGE_M for track in snap.tracks)
    action = np.asarray(row["policy_action"], dtype=np.float32)
    rejoin = v3.SafetyFilterV3._rejoin(None, snap)
    candidates = np.asarray([tuple(action)] + [(r,t) for r in v2.RUDDERS for t in v2.THROTTLES]
                            + [tuple(rejoin)] + ([(r,np.nan) for r in v2.BRAKE_RUDDERS] if traffic else []))
    recoveries = [(r, np.nan if t is v2.BRAKE else t) for r,t in v2.RECOVERY if traffic or t is not v2.BRAKE]
    templates = v4.recovery_templates(recoveries)
    bank = v4.sequence_bank(candidates, templates)
    bf, bc = v2.SafetyFilterV2._evaluate(None, snap, v3.rollout_seq(snap, act, bank))
    bf, bc = bf.reshape(len(candidates), -1), bc.reshape(len(candidates), -1)
    margins = np.where(np.isposinf(bf), bc, -np.inf).max(axis=1)
    nonbrake = margins[~np.isnan(candidates[:, 1])]
    continuation_clearance = -np.inf
    if first and rows[first-1]["plan"] is not None and len(rows[first-1]["plan"]) > 1:
        continuation = v3.continuation(np.asarray(rows[first-1]["plan"], dtype=float))
        cf, cv = v2.SafetyFilterV2._evaluate(None, snap, v3.rollout_seq(snap, act, continuation[None]))
        if np.isposinf(cf[0]):
            continuation_clearance = float(cv[0])
    best = max(float(nonbrake.max()), continuation_clearance)
    floor = min(v2.TRIGGER_MARGIN_M, best-v2.ROOM_SLACK_M) if np.isfinite(best) else None
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope=__doc__, case=case,
        evaluated_result=result, reference_outcome=episode["outcome"], first_divergence_step=first+1,
        matched_pre_states_absolute_tolerance=1e-9, model="recorded onboard state; identified nominal dynamics; weak brake",
        full_horizon_s=float(ro.times[-1]), fixed_recorded_policy_commands=sequences[0],
        predicted_first_violation_s=float(t[0]), predicted_clearance_m=float(clearance[0]),
        predicted_components=components(ro.positions, ro.headings, ro.times),
        actual_recorded_pose_components_against_frozen_onboard_scene=components(actual_positions, actual_headings, actual_times),
        actual_pose_check_limits="0.5s samples only; no terminal check because true sway speed is absent in OFF trace; frozen observed scene, not true future geometry",
        prediction_errors=errors, current_bank=dict(traffic=traffic, nonbrake_best_margin=float(nonbrake.max()),
            continuation_clearance=continuation_clearance, ordinary_selection_best=best, ordinary_selection_floor=floor,
            parent_checked_clearance=flags.get("checked_clearance"), parent_reason=flags.get("why")),
        source_sha256=source_hashes, trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        reference_trace_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=["The successful future commands are a recorded counterfactual, not an online-available oracle.",
            "This does not recreate future policy observations or claim an alternative closed-loop outcome.",
            "Truth poses score the forecast only; no hidden state initializes or chooses a prediction."])
    if args.output_dir:
        with (args.output_dir / "audit.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(clean(output), indent=2, allow_nan=False)+"\n")
        with (args.output_dir / "reproducer_evaluated_bytes.py").open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
        print("OUTPUT", args.output_dir / "audit.json")
    print(json.dumps(clean({k:output[k] for k in ("first_divergence_step", "predicted_first_violation_s",
        "predicted_clearance_m", "predicted_components", "actual_recorded_pose_components_against_frozen_onboard_scene", "current_bank", "prediction_errors")}),indent=2))


if __name__ == "__main__":
    main()
