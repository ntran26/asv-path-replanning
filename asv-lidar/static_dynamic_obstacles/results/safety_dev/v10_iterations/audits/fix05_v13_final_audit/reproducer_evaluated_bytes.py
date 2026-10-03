"""Score saved V13 plans under two existing brake models; no new episodes.

Uses recorded onboard snapshots/actuator history to forecast. Simulator truth
is used only to score prediction error and labelled target-motion sensitivity.
Without --tag this is read-only. A supplied output tag must not already exist.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.diagnostics.safety.audit_v11_prediction import audit_arguments, clean
from tools.diagnostics.safety.suite_status import read_shared_bytes


def main():
    args = audit_arguments(__doc__)
    run = args.run_dir
    trace_bytes = read_shared_bytes(run / "traces" / args.trace)
    manifest_bytes = read_shared_bytes(run / "manifest.json")
    manifest = json.loads(manifest_bytes)
    rows = [json.loads(line) for line in trace_bytes.splitlines() if line]
    result = json.loads(read_shared_bytes(run / "attempts" / f"{args.attempt:03d}_result.json"))
    if len(rows) != result["steps"]:
        raise ValueError("Completed result and trace length disagree")
    case = next(item for item in manifest["cases"] if item["case"] == result["case"])
    dependencies = ["src/constants.py", "src/constant_temp.py", "src/classical/common.py",
                    "src/safety_v2.py", "src/safety_v3.py", "src/safety_v4.py",
                    "src/safety_v6.py", "src/safety_v7.py", "src/safety_prediction.py",
                    "src/ship.py", "src/reference_controller.py", "bluefin/dynamics.py"]
    matched = {}
    for name in dependencies:
        digest = hashlib.sha256(read_shared_bytes(ROOT / name)).hexdigest()
        if digest != manifest["source_sha256"][name]:
            raise ValueError("Prediction source differs from evaluated source: " + name)
        matched[name] = digest
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
    from safety_prediction import rollout_seq
    from reference_controller import hull_separation
    import ship

    def snapshot(row):
        values = copy.deepcopy(row["diagnostic_decision"]["snapshot"])
        values.pop("units", None)
        for key in ["tangent", "right", "centre", "points", "edges_a", "edges_b"]:
            values[key] = np.asarray(values[key], dtype=float)
        values["tracks"] = [cc.TrackView(t["id"], np.asarray(t["position"]),
                                        np.asarray(t["velocity"]), t["heading"])
                            for t in values["tracks"]]
        return cc.Snapshot(**values)

    def actuators(row):
        saved = row["diagnostic_decision"]["actuators_before_decision"]
        act = cc.Actuators()
        act.servo, act.executed = saved["servo"], saved["executed"]
        act.buffer = copy.deepcopy(saved["buffer"])
        return act

    def command(row):
        rpm = float(row["signed_rpm_command"])
        if rpm < 0 and not np.isclose(rpm, v2.ASTERN_RPM):
            raise ValueError("Unmodelled partial astern command")
        return np.array([row["rudder_command"], np.nan if rpm < 0 else
                         (rpm - cfg.CRUISE_RPM) / cfg.RPM_DELTA])

    def truth_target(row):
        targets = row["diagnostic_before"]["truth_scoring_only"]["targets"]
        if len(targets) != 1:
            raise ValueError("This audit requires exactly one recorded target")
        t = targets[0]
        return cc.TrackView(0, np.array([t["x"], t["y"]]), np.array(t["velocity"]),
                            np.radians(t["heading"]))

    def rollout(snap, act, sequences, fast):
        if fast:
            return rollout_seq(snap, act, sequences,
                               brake_efficiency=ship.REVERSE_THRUST_EFFICIENCY,
                               brake_delay_s=0.)
        return v3.rollout_seq(snap, act, sequences)

    def body(snap, act, row, fast):
        state = np.array([max(0., snap.u), snap.v, snap.r, snap.heading,
                          act.servo, snap.x, snap.y])[:, None]
        params = {key: np.array([value]) for key, value in cc.IDENTIFIED.items()}
        rudder = np.array([-cc.MAX_RUDDER_RAD * row["rudder_command"]])
        pending = ([np.array([x]) for x in act.buffer] if act.buffer is not None
                   else [rudder.copy() for _ in range(cc.DELAY_STEPS)])
        rpm = row["signed_rpm_command"]
        efficiency = ship.REVERSE_THRUST_EFFICIENCY if fast else v2.BRAKE_EFFICIENCY
        delay = 0. if fast else v2.BRAKE_DELAY_S
        deceleration = ship.braking_thrust(rpm, efficiency=efficiency) / ship.M11
        for j in range(cc.SUBSTEPS):
            pending.append(rudder)
            state = cc.dyn.rk4_step(state, np.array([max(0., rpm)]),
                                    pending.pop(0), params, cc.PRED_DT)
            if rpm < 0 and j * cc.PRED_DT >= delay:
                state[0, 0] = max(0., state[0, 0] - deceleration * cc.PRED_DT)
        return state[:, 0]

    def metrics(snap, ro):
        first, clear = v2.SafetyFilterV2._evaluate(None, snap, ro)
        components = {
            "static": float((cc.point_clearance(ro.positions, ro.headings, snap.points,
                reach=cc.HALF_L + 1.) - v2.GAP_STATIC_M).min()),
            "boundary": float((cc.boundary_clearance(ro.positions, ro.headings,
                snap.edges_a, snap.edges_b) - v2.GAP_BOUNDARY_M).min())}
        for track in snap.tracks:
            components[str(track.id)] = float((cc.target_gap(ro.positions, ro.headings,
                ro.times, track) - v2.GAP_TARGET_M).min())
        return dict(first_violation_s=float(first[0]), clearance_m=float(clear[0]),
                    passes=bool(np.isposinf(first[0]) and clear[0] >= 0), components=components)

    timeline, plans, tracks, conditioned = [], [], [], []
    times = np.array([r["diagnostic_before"]["elapsed_time"] for r in rows])
    target_positions = np.array([truth_target(r).position for r in rows])
    for index, row in enumerate(rows):
        flags = row["filter"]
        if (flags.get("calibrated_braking") or flags.get("dual_brake_sequences", 0)
                or flags.get("v10_target_turn_rate_deg_s", 0)
                or flags.get("countersteer_recovery_enabled")):
            raise ValueError("Saved evaluator is outside this audit's supported nominal-CV preset")
        snap, act, target = snapshot(row), actuators(row), truth_target(row)
        own = row["diagnostic_before"]["truth_scoring_only"]["own"]
        actual = np.asarray(row["post_state"])
        one = dict(step=row["step"], time_s=float(times[index]), why=flags.get("why"),
                   changed=row["changed"], rpm=row["signed_rpm_command"],
                   requested_brake=row["brake"], target_position=target.position,
                   target_velocity=target.velocity, checked_clearance=flags.get("checked_clearance"),
                   any_safe=flags.get("any_safe"), plan_recorded=row["plan"] is not None)
        for fast, label in [(False, "weak"), (True, "fast")]:
            ro = rollout(snap, act, command(row)[None, None], fast)
            state = body(snap, act, row, fast)
            np.testing.assert_allclose(ro.positions[-1, 0], state[5:], atol=1e-12)
            np.testing.assert_allclose(ro.headings[-1, 0], state[3], atol=1e-12)
            one[label] = dict(predicted_end_position=state[5:],
                position_error_m=float(np.linalg.norm(state[5:] - actual[:2])),
                displacement_error_m=float(np.linalg.norm((state[5:] - snap.position)
                    - (actual[:2] - np.asarray(row["pre_state"][:2])))),
                predicted_end_u_mps=float(state[0]), actual_end_u_mps=float(actual[3]),
                u_error_mps=float(state[0] - actual[3]),
                u_increment_error_mps=float(state[0] - snap.u - (actual[3] - own["u_body"])),
                heading_error_deg=float(np.degrees(cc.wrap_pi(state[3] - np.radians(actual[2])))))
        timeline.append(one)
        for track in snap.tracks:
            tracks.append(dict(step=row["step"], id=track.id, position=track.position,
                velocity=track.velocity, truth_position=target.position, truth_velocity=target.velocity,
                position_error_m=float(np.linalg.norm(track.position - target.position)),
                velocity_error_mps=float(np.linalg.norm(track.velocity - target.velocity)),
                heading_error_deg=float(np.degrees(cc.wrap_pi(track.heading - target.heading)))))

        plan, origin = None, None
        if row["plan"] is not None:
            plan = replace_first_action(np.asarray(row["plan"], dtype=float),
                                        np.asarray(flags["chosen"], dtype=np.float32))
            if row["brake"]:
                plan[0, 1] = np.nan
            origin = "recorded retained plan, padded by unchanged V7 convention"
        elif flags.get("why") in ("nominal", "handback"):
            traffic = any(np.linalg.norm(t.position - snap.position) < v2.ENGAGE_RANGE_M
                          for t in snap.tracks)
            recoveries = [(r, np.nan if t is v2.BRAKE else t) for r, t in v2.RECOVERY
                          if traffic or t is not v2.BRAKE]
            bank = v4.sequence_bank(np.asarray(row["policy_action"])[None],
                                    v4.recovery_templates(recoveries))
            first, clear = v2.SafetyFilterV2._evaluate(None, snap, rollout(snap, act, bank, False))
            safe_clear = np.where(np.isposinf(first), clear, -np.inf)
            if np.isfinite(safe_clear.max()):
                plan = bank[int(np.argmax(safe_clear))]
                origin = "reconstructed best policy primitive, not retained by preset"
        if plan is not None:
            record = dict(step=row["step"], why=flags.get("why"), plan_origin=origin,
                          plan=plan, contains_brake=bool(np.isnan(plan[:, 1]).any()),
                          logged_clearance=flags.get("checked_clearance"))
            for fast, label in [(False, "weak"), (True, "fast")]:
                ro = rollout(snap, act, plan[None], fast)
                record[label] = metrics(snap, ro)
                truth_snap = copy.copy(snap)
                truth_snap.tracks = [target]
                record[label]["truth_initial_target_cv_scoring"] = metrics(truth_snap, ro)
                if row["step"] in (29, 34):
                    # Component-at-a-time sensitivity; no new plan or command.
                    # Truth substitutions are scoring only. Previous same-ID
                    # heading is an explicitly labelled onboard-only counterfactual.
                    sensitivities = {}
                    previous = {t.id: t for t in snapshot(rows[index-1]).tracks} if index else {}
                    for attribute in ("position", "velocity", "heading", "previous_same_id_heading"):
                        alternate = copy.copy(snap)
                        alternate.tracks = []
                        for track in snap.tracks:
                            changed_track = copy.copy(track)
                            if track.id >= 0:
                                if attribute == "previous_same_id_heading":
                                    if track.id in previous:
                                        changed_track.heading = previous[track.id].heading
                                else:
                                    setattr(changed_track, attribute, getattr(target, attribute))
                            alternate.tracks.append(changed_track)
                        sensitivities[attribute] = metrics(alternate, ro)
                    record[label]["positive_track_component_sensitivity"] = sensitivities
                # Recorded target future only; truncate before trace ends and
                # interpolate positions, never extrapolate or feed into control.
                valid = times[index] + ro.times <= times[-1] + 1e-9
                count = int(valid.sum())
                if count:
                    query = times[index] + ro.times[valid]
                    future = np.column_stack([np.interp(query, times, target_positions[:, axis])
                                              for axis in (0, 1)])
                    heads = np.unwrap([truth_target(r).heading for r in rows])
                    future_heads = np.interp(query, times, heads)
                    gaps = np.array([hull_separation(ro.positions[k:k+1], ro.headings[k:k+1],
                        future[k][None, None], float(future_heads[k]), margin=cc.HULL_MARGIN)[0, 0]
                        - v2.GAP_TARGET_M for k in range(count)])
                    record[label]["recorded_future_target_scoring"] = dict(
                        available_horizon_s=float(ro.times[count-1]), minimum_target_clearance_m=float(gaps.min()),
                        first_target_violation_s=float(ro.times[np.flatnonzero(gaps < 0)[0]])
                            if (gaps < 0).any() else math.inf,
                        interpolation="piecewise linear between recorded 0.5s target samples")
            logged = float(flags["checked_clearance"])
            if math.isfinite(logged) and not np.isclose(record["weak"]["clearance_m"], logged, atol=2e-6):
                raise ValueError(f"Reconstructed plan does not reproduce clearance at step {row['step']}")
            plans.append(record)
        if row["step"] in (13, 14, 17, 21, 25, 29, 34):
            future_commands = np.array([command(r) for r in rows[index:]])[None]
            for fast, label in [(False, "weak"), (True, "fast")]:
                ro = rollout(snap, act, future_commands, fast)
                for horizon in (.5, 1., 2., 4.):
                    offset = int(round(horizon / cfg.UPDATE_RATE))
                    if index + offset > len(rows):
                        continue
                    sample = offset * cc.SUBSTEPS - 1
                    endpoint = rows[index + offset - 1]["post_state"]
                    conditioned.append(dict(start_step=row["step"], model=label, horizon_s=horizon,
                        position_error_m=float(np.linalg.norm(ro.positions[sample, 0] - endpoint[:2])),
                        heading_error_deg=float(np.degrees(cc.wrap_pi(ro.headings[sample, 0]
                            - np.radians(endpoint[2]))))))

    # Follow only already-recorded checked plans through later recorded states.
    # These are counterfactual tails; commands were not necessarily executed.
    continuations = []
    for recorded in plans:
        if recorded["step"] not in (17, 20, 24, 28, 34):
            continue
        if not recorded["weak"]["passes"]:
            continue
        for offset in range(1, min(6, len(recorded["plan"]))):
            later_index = recorded["step"] - 1 + offset
            if later_index >= len(rows):
                break
            later = rows[later_index]
            snap, act = snapshot(later), actuators(later)
            remainder = recorded["plan"][offset:].copy()
            first = remainder[0].copy()
            transport = first.copy()
            if np.isnan(transport[1]):
                transport[1] = -1.
            padded = replace_first_action(remainder, transport)
            padded[0] = first
            item = dict(plan_step=recorded["step"], check_step=later["step"],
                        plan_was_retained=rows[recorded["step"]-1]["plan"] is not None,
                        expected_first_action=first, actual_first_action=command(later),
                        later_reason=later["filter"].get("why"))
            for fast, label in [(False, "weak"), (True, "fast")]:
                item[label] = metrics(snap, rollout(snap, act, padded[None], fast))
            continuations.append(item)

    groups = {}
    for label, selected in [("all", timeline), ("astern", [r for r in timeline if r["rpm"] < 0]),
                            ("non_astern", [r for r in timeline if r["rpm"] >= 0])]:
        groups[label] = {model: {key: float(np.mean([abs(r[model][key]) for r in selected]))
            if selected else None for key in ["position_error_m", "displacement_error_m",
                "u_error_mps", "u_increment_error_mps", "heading_error_deg"]}
            for model in ("weak", "fast")}
        groups[label]["count"] = len(selected)
    summary = dict(steps=len(rows), outcome=result["outcome"], changed_steps=[r["step"] for r in rows if r["changed"]],
        actual_astern_steps=[r["step"] for r in timeline if r["rpm"] < 0],
        weak_passing_fast_failing_plan_steps=[r["step"] for r in plans if r["weak"]["passes"] and not r["fast"]["passes"]],
        target_zero_speed_first_step=next((r["step"] for r in timeline if np.linalg.norm(r["target_velocity"]) < 1e-12), None),
        errors_mae=groups)
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="saved prediction audit; zero new policy episodes; no environment construction/reset/step",
        case=case, result=result, working_reproducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        checked_prediction_source_hashes=matched, models={"weak": dict(efficiency=v2.BRAKE_EFFICIENCY, delay_s=v2.BRAKE_DELAY_S),
            "fast": dict(efficiency=ship.REVERSE_THRUST_EFFICIENCY, delay_s=0.)},
        truth_usage="scoring only, including labelled fixed-plan target-only sensitivities; no action selection",
        limitations=["Neither brake scenario bounds all model error or guarantees safety.",
            "Fixed-plan scoring does not predict the outcome of a modified closed-loop filter.",
            "Recorded future target is interpolated and available only before the last pre-state.",
            "SAT inflated-rectangle gaps are filter surrogates, not the simulator's polygon contact test."],
        summary=summary, one_step=timeline, plan_checks=plans, track_errors=tracks,
        recorded_plan_tails_rechecked_at_later_observed_states=continuations,
        conditioned_on_recorded_future_commands=conditioned)
    if args.output_dir is not None:
        with (args.output_dir / "audit.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(clean(output), indent=2, allow_nan=False) + "\n")
        print("OUTPUT", args.output_dir / "audit.json")
    else:
        print("READ_ONLY: no files written")
    print(json.dumps(clean(summary), indent=2))
    print("PLAN_CHECKS", json.dumps(clean([{k: v for k, v in r.items() if k != "plan"} for r in plans])))


if __name__ == "__main__":
    main()
