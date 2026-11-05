"""
Bench parity test of the Paper 3 bridge (FIELD_CALIBRATION_PLAN.md P-9; FIELD_TRIALS_PLAN.md
section 3, item 3): the bridge's adapter against `ASVLidarEnv`, decision by decision.

Each selected test set v4 episode runs in the simulator, and its sensor stream (the scan after
the sensor model, the estimated pose, the measured ego motion, the staleness flag) is replayed
through the bridge (`p3_bridge`).  Required at every decision:

* the observation equals the environment's, all 70 values, exactly;
* the policy action, the issued command and the executed action are equal;
* the safety layer's decision is equal: branch, command, brake, stop commitment, plan, and
  every diagnostic it records (run times excepted).

The simulator passes run `stopping._profile` itself; the bridge passes run the precomputed
table the bridge loads (`p3_bridge.use_stopping_table`), so the comparison covers it too.

Two safety configurations per episode:

* `shadow`: the simulator with the layer off (the behaviour the paper reports), E1 in shadow
  beside it; on the bridge side the shadow layer runs inline on the bridge's environment, and
  again in the worker's code path (`ShadowRunner`) from the bridge's own decision inputs;
* `active`: the simulator with E1 in the loop under the stop protocol (the reported E1 runs);
  the bridge with E1 in its command path.  The simulator side is also checked against the
  recorded trace of the same episode (`results/safety_dev/testset_v4_main/runs/ts4_final_e1_*`),
  so the reference is the reported configuration.

Telemetry check (reported, not required): the stream is encoded as Bluefin telemetry lines
and decoded by `log_parser.BluefinStreamDecoder` and `p3_bridge.frame_to_sensor`.  At full
precision it tests beam order, axes and the sign of each ego component against the
simulator's own motion; at the field's resolution (0.1 m pose, 0.1 m ranges, ego from pose
differences, as in the July logs) it measures how far the observation moves.

Track identifiers come from one process-wide counter; it is restarted before each pass so the
passes number their tracks alike, as separate processes would.

Usage (from field_deployment/):
    python p3_parity_replay.py
    python p3_parity_replay.py --cases TS4:BAS-CR-CV-013,TS4:FS-NT-023 --modes shadow
Output: p3_bench/parity/<tag>/ (summary.json, cases.csv, timing.csv).
"""

from __future__ import annotations

import p3_bridge as pb  # first: puts the simulator's modules ahead of this directory's

import argparse
import contextlib
import copy
import io
import csv
import glob
import gzip
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import constants as cfg
import tracking
from env import ASVLidarEnv
from log_parser import BluefinStreamDecoder

sys.path.insert(1, os.path.join(pb.SIM_ROOT, "tools", "diagnostics", "safety"))
import estop_protocol  # noqa: E402

E1_RUNS = os.path.join(pb.SIM_ROOT, "results", "safety_dev", "testset_v4_main", "runs")
OUT_ROOT = os.path.join(pb.HERE, "p3_bench", "parity")

# Stops, target contacts, interventions, every encounter class, the basin layouts L1-L3,
# a channel, a null encounter and a no-target leg (outcomes from the recorded E1 runs).
DEFAULT_CASES = (
    "TS4:BAS-CR-CV-013", "TS4:BAS-CR-CV-037", "TS4:CH-HO-CV-088", "TS4:P2-L1-CRP-FIX-05",
    "TS4:BAS-HO-NC-071", "TS4:BAS-BO-CV-061", "TS4:BAS-HO-RE-081", "TS4:BAS-CR-CV-051",
    "TS4:P2-L1-CRS-VAR-08", "TS4:P2-L2-CRP-VAR-14", "TS4:P2-L3-HO-FIX-05", "TS4:BAS-OT-RE-022",
    "TS4:FS-NT-023", "TS4:BAS-NU-CV-004", "TS4:BAS-BO-RE-006", "TS4:CH-HO-NC-005",
)


def restart_track_ids() -> None:
    tracking._next_track_id = itertools.count(1)


def comparable(details: Dict) -> Dict:
    """A decision's diagnostics as JSON, run times and the bulky keys left out."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if not str(k).endswith("seconds")}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return strip(pb.jsonable(pb.without_keys(details, pb.COMPACT_DROP)))


def first_difference(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: present on one side only"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: lengths {len(a)} and {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


# ---------------------------------------------------------------------------
# Simulator reference
# ---------------------------------------------------------------------------
def sensor_from_env(env: ASVLidarEnv) -> pb.SensorFrame:
    """The sensor stream as the environment perceived it: what the bridge receives."""
    return pb.SensorFrame(t=float(env.elapsed_time), ranges=np.asarray(env.raw_ranges, dtype=np.float64).copy(),
                          pose=tuple(env.estimated_pose()), ego=tuple(env._measured_ego()),
                          pose_stale=bool(env.pose_stale), missed_cycles=0)


def simulate(built, seed: int, mode: str, policy: pb.Policy, pose_stale_prob: float = 0.0) -> Dict:
    """`pose_stale_prob` > 0 turns on the environment's own staleness model (off in every
    reported run), to exercise the bridge's stale-pose path."""
    restart_track_ids()
    cfg.SAFETY_VERSION = 21 if mode == "active" else 1
    env = ASVLidarEnv(render_mode=None, emergency_stop=mode == "active", low_speed_start_frac=0.0,
                      pose_stale_prob=pose_stale_prob)
    obs, _ = env.reset(seed=seed, options={"generated": copy.deepcopy(built)})
    if mode == "active":
        env._safety_v2 = pb.E1_CLASS(**pb.E1_OPTIONS)
        estop_protocol.install(env)
    shadow = pb.SafetyLayer("shadow") if mode == "shadow" else None
    actor = pb.tf.EpisodeActor(policy.model)
    rows, outcome = [], "timeout"
    while True:
        row = {"sensor": sensor_from_env(env), "executed_prev": env._executed_action.copy(),
               "obs": {k: v.copy() for k, v in obs.items()},
               "truth": (env.asv_x, env.asv_y, env.asv_h, env.u_body, env.v_body, env.asv_w)}
        action = np.asarray(actor(obs), dtype=np.float32).reshape(2)
        row["action"] = action.copy()
        decision = shadow.decide(env, action) if shadow is not None else None
        result = env.step(action)
        if shadow is not None:
            shadow.issued(env, env.rudder, env.rpm)
        else:
            result = estop_protocol.after_step(env, result, len(rows) + 1)
            filt = env._safety_v2
            decision = pb.SafetyDecision(
                out=np.full(2, np.nan, dtype=np.float32), changed=bool(env.safety_v2_changed),
                brake=bool(env._v2_brake), why=str(filt.last.get("why", "idle")),
                mode=str(filt.last.get("mode", "idle")), committed=filt.estop_committed_decision,
                seconds=0.0, details=copy.deepcopy(filt.last),
                plan=None if filt.plan is None else np.array(filt.plan))
        row.update(decision=decision, rudder_pct=float(env.rudder), rpm=float(env.rpm),
                   executed=env._executed_action.copy())
        rows.append(row)
        obs, _, term, trunc, info = result
        if term or trunc:
            if mode == "active" and env.estop_protocol["outcome"]:
                outcome = env.estop_protocol["outcome"]
            elif info.get("collided"):
                outcome = f"collision:{info['collision_kind']}"
            elif info.get("reached_goal"):
                outcome = "goal"
            break
    env.close()
    return {"rows": rows, "outcome": outcome}


# ---------------------------------------------------------------------------
# Bridge replay
# ---------------------------------------------------------------------------
def replay(built, seed: int, mode: str, policy: pb.Policy, rows: List[Dict]) -> Dict:
    """The bridge on the simulator's sensor stream.  It reads nothing else of the run."""
    restart_track_ids()
    layer = pb.SafetyLayer("active") if mode == "active" else None
    bridge = pb.Bridge(built, policy, seed=seed, layer=layer)
    inline = pb.SafetyLayer("shadow") if mode == "shadow" else None
    out = []
    for row in rows:
        decision = bridge.decide(row["sensor"])
        shadow = None
        if inline is not None:
            c = decision.command
            shadow = pb.shadow_inline(inline, bridge.env, decision.action, c.rudder_pct, c.rpm)
        out.append({"decision": decision, "shadow": shadow})
    return {"rows": out}


def replay_worker_path(built, seed: int, bridged: List[Dict]) -> List[Dict]:
    """The shadow worker's code path, fed the bridge's own decision inputs."""
    restart_track_ids()
    runner = pb.ShadowRunner(built, seed=seed)
    out = []
    for item in bridged:
        d = item["decision"]
        decision, same, _ = runner.process(d.sensor, d.executed_prev, d.action, d.command.rudder_pct,
                                           d.command.rpm, obs=d.obs)
        out.append({"decision": decision, "obs_matches_bridge": same})
    return out


def compare_decision(ref: pb.SafetyDecision, got: pb.SafetyDecision, *, compare_out: bool) -> Optional[str]:
    for name in ("why", "mode", "changed", "brake", "committed"):
        if getattr(ref, name) != getattr(got, name):
            return f"{name}: {getattr(ref, name)!r} vs {getattr(got, name)!r}"
    if compare_out and not np.array_equal(ref.out, got.out):
        return f"out: {ref.out.tolist()} vs {got.out.tolist()}"
    if (ref.plan is None) != (got.plan is None) or (ref.plan is not None and not np.array_equal(
            ref.plan, got.plan, equal_nan=True)):
        return "plan differs"
    return first_difference(comparable(ref.details), comparable(got.details), "details")


def compare(case: str, mode: str, sim: Dict, bridged: Dict, worker: Optional[List[Dict]]) -> Dict:
    first = {}
    counts = {"decisions": len(sim["rows"]), "obs": 0, "action": 0, "command": 0, "layer": 0,
              "layer_worker": 0, "worker_obs": 0, "layer_changed": 0, "layer_stop_at": None}
    for k, (ref, got) in enumerate(zip(sim["rows"], bridged["rows"])):
        d = got["decision"]
        checks = {
            "obs": None if pb.obs_equal(ref["obs"], d.obs) else "observation differs",
            "action": None if np.array_equal(ref["action"], d.action) else f"{ref['action']} vs {d.action}",
            "command": None if (ref["rudder_pct"] == d.command.rudder_pct and ref["rpm"] == d.command.rpm
                                and np.array_equal(ref["executed"], d.command.executed))
            else f"rudder {ref['rudder_pct']} vs {d.command.rudder_pct}, rpm {ref['rpm']} vs {d.command.rpm}",
        }
        theirs = got["shadow"] if mode == "shadow" else d.safety
        checks["layer"] = compare_decision(ref["decision"], theirs, compare_out=mode == "shadow")
        if worker is not None:
            checks["layer_worker"] = compare_decision(ref["decision"], worker[k]["decision"], compare_out=True)
            checks["worker_obs"] = None if worker[k]["obs_matches_bridge"] else "worker observation differs"
        for name, problem in checks.items():
            if problem:
                counts[name] += 1
                first.setdefault(name, f"decision {k}: {problem}")
        counts["layer_changed"] += int(ref["decision"].changed)
        if ref["decision"].committed is not None and counts["layer_stop_at"] is None:
            counts["layer_stop_at"] = k
    if len(sim["rows"]) != len(bridged["rows"]):
        first["length"] = f"{len(sim['rows'])} vs {len(bridged['rows'])}"
    passed = not first
    return {"case": case, "mode": mode, "outcome": sim["outcome"], "passed": passed, **counts,
            "first_mismatch": first}


# ---------------------------------------------------------------------------
# The recorded E1 traces
# ---------------------------------------------------------------------------
def recorded_trace(case: str) -> Optional[List[Dict]]:
    for attempt in glob.glob(os.path.join(E1_RUNS, "ts4_final_e1_c*", "attempts", "[0-9][0-9][0-9].json")):
        identity = json.load(open(attempt))
        if identity.get("case") == case and identity.get("mode") == "v21_e1_estop_bt":
            path = os.path.join(os.path.dirname(os.path.dirname(attempt)), "traces",
                                f"{identity['attempt']:03d}_v21_e1_estop_bt.jsonl.gz")
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return [json.loads(line) for line in fh]
    return None


def compare_trace(sim: Dict, trace: List[Dict]) -> Dict:
    first, bad = None, 0
    for k, (row, rec) in enumerate(zip(sim["rows"], trace)):
        problems = []
        if pb.obs_record(row["obs"]) != rec["observation"]:
            problems.append("observation")
        if row["action"].tolist() != rec["policy_action"]:
            problems.append("action")
        if row["rudder_pct"] / 100.0 != rec["rudder_command"] or row["rpm"] != rec["signed_rpm_command"]:
            problems.append("command")
        if row["decision"].changed != rec["changed"] or row["decision"].brake != rec["brake"]:
            problems.append("changed/brake")
        d = first_difference(comparable(row["decision"].details), comparable(rec["filter"]), "filter")
        if d:
            problems.append(d)
        if problems:
            bad += 1
            first = first or f"decision {k}: {', '.join(problems)}"
    length = len(sim["rows"]) == len(trace)
    return {"trace_decisions": len(trace), "trace_mismatches": bad, "trace_first": first,
            "trace_length_equal": length, "trace_passed": bad == 0 and length}


# ---------------------------------------------------------------------------
# Telemetry check
# ---------------------------------------------------------------------------
def _ts(seconds: float) -> str:
    whole = int(seconds)
    micros = int(round((seconds - whole) * 1e6))
    return f"{whole // 3600:02d}:{(whole // 60) % 60:02d}:{whole % 60:02d}.{micros:06d}"


def telemetry_lines(t: float, pose, ranges, *, decimals: int) -> List[str]:
    """One cycle as the vessel sends it: the pose line, then the LiDAR line.

    The inverse of `log_parser`: the pose line holds (-y, x, -heading); the scan is in
    decimetres, counter-clockwise, 0 for no return.
    """
    x, y, heading = pose
    p = f"[{_ts(t)}]{-y:+.{decimals}f},{x:+.{decimals}f},{-heading:+.{decimals}f}"
    r = np.asarray(ranges, dtype=np.float64)
    dm = np.where(r >= cfg.LIDAR_RANGE - 1e-9, 0, np.round(r * 10.0)).astype(int)[::-1]
    return [p, f"[{_ts(t + 0.003)}][{', '.join(str(v) for v in dm)}]"]


def decode(rows: List[Dict], *, decimals: int, truth_pose: bool, pose_point: str = "origin") -> List[pb.SensorFrame]:
    """`pose_point="sensor"` sends the LiDAR's pose (the true one) and decodes it as such."""
    from ship import LIDAR_OFFSET_M
    decoder = BluefinStreamDecoder(lidar_out_beams=cfg.LIDAR_BEAMS)
    frames = []
    for k, row in enumerate(rows):
        pose = row["truth"][:3] if truth_pose else row["sensor"].pose
        if pose_point == "sensor":
            x, y, h = pose
            pose = (x + LIDAR_OFFSET_M * np.sin(np.radians(h)), y + LIDAR_OFFSET_M * np.cos(np.radians(h)), h)
        frame = None
        with contextlib.redirect_stdout(io.StringIO()):     # the decoder prints each new top speed
            for line in telemetry_lines(36000.0 + 0.5 * k, pose, row["sensor"].ranges, decimals=decimals):
                frame = decoder.feed(line) or frame
        frames.append(pb.frame_to_sensor(frame, pb.PoseTransform(), lidar_index0_deg=0.0, pose_point=pose_point))
    return frames


def telemetry_check(case: str, built, seed: int, rows: List[Dict]) -> Dict:
    exact = decode(rows, decimals=6, truth_pose=True)
    range_err = max(float(np.max(np.abs(f.ranges - r["sensor"].ranges))) for f, r in zip(exact, rows))
    pose_err = max(float(np.max(np.abs(np.subtract(f.pose[:2], r["truth"][:2])))) for f, r in zip(exact, rows))
    head_err = max(abs((f.pose[2] - r["truth"][2] + 180.0) % 360.0 - 180.0) for f, r in zip(exact, rows))
    # Pose differences over a step against the simulator's motion at its end: slope ~ +1 per
    # component when the signs agree (decoded from the second frame on).
    fits = {}
    for i, name in enumerate(("u", "v", "r")):
        got = np.array([f.ego[i] for f in exact[2:]])
        true = np.array([r["truth"][3 + i] for r in rows[2:]])
        spread = float(np.std(true))
        slope = float(np.polyfit(true, got, 1)[0]) if spread > 1e-6 and len(true) > 3 else float("nan")
        fits[name] = {"slope": slope, "mean_abs_error": float(np.mean(np.abs(got - true))) if len(true) else None,
                      "true_spread": spread}
    # A pose that belongs to the LiDAR, decoded as such.
    sensor = decode(rows, decimals=6, truth_pose=True, pose_point="sensor")
    sensor_pose_err = max(float(np.max(np.abs(np.subtract(f.pose[:2], r["truth"][:2])))) for f, r in zip(sensor, rows))
    got = np.array([f.ego[1] for f in sensor[2:]])
    true = np.array([r["truth"][4] for r in rows[2:]])
    sensor_v = {"slope": float(np.polyfit(true, got, 1)[0]) if len(true) > 3 and np.std(true) > 1e-6 else float("nan"),
                "mean_abs_error": float(np.mean(np.abs(got - true))) if len(true) else None}
    # The field's resolution, open loop on the simulator's executed actions; from the third
    # frame, once the decoder has a pose difference.
    field = decode(rows, decimals=1, truth_pose=False)
    restart_track_ids()
    env = pb.LiveEnv()
    diffs: Dict[str, List[float]] = {}
    for k, (frame, row) in enumerate(zip(field, rows)):
        obs = env.begin(built, frame, seed=seed) if k == 0 else env.observe(frame, row["executed_prev"])
        if k >= 2:
            for key in obs:
                diffs.setdefault(key, []).append(float(np.max(np.abs(obs[key] - row["obs"][key]))))
    env.close()
    return {"case": case, "range_max_error_m": range_err, "pose_max_error_m": pose_err,
            "heading_max_error_deg": head_err, "ego_fit": fits,
            "sensor_point": {"pose_max_error_m": sensor_pose_err, "v_fit": sensor_v},
            "field_resolution_obs_max_abs_diff": {k: max(v) for k, v in diffs.items()},
            "field_resolution_obs_mean_abs_diff": {k: float(np.mean(v)) for k, v in diffs.items()}}


# ---------------------------------------------------------------------------
def percentile_row(name: str, values) -> Dict:
    v = np.asarray(list(values), dtype=float)
    if not len(v):
        return {"component": name, "n": 0}
    p50, p95, p99 = np.percentile(v, [50, 95, 99])
    return {"component": name, "n": int(len(v)), "p50_ms": 1e3 * p50, "p95_ms": 1e3 * p95,
            "p99_ms": 1e3 * p99, "max_ms": 1e3 * float(v.max()), "mean_ms": 1e3 * float(v.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--modes", default="shadow,active")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--model-path", default=pb.DEFAULT_MODEL)
    ap.add_argument("--no-telemetry-check", action="store_true")
    ap.add_argument("--pose-stale-prob", type=float, default=0.0,
                    help="the environment's staleness model in the simulator passes (0 in the reported runs)")
    ap.add_argument("--native-stopping", action="store_true",
                    help="bridge passes on stopping._profile itself, not the precomputed table")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=False)
    policy = pb.Policy(args.model_path)
    table = pb.use_stopping_table() if not args.native_stopping else {"installed": False, "reason": "--native-stopping"}
    print(f"stopping table: {table}")
    results, telemetry, timing = [], [], {"observation": [], "policy": [], "layer": [], "layer_active": []}
    started = time.perf_counter()
    for case in [c.strip() for c in args.cases.split(",") if c.strip()]:
        built, seed = pb.load_test_case(case)
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            t0 = time.perf_counter()
            with pb.native_stopping():
                sim = simulate(built, seed, mode, policy, pose_stale_prob=args.pose_stale_prob)
            bridged = replay(built, seed, mode, policy, sim["rows"])
            worker = replay_worker_path(built, seed, bridged["rows"]) if mode == "shadow" else None
            row = compare(case, mode, sim, bridged, worker)
            row["stale_frames"] = sum(r["sensor"].pose_stale for r in sim["rows"])
            if mode == "active" and not args.pose_stale_prob:
                trace = recorded_trace(case)
                row.update(compare_trace(sim, trace) if trace is not None else {"trace_passed": None})
            for item in bridged["rows"]:
                d = item["decision"]
                timing["observation"].append(d.seconds["observation"])
                timing["policy"].append(d.seconds["policy"])
                if mode == "active":
                    timing["layer_active"].append(d.seconds["layer_active"])
                if item["shadow"] is not None:
                    timing["layer"].append(item["shadow"].seconds)
            row["elapsed_s"] = time.perf_counter() - t0
            results.append(row)
            print(f"{case} {mode}: {'PASS' if row['passed'] else 'FAIL'} decisions={row['decisions']} "
                  f"outcome={row['outcome']} layer_changed={row['layer_changed']} stop_at={row['layer_stop_at']} "
                  f"trace={row.get('trace_passed', '-')} ({row['elapsed_s']:.0f} s)"
                  + ("" if row["passed"] else f" {row['first_mismatch']}"), flush=True)
            if mode == "shadow" and not args.no_telemetry_check:
                telemetry.append(telemetry_check(case, built, seed, sim["rows"]))
    timing_rows = [percentile_row(k, v) for k, v in timing.items()]
    summary = {"tag": args.tag, "model_sha256": policy.sha256, "stopping_table": table, "cases": len({r["case"] for r in results}),
               "runs": len(results), "passed": sum(bool(r["passed"]) for r in results),
               "trace_checked": sum(r.get("trace_passed") is not None for r in results),
               "trace_passed": sum(bool(r.get("trace_passed")) for r in results),
               "decisions": sum(r["decisions"] for r in results), "elapsed_s": time.perf_counter() - started,
               "results": results, "telemetry": telemetry, "timing": timing_rows,
               "git_head": pb.git_head(), "machine": {"platform": sys.platform, "cpu": os.cpu_count(),
                                                      "processor": __import__("platform").processor()}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(pb.jsonable(summary), fh, indent=1, allow_nan=False)
    with open(os.path.join(out_dir, "cases.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = ["case", "mode", "outcome", "passed", "decisions", "obs", "action", "command", "layer",
                "layer_worker", "worker_obs", "layer_changed", "layer_stop_at", "trace_passed", "trace_mismatches",
                "elapsed_s", "first_mismatch"]
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(dict(r, first_mismatch=json.dumps(r["first_mismatch"])))
    with open(os.path.join(out_dir, "timing.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["component", "n", "p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms"])
        writer.writeheader()
        writer.writerows(timing_rows)
    print(json.dumps({k: summary[k] for k in ("runs", "passed", "trace_checked", "trace_passed", "decisions")}))
    for t in timing_rows:
        print(t)
    return 0 if summary["passed"] == summary["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
