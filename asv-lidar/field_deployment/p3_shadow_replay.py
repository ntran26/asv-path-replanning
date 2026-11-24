"""
Bench shadow session of the Paper 3 bridge on the July 2026 logs (FIELD_TRIALS_PLAN.md section 3,
item 3, "shadow run"; FIELD_CALIBRATION_PLAN.md P-9, P-10).

Part A, in process, every July log: the log's lines go through the bridge's own per-frame path
(`udp_live_p3.FrameHandler`, with `PoseSync` fed the log's time stamps as arrival times) and the
shadow layer runs in the worker's code path (`ShadowRunner`).  Checked: every frame decided,
every observation finite and inside the observation space, one shadow decision per decision,
the shadow layer's own perception equal to the bridge's, no errors.

Part B, end to end over UDP, selected logs: `fake_vessel_replay.py` streams the log in real time
and `udp_live_p3.py` runs headless with the shadow layer in its worker process, as on the water
(--no-send: the fake vessel reads no commands).  Checked against part A: the same decisions
(observation, action, command) and the same shadow decisions; and the decision log alone
reproduces every shadow decision offline (a fresh `ShadowRunner` fed `decisions.jsonl`).
Timing (P-10) comes from part B: observation, policy, frame release to command sent, the LiDAR
line to command sent (pose-sync wait included), and the shadow layer per decision.

The July runs had no target, so the layer is mostly idle here; the encounter-rich timing is
`p3_timing.py`'s.

Raw vessel logs without the July bridge's #CONFIG line (the February 2026 logs in data/) take
their geometry from --test-case.

Usage (from field_deployment/):
    python p3_shadow_replay.py
    python p3_shadow_replay.py --udp-logs 2026-07-03/1.log --skip-inproc
    python p3_shadow_replay.py --logs "data/test_*.log" --udp-logs data/test_3.log --test-case 0
Output: p3_bench/shadow_replay/<tag>/ (summary.json, timing.csv, the runs' own logs).
"""

from __future__ import annotations

import udp_live_p3 as bridge_cli  # first: the simulator's modules, then the July bridge's pieces

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import p3_bridge as pb
import constants as cfg
import observation as obsmod
from log_parser import BluefinStreamDecoder

OUT_ROOT = os.path.join(pb.HERE, "p3_bench", "shadow_replay")
TS_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{6})\]")
DEFAULT_UDP_LOGS = ("2026-07-03/1.log", "2026-07-03/2.log", "2026-07-03/3.log")
FALLBACK_TEST_CASE: Optional[int] = None    # --test-case, for logs without a #CONFIG line


def line_time(line: str) -> Optional[float]:
    """As `fake_vessel_replay.ts_to_seconds`: lines without a time stamp are not sent."""
    m = TS_RE.match(line)
    if not m:
        return None
    hh, mm, ss, us = (int(v) for v in m.groups())
    return hh * 3600 + mm * 60 + ss + us / 1e6


def log_test_case(path: str) -> int:
    with open(path, errors="ignore") as fh:
        first = fh.readline()
    m = re.search(r"test_case=(\d+)", first)
    if m:
        return int(m.group(1))
    if FALLBACK_TEST_CASE is not None:
        return FALLBACK_TEST_CASE
    raise ValueError(f"{path}: no test_case in its #CONFIG line (give --test-case)")


def run_id_for(path: str) -> str:
    rel = os.path.relpath(path, pb.HERE).replace(os.sep, "/")
    prefix = "july-" if rel.startswith("2026-07") else "log-"
    return prefix + rel.replace("/", "-").replace(".log", "")


def args_for(path: str, out_dir: str, extra: List[str] = ()) -> argparse.Namespace:
    return bridge_cli.build_parser().parse_args(
        ["--test-case", str(log_test_case(path)), "--run-id", run_id_for(path), "--out-dir", out_dir,
         "--safety", "shadow", "--no-send", "--headless", *extra])


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def replay_inproc(path: str, out_dir: str) -> Dict:
    args = args_for(path, out_dir)
    handler, run_dir, geometry = bridge_cli.setup(args, shadow_kind="runner")
    decoder = BluefinStreamDecoder(lidar_out_beams=cfg.LIDAR_BEAMS)
    sync = bridge_cli.p2.PoseSync(decoder, enabled=True, wait_s=args.pose_wait_s)
    stats = {"lines": 0, "backlog_frames": 0, "backlog_max": 0}
    t0 = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            t = line_time(line)
            if t is None or not line.strip():
                continue
            t0 = t if t0 is None else t0
            now = t - t0
            stats["lines"] += 1
            frame = handler.take(line, now, sync)
            if frame is not None:
                handler.handle(frame, fresh=sync.last_fresh, pose_age_s=sync.last_pose_age_s,
                               pose_wait_s=sync.last_wait_s, t_release=time.perf_counter(),
                               send=lambda message: False)
    summary = bridge_cli.finish(handler, run_dir, args=args, geometry=geometry, stats=stats, pose_sync=sync)
    return check_run(run_dir, summary)


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_run(run_dir: str, summary: Dict) -> Dict:
    decisions = [r for r in read_jsonl(os.path.join(run_dir, "decisions.jsonl")) if r["type"] == "decision"]
    shadows = [r for r in read_jsonl(os.path.join(run_dir, "shadow.jsonl")) if r["type"] == "shadow"]
    space = obsmod.observation_space(cfg.N_MAX_TARGETS)
    outside = 0
    for r in decisions:
        for key, values in r["observation"].items():
            v = np.asarray(values, dtype=np.float32)
            if not (np.all(np.isfinite(v)) and space[key].contains(v)):
                outside += 1
    by_k = {r["k"]: r for r in shadows}
    problems = []
    if summary.get("errors"):
        problems.append(f"{summary['errors']} frame errors")
    if outside:
        problems.append(f"{outside} observation branches outside the space")
    if sorted(by_k) != [r["k"] for r in decisions]:
        problems.append(f"{len(shadows)} shadow decisions for {len(decisions)} decisions")
    if any(not r.get("obs_matches_bridge", True) for r in shadows):
        problems.append("shadow perception differs from the bridge's")
    whys: Dict[str, int] = {}
    for r in shadows:
        whys[r["safety"]["why"]] = whys.get(r["safety"]["why"], 0) + 1
    # The July runs had no target: every dynamic track and encounter context is a phantom.
    dynamic = sum(any(t["dynamic"] for t in r["perception"]["tracks"]) for r in decisions)
    with_context = sum(bool(r["perception"]["contexts"]) for r in decisions)
    engaged = sum(any(c.get("state") != "idle" for c in r["perception"]["contexts"].values()) for r in decisions)
    target_branch = sum(any(v != 0.0 for v in r["observation"]["target"]) for r in decisions)
    return {"run_dir": run_dir, "decisions": len(decisions), "shadow_decisions": len(shadows),
            "phantom": {"decisions_with_dynamic_track": dynamic, "decisions_with_context": with_context,
                        "decisions_engaged": engaged, "decisions_target_branch_nonzero": target_branch},
            "would_change": sum(bool(r["safety"]["changed"]) for r in shadows),
            "first_would_change": next((r["k"] for r in shadows if r["safety"]["changed"]), None),
            "would_stop_at": next((r["k"] for r in shadows if r["safety"]["committed"] is not None), None),
            "why_counts": whys, "pose_stale": summary["pose_sync"]["stale"],
            "backlog_frames": summary.get("backlog_frames"), "passed": not problems, "problems": problems}


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def replay_udp(path: str, out_dir: str, *, port: int, local_port: int, idle_exit_s: float) -> Dict:
    log = os.path.join(out_dir, run_id_for(path) + ".console.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(log, "w", encoding="utf-8") as console:
        vessel = subprocess.Popen([sys.executable, "fake_vessel_replay.py", "--log", path, "--port", str(port)],
                                  cwd=pb.HERE, stdout=console, stderr=subprocess.STDOUT)
        time.sleep(2.0)
        started = time.monotonic()
        bridge = subprocess.run([sys.executable, "udp_live_p3.py", "--test-case", str(log_test_case(path)),
                                 "--run-id", run_id_for(path), "--out-dir", out_dir, "--server-port", str(port),
                                 "--local-port", str(local_port),
                                 "--safety", "shadow", "--no-send", "--headless", "--idle-exit-s", str(idle_exit_s)],
                                cwd=pb.HERE, stdout=console, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
        try:
            vessel.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            vessel.kill()
    run_dir = os.path.join(out_dir, run_id_for(path))
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    result = check_run(run_dir, summary)
    result.update(bridge_exit=bridge.returncode, elapsed_s=elapsed, console=log)
    if bridge.returncode != 0:
        result["passed"] = False
        result["problems"].append(f"bridge exit code {bridge.returncode}")
    return result


def same_decisions(run_a: str, run_b: str) -> Dict:
    """Part A's and part B's decisions and shadow decisions, decision by decision."""
    def load(run_dir):
        d = [r for r in read_jsonl(os.path.join(run_dir, "decisions.jsonl")) if r["type"] == "decision"]
        s = [r for r in read_jsonl(os.path.join(run_dir, "shadow.jsonl")) if r["type"] == "shadow"]
        return d, s
    da, sa = load(run_a)
    db, sb = load(run_b)
    first = None
    for x, y in zip(da, db):
        if (x["observation"] != y["observation"] or x["action"] != y["action"]
                or x["command"]["message"] != y["command"]["message"] or x["sensor"] != y["sensor"]):
            first = first or f"decision {x['k']}"
    for x, y in zip(sa, sb):
        if strip_seconds(x["safety"]) != strip_seconds(y["safety"]):
            first = first or f"shadow decision {x['k']}"
    equal = first is None and len(da) == len(db) and len(sa) == len(sb)
    return {"equal": equal, "first_difference": first, "decisions": [len(da), len(db)],
            "shadow_decisions": [len(sa), len(sb)]}


def strip_seconds(value):
    if isinstance(value, dict):
        return {k: strip_seconds(v) for k, v in value.items() if not str(k).endswith("seconds")}
    if isinstance(value, list):
        return [strip_seconds(v) for v in value]
    return value


def offline_from_log(run_dir: str, built) -> Dict:
    """A fresh shadow layer fed only `decisions.jsonl` must reproduce `shadow.jsonl`."""
    decisions = [r for r in read_jsonl(os.path.join(run_dir, "decisions.jsonl")) if r["type"] == "decision"]
    shadows = {r["k"]: r for r in read_jsonl(os.path.join(run_dir, "shadow.jsonl")) if r["type"] == "shadow"}
    runner = pb.ShadowRunner(built)
    first, compared = None, 0
    for r in decisions:
        s = r["sensor"]
        sensor = pb.SensorFrame(t=r["t_frame"], ranges=np.asarray(s["ranges"], dtype=np.float64),
                                pose=tuple(s["pose"]), ego=tuple(s["ego"]), pose_stale=bool(s["pose_stale"]),
                                missed_cycles=int(s["missed_cycles"]))
        obs = {k: np.asarray(v, dtype=np.float32) for k, v in r["observation"].items()}
        decision, same, _ = runner.process(sensor, np.asarray(r["executed_prev"], dtype=np.float32),
                                           np.asarray(r["action"], dtype=np.float32),
                                           r["command"]["rudder_pct"], r["command"]["rpm"], obs=obs)
        mine = strip_seconds(pb.jsonable(pb.safety_record(decision)))
        theirs = strip_seconds(shadows[r["k"]]["safety"]) if r["k"] in shadows else None
        compared += 1
        if not same:
            first = first or f"decision {r['k']}: observation"
        elif mine != theirs:
            first = first or f"decision {r['k']}: shadow decision"
    return {"reproduced": first is None and compared == len(shadows), "compared": compared,
            "first_difference": first}


def percentiles(name: str, values) -> Dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"component": name, "n": 0}
    p50, p95, p99 = np.percentile(v, [50, 95, 99])
    return {"component": name, "n": int(len(v)), "p50_ms": 1e3 * p50, "p95_ms": 1e3 * p95, "p99_ms": 1e3 * p99,
            "max_ms": 1e3 * float(v.max())}


def timing(run_dirs: List[str]) -> List[Dict]:
    rows: Dict[str, List[float]] = {}
    for run_dir in run_dirs:
        for r in read_jsonl(os.path.join(run_dir, "decisions.jsonl")):
            if r["type"] != "decision" or r["k"] == 0:        # decision 0 also loads the map
                continue
            s = r["seconds"]
            for key in ("observation", "policy", "command", "release_to_sent", "lidar_line_to_sent",
                        "shadow_handoff"):
                rows.setdefault(key, []).append(s.get(key))
            rows.setdefault("pose_wait", []).append(r["telemetry"]["pose_wait_s"])
        for r in read_jsonl(os.path.join(run_dir, "shadow.jsonl")):
            if r["type"] == "shadow" and r["k"] > 0:
                rows.setdefault("shadow_layer", []).append(r["seconds"]["layer"])
                rows.setdefault("shadow_behind_frame", []).append(r["seconds"].get("behind_frame"))
    return [percentiles(k, v) for k, v in rows.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--logs", default="2026-07-0*/*.log", help="glob, relative to field_deployment/")
    ap.add_argument("--udp-logs", default=",".join(DEFAULT_UDP_LOGS))
    ap.add_argument("--skip-inproc", action="store_true")
    # UDP 5000, 5001 and 5050 are held by system services on the bench computer.
    ap.add_argument("--port", type=int, default=15050, help="the fake vessel's port")
    ap.add_argument("--local-port", type=int, default=15000, help="the bridge's port")
    ap.add_argument("--idle-exit-s", type=float, default=5.0)
    ap.add_argument("--test-case", type=int, default=None,
                    help="geometry for logs without a #CONFIG line (the July bridge's test_run case)")
    args = ap.parse_args()
    global FALLBACK_TEST_CASE
    FALLBACK_TEST_CASE = args.test_case
    out = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out, exist_ok=False)
    summary = {"tag": args.tag, "inproc": [], "udp": [], "git_head": pb.git_head()}
    logs = sorted(glob.glob(os.path.join(pb.HERE, args.logs)))
    if not args.skip_inproc:
        for path in logs:
            row = dict(replay_inproc(path, os.path.join(out, "inproc")), log=os.path.relpath(path, pb.HERE))
            summary["inproc"].append(row)
            print(f"A {row['log']}: {'PASS' if row['passed'] else 'FAIL'} decisions={row['decisions']} "
                  f"would_change={row['would_change']} stop_at={row['would_stop_at']} {row['why_counts']} "
                  f"phantom={row['phantom']}"
                  + ("" if row["passed"] else f" {row['problems']}"), flush=True)
    udp_dirs = []
    for rel in [p for p in args.udp_logs.split(",") if p.strip()]:
        path = os.path.join(pb.HERE, rel.strip())
        row = dict(replay_udp(path, os.path.join(out, "udp"), port=args.port, local_port=args.local_port,
                              idle_exit_s=args.idle_exit_s),
                   log=rel.strip())
        udp_dirs.append(row["run_dir"])
        inproc_dir = os.path.join(out, "inproc", run_id_for(path))
        if os.path.isdir(inproc_dir):
            row["same_as_inproc"] = same_decisions(inproc_dir, row["run_dir"])
            row["passed"] = row["passed"] and row["same_as_inproc"]["equal"]
        row["offline"] = offline_from_log(row["run_dir"], bridge_cli.test_case_scenario(log_test_case(path)))
        row["passed"] = row["passed"] and row["offline"]["reproduced"]
        summary["udp"].append(row)
        print(f"B {row['log']}: {'PASS' if row['passed'] else 'FAIL'} decisions={row['decisions']} "
              f"same_as_A={row.get('same_as_inproc', {}).get('equal')} offline={row['offline']['reproduced']} "
              f"({row['elapsed_s']:.0f} s)" + ("" if row["passed"] else f" {row['problems']}"), flush=True)
    summary["timing_udp"] = timing(udp_dirs)
    summary["passed"] = all(r["passed"] for r in summary["inproc"] + summary["udp"])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(pb.jsonable(summary), fh, indent=1)
    with open(os.path.join(out, "timing.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["component", "n", "p50_ms", "p95_ms", "p99_ms", "max_ms"])
        writer.writeheader()
        writer.writerows(summary["timing_udp"])
    for t in summary["timing_udp"]:
        print(t)
    print("PASSED" if summary["passed"] else "FAILED")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
