"""
Per-decision run time of the Paper 3 bridge on this computer (FIELD_CALIBRATION_PLAN.md P-10,
threshold T-14: p99 below the 0.5 s decision interval, communication included).

Test set v4 episodes, stratified by cell, are run in the simulator with the layer off (the
behaviour the paper reports), and their sensor streams are replayed through the bridge with
the shadow layer inline on the bridge's environment.  The bridge pass is timed, with nothing
else of the bench running:

* observation   the simulator's perception on the frame (the stopping table loaded);
* policy        the deployed SAC policy;
* command       command resolution;
* command_path  observation + policy + command: what delays a `$CMD` in shadow mode;
* layer         one E1 decision;
* layer_in_loop command_path + layer: what an active layer would put before the command;
* worker_lag    from a frame to its shadow decision, when the layer runs in a worker that
                receives a decision every 0.5 s (queue of the measured service times, the
                worker's own perception included).

The first decision of each episode also loads the map, and is reported apart.  UDP transport
and pose-sync waits are measured by `p3_shadow_replay.py` (part B); the T-14 check adds their p99.

Usage (from field_deployment/):
    python p3_timing.py                       # 2 episodes per cell
    python p3_timing.py --per-cell 4
Output: p3_bench/timing/<tag>/ (summary.json, timing.csv, decisions.csv).
"""

from __future__ import annotations

import p3_bridge as pb  # first: puts the simulator's modules ahead of this directory's

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

import p3_parity_replay as parity

OUT_ROOT = os.path.join(pb.HERE, "p3_bench", "timing")


def select_cases(per_cell: int) -> List[str]:
    with open(os.path.join(pb.TEST_SET_DIR, "definition.csv"), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_cell: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell"]].append(row["test_id"])
    out = []
    for cell in sorted(by_cell):
        ids = sorted(by_cell[cell])
        picks = np.linspace(0, len(ids) - 1, min(per_cell, len(ids))).round().astype(int)
        out += [ids[i] for i in sorted(set(picks))]
    return out


def timed_pass(built, seed: int, policy: pb.Policy, rows: List[Dict]) -> List[Dict]:
    parity.restart_track_ids()
    bridge = pb.Bridge(built, policy, seed=seed)
    layer = pb.SafetyLayer("shadow")
    out = []
    for row in rows:
        d = bridge.decide(row["sensor"])
        c = d.command
        sd = pb.shadow_inline(layer, bridge.env, d.action, c.rudder_pct, c.rpm)
        s = d.seconds
        command_path = s["observation"] + s["policy"] + s["command"]
        out.append({"k": d.k, "observation": s["observation"], "policy": s["policy"], "command": s["command"],
                    "command_path": command_path, "layer": sd.seconds, "layer_in_loop": command_path + sd.seconds,
                    "why": sd.why, "tracks": len(bridge.env.tracks), "contexts": len(bridge.env.encounter_contexts)})
    return out


def worker_lag(episode: List[Dict], interval: float = 0.5) -> List[float]:
    """Frame-to-decision time of a worker serving one decision per `interval` (Lindley)."""
    lags, backlog = [], 0.0
    for row in episode:
        service = row["observation"] + row["layer"]        # the worker perceives the frame again
        lags.append(backlog + service)
        backlog = max(0.0, backlog + service - interval)
    return lags


def percentile_row(name: str, values) -> Dict:
    v = np.asarray(list(values), dtype=float)
    p50, p95, p99 = np.percentile(v, [50, 95, 99])
    return {"component": name, "n": int(len(v)), "p50_ms": 1e3 * p50, "p95_ms": 1e3 * p95, "p99_ms": 1e3 * p99,
            "max_ms": 1e3 * float(v.max()), "mean_ms": 1e3 * float(v.mean()),
            "share_over_500ms": float(np.mean(v > 0.5))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--cases", default="", help="comma-separated test set v4 IDs instead of the stratified pick")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--model-path", default=pb.DEFAULT_MODEL)
    args = ap.parse_args()
    out_dir = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=False)
    policy = pb.Policy(args.model_path)
    table = pb.use_stopping_table()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or select_cases(args.per_cell)
    print(f"{len(cases)} episodes; stopping table: {table}", flush=True)
    decisions, first, lags = [], [], []
    for i, case in enumerate(cases):
        built, seed = pb.load_test_case(case)
        with pb.native_stopping():
            sim = parity.simulate(built, seed, "shadow", policy)
        rows = timed_pass(built, seed, policy, sim["rows"])
        for r in rows:
            r["case"] = case
        first.append(rows[0])
        decisions += rows[1:]
        lags += worker_lag(rows[1:])
        print(f"{i + 1}/{len(cases)} {case}: {len(rows)} decisions, layer max "
              f"{1e3 * max(r['layer'] for r in rows):.0f} ms", flush=True)
    names = ("observation", "policy", "command", "command_path", "layer", "layer_in_loop")
    timing = [percentile_row(n, [r[n] for r in decisions]) for n in names]
    timing.append(percentile_row("worker_lag", lags))
    timing += [dict(percentile_row(n, [r[n] for r in first]), component=f"first_decision_{n}")
               for n in ("observation", "layer")]
    engaged = [r for r in decisions if r["contexts"] > 0]
    if engaged:
        timing.append(dict(percentile_row("layer", [r["layer"] for r in engaged]), component="layer_with_context"))
    summary = {"tag": args.tag, "episodes": len(cases), "decisions": len(decisions), "stopping_table": table,
               "machine": {"platform": platform.platform(), "processor": platform.processor(),
                           "logical_cpus": os.cpu_count(), "python": sys.version.split()[0],
                           "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
               "timing": timing, "cases": cases, "git_head": pb.git_head()}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(pb.jsonable(summary), fh, indent=1)
    with open(os.path.join(out_dir, "timing.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(timing[0]))
        writer.writeheader()
        writer.writerows(timing)
    with open(os.path.join(out_dir, "decisions.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case", "k", *names, "why", "tracks", "contexts"])
        writer.writeheader()
        writer.writerows(decisions)
    for t in timing:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
