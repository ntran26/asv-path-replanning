"""Attribute G3 recheck failures to own-state or world-model change (saved data only).

For every decision k+1 at which the contingency committed at k (V16's issued
command, nominal certificate, default stop family) fails its recheck in a
merged G2/G3 replay, the recheck is repeated with one input changed at a time:

* own_only: the new own state and actuator history with the world of decision
  k (static points, polygon and tracks);
* world_only: the decision-k certificate recomputed with the world of k+1.

The failure is labelled `own_state` (only own_only fails), `world_update`
(only world_only fails), `both`, or `interaction` (neither alone fails). The
component with the smallest slack (static, boundary, target) at k+1 and the
own-ship hull-point displacement from the committed prediction are recorded,
and whether decision k+1 is one of V16's unchecked decisions. Truth is not
read. Usage (project root):
    python -B tools/diagnostics/safety/v20_recheck_attribution.py --replay g2g3_v3 --tag attribution_v1
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import safety_v20_contingency as sc  # noqa: E402
import v20_replay_saved_decisions as replay  # noqa: E402

DEV = ROOT / "results/safety_dev/v20_development"
WORLD = ("points", "tracks", "edges_a", "edges_b")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set(out, world):
    """`out` (a fresh snapshot) with the static points, polygon and tracks of `world`."""
    for k in WORLD:
        setattr(out, k, getattr(world, k))
    return out


def attribute(replay_tag):
    files = sorted((DEV / "saved_state_replay" / replay_tag / "episodes").glob("*.json"))
    sources = {case: trace for case, _, trace, _ in replay.sources({})}
    rows, counts = [], Counter()
    for f in files:
        saved = json.loads(f.read_text())
        case = saved["episode"]["case"]
        by_step = {r["step"]: r for r in saved["rows"]}
        failing = {r["step"] for r in saved["rows"] if r.get("none_survived") is False}
        if not failing:
            continue
        previous = None
        for rec in replay.stream_records(sources[case]):
            dd = rec["diagnostic_decision"]
            s, a = dd["snapshot"], dd["actuators_before_decision"] or {}
            if s is None or a.get("servo") is None:
                previous = None
                continue
            current = (rec, replay.snapshot(s), SimpleNamespace(servo=a["servo"], buffer=a["buffer"]))
            if rec["step"] in failing and previous is not None and previous[0]["step"] == rec["step"] - 1:
                prec, psnap, pact = previous
                cert = sc.ContingencyChecker(psnap, pact).certify(replay.issued(prec))
                if cert.certified:
                    shifted = sc.shift_sequence(cert.sequence)
                    ro = sc.rollout_states(psnap, pact, cert.sequence[None], *sc.WEAK)
                    k = sc.cc.SUBSTEPS                      # one decision ahead
                    dpos = ro.positions[k, 0] - np.array([current[1].x, current[1].y])
                    dpsi = (ro.headings[k, 0] - current[1].heading + math.pi) % (2 * math.pi) - math.pi
                    own_error = float(math.hypot(*dpos) + sc.INFLATED_HALF_DIAGONAL * abs(dpsi))
                    new = sc.ContingencyChecker(current[1], current[2]).certify_sequence(shifted)
                    own_only = sc.ContingencyChecker(_set(replay.snapshot(s), psnap), current[2]).certify_sequence(shifted)
                    world_only = sc.ContingencyChecker(
                        _set(replay.snapshot(prec["diagnostic_decision"]["snapshot"]), current[1]),
                        pact).certify(replay.issued(prec))
                    cause = ("both" if not own_only.certified and not world_only.certified else
                             "own_state" if not own_only.certified else
                             "world_update" if not world_only.certified else "interaction")
                    parts = {key: new.diagnostics[key] for key in ("static", "boundary", "target")
                             if key in new.diagnostics}
                    component = min(parts, key=parts.get) if parts else None
                    if new.diagnostics.get("qualifies", 1.0) < 0.5:
                        component = "not_at_rest"
                    unchecked = bool(by_step[rec["step"]]["unchecked"])
                    row = {"case": case, "step": rec["step"], "cause": cause, "component": component,
                           "own_error_m": round(own_error, 4), "recheck_slack": new.slack,
                           "committed_slack": cert.slack, "unchecked_at_recheck": unchecked,
                           "unchecked_before": bool(by_step.get(rec["step"] - 1, {}).get("unchecked", False)),
                           "traffic": bool(by_step[rec["step"]]["traffic"]),
                           "tracks_before": len(psnap.tracks), "tracks_after": len(current[1].tracks)}
                    rows.append(row)
                    counts[("cause", cause)] += 1
                    counts[("component", str(component))] += 1
                    if unchecked and not row["unchecked_before"]:
                        counts[("entering_unchecked", cause)] += 1
            previous = current
    summary = {"failures": len(rows),
               "cause": {k[1]: v for k, v in counts.items() if k[0] == "cause"},
               "component": {k[1]: v for k, v in counts.items() if k[0] == "component"},
               "entering_unchecked_cause": {k[1]: v for k, v in counts.items() if k[0] == "entering_unchecked"},
               "own_error_m_quartiles_own_state": (np.percentile([r["own_error_m"] for r in rows
                                                                  if r["cause"] == "own_state"], [0, 25, 50, 75, 100]).tolist()
                                                   if any(r["cause"] == "own_state" for r in rows) else None),
               "committed_slack_median": float(np.median([r["committed_slack"] for r in rows])) if rows else None}
    return summary, rows, {f.relative_to(ROOT).as_posix(): sha(f) for f in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        parser.error("Use a simple tag")
    out = DEV / "recheck_attribution" / args.tag
    out.mkdir(parents=True, exist_ok=False)
    summary, rows, inputs = attribute(args.replay)
    result = {"schema": 1, "replay": args.replay, "summary": summary, "rows": rows,
              "input_sha256": inputs, "script_sha256": sha(Path(__file__)),
              "module_sha256": {m: sha(ROOT / "src" / m) for m in ("safety_v20_contingency.py",)}}
    (out / "attribution.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
