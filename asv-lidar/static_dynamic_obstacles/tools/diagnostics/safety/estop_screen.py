"""Choose the emergency-stop window W on saved development traces (protocol section 4).

Saved data only; no episode. For every saved V16-equivalent development
trace (the G2/G3 replay sources) and every decision at which V16 had no
passing plan, the detector of src/safety_estop.py is evaluated from the saved
decision snapshot and pre-command actuator history, giving T* (the longest
predicted time before contact over all candidate sequences, up to 6 s).
Along V16's own trajectory nothing changes before the first firing, so the
first decision with T* <= W is where V16 with the stop would commit.

For each W the screen counts false alarms (V16-successful episodes in which
it fires) and caught contacts (V16 contact episodes in which it first fires
at least one decision before the last decision). The chosen W maximises
caught contacts with zero false alarms; ties go to the smaller W.

Usage (project root):
    python -B tools/diagnostics/safety/estop_screen.py --replay g2g3_v3 --tag screen_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import safety_estop as se  # noqa: E402
import safety_v2 as v2  # noqa: E402
import v20_replay_saved_decisions as replay  # noqa: E402

DEV = ROOT / "results/safety_dev/v20_development"
OUT = ROOT / "results/safety_dev/estop_protocol"
WINDOWS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
SUCCESS = ("rescue", "both_goal")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen_episode(trace, unchecked_steps):
    rows = []
    grid = [(r, t) for r in v2.RUDDERS for t in v2.THROTTLES] + [(r, np.nan) for r in v2.BRAKE_RUDDERS]
    for rec in replay.stream_records(trace):
        if rec["step"] not in unchecked_steps:
            continue
        dd = rec["diagnostic_decision"]
        s, a = dd["snapshot"], dd["actuators_before_decision"] or {}
        if s is None or a.get("servo") is None:
            continue
        snap = replay.snapshot(s)
        act = SimpleNamespace(servo=a["servo"], buffer=a["buffer"])
        sac = np.clip(np.asarray(rec["policy_action"], float), -1, 1)
        firsts = np.vstack([sac[None], replay.issued(rec)[None], np.asarray(grid, float)])
        t0 = time.perf_counter()
        t_star, details = se.escape_time(snap, act, firsts)
        rows.append({"step": rec["step"], "why": rec["filter"]["why"], "t_star": t_star,
                     "seconds": time.perf_counter() - t0, "best_first_command": details["best_sequence_first_command"]})
    return rows


def score(episodes):
    table = []
    for w in WINDOWS:
        false_alarm, caught, missed, fired_late, leads = [], [], [], [], []
        for ep in episodes:
            fire = next((r["step"] for r in ep["rows"] if r["t_star"] <= w), None)
            if ep["category"] in SUCCESS:
                if fire is not None:
                    false_alarm.append(ep["case"])
            elif ep["contact"]:
                if fire is None:
                    missed.append(ep["case"])
                elif fire <= ep["last_step"] - 1:
                    caught.append(ep["case"])
                    leads.append(ep["last_step"] - fire)
                else:
                    fired_late.append(ep["case"])
        table.append({"window_s": w, "false_alarms": false_alarm, "caught": caught, "missed": missed,
                      "fired_at_last_decision": fired_late,
                      "lead_decisions": sorted(leads)})
    eligible = [row for row in table if not row["false_alarms"] and row["caught"]]
    chosen = None
    if eligible:
        most = max(len(row["caught"]) for row in eligible)
        chosen = min(row["window_s"] for row in eligible if len(row["caught"]) == most)
    return table, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    out = OUT / args.tag
    if Path(args.tag).name != args.tag:
        parser.error("Use a simple tag")
    out.mkdir(parents=True, exist_ok=False)
    saved = {}
    for f in sorted((DEV / "saved_state_replay" / args.replay / "episodes").glob("*.json")):
        ep = json.loads(f.read_text())
        saved[ep["episode"]["case"]] = ep
    episodes = []
    for case, label, trace, audit_row in replay.sources({}):
        ep = saved[case]
        unchecked = {r["step"] for r in ep["rows"] if r["unchecked"]}
        last_step = max(r["step"] for r in ep["rows"])
        t0 = time.perf_counter()
        rows = screen_episode(trace, unchecked) if unchecked else []
        episodes.append({"case": case, "source": label, "category": audit_row["category"],
                         "contact": bool(audit_row["contact"]), "v16_outcome": audit_row["v16_outcome"],
                         "last_step": last_step, "unchecked_decisions": len(unchecked), "rows": rows,
                         "trace_sha256": sha(trace)})
        print(f"{case} {len(rows)} unchecked decisions {time.perf_counter() - t0:.0f}s", flush=True)
    table, chosen = score(episodes)
    result = {"schema": 1, "replay": args.replay, "windows": table, "chosen_window_s": chosen,
              "rule": "max caught contacts with zero false alarms; ties to the smaller window",
              "episodes": episodes, "script_sha256": sha(Path(__file__)),
              "module_sha256": {m: sha(ROOT / "src" / m) for m in ("safety_estop.py", "safety_v20_contingency.py")}}
    (out / "screen.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for row in table:
        print(json.dumps({k: (len(v) if isinstance(v, list) and k != "lead_decisions" else v)
                          for k, v in row.items()}))
    print("chosen window:", chosen)


if __name__ == "__main__":
    main()
