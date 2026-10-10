"""Gates G2 and G3 for V20: replay the contingency certificate at saved V16 states.

Saved data only; no episode, policy call or environment construction. Each
saved full decision snapshot of a V16 run (or of a V15 run whose complete
command sequence is identical to V16's) supplies the onboard snapshot, the
pre-command actuator history, SAC's action, V16's branch and V16's issued
command. For both allowance settings ("none" and "calibrated") the tool
records:

* G2: whether V16's issued command is certified; at V16's unchecked decisions
  (`last certificate`, `no escape`) the option V20 would issue (SAC, V16,
  projection grid, committed contingency, or out of contract);
* G3: whether the contingency certified for V16's command at decision k is
  still certified when rechecked from the snapshot at k+1. This is valid along
  V16's trajectory because the vessel executed exactly that command.

Results describe V20's first decisions along V16's recorded trajectory only;
after the first predicted divergence the closed loop would differ.
Truth is not read. Usage (project root):
    python -B tools/diagnostics/safety/v20_replay_saved_decisions.py --tag TAG --part 1 --parts 2
    python -B tools/diagnostics/safety/v20_replay_saved_decisions.py --tag TAG --merge --parts 2
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src")]
from classical import common as cc  # noqa: E402
import safety_v2 as v2  # noqa: E402
import safety_v20 as v20  # noqa: E402
import safety_v20_contingency as sc  # noqa: E402
import safety_v20_tubes as tubes  # noqa: E402

CAMPAIGN = ROOT / "results/safety_dev/v10_iterations"
OUT_ROOT = ROOT / "results/safety_dev/v20_development/saved_state_replay"
AUDIT = ROOT / "results/safety_dev/v20_development/phase1_saved_cascade_audit/audit.json"
UNCHECKED = ("last certificate", "no escape", "hold back")
SETTINGS = {"none": (None, None), "calibrated": (tubes.OWN_TABLE, tubes.TARGET_TABLE)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find(tag, case, mode):
    d = CAMPAIGN / tag
    for f in sorted((d / "attempts").glob("[0-9][0-9][0-9]_result.json")):
        r = json.loads(f.read_text())
        if r["case"] == case and r["mode"] == mode:
            return d / "traces" / f"{r['attempt']:03d}_{mode}.jsonl", r
    return None, None


def commands(trace):
    out = []
    for line in trace.open(encoding="utf-8"):
        rec = json.loads(line)
        out.append((rec["rudder_command"], rec["signed_rpm_command"]))
    return out


def sources(inputs):
    """(case, label, trace) for every case with a V16-equivalent full-snapshot trace."""
    audit = json.loads(AUDIT.read_text())
    rows = {r["case"]: r for r in audit["rows"]}
    chosen = {}
    for tag in ("v16_broader40_paired", "motion_axis_probe5"):
        d = CAMPAIGN / tag
        for f in sorted((d / "attempts").glob("[0-9][0-9][0-9]_result.json")):
            r = json.loads(f.read_text())
            if r["mode"] == "v16" and r["case"] in rows and r["case"] not in chosen:
                chosen[r["case"]] = (f"{tag}:v16", d / "traces" / f"{r['attempt']:03d}_v16.jsonl")
    paired = {r["case"]: r for r in csv.DictReader(
        (CAMPAIGN / "reports/motion_axis32/paired.csv").open(encoding="utf-8")) if r["mode"] == "v16"}
    for case, row in paired.items():
        if case in chosen:
            continue
        v15, _ = find("conditional_prefix32", case, "v15")
        v16, _ = find(row["source_iteration"], case, "v16")
        if v15 is None or v16 is None:
            continue
        a, b = commands(v15), commands(v16)
        if len(a) == len(b) and all(abs(x[0] - y[0]) <= 1e-9 and abs(x[1] - y[1]) <= 1e-9 for x, y in zip(a, b)):
            chosen[case] = ("conditional_prefix32:v15_identical_to_v16", v15)
    out = []
    for case in sorted(chosen):
        label, trace = chosen[case]
        inputs[trace.relative_to(ROOT).as_posix()] = sha(trace)
        out.append((case, label, trace, rows[case]))
    return out


def snapshot(s):
    tracks = [cc.TrackView(int(t["id"]), np.asarray(t["position"], float),
                           np.asarray(t["velocity"], float), float(t["heading"])) for t in s["tracks"]]
    return cc.Snapshot(s["x"], s["y"], s["heading"], s["u"], s["v"], s["r"],
                       np.asarray(s["tangent"], float), np.asarray(s["right"], float),
                       np.asarray(s["centre"], float), s["base_heading"], s["lateral"], s["remaining"],
                       np.asarray(s["points"], float).reshape(-1, 2), tracks,
                       np.asarray(s["edges_a"], float), np.asarray(s["edges_b"], float))


def issued(rec):
    rpm = float(rec["signed_rpm_command"])
    throttle = np.nan if (rpm < 0 or bool(rec.get("brake"))) else float(np.clip((rpm - 6.0) / 6.0, -1, 1))
    return np.array([float(rec["rudder_command"]), throttle])


def stream_records(trace):
    """Yield one decision at a time with only the fields the replay needs.

    Full-snapshot lines are about 9 MB each; parsing a whole episode at once
    exhausted memory, so each line is reduced as soon as it is parsed.
    """
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            dd = rec.get("diagnostic_decision") or {}
            yield {"step": rec["step"], "filter": {"why": (rec.get("filter") or {}).get("why", "idle")},
                   "policy_action": rec["policy_action"], "rudder_command": rec["rudder_command"],
                   "signed_rpm_command": rec["signed_rpm_command"], "brake": rec.get("brake"),
                   "diagnostic_decision": {"snapshot": dd.get("snapshot"),
                                           "actuators_before_decision": dd.get("actuators_before_decision")}}
            del rec, dd


def replay_episode(case, label, trace, audit_row):
    rows = []
    committed = {name: None for name in SETTINGS}
    for rec in stream_records(trace):
        dd = rec.get("diagnostic_decision") or {}
        s = dd.get("snapshot")
        a = dd.get("actuators_before_decision") or {}
        if s is None or a.get("servo") is None:
            committed = {name: None for name in SETTINGS}
            continue
        snap = snapshot(s)
        act = SimpleNamespace(servo=a["servo"], buffer=a["buffer"])
        why = rec["filter"].get("why", "idle")
        sac = np.clip(np.asarray(rec["policy_action"], float), -1, 1)
        v16 = issued(rec)
        traffic = any(np.hypot(*(t.position - snap.position)) < v2.ENGAGE_RANGE_M for t in snap.tracks)
        row = {"case": case, "source": label, "category": audit_row["category"], "step": rec["step"],
               "why": why, "unchecked": why in UNCHECKED or why not in v20.CHECKED_BRANCHES | {"idle"},
               "traffic": traffic, "u": s["u"]}
        for name, (own, tgt) in SETTINGS.items():
            t0 = time.perf_counter()
            checker = sc.ContingencyChecker(snap, act, own, tgt)
            prev = committed[name]
            if prev is not None:
                rc = checker.certify_sequence(prev)
                row[f"{name}_survived"] = bool(rc.certified)
                row[f"{name}_survival_slack"] = rc.slack
            else:
                rc = None
            cv = checker.certify(v16)
            row[f"{name}_v16_certified"] = bool(cv.certified)
            row[f"{name}_v16_slack"] = cv.slack
            if v20._same_command(sac, v16):
                cs = cv
            else:
                cs = None
            choice = "v16_unchanged"
            if row["unchecked"]:
                options = [("sac", sac, cs), ("v16", v16, cv)]
                options += [("projection", c, None) for c in sc.projection_candidates(sac)]
                pending = [i for i, o in enumerate(options) if o[2] is None]
                batch = checker.certify_many(np.array([options[i][1] for i in pending])) if pending else []
                computed = dict(zip(pending, batch))
                found = None
                for i, (opt, cand, known) in enumerate(options):
                    cert = known if known is not None else computed[i]
                    if opt == "sac":
                        cs = cert
                    if cert.certified:
                        found = (opt, cand)
                        break
                if found is None and rc is not None and rc.certified:
                    found = ("committed", prev[0])
                if found is None:
                    choice = "out_of_contract_committed" if prev is not None else "out_of_contract_stop"
                else:
                    opt, cand = found
                    choice = opt if not v20._same_command(cand, v16) else f"{opt}_same_as_v16"
            if cs is not None:
                row[f"{name}_sac_certified"] = bool(cs.certified)
            row[f"{name}_choice"] = choice
            row[f"{name}_seconds"] = time.perf_counter() - t0
            # Commitment along V16's trajectory: only V16's own certified command.
            committed[name] = sc.shift_sequence(cv.sequence) if cv.certified else None
        rows.append(row)
    return rows


def summarise(rows, episodes):
    out = {}
    for name in SETTINGS:
        s = {}
        for group, pred in (("checked_or_idle", lambda r: not r["unchecked"]),
                            ("unchecked", lambda r: r["unchecked"])):
            sel = [r for r in rows if pred(r)]
            s[f"{group}_decisions"] = len(sel)
            s[f"{group}_v16_certified"] = sum(r[f"{name}_v16_certified"] for r in sel)
        for stratum, pred in (("traffic", lambda r: r["traffic"]), ("no_traffic", lambda r: not r["traffic"])):
            sel = [r for r in rows if f"{name}_survived" in r and pred(r)]
            n, k = len(sel), sum(r[f"{name}_survived"] for r in sel)
            s[f"survival_{stratum}"] = {"rechecks": n, "survived": k, "rate": k / n if n else None,
                                        "wilson95": wilson(k, n)}
        choice = Counter()
        for r in rows:
            if r["unchecked"] and r["category"] in ("rescue", "both_goal"):
                choice[r[f"{name}_choice"]] += 1
        s["unchecked_choice_in_successful_v16_episodes"] = dict(choice)
        choice = Counter(r[f"{name}_choice"] for r in rows if r["unchecked"])
        s["unchecked_choice_all"] = dict(choice)
        # Contact precursors.
        prec = []
        for ep in episodes:
            if not ep["contact"]:
                continue
            er = [r for r in rows if r["case"] == ep["case"]]
            if not er:
                continue
            tail_start = None
            for i in range(len(er) - 1, -1, -1):
                if not er[i]["unchecked"]:
                    break
                tail_start = i
            if tail_start is None:
                prec.append({"case": ep["case"], "unchecked_tail": 0})
                continue
            last_checked = er[tail_start - 1] if tail_start > 0 else None
            first_unchecked = er[tail_start]
            prec.append({
                "case": ep["case"], "unchecked_tail": len(er) - tail_start,
                "last_checked_step": None if last_checked is None else last_checked["step"],
                "last_checked_v16_certified": None if last_checked is None else last_checked[f"{name}_v16_certified"],
                "first_unchecked_step": first_unchecked["step"],
                "first_unchecked_choice": first_unchecked[f"{name}_choice"],
                "unchecked_with_certified_option": sum(
                    r[f"{name}_choice"] not in ("out_of_contract_committed", "out_of_contract_stop")
                    for r in er[tail_start:]),
            })
        s["contact_precursors"] = prec
        s["contact_precursors_with_certified_last_checked"] = sum(
            bool(p.get("last_checked_v16_certified")) for p in prec)
        s["contact_precursors_with_certified_option_at_first_unchecked"] = sum(
            p.get("first_unchecked_choice") not in (None, "out_of_contract_committed", "out_of_contract_stop")
            for p in prec if p.get("unchecked_tail"))
        s["contact_precursor_episodes"] = sum(1 for p in prec if p.get("unchecked_tail"))
        out[name] = s
    return out


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    out = OUT_ROOT / args.tag
    if Path(args.tag).name != args.tag:
        parser.error("Use a simple tag")
    if args.merge:
        rows, episodes, inputs = [], [], {}
        for p in range(1, args.parts + 1):
            part = json.loads((out / f"part{p}.json").read_text())
            rows += part["rows"]
            episodes += part["episodes"]
            inputs.update(part["input_sha256"])
        if (out / "replay.json").exists():
            parser.error("Merged output exists; results are never overwritten")
        summary = summarise(rows, episodes)
        identity = out / "replay_code_identity.json"
        result = {"schema": 1, "episodes": episodes, "summary": summary, "input_sha256": inputs,
                  "replay_code_identity": json.loads(identity.read_text()) if identity.exists() else None,
                  "merge_script_sha256": sha(Path(__file__)),
                  "module_sha256_on_disk_at_merge": {m: sha(ROOT / "src" / m) for m in
                                                     ("safety_v20.py", "safety_v20_contingency.py",
                                                      "safety_v20_tubes.py")}}
        (out / "replay.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        with (out / "decisions.csv").open("x", encoding="utf-8", newline="") as stream:
            fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case", "step"), k))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "contact_precursors"}
                          for k, v in summary.items()}, indent=1))
        return
    target = out / f"part{args.part}.json"
    if target.exists():
        parser.error("Existing part; results are never overwritten")
    inputs = {AUDIT.relative_to(ROOT).as_posix(): sha(AUDIT)}
    todo = sources(inputs)[args.part - 1::args.parts]
    episodes_dir = out / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    rows, episodes = [], []
    for case, label, trace, audit_row in todo:
        name = episodes_dir / (case.replace(":", "_") + ".json")
        if name.exists():
            # Completed earlier by this same tag and code; never recomputed or overwritten.
            saved = json.loads(name.read_text())
            rows += saved["rows"]
            episodes.append(saved["episode"])
            print(f"{case} reused completed episode file", flush=True)
            continue
        t0 = time.perf_counter()
        er = replay_episode(case, label, trace, audit_row)
        episode = {"case": case, "source": label, "category": audit_row["category"],
                   "contact": audit_row["contact"], "decisions": len(er)}
        with name.open("x", encoding="utf-8") as stream:
            json.dump({"episode": episode, "rows": er}, stream, default=float)
        rows += er
        episodes.append(episode)
        print(f"{case} {label} {len(er)} decisions {time.perf_counter() - t0:.0f}s", flush=True)
    target.write_text(json.dumps({"rows": rows, "episodes": episodes, "input_sha256": inputs},
                                 indent=None, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
