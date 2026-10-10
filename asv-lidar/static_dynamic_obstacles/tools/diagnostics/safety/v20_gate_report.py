"""Score the pre-registered V20 offline gates (planning/SAFETY_V20_PLAN.md, 9.1).

Inputs are saved development artifacts only: the G1 calibration
(`allowance_calibration/<g1 tag>/calibration.json`), the merged G2/G3 replay
(`saved_state_replay/<replay tag>/replay.json` and `decisions.csv`) and the
phase-1 audit. No episode, policy call or test-set file is read. G0 is the
pytest result passed on the command line, recorded verbatim.

Advancement to episodes requires G0 and G1 to pass, G3 survival of at least
0.90 in both strata, and a certified option at the last checked decision in at
least 11 of the 21 contact precursors. A precursor whose episode has no saved
V16-equivalent full-snapshot trace counts as not demonstrated.

Usage (project root):
    python -B tools/diagnostics/safety/v20_gate_report.py --g1 g1_v1 --replay g2g3_v3 \
        --g0 "41 passed" --tag gates_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src")]
from classical import common as cc  # noqa: E402
import safety_v20_contingency as sc  # noqa: E402

DEV = ROOT / "results/safety_dev/v20_development"
AUDIT = DEV / "phase1_saved_cascade_audit/audit.json"
COVERAGE_MIN = 0.90
SURVIVAL_MIN = 0.90
PRECURSORS_REQUIRED, PRECURSORS_TOTAL = 11, 21


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_rest_time(u0: float) -> float:
    """Latest rest time over the default stop family from a straight run at surge u0.

    Bounds the horizon bins that a certified default-family contingency can use
    from any saved state with surge at most u0 (later of the two reverse models,
    worst rudder history: servo and buffer at full deflection or centred).
    """
    edges = np.array([[-50.0, -50.0], [50.0, -50.0], [50.0, 50.0], [-50.0, 50.0]])
    snap = cc.Snapshot(0.0, 0.0, 0.0, u0, 0.0, 0.0, np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                       np.array([0.0, 0.0]), 0.0, 0.0, 40.0, np.empty((0, 2)), [], edges,
                       np.roll(edges, -1, 0))
    worst = 0.0
    for servo in (-cc.MAX_RUDDER_RAD, 0.0, cc.MAX_RUDDER_RAD):
        act = SimpleNamespace(servo=servo, buffer=[servo] * cc.DELAY_STEPS)
        checker = sc.ContingencyChecker(snap, act)
        firsts = np.array([[r, t] for r in (-1.0, 0.0, 1.0) for t in (1.0, np.nan)])
        seqs = sc.sequences_for(firsts, sc.TAILS)
        _, rest, _ = checker.evaluate(seqs)
        finite = rest[np.isfinite(rest)]
        if len(finite):
            worst = max(worst, float(finite.max()))
    return worst


def g1(calibration, max_rest_s):
    own = calibration["own"]
    out = {"threshold": COVERAGE_MIN, "max_rest_time_s_default_family": max_rest_s, "splits": {}}
    passed = True
    for split in ("calibrate_A_validate_B", "calibrate_B_validate_A"):
        rows = own[split]["onboard_coverage"]
        failing = [r for r in rows if r["coverage"] < COVERAGE_MIN]
        used_failing = [r for r in failing if r["horizon_s"] <= max_rest_s + 0.5]
        out["splits"][split] = {
            "min_coverage": min(r["coverage"] for r in rows),
            "failing_bins": [(r["horizon_s"], round(r["coverage"], 4)) for r in failing],
            "failing_bins_within_default_rest_time": [(r["horizon_s"], round(r["coverage"], 4))
                                                       for r in used_failing]}
        passed &= not used_failing
    out["pass"] = bool(passed)
    tgt = calibration["target"]["frozen_truth_coverage_pooled"]
    out["target_truth_coverage_constant_velocity_reported"] = [
        (r["horizon_s"], round(r["coverage"], 4)) for r in tgt if r["horizon_s"] in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)]
    return out


def g2_g3(replay, rows, audit_rows):
    contacts = [r for r in audit_rows if r["contact"]]
    precursors = [r for r in contacts if (r.get("consecutive_unchecked_decisions_before_end") or 0) > 0]
    covered = {e["case"] for e in replay["episodes"]}
    out = {"audit_contacts": len(contacts), "audit_contact_precursors": len(precursors),
           "precursors_without_saved_trace": sorted(r["case"] for r in precursors if r["case"] not in covered),
           "settings": {}}
    for name, s in replay["summary"].items():
        sel = [r for r in rows if r["unchecked"] == "False"]
        unc = [r for r in rows if r["unchecked"] == "True"]
        prec = {p["case"]: p for p in s["contact_precursors"] if p.get("unchecked_tail")}
        demonstrated = sum(bool(prec[c].get("last_checked_v16_certified")) for c in prec)
        survival = {k: s[f"survival_{k}"] for k in ("traffic", "no_traffic")}
        surv_pass = all(v["rate"] is not None and v["rate"] >= SURVIVAL_MIN for v in survival.values())
        unchecked_any = sum(r[f"{name}_choice"] not in ("out_of_contract_committed", "out_of_contract_stop")
                            for r in unc)
        out["settings"][name] = {
            "g2_checked_or_idle_v16_certified": [s["checked_or_idle_v16_certified"], s["checked_or_idle_decisions"]],
            "g2_unchecked_v16_certified": [s["unchecked_v16_certified"], s["unchecked_decisions"]],
            "g2_unchecked_with_certified_option": [unchecked_any, len(unc)],
            "g2_unchecked_choice_all": s["unchecked_choice_all"],
            "g2_unchecked_choice_in_successful_v16_episodes": s["unchecked_choice_in_successful_v16_episodes"],
            "g2_precursors_with_certified_option_at_first_unchecked": [
                s["contact_precursors_with_certified_option_at_first_unchecked"], len(prec)],
            "g3_survival": survival,
            "g3_survival_pass": bool(surv_pass),
            "g3_precursors_certified_at_last_checked": [demonstrated, PRECURSORS_TOTAL],
            "g3_precursor_pass": demonstrated >= PRECURSORS_REQUIRED,
            "precursor_detail": [prec[c] for c in sorted(prec)],
            "checked_rows": len(sel)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--g0", required=True, help="pytest summary line for tests/test_safety_v20_*.py")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    out_dir = DEV / "offline_gates" / args.tag
    if Path(args.tag).name != args.tag:
        parser.error("Use a simple tag")
    out_dir.mkdir(parents=True, exist_ok=False)
    cal_path = DEV / "allowance_calibration" / args.g1 / "calibration.json"
    rep_dir = DEV / "saved_state_replay" / args.replay
    calibration = json.loads(cal_path.read_text())
    replay = json.loads((rep_dir / "replay.json").read_text())
    rows = list(csv.DictReader((rep_dir / "decisions.csv").open(encoding="utf-8")))
    audit_rows = json.loads(AUDIT.read_text())["rows"]
    max_u = max(float(r["u"]) for r in rows)
    rest = latest_rest_time(max_u)
    gate1 = g1(calibration, rest)
    gate23 = g2_g3(replay, rows, audit_rows)
    g0_pass = "failed" not in args.g0 and "error" not in args.g0 and "passed" in args.g0
    advancement = {name: bool(g0_pass and gate1["pass"] and v["g3_survival_pass"] and v["g3_precursor_pass"])
                   for name, v in gate23["settings"].items()}
    result = {"schema": 1, "g0": {"pytest": args.g0, "pass": g0_pass}, "g1": gate1, "g2_g3": gate23,
              "max_saved_surge_mps": max_u, "advancement_by_allowance_setting": advancement,
              "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                               (cal_path, rep_dir / "replay.json", rep_dir / "decisions.csv", AUDIT)},
              "script_sha256": sha(Path(__file__))}
    (out_dir / "gates.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "g2_g3"}, indent=1, default=float))
    for name, v in gate23["settings"].items():
        print(name, json.dumps({k: vv for k, vv in v.items() if k != "precursor_detail"}, default=float))
    print("precursors without saved trace:", gate23["precursors_without_saved_trace"])


if __name__ == "__main__":
    main()
