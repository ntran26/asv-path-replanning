"""Gate G1 for V20: calibrate horizon-indexed forecast-error allowances.

Saved data only; no episode, policy call or environment construction.

Own ship: from every saved full decision snapshot of the V14/V15/V16 runs
(which share V16's predictor and observer), predict the pose with the V20
contingency predictor driven by the commands that were actually issued at the
following decisions, and score

    s_own(t) = |p_pred(t) - p_onboard(t)| + R * |wrap(psi_pred(t) - psi_onboard(t))|

against the onboard pose estimate saved at the later decision, where R is the
inflated hull half-diagonal. For segments with astern commands the smaller
score of the two reverse-thrust models is used (the bracketing assumption).

Targets: constant-velocity forecasts from each decision's track views scored
against the later onboard view with the same source ID, in the same form.
The contract population is targets in constant-velocity scenarios (case IDs
labelled CV or FIX); the label is read offline only.

Quantiles: split-conformal, 95 % per horizon, finite-sample rank
ceil((n + 1) * 0.95), monotone in horizon. Split by scenario: the 32-case
cohort calibrates and the 40-case cohort validates, then the roles swap; the
frozen tables use pooled data. Truth (simulator own pose and target hulls) is
read only to report coverage, never to set a table value. Recorded future
commands characterise model error offline; they never select an action.

Usage (project root):
    python -B tools/diagnostics/safety/v20_calibrate_allowances.py --tag NEW_TAG
Existing tags are refused.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src")]
import safety_v20_contingency as sc  # noqa: E402

CAMPAIGN = ROOT / "results/safety_dev/v10_iterations"
OUT_ROOT = ROOT / "results/safety_dev/v20_development/allowance_calibration"
RUNS = (("persistent_prefix32", "v14"), ("conditional_prefix32", "v15"),
        ("motion_axis_probe5", "v16"), ("v16_feasible_probe9", "v16"),
        ("v16_broader40_paired", "v16"))
COHORT32 = CAMPAIGN / "reports/motion_axis32/paired.csv"
COHORT40 = CAMPAIGN / "reports/v16_broader40_paired/paired.csv"
OWN_STEPS = sc.DECISIONS                  # 0.5 .. 12 s
TARGET_STEPS = 40                         # 0.5 .. 20 s
LEVEL = 0.95
MIN_PAIRS = 50
R = sc.INFLATED_HALF_DIAGONAL


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def throttle_of(rpm: float) -> float:
    if rpm < 0.0:
        return float("nan")
    return float(np.clip((rpm - 6.0) / 6.0, -1.0, 1.0))


def contract_label(case: str) -> bool:
    return "-CV-" in case or "-FIX-" in case


def load_run(tag, mode, inputs):
    directory = CAMPAIGN / tag
    inputs[(directory / "manifest.json").relative_to(ROOT).as_posix()] = sha(directory / "manifest.json")
    for result in sorted((directory / "attempts").glob("[0-9][0-9][0-9]_result.json")):
        row = json.loads(result.read_text())
        if row["mode"] != mode:
            continue
        trace = directory / "traces" / f"{row['attempt']:03d}_{mode}.jsonl"
        inputs[trace.relative_to(ROOT).as_posix()] = sha(trace)
        records = []
        with trace.open(encoding="utf-8") as stream:
            for line in stream:
                rec = json.loads(line)
                decision = rec.get("diagnostic_decision") or {}
                snap = decision.get("snapshot")
                before = decision.get("actuators_before_decision") or {}
                truth = (rec.get("diagnostic_before") or {}).get("truth_scoring_only") or {}
                records.append(dict(
                    step=rec["step"], snap=snap, servo=before.get("servo"), buffer=before.get("buffer"),
                    rudder=float(rec["rudder_command"]), rpm=float(rec["signed_rpm_command"]),
                    truth_own=(truth.get("own") or None), truth_targets=truth.get("targets") or []))
        yield row["case"], f"{tag}:{row['attempt']:03d}", records


def own_scores(records):
    """Yield (horizon_index, onboard_score, truth_score) for one episode."""
    n = len(records)
    for k, rec in enumerate(records):
        snap = rec["snap"]
        if snap is None or rec["servo"] is None:
            continue
        horizon = min(OWN_STEPS, n - 1 - k)
        if horizon < 1:
            continue
        seq = np.array([[records[k + j]["rudder"], throttle_of(records[k + j]["rpm"])]
                        for j in range(horizon)], dtype=float)
        state = SimpleNamespace(x=snap["x"], y=snap["y"], heading=snap["heading"],
                                u=snap["u"], v=snap["v"], r=snap["r"])
        act = SimpleNamespace(servo=rec["servo"], buffer=rec["buffer"])
        models = [sc.WEAK, sc.STRONG] if np.isnan(seq[:, 1]).any() else [sc.WEAK]
        preds = [sc.rollout_states(state, act, seq[None], *m) for m in models]
        for j in range(1, horizon + 1):
            later = records[k + j]["snap"]
            if later is None:
                continue
            idx = j * 4                          # sample at t = 0.5 j (sample 0 is now)
            best_on, best_tr = math.inf, math.inf
            for p in preds:
                pos, hdg = p.positions[idx, 0], p.headings[idx, 0]
                on = math.hypot(pos[0] - later["x"], pos[1] - later["y"]) + R * abs(float(wrap(hdg - later["heading"])))
                best_on = min(best_on, on)
                # diagnostic_before of record k+j is the true state at decision k+j.
                truth = records[k + j]["truth_own"]
                if truth:
                    tr = (math.hypot(pos[0] - truth["asv_x"], pos[1] - truth["asv_y"])
                          + R * abs(float(wrap(hdg - math.radians(truth["asv_h"])))))
                    best_tr = min(best_tr, tr)
            yield j, best_on, (best_tr if math.isfinite(best_tr) else None)


def target_scores(records):
    n = len(records)
    for k, rec in enumerate(records):
        snap = rec["snap"]
        if snap is None or not snap.get("tracks"):
            continue
        for track in snap["tracks"]:
            p0 = np.asarray(track["position"], float)
            vel = np.asarray(track["velocity"], float)
            h0 = float(track["heading"])
            truth_index = None
            if rec["truth_targets"]:
                d = [math.hypot(t["x"] - p0[0], t["y"] - p0[1]) for t in rec["truth_targets"]]
                truth_index = int(np.argmin(d))
            for j in range(1, min(TARGET_STEPS, n - 1 - k) + 1):
                later = records[k + j]["snap"]
                if later is None:
                    continue
                same = [t for t in later.get("tracks", []) if t["id"] == track["id"]]
                pred = p0 + 0.5 * j * vel
                onboard = None
                if same:
                    q = np.asarray(same[0]["position"], float)
                    onboard = float(np.hypot(*(pred - q)) + R * abs(float(wrap(h0 - same[0]["heading"]))))
                truth = None
                if truth_index is not None and records[k + j]["truth_targets"]:
                    t = records[k + j]["truth_targets"][truth_index]
                    truth = float(math.hypot(pred[0] - t["x"], pred[1] - t["y"])
                                  + R * abs(float(wrap(h0 - math.radians(t["heading"])))))
                yield j, onboard, truth


def quantile(values, level=LEVEL):
    values = np.sort(np.asarray(values, float))
    n = len(values)
    if n == 0:
        return math.inf
    rank = math.ceil((n + 1) * level)
    return float(values[rank - 1]) if rank <= n else math.inf


def table(by_h, steps):
    """Monotone per-horizon quantiles; thin bins extrapolated, never decreased."""
    rows, last_ok = [], []
    for j in range(1, steps + 1):
        vals = by_h.get(j, [])
        q = quantile(vals) if len(vals) >= MIN_PAIRS else None
        rows.append([0.5 * j, q, len(vals)])
    supported = [(h, q) for h, q, n in rows if q is not None and math.isfinite(q)]
    out, running = [], 0.0
    for h, q, n in rows:
        if q is None or not math.isfinite(q):
            if len(supported) >= 2:
                (h1, q1), (h2, q2) = supported[-2], supported[-1]
                slope = max(0.0, (q2 - q1) / (h2 - h1))
                q = q2 + slope * (h - h2) if h > h2 else q2
            else:
                q = math.inf
            extrapolated = True
        else:
            extrapolated = False
        running = max(running, q)
        out.append({"horizon_s": h, "quantile": running, "pairs": n, "extrapolated": extrapolated})
    return out


def coverage(tab, by_h):
    rows = []
    for row in tab:
        j = int(round(row["horizon_s"] / 0.5))
        vals = np.asarray(by_h.get(j, []), float)
        rows.append({"horizon_s": row["horizon_s"], "pairs": int(len(vals)),
                     "coverage": float(np.mean(vals <= row["quantile"])) if len(vals) else None})
    return rows


def cohort_cases(path):
    return {r["case"] for r in csv.DictReader(path.open(encoding="utf-8")) if r["mode"] == "v16"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    out = OUT_ROOT / args.tag
    if Path(args.tag).name != args.tag or out.exists():
        parser.error("Use a new simple tag; results are never overwritten")
    inputs = {COHORT32.relative_to(ROOT).as_posix(): sha(COHORT32),
              COHORT40.relative_to(ROOT).as_posix(): sha(COHORT40)}
    split = {"A": cohort_cases(COHORT32), "B": cohort_cases(COHORT40)}
    own = {"A": defaultdict(list), "B": defaultdict(list)}
    own_truth = {"A": defaultdict(list), "B": defaultdict(list)}
    tgt = {"A": defaultdict(list), "B": defaultdict(list)}
    tgt_truth = {"A": defaultdict(list), "B": defaultdict(list)}
    tgt_truth_all = {"A": defaultdict(list), "B": defaultdict(list)}
    case_max = {"A": defaultdict(dict), "B": defaultdict(dict)}
    episodes = []
    for tag, mode in RUNS:
        for case, label, records in load_run(tag, mode, inputs):
            part = "A" if case in split["A"] else "B" if case in split["B"] else None
            if part is None:
                continue
            episodes.append({"case": case, "run": label, "split": part, "decisions": len(records)})
            for j, on, tr in own_scores(records):
                own[part][j].append(on)
                prev = case_max[part][case].get(j, 0.0)
                case_max[part][case][j] = max(prev, on)
                if tr is not None:
                    own_truth[part][j].append(tr)
            contract = contract_label(case)
            for j, on, tr in target_scores(records):
                if tr is not None:
                    tgt_truth_all[part][j].append(tr)
                if not contract:
                    continue
                if on is not None:
                    tgt[part][j].append(on)
                if tr is not None:
                    tgt_truth[part][j].append(tr)
            print(f"{label} {case} split {part}: {len(records)} decisions", flush=True)

    def pooled(d):
        merged = defaultdict(list)
        for part in ("A", "B"):
            for j, vals in d[part].items():
                merged[j].extend(vals)
        return merged

    result = {"schema": 1, "level": LEVEL, "min_pairs": MIN_PAIRS, "hull_radius_m": R,
              "split": {"A": "32-case cohort scenarios", "B": "40-case cohort scenarios"},
              "episodes": episodes, "input_sha256": inputs, "script_sha256": sha(Path(__file__)),
              "contingency_module_sha256": sha(ROOT / "src/safety_v20_contingency.py")}
    for name, data, truth, steps in (("own", own, own_truth, OWN_STEPS),
                                     ("target", tgt, tgt_truth, TARGET_STEPS)):
        section = {}
        for cal, val in (("A", "B"), ("B", "A")):
            tab = table(data[cal], steps)
            section[f"calibrate_{cal}_validate_{val}"] = {
                "table": tab, "onboard_coverage": coverage(tab, data[val]),
                "truth_coverage": coverage(tab, truth[val])}
        frozen = table(pooled(data), steps)
        section["frozen"] = frozen
        section["frozen_truth_coverage_pooled"] = coverage(frozen, pooled(truth))
        if name == "target":
            section["frozen_truth_coverage_all_targets"] = coverage(frozen, pooled(tgt_truth_all))
        result[name] = section
    case_cov = {}
    for cal, val in (("A", "B"), ("B", "A")):
        tab = {round(r["horizon_s"] / 0.5): r["quantile"]
               for r in result["own"][f"calibrate_{cal}_validate_{val}"]["table"]}
        rows = []
        for j in range(1, OWN_STEPS + 1):
            maxima = [m[j] for m in case_max[val].values() if j in m]
            rows.append({"horizon_s": 0.5 * j, "cases": len(maxima),
                         "case_max_coverage": float(np.mean(np.asarray(maxima) <= tab[j])) if maxima else None})
        case_cov[f"calibrate_{cal}_validate_{val}"] = rows
    result["own"]["case_level_max_coverage"] = case_cov
    out.mkdir(parents=True)
    (out / "calibration.json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n", encoding="utf-8")
    for name in ("own", "target"):
        print(name, [(r["horizon_s"], round(r["quantile"], 3), r["pairs"]) for r in result[name]["frozen"]])
        for key in ("calibrate_A_validate_B", "calibrate_B_validate_A"):
            cov = result[name][key]["onboard_coverage"]
            print(" ", key, "onboard", [None if c["coverage"] is None else round(c["coverage"], 3) for c in cov])
            cov = result[name][key]["truth_coverage"]
            print(" ", key, "truth  ", [None if c["coverage"] is None else round(c["coverage"], 3) for c in cov])
    print(f"Wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
