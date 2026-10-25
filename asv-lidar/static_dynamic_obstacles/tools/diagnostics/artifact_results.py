"""Results data for the master explainer artifact's results section (2026-10-22).

Collects from the recorded result files everything the artifact's results section draws, so
the page carries numbers read from the runs and not retyped:

  * per run (learner x seed), test set v4 outcome counts for the best validation-set
    checkpoint and for the final 3.0 M model: all episodes, decoupled and coupled families,
    encounter type (crossing split by the side the target comes from), test-set cell,
    stratum (basin / channel), target behaviour and coupled target speed;
  * per run, interquartile means (IQM) of the continuous metrics (`paper_table.py` rows);
  * the pooled table with stratified bootstrap 95 % intervals, read from
    `results/baseline_v3/paper_table.csv` (run `paper_table.py` first with every finished seed);
  * the validation-set learning curves (goal - 2 x collision, safety layer off);
  * closest-approach and hull-clearance distributions (empirical CDFs) on target episodes;
  * the ideal compute budget (`results/baseline_v3/compute_budget.md`);
  * the reward scale audit of the deployed policy, Table R7
    (`results/baseline_v3/reward_scale_audit.csv`, from `reward_scale_audit.py`).

Writes `planning/master_artifact_results.json` (chart data, counts not rates, so the page
computes rates and Wilson intervals itself) and `planning/MASTER_ARTIFACT_RESULTS.md` (the
same numbers as text, for checking and for the brief). Safety layer off throughout. The
COLREGs-VO run is not read: it is not a reported comparator (decision S18).

    python tools/diagnostics/artifact_results.py --table   # pooled table for every finished seed first
    python tools/diagnostics/artifact_results.py           # reuse the pooled table already written

`--table` runs `paper_table.py` with SAC, TQC, PPO, RecurrentPPO (every seed with a test-set run)
and LOS-DWA, which rewrites `results/baseline_v3/paper_table.*`; SAC is the paired reference.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import iqm  # noqa: E402
from paper_table import load, subset  # noqa: E402

TEST = ROOT / "results" / "test_set" / "v4"
RUNS = ROOT / "runs"
PAPER_TABLE = ROOT / "results" / "baseline_v3" / "paper_table.csv"
PAIRED = ROOT / "results" / "baseline_v3" / "paper_table_paired.csv"
BUDGET = ROOT / "results" / "baseline_v3" / "compute_budget.md"
AUDIT = ROOT / "results" / "baseline_v3" / "reward_scale_audit.csv"
AUDIT_MD = ROOT / "results" / "baseline_v3" / "reward_scale_audit.md"
OUT_JSON = ROOT / "planning" / "master_artifact_results.json"
OUT_MD = ROOT / "planning" / "MASTER_ARTIFACT_RESULTS.md"

LEARNERS = [  # id, display name, family, run-name prefix, test-set tag prefix
    ("sac", "SAC", "off-policy", "sac", "sacs"),
    ("tqc", "TQC", "off-policy", "tqc", "tqcs"),
    ("ppo", "PPO", "on-policy", "ppo", "ppos"),
    ("rppo", "RecurrentPPO", "on-policy", "recurrent_ppo", "recurrent_ppos"),
]
SEEDS = (0, 1, 2)
COMPARATORS = [  # id, display name, test-set tag or None, status
    ("los_dwa", "LOS-PID + DWA", "los_dwa_v2", "comparators v2: values unchanged from v1 (tuned on the 120 "
     "decoupled validation episodes; DWA kept its defaults); reproduces its v1 run in all 1,000 episodes"),
    ("evo", "Encounter-specific VO", "encounter_vo_v2", "comparators v2 (decision S18, A35): passing side judged on "
     "the engaged encounter; tuned on the 120 decoupled validation episodes (course-change weight 0.3 -> 0.8)"),
]
TABLE_NAME = {"los_dwa": "LOS-DWA", "evo": "Encounter-VO"}  # method names in paper_table.py
STAGES = [0, 160_000, 360_000, 640_000, 1_000_000, 1_500_000, 2_000_000]
TOTAL = 3_000_000
GRID = list(range(200_000, TOTAL + 1, 200_000))

TYPES = [  # key, label
    ("head_on", "Head-on"), ("crossing_port", "Crossing from port"),
    ("crossing_starboard", "Crossing from starboard"), ("overtaking", "Overtaking"),
    ("being_overtaken", "Being overtaken"), ("null", "Null (target, no risk)"), ("no_target", "No target"),
]
DEC_CELLS = ["basin-head_on", "basin-crossing", "basin-overtaking", "basin-being_overtaken", "basin-null",
             "channel-head_on", "channel-crossing", "channel-overtaking"]
BEHAVIOUR = {"CV": "constant velocity", "RE": "reactive (gives way by the COLREGs-VO rule from its own side)",
             "NC": "non-compliant (head-on: turns to port)"}
SPEED = {"FIX": "constant speed", "VAR": "one speed change on the approach", "": "no target"}

# Continuous metrics per run: key, column or callable, filter, kind (paper_table conventions).
CONT = [
    ("rms_cte_m", "rms_cte", None, "iqm"),
    ("completion_time_s", "completion_time_s", "success", "iqm"),
    ("path_ratio", "path_efficiency", "success", "iqm"),
    ("mean_speed_mps", "mean_speed", None, "iqm"),
    ("min_speed_mps", "min_speed", None, "iqm"),
    ("min_obstacle_clearance_m", "min_obstacle_clearance", "obstacles", "iqm"),
    ("domain_intrusion_rate", lambda d: (d.domain_intrusion_max > 0.0).astype(float), "target", "rate"),
    ("closest_approach_m", "min_target_range", "target", "iqm"),
    ("hull_clearance_m", "min_target_clearance", "target", "iqm"),
    ("rudder_saturation", "rudder_saturation_fraction", None, "rate"),
    ("rudder_rate_dps", "mean_abs_rudder_rate", None, "iqm"),
    ("control_effort", "control_effort", None, "iqm"),
]
NOTES = {  # results recorded elsewhere that the section quotes (source in each entry)
    "rppo_seed1_collapse": "RecurrentPPO seed 1 collapsed mid-training: between 1.0 and 2.4 M steps, 258 of 273 "
                           "updates stopped at the first epoch on the KL limit, the validation-set score fell to "
                           "-0.91 at 2.0 M, and it recovered after 2.4 M. No defect was found, so the seed is kept "
                           "and reported beside seeds 0 and 2, which show none of it (seed retention, skeleton S16; "
                           "planning/PAPER3_CLAIMS_AND_TABLES.md).",
    "channel_widths": None,  # filled from the test set in main()
}
CDF_GRID = {"closest_approach_m": ("min_target_range", np.round(np.arange(0.0, 8.01, 0.1), 2)),
            "hull_clearance_m": ("min_target_clearance", np.round(np.arange(0.0, 6.01, 0.1), 2))}


def r3(x):
    # Nine decimals: readers round again to three, and rounding to four first moved some values
    # by 0.001 (0.87048 -> 0.8705 -> 0.871); six still did once (0.81649995 -> 0.8165 -> 0.817).
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 9)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def counts(d):
    o = d.outcome
    return {"n": int(len(d)), "goal": int((o == "goal").sum()), "target": int((o == "collision:target").sum()),
            "obstacle": int((o == "collision:obstacle").sum()), "boundary": int((o == "collision:boundary").sum()),
            "timeout": int((o == "timeout").sum())}


def type_key(d):
    cls = d["class"].astype(str)
    side = d["crossing_side"].astype(str)
    return np.where(cls == "crossing", "crossing_" + side, cls)


def coupled_cell(d):
    # Coupled cells by encounter, pooled over the generated layouts and the reference layouts.
    return d["test_id"].astype(str).str.extract(r"-(HO|CRP|CRS|OT|BO|NT)-")[0]


def breakdown(d):
    d = d.copy()
    d["type"] = type_key(d)
    dec, cou = d[d.source == "frozen"], d[d.source == "paper2"]
    out = {"all": counts(d), "decoupled": counts(dec), "coupled": counts(cou)}
    for scope, sd in (("all", d), ("decoupled", dec), ("coupled", cou)):
        out[f"by_type_{scope}"] = {k: counts(sd[sd.type == k]) for k, _ in TYPES if (sd.type == k).any()}
    out["by_cell_decoupled"] = {c: counts(dec[dec.cell == c]) for c in DEC_CELLS}
    cc = coupled_cell(cou)
    out["by_cell_coupled"] = {c: counts(cou[cc == c]) for c in ("NT", "HO", "CRP", "CRS", "OT", "BO")}
    out["by_stratum_decoupled"] = {s: counts(dec[dec.stratum == s]) for s in ("basin", "channel")}
    shared = dec["class"].isin(["head_on", "crossing", "overtaking"])
    out["by_stratum_shared_classes"] = {s: counts(dec[shared & (dec.stratum == s)]) for s in ("basin", "channel")}
    out["by_behaviour_decoupled"] = {v: counts(dec[dec.variant == v]) for v in BEHAVIOUR}
    out["by_behaviour_head_on"] = {v: counts(dec[(dec["class"] == "head_on") & (dec.variant == v)]) for v in BEHAVIOUR}
    out["by_speed_coupled"] = {(v or "none"): counts(cou[cou.variant == v]) for v in SPEED}
    return out


def continuous(d):
    out = {}
    for scope, sd in (("all", d), ("decoupled", d[d.source == "frozen"]), ("coupled", d[d.source == "paper2"])):
        row = {}
        for key, col, flt, kind in CONT:
            s = subset(sd, flt)
            v = (col(s) if callable(col) else s[col]).astype(float).to_numpy()
            v = v[np.isfinite(v)]
            row[key] = r3(np.mean(v) if kind == "rate" else iqm(v)) if v.size else None
        out[scope] = row
    return out


def cdf(frames):
    d = pd.concat(frames) if frames else None
    out = {}
    for key, (col, grid) in CDF_GRID.items():
        if d is None:
            out[key] = None
            continue
        v = d[d.has_target][col].astype(float).to_numpy()
        v = np.sort(v[np.isfinite(v)])
        out[key] = [r3(np.searchsorted(v, g, side="right") / v.size) for g in grid]
    return out


def tag_for(prefix, seed, final=False):
    base = f"{prefix}{seed}_bl3"
    if final:
        return base + "_final" if (TEST / (base + "_final") / "episodes.csv").exists() else None
    for t in (base + "_metrics", base):
        if (TEST / t / "episodes.csv").exists() and "rms_cte" in (TEST / t / "episodes.csv").open().readline():
            return t
    return None


def validation(run_dir):
    p = run_dir / "eval_summary.json"
    if not p.exists():
        return None
    rows = [e for e in json.loads(p.read_text()) if e.get("supervisor", e.get("safety")) == "off"]
    # The scheduled checks (every 200,000 steps; 2.5 M for SAC and TQC), then one end-of-training
    # check of the final model (logged at the last step, or as a second 3.0 M entry).
    seen, checks, best, end = set(), [], None, None
    for e in rows:
        score = e["goal_rate"] - 2 * e["collision_rate"]
        is_end = e["timesteps"] in seen or e["timesteps"] > TOTAL
        if best is None or score > best[1]:
            best = (int(e["timesteps"]), score, is_end)
        rec = [int(e["timesteps"]), r3(score), r3(e["goal_rate"]), r3(e["collision_rate"]),
               r3(e.get("dev/goal")), r3(e.get("field/goal"))]
        if is_end:
            end = rec
            continue
        seen.add(e["timesteps"])
        checks.append(rec)
    return {"checks": checks, "end_check": end, "best_step": best[0] if best else None,
            "best_is_end_check": best[2] if best else None, "best_score": r3(best[1]) if best else None}


def budget():
    if not BUDGET.exists():
        return None
    text = BUDGET.read_text(encoding="utf-8")
    speeds, hours = {}, {}
    for name in ("PPO", "RecurrentPPO", "SAC", "TQC"):
        m = re.search(rf"^\| {name} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$",
                      text, re.M)
        if m:
            speeds[name] = [float(x) for x in m.groups()]
        m = re.search(rf"^\| {name} \| 0 \| [^|]+\| ([\d.]+) \| ([\d.]+) \|", text, re.M)
        if m:
            hours[name] = {"training_h": float(m.group(1)), "evaluations_h": float(m.group(2))}
    return {"source": "results/baseline_v3/compute_budget.md (ideal conditions: mains power, nothing else running)",
            "steps_per_s_by_stage": speeds, "ideal_hours_per_run": hours}


def reward_audit():
    # Table R7: how the deployed policy realises each reward term, per scope (all, decoupled,
    # coupled, goal and collision episodes). Rows: the eight dense terms, the terminal reward
    # and the episode return; the audit's own checks come from its report.
    if not AUDIT.exists():
        return None
    scopes = {}
    for r in pd.read_csv(AUDIT).itertuples():
        s = scopes.setdefault(r.scope, {"n": int(r.n), "rows": []})
        s["rows"].append({"term": r.term, "label": r.label, "weight": r3(r.weight),
                          "mean_integral": r3(r.mean_integral), "share_of_dense_abs": r3(r.share_of_dense_abs),
                          "episodes_active": r3(r.episodes_active), "mean_active_steps": r3(r.mean_active_steps),
                          "per_step_when_active": r3(r.per_step_when_active)})
    text = AUDIT_MD.read_text(encoding="utf-8") if AUDIT_MD.exists() else ""
    run = re.search(r"^Run `([^`]+)`: (.+?)\. Generated", text, re.M)
    checks = (re.findall(r"^- (.+)$", text.split("## Checks", 1)[1].split(chr(10) + "## ", 1)[0], re.M)
              if "## Checks" in text else [])
    return {"source": "results/baseline_v3/reward_scale_audit.csv (tools/diagnostics/reward_scale_audit.py)",
            "run": run.group(1) if run else None, "policy": run.group(2) if run else None,
            "checks": checks, "dense_terms": [r["term"] for r in next(iter(scopes.values()))["rows"]
                                              if r["weight"] is not None] if scopes else [],
            "scopes": scopes}


def pooled():
    if not PAPER_TABLE.exists():
        return None, None
    t = pd.read_csv(PAPER_TABLE)
    rows = [{"scope": r.scope, "metric": r.metric, "method": r.method, "kind": r.kind,
             "point": r3(r.point), "lo": r3(r.ci_lo), "hi": r3(r.ci_hi)} for r in t.itertuples()]
    head = (ROOT / "results" / "baseline_v3" / "paper_table.md").read_text(encoding="utf-8")
    runs = dict(re.findall(r"^\| ([^|]+?) \| ([a-z0-9_, ]+) \| \d+ / \d+ / \d+ \|$", head, re.M))
    paired = pd.read_csv(PAIRED).to_dict("records") if PAIRED.exists() else []
    return {"runs_per_method": runs, "rows": rows}, paired


def run_table():
    parts = []
    for lid, name, fam, rprefix, tprefix in LEARNERS:
        tags = [tag_for(tprefix, s) for s in SEEDS if (RUNS / f"{rprefix}_formulation_seed{s}_bl3" / "final_model.zip").exists()]
        tags = [t for t in tags if t]
        if tags:
            parts.append(f"{name}={','.join(tags)}")
    parts += [f"{TABLE_NAME.get(cid, n)}={t}" for cid, n, t, _ in COMPARATORS if t]
    methods = ";".join(parts)
    print("paper_table.py --methods", methods, flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "paper_table.py"), "--methods", methods],
                   check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    if "--table" in sys.argv[1:]:
        run_table()
    runs, frames_by = [], {}
    for lid, name, fam, rprefix, tprefix in LEARNERS:
        for s in SEEDS:
            run_dir = RUNS / f"{rprefix}_formulation_seed{s}_bl3"
            val = validation(run_dir)
            done = (run_dir / "final_model.zip").exists()
            best_tag, final_tag = tag_for(tprefix, s), tag_for(tprefix, s, final=True)
            rec = {"learner": lid, "seed": s, "trained": done, "validation": val,
                   "best_tag": best_tag if done else None, "final_tag": final_tag if done else None}
            if done and best_tag:
                d = load([best_tag])
                frames_by.setdefault(lid, []).append(d)
                rec["best"] = breakdown(d)
                rec["best_continuous"] = continuous(d)
                k, n = rec["best"]["all"]["goal"], rec["best"]["all"]["n"]
                rec["best_success_wilson"] = [r3(x) for x in wilson(k, n)]
            if done and final_tag:
                f = load([final_tag])
                rec["final"] = breakdown(f)
                # Same model when every episode ends the same way at the same step with the same return
                # (the best checkpoint was the end-of-training check).
                if best_tag:
                    a = d.set_index("test_id")[["outcome", "steps", "return"]].sort_index()
                    b = f.set_index("test_id")[["outcome", "steps", "return"]].sort_index()
                    rec["final_is_best"] = bool(a.index.equals(b.index) and (a.outcome == b.outcome).all()
                                                and (a.steps == b.steps).all()
                                                and np.allclose(a["return"].astype(float), b["return"].astype(float)))
            runs.append(rec)

    comps = []
    for cid, name, tag, status in COMPARATORS:
        rec = {"id": cid, "name": name, "status": status, "tag": tag}
        if tag:
            d = load([tag])
            frames_by[cid] = [d]
            rec["test"] = breakdown(d)
            rec["continuous"] = continuous(d)
            rec["success_wilson"] = [r3(x) for x in wilson(rec["test"]["all"]["goal"], rec["test"]["all"]["n"])]
        comps.append(rec)

    summary = {}
    for lid, name, *_ in LEARNERS:
        done = [r for r in runs if r["learner"] == lid and "best" in r]
        succ = [r["best"]["all"]["goal"] / r["best"]["all"]["n"] for r in done]
        dec = [r["best"]["decoupled"]["goal"] / r["best"]["decoupled"]["n"] for r in done]
        cou = [r["best"]["coupled"]["goal"] / r["best"]["coupled"]["n"] for r in done]
        summary[lid] = {"seeds_done": [r["seed"] for r in done],
                        "success": {"mean": r3(np.mean(succ)), "min": r3(min(succ)), "max": r3(max(succ))} if succ else None,
                        "decoupled": {"mean": r3(np.mean(dec)), "min": r3(min(dec)), "max": r3(max(dec))} if dec else None,
                        "coupled": {"mean": r3(np.mean(cou)), "min": r3(min(cou)), "max": r3(max(cou))} if cou else None}

    bands = {}
    for lid, *_ in LEARNERS:
        vals = [r["validation"] for r in runs if r["learner"] == lid and r["trained"] and r["validation"]]
        band = []
        for g in GRID:
            pts = [next((c[1] for c in v["checks"] if c[0] == g), None) for v in vals]
            pts = [p for p in pts if p is not None]
            if len(pts) == len(vals) and pts:
                band.append([g, r3(np.mean(pts)), r3(min(pts)), r3(max(pts))])
        bands[lid] = band

    goals = {lid: sum(r["best"]["all"]["goal"] for r in runs if r["learner"] == lid and "best" in r) for lid, *_ in LEARNERS}
    sac, tqc = summary["sac"], summary["tqc"]
    need = None
    if len(tqc["seeds_done"]) == 2 and len(sac["seeds_done"]) == 3:
        need = goals["sac"] - goals["tqc"]   # TQC seed 2 must reach more goals than this (of 1,000) to lead
    deployment = {"rule": "the better of SAC and TQC by three-seed mean success on test set v4; within it, the seed "
                          "with the best validation-set score (decision 2026-10-10)",
                  "status": "pending TQC seed 2" if len(tqc["seeds_done"]) < 3 else "decidable",
                  "deployed": None,
                  "tqc_seed2_success_needed_to_lead": need,
                  "note_threshold": "TQC seed 2 must succeed in more than this many of the 1,000 test episodes for TQC's three-seed mean to exceed SAC's"}
    if len(tqc["seeds_done"]) == 3 and len(sac["seeds_done"]) == 3:
        if goals["sac"] == goals["tqc"]:
            deployment["status"] = "tie on three-seed mean success: decision needed"
        else:
            lid = "sac" if goals["sac"] > goals["tqc"] else "tqc"
            pick = max((r for r in runs if r["learner"] == lid and "best" in r),
                       key=lambda r: r["validation"]["best_score"])
            deployment["status"] = "decided by the rule"
            deployment["deployed"] = {"learner": lid, "seed": pick["seed"], "tag": pick["best_tag"],
                                      "validation_score": pick["validation"]["best_score"]}

    ref = load([runs[0]["best_tag"]])
    w = ref[(ref.source == "frozen") & (ref.stratum == "channel")].width.astype(float)
    NOTES["channel_widths"] = (f"Decoupled channel cells: recorded nominal widths {w.min():.1f}-{w.max():.1f} m "
                               f"(median {w.median():.1f} m); {int((w < 7.60).sum())} of {len(w)} below 7.60 m.")
    pool, paired = pooled()
    meta = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "generator": "tools/diagnostics/artifact_results.py",
        "safety_layer": "off in every number here",
        "test_set": {"name": "test set v4", "episodes": 1000, "decoupled": 664, "coupled": 336,
                     "digest": "a8c40beb234ea7f9", "replaced_by_fairness_check": 67},
        "validation_set": {"episodes": 270, "decoupled": 120, "coupled": 150,
                           "score": "goal rate - 2 x collision rate", "checks": "every 200,000 steps, plus 2.5 M for SAC and TQC"},
        "curriculum_stage_starts": STAGES, "total_steps": TOTAL,
        "learners": [{"id": l, "name": n, "family": f} for l, n, f, *_ in LEARNERS],
        "types": [{"key": k, "label": v} for k, v in TYPES],
        "decoupled_cells": DEC_CELLS,
        "coupled_cells": {"NT": "no target", "HO": "head-on", "CRP": "crossing from port",
                          "CRS": "crossing from starboard", "OT": "overtaking", "BO": "being overtaken"},
        "behaviour": BEHAVIOUR, "coupled_speed": SPEED,
        "counts_keys": "n = episodes; goal; target, obstacle, boundary = collisions by type; timeout",
        "validation_check_columns": ["timesteps", "score (goal - 2 x collision)", "goal rate", "collision rate",
                                     "decoupled goal rate", "coupled goal rate"],
        "validation_band_columns": ["timesteps", "seed mean score", "seed min", "seed max"],
        "continuous_definitions": "IQM except rates; completion time and path ratio over successful episodes; "
                                  "target metrics over episodes with a target; obstacle clearance over episodes with "
                                  "a static obstacle; ship domain 3.14 m ahead, 1.57 m astern, 1.25 m abeam, entered "
                                  "by the target's centre; clearances from the hull polygon with its 0.15 m margin",
        "cdf_grid_m": {k: [float(x) for x in g] for k, (_, g) in CDF_GRID.items()},
    }
    data = {"meta": meta, "runs": runs, "comparators": comps, "learner_summary": summary,
            "validation_bands": bands,
            "cdf": {k: cdf(v) for k, v in frames_by.items()},
            "pooled_table": pool, "paired_vs_first": paired, "compute": budget(), "deployment": deployment,
            "notes": NOTES, "reward_audit": reward_audit()}
    OUT_JSON.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    OUT_MD.write_text(markdown(data), encoding="utf-8")
    print(f"-> {OUT_JSON} ({OUT_JSON.stat().st_size / 1024:.0f} kB)\n-> {OUT_MD}")
    return 0


def rate(c, key="goal"):
    return c[key] / c["n"] if c and c["n"] else float("nan")


def markdown(data):
    names = {l["id"]: l["name"] for l in data["meta"]["learners"]}
    L = ["# Paper 3 results for the master artifact (generated)", "",
         f"Generated {data['meta']['generated']} by `tools/diagnostics/artifact_results.py` from the recorded "
         "result files; chart data in `master_artifact_results.json`. Safety layer off throughout. Test set v4: "
         "1,000 episodes (664 decoupled, 336 coupled). Each run is represented by its best validation-set "
         "checkpoint; the final 3.0 M model is the selection-sensitivity check. Rates are fractions of episodes; "
         "the bracket after a run's success is its Wilson 95 % interval.", ""]
    L += ["## 1. Status", ""]
    for lid, s in data["learner_summary"].items():
        L.append(f"- {names[lid]}: seeds done {', '.join(map(str, s['seeds_done'])) or 'none'}")
    for c in data["comparators"]:
        L.append(f"- {c['name']}: {c['status']}")
    dep = data["deployment"]
    L += [f"- Deployed policy: {dep['rule']}. Status: {dep['status']}."
          + (f" Deployed: {names[dep['deployed']['learner']]} seed {dep['deployed']['seed']} "
             f"(validation score {dep['deployed']['validation_score']:+.3f})." if dep["deployed"] else "")
          + (f" TQC leads only if seed 2 succeeds in more than {dep['tqc_seed2_success_needed_to_lead']} of the "
             f"1,000 test episodes (success above {dep['tqc_seed2_success_needed_to_lead'] / 1000:.3f}; equal means at exactly that)."
             if dep["tqc_seed2_success_needed_to_lead"] else ""), ""]

    L += ["## 2. Test set v4 by run (success; collisions by type)", "",
          "| Run | Best step | Validation score | Success (Wilson 95 %) | Decoupled | Coupled | Target coll. | Obstacle coll. | Boundary coll. | Timeout | Final 3.0 M model: success (dec. / coup.) |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in data["runs"]:
        nm = f"{names[r['learner']]} seed {r['seed']}"
        if "best" not in r:
            L.append(f"| {nm} | — | — | {'training' if not r['trained'] else 'no test run'} | | | | | | | |")
            continue
        b, v, w = r["best"], r["validation"], r["best_success_wilson"]
        if r.get("final_is_best"):
            fin = "same model (the best checkpoint is the final model)"
        elif "final" in r:
            f = r["final"]
            fin = f"{rate(f['all']):.3f} ({rate(f['decoupled']):.3f} / {rate(f['coupled']):.3f})"
        else:
            fin = "not run"
        step = "final (end check)" if v["best_is_end_check"] else f"{v['best_step'] / 1e6:.1f} M"
        L.append(f"| {nm} | {step} | {v['best_score']:+.3f} | {rate(b['all']):.3f} ({w[0]:.3f}–{w[1]:.3f}) | "
                 f"{rate(b['decoupled']):.3f} | {rate(b['coupled']):.3f} | {rate(b['all'], 'target'):.3f} | "
                 f"{rate(b['all'], 'obstacle'):.3f} | {rate(b['all'], 'boundary'):.3f} | {rate(b['all'], 'timeout'):.3f} | {fin} |")
    for c in data["comparators"]:
        if "test" in c:
            t, w = c["test"], c["success_wilson"]
            L.append(f"| {c['name']} (v2) | — | — | {rate(t['all']):.3f} ({w[0]:.3f}–{w[1]:.3f}) | {rate(t['decoupled']):.3f} | "
                     f"{rate(t['coupled']):.3f} | {rate(t['all'], 'target'):.3f} | {rate(t['all'], 'obstacle'):.3f} | "
                     f"{rate(t['all'], 'boundary'):.3f} | {rate(t['all'], 'timeout'):.3f} | deterministic |")
        else:
            L.append(f"| {c['name']} | — | — | pending | | | | | | | |")
    L.append("")

    L += ["## 3. Learner means over the finished seeds (best checkpoints; range in brackets)", "",
          "| Learner | Seeds | Success | Decoupled | Coupled |", "|---|---|---|---|---|"]
    for lid, s in data["learner_summary"].items():
        if not s["success"]:
            continue
        f = lambda x: f"{x['mean']:.3f} ({x['min']:.3f}–{x['max']:.3f})"
        L.append(f"| {names[lid]} | {', '.join(map(str, s['seeds_done']))} | {f(s['success'])} | {f(s['decoupled'])} | {f(s['coupled'])} |")
    L.append("")

    def per_group(title, key, labels):
        rows = ["| Method | " + " | ".join(labels.values()) + " |", "|---|" + "---:|" * len(labels)]
        for lid, s in data["learner_summary"].items():
            done = [r for r in data["runs"] if r["learner"] == lid and "best" in r]
            if not done:
                continue
            cells = []
            for k in labels:
                vals = [rate(r["best"][key].get(k)) for r in done if r["best"][key].get(k)]
                n = done[0]["best"][key].get(k, {}).get("n", 0)
                cells.append(f"{np.mean(vals):.3f} ({min(vals):.2f}–{max(vals):.2f}) n={n}" if vals else "—")
            rows.append(f"| {names[lid]} | " + " | ".join(cells) + " |")
        for c in data["comparators"]:
            if "test" in c:
                rows.append(f"| {c['name']} (v2) | " + " | ".join(
                    f"{rate(c['test'][key].get(k)):.3f}" if c["test"][key].get(k) else "—" for k in labels) + " |")
        return [f"## {title}", "", "Success: seed mean (seed range), n = episodes per run.", ""] + rows + [""]

    tl = {k: v for k, v in TYPES}
    L += per_group("4. Success by encounter type, whole test set", "by_type_all", tl)
    L += per_group("5. Success by encounter type, decoupled family", "by_type_decoupled",
                   {k: v for k, v in tl.items() if k != "no_target"})
    L += per_group("6. Success by encounter type, coupled family", "by_type_coupled",
                   {k: v for k, v in tl.items() if k != "null"})
    L += per_group("7. Success by decoupled cell", "by_cell_decoupled", {c: c for c in DEC_CELLS})
    L += per_group("8. Success by stratum, decoupled (all classes)", "by_stratum_decoupled",
                   {"basin": "basin", "channel": "channel"})
    L += per_group("9. Success by stratum, classes both strata hold (head-on, crossing, overtaking)",
                   "by_stratum_shared_classes", {"basin": "basin", "channel": "channel"})
    L += per_group("10. Success by target behaviour, decoupled family", "by_behaviour_decoupled",
                   {k: k + ": " + v for k, v in BEHAVIOUR.items()})
    L += per_group("11. Success by target behaviour, decoupled head-on only", "by_behaviour_head_on",
                   {k: k for k in BEHAVIOUR})
    L += per_group("12. Success by target speed, coupled family", "by_speed_coupled",
                   {("none" if k == "" else k): v for k, v in SPEED.items()})

    pool = data["pooled_table"]
    if pool:
        methods = list(pool["runs_per_method"])
        L += ["## 13. Pooled table (seeds pooled; stratified bootstrap 95 % intervals)", "",
              "From `results/baseline_v3/paper_table.csv`. Runs pooled per method: " +
              "; ".join(f"{m}: {r}" for m, r in pool["runs_per_method"].items()) + ".", ""]
        for scope in ("All episodes", "Decoupled", "Coupled"):
            L += [f"### {scope}", "", "| Metric | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
            metrics = []
            for r in pool["rows"]:
                if r["scope"] == scope and r["metric"] not in metrics:
                    metrics.append(r["metric"])
            for m in metrics:
                cells = []
                for meth in methods:
                    r = next((x for x in pool["rows"] if x["scope"] == scope and x["metric"] == m and x["method"] == meth), None)
                    cells.append("—" if r is None or r["point"] is None else
                                 (f"{r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if abs(r["point"]) < 10 else
                                  f"{r['point']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]"))
                L.append(f"| {m} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## 14. Validation-set learning curves (score = goal − 2 × collision; safety off)", "",
          "| Run | Best step | Best score | Score at 1.0 / 2.0 / 3.0 M |", "|---|---:|---:|---|"]
    for r in data["runs"]:
        v = r["validation"]
        if not v or not v["checks"]:
            continue
        at = {c[0]: c[1] for c in v["checks"]}
        step = "final (end check)" if v["best_is_end_check"] else f"{v['best_step'] / 1e6:.1f} M"
        L.append(f"| {names[r['learner']]} seed {r['seed']}{'' if r['trained'] else ' (training)'} | "
                 f"{step} | {v['best_score']:+.3f} | " +
                 " / ".join(f"{at[g]:+.3f}" if g in at else "—" for g in (1_000_000, 2_000_000, 3_000_000)) + " |")
    L.append("")

    comp = data["compute"]
    if comp:
        L += ["## 15. Compute (ideal conditions, per run)", "", "| Learner | Training (h) | Evaluations (h) | Steps/s, stages 1–7 |",
              "|---|---:|---:|---|"]
        for nm, h in comp["ideal_hours_per_run"].items():
            L.append(f"| {nm} | {h['training_h']:.1f} | {h['evaluations_h']:.1f} | "
                     + ", ".join(f"{x:.0f}" for x in comp["steps_per_s_by_stage"].get(nm, [])) + " |")
        L.append("")
    L += ["## 16. Notes", ""] + [f"- {v}" for v in data["notes"].values() if v] + [""]
    aud = data.get("reward_audit")
    if aud and aud["scopes"]:
        dense = aud["dense_terms"]
        allr = {r["term"]: r for r in aud["scopes"]["All episodes"]["rows"]}
        L += ["## 17. Reward scale audit of the deployed policy (Table R7)", "",
              f"Run `{aud['run']}`: {aud['policy']}. From `{aud['source']}`. "
              "Weight = the term's largest per-step contribution; integral = mean weighted episode integral; "
              "share = mean share of the episode's absolute dense reward; active = share of episodes in which "
              "the term was ever non-zero; per step = mean magnitude per step when active.", ""]
        L += [f"- {c}" for c in aud["checks"]] + [""]
        L += ["### All episodes", "", "| Term | Weight | Mean episode integral | Share of dense abs. | Episodes active "
              "| Per step when active |", "|---|---:|---:|---:|---:|---:|"]
        f = lambda x, fmt: "—" if x is None else format(x, fmt)
        for r in aud["scopes"]["All episodes"]["rows"]:
            L.append(f"| {r['label']} | {f(r['weight'], '.2f')} | {f(r['mean_integral'], '+.2f')} | "
                     f"{f(r['share_of_dense_abs'], '.3f')} | {f(r['episodes_active'], '.3f')} | "
                     f"{f(r['per_step_when_active'], '.3f')} |")
        by = lambda key: " > ".join(sorted(dense, key=lambda t: -abs(allr[t][key])))
        L += ["", f"Order by per-step magnitude when active: {by('per_step_when_active')} "
              f"(designed, by weight: {by('weight')}).", "", "### Per step when active, by scope", "",
              "| Term | " + " | ".join(f"{k} (n = {v['n']})" for k, v in aud["scopes"].items()) + " |",
              "|---|" + "---:|" * len(aud["scopes"])]
        for t in dense:
            cells = [next(r for r in v["rows"] if r["term"] == t)["per_step_when_active"] for v in aud["scopes"].values()]
            L.append(f"| {allr[t]['label']} | " + " | ".join(f(c, '.3f') for c in cells) + " |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
