"""The v4.3 SAC pair re-scored on new seeds (2026-10-05; BASELINE_V4_PLAN.md section 5e).

Declared before it ran.  The gate (field >= +5 points, frozen-like >= -2) failed
narrowly on the training-time evaluation seeds (field +3.8).  The v4.3 arm's
selected checkpoint (3.4 M) was the best of six evaluations on those seeds, so
they flatter it.  This scores the same checkpoints on the same 330 development
episodes (formulation v4.2: frozen-like 120 + field 210, in evaluation order)
under three new episode-seed sets, and decides on those seeds only:

* primary: the selected checkpoints, v4.3 arm `best_model.zip` (3.4 M) against
  v3 arm `best_model.zip` (its 3.0 M start);
* secondary: both arms' final 3.5 M models.

The difference v4.3 minus v3 comes with a 95 % interval from a bootstrap over
development scenarios (each resampled scenario keeps its three seeds).

    python tools/diagnostics/v4_gates/pair_reseed.py --processes 6

Writes `results/v4_gates/pair_reseed_episodes.csv` and `pair_reseed_summary.txt`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(ROOT / "tools" / "diagnostics" / "feasibility"),
                str(Path(__file__).parent)]

import numpy as np
import pandas as pd

import curriculum
import train_formulation as tf

SEED_SETS = (1_100_000, 1_200_000, 1_300_000)
MODELS = {
    ("v4.3", "selected"): ROOT / "runs" / "sac_formulation_seed0_bl3_v43pair" / "best_model.zip",
    ("v3", "selected"): ROOT / "runs" / "sac_formulation_seed0_bl3_v3pair" / "best_model.zip",
    ("v4.3", "final"): ROOT / "runs" / "sac_formulation_seed0_bl3_v43pair" / "final_model.zip",
    ("v3", "final"): ROOT / "runs" / "sac_formulation_seed0_bl3_v3pair" / "final_model.zip",
}
OUT = ROOT / "results" / "v4_gates"
FIELD_GATE, FROZEN_GATE = 0.05, -0.02


def development_episodes():
    """Formulation v4.2's development episodes in evaluation order: the cached v3-era
    sets (`oracle_sets.deveval_items`) with the recorded replacements applied in place --
    the same list `dev_set_v4.development_sets` builds, without rebuilding it."""
    import dev_set_v4
    import oracle_sets
    items = oracle_sets.deveval_items()
    rec = json.loads((ROOT / "configs" / "dev_set_v4_replacements.json").read_text(encoding="utf-8"))
    out = [{"position": m["position"], "built": b, "set": "dev" if m["set"] == "dev_v2" else "field"} for _, b, _, m in items]
    for r in rec["replacements"]:
        out[int(r["position"])]["built"] = dev_set_v4.from_recipe(r["recipe"])
    for e in out:
        b = e["built"]
        e["case_id"] = getattr(b, "case_id", "")
        e["crossing"] = str(b.encounter_class) == "crossing"
        e["no_target"] = str(b.encounter_class) == "no_target"
    return out


def run(processes: int):
    from common import run_pool
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    eps = development_episodes()
    print(f"[reseed] {len(eps)} development episodes x {len(SEED_SETS)} seed sets x {len(MODELS)} models", flush=True)
    rows, t0 = [], time.time()
    for (arm, ckpt), path in MODELS.items():
        jobs = [(e["built"], base + e["position"], "model", None,
                 {"arm": arm, "checkpoint": ckpt, "seed_set": base, "position": e["position"], "set": e["set"],
                  "case_id": e["case_id"], "is_crossing": e["crossing"], "is_no_target": e["no_target"]})
                for base in SEED_SETS for e in eps]
        rows += run_pool(jobs, model_path=path, processes=processes, overrides={"EMERGENCY_STOP_ENABLED": False})
        print(f"[reseed] {arm} {ckpt}: {len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "pair_reseed_episodes.csv", index=False)


def _rates(x):
    f, z = x[x.set == "field"], x[x.set == "dev"]
    return {"field": f.goal.mean(), "frozen_like": z.goal.mean(), "field_crossing": f[f.is_crossing].goal.mean(),
            "all": x.goal.mean(), "no_target_rate": x[x.is_no_target].goal.mean()}


def report():
    pd.set_option("display.width", 220)
    d = pd.read_csv(OUT / "pair_reseed_episodes.csv", keep_default_na=False, na_values=[""])
    d["goal"] = d.outcome == "goal"
    for c in ("is_crossing", "is_no_target"):
        d[c] = d[c].astype(str) == "True"
    lines = ["v4.3 SAC pair re-scored on three new seed sets (decision on these seeds only)",
             f"{d.position.nunique()} development episodes x {d.seed_set.nunique()} seed sets per checkpoint, safety off", ""]
    t = pd.DataFrame({(a, c): _rates(g) for (a, c), g in d.groupby(["arm", "checkpoint"])}).T.round(3)
    lines += [t.to_string(), ""]
    rng = np.random.default_rng(0)
    for ckpt in ("selected", "final"):
        a = d[(d.arm == "v4.3") & (d.checkpoint == ckpt)].set_index(["position", "seed_set"]).sort_index()
        b = d[(d.arm == "v3") & (d.checkpoint == ckpt)].set_index(["position", "seed_set"]).sort_index()
        diff = {}
        for key in ("field", "frozen_like", "field_crossing"):
            diff[key] = _rates(a.reset_index())[key] - _rates(b.reset_index())[key]
        # bootstrap over scenarios (positions), each keeping its three seeds
        pos_f = a[a.set == "field"].index.get_level_values(0).unique().to_numpy()
        pos_z = a[a.set == "dev"].index.get_level_values(0).unique().to_numpy()
        ga, gb = a.goal.groupby(level=0).mean(), b.goal.groupby(level=0).mean()
        boot = {"field": [], "frozen_like": []}
        for _ in range(2000):
            sf = rng.choice(pos_f, len(pos_f)); sz = rng.choice(pos_z, len(pos_z))
            boot["field"].append(ga[sf].mean() - gb[sf].mean())
            boot["frozen_like"].append(ga[sz].mean() - gb[sz].mean())
        ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}
        gained = int((a.goal & ~b.goal).sum()); lost = int((~a.goal & b.goal).sum())
        line = (f"{ckpt}: v4.3 minus v3 -- field {diff['field']:+.1%} (95 % CI {ci['field'][0]:+.1%} to {ci['field'][1]:+.1%}), "
                f"frozen-like {diff['frozen_like']:+.1%} (95 % CI {ci['frozen_like'][0]:+.1%} to {ci['frozen_like'][1]:+.1%}), "
                f"field crossings {diff['field_crossing']:+.1%}; episodes gained {gained}, lost {lost} of {len(a)}")
        if ckpt == "selected":
            verdict = "PASS" if diff["field"] >= FIELD_GATE and diff["frozen_like"] >= FROZEN_GATE else "FAIL"
            line += f"  -> gate {verdict} (field >= +5, frozen-like >= -2 points)"
        lines.append(line)
    nt = d[d.is_no_target].groupby(["arm", "checkpoint"]).goal.agg(["sum", "size"])
    lines += ["", "-- no-target (development, 50 episodes x 3 seeds)", nt.to_string()]
    text = "\n".join(lines) + "\n"
    (OUT / "pair_reseed_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "report"), nargs="?", default="run")
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    if a.what == "run":
        run(a.processes)
    report()


if __name__ == "__main__":
    main()
