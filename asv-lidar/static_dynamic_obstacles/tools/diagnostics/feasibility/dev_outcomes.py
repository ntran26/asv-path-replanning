"""Controller outcomes on the oracle's development episodes (2026-10-03).

Checks the oracle (`oracle_sets.py dev`) against controllers that actually ran:
an episode the oracle calls unsolvable but a controller solves is a miss, and
the near-impossible rule must not remove it.  Runs the kept SAC 3 M policy,
LOS-DWA and COLREGs-VO (safety off) on the same 370 development episodes with
the same seeds; G4's PPO outcomes (control, A, B) pair by seed too.

    python tools/diagnostics/feasibility/dev_outcomes.py --processes 3

Writes `results/feasibility/dev_outcomes.csv`.  The test set is never used.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(Path(__file__).parent)]

import pandas as pd

import oracle_sets
from common import run_pool

MODEL = ROOT / "runs" / "sac_formulation_seed0_bl3" / "kept_best_3M" / "best_model.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=3)
    a = ap.parse_args()
    items = oracle_sets.dev_items()
    out = oracle_sets.OUT / "dev_outcomes.csv"
    frames = []
    for name, policy, model in (("sac", "model", MODEL), ("los_dwa", "los_dwa", None), ("colregs_vo", "colregs_vo", None)):
        t0 = time.time()
        jobs = [(b, seed, policy, None, {"key": key, "seed": seed, "controller": name}) for key, b, seed, _ in items]
        rows = run_pool(jobs, model_path=model, processes=a.processes, overrides={"EMERGENCY_STOP_ENABLED": False})
        frames.append(pd.DataFrame(rows)[["key", "seed", "controller", "outcome", "steps", "min_target_range"]])
        pd.concat(frames).to_csv(out, index=False)
        print(f"[dev outcomes] {name}: {len(rows)} episodes in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
