"""Hard-state starts for baseline v4.3 (2026-10-04; BASELINE_V4_PLAN.md section 5e, item G).

Decision (2026-10-04): give every learner more practice in exactly the states
the crossing failures come from, without handing any learner an action.  The
reward check showed the reward already ranks the successful manoeuvre first in
every failed crossing; the policy has not learned it in these rare states,
where the decision window after the first track is 2-4 s.

A pool record is a start state for `ASVLidarEnv.set_start_pool` (the hand-back
mechanism: the recorded prefix is replayed at reset, so the targets, tracker
and encounter contexts are where they were; the last own-ship state is
jittered by +-10 deg and +-20 % speed).  Each record comes from:

* a fresh dense field-style layout from the **training** namespace
  (`field_training.sample`, namespace "hard_v4", generator seeds
  `HARD_SEED_BASE`+), drawn like v4.1's stage 7: straight legs 70 %, varying-speed
  targets 50 %, near-deployment layouts 50 %, encounters weighted to the failures
  (`CODE_WEIGHTS`);
* a policy-independent approach: the oracle's route follower at cruise until the
  onboard perception first tracks the target (plus `LEAD_STEPS` for half the
  records), through the full `env.step`;
* the oracle test from that state (`src/oracle_feasibility.py`): standing on
  (the route follower) does **not** reach the goal with `MARGIN_M` clearance --
  action is required -- and some manoeuvre of the library started 0-2 s later
  **does** -- the state is still solvable.

No learner is involved in choosing the states, so the pool is the same extra
practice for PPO, RecurrentPPO, SAC and TQC; the oracle supplies which states,
never which action.  The test set and the development sets are not used.

    python tools/diagnostics/feasibility/harvest_hard_starts.py --candidates 1200 --processes 6

Writes `results/hard_starts/pool_v4.pkl` ({"records", "meta"}) and `pool_v4.csv`.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

import numpy as np
import pandas as pd

import curriculum
import train_formulation as tf

OUT = ROOT / "results" / "hard_starts"
HARD_SEED_BASE = 640_000            # generator seeds (seed_fn adds 200,000): unused by every other set
HARD_EPISODE_SEED = 940_000
CODE_WEIGHTS = {"CRP": 0.35, "CRS": 0.35, "HO": 0.15, "OT": 0.08, "BO": 0.07}
STRAIGHT_SHARE, VARYING_SHARE, NEAR_SHARE = 0.70, 0.50, 0.50
LEAD_STEPS = 2                      # half the records start 1 s after the first track
SCREEN_T0_S = (0.0, 1.0, 2.0)       # manoeuvre starts tried from the record's state
ENOUGH = 3                          # stop screening after this many solutions with margin
_W = {}


def recipes(n: int):
    """(code, varying, straight, near, i, lead) for `n` candidates, shares spread exactly."""
    out, j = [], 0
    for code, w in CODE_WEIGHTS.items():
        for i in range(int(round(w * n))):
            spread = lambda share, k=i: int((k + 1) * share) > int(k * share)
            out.append((code, spread(VARYING_SHARE), spread(STRAIGHT_SHARE),
                        spread(NEAR_SHARE, i // 2), i, LEAD_STEPS if i % 2 else 0))
            j += 1
    return out


def build(recipe):
    import field_training as ft
    import scenario as scn
    code, vary, straight, near, i, _ = recipe
    base = HARD_SEED_BASE + list(CODE_WEIGHTS).index(code) * 20_000 + i * 20
    built = ft.sample(np.random.default_rng(base), namespace="hard_v4", encounter=code, varying=vary,
                      generator=scn.ScenarioGenerator(stage=5, seed_namespace="training"),
                      seed_fn=lambda k, base=base: base + 200_000 + k, solvable_only=True,
                      near=near, straight=straight)
    built.case_id = f"HS-{code}-{'VS' if vary else 'CV'}-{'S' if straight else 'L'}-{i + 1:03d}"
    return built


def _init():
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)


def harvest(job):
    import oracle_feasibility as of
    j, recipe = job
    env = _W["env"]
    built = build(recipe)
    seed = HARD_EPISODE_SEED + j
    env.reset(seed=seed, options={"generated": built})
    route = of.Route(env)
    states, actions, brakes, tracked_at = [], [], [], None
    row = {"case": built.case_id, "code": recipe[0], "varying": recipe[1], "straight": recipe[2],
           "near": recipe[3], "lead": recipe[5], "episode_seed": seed, "kept": False, "why": ""}
    while True:
        before = env.own_state()
        r, tau = route.action(env, 0.0)
        _, _, term, trunc, _ = env.step(np.array([r, tau], dtype=np.float32))
        states.append(before)
        actions.append(env._executed_action.copy().tolist())
        brakes.append(False)
        if term or trunc:
            row["why"] = "ended before the start state"
            return None, row
        if tracked_at is None and env.encounter_contexts:
            tracked_at = len(states)
        if tracked_at is not None and len(states) >= tracked_at + recipe[5]:
            break
    row.update(k=len(states), t_start_s=len(states) * 0.5)
    snap = copy.deepcopy(env)
    r2 = of.Route(snap)
    nominal = of._run(copy.deepcopy(snap), lambda e, k: r2.action(e, 0.0))
    row["nominal_outcome"] = nominal["outcome"]
    if nominal["outcome"] == "goal" and nominal["clearance"] >= of.MARGIN_M:
        row["why"] = "standing on succeeds"
        return None, row
    pre, k_done, found, latest = copy.deepcopy(snap), 0, 0, None
    for t0 in SCREEN_T0_S:
        end = None
        while k_done < int(round(t0 / 0.5)):
            end = of.physics_step(pre, *r2.action(pre, 0.0))
            k_done += 1
            if end is not None:
                break
        if end is not None:
            break
        for (_, dpsi, tau, dur) in (m for m in of.library(t0s=(t0,))):
            res = of._run(copy.deepcopy(pre), of.manoeuvre(r2, dpsi, tau, dur))
            if res["outcome"] == "goal" and res["clearance"] >= of.MARGIN_M:
                found += 1
                latest = t0
                if found >= ENOUGH:
                    break
        if found >= ENOUGH:
            break
    row.update(solutions_found=found, latest_screened_start_s=latest)
    if not found:
        row["why"] = "no solution with margin from the start state"
        return None, row
    row.update(kept=True, why="action required and possible")
    rec = {"built": built, "source": "hard_v4", "case": built.case_id, "episode_seed": seed, "k": len(states),
           "states": states, "actions": actions, "brakes": brakes,
           "meta": {k: row[k] for k in ("code", "varying", "straight", "near", "lead", "nominal_outcome",
                                         "solutions_found")}}
    return rec, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=1200)
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--out", default=str(OUT / "pool_v4.pkl"))
    a = ap.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = list(enumerate(recipes(a.candidates)))
    print(f"[hard starts] {len(jobs)} candidate layouts, {a.processes} processes", flush=True)
    t0 = time.time()
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    records, rows = [], []
    with ProcessPoolExecutor(max_workers=a.processes, initializer=_init) as pool:
        for n, (rec, row) in enumerate(pool.map(harvest, jobs, chunksize=1), 1):
            rows.append(row)
            if rec is not None:
                records.append(rec)
            if n % 100 == 0:
                print(f"[hard starts] {n}/{len(jobs)}: {len(records)} kept, {time.time() - t0:.0f} s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "pool_v4.csv", index=False)
    meta = {"decided": "2026-10-04", "candidates": len(jobs), "kept": len(records),
            "code_weights": CODE_WEIGHTS, "shares": {"straight": STRAIGHT_SHARE, "varying": VARYING_SHARE,
                                                      "near": NEAR_SHARE}, "lead_steps": LEAD_STEPS,
            "rule": "standing on fails with margin; a library manoeuvre started 0-2 s later succeeds with margin",
            "kept_by_code": d[d.kept].code.value_counts().to_dict(), "why": d.why.value_counts().to_dict()}
    with open(a.out, "wb") as fh:
        pickle.dump({"records": records, "meta": meta}, fh)
    print(f"[hard starts] {len(records)} of {len(jobs)} kept in {time.time() - t0:.0f} s -> {a.out}", flush=True)
    print(d.groupby("code").kept.agg(["size", "sum"]).to_string(), flush=True)
    print(d.why.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
