"""Harvest hand-back starts (planning/HANDBACK_STARTS_PLAN.md, 2026-10-01).

Runs a policy with safety layer v2 on and records every state where the filter,
having overridden the policy, hands the helm back -- the slow, off-heading,
near-wall states the policy then has to recover from.  Each record is the
episode's scenario plus the replayable prefix up to that step (own-ship state
before each step, the executed action, the brake flag), which
`ASVLidarEnv.set_start_pool` replays at reset.

Sources, as decided (your call, 2026-10-01: "the dev set and the field set"):
* `dev`   -- the v3 field development set (150), two episode seeds each;
* `field` -- fresh layouts near the three deployment layouts L1-L3 (training
  seeds, `field_training.sample(near=True)`), every encounter, constant and
  varying speed.  The Paper 2 set's own 630 episodes are not used: they are the
  held-out deployment test, and training on their states would leak it.

    python tools/diagnostics/harvest_handback_starts.py \\
        --model runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip \\
        --out results/handback_starts/pool_v1.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

import numpy as np

import constants as cfg
import curriculum
import train_formulation as tf

FIELD_SEED_BASE = 380_000            # generator seeds: a gap no namespace or set uses
FIELD_PER_CELL = 20                  # per encounter x speed (and no-target)
EPISODE_SEED_BASE = 920_000
MIN_GAP_STEPS = 6                    # hand-backs closer than 3 s count once
_W = {}


def field_recipes():
    import field_training as ft
    out = [("NT", False, i) for i in range(FIELD_PER_CELL)]
    for code in ft.ENCOUNTER_CODES:
        out += [(code, False, i) for i in range(FIELD_PER_CELL)]
        out += [(code, True, i) for i in range(FIELD_PER_CELL)]
    return out


def build_field(recipe):
    import field_training as ft
    import scenario as scn
    code, vary, i = recipe
    b = ["NT", *ft.ENCOUNTER_CODES].index(code) * 2 + int(vary)
    base = FIELD_SEED_BASE + b * 1_000 + i * 20
    built = ft.sample(np.random.default_rng(base), namespace="handback_v1", encounter=code, varying=vary,
                      generator=scn.ScenarioGenerator(stage=5, seed_namespace="training"),
                      seed_fn=lambda k, base=base: base + 200_000 + k, solvable_only=True, near=True)
    built.case_id = f"HB-{code}-{'VS' if vary else 'CV'}-{i + 1:02d}"
    return built


def _init(model_path):
    import torch
    torch.set_num_threads(1)
    cfg.SAFETY_VERSION = 2
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    from common import load_model
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=True)
    _W["model"] = load_model(model_path)


def collect(env, built, seed: int, act, source: str = ""):
    """One episode under safety layer v2: (hand-back records, outcome, after-states).
    `after[k]` is the own-ship state after step k, for checking a replay."""
    obs, _ = env.reset(seed=seed, options={"generated": built})
    states, actions, brakes, records, after = [], [], [], [], []
    overridden, last_k = False, -MIN_GAP_STEPS
    while True:
        before = env.own_state()
        brake_count = env.safety_v2_brake_steps
        obs, _, term, trunc, info = env.step(act(obs))
        k = len(states)
        if overridden and not env.safety_v2_changed and k - last_k >= MIN_GAP_STEPS:
            # The policy's action passed again after an override: a hand-back.  The
            # prefix is steps 0 .. k-1; replayed, the policy acts at step k.
            records.append({"built": built, "source": source, "case": built.case_id,
                            "episode_seed": seed, "k": k,
                            "states": list(states), "actions": list(actions), "brakes": list(brakes)})
            last_k = k
        overridden = env.safety_v2_changed
        states.append(before)
        actions.append(env._executed_action.copy().tolist())
        brakes.append(bool(env.safety_v2_brake_steps > brake_count))
        after.append(env.own_state())
        if term or trunc:
            outcome = ("goal" if info.get("reached_goal") else
                       f"collision:{info.get('collision_kind')}" if info.get("collided") else "timeout")
            break
    for r in records:
        r["episode_outcome"] = outcome
    return records, outcome, after


def _episode(job):
    source, built, seed = job
    if isinstance(built, tuple):
        built = build_field(built)
    records, outcome, _ = collect(_W["env"], built, seed, tf.EpisodeActor(_W["model"]), source)
    return records, {"source": source, "case": built.case_id, "outcome": outcome, "handbacks": len(records)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip")
    ap.add_argument("--out", default="results/handback_starts/pool_v1.pkl")
    ap.add_argument("--dev-seeds", type=int, default=2)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N jobs only (a smoke run)")
    args = ap.parse_args()
    os.chdir(ROOT)
    import formulation_v3 as fv
    t0 = time.time()
    dev = fv.field_development_set()
    jobs = [("dev", b, EPISODE_SEED_BASE + 1_000 * r + i) for r in range(args.dev_seeds) for i, b in enumerate(dev)]
    jobs += [("field", rec, EPISODE_SEED_BASE + 50_000 + i) for i, rec in enumerate(field_recipes())]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} episodes ({len(dev)} dev scenarios built in {time.time() - t0:.0f} s)", flush=True)
    with ProcessPoolExecutor(max_workers=args.processes, initializer=_init, initargs=(args.model,)) as pool:
        results = list(pool.map(_episode, jobs, chunksize=1))
    records = [r for recs, _ in results for r in recs]
    rows = [row for _, row in results]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "safety_version": 2, "created": time.strftime("%Y-%m-%d %H:%M"),
            "episodes": len(jobs), "records": len(records),
            "by_source": dict(Counter(r["source"] for r in records)),
            "by_episode_outcome": dict(Counter(r["episode_outcome"] for r in records)),
            "harvest_outcomes": dict(Counter(f"{row['source']}:{row['outcome']}" for row in rows)),
            "prefix_steps_mean": float(np.mean([r["k"] for r in records])) if records else 0.0}
    with open(out, "wb") as fh:
        pickle.dump({"meta": meta, "records": records}, fh)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), f"\n{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
