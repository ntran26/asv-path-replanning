"""Replace the near-impossible episodes of the formulation-v4 development set (2026-10-04).

Decision (2026-10-04): episodes that fail the oracle test -- no manoeuvre that
starts at or after the first track reaches the goal with 0.2 m clearance
(tight, decided before tracking, unsolvable) -- leave the development set as
well as the test set, replaced so the set keeps its size and balance.

Reads the oracle rows graded with the training evaluation's seeds
(`results/feasibility/oracle_deveval.csv`, `oracle_sets.py deveval`).  Each
failing episode is replaced in place -- same position, so the same evaluation
seed `900_000 + position`, and every other episode's seed is unchanged -- by
the next draw of the generator that made its set:

* frozen-like (`train_formulation.development_set`): the class's next
  development-namespace index past those the set uses;
* v3 field development set (`formulation_v3.field_development_set`): the same
  encounter / speed block, item indices past the block's count, the same
  near-deployment flag;
* 4.1 extension (`dev_set_v4.extension`): the same encounter block, indices
  past 10, the same straight / varying-speed flags.

A candidate is accepted only if it passes the same oracle test at that seed.
The accepted recipes go to `configs/dev_set_v4_replacements.json`, which
`dev_set_v4.development_sets` reads; nothing else in the development set moves.

    python tools/diagnostics/feasibility/dev_replacements.py --processes 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(Path(__file__).parent),
                str(ROOT / "tools" / "diagnostics" / "v4_gates")]

import pandas as pd

import curriculum
import train_formulation as tf

ORACLE = ROOT / "results" / "feasibility" / "oracle_deveval.csv"
OUT = ROOT / "configs" / "dev_set_v4_replacements.json"
CANDIDATES = 2
ROUNDS = 8


def next_recipes(meta: dict, key: str, built, start: int, n: int):
    """`n` recipes for replacements of one episode, from generator index `start`."""
    s = meta["set"]
    if s == "dev_v2":
        return [{"set": "dev_v2", "class": str(built.encounter_class), "index": start + j} for j in range(n)]
    cid = key.split(":", 2)[-1]
    parts = cid.split("-")
    if s == "field_dev":                       # DV3-<code>-<CV|VS>-<nn>
        code, speed, item = parts[1], parts[2], int(parts[3]) - 1
        import formulation_v3 as fv
        return [{"set": "field_dev", "code": code, "varying": speed == "VS",
                 "near": (item % 10) < fv.DEV_NEAR_PER_TEN, "i": start + j} for j in range(n)]
    code, speed, leg = parts[1], parts[2], parts[3]   # DV4X-<code>-<VS|CV>-<S|L>-<nn>
    return [{"set": "dv4x", "code": code, "varying": speed == "VS", "straight": leg == "S", "i": start + j}
            for j in range(n)]


def first_index(meta: dict, built) -> int:
    """The first generator index past those the set itself uses."""
    import dev_set_v4
    import formulation_v3 as fv
    if meta["set"] == "dev_v2":
        return dev_set_v4.dev_v2_used(str(built.encounter_class))
    if meta["set"] == "field_dev":
        return max(fv.DEV_NO_TARGET, fv.DEV_PER_ENCOUNTER_CV, fv.DEV_PER_ENCOUNTER_VS)
    return dev_set_v4.EXT_PER_CODE


_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)


def _job(args):
    import dev_set_v4
    import oracle_feasibility as of
    slot, recipe, seed = args
    built = dev_set_v4.from_recipe(recipe)
    if built is None:
        return slot, recipe, None
    return slot, recipe, of.assess(_W["env"], built, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    import oracle_sets
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    d = pd.read_csv(ORACLE, keep_default_na=False, na_values=[""]).set_index("key")
    items = oracle_sets.deveval_items()
    assert all(k in d.index for k, _, _, _ in items), "every development episode needs an oracle row"
    fail = [(k, b, s, m) for k, b, s, m in items
            if not (bool(d.loc[k, "solved_after_track"]) and bool(d.loc[k, "margin_after_track"]))]
    print(f"[dev v4] {len(fail)} of {len(items)} development episodes fail the oracle test", flush=True)
    nxt = {k: first_index(m, b) for k, b, s, m in fail}
    accepted, log, t0 = {}, [], time.time()
    with ProcessPoolExecutor(max_workers=a.processes, initializer=_init) as pool:
        for rnd in range(ROUNDS):
            jobs = []
            for k, b, s, m in fail:
                if k in accepted:
                    continue
                for r in next_recipes(m, k, b, nxt[k], CANDIDATES):
                    jobs.append((k, r, s))
                nxt[k] += CANDIDATES
            if not jobs:
                break
            print(f"[dev v4] round {rnd + 1}: {len(jobs)} candidates", flush=True)
            for slot, recipe, rep in pool.map(_job, jobs, chunksize=1):
                ok = rep is not None and bool(rep["solved_after_track"]) and bool(rep["margin_after_track"])
                log.append({"replaces": slot, **recipe, "passes": ok})
                if ok and slot not in accepted:
                    accepted[slot] = recipe
    meta = {k: (s, m) for k, b, s, m in fail}
    out = {"revision": "4.2", "decided": "2026-10-04", "oracle": str(ORACLE.relative_to(ROOT)),
           "rule": "solved_after_track and margin_after_track (0.2 m)",
           "replacements": [{"position": meta[k][1]["position"], "replaces": k.split(":", 1)[1], "seed": meta[k][0],
                             "recipe": accepted[k]} for k in sorted(accepted, key=lambda k: meta[k][1]["position"])],
           "unfilled": sorted(k for k, *_ in fail if k not in accepted)}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    pd.DataFrame(log).to_csv(ROOT / "results" / "feasibility" / "dev_replacement_candidates.csv", index=False)
    print(f"[dev v4] {len(accepted)} of {len(fail)} replaced in {time.time() - t0:.0f} s -> {OUT.relative_to(ROOT)}"
          + (f"; unfilled {out['unfilled']}" if out["unfilled"] else ""), flush=True)


if __name__ == "__main__":
    main()
