"""Oracle feasibility of the development sets and the test set (2026-10-03).

Decision (2026-10-03): near-impossible episodes should be left out of the test
set and the development set, or kept to a very small share.  This runs
`src/oracle_feasibility.py` (a manoeuvre library rolled out with perfect
foresight, the true vessel model and the policy's action space) on every
episode, with the episode's own evaluation seed:

    python tools/diagnostics/feasibility/oracle_sets.py dev  --processes 5
    python tools/diagnostics/feasibility/oracle_sets.py test --processes 5
    python tools/diagnostics/feasibility/oracle_sets.py deveval --processes 5
    python tools/diagnostics/feasibility/oracle_sets.py report

Development side (the near-impossible rule is calibrated here, before the test
set is read):
* field_dev   the v3 field development set (150), G4's seeds 960,000+;
* g1_conflict G1's 40 near-deployment conflicts, G4's seeds 961,000+;
* dev_v2      the frozen-like development set (120), G4's seeds 962,000+;
* dv4x        the 4.1 extension (`dev_set_v4.extension`, 60), seeds 963,000+.
The same seeds as G4's evaluation, so its PPO outcomes pair with these rows.
Crossings get the full library (`latest_start_s` feeds the crossing trace);
everything else is screened (stops at `STOP_AFTER` solutions with margin).

Development set as the training evaluation sees it (`deveval`): the
formulation's frozen-like set (120) then its field set -- the v3 field
development set (150) and the 4.1 extension (60) -- with the seeds the training
evaluation gives them, `900_000 + position` (`FieldEvalCallback`).  The oracle's
tracking time depends on the seed (sensor noise), so the v4 development-set
filter is decided on these rows.

Test side: test set v3's 1,000 episodes with their own episode seeds,
screened.  Rows are appended as episodes finish, so a stopped run resumes.
Results: `results/feasibility/oracle_<side>.csv`, `oracle_summary.txt`.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(ROOT / "tools" / "diagnostics" / "v4_gates")]

import pandas as pd

import curriculum
import train_formulation as tf

OUT = ROOT / "results" / "feasibility"
DEV_CACHE = OUT / "dev_sets.pkl"
TEST_PKL = ROOT / "results" / "test_set" / "v3" / "set_v3.0.pkl"
_W = {}


def _code(built) -> str:
    cid = str(getattr(built, "case_id", "") or "")
    parts = cid.split("-")
    if cid.startswith(("DV3-", "DV4X-", "G-")) and len(parts) > 1:
        return parts[1]
    return {"head_on": "HO", "crossing": "CR", "overtaking": "OT", "being_overtaken": "BO",
            "no_target": "NT"}.get(str(built.encounter_class), str(built.encounter_class))


def dev_items():
    """[(key, built, seed, meta)], cached: building DV3 and the extension takes minutes."""
    if DEV_CACHE.exists():
        with open(DEV_CACHE, "rb") as fh:
            return pickle.load(fh)
    import formulation_v3 as fv
    import gates
    import dev_set_v4
    items = []
    for i, b in enumerate(fv.field_development_set()):
        items.append((f"field_dev:{b.case_id}", b, 960_000 + i, {"set": "field_dev"}))
    for j, s in enumerate(gates.conflict_scenarios()):
        items.append((f"g1_conflict:{s['case']}", s["built"], 961_000 + j, {"set": "g1_conflict"}))
    for k, b in enumerate(tf.development_set(20)):
        items.append((f"dev_v2:{k:03d}", b, 962_000 + k, {"set": "dev_v2"}))
    for m, b in enumerate(dev_set_v4.extension()):
        items.append((f"dv4x:{b.case_id}", b, 963_000 + m, {"set": "dv4x"}))
    for key, b, seed, meta in items:
        meta.update({"code": _code(b), "class": str(b.encounter_class),
                     "varying": bool((b.flags or {}).get("speed_profile")),
                     "panels": len((b.flags or {}).get("fixed_obstacles") or [])})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(DEV_CACHE, "wb") as fh:
        pickle.dump(items, fh)
    return items


def deveval_items():
    """The formulation-v4 development set in evaluation order, with its evaluation seeds."""
    by = {}
    for key, b, seed, meta in dev_items():
        by.setdefault(meta["set"], []).append((key, b, meta))
    ordered = by["dev_v2"] + by["field_dev"] + by["dv4x"]
    return [(f"deveval:{key}", b, 900_000 + i, {**meta, "position": i}) for i, (key, b, meta) in enumerate(ordered)]


def test_items():
    with open(TEST_PKL, "rb") as fh:
        kept, _ = pickle.load(fh)
    return [(f"test:{it['test_id']}", it["built"], int(it["episode_seed"]),
             {"set": "test_v3", "source": it["source"], "cell": it["cell"], "variant": it["variant"],
              "code": _code(it["built"]), "class": str(it["built"].encounter_class),
              "varying": bool((it["built"].flags or {}).get("speed_profile")), "panels": it.get("panels")})
            for it in kept]


def _init():
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)


def _job(args):
    import oracle_feasibility as of
    key, built, seed, meta, full = args
    rep = of.assess(_W["env"], built, seed, full=full)
    return {"key": key, "seed": seed, **meta, "full": full, **rep}


def _read(path):
    """The rows so far; "null" is an encounter class here, not a missing value."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def run(side: str, processes: int, rounds: int = 3):
    """Runs every episode not yet in the CSV.  Appends to a file on OneDrive were
    once lost silently (5 of 1,000 rows, 2026-10-04), so the run ends by checking
    that every episode is present and repeats the missing ones."""
    for r in range(rounds):
        if not _run_missing(side, processes):
            return
        print(f"[oracle {side}] round {r + 1}: some rows did not reach the CSV; repeating them", flush=True)
    raise SystemExit(f"[oracle {side}] rows still missing after {rounds} rounds")


def _run_missing(side: str, processes: int) -> bool:
    OUT.mkdir(parents=True, exist_ok=True)
    items = {"dev": dev_items, "deveval": deveval_items, "test": test_items}[side]()
    path = OUT / f"oracle_{side}.csv"
    done = set(_read(path).key) if path.exists() else set()
    jobs = [(k, b, s, m, side == "dev" and m["code"] in ("CRP", "CRS", "CR")) for k, b, s, m in items if k not in done]
    jobs.sort(key=lambda j: not j[4])            # full-library crossings first (the trace waits on them)
    print(f"[oracle {side}] {len(items)} episodes, {len(done)} done, {len(jobs)} to run, {processes} processes", flush=True)
    t0, n = time.time(), 0
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    with ProcessPoolExecutor(max_workers=processes, initializer=_init) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for f in as_completed(futures):
            row = f.result()
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            n += 1
            if n % 25 == 0 or n == len(jobs):
                print(f"[oracle {side}] {n}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
    print(f"[oracle {side}] done in {time.time() - t0:.0f} s", flush=True)
    present = set(_read(path).key)
    return any(k not in present for k, _, _, _ in items)


def report():
    pd.set_option("display.width", 220)
    lines = []
    for side in ("dev", "deveval", "test"):
        path = OUT / f"oracle_{side}.csv"
        if not path.exists():
            continue
        d = _read(path)
        d["solved"] = (d.n_success > 0) | (d.nominal_outcome == "goal")
        d["margin"] = (d.n_success_margin > 0) | ((d.nominal_outcome == "goal") & (d.nominal_clearance_m >= 0.2))
        grp = ["source", "code"] if side == "test" else ["set", "code"]
        t = d.groupby(grp).agg(n=("key", "size"), solved=("solved", "mean"), with_margin=("margin", "mean"),
                               after_track=("solved_after_track", "mean"), margin_after_track=("margin_after_track", "mean"),
                               median_t_track_s=("t_track_s", "median"), median_window_s=("decision_window_s", "median"),
                               oracle_s=("oracle_s", "mean")).round(2)
        lines += [f"== {side}: {len(d)} episodes; oracle-unsolved {int((~d.solved).sum())}, "
                  f"no solution with 0.2 m margin {int((~d.margin).sum())}, "
                  f"unsolvable once tracked {int((~d.solved_after_track.astype(bool)).sum())}, "
                  f"no margin once tracked {int((~d.margin_after_track.astype(bool)).sum())}", t.to_string(), ""]
    text = "\n".join(lines) + "\n"
    (OUT / "oracle_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("dev", "deveval", "test", "report", "build"))
    ap.add_argument("--processes", type=int, default=max(1, min(5, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()
    if a.what == "report":
        report()
    elif a.what == "build":
        print(len(dev_items()), "development episodes cached", flush=True)
    else:
        run(a.what, a.processes)
        report()


if __name__ == "__main__":
    main()
