"""Unified evaluation harness: any controller, any layout set, one CSV schema.

A controller is anything exposing

    predict(obs, deterministic=True) -> action            (or (action, state))

which covers SB3 models and plain Python controllers alike.  If it also exposes
`reset()`, the harness calls it at the start of every episode -- classical
controllers carry integrator and derivative state that must not leak across
episodes.

    # SAC on the frozen 500
    python src/evaluate.py --controller sb3:sac:models/sac_model_1M.zip \
        --layouts eval_layouts/eval_layouts_v1.json --tag sac_1M --workers 6

    # a PPO seed
    python src/evaluate.py --controller sb3:ppo:models/ppo_seed0/best_model.zip \
        --tag ppo_seed0 --workers 6

    # the classical baseline, parameters from JSON
    python src/evaluate.py --controller los_apf:eval_results/baselines/los_apf_best.json \
        --tag los_apf --workers 6 --deterministic-controller

Writes `eval_results/baselines/<tag>/episodes.csv` (one row per episode) and
`summary.json` (mean / median / IQM / stratified bootstrap 95 % CIs).

Determinism
-----------
Every episode is fully pinned by its layout record, so splitting the set across
worker processes cannot change any result.  `--check-workers` verifies that
claim by re-running a slice single-process and diffing.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from env import ASVLidarEnv
from eval_layouts import EVAL_LAYOUTS, load_layouts, reset_to_layout
from metrics import EpisodeRecorder, summarise

OUT_ROOT = os.path.join("eval_results", "baselines")
MAX_STEPS = 2_000          # the env truncates at 700 first; matches evaluate_suite.py


# ---------------------------------------------------------------------------
# Controller construction
# ---------------------------------------------------------------------------
def build_controller(spec):
    """Build a controller from a `--controller` string, or from a dict.

    `sb3:<algo>:<path>`   a stable-baselines3 checkpoint
    `los_apf:<json>`      the classical baseline, parameters from a JSON file
    `los_apf:`            the classical baseline at its default parameters

    A dict form is also accepted so callers can pass parameters directly
    without a file -- the tuning search uses this:

        {"kind": "los_apf", "params": {...}}
        {"kind": "sb3", "algo": "sac", "path": "..."}
    """
    if isinstance(spec, dict):
        kind = spec["kind"]
        if kind == "los_apf":
            from baselines.los_apf import LosApfController
            return LosApfController(**spec.get("params", {}))
        if kind == "sb3":
            from stable_baselines3 import PPO, SAC, TD3, DDPG
            cls = {"sac": SAC, "ppo": PPO, "td3": TD3, "ddpg": DDPG}[spec["algo"].lower()]
            return cls.load(spec["path"], device="cpu")
        raise ValueError(f"unrecognised controller kind: {kind!r}")

    if spec.startswith("sb3:"):
        _, algo, path = spec.split(":", 2)
        from stable_baselines3 import PPO, SAC, TD3, DDPG
        cls = {"sac": SAC, "ppo": PPO, "td3": TD3, "ddpg": DDPG}[algo.lower()]
        return cls.load(path, device="cpu")

    if spec.startswith("los_apf"):
        from baselines.los_apf import LosApfController
        _, _, cfg_path = spec.partition(":")
        params: Dict[str, Any] = {}
        if cfg_path:
            with open(cfg_path, "r") as f:
                loaded = json.load(f)
            # Accept either a bare parameter dict or a tuning-result wrapper.
            params = loaded.get("params", loaded)
        return LosApfController(**params)

    raise ValueError(f"unrecognised controller spec: {spec!r}")


def _as_action(result) -> np.ndarray:
    """Normalise `predict` output; SB3 returns (action, state), plain ones don't."""
    action = result[0] if isinstance(result, tuple) else result
    return np.asarray(action, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def run_episode(controller, env: ASVLidarEnv, record: Dict[str, Any],
                *, deterministic: bool = True,
                max_steps: int = MAX_STEPS) -> Dict[str, Any]:
    """Run one layout and return its metric row."""
    obs, _ = reset_to_layout(env, record)
    if hasattr(controller, "reset"):
        controller.reset()

    rec = EpisodeRecorder(env, record)
    done = False
    truncated = False
    while rec.steps < max_steps:
        action = _as_action(controller.predict(obs, deterministic=deterministic))
        obs, reward, terminated, truncated, info = env.step(action)
        rec.observe(action, reward, info)
        if terminated or truncated:
            done = True
            break

    return rec.finish(truncated=bool(truncated),
                      hit_max_steps=rec.steps >= max_steps and not done)


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------
_WORKER: Dict[str, Any] = {}


def _init_worker(spec: str, deterministic: bool) -> None:
    _WORKER["controller"] = build_controller(spec)
    _WORKER["env"] = ASVLidarEnv(map_width=10.0, map_height=25.0,
                                 max_obs=5, path_mode="straight")
    _WORKER["deterministic"] = deterministic


def _run_in_worker(record: Dict[str, Any]) -> Dict[str, Any]:
    return run_episode(_WORKER["controller"], _WORKER["env"], record,
                       deterministic=_WORKER["deterministic"])


def evaluate(spec: str, records: Sequence[Dict[str, Any]], *,
             deterministic: bool = True, workers: int = 1,
             progress_every: int = 25) -> List[Dict[str, Any]]:
    """Run a controller over a layout set, optionally across processes."""
    started = time.time()

    if workers <= 1:
        controller = build_controller(spec)
        env = ASVLidarEnv(map_width=10.0, map_height=25.0, max_obs=5,
                          path_mode="straight")
        rows = []
        for i, rec in enumerate(records, 1):
            rows.append(run_episode(controller, env, rec,
                                    deterministic=deterministic))
            if progress_every and (i % progress_every == 0 or i == len(records)):
                _progress(i, len(records), rows, started)
        env.close()
        return rows

    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(spec, deterministic)) as pool:
        rows = []
        for i, row in enumerate(pool.imap(_run_in_worker, records, chunksize=4), 1):
            rows.append(row)
            if progress_every and (i % progress_every == 0 or i == len(records)):
                _progress(i, len(records), rows, started)

    # imap preserves input order, but sort defensively so the CSV order is
    # always the layout order regardless of pool behaviour.
    order = {int(r["case_id"]): i for i, r in enumerate(records)}
    rows.sort(key=lambda r: order[int(r["episode_id"])])
    return rows


def _progress(i: int, n: int, rows: List[Dict[str, Any]], started: float) -> None:
    succ = float(np.mean([r["success"] for r in rows]))
    rate = i / max(time.time() - started, 1e-9)
    eta = (n - i) / max(rate, 1e-9)
    print(f"  {i:4d}/{n}  success={succ:.3f}  {rate:.2f} ep/s  eta {eta / 60:.1f} min",
          flush=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, Any]], tag: str) -> None:
    n = len(rows)
    def rate(key):
        return float(np.mean([r[key] for r in rows]))
    print(f"\n{tag}: {n} episodes")
    print(f"  success           {rate('success'):.3f}")
    print(f"  obstacle coll.    {rate('obstacle_collision'):.3f}")
    print(f"  border coll.      {rate('border_collision'):.3f}")
    print(f"  timeout           {rate('timeout'):.3f}")
    print(f"  rms cte           {np.nanmean([r['rms_cte'] for r in rows]):.3f} m")
    print(f"  min obst clear    {np.nanmean([r['min_obstacle_clearance'] for r in rows]):.3f} m")
    print(f"  min border clear  {np.nanmean([r['min_border_clearance'] for r in rows]):.3f} m")
    print(f"  min lat.  clear   {np.nanmean([r['min_lateral_border_clearance'] for r in rows]):.3f} m")
    print(f"  control effort    {np.nanmean([r['control_effort'] for r in rows]):.3f}")
    print(f"  rudder sat frac   {np.nanmean([r['rudder_saturation_fraction'] for r in rows]):.3f}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", required=True,
                    help="sb3:<algo>:<path> | los_apf[:<params.json>]")
    ap.add_argument("--layouts", default=EVAL_LAYOUTS)
    ap.add_argument("--tag", required=True, help="output directory name")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N layouts (smoke tests)")
    ap.add_argument("--stochastic", action="store_true",
                    help="pass deterministic=False to predict")
    ap.add_argument("--deterministic-controller", action="store_true",
                    help="controller has no sampling variability, so suppress "
                         "bootstrap CIs in the summary (use for LOS+APF)")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--check-workers", action="store_true",
                    help="verify parallel and serial runs agree on the first 20")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = load_layouts(args.layouts)
    if args.limit:
        records = records[:args.limit]

    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    deterministic = not args.stochastic

    if args.check_workers:
        subset = records[:20]
        a = evaluate(args.controller, subset, deterministic=deterministic,
                     workers=1, progress_every=0)
        b = evaluate(args.controller, subset, deterministic=deterministic,
                     workers=max(2, args.workers), progress_every=0)
        same = all(_rows_equal(x, y) for x, y in zip(a, b))
        print(f"worker-count determinism over {len(subset)} episodes: "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            for x, y in zip(a, b):
                if not _rows_equal(x, y):
                    print(f"  episode {x['episode_id']} differs")
            raise SystemExit(1)

    print(f"Evaluating {args.controller} over {len(records)} layouts "
          f"({args.workers} worker{'s' if args.workers != 1 else ''}) ...")
    started = time.time()
    rows = evaluate(args.controller, records, deterministic=deterministic,
                    workers=args.workers)
    wall = time.time() - started

    summary = summarise(rows, method=args.tag,
                        deterministic=args.deterministic_controller)
    summary["wall_clock_seconds"] = wall
    summary["controller"] = args.controller
    summary["layouts"] = args.layouts
    summary["n_workers"] = args.workers
    summary["predict_deterministic"] = deterministic

    csv_path = os.path.join(out_dir, "episodes.csv")
    write_rows(csv_path, rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(rows, args.tag)
    print(f"\n  wall clock {wall / 60:.1f} min")
    print(f"  wrote {csv_path}")
    print(f"  wrote {os.path.join(out_dir, 'summary.json')}")


def _rows_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not (np.isnan(x) and np.isnan(y)) and abs(x - y) > 1e-12:
                return False
        elif x != y:
            return False
    return True


if __name__ == "__main__":
    mp.freeze_support()
    main()
