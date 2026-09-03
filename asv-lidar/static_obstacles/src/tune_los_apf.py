"""Random search over LOS+APF parameters, on the tuning layouts only.

    python src/tune_los_apf.py --n-configs 250 --workers 4

Under-tuning is the standard objection to a classical baseline in a DRL paper,
so the procedure is fixed in advance and the full record is kept:

* **Search space** -- 20 parameters, ranges declared in `SEARCH_SPACE` below.
  Random search is used rather than grid search because at equal budget it
  covers a 20-dimensional space far better; a grid dense enough to matter would
  need orders of magnitude more evaluations.
* **Budget** -- at least 200 configurations (default 250), each evaluated on all
  100 tuning layouts.  Every configuration and every score is written to
  `apf_tuning_results.csv`, so the search budget and ranges can be quoted and
  the record produced on request.
* **Objective** -- success rate first; among configurations within
  `--tie-margin` of the best success rate, the lowest mean RMS cross-track
  error wins.
* **Data hygiene** -- the search reads `eval_layouts/tune_layouts_v1.json` and
  nothing else.  The frozen 500-episode evaluation set is never touched here.
  Only the single best configuration is afterwards evaluated on it.

The default (hand-chosen) parameters are always included as configuration 0, so
the search can be shown to have improved on them rather than merely differed.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from baselines.los_apf import DEFAULTS, LosApfController
from env import ASVLidarEnv
from eval_layouts import TUNE_LAYOUTS, load_layouts
from evaluate import run_episode

OUT_DIR = os.path.join("eval_results", "baselines")
RESULTS_CSV = os.path.join(OUT_DIR, "apf_tuning_results.csv")
BEST_JSON = os.path.join(OUT_DIR, "los_apf_best.json")

# name -> (kind, low, high) with kind in {"uniform", "loguniform", "choice", "int"}
SEARCH_SPACE: Dict[str, Tuple] = {
    # LOS guidance
    "delta_lookahead":  ("uniform", 2.0, 14.0),
    "w_lookahead":      ("uniform", 0.0, 0.6),
    "max_los_deg":      ("uniform", 45.0, 90.0),

    # APF repulsion
    "k_rep":            ("loguniform", 4.0, 70.0),
    "c_threshold":      ("uniform", 0.30, 0.80),
    "rep_power":        ("uniform", 1.0, 4.0),
    "rep_sigma_deg":    ("uniform", 20.0, 90.0),
    "max_rep_deg":      ("uniform", 35.0, 90.0),

    # head-on symmetry breaking
    "k_headon":         ("loguniform", 8.0, 100.0),
    "headon_deg":       ("uniform", 8.0, 40.0),
    "side_tie":         ("uniform", 0.02, 0.80),
    "default_side":     ("choice", (1.0, -1.0)),

    # heading PID
    "kp":               ("loguniform", 0.004, 0.060),
    "ki":               ("uniform", 0.0, 0.003),
    "kd":               ("uniform", 0.0, 0.040),
    "integral_limit":   ("uniform", 40.0, 400.0),

    # speed control
    "throttle_base":    ("uniform", 0.1, 1.0),
    "k_speed_obs":      ("uniform", 0.0, 2.5),
    "k_speed_head":     ("uniform", 0.0, 2.0),

    # sector geometry: nominal grid (what the env uses) vs true chunk centres
    "bearing_mode":     ("choice", ("nominal", "actual")),
}


def sample_config(rng: np.random.Generator) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "uniform":
            cfg[name] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "loguniform":
            cfg[name] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif kind == "choice":
            options = spec[1]
            cfg[name] = options[int(rng.integers(len(options)))]
        elif kind == "int":
            cfg[name] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            raise ValueError(f"unknown sampler {kind!r}")
    return cfg


# ---------------------------------------------------------------------------
# Worker pool: one env per worker, a fresh controller per episode
# ---------------------------------------------------------------------------
_ENV: Dict[str, Any] = {}


def _init_worker() -> None:
    _ENV["env"] = ASVLidarEnv(map_width=10.0, map_height=25.0, max_obs=5,
                              path_mode="straight")


def _run_task(task):
    config_id, params, record = task
    controller = LosApfController(**params)
    row = run_episode(controller, _ENV["env"], record, deterministic=True)
    row["config_id"] = config_id
    return row


def score_config(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse one configuration's 100 episodes into its scoring row."""
    succ = np.array([r["success"] for r in rows], dtype=np.float64)
    rms = np.array([r["rms_cte"] for r in rows], dtype=np.float64)
    ok = succ > 0.5
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean(succ)),
        "obstacle_collision_rate": float(np.mean([r["obstacle_collision"] for r in rows])),
        "border_collision_rate": float(np.mean([r["border_collision"] for r in rows])),
        "timeout_rate": float(np.mean([r["timeout"] for r in rows])),
        "mean_rms_cte": float(np.nanmean(rms)),
        # Tracking quality on successful episodes only: a collision truncates
        # the trajectory and flatters the RMS, so the all-episode figure is not
        # a clean tie-breaker on its own.
        "mean_rms_cte_success": float(np.nanmean(rms[ok])) if ok.any() else float("nan"),
        "mean_min_obstacle_clearance": float(np.nanmean(
            [r["min_obstacle_clearance"] for r in rows])),
        "mean_min_border_clearance": float(np.nanmean(
            [r["min_border_clearance"] for r in rows])),
        "mean_control_effort": float(np.nanmean([r["control_effort"] for r in rows])),
        "mean_abs_rudder_rate": float(np.nanmean([r["mean_abs_rudder_rate"] for r in rows])),
        "mean_completion_time_s": float(np.nanmean(
            [r["path_completion_time_s"] for r in rows if r["success"] > 0.5]))
        if ok.any() else float("nan"),
    }


def select_best(results: List[Dict[str, Any]], tie_margin: float) -> Dict[str, Any]:
    """Success rate first; RMS CTE breaks ties within `tie_margin` of the best."""
    best_success = max(r["success_rate"] for r in results)
    contenders = [r for r in results
                  if r["success_rate"] >= best_success - tie_margin]
    # Prefer the tracking error measured on successful episodes; fall back to
    # the all-episode figure if a contender never succeeded.
    def key(r):
        v = r["mean_rms_cte_success"]
        return r["mean_rms_cte"] if not np.isfinite(v) else v
    return min(contenders, key=key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-configs", type=int, default=250,
                    help="random configurations to try (brief requires >= 200)")
    ap.add_argument("--layouts", default=TUNE_LAYOUTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20240818)
    ap.add_argument("--tie-margin", type=float, default=0.02,
                    help="success-rate margin within which RMS CTE decides")
    ap.add_argument("--results-csv", default=RESULTS_CSV)
    ap.add_argument("--best-json", default=BEST_JSON)
    args = ap.parse_args()

    if args.n_configs < 200:
        print(f"WARNING: {args.n_configs} configurations is below the 200 the "
              f"brief requires; the search budget will not be defensible.")

    records = load_layouts(args.layouts)
    os.makedirs(OUT_DIR, exist_ok=True)

    if "eval_layouts_v1" in args.layouts:
        raise SystemExit(
            "refusing to tune on the frozen evaluation set -- that is the exact "
            "objection this procedure exists to pre-empt")

    rng = np.random.default_rng(args.seed)
    configs = [dict(DEFAULTS)] + [sample_config(rng) for _ in range(args.n_configs - 1)]

    # Validate every configuration before spending compute on any of them.
    for i, c in enumerate(configs):
        LosApfController(**c)

    print(f"Random search: {len(configs)} configurations x {len(records)} layouts "
          f"= {len(configs) * len(records):,} episodes")
    print(f"  layouts : {args.layouts}")
    print(f"  workers : {args.workers}")
    print(f"  seed    : {args.seed}")
    print(f"  config 0 is the hand-chosen default\n")

    tasks = [(i, c, r) for i, c in enumerate(configs) for r in records]
    started = time.time()
    per_config: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(configs))}
    done = 0

    with mp.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for row in pool.imap_unordered(_run_task, tasks, chunksize=8):
            per_config[row["config_id"]].append(row)
            done += 1
            if done % 500 == 0 or done == len(tasks):
                rate = done / max(time.time() - started, 1e-9)
                eta = (len(tasks) - done) / max(rate, 1e-9)
                complete = sum(1 for v in per_config.values() if len(v) == len(records))
                best = max((score_config(v)["success_rate"]
                            for v in per_config.values() if len(v) == len(records)),
                           default=float("nan"))
                print(f"  {done:6d}/{len(tasks)} episodes | {complete:3d} configs done "
                      f"| best success {best:.3f} | {rate:.1f} ep/s "
                      f"| eta {eta / 60:.0f} min", flush=True)

    results = []
    for i, c in enumerate(configs):
        rows = per_config[i]
        if len(rows) != len(records):
            print(f"  WARNING: config {i} produced {len(rows)} rows, skipping")
            continue
        entry = {"config_id": i, "is_default": int(i == 0)}
        entry.update(score_config(rows))
        entry.update({f"param_{k}": v for k, v in c.items()})
        results.append(entry)

    with open(args.results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = select_best(results, args.tie_margin)
    best_params = {k[len("param_"):]: v for k, v in best.items()
                   if k.startswith("param_")}

    payload = {
        "params": best_params,
        "selection": {
            "config_id": best["config_id"],
            "success_rate": best["success_rate"],
            "mean_rms_cte": best["mean_rms_cte"],
            "mean_rms_cte_success": best["mean_rms_cte_success"],
            "rule": ("highest success rate; ties within "
                     f"{args.tie_margin} broken by mean RMS CTE on successful episodes"),
        },
        "search": {
            "n_configs": len(results),
            "n_layouts": len(records),
            "layouts": args.layouts,
            "total_episodes": len(results) * len(records),
            "seed": args.seed,
            "tie_margin": args.tie_margin,
            "wall_clock_seconds": time.time() - started,
            "space": {k: list(v) if not isinstance(v[1], tuple) else [v[0], list(v[1])]
                      for k, v in SEARCH_SPACE.items()},
            "results_csv": args.results_csv,
        },
    }
    with open(args.best_json, "w") as f:
        json.dump(payload, f, indent=2)

    ranked = sorted(results, key=lambda r: (-r["success_rate"], r["mean_rms_cte"]))
    print(f"\nTop 10 configurations by success rate:")
    print(f"  {'id':>4} {'succ':>6} {'rmsCTE':>7} {'rmsSucc':>8} {'obst':>6} "
          f"{'bord':>6} {'tmo':>6} {'effort':>7}")
    for r in ranked[:10]:
        print(f"  {r['config_id']:>4} {r['success_rate']:>6.3f} {r['mean_rms_cte']:>7.3f} "
              f"{r['mean_rms_cte_success']:>8.3f} {r['obstacle_collision_rate']:>6.3f} "
              f"{r['border_collision_rate']:>6.3f} {r['timeout_rate']:>6.3f} "
              f"{r['mean_control_effort']:>7.3f}")

    default_row = next(r for r in results if r["config_id"] == 0)
    print(f"\n  default config : success {default_row['success_rate']:.3f}, "
          f"rms cte {default_row['mean_rms_cte']:.3f}")
    print(f"  selected config: id {best['config_id']}, "
          f"success {best['success_rate']:.3f}, "
          f"rms cte {best['mean_rms_cte']:.3f}")
    print(f"\n  wrote {args.results_csv}")
    print(f"  wrote {args.best_json}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
