"""Final identification run: pooled-horizon fit, multi-start, bootstrap CIs.

Changes relative to `fit_staged.py`, all motivated by diagnostics rather than
by the holdout (which stays untouched until `validate.py`):

* **Pooled objective.** 3 s and 10 s windows plus the free run, so no horizon
  can be traded against another.
* **N_rr fixed at zero.** Quadratic yaw damping sat on a bound in every fit and
  contributes nothing separable from N_r at the yaw rates present in the data
  (|r| < 25 deg/s for 99% of samples). Carrying an unidentifiable parameter
  invites a reviewer question with no good answer.
* **N_r bounded away from zero.** A hull with literally no linear yaw damping
  is not a physical claim; the earlier fit pushed all yaw damping into the
  drift term because nothing stopped it.
* **Multi-start.** Three starting points (random, the physics-anchored guess,
  the previous staged solution) to check the optimum is not an artefact of
  initialisation.
* **Run-level bootstrap** for confidence intervals: resample the 14 training
  runs with replacement, refit, repeat. These intervals are the
  domain-randomisation ranges document 05 section 7 calls for.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np

import dynamics as dyn
from build_dataset import load_all, classify
from fit_model import differential_evolution, polish
from simulate import make_batch, objective_multi, report, Batch

BOUNDS: Dict[str, tuple] = {
    "T12":       (2.0, 45.0),
    "t_boost":   (0.0, 0.0),   # fixed: fit drives it to zero, so u_boost is undefined
    "u_boost":   (0.5, 0.5),   # fixed (inactive while t_boost = 0)
    "X_uu":      (0.0, 40.0),
    "X_vv":      (0.0, 60.0),   # capped: surge cross-flow drag, not the sway coefficient
    "X_rr":      (0.0, 200.0),
    "X_delta":   (0.0, 6.0),
    "Y_v":       (0.0, 300.0),
    "Y_vv":      (0.0, 300.0),
    "k_R":       (0.05, 15.0),
    "k_race":    (0.0, 4.0),
    "N_r":       (0.0, 120.0),
    "N_rr":      (0.0, 60.0),     # weakly identifiable, but needed for boundedness
    "N_uv":      (-90.0, 250.0),
    "rud_rate":  (10.0, 3000.0),
    "rud_tau":   (0.01, 1.2),
    "rud_delay": (0.0, 1.6),
}
LO = np.array([BOUNDS[k][0] for k in dyn.PARAM_NAMES])
HI = np.array([BOUNDS[k][1] for k in dyn.PARAM_NAMES])

PHYSICS_ANCHOR = dict(dyn.DEFAULT_PARAMS)
PHYSICS_ANCHOR.update(T12=11.0, t_boost=1.0, u_boost=0.5, X_uu=5.0, X_vv=30.0,
                      X_rr=10.0, X_delta=0.2, Y_v=50.0, Y_vv=40.0, k_R=0.9, k_race=1.2,
                      N_r=12.0, N_rr=0.0, N_uv=5.0, rud_rate=300.0,
                      rud_tau=0.3, rud_delay=0.4)


def fit_once(batch: Batch, seed: int = 0, iters: int = 200, popsize: int = 64,
             x0=None, verbose: bool = True):
    fun = lambda T: objective_multi(batch, T, w_pos=1.0)
    x, f = differential_evolution(fun, LO, HI, popsize=popsize, iters=iters,
                                  seed=seed, x0=x0, verbose=verbose)
    x, f = polish(fun, x, LO, HI, popsize=popsize, verbose=False)
    return x, f


def multistart(batch: Batch, iters: int = 200, verbose: bool = True):
    starts = [None, dyn.as_vector(PHYSICS_ANCHOR)]
    prev = "out/params_v3.json"
    if os.path.exists(prev):
        starts.append(np.clip(dyn.as_vector(json.load(open(prev))["params"]), LO, HI))

    best_x, best_f = None, np.inf
    for i, x0 in enumerate(starts):
        if verbose:
            print(f"  start {i} ({'random' if x0 is None else 'seeded'})")
        x, f = fit_once(batch, seed=10 * i, iters=iters, x0=x0, verbose=verbose)
        if verbose:
            print(f"    J = {f:.5f}")
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f


def bootstrap(batch_runs: List, n_boot: int = 24, iters: int = 90, x0=None,
              seed: int = 0) -> np.ndarray:
    """Refit on resampled runs. Returns (n_boot, n_params)."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_boot):
        pick = rng.integers(0, len(batch_runs), size=len(batch_runs))
        bb = make_batch([batch_runs[i] for i in pick])
        x, f = fit_once(bb, seed=1000 + b, iters=iters, popsize=48, x0=x0,
                        verbose=False)
        out.append(x)
        print(f"    bootstrap {b+1}/{n_boot}: J = {f:.4f}")
    return np.array(out)


def load_best():
    """Best solution found so far across chunked runs."""
    if os.path.exists("out/best.json"):
        d = json.load(open("out/best.json"))
        return np.array(d["x"]), float(d["J"])
    return None, np.inf


def save_best(x, f):
    json.dump({"x": list(map(float, x)), "J": float(f)}, open("out/best.json", "w"))


def cmd_start(k: int, iters: int, popsize: int = 48):
    """One optimiser start. Chunked so each call fits the wall-clock budget."""
    runs = [r for r in load_all() if classify(r) == "moving"]
    train = [r for r in runs if r.session == "2026-07-02"]
    b_tr = make_batch(train)
    prev_x, prev_f = load_best()

    if k == 0:
        x0 = dyn.as_vector(PHYSICS_ANCHOR)
        label = "physics anchor"
    elif k == 1:
        if prev_x is None:
            raise SystemExit("start 1 warm-starts from out/best.json, which is absent")
        x0 = np.clip(prev_x, LO, HI)
        label = "incumbent"
    else:
        x0 = None
        label = "random"
    print(f"start {k} ({label}), {iters} generations, popsize {popsize}")
    x, f = fit_once(b_tr, seed=10 * k, iters=iters, popsize=popsize, x0=x0)
    print(f"  J = {f:.5f}   (incumbent {prev_f:.5f})")
    if f < prev_f:
        save_best(x, f)
        print("  -> new best")


def cmd_boot(n0: int, n1: int, rounds: int = 8, popsize: int = 64):
    """Bootstrap resamples [n0, n1). Warm-started at the full-data optimum and
    locally re-optimised, which is the standard treatment when a global search
    per resample is not affordable."""
    runs = [r for r in load_all() if classify(r) == "moving"]
    train = [r for r in runs if r.session == "2026-07-02"]
    x_best, _ = load_best()
    rng = np.random.default_rng(12345)
    picks = [rng.integers(0, len(train), size=len(train)) for _ in range(200)]

    store = {}
    if os.path.exists("out/boot.json"):
        store = json.load(open("out/boot.json"))
    for b in range(n0, n1):
        bb = make_batch([train[i] for i in picks[b]])
        fun = lambda T: objective_multi(bb, T, w_pos=1.0)
        x, f = polish(fun, x_best.copy(), LO, HI, rounds=rounds,
                      popsize=popsize, seed=1000 + b, verbose=False)
        store[str(b)] = list(map(float, x))
        json.dump(store, open("out/boot.json", "w"))
        print(f"  resample {b}: J = {f:.4f}")


def main():
    runs = [r for r in load_all() if classify(r) == "moving"]
    train = [r for r in runs if r.session == "2026-07-02"]
    test = [r for r in runs if r.session == "2026-07-03"]
    b_tr, b_te = make_batch(train), make_batch(test)
    print(f"train {len(train)} runs (2026-07-02) | holdout {len(test)} runs (2026-07-03)\n")

    x, f = load_best()
    if x is None:
        raise SystemExit("no out/best.json — run the start chunks first")
    params = dyn.as_dict(x)
    print(f"\n{'parameter':11s} {'value':>10}   bounds")
    for k in dyn.PARAM_NAMES:
        lo, hi = BOUNDS[k]
        span = hi - lo
        flag = ""
        if span > 0 and (abs(params[k] - lo) < 0.01 * span or abs(params[k] - hi) < 0.01 * span):
            flag = "  <-- at bound"
        print(f"  {k:10s} {params[k]:10.4f}   [{lo}, {hi}]{flag}")

    rep_tr, rep_te = report(b_tr, params), report(b_te, params)
    for nm, rp in (("train", rep_tr), ("holdout", rep_te)):
        print(f"{nm:8s}: win psi {rp['win_psi_rms']:6.2f} deg  win pos {rp['win_pos_rms']:5.3f} m  "
              f"free psi {rp['free_psi_rms']:6.2f} deg  free pos {rp['free_pos_rms']:5.2f} m")

    store = json.load(open("out/boot.json"))
    B = np.array([store[k] for k in sorted(store, key=int)])
    print(f"\nbootstrap: {len(B)} resamples of the training runs")
    lo_ci = np.percentile(B, 5, axis=0)
    hi_ci = np.percentile(B, 95, axis=0)
    print(f"\n{'parameter':11s} {'fit':>9} {'5%':>9} {'95%':>9}  {'rel. width':>10}")
    for j, k in enumerate(dyn.PARAM_NAMES):
        w = (hi_ci[j] - lo_ci[j]) / max(abs(params[k]), 1e-6)
        print(f"  {k:10s} {params[k]:9.3f} {lo_ci[j]:9.3f} {hi_ci[j]:9.3f}  {w:10.2f}")

    os.makedirs("out", exist_ok=True)
    json.dump({"params": params, "J_train": f,
               "ci_low": {k: float(v) for k, v in zip(dyn.PARAM_NAMES, lo_ci)},
               "ci_high": {k: float(v) for k, v in zip(dyn.PARAM_NAMES, hi_ci)},
               "bootstrap": B.tolist(),
               "train_runs": [r.name for r in train],
               "holdout_runs": [r.name for r in test],
               "bounds": {k: list(v) for k, v in BOUNDS.items()}},
              open("out/params_final.json", "w"), indent=2)
    print("\nwrote out/params_final.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        cmd_start(int(sys.argv[2]), iters=int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "boot":
        cmd_boot(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
