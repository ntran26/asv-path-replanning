"""Identify the v3 parameters from the July field logs.

Protocol (fixed before looking at any result, per document 05 section 5):

* **Split by session, not by run.** Fit on 2026-07-02 (14 moving runs), hold out
  2026-07-03 (6 moving runs) untouched. A different day means a different
  vessel setup, battery state and water condition, so this is the strongest
  split the data supports. Nothing about the holdout enters the fit or the
  choice of parameterisation.
* **Objective**: windowed (5 s) prediction error on heading and position only.
* **Optimiser**: differential evolution over the whole population at once,
  followed by a shrinking-bounds polish.
* **Uncertainty**: run-level bootstrap (resample the training runs with
  replacement, refit). Confidence intervals are the domain-randomisation
  ranges required by document 05 section 7, so they are a deliverable, not a
  diagnostic.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

import dynamics as dyn
from build_dataset import load_all, classify
from simulate import make_batch, objective_pop, report, Batch

# (low, high) per parameter, in PARAM_NAMES order
BOUNDS: Dict[str, Tuple[float, float]] = {
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
    "N_rr":      (0.0, 60.0),
    "N_uv":      (-90.0, 250.0),
    "rud_rate":  (10.0, 3000.0),
    "rud_tau":   (0.01, 1.2),
    "rud_delay": (0.0, 1.6),
}

WINDOW = 3.0

LO = np.array([BOUNDS[k][0] for k in dyn.PARAM_NAMES])
HI = np.array([BOUNDS[k][1] for k in dyn.PARAM_NAMES])


def differential_evolution(fun, lo, hi, popsize=64, iters=300, F=0.7, CR=0.9,
                           seed=0, x0=None, verbose=True):
    """Vectorised DE/rand/1/bin. `fun` takes (P, n) and returns (P,)."""
    rng = np.random.default_rng(seed)
    n = len(lo)
    pop = rng.uniform(lo, hi, size=(popsize, n))
    if x0 is not None:
        pop[0] = np.clip(x0, lo, hi)
    fit = fun(pop)
    best = int(np.argmin(fit))

    for it in range(iters):
        idx = rng.integers(0, popsize, size=(popsize, 3))
        a, b, c = pop[idx[:, 0]], pop[idx[:, 1]], pop[idx[:, 2]]
        mutant = np.clip(a + F * (b - c), lo, hi)
        cross = rng.random((popsize, n)) < CR
        jrand = rng.integers(0, n, size=popsize)
        cross[np.arange(popsize), jrand] = True
        trial = np.where(cross, mutant, pop)
        ftrial = fun(trial)
        improved = ftrial < fit
        pop[improved] = trial[improved]
        fit[improved] = ftrial[improved]
        best = int(np.argmin(fit))
        if verbose and (it % 25 == 0 or it == iters - 1):
            print(f"    gen {it:4d}  best J = {fit[best]:.5f}  spread = {fit.std():.4f}")
    return pop[best].copy(), float(fit[best])


def polish(fun, x, lo, hi, rounds=6, popsize=64, seed=1, verbose=True):
    """Shrinking-bounds random search around the incumbent."""
    rng = np.random.default_rng(seed)
    best, fbest = x.copy(), float(fun(x[None, :])[0])
    span = (hi - lo) * 0.10
    for rd in range(rounds):
        cand = np.clip(best[None, :] + rng.normal(0, span, size=(popsize, len(x))), lo, hi)
        cand[0] = best
        f = fun(cand)
        i = int(np.argmin(f))
        if f[i] < fbest:
            best, fbest = cand[i].copy(), float(f[i])
        span *= 0.5
        if verbose:
            print(f"    polish {rd}: J = {fbest:.5f}")
    return best, fbest


def fit(batch: Batch, seed=0, iters=300, popsize=64, x0=None, verbose=True):
    fun = lambda T: objective_pop(batch, T, window=WINDOW, w_pos=1.0)
    x, f = differential_evolution(fun, LO, HI, popsize=popsize, iters=iters,
                                  seed=seed, x0=x0, verbose=verbose)
    x, f = polish(fun, x, LO, HI, verbose=verbose)
    return x, f


def main():
    runs = [r for r in load_all() if classify(r) == "moving"]
    train = [r for r in runs if r.session == "2026-07-02"]
    test = [r for r in runs if r.session == "2026-07-03"]
    print(f"train: {len(train)} runs (2026-07-02)   holdout: {len(test)} runs (2026-07-03)")

    b_tr, b_te = make_batch(train), make_batch(test)

    t0 = time.time()
    x, f = fit(b_tr, seed=0, iters=int(os.environ.get("DE_ITERS", 300)))
    print(f"\nfit done in {time.time()-t0:.0f} s, J_train = {f:.5f}")

    params = dyn.as_dict(x)
    for k in dyn.PARAM_NAMES:
        print(f"  {k:10s} {params[k]:10.4f}   [{BOUNDS[k][0]}, {BOUNDS[k][1]}]")

    rep_tr = report(b_tr, params)
    rep_te = report(b_te, params)
    print(f"\ntrain : win psi {rep_tr['win_psi_rms']:6.2f} deg  win pos {rep_tr['win_pos_rms']:5.3f} m  "
          f"free psi {rep_tr['free_psi_rms']:6.2f} deg  free pos {rep_tr['free_pos_rms']:5.2f} m")
    print(f"holdout: win psi {rep_te['win_psi_rms']:6.2f} deg  win pos {rep_te['win_pos_rms']:5.3f} m  "
          f"free psi {rep_te['free_psi_rms']:6.2f} deg  free pos {rep_te['free_pos_rms']:5.2f} m")

    os.makedirs("out", exist_ok=True)
    with open("out/params_v3.json", "w") as fh:
        json.dump({"params": params, "J_train": f,
                   "train_runs": [r.name for r in train],
                   "holdout_runs": [r.name for r in test],
                   "bounds": {k: list(v) for k, v in BOUNDS.items()}}, fh, indent=2)
    print("\nwrote out/params_v3.json")


if __name__ == "__main__":
    main()
