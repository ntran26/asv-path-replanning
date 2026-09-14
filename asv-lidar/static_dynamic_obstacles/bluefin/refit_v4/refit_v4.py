"""Refit v4: the v3 identification re-run with the retrieval windows trimmed.

`bluefin/REVIEW.md` §3.5 found that three training segments (`trial`,
`trial_2`, `calibration#1`) contain several seconds of sustained motion
against the heading at the end of the run -- most likely the vessel being
retrieved -- and §4 found `N_uv` pinned 1.8 % from its lower bound.  Its
recommendations 1 and 2 are exactly this script:

1. trim the retrieval windows, by a rule applied to every run in both sessions;
2. widen the `N_uv` bound from [-90, 250] to [-250, 250];
3. otherwise run `fit_final.py`'s pipeline unchanged: the same pooled
   objective, the same optimiser, three starts, 28 run-level bootstrap
   resamples warm-started at the optimum.

Pre-registered before the fit was run
-------------------------------------
**Trim rule.**  A pose step is *against the heading* when it covers at least
`AGAINST_MIN_STEP_M` and its course differs from the mid-step heading by more
than 90 deg.  The first run of `AGAINST_MIN_RUN` consecutive such steps marks
the start of un-self-propelled motion (the bridge clips S2 at zero, so the
vessel cannot reverse under its own power); the run is truncated there.
Applied to every run, training and holdout alike, and every trim is reported.

**Adoption.**  v4 replaces v3 in the simulator only if, on the untrimmed
2026-07-03 holdout -- the same six runs `REPORT.md` validates on -- none of
these is worse than v3's by more than `ADOPT_TOLERANCE` (relative):

    free-run heading RMSE, 10 s heading RMSE, free-run position RMSE,
    turn-rate KS distance, mean-absolute free-run path-length error

and acceptance tests A1-A5 pass on the v4 parameters and bootstrap.  The
tolerance exists because the training set changed: a model fitted to data
that contained backwards motion is not preferred merely for scoring a hair
better on six runs, and a correction on data-validity grounds should not be
rejected for a difference inside run-to-run noise.

Usage (from this directory)::

    python refit_v4.py dry          # trims and one objective timing, no fit
    python refit_v4.py fit          # starts + bootstrap -> out/
    python refit_v4.py report       # validation and adoption -> out/
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
BLUEFIN = HERE.parent
sys.path.insert(0, str(BLUEFIN))

import dynamics as dyn  # noqa: E402
from build_dataset import Run, classify, load_all  # noqa: E402
from fit_final import BOUNDS as V3_BOUNDS, PHYSICS_ANCHOR  # noqa: E402
from fit_model import differential_evolution, polish  # noqa: E402
from scipy import stats  # noqa: E402
from simulate import (make_batch, objective_multi, residuals, rms,  # noqa: E402
                      simulate, _smooth_derivative)
import ship_model_v3 as v3  # noqa: E402

DATA_DIR = BLUEFIN.parent.parent / "field_deployment"
OUT = HERE / ("out" if os.environ.get("REFIT_VARIANT", "a") == "a" else "out_b")

AGAINST_MIN_STEP_M = 0.15
AGAINST_MIN_RUN = 3
ADOPT_TOLERANCE = 0.05

BOUNDS = dict(V3_BOUNDS)
BOUNDS["N_uv"] = (-250.0, 250.0)

# Variant "b" -- **post hoc**, added after variant "a" failed A5 (11/12 draws)
# with `N_rr` at its upper bound.  `fit_final.py`'s docstring says `N_rr` is
# "fixed at zero" (unidentifiable against `N_r`), but its BOUNDS allowed
# [0, 60]; v3 landed at 0.017 by itself.  "b" fixes it at zero as documented.
# Same trims, same adoption criteria -- but a variant run because another
# failed is weaker evidence, and it is reported as such.
VARIANT = os.environ.get("REFIT_VARIANT", "a")
if VARIANT == "b":
    BOUNDS["N_rr"] = (0.0, 0.0)
LO = np.array([BOUNDS[k][0] for k in dyn.PARAM_NAMES])
HI = np.array([BOUNDS[k][1] for k in dyn.PARAM_NAMES])

START_ITERS = 110
START_POPSIZE = 48
N_BOOT = 28


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def against_heading_start(run: Run) -> int:
    """Pose index where sustained motion against the heading begins, or -1."""
    dx, dy = np.diff(run.x), np.diff(run.y)
    step = np.hypot(dx, dy)
    course = np.degrees(np.arctan2(dx, dy))
    mid = 0.5 * (run.yaw[1:] + run.yaw[:-1])
    diff = np.abs((course - mid + 180.0) % 360.0 - 180.0)
    against = (step >= AGAINST_MIN_STEP_M) & (diff > 90.0)
    count = 0
    for k, flag in enumerate(against):
        count = count + 1 if flag else 0
        if count >= AGAINST_MIN_RUN:
            return k - AGAINST_MIN_RUN + 1
    return -1


def trim(run: Run) -> Tuple[Run, Dict]:
    k = against_heading_start(run)
    if k < 0:
        return run, {}
    t_end = float(run.t[k])
    keep_c = run.tc <= t_end + 1e-9
    trimmed = Run(name=run.name, session=run.session,
                  t=run.t[:k + 1], x=run.x[:k + 1], y=run.y[:k + 1], yaw=run.yaw[:k + 1],
                  tc=run.tc[keep_c], rudder_cmd=run.rudder_cmd[keep_c],
                  rpm_cmd=run.rpm_cmd[keep_c], meta=dict(run.meta))
    return trimmed, {"run": run.name, "t_cut_s": round(t_end, 2),
                     "duration_s": round(run.duration, 2),
                     "samples_removed": int(run.n - trimmed.n)}


def datasets():
    runs = [r for r in load_all(str(DATA_DIR)) if classify(r) == "moving"]
    train_raw = [r for r in runs if r.session == "2026-07-02"]
    test_raw = [r for r in runs if r.session == "2026-07-03"]
    trims = []
    train, test_trim = [], []
    for r in train_raw:
        t, info = trim(r)
        train.append(t)
        if info:
            trims.append({**info, "split": "train"})
    for r in test_raw:
        t, info = trim(r)
        test_trim.append(t)
        if info:
            trims.append({**info, "split": "holdout"})
    return train_raw, train, test_raw, test_trim, trims


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_once(batch, seed, iters, popsize, x0):
    fun = lambda T: objective_multi(batch, T, w_pos=1.0)
    x, f = differential_evolution(fun, LO, HI, popsize=popsize, iters=iters,
                                  seed=seed, x0=x0, verbose=True)
    return polish(fun, x, LO, HI, popsize=popsize, verbose=False)


def cmd_dry():
    train_raw, train, test_raw, _, trims = datasets()
    print(f"train {len(train)} runs, holdout {len(test_raw)} runs")
    for t in trims:
        print("  trimmed", t)
    b = make_batch(train)
    theta = np.vstack([np.clip(dyn.as_vector(v3.IDENTIFIED), LO, HI)] * START_POPSIZE)
    t0 = time.time()
    j = objective_multi(b, theta, w_pos=1.0)
    print(f"one population evaluation: {time.time() - t0:.2f} s, J(v3) = {j[0]:.5f}")
    j_raw = objective_multi(make_batch(train_raw), theta[:1], w_pos=1.0)[0]
    print(f"J(v3) on the untrimmed training set = {j_raw:.5f}")


def cmd_fit():
    OUT.mkdir(exist_ok=True)
    _, train, _, _, trims = datasets()
    json.dump(trims, open(OUT / "trims.json", "w"), indent=2)
    b_tr = make_batch(train)

    starts = [("physics anchor", dyn.as_vector(PHYSICS_ANCHOR)),
              ("v3 incumbent", np.clip(dyn.as_vector(v3.IDENTIFIED), LO, HI)),
              ("random", None)]
    results = []
    for k, (label, x0) in enumerate(starts):
        t0 = time.time()
        print(f"start {k} ({label})")
        x, f = fit_once(b_tr, seed=10 * k, iters=START_ITERS, popsize=START_POPSIZE, x0=x0)
        print(f"  J = {f:.5f}  ({time.time() - t0:.0f} s)")
        results.append({"label": label, "J": float(f), "x": list(map(float, x))})
        json.dump(results, open(OUT / "starts.json", "w"), indent=2)
    best = min(results, key=lambda r: r["J"])
    x_best = np.array(best["x"])

    rng = np.random.default_rng(12345)
    picks = [rng.integers(0, len(train), size=len(train)) for _ in range(N_BOOT)]
    boot = []
    for b, pick in enumerate(picks):
        bb = make_batch([train[i] for i in pick])
        fun = lambda T: objective_multi(bb, T, w_pos=1.0)
        x, f = polish(fun, x_best.copy(), LO, HI, rounds=8, popsize=64,
                      seed=1000 + b, verbose=False)
        boot.append(list(map(float, x)))
        print(f"  resample {b}: J = {f:.4f}")
        json.dump(boot, open(OUT / "boot.json", "w"))

    B = np.array(boot)
    params = dyn.as_dict(x_best)
    json.dump({"params": params, "J_train": best["J"], "start": best["label"],
               "ci_low": dict(zip(dyn.PARAM_NAMES, np.percentile(B, 5, axis=0).tolist())),
               "ci_high": dict(zip(dyn.PARAM_NAMES, np.percentile(B, 95, axis=0).tolist())),
               "bootstrap": B.tolist(),
               "train_runs": [r.name for r in train],
               "trims": trims,
               "bounds": {k: list(v) for k, v in BOUNDS.items()}},
              open(OUT / "params_v4.json", "w"), indent=2)
    print("wrote out/params_v4.json")


# ---------------------------------------------------------------------------
# Validation and the adoption decision
# ---------------------------------------------------------------------------
def naive_rms(runs: List[Run], H: float, mode: str) -> float:
    errs = []
    for run in runs:
        psi, t = run.yaw, run.t
        r_meas = _smooth_derivative(t, psi)
        out = np.empty_like(psi)
        for k in range(len(t)):
            j = int(np.clip(np.searchsorted(t, np.floor(t[k] / H) * H, side="left"), 0, len(t) - 1))
            out[k] = psi[j] if mode == "freeze" else psi[j] + r_meas[j] * (t[k] - t[j])
        errs.append((out - psi)[1:])
    return rms(errs)


def metrics(runs: List[Run], params: Dict[str, float]) -> Dict:
    batch = make_batch(runs)
    row = {}
    for H in (1.0, 2.0, 3.0, 5.0, 10.0, None):
        pred = simulate(batch, params, reset_every=H)
        e_psi, e_pos = residuals(batch, pred, skip_after_reset=1)
        tag = "free" if H is None else f"{H:.0f}s"
        row[f"psi_{tag}"] = rms(e_psi)
        row[f"pos_{tag}"] = rms(e_pos)

    free = simulate(batch, params, reset_every=None)
    meas_r, sim_r, path_err, drift_m, drift_s = [], [], [], [], []
    for i, run in enumerate(runs):
        dt = np.diff(run.t)
        meas_r.append(np.diff(run.yaw) / dt)
        sim_r.append(np.diff(np.rad2deg(free["psi"][i])) / dt)
        sim_len = float(np.sum(np.hypot(np.diff(free["x"][i]), np.diff(free["y"][i]))))
        path_err.append(sim_len - run.path_length())
        for xs, ys, psi, store in ((run.x, run.y, run.yaw, drift_m),
                                   (free["x"][i], free["y"][i], np.rad2deg(free["psi"][i]), drift_s)):
            ddx, ddy = np.diff(xs), np.diff(ys)
            spd = np.hypot(ddx, ddy) / dt
            d = (np.degrees(np.arctan2(ddx, ddy)) - 0.5 * (psi[1:] + psi[:-1]) + 180.0) % 360.0 - 180.0
            store.append(d[spd > 0.5])
    meas_r, sim_r = np.concatenate(meas_r), np.concatenate(sim_r)
    row["ks_turn_rate"] = float(stats.ks_2samp(meas_r, sim_r).statistic)
    row["path_err_mean"] = float(np.mean(path_err))
    row["path_err_abs"] = float(np.mean(np.abs(path_err)))
    row["drift_sd_meas"] = float(np.concatenate(drift_m).std())
    row["drift_sd_sim"] = float(np.concatenate(drift_s).std())
    return row


def sample_from(bootstrap: np.ndarray, rng, scale=1.0, jitter=0.05) -> Dict[str, float]:
    """`ship_model_v3.sample_params`, over an arbitrary bootstrap."""
    n = bootstrap.shape[0]
    i, k = rng.integers(0, n), rng.integers(0, n)
    w = rng.random()
    vec = w * bootstrap[i] + (1.0 - w) * bootstrap[k]
    mean = bootstrap.mean(axis=0)
    vec = mean + (vec - mean) * scale
    if jitter > 0:
        vec = vec + rng.normal(0.0, jitter * scale, size=vec.shape) * bootstrap.std(axis=0)
    vec = np.maximum(vec, 0.0) * (mean >= 0) + np.minimum(vec, 0.0) * (mean < 0)
    return dict(zip(dyn.PARAM_NAMES, map(float, vec)))


def cmd_report():
    import acceptance as acc
    from vessel_sim import VesselSim, turning_circle

    fit = json.load(open(OUT / "params_v4.json"))
    p4, p3 = fit["params"], dict(v3.IDENTIFIED)
    train_raw, train, test_raw, test_trim, trims = datasets()

    table = {}
    for split, runs in (("train_trimmed", train), ("train_untrimmed", train_raw),
                        ("holdout", test_raw), ("holdout_trimmed", test_trim)):
        table[split] = {"v3": metrics(runs, p3), "v4": metrics(runs, p4)}
        table[split]["naive"] = {f"freeze_{H:.0f}s": naive_rms(runs, H, "freeze")
                                 for H in (1.0, 2.0, 3.0, 5.0, 10.0)}
        table[split]["naive"]["freeze_free"] = naive_rms(runs, 1e6, "freeze")

    hold = table["holdout"]
    criteria = {k: (hold["v4"][k], hold["v3"][k],
                    hold["v4"][k] <= hold["v3"][k] * (1.0 + ADOPT_TOLERANCE))
                for k in ("psi_free", "psi_10s", "pos_free", "ks_turn_rate", "path_err_abs")}

    # A1-A5 on the v4 hull and bootstrap.
    acc.RESULTS.clear()
    a1 = acc.a1_bounded(p4, quiet=True)
    a2 = acc.a2_steady_turn(p4, quiet=True)
    a3 = acc.a3_timestep(p4, quiet=True)
    rng = np.random.default_rng(0)
    B = np.array(fit["bootstrap"])
    draws_ok = 0
    for _ in range(12):
        p = sample_from(B, rng)
        draws_ok += int(acc.a1_bounded(p, quiet=True) and acc.a2_steady_turn(p, quiet=True)
                        and acc.a3_timestep(p, quiet=True))
    turns = [turning_circle(VesselSim(params=p4), helm=h, duration=60.0)
             for h in (0.25, 0.5, 0.75, 1.0)]
    acceptance = {"A1": a1, "A2": a2, "A3": a3, "A5_draws_usable": f"{draws_ok}/12",
                  "A5": draws_ok == 12,
                  "steady_turn_degps": [round(abs(t["yaw_rate_degps"]), 2) for t in turns],
                  "radius_L": round(turns[-1]["radius_L"], 2),
                  "speed_ratio": round(turns[-1]["speed_ratio"], 3)}

    at_bound = []
    for k in dyn.PARAM_NAMES:
        lo, hi = BOUNDS[k]
        span = hi - lo
        if span > 0 and min(abs(p4[k] - lo), abs(p4[k] - hi)) < 0.03 * span:
            at_bound.append(k)

    adopt = all(c[2] for c in criteria.values()) and a1 and a2 and a3 and draws_ok == 12
    report = {"trims": trims, "table": table,
              "criteria": {k: {"v4": v[0], "v3": v[1], "ok": v[2]} for k, v in criteria.items()},
              "acceptance": acceptance, "params_v3": p3, "params_v4": p4,
              "ci_low": fit["ci_low"], "ci_high": fit["ci_high"],
              "at_bound_v4": at_bound, "adopt": bool(adopt),
              "tolerance": ADOPT_TOLERANCE}
    json.dump(report, open(OUT / "validation_v4.json", "w"), indent=2)

    print(json.dumps({"trims": trims, "criteria": report["criteria"],
                      "acceptance": acceptance, "at_bound_v4": at_bound,
                      "adopt": adopt}, indent=2))
    for split in ("holdout", "train_trimmed"):
        print(f"\n{split}")
        for key in ("psi_1s", "psi_2s", "psi_3s", "psi_5s", "psi_10s", "psi_free",
                    "pos_3s", "pos_10s", "pos_free", "ks_turn_rate", "path_err_mean",
                    "path_err_abs", "drift_sd_meas", "drift_sd_sim"):
            print(f"  {key:14s} v3 {table[split]['v3'][key]:8.3f}   v4 {table[split]['v4'][key]:8.3f}")
        print("  naive", {k: round(v, 2) for k, v in table[split]["naive"].items()})
    print("\nparameters")
    for k in dyn.PARAM_NAMES:
        print(f"  {k:10s} v3 {p3[k]:10.4f}   v4 {p4[k]:10.4f}   "
              f"[{fit['ci_low'][k]:.3f}, {fit['ci_high'][k]:.3f}]   bounds {BOUNDS[k]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dry"
    {"dry": cmd_dry, "fit": cmd_fit, "report": cmd_report}[cmd]()
