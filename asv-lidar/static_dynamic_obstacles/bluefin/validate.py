"""Validation: does the identified model actually match the vessel better?

Six tests, fixed before the holdout was looked at. Every test reports the
current model (v2), the identified model (v3) and at least one *naive*
reference, because "better than the old model" is a weak claim if the old model
is worse than assuming the vessel does not turn — which, on this data, it
nearly is.

  V1  Heading error vs prediction horizon (1, 2, 3, 5, 10 s and free run)
  V2  Free-run trajectory overlay (figure)
  V3  Per-run paired comparison + Wilcoxon signed-rank over the holdout
  V4  Turn-rate distribution match (two-sample KS + summary statistics)
  V5  Speed / path-length profile match
  V6  Drift angle (course minus heading) distribution match

Naive references:
  freeze   heading held at the value it had at the window start
  const-r  heading extrapolated at the yaw rate measured at the window start

Runs split by session: fitted on 2026-07-02, evaluated on 2026-07-03.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from scipy import stats

import dynamics as dyn
from build_dataset import load_all, classify, Run
from simulate import (Batch, make_batch, simulate, residuals, rms,
                      _smooth_derivative)
from baseline_v2 import simulate_v2

HORIZONS = [1.0, 2.0, 3.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# naive references
# ---------------------------------------------------------------------------
def naive_heading(run: Run, H: float, mode: str) -> np.ndarray:
    """Predicted heading (deg) under a naive rule with the same window resets."""
    psi = run.yaw
    t = run.t
    out = np.empty_like(psi)
    r_meas = _smooth_derivative(t, psi)
    for k in range(len(t)):
        w0 = np.floor(t[k] / H) * H
        j = int(np.clip(np.searchsorted(t, w0, side="left"), 0, len(t) - 1))
        if mode == "freeze":
            out[k] = psi[j]
        else:
            out[k] = psi[j] + r_meas[j] * (t[k] - t[j])
    return out


def naive_rms(runs: List[Run], H: float, mode: str) -> float:
    errs = []
    for run in runs:
        e = naive_heading(run, H, mode) - run.yaw
        errs.append(e[1:])
    return rms(errs)


# ---------------------------------------------------------------------------
# model errors at a given horizon
# ---------------------------------------------------------------------------
def model_errors(batch: Batch, params: Dict[str, float], H):
    reset = None if H is None else H
    p3 = simulate(batch, params, reset_every=reset)
    e3_psi, e3_pos = residuals(batch, p3, skip_after_reset=1)
    p2 = simulate_v2(batch, reset_every=reset)
    e2_psi, e2_pos = residuals(batch, p2, skip_after_reset=1)
    return (e2_psi, e2_pos), (e3_psi, e3_pos)


def v1_horizon_table(batch: Batch, runs: List[Run], params) -> List[Dict]:
    rows = []
    for H in HORIZONS + [None]:
        (e2p, e2x), (e3p, e3x) = model_errors(batch, params, H)
        row = {
            "horizon": "free" if H is None else f"{H:.0f} s",
            "v2_psi": rms(e2p), "v3_psi": rms(e3p),
            "v2_pos": rms(e2x), "v3_pos": rms(e3x),
        }
        if H is not None:
            row["freeze_psi"] = naive_rms(runs, H, "freeze")
            row["constr_psi"] = naive_rms(runs, H, "constr")
        else:
            row["freeze_psi"] = naive_rms(runs, 1e6, "freeze")
            row["constr_psi"] = naive_rms(runs, 1e6, "constr")
        rows.append(row)
    return rows


def v3_paired(batch: Batch, params: Dict[str, float], H: float = 3.0):
    """Per-run paired comparison and a signed-rank test."""
    (e2p, _), (e3p, _) = model_errors(batch, params, H)
    a = np.array([np.sqrt(np.mean(e ** 2)) for e in e2p])
    b = np.array([np.sqrt(np.mean(e ** 2)) for e in e3p])
    try:
        stat, p = stats.wilcoxon(a, b, alternative="greater")
    except ValueError:
        stat, p = np.nan, np.nan
    return a, b, float(p)


def v4_turn_rates(batch: Batch, params: Dict[str, float]):
    p3 = simulate(batch, params, reset_every=None)
    p2 = simulate_v2(batch, reset_every=None)
    meas, m2, m3 = [], [], []
    for i, run in enumerate(batch.runs):
        dt = np.diff(run.t)
        meas.append(np.diff(run.yaw) / dt)
        m2.append(np.diff(np.rad2deg(p2["psi"][i])) / dt)
        m3.append(np.diff(np.rad2deg(p3["psi"][i])) / dt)
    meas, m2, m3 = np.concatenate(meas), np.concatenate(m2), np.concatenate(m3)
    return {
        "meas": meas, "v2": m2, "v3": m3,
        "ks_v2": float(stats.ks_2samp(meas, m2).statistic),
        "ks_v3": float(stats.ks_2samp(meas, m3).statistic),
    }


def v5_path(batch: Batch, params: Dict[str, float]):
    p3 = simulate(batch, params, reset_every=None)
    p2 = simulate_v2(batch, reset_every=None)
    rows = []
    for i, run in enumerate(batch.runs):
        meas = float(run.path_length())
        d2 = float(np.sum(np.hypot(np.diff(p2["x"][i]), np.diff(p2["y"][i]))))
        d3 = float(np.sum(np.hypot(np.diff(p3["x"][i]), np.diff(p3["y"][i]))))
        rows.append({"run": run.name, "meas": meas, "v2": d2, "v3": d3})
    return rows


def v6_drift(batch: Batch, params: Dict[str, float]):
    """Drift angle = course over ground minus heading, where speed is adequate."""
    def drift(x, y, psi_deg, t):
        dx, dy = np.diff(x), np.diff(y)
        spd = np.hypot(dx, dy) / np.diff(t)
        course = np.degrees(np.arctan2(dx, dy))
        mid = 0.5 * (psi_deg[1:] + psi_deg[:-1])
        d = (course - mid + 180) % 360 - 180
        return d[spd > 0.5]

    p3 = simulate(batch, params, reset_every=None)
    p2 = simulate_v2(batch, reset_every=None)
    dm, d2, d3 = [], [], []
    for i, run in enumerate(batch.runs):
        dm.append(drift(run.x, run.y, run.yaw, run.t))
        d2.append(drift(p2["x"][i], p2["y"][i], np.rad2deg(p2["psi"][i]), run.t))
        d3.append(drift(p3["x"][i], p3["y"][i], np.rad2deg(p3["psi"][i]), run.t))
    return (np.concatenate(dm), np.concatenate(d2), np.concatenate(d3))


# ---------------------------------------------------------------------------
def main():
    params = json.load(open("out/params_final.json"))["params"]
    runs = [r for r in load_all() if classify(r) == "moving"]
    train = [r for r in runs if r.session == "2026-07-02"]
    test = [r for r in runs if r.session == "2026-07-03"]
    b_tr, b_te = make_batch(train), make_batch(test)

    results = {}
    for tag, bb, rr in (("train", b_tr, train), ("holdout", b_te, test)):
        print(f"\n=== V1 heading error vs horizon — {tag} ({len(rr)} runs) ===")
        print(f"{'horizon':>8} {'freeze':>8} {'const-r':>8} {'v2':>8} {'v3':>8} "
              f"{'v2 pos':>8} {'v3 pos':>8}")
        rows = v1_horizon_table(bb, rr, params)
        for row in rows:
            print(f"{row['horizon']:>8} {row['freeze_psi']:8.2f} {row['constr_psi']:8.2f} "
                  f"{row['v2_psi']:8.2f} {row['v3_psi']:8.2f} "
                  f"{row['v2_pos']:8.3f} {row['v3_pos']:8.3f}")
        results[f"v1_{tag}"] = rows

        a, b, p = v3_paired(bb, params, H=3.0)
        print(f"\n=== V3 paired per-run heading RMSE at 3 s — {tag} ===")
        print(f"{'run':26s} {'v2':>8} {'v3':>8} {'better?':>8}")
        for run, x, y in zip(rr, a, b):
            print(f"{run.name:26s} {x:8.2f} {y:8.2f} {'v3' if y < x else 'v2':>8}")
        print(f"  v3 better in {int(np.sum(b < a))}/{len(a)} runs, "
              f"Wilcoxon one-sided p = {p:.4f}")
        results[f"v3_{tag}"] = {"v2": a.tolist(), "v3": b.tolist(), "p": p,
                                "runs": [r.name for r in rr]}

        tr = v4_turn_rates(bb, params)
        print(f"\n=== V4 turn-rate distribution — {tag} ===")
        for nm in ("meas", "v2", "v3"):
            z = tr[nm]
            print(f"  {nm:5s}: mean|r| {np.abs(z).mean():6.2f}  p90|r| "
                  f"{np.percentile(np.abs(z),90):6.2f}  sd {z.std():6.2f} deg/s")
        print(f"  KS distance to measurement: v2 {tr['ks_v2']:.3f}   v3 {tr['ks_v3']:.3f}")
        results[f"v4_{tag}"] = {"ks_v2": tr["ks_v2"], "ks_v3": tr["ks_v3"]}

        pr = v5_path(bb, params)
        m = np.array([q["meas"] for q in pr])
        e2 = np.array([q["v2"] for q in pr]) - m
        e3 = np.array([q["v3"] for q in pr]) - m
        print(f"\n=== V5 free-run path length — {tag} ===")
        print(f"  measured mean {m.mean():5.1f} m | v2 error mean {e2.mean():+5.2f} m "
              f"(abs {np.abs(e2).mean():.2f}) | v3 error mean {e3.mean():+5.2f} m "
              f"(abs {np.abs(e3).mean():.2f})")
        results[f"v5_{tag}"] = {"meas": m.tolist(), "v2_err": e2.tolist(),
                                "v3_err": e3.tolist()}

        dm, d2, d3 = v6_drift(bb, params)
        print(f"\n=== V6 drift angle (course - heading) — {tag} ===")
        print(f"  measured sd {dm.std():5.2f} deg | v2 sd {d2.std():5.2f} | v3 sd {d3.std():5.2f}")
        print(f"  KS to measurement: v2 {stats.ks_2samp(dm,d2).statistic:.3f}  "
              f"v3 {stats.ks_2samp(dm,d3).statistic:.3f}")
        results[f"v6_{tag}"] = {"meas_sd": float(dm.std()), "v2_sd": float(d2.std()),
                                "v3_sd": float(d3.std())}

    os.makedirs("out", exist_ok=True)
    json.dump(results, open("out/validation.json", "w"), indent=2)
    print("\nwrote out/validation.json")


if __name__ == "__main__":
    main()
