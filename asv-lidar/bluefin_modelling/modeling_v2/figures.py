"""Figures for the identification report."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dynamics as dyn
from build_dataset import load_all, classify
from simulate import make_batch, simulate
from baseline_v2 import simulate_v2
from validate import v4_turn_rates, HORIZONS

C_MEAS, C_V2, C_V3 = "#111111", "#c1121f", "#1f77b4"


def main():
    params = json.load(open("out/params_final.json"))["params"]
    val = json.load(open("out/validation.json"))
    runs = [r for r in load_all() if classify(r) == "moving"]
    test = [r for r in runs if r.session == "2026-07-03"]
    b = make_batch(test)
    p3 = simulate(b, params, reset_every=None)
    p2 = simulate_v2(b, reset_every=None)

    # --- Fig 1: free-run trajectories, holdout -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, i in zip(axes.ravel(), range(len(test))):
        run = test[i]
        ax.plot(run.x, run.y, color=C_MEAS, lw=2.2, label="measured")
        ax.plot(p2["x"][i], p2["y"][i], color=C_V2, lw=1.4, ls="--", label="v2")
        ax.plot(p3["x"][i], p3["y"][i], color=C_V3, lw=1.6, label="v3")
        ax.plot(run.x[0], run.y[0], "o", ms=5, color=C_MEAS)
        ax.set_title(run.name, fontsize=9)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Free-run open-loop replay, holdout session 2026-07-03 "
                 "(no state reset; logged commands only)", fontsize=11)
    fig.tight_layout()
    fig.savefig("out/fig1_trajectories.png", dpi=140)
    plt.close(fig)

    # --- Fig 2: heading traces --------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, i in zip(axes.ravel(), range(len(test))):
        run = test[i]
        ax.plot(run.t, run.yaw, color=C_MEAS, lw=2.0, label="measured")
        ax.plot(run.t, np.rad2deg(p2["psi"][i]), color=C_V2, lw=1.3, ls="--", label="v2")
        ax.plot(run.t, np.rad2deg(p3["psi"][i]), color=C_V3, lw=1.5, label="v3")
        ax2 = ax.twinx()
        ax2.plot(run.tc, run.rudder_cmd, color="0.7", lw=0.8)
        ax2.set_ylim(-260, 110)
        ax2.set_yticks([])
        ax.set_title(run.name, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("t (s)")
    fig.suptitle("Heading, free-run replay (grey: transmitted rudder command)", fontsize=11)
    fig.tight_layout()
    fig.savefig("out/fig2_heading.png", dpi=140)
    plt.close(fig)

    # --- Fig 3: error vs horizon ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = val["v1_holdout"]
    xs = np.arange(len(rows))
    labels = [r["horizon"] for r in rows]
    axes[0].plot(xs, [r["freeze_psi"] for r in rows], "o-", color="0.55", label="naive: freeze heading")
    axes[0].plot(xs, [r["constr_psi"] for r in rows], "s-", color="0.75", label="naive: constant yaw rate")
    axes[0].plot(xs, [r["v2_psi"] for r in rows], "^--", color=C_V2, label="v2")
    axes[0].plot(xs, [r["v3_psi"] for r in rows], "o-", color=C_V3, label="v3")
    axes[0].set_ylabel("heading RMSE (deg)")
    axes[1].plot(xs, [r["v2_pos"] for r in rows], "^--", color=C_V2, label="v2")
    axes[1].plot(xs, [r["v3_pos"] for r in rows], "o-", color=C_V3, label="v3")
    axes[1].set_ylabel("position RMSE (m)")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("prediction horizon")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("V1: prediction error vs horizon, holdout session", fontsize=11)
    fig.tight_layout()
    fig.savefig("out/fig3_horizon.png", dpi=140)
    plt.close(fig)

    # --- Fig 4: turn-rate distribution ------------------------------------
    tr = v4_turn_rates(b, params)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-45, 45, 51)
    ax.hist(tr["meas"], bins=bins, density=True, color=C_MEAS, alpha=0.30, label="measured")
    ax.hist(tr["v2"], bins=bins, density=True, histtype="step", lw=1.8, color=C_V2, label="v2")
    ax.hist(tr["v3"], bins=bins, density=True, histtype="step", lw=1.8, color=C_V3, label="v3")
    ax.set_xlabel("yaw rate (deg/s)")
    ax.set_ylabel("density")
    ax.set_title(f"V4: turn-rate distribution, holdout "
                 f"(KS: v2 {tr['ks_v2']:.3f}, v3 {tr['ks_v3']:.3f})", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("out/fig4_turnrate.png", dpi=140)
    plt.close(fig)

    # --- Fig 5: bootstrap spread ------------------------------------------
    j = json.load(open("out/params_final.json"))
    B = np.array(j["bootstrap"])
    names = [k for k in dyn.PARAM_NAMES if j["bounds"][k][0] != j["bounds"][k][1]]
    cols = [dyn.PARAM_NAMES.index(k) for k in names]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    norm = []
    for c, k in zip(cols, names):
        v = B[:, c]
        m = j["params"][k]
        norm.append(v / m if abs(m) > 1e-9 else v)
    ax.boxplot(norm, labels=names, showfliers=False)
    ax.axhline(1.0, color=C_V3, lw=1, ls="--")
    ax.set_ylabel("bootstrap value / fitted value")
    ax.set_title("Parameter uncertainty: 28 run-level bootstrap resamples "
                 "(training session only)", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig("out/fig5_bootstrap.png", dpi=140)
    plt.close(fig)

    print("wrote out/fig1_trajectories.png .. fig5_bootstrap.png")


if __name__ == "__main__":
    main()
