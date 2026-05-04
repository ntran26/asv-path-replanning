#!/usr/bin/env python3
"""
Plot major RL training/evaluation figures for the ASV project.

Main plots:
  1) Smoothed training reward vs timesteps            (from train_monitor.csv)
  2) Smoothed episode length vs timesteps             (from train_monitor.csv)
  3) Evaluation success rate vs checkpoint timesteps  (from eval_summary.{json,csv})
  4) Mean absolute CTE vs checkpoint timesteps        (from eval_summary.{json,csv})

Optional extras:
  - failure mode rates vs timesteps
  - heading/course error vs timesteps
  - per-case bar charts for the latest checkpoint
  - TensorBoard scalar plots (if tensorboard package + event files are available)

Examples:
  python plot_training_results.py \
      --monitor /mnt/data/train_monitor.csv \
      --eval-summary /mnt/data/eval_summary.json \
      --eval-metrics /mnt/data/eval_metrics.json \
      --outdir /mnt/data/plots

  python plot_training_results.py \
      --monitor train_monitor.csv --eval-summary eval_summary.csv --outdir plots \
      --with-extras --tb-dir ./sac_log
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional TensorBoard support
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    HAVE_TENSORBOARD = True
except Exception:
    HAVE_TENSORBOARD = False
    EventAccumulator = None  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", type=str, required=True, help="Path to train_monitor.csv")
    p.add_argument("--eval-summary", type=str, required=True, help="Path to eval_summary.json/csv")
    p.add_argument("--eval-metrics", type=str, default=None, help="Path to eval_metrics.json/csv (optional but recommended)")
    p.add_argument("--outdir", type=str, default="plots", help="Directory for output plots")
    p.add_argument("--window", type=int, default=50, help="Rolling window (episodes) for smoothed training plots")
    p.add_argument("--with-extras", action="store_true", help="Generate extra evaluation plots")
    p.add_argument("--tb-dir", type=str, default=None, help="TensorBoard log directory (optional)")
    p.add_argument(
        "--tb-tags",
        nargs="*",
        default=None,
        help="Specific TensorBoard scalar tags to plot. If omitted, common useful tags are attempted.",
    )
    return p.parse_args()

def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    if path.suffix.lower() == ".csv":
        # SB3 monitor files have a JSON comment in the first line.
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
        if first.startswith("#"):
            return pd.read_csv(path, comment="#")
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def save_plot(fig: plt.Figure, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    w = max(int(window), 1)
    return series.rolling(window=w, min_periods=1).mean()

def add_cumulative_timesteps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "l" not in out.columns:
        raise KeyError("train monitor file must have column 'l' (episode length)")
    out["timesteps"] = out["l"].cumsum()
    return out

def pick_cte_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["mean_abs_cte", "mean_abs_tgt", "mean_abs_cross_track_error"]:
        if col in df.columns:
            return col
    return None

def pick_course_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["mean_abs_course_error", "mean_abs_angle_diff", "mean_abs_heading_error"]:
        if col in df.columns:
            return col
    return None

def plot_training_reward(monitor_df: pd.DataFrame, window: int, outdir: Path) -> None:
    df = add_cumulative_timesteps(monitor_df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["timesteps"], df["r"], alpha=0.25, label="Episode reward")
    ax.plot(df["timesteps"], rolling_mean(df["r"], window), linewidth=2.0, label=f"Reward ({window}-ep MA)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training reward vs timesteps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "01_training_reward_vs_timesteps.png")

def plot_episode_length(monitor_df: pd.DataFrame, window: int, outdir: Path) -> None:
    df = add_cumulative_timesteps(monitor_df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["timesteps"], df["l"], alpha=0.25, label="Episode length")
    ax.plot(df["timesteps"], rolling_mean(df["l"], window), linewidth=2.0, label=f"Length ({window}-ep MA)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title("Episode length vs timesteps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "02_episode_length_vs_timesteps.png")

def plot_eval_success(eval_summary: pd.DataFrame, outdir: Path) -> None:
    if "timesteps" not in eval_summary.columns or "success_rate" not in eval_summary.columns:
        raise KeyError("eval summary must contain 'timesteps' and 'success_rate'")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eval_summary["timesteps"], eval_summary["success_rate"], marker="o", linewidth=2.0)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Evaluation success rate vs timesteps")
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "03_eval_success_rate_vs_timesteps.png")

def plot_eval_cte(eval_summary: pd.DataFrame, outdir: Path) -> None:
    cte_col = pick_cte_col(eval_summary)
    if cte_col is None:
        raise KeyError("eval summary does not contain a recognized CTE column")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eval_summary["timesteps"], eval_summary[cte_col], marker="o", linewidth=2.0)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean absolute CTE")
    ax.set_title("Mean absolute cross-track error vs timesteps")
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "04_eval_mean_abs_cte_vs_timesteps.png")

def plot_failure_modes(eval_summary: pd.DataFrame, outdir: Path) -> None:
    cols = [c for c in ["border_rate", "obstacle_rate", "timeout_rate", "collision_rate"] if c in eval_summary.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in cols:
        ax.plot(eval_summary["timesteps"], eval_summary[c], marker="o", linewidth=1.8, label=c)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Failure rates vs timesteps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "05_eval_failure_rates_vs_timesteps.png")

def plot_heading_errors(eval_summary: pd.DataFrame, outdir: Path) -> None:
    cols = []
    course_col = pick_course_col(eval_summary)
    if course_col:
        cols.append(course_col)
    if "mean_abs_lookahead_error" in eval_summary.columns:
        cols.append("mean_abs_lookahead_error")
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in cols:
        ax.plot(eval_summary["timesteps"], eval_summary[c], marker="o", linewidth=1.8, label=c)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Error (deg)")
    ax.set_title("Heading/course errors vs timesteps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "06_eval_heading_errors_vs_timesteps.png")

def plot_final_per_case(eval_metrics: pd.DataFrame, outdir: Path) -> None:
    if eval_metrics is None or eval_metrics.empty:
        return
    if "timesteps" not in eval_metrics.columns:
        return
    latest_t = eval_metrics["timesteps"].max()
    df = eval_metrics.loc[eval_metrics["timesteps"] == latest_t].copy()
    if df.empty or "test_case" not in df.columns:
        return
    df = df.sort_values("test_case")
    cte_col = pick_cte_col(df)
    course_col = pick_course_col(df)

    nrows = 2 if (cte_col and course_col) else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, 6 if nrows == 2 else 4.5), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax0 = axes[0]
    success_vals = df["success"].astype(float).values if "success" in df.columns else np.zeros(len(df))
    ax0.bar(df["test_case"].astype(str), success_vals)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_ylabel("Success")
    ax0.set_title(f"Per-case performance at latest checkpoint ({int(latest_t)} steps)")
    ax0.grid(True, axis="y", alpha=0.3)

    if nrows == 2:
        ax1 = axes[1]
        width = 0.35 if course_col else 0.6
        x = np.arange(len(df))
        if cte_col:
            ax1.bar(x - width/2 if course_col else x, df[cte_col].astype(float).values, width=width, label=cte_col)
        if course_col:
            ax1.bar(x + width/2, df[course_col].astype(float).values, width=width, label=course_col)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["test_case"].astype(str).tolist())
        ax1.set_ylabel("Metric value")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.set_xlabel("Test case")
    else:
        axes[0].set_xlabel("Test case")

    save_plot(fig, outdir / "07_final_per_case_metrics.png")

def find_event_files(tb_dir: Path) -> list[Path]:
    return sorted([p for p in tb_dir.rglob("events.out.tfevents.*") if p.is_file()])

def default_tb_tags() -> list[str]:
    return [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "eval/mean_ep_reward",
        "eval/success_rate",
        "eval/mean_abs_cte",
        "eval/mean_abs_tgt",
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
        "train/ent_coef_loss",
        "train/learning_rate",
        "train/n_updates",
        # possible extras if user logged them manually
        "train/qf1_loss",
        "train/qf2_loss",
        "train/value_loss",
        "train/policy_loss",
        "train/entropy_loss",
        "train/q_values",
    ]

def sanitize_filename(tag: str) -> str:
    keep = []
    for ch in tag:
        keep.append(ch if ch.isalnum() else "_")
    return "".join(keep).strip("_")

def collect_tb_scalars(tb_dir: Path, tags: Optional[Iterable[str]]) -> dict[str, pd.DataFrame]:
    if not HAVE_TENSORBOARD:
        print("[warn] TensorBoard package not installed; skipping TensorBoard plots.")
        return {}
    event_files = find_event_files(tb_dir)
    if not event_files:
        print(f"[warn] No TensorBoard event files found under: {tb_dir}")
        return {}

    # Use the newest event file by default; SB3 typically writes one active file per run.
    event_file = event_files[-1]
    print(f"[info] Reading TensorBoard scalars from: {event_file}")
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    wanted = list(tags) if tags else default_tb_tags()
    found = [t for t in wanted if t in available]
    missing = [t for t in wanted if t not in available]
    if missing:
        print("[info] TensorBoard tags not found (skipped):")
        for t in missing:
            print(f"  - {t}")

    out: dict[str, pd.DataFrame] = {}
    for tag in found:
        events = ea.Scalars(tag)
        out[tag] = pd.DataFrame({
            "wall_time": [e.wall_time for e in events],
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        })
    return out

def plot_tb_scalars(tb_scalars: dict[str, pd.DataFrame], outdir: Path) -> None:
    if not tb_scalars:
        return
    tb_out = outdir / "tb_plots"
    tb_out.mkdir(parents=True, exist_ok=True)
    for tag, df in tb_scalars.items():
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df["step"], df["value"], linewidth=1.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)
        save_plot(fig, tb_out / f"{sanitize_filename(tag)}.png")

def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    monitor_df = load_table(args.monitor)
    eval_summary_df = load_table(args.eval_summary)
    eval_metrics_df = load_table(args.eval_metrics) if args.eval_metrics else None

    # Major plots
    plot_training_reward(monitor_df, args.window, outdir)
    plot_episode_length(monitor_df, args.window, outdir)
    plot_eval_success(eval_summary_df, outdir)
    plot_eval_cte(eval_summary_df, outdir)

    # Optional extras
    if args.with_extras:
        plot_failure_modes(eval_summary_df, outdir)
        plot_heading_errors(eval_summary_df, outdir)
        if eval_metrics_df is not None:
            plot_final_per_case(eval_metrics_df, outdir)

    if args.tb_dir:
        tb_scalars = collect_tb_scalars(Path(args.tb_dir), args.tb_tags)
        plot_tb_scalars(tb_scalars, outdir)

    print(f"Plots written to: {outdir}")

if __name__ == "__main__":
    main()
