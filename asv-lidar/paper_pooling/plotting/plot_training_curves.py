#!/usr/bin/env python3
"""
Plot ASV SAC training/evaluation curves for the paper.

Recommended main input:
    eval_summary.json

Optional input:
    train_monitor.csv

Example:
    python plot_training_curves.py \
        --eval-summary eval_summary.json \
        --monitor train_monitor.csv \
        --out-dir paper_plots \
        --title "Feasible pooling SAC with Stage-1 speed control"

Outputs:
    paper_plots/eval_reward_success_rpm.png/.svg        # default full figure
    paper_plots/eval_mean_reward.png/.svg               # with --only-reward
    paper_plots/group_success_by_obstacle.png/.svg       # default full figure
    paper_plots/train_monitor_reward.png/.svg            # only if --monitor is given
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list of dictionaries.")
    rows = [r for r in data if isinstance(r, dict) and "timesteps" in r]
    rows.sort(key=lambda r: int(r.get("timesteps", 0)))
    if not rows:
        raise ValueError(f"No rows with a 'timesteps' field found in {path}.")
    return rows


def arr(rows: List[Dict[str, Any]], key: str, scale: float = 1.0) -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(key, np.nan)
        try:
            vals.append(float(v) * scale)
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


def finite_any(a: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(a)))


def x_values(rows: List[Dict[str, Any]], x_units: str) -> tuple[np.ndarray, str]:
    steps = arr(rows, "timesteps")
    if x_units == "steps":
        return steps, "Timesteps"
    if x_units == "k":
        return steps / 1_000.0, "Timesteps (k)"
    if x_units == "m":
        return steps / 1_000_000.0, "Timesteps (million)"
    raise ValueError(f"Unknown x_units: {x_units}")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    print(f"Saved: {out_dir / (stem + '.png')}")
    print(f"Saved: {out_dir / (stem + '.svg')}")


def _steps_to_x(steps: float, x_units: str) -> float:
    if x_units == "steps":
        return steps
    if x_units == "k":
        return steps / 1_000.0
    if x_units == "m":
        return steps / 1_000_000.0
    raise ValueError(f"Unknown x_units: {x_units}")


def add_training_stage_markers(ax: plt.Axes, x_units: str) -> None:
    """
    Add dashed vertical lines and labels for the training curriculum:
      - cruise (fixed-speed): 0 to 700k
      - stage 1: 700k to 800k
      - stage 2: 800k to 900k
      - stage 3: 900k to 1.0M
    """
    # boundaries between phases
    boundaries = [700_000, 800_000, 900_000]

    # label centers for each phase
    label_centers = [350_000, 750_000, 850_000, 950_000]
    label_texts = ["cruise", "stage 1", "stage 2", "stage 3"]

    # dashed boundaries
    for s in boundaries:
        ax.axvline(
            _steps_to_x(s, x_units),
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            zorder=0,
        )

    # labels at the top of the axes
    for s, txt in zip(label_centers, label_texts):
        ax.text(
            _steps_to_x(s, x_units),
            0.98,
            txt,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )


# def plot_eval_reward_only(rows: List[Dict[str, Any]], out_dir: Path, title: str, x_units: str) -> None:
#     """Plot only the mean evaluation reward curve, with optional ±1 std. band."""
#     x, xlabel = x_values(rows, x_units)
#     mean_reward = arr(rows, "mean_ep_reward")
#     std_reward = arr(rows, "std_ep_reward")

#     fig, ax = plt.subplots(figsize=(7.2, 4.4))
#     ax.plot(x, mean_reward, marker="o", linewidth=1.8, label="Mean eval reward")
#     if finite_any(std_reward):
#         lower = mean_reward - std_reward
#         upper = mean_reward + std_reward
#         ax.fill_between(x, lower, upper, alpha=0.18, label="±1 std. reward")
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Mean episode reward")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="lower right", frameon=True)
#     if title:
#         ax.set_title(title)
#     fig.tight_layout()
#     save_figure(fig, out_dir, "eval_mean_reward")
#     plt.close(fig)


def plot_eval_reward_only(rows: List[Dict[str, Any]], out_dir: Path, title: str, x_units: str) -> None:
    """Plot only the mean evaluation reward curve, with optional ±1 std. band."""
    x, xlabel = x_values(rows, x_units)
    mean_reward = arr(rows, "mean_ep_reward")
    std_reward = arr(rows, "std_ep_reward")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, mean_reward, marker="o", linewidth=1.8, label="Mean eval reward")
    if finite_any(std_reward):
        lower = mean_reward - std_reward
        upper = mean_reward + std_reward
        ax.fill_between(x, lower, upper, alpha=0.18, label="±1 std. reward")

    add_training_stage_markers(ax, x_units)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean episode reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, out_dir, "eval_mean_reward")
    plt.close(fig)


def plot_eval_reward_success_rpm(rows: List[Dict[str, Any]], out_dir: Path, title: str, x_units: str) -> None:
    x, xlabel = x_values(rows, x_units)

    mean_reward = arr(rows, "mean_ep_reward")
    std_reward = arr(rows, "std_ep_reward")
    success = arr(rows, "success_rate", 100.0)
    obstacle = arr(rows, "obstacle_rate", 100.0)
    border = arr(rows, "border_rate", 100.0)
    timeout = arr(rows, "timeout_rate", 100.0)

    mean_rpm = arr(rows, "mean_rpm")
    min_rpm = arr(rows, "min_rpm")
    max_rpm = arr(rows, "max_rpm")
    has_rpm = finite_any(mean_rpm)

    nrows = 3 if has_rpm else 2
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(7.2, 3.0 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(x, mean_reward, marker="o", linewidth=1.8, label="Mean eval reward")
    if finite_any(std_reward):
        lower = mean_reward - std_reward
        upper = mean_reward + std_reward
        ax.fill_between(x, lower, upper, alpha=0.18, label="±1 std. reward")
    ax.set_ylabel("Mean episode reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    ax = axes[1]
    ax.plot(x, success, marker="o", linewidth=1.8, label="Success")
    if finite_any(obstacle):
        ax.plot(x, obstacle, marker="o", linewidth=1.4, label="Obstacle collision")
    if finite_any(border):
        ax.plot(x, border, marker="o", linewidth=1.4, label="Border collision")
    if finite_any(timeout):
        ax.plot(x, timeout, marker="o", linewidth=1.4, label="Timeout")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, ncols=2)

    if has_rpm:
        ax = axes[2]
        ax.plot(x, mean_rpm, marker="o", linewidth=1.8, label="Mean RPM")
        if finite_any(min_rpm) and finite_any(max_rpm):
            ax.fill_between(x, min_rpm, max_rpm, alpha=0.18, label="Min–max RPM")
        ax.set_ylabel("RPM")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)

    axes[-1].set_xlabel(xlabel)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    save_figure(fig, out_dir, "eval_reward_success_rpm")
    plt.close(fig)


def group_success_keys(rows: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in rows:
        for k in r:
            if k.startswith("obs_") and k.endswith("_success_rate"):
                keys.add(k)

    def obs_index(key: str) -> int:
        # key format: obs_3_success_rate
        try:
            return int(key.split("_")[1])
        except Exception:
            return 999

    return sorted(keys, key=obs_index)


def plot_group_success(rows: List[Dict[str, Any]], out_dir: Path, title: str, x_units: str) -> None:
    keys = group_success_keys(rows)
    if not keys:
        print("No per-obstacle success fields found; skipping group_success_by_obstacle plot.")
        return

    x, xlabel = x_values(rows, x_units)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for key in keys:
        label = key.replace("_success_rate", "").replace("obs_", "obs ")
        ax.plot(x, arr(rows, key, 100.0), marker="o", linewidth=1.6, label=label)

    overall = arr(rows, "success_rate", 100.0)
    if finite_any(overall):
        ax.plot(x, overall, marker="o", linewidth=2.2, linestyle="--", label="overall")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, ncols=2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, out_dir, "group_success_by_obstacle")
    plt.close(fig)


def read_monitor_csv(path: Path) -> List[Dict[str, float]]:
    # Stable-Baselines monitor files often start with one JSON comment line beginning with '#'.
    text_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            text_lines.append(line)
    if not text_lines:
        return []

    reader = csv.DictReader(text_lines)
    rows = []
    for r in reader:
        try:
            reward = float(r.get("r", r.get("reward", "nan")))
            length = float(r.get("l", r.get("length", "nan")))
            wall_time = float(r.get("t", r.get("time", "nan")))
        except ValueError:
            continue
        if math.isfinite(reward) and math.isfinite(length):
            rows.append({"reward": reward, "length": length, "time": wall_time})
    return rows


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    out = np.empty_like(values, dtype=float)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        out[i] = (cumsum[i + 1] - cumsum[start]) / (i + 1 - start)
    return out


def plot_monitor_reward(monitor_path: Optional[Path], out_dir: Path, title: str, x_units: str, rolling_window: int) -> None:
    if monitor_path is None:
        return
    if not monitor_path.exists():
        raise FileNotFoundError(monitor_path)

    rows = read_monitor_csv(monitor_path)
    if not rows:
        print(f"No monitor rows found in {monitor_path}; skipping monitor plot.")
        return

    rewards = np.asarray([r["reward"] for r in rows], dtype=float)
    lengths = np.asarray([r["length"] for r in rows], dtype=float)
    steps = np.cumsum(lengths)
    if x_units == "steps":
        x = steps
        xlabel = "Training timesteps (approx. from episode lengths)"
    elif x_units == "k":
        x = steps / 1_000.0
        xlabel = "Training timesteps (k, approx. from episode lengths)"
    else:
        x = steps / 1_000_000.0
        xlabel = "Training timesteps (million, approx. from episode lengths)"

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, rewards, linewidth=0.7, alpha=0.35, label="Episode reward")
    ax.plot(x, rolling_mean(rewards, rolling_window), linewidth=2.0, label=f"Rolling mean ({rolling_window} episodes)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training episode reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, out_dir, "train_monitor_reward")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ASV SAC training/evaluation curves for paper figures.")
    parser.add_argument("--eval-summary", type=Path, required=True, help="Path to eval_summary.json from train_test_asv.py.")
    parser.add_argument("--monitor", type=Path, default=None, help="Optional SB3 train_monitor.csv / monitor.csv.")
    parser.add_argument("--out-dir", type=Path, default=Path("paper_plots"), help="Output directory for PNG/SVG figures.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument("--x-units", choices=["steps", "k", "m"], default="m", help="X-axis units.")
    parser.add_argument("--rolling-window", type=int, default=50, help="Rolling window for monitor reward plot.")
    parser.add_argument(
        "--only-reward",
        action="store_true",
        help="Generate only the mean evaluation reward plot from eval_summary.json.",
    )
    parser.add_argument(
        "--plot-set",
        choices=["all", "reward", "main", "group", "monitor"],
        default="all",
        help=(
            "Which plots to generate: 'reward' creates only eval_mean_reward; "
            "'main' creates the 3-panel reward/success/RPM figure; "
            "'group' creates per-obstacle success; 'monitor' creates only monitor reward; "
            "'all' keeps the old behavior. --only-reward is equivalent to --plot-set reward."
        ),
    )
    args = parser.parse_args()

    rows = load_json_rows(args.eval_summary)
    plot_set = "reward" if args.only_reward else args.plot_set

    if plot_set == "reward":
        plot_eval_reward_only(rows, args.out_dir, args.title, args.x_units)
    elif plot_set == "main":
        plot_eval_reward_success_rpm(rows, args.out_dir, args.title, args.x_units)
    elif plot_set == "group":
        plot_group_success(rows, args.out_dir, "Per-obstacle-count evaluation success", args.x_units)
    elif plot_set == "monitor":
        plot_monitor_reward(args.monitor, args.out_dir, "Raw training episode reward", args.x_units, args.rolling_window)
    else:
        plot_eval_reward_success_rpm(rows, args.out_dir, args.title, args.x_units)
        plot_group_success(rows, args.out_dir, "Per-obstacle-count evaluation success", args.x_units)
        plot_monitor_reward(args.monitor, args.out_dir, "Raw training episode reward", args.x_units, args.rolling_window)


if __name__ == "__main__":
    main()
