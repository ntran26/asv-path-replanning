"""
Train:
    python train_test_asv.py --mode train --algo ppo --timesteps 1000000

Test (render):
    python train_test_asv.py --mode test --algo ppo

Plot benchmark:
    python train_test_asv.py --mode plot --plot-input benchmark_history.json --plot-dir benchmark_plots

Optional:
  --num-envs 8 --eval-freq 50000 --save-freq 500000
  python train_test_asv.py --mode train --algo ppo --timesteps 500000 --train-stage 1 --eval-freq 50000
"""

import os
import csv
import json
import argparse
import multiprocessing
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from rl_env import ASVLidarEnv, RPM_MAX, RPM_MIN
from ship_model_selector import VESSEL_LENGTH, MAX_RUD_ANGLE

DEFAULT_BENCHMARK_CASES = [0, 1, 2, 3, 4, 5]
DEFAULT_EVAL_FREQ = 50_000
DEFAULT_EVAL_MAX_STEPS = 600
DEFAULT_PLOT_DIR = "benchmark_plots"
DEFAULT_BEST_MODEL_PATH = os.path.join("models", "best_benchmark_model")
DEFAULT_BEST_BENCHMARK_JSON = "best_benchmark_result.json"
DEFAULT_TRAIN_STAGE = 3


def action_to_rpm(throttle_cmd: float) -> float:
    throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))
    return float(RPM_MIN + (throttle_cmd + 1.0) * 0.5 * (RPM_MAX - RPM_MIN))


def action_to_rudder_deg(rudder_cmd: float) -> float:
    rudder_cmd = float(np.clip(rudder_cmd, -1.0, 1.0))
    return float(rudder_cmd * float(MAX_RUD_ANGLE))


def lidar_front_stats(env: ASVLidarEnv) -> Dict[str, float]:
    out = {"min_lidar": float("inf"), "p10_front": float("inf")}
    ranges = np.asarray(env.lidar.ranges, dtype=np.float32)
    angles = np.asarray(env.lidar.angles, dtype=np.float32)

    finite = ranges[np.isfinite(ranges)]
    if finite.size:
        out["min_lidar"] = float(np.min(finite))

    front_mask = np.abs(angles) <= 45.0
    front = ranges[front_mask] if np.any(front_mask) else ranges
    front = front[np.isfinite(front)]
    if front.size:
        out["p10_front"] = float(np.percentile(front, 10))
    return out


def infer_term_reason(env: ASVLidarEnv, last_info: Dict[str, Any], terminated: bool, truncated: bool, hit_max_steps: bool) -> str:
    if hit_max_steps or truncated:
        return "timeout"

    if bool(last_info.get("collided", False)):
        collision_type = str(last_info.get("collision_type", "obstacle") or "obstacle")
        return "border" if collision_type == "border" else "collision"

    if bool(last_info.get("reached_goal", False)) or getattr(env, "distance_to_goal", float("inf")) <= (VESSEL_LENGTH / 2.0):
        return "goal"

    if terminated:
        return "collision"

    return "timeout"


def rollout_episode(actor, env: ASVLidarEnv, case_id: int, max_steps: int, deterministic: bool = True) -> Dict[str, Any]:
    env.test_case = case_id
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    last_info: Dict[str, Any] = {}

    speeds: List[float] = []
    rpms: List[float] = []
    rudders: List[float] = []
    min_lidars: List[float] = []
    front_p10s: List[float] = []
    front_mins: List[float] = []
    near_flags: List[float] = []
    r_pfs: List[float] = []
    r_oas: List[float] = []
    r_exists: List[float] = []
    lambdas: List[float] = []
    abs_tgts: List[float] = []
    abs_heading_errors: List[float] = []
    distances_to_goal: List[float] = []

    while steps < max_steps:
        action, _ = actor.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        total_reward += float(reward)
        steps += 1

        speeds.append(float(info.get("speed_mps", 0.0)))
        rpms.append(action_to_rpm(float(action[1])))
        rudders.append(action_to_rudder_deg(float(action[0])))
        r_pfs.append(float(info.get("r_pf", 0.0)))
        r_oas.append(float(info.get("r_oa", 0.0)))
        r_exists.append(float(info.get("r_exist", 0.0)))
        lambdas.append(float(info.get("lambda_reward", 0.0)))
        abs_tgts.append(abs(float(info.get("cross_track_error", 0.0))))
        abs_heading_errors.append(abs(float(info.get("heading_error", 0.0))))
        front_mins.append(float(info.get("front_min", np.inf)))
        front_p10s.append(float(info.get("front_p10", np.inf)))
        near_flags.append(float(info.get("near_flag", 0.0)))
        distances_to_goal.append(float(info.get("distance_to_goal", np.inf)))

        ls = lidar_front_stats(env)
        min_lidars.append(ls["min_lidar"])

        if terminated or truncated:
            break

    hit_max_steps = steps >= max_steps and not (terminated or truncated)
    term_reason = infer_term_reason(env, last_info, terminated, truncated, hit_max_steps)

    return {
        "case": int(case_id),
        "ep_reward": float(total_reward),
        "ep_len": int(steps),
        "term_reason": term_reason,
        "success": int(term_reason == "goal"),
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_rpm": float(np.mean(rpms)) if rpms else 0.0,
        "mean_abs_rudder": float(np.mean(np.abs(rudders))) if rudders else 0.0,
        "min_lidar": float(np.min(min_lidars)) if min_lidars else float("inf"),
        "p10_front": float(np.min(front_p10s)) if front_p10s else float("inf"),
        "front_min": float(np.min(front_mins)) if front_mins else float("inf"),
        "mean_distance_to_goal": float(np.mean(distances_to_goal)) if distances_to_goal else float("inf"),
        "mean_near_flag": float(np.mean(near_flags)) if near_flags else 0.0,
        "d_end": float(getattr(env, "distance_to_goal", float("inf"))),
        "start": [float(env.start_x), float(env.start_y)],
        "goal": [float(env.goal_x), float(env.goal_y)],
        "mean_r_pf": float(np.mean(r_pfs)) if r_pfs else 0.0,
        "mean_r_oa": float(np.mean(r_oas)) if r_oas else 0.0,
        "mean_r_exist": float(np.mean(r_exists)) if r_exists else 0.0,
        "mean_lambda": float(np.mean(lambdas)) if lambdas else 0.0,
        "mean_abs_tgt": float(np.mean(abs_tgts)) if abs_tgts else 0.0,
        "mean_abs_heading_error": float(np.mean(abs_heading_errors)) if abs_heading_errors else 0.0,
        "final_x": float(env.asv_x),
        "final_y": float(env.asv_y),
        "final_heading": float(env.asv_h),
    }


def evaluate_benchmark(actor, env: ASVLidarEnv, cases: List[int], max_steps: int) -> Dict[str, Any]:
    rows = [rollout_episode(actor, env, case_id=case, max_steps=max_steps) for case in cases]
    term_reasons = [row["term_reason"] for row in rows]
    summary = {
        "n_cases": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "mean_reward": float(np.mean([row["ep_reward"] for row in rows])) if rows else 0.0,
        "mean_ep_len": float(np.mean([row["ep_len"] for row in rows])) if rows else 0.0,
        "mean_speed": float(np.mean([row["mean_speed"] for row in rows])) if rows else 0.0,
        "mean_distance_to_goal": float(np.mean([row["mean_distance_to_goal"] for row in rows])) if rows else float("inf"),
        "min_front_min": float(np.min([row["front_min"] for row in rows])) if rows else float("inf"),
        "min_p10_front": float(np.min([row["p10_front"] for row in rows])) if rows else float("inf"),
        "min_lidar": float(np.min([row["min_lidar"] for row in rows])) if rows else float("inf"),
        "goal_rate": float(np.mean([r == "goal" for r in term_reasons])) if rows else 0.0,
        "collision_rate": float(np.mean([r == "collision" for r in term_reasons])) if rows else 0.0,
        "obstacle_rate": float(np.mean([r == "collision" for r in term_reasons])) if rows else 0.0,
        "border_rate": float(np.mean([r == "border" for r in term_reasons])) if rows else 0.0,
        "timeout_rate": float(np.mean([r == "timeout" for r in term_reasons])) if rows else 0.0,
        "mean_r_pf": float(np.mean([row["mean_r_pf"] for row in rows])) if rows else 0.0,
        "mean_r_oa": float(np.mean([row["mean_r_oa"] for row in rows])) if rows else 0.0,
        "mean_r_exist": float(np.mean([row["mean_r_exist"] for row in rows])) if rows else 0.0,
        "mean_lambda": float(np.mean([row["mean_lambda"] for row in rows])) if rows else 0.0,
        "mean_abs_tgt": float(np.mean([row["mean_abs_tgt"] for row in rows])) if rows else 0.0,
        "mean_abs_heading_error": float(np.mean([row["mean_abs_heading_error"] for row in rows])) if rows else 0.0,
        "mean_near_flag": float(np.mean([row["mean_near_flag"] for row in rows])) if rows else 0.0,
    }
    return {"rows": rows, "summary": summary}


def _save_plot(fig, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_history(history_path: str, output_dir: str) -> List[str]:
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Benchmark history not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise ValueError(f"Benchmark history is empty: {history_path}")

    os.makedirs(output_dir, exist_ok=True)

    timesteps = [int(entry["timesteps"]) for entry in history]
    summaries = [entry["summary"] for entry in history]
    latest_rows = history[-1].get("rows", [])

    def series(key: str, default=0.0):
        values = []
        for summary in summaries:
            values.append(float(summary.get(key, default)))
        return values

    saved_paths: List[str] = []

    # 1. Success / failure rates
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, series("success_rate"), marker="o", linewidth=2, label="success")
    ax.plot(timesteps, series("goal_rate"), marker="o", linewidth=2, label="goal")
    ax.plot(timesteps, series("collision_rate", default=np.nan), marker="o", linewidth=2, label="collision")
    ax.plot(timesteps, series("timeout_rate"), marker="o", linewidth=2, label="timeout")
    ax.set_title("Benchmark Rates vs Timesteps")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    rates_path = os.path.join(output_dir, "benchmark_rates.png")
    _save_plot(fig, rates_path)
    saved_paths.append(rates_path)

    # 2. Reward / episode length / speed
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(timesteps, series("mean_reward"), marker="o", linewidth=2, color="tab:blue")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Benchmark Performance")

    axes[1].plot(timesteps, series("mean_ep_len"), marker="o", linewidth=2, color="tab:orange")
    axes[1].set_ylabel("Mean Episode Length")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(timesteps, series("mean_speed"), marker="o", linewidth=2, color="tab:green")
    axes[2].set_ylabel("Mean Speed (m/s)")
    axes[2].set_xlabel("Timesteps")
    axes[2].grid(True, alpha=0.3)

    perf_path = os.path.join(output_dir, "benchmark_performance.png")
    _save_plot(fig, perf_path)
    saved_paths.append(perf_path)

    # 3. Reward components / lambda
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(timesteps, series("mean_r_pf"), marker="o", linewidth=2, label="mean_r_pf")
    axes[0].plot(timesteps, series("mean_r_oa"), marker="o", linewidth=2, label="mean_r_oa")
    axes[0].plot(timesteps, series("mean_r_exist"), marker="o", linewidth=2, label="mean_r_exist")
    axes[0].set_ylabel("Reward Term")
    axes[0].set_title("Reward Components")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(timesteps, series("mean_lambda"), marker="o", linewidth=2, color="tab:purple", label="mean_lambda")
    axes[1].plot(timesteps, series("mean_abs_tgt"), marker="o", linewidth=2, color="tab:red", label="mean_abs_tgt")
    axes[1].plot(timesteps, series("mean_abs_heading_error"), marker="o", linewidth=2, color="tab:brown", label="mean_abs_heading_error")
    axes[1].set_ylabel("Weight / Gate")
    axes[1].set_xlabel("Timesteps")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    reward_path = os.path.join(output_dir, "benchmark_reward_terms.png")
    _save_plot(fig, reward_path)
    saved_paths.append(reward_path)

    # 4. Tracking / progress / lidar safety metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    flat_axes = axes.flatten()
    plot_specs = [
        ("mean_abs_tgt", "Mean |Cross-Track Error|"),
        ("mean_abs_heading_error", "Mean |Heading Error|"),
        ("mean_distance_to_goal", "Mean Distance To Goal"),
        ("min_front_min", "Min Front Clearance"),
    ]
    for ax, (key, title) in zip(flat_axes, plot_specs):
        ax.plot(timesteps, series(key), marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

    tracking_path = os.path.join(output_dir, "benchmark_tracking.png")
    _save_plot(fig, tracking_path)
    saved_paths.append(tracking_path)

    # 5. Latest checkpoint per-case outcome bar chart
    if latest_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        case_ids = [int(row["case"]) for row in latest_rows]
        rewards = [float(row["ep_reward"]) for row in latest_rows]
        reason_colors = {
            "goal": "tab:green",
            "collision": "tab:red",
            "border": "tab:orange",
            "timeout": "tab:gray",
        }
        colors = [reason_colors.get(row.get("term_reason", ""), "tab:blue") for row in latest_rows]
        ax.bar(case_ids, rewards, color=colors)
        for case_id, row in zip(case_ids, latest_rows):
            ax.text(case_id, rewards[case_ids.index(case_id)], row["term_reason"], ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_title(f"Latest Checkpoint Per-Case Reward ({timesteps[-1]} steps)")
        ax.set_xlabel("Case")
        ax.set_ylabel("Episode Reward")
        ax.grid(True, axis="y", alpha=0.3)
        latest_path = os.path.join(output_dir, "benchmark_latest_cases.png")
        _save_plot(fig, latest_path)
        saved_paths.append(latest_path)

    manifest_path = os.path.join(output_dir, "plot_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history_path": history_path,
                "latest_timesteps": timesteps[-1],
                "files": saved_paths,
            },
            f,
            indent=2,
        )
    saved_paths.append(manifest_path)

    return saved_paths


class FixedBenchmarkCallback(BaseCallback):
    def __init__(
        self,
        eval_env: ASVLidarEnv,
        cases: List[int],
        eval_freq: int = DEFAULT_EVAL_FREQ,
        max_steps: int = DEFAULT_EVAL_MAX_STEPS,
        out_json: str = "benchmark_history.json",
        out_csv: str = "benchmark_summary.csv",
        best_model_path: str = DEFAULT_BEST_MODEL_PATH,
        best_json: str = DEFAULT_BEST_BENCHMARK_JSON,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.cases = list(cases)
        self.eval_freq = int(eval_freq)
        self.max_steps = int(max_steps)
        self.out_json = out_json
        self.out_csv = out_csv
        self.best_model_path = best_model_path
        self.best_json = best_json
        self.history: List[Dict[str, Any]] = []
        self._csv_initialized = False
        self.best_metric: Tuple[float, ...] | None = None
        self.best_summary: Dict[str, Any] | None = None

    def _metric_tuple(self, summary: Dict[str, Any]) -> Tuple[float, ...]:
        return (
            float(summary["success_rate"]),
            float(summary["goal_rate"]),
            -float(summary["collision_rate"]),
            -float(summary["border_rate"]),
            -float(summary["timeout_rate"]),
            float(summary["mean_reward"]),
        )

    def _save_best_model(self, result: Dict[str, Any], summary: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.best_model_path) or ".", exist_ok=True)
        self.model.save(self.best_model_path)
        payload = {
            "timesteps": int(self.num_timesteps),
            "metric": list(self.best_metric) if self.best_metric is not None else None,
            "summary": summary,
            "rows": result["rows"],
            "model_path": f"{self.best_model_path}.zip",
        }
        with open(self.best_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _init_csv(self) -> None:
        if self._csv_initialized:
            return
        write_header = not os.path.exists(self.out_csv)
        with open(self.out_csv, "a", newline="") as f:
            if write_header:
                csv.writer(f).writerow(
                    [
                        "timesteps",
                        "n_cases",
                        "success_rate",
                        "mean_reward",
                        "mean_ep_len",
                        "mean_speed",
                        "mean_distance_to_goal",
                        "min_front_min",
                        "min_p10_front",
                        "min_lidar",
                        "goal_rate",
                        "collision_rate",
                        "obstacle_rate",
                        "border_rate",
                        "timeout_rate",
                        "mean_r_pf",
                        "mean_r_oa",
                        "mean_r_exist",
                        "mean_lambda",
                        "mean_abs_tgt",
                        "mean_abs_heading_error",
                        "mean_near_flag",
                    ]
                )
        self._csv_initialized = True

    def _append_csv(self, row: Dict[str, Any]) -> None:
        self._init_csv()
        with open(self.out_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    row["timesteps"],
                    row["n_cases"],
                    row["success_rate"],
                    row["mean_reward"],
                    row["mean_ep_len"],
                    row["mean_speed"],
                    row["mean_distance_to_goal"],
                    row["min_front_min"],
                    row["min_p10_front"],
                    row["min_lidar"],
                    row["goal_rate"],
                    row["collision_rate"],
                    row["obstacle_rate"],
                    row["border_rate"],
                    row["timeout_rate"],
                    row["mean_r_pf"],
                    row["mean_r_oa"],
                    row["mean_r_exist"],
                    row["mean_lambda"],
                    row["mean_abs_tgt"],
                    row["mean_abs_heading_error"],
                    row["mean_near_flag"],
                ]
            )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        result = evaluate_benchmark(self.model, self.eval_env, self.cases, self.max_steps)
        summary = {"timesteps": int(self.num_timesteps), **result["summary"]}
        self.history.append({"timesteps": int(self.num_timesteps), **result})

        with open(self.out_json, "w") as f:
            json.dump(self.history, f, indent=2)
        self._append_csv(summary)

        self.logger.record("benchmark/success_rate", summary["success_rate"])
        self.logger.record("benchmark/mean_reward", summary["mean_reward"])
        self.logger.record("benchmark/collision_rate", summary["collision_rate"])
        self.logger.record("benchmark/obstacle_rate", summary["obstacle_rate"])
        self.logger.record("benchmark/border_rate", summary["border_rate"])
        self.logger.record("benchmark/timeout_rate", summary["timeout_rate"])
        self.logger.record("benchmark/mean_distance_to_goal", summary["mean_distance_to_goal"])
        self.logger.record("benchmark/min_front_min", summary["min_front_min"])
        self.logger.record("benchmark/min_p10_front", summary["min_p10_front"])
        self.logger.record("benchmark/mean_r_pf", summary["mean_r_pf"])
        self.logger.record("benchmark/mean_r_oa", summary["mean_r_oa"])
        self.logger.record("benchmark/mean_r_exist", summary["mean_r_exist"])
        self.logger.record("benchmark/mean_lambda", summary["mean_lambda"])
        self.logger.record("benchmark/mean_abs_tgt", summary["mean_abs_tgt"])
        self.logger.record("benchmark/mean_abs_heading_error", summary["mean_abs_heading_error"])
        self.logger.record("benchmark/mean_near_flag", summary["mean_near_flag"])

        metric = self._metric_tuple(summary)
        improved = self.best_metric is None or metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.best_summary = summary
            self._save_best_model(result, summary)
            if self.verbose:
                print(
                    f"[BEST @ {self.num_timesteps}] "
                    f"success={summary['success_rate']:.2f} "
                    f"goal={summary['goal_rate']:.2f} "
                    f"collision={summary['collision_rate']:.2f} "
                    f"border={summary['border_rate']:.2f}"
                )

        self.logger.record(
            "benchmark/best_success_rate",
            0.0 if self.best_summary is None else float(self.best_summary["success_rate"]),
        )

        if self.verbose:
            print(
                f"[BENCHMARK @ {self.num_timesteps}] "
                f"success={summary['success_rate']:.2f} "
                f"reward={summary['mean_reward']:.2f} "
                f"collision={summary['collision_rate']:.2f} "
                f"border={summary['border_rate']:.2f} "
                f"timeout={summary['timeout_rate']:.2f}"
            )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "eval", "plot"], default="test")
    parser.add_argument("--algo", choices=["ppo"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--test-case", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--eval-max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS)
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--benchmark-cases", type=int, nargs="*", default=DEFAULT_BENCHMARK_CASES)
    parser.add_argument("--plot-input", type=str, default="benchmark_history.json")
    parser.add_argument("--plot-dir", type=str, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--best-model-path", type=str, default=DEFAULT_BEST_MODEL_PATH)
    parser.add_argument("--best-benchmark-json", type=str, default=DEFAULT_BEST_BENCHMARK_JSON)
    parser.add_argument("--train-stage", type=int, choices=[1, 2, 3], default=DEFAULT_TRAIN_STAGE)
    return parser.parse_args()


def make_train_env(seed: int, rank: int, train_stage: int):
    def _init():
        env = ASVLidarEnv(render_mode=None, train_stage=train_stage)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_model(env):
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh)
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_log/",
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
    )


def load_model(model_path: str):
    return PPO.load(model_path)


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()
    model_path = args.model_path or "ppo_asv_model.zip"

    if args.mode == "train":
        print(f"Training with train_stage={args.train_stage}")
        env_fns = [make_train_env(args.seed, i, args.train_stage) for i in range(args.num_envs)]
        vec_env = VecMonitor(SubprocVecEnv(env_fns), filename="train_monitor.csv")
        model = build_model(vec_env)

        eval_env = ASVLidarEnv(render_mode=None, train_stage=args.train_stage)
        eval_env.reset(seed=args.seed + 10_000)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(int(args.save_freq // max(args.num_envs, 1)), 1),
            save_path="models",
            name_prefix="ppo_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        benchmark_cb = FixedBenchmarkCallback(
            eval_env=eval_env,
            cases=args.benchmark_cases,
            eval_freq=args.eval_freq,
            max_steps=args.eval_max_steps,
            out_json="benchmark_history.json",
            out_csv="benchmark_summary.csv",
            best_model_path=args.best_model_path,
            best_json=args.best_benchmark_json,
            verbose=1,
        )

        model.learn(
            total_timesteps=int(args.timesteps),
            tb_log_name="asv_ppo",
            callback=CallbackList([checkpoint_cb, benchmark_cb]),
            progress_bar=True,
        )
        model.save(model_path)
        print(f"Saved model -> {model_path}")
        try:
            saved = plot_benchmark_history("benchmark_history.json", args.plot_dir)
            print("Saved benchmark plots:")
            for path in saved:
                print(f"  {path}")
        except Exception as exc:
            print(f"Plot export skipped: {exc}")

        vec_env.close()
        eval_env.close()
        return

    if args.mode == "test":
        model = load_model(model_path)
        env = ASVLidarEnv(render_mode="human", train_stage=args.train_stage)
        env.test_case = args.test_case

        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
            print(f"action={np.asarray(action).round(3).tolist()} reward={reward:.3f}")

        print(f"Test case {args.test_case} completed. Total reward: {total_reward:.2f}")

        result = {
            "test_case": int(args.test_case) if args.test_case is not None else -1,
            "heading": float(env.asv_h),
            "start": [float(env.start_x), float(env.start_y)],
            "goal": [float(env.goal_x), float(env.goal_y)],
            "obstacles": env.obstacles,
            "path": env.path.tolist() if hasattr(env.path, "tolist") else env.path,
            "asv_path": env.asv_path,
        }
        with open("asv_data.json", "w") as f:
            json.dump(result, f, indent=2)

        env.close()
        return

    if args.mode == "eval":
        model = load_model(model_path)
        eval_env = ASVLidarEnv(render_mode=None, train_stage=args.train_stage)
        eval_env.reset(seed=args.seed + 10_000)
        result = evaluate_benchmark(model, eval_env, args.benchmark_cases, args.eval_max_steps)

        print("Benchmark summary:")
        for k, v in result["summary"].items():
            print(f"  {k}: {v}")

        for row in result["rows"]:
            print(
                f"case={row['case']} reward={row['ep_reward']:.2f} len={row['ep_len']} "
                f"term={row['term_reason']} p10_front={row['p10_front']:.2f}"
            )

        with open("benchmark_eval.json", "w") as f:
            json.dump(result, f, indent=2)

        try:
            saved = plot_benchmark_history(args.plot_input, args.plot_dir)
            print("Saved benchmark plots:")
            for path in saved:
                print(f"  {path}")
        except Exception as exc:
            print(f"Plot export skipped: {exc}")

        eval_env.close()
        return

    if args.mode == "plot":
        saved = plot_benchmark_history(args.plot_input, args.plot_dir)
        print("Saved benchmark plots:")
        for path in saved:
            print(f"  {path}")
        return

if __name__ == "__main__":
    main()
