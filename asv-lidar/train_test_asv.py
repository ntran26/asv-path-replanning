import os
import csv
import json
import argparse
import multiprocessing
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from asv_lidar_rudder_speed_control import ASVLidarEnv, RPM_MAX, RPM_MIN

"""
Train:
  python train_test_asv.py --mode train --algo sac --timesteps 1000000

Test (render):
  python train_test_asv.py --mode test --algo sac

Optional:
  --num-envs 8 --eval-freq 50000 --n-eval-episodes 5 --save-freq 500000
"""


# -------------------------------
# Action decoding helpers (for logging)
# -------------------------------
def action_to_rpm(throttle_cmd: float) -> float:
    """Map normalized throttle [-1,1] to rpm [RPM_MIN, RPM_MAX]."""
    throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))
    return float(RPM_MIN + (throttle_cmd + 1.0) * 0.5 * (RPM_MAX - RPM_MIN))


def action_to_rudder_deg(rudder_cmd: float) -> float:
    """Map normalized rudder [-1,1] to degrees (same mapping as env.step)."""
    rudder_cmd = float(np.clip(rudder_cmd, -1.0, 1.0))
    return float(rudder_cmd * 25.0)


# -------------------------------
# Domain-specific evaluation helpers
# -------------------------------
def lidar_clearance_stats(env: ASVLidarEnv) -> Dict[str, float]:
    """
    Clearance stats from lidar beams (handles invalid/0 by treating as out-of-range).

    Returns:
      min_lidar_all: minimum finite beam distance across all beams
      p10_front: 10th percentile of the +/-45deg "front" beams (worst-ish front clearance)
      p50_front: median of the +/-45deg "front" beams
    """
    out = {"min_lidar_all": float("inf"), "p10_front": float("inf"), "p50_front": float("inf")}

    if not (hasattr(env, "lidar") and hasattr(env.lidar, "ranges") and hasattr(env.lidar, "angles")):
        return out

    r = np.array(env.lidar.ranges, dtype=np.float32)

    # If your lidar outputs 0 when out of range (<1m or >16m), treat 0 as invalid
    r[r <= 0.0] = np.inf

    finite = r[np.isfinite(r)]
    if finite.size > 0:
        out["min_lidar_all"] = float(np.min(finite))

    ang = np.array(env.lidar.angles, dtype=np.float32)
    front_mask = np.abs(ang) <= 45.0
    front = r[front_mask] if np.any(front_mask) else r
    front_finite = front[np.isfinite(front)]
    if front_finite.size > 0:
        out["p10_front"] = float(np.percentile(front_finite, 10))
        out["p50_front"] = float(np.percentile(front_finite, 50))

    return out


def termination_reason(env: ASVLidarEnv, done: bool, hit_max_steps: bool) -> str:
    """
    Infer termination reason without changing the env.

    We avoid hard-coded goal radii here because your project has used multiple unit
    conventions (pixels / dm / meters) over time.

    Returns: goal / obstacle / border / timeout / terminated
    """
    if hit_max_steps:
        return "timeout"

    # Border check (prefer hull polygon if available)
    border_outside = False
    try:
        if hasattr(env, "_hull_polygon_world"):
            hull = env._hull_polygon_world()
            xs = [p[0] for p in hull]
            ys = [p[1] for p in hull]
            border_outside = (min(xs) <= 0 or max(xs) >= env.map_width or min(ys) <= 0 or max(ys) >= env.map_height)
        else:
            border_outside = (env.asv_x <= 0 or env.asv_x >= env.map_width or env.asv_y <= 0 or env.asv_y >= env.map_height)
    except Exception:
        border_outside = False

    if border_outside:
        return "border"

    # Obstacle collision check (geometry)
    collided = False
    if hasattr(env, "_check_collision_geom"):
        try:
            collided = bool(env._check_collision_geom())
        except Exception:
            collided = False

    if collided:
        return "obstacle"

    # If the env terminated but we didn't detect border/collision, treat as goal
    if done:
        return "goal"

    return "terminated" if done else "timeout"



def eval_one_episode(model, env: ASVLidarEnv, deterministic: bool = True, max_steps: int = 5000) -> Dict[str, Any]:
    """
    Run 1 evaluation episode and return episode-level metrics.

    Logs both:
      - task metrics (progress, speed, clearance, etc.)
      - reward component breakdown (r_pf, r_oa, r_exist, cos_chi, lambda) from env.info
    """
    obs, _ = env.reset()
    done = False
    terminated = False
    truncated = False

    ep_reward = 0.0
    step_count = 0

    # Episode-level lists for stats
    speed_list: List[float] = []
    rpm_list: List[float] = []
    rudder_deg_list: List[float] = []
    tgt_list: List[float] = []
    angle_diff_list: List[float] = []

    min_lidar_list: List[float] = []
    p10_front_list: List[float] = []

    # Reward component breakdown (from env.info)
    lam_list: List[float] = []
    r_pf_list: List[float] = []
    r_oa_list: List[float] = []
    r_exist_list: List[float] = []
    r_heading_list: List[float] = []  # cos_chi / r_heading
    collided_steps = 0

    d_start = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))

    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.array(action, dtype=np.float32).reshape(-1)

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        ep_reward += float(reward)
        step_count += 1

        # Speed: pose-based speed (env.speed_mps should match obs["speed"][0])
        if isinstance(obs, dict) and "speed" in obs:
            spd = float(obs["speed"][0])
        else:
            spd = float(getattr(env, "speed_mps", 0.0))
        speed_list.append(spd)

        # Decode commands for logging
        rudder_deg_list.append(action_to_rudder_deg(float(action[0])))
        rpm_list.append(action_to_rpm(float(action[1])))

        # Task-state stats
        tgt_list.append(float(getattr(env, "tgt", 0.0)))
        angle_diff_list.append(float(getattr(env, "angle_diff", 0.0)))

        # Lidar clearance stats
        cs = lidar_clearance_stats(env)
        min_lidar_list.append(cs["min_lidar_all"])
        p10_front_list.append(cs["p10_front"])

        # Reward breakdown (if env provides it)
        if isinstance(info, dict):
            # lambda key variants
            for k in ("lam", "lambda", "lambda_"):
                if k in info:
                    lam_list.append(float(info[k]))
                    break

            if "r_pf" in info:
                r_pf_list.append(float(info["r_pf"]))
            if "r_oa" in info:
                r_oa_list.append(float(info["r_oa"]))
            if "r_exist" in info:
                r_exist_list.append(float(info["r_exist"]))
            if "r_heading" in info:
                r_heading_list.append(float(info["r_heading"]))
            if bool(info.get("collision", False)):
                collided_steps += 1

        if done:
            break

    hit_max_steps = (step_count >= max_steps and not done)
    d_end = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))

    prog_total = d_start - d_end
    prog_per_step = prog_total / float(step_count) if step_count > 0 else 0.0

    reason = termination_reason(env, done=done, hit_max_steps=hit_max_steps)
    success = 1 if reason == "goal" else 0

    # Helpers
    def safe_mean(x: List[float]) -> float:
        return float(np.mean(x)) if len(x) else 0.0

    def safe_min(x: List[float]) -> float:
        if not len(x):
            return float("inf")
        return float(np.min(x))

    def safe_max(x: List[float]) -> float:
        return float(np.max(x)) if len(x) else 0.0

    metrics: Dict[str, Any] = {
        "ep_reward": float(ep_reward),
        "ep_len": int(step_count),
        "success": int(success),
        "term_reason": str(reason),
        "d_start": float(d_start),
        "d_end": float(d_end),
        "progress_total": float(prog_total),
        "progress_per_step": float(prog_per_step),

        "mean_speed": safe_mean(speed_list),
        "min_speed": safe_min(speed_list),
        "max_speed": safe_max(speed_list),

        "mean_rpm": safe_mean(rpm_list),
        "min_rpm": safe_min(rpm_list),
        "max_rpm": safe_max(rpm_list),

        "mean_abs_rudder": safe_mean([abs(x) for x in rudder_deg_list]),
        "std_rudder": float(np.std(rudder_deg_list)) if len(rudder_deg_list) else 0.0,

        "mean_abs_tgt": safe_mean([abs(x) for x in tgt_list]),
        "max_abs_tgt": safe_max([abs(x) for x in tgt_list]),

        "mean_abs_angle_diff": safe_mean([abs(x) for x in angle_diff_list]),
        "max_abs_angle_diff": safe_max([abs(x) for x in angle_diff_list]),

        "min_lidar_all": safe_min(min_lidar_list),
        "p10_front": safe_min(p10_front_list),

        # Reward component breakdown (means over the episode)
        "mean_r_pf": safe_mean(r_pf_list),
        "mean_r_oa": safe_mean(r_oa_list),
        "mean_r_exist": safe_mean(r_exist_list),
        "mean_cos_chi": safe_mean(r_heading_list),
        "mean_lambda": safe_mean(lam_list),
        "has_reward_info": int(bool(len(r_pf_list) or len(r_oa_list) or len(r_exist_list) or len(lam_list) or len(r_heading_list))),

        # Sanity: average reward per step and whether collision flag happened at any step
        "reward_per_step": float(ep_reward / float(step_count)) if step_count > 0 else 0.0,
        "collision_steps": int(collided_steps),
    }

    return metrics


# -------------------------------
# Callbacks
# -------------------------------
class EvalMetricsCallback(BaseCallback):
    """
    Periodic evaluation on a single non-vector env.

    Outputs:
      - eval_metrics.csv / eval_metrics.json  (per-episode eval logs)
      - eval_summary.csv / eval_summary.json  (aggregated per-eval interval)

    Also logs aggregated metrics into TensorBoard under the "eval/" namespace.
    """

    def __init__(
        self,
        eval_env: ASVLidarEnv,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 3,
        max_steps: int = 5_000,
        csv_path: str = "eval_metrics.csv",
        json_path: str = "eval_metrics.json",
        summary_csv_path: str = "eval_summary.csv",
        summary_json_path: str = "eval_summary.json",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_steps = int(max_steps)

        self.csv_path = csv_path
        self.json_path = json_path
        self.summary_csv_path = summary_csv_path
        self.summary_json_path = summary_json_path

        self.rows: List[Dict[str, Any]] = []
        self.summary_rows: List[Dict[str, Any]] = []

        self._csv_inited = False
        self._summary_csv_inited = False

        # Per-episode CSV columns
        self.header = [
            "timesteps", "episode",
            "ep_reward", "ep_len", "success", "term_reason",
            "d_start", "d_end", "progress_total", "progress_per_step",
            "mean_speed", "min_speed", "max_speed",
            "mean_rpm", "min_rpm", "max_rpm",
            "mean_abs_rudder", "std_rudder",
            "mean_abs_tgt", "max_abs_tgt",
            "mean_abs_angle_diff", "max_abs_angle_diff",
            "min_lidar_all", "p10_front",
            "mean_r_pf", "mean_r_oa", "mean_r_exist", "mean_cos_chi", "mean_lambda",
            "reward_per_step", "collision_steps", "has_reward_info",
        ]

        # Aggregated summary per eval point
        self.summary_header = [
            "timesteps",
            "mean_ep_reward", "std_ep_reward",
            "mean_ep_len",
            "success_rate",
            "collision_rate", "border_rate", "obstacle_rate", "timeout_rate",
            "mean_progress_per_step",
            "mean_d_end",
            "mean_speed",
            "min_min_lidar_all",
            "min_p10_front",
            "mean_r_pf", "mean_r_oa", "mean_r_exist", "mean_cos_chi", "mean_lambda", "reward_info_rate",
        ]

    def _init_csv(self):
        if self._csv_inited:
            return
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(self.header)
        self._csv_inited = True

    def _init_summary_csv(self):
        if self._summary_csv_inited:
            return
        write_header = not os.path.exists(self.summary_csv_path)
        with open(self.summary_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(self.summary_header)
        self._summary_csv_inited = True

    def _append_row(self, row: List[Any]):
        self._init_csv()
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _append_summary_row(self, row: List[Any]):
        self._init_summary_csv()
        with open(self.summary_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True

        ep_metrics: List[Dict[str, Any]] = []
        for ep_i in range(self.n_eval_episodes):
            m = eval_one_episode(self.model, self.eval_env, deterministic=True, max_steps=self.max_steps)
            ep_metrics.append(m)

            if self.verbose:
                print(
                    f"[EVAL @ {self.num_timesteps}] ep#{ep_i} "
                    f"R={m['ep_reward']:.1f} len={m['ep_len']} "
                    f"succ={m['success']} reason={m['term_reason']} "
                    f"d_end={m['d_end']:.1f} prog/step={m['progress_per_step']:.3f} "
                    f"v_mean={m['mean_speed']:.2f} rpm_mean={m['mean_rpm']:.1f} "
                    f"min_lidar={m['min_lidar_all']:.2f} p10_front={m['p10_front']:.2f} "
                    f"(r_pf={m['mean_r_pf']:.3f}, r_oa={m['mean_r_oa']:.3f}, r_exist={m['mean_r_exist']:.3f})"
                )

            row = [self.num_timesteps, ep_i] + [m.get(k) for k in self.header[2:]]
            self._append_row(row)
            self.rows.append({"timesteps": int(self.num_timesteps), "episode": int(ep_i), **m})

        # Aggregate summary
        def mean_of(key: str) -> float:
            vals = [float(x.get(key, 0.0)) for x in ep_metrics]
            return float(np.mean(vals)) if len(vals) else 0.0

        def std_of(key: str) -> float:
            vals = [float(x.get(key, 0.0)) for x in ep_metrics]
            return float(np.std(vals)) if len(vals) else 0.0

        term_reasons = [str(x.get("term_reason", "")) for x in ep_metrics]
        success_rate = float(np.mean([int(x.get("success", 0)) for x in ep_metrics])) if len(ep_metrics) else 0.0

        collision_rate = float(np.mean([1 if r in ("obstacle", "border") else 0 for r in term_reasons])) if len(term_reasons) else 0.0
        border_rate = float(np.mean([1 if r == "border" else 0 for r in term_reasons])) if len(term_reasons) else 0.0
        obstacle_rate = float(np.mean([1 if r == "obstacle" else 0 for r in term_reasons])) if len(term_reasons) else 0.0
        timeout_rate = float(np.mean([1 if r == "timeout" else 0 for r in term_reasons])) if len(term_reasons) else 0.0

        summary = {
            "timesteps": int(self.num_timesteps),
            "mean_ep_reward": mean_of("ep_reward"),
            "std_ep_reward": std_of("ep_reward"),
            "mean_ep_len": mean_of("ep_len"),
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "border_rate": border_rate,
            "obstacle_rate": obstacle_rate,
            "timeout_rate": timeout_rate,
            "mean_progress_per_step": mean_of("progress_per_step"),
            "mean_d_end": mean_of("d_end"),
            "mean_speed": mean_of("mean_speed"),
            "min_min_lidar_all": float(np.min([float(x.get("min_lidar_all", float("inf"))) for x in ep_metrics])) if len(ep_metrics) else float("inf"),
            "min_p10_front": float(np.min([float(x.get("p10_front", float("inf"))) for x in ep_metrics])) if len(ep_metrics) else float("inf"),
            "mean_r_pf": mean_of("mean_r_pf"),
            "mean_r_oa": mean_of("mean_r_oa"),
            "mean_r_exist": mean_of("mean_r_exist"),
            "mean_cos_chi": mean_of("mean_cos_chi"),
            "mean_lambda": mean_of("mean_lambda"),
            "reward_info_rate": mean_of("has_reward_info"),
        }
        self.summary_rows.append(summary)

        srow = [summary.get(k) for k in self.summary_header]
        self._append_summary_row(srow)

        with open(self.json_path, "w") as f:
            json.dump(self.rows, f, indent=2)
        with open(self.summary_json_path, "w") as f:
            json.dump(self.summary_rows, f, indent=2)

        # TensorBoard logging
        self.logger.record("eval/mean_ep_reward", summary["mean_ep_reward"])
        self.logger.record("eval/std_ep_reward", summary["std_ep_reward"])
        self.logger.record("eval/success_rate", summary["success_rate"])
        self.logger.record("eval/collision_rate", summary["collision_rate"])
        self.logger.record("eval/border_rate", summary["border_rate"])
        self.logger.record("eval/obstacle_rate", summary["obstacle_rate"])
        self.logger.record("eval/timeout_rate", summary["timeout_rate"])
        self.logger.record("eval/mean_progress_per_step", summary["mean_progress_per_step"])
        self.logger.record("eval/mean_d_end", summary["mean_d_end"])
        self.logger.record("eval/mean_speed", summary["mean_speed"])
        self.logger.record("eval/min_min_lidar_all", summary["min_min_lidar_all"])
        self.logger.record("eval/min_p10_front", summary["min_p10_front"])
        self.logger.record("eval/mean_r_pf", summary["mean_r_pf"])
        self.logger.record("eval/mean_r_oa", summary["mean_r_oa"])
        self.logger.record("eval/mean_r_exist", summary["mean_r_exist"])
        self.logger.record("eval/mean_cos_chi", summary["mean_cos_chi"])
        self.logger.record("eval/mean_lambda", summary["mean_lambda"])
        self.logger.record("eval/reward_info_rate", summary["reward_info_rate"])

        return True


# -------------------------------
# Main
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # evaluation / checkpoint
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--n-eval-episodes", type=int, default=3)
    p.add_argument("--eval-max-steps", type=int, default=5_000)
    p.add_argument("--save-freq", type=int, default=500_000)

    # model paths
    p.add_argument("--model-path", type=str, default=None)
    return p.parse_args()


def make_env(seed: int, rank: int):
    """
    Factory for SubprocVecEnv.
    Seeds each env. VecMonitor will collect episode stats for logging.
    """
    def _init():
        env = ASVLidarEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()

    algo = args.algo.lower()
    model_path = args.model_path or f"{algo}_asv_model.zip"

    if args.mode == "train":
        # Vector env for training
        env_fns = [make_env(args.seed, i) for i in range(args.num_envs)]
        vec_env = SubprocVecEnv(env_fns)

        # VecMonitor writes a single monitor file with episode returns/lengths (for training curves)
        vec_env = VecMonitor(vec_env, filename="train_monitor.csv")

        # Eval env (single)
        eval_env = ASVLidarEnv(render_mode=None)
        eval_env.reset(seed=args.seed + 10_000)

        # Hyperparameters (keep your current values; tune here later)
        learn_rate = 1e-4
        batch_size = 512
        gamma = 0.99

        if algo == "ppo":
            n_steps = args.num_envs * 1024
            n_epochs = 10
            gae_lambda = 0.95
            clip_range = 0.2
            ent_coef = 0.01
            vf_coef = 0.5

            model = PPO(
                "MultiInputPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=f"./{algo}_log/",
                learning_rate=learn_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
            )
        elif algo == "sac":
            model = SAC(
                "MultiInputPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=f"./{algo}_log/",
                learning_rate=learn_rate,
                batch_size=batch_size,
                gamma=gamma,
                buffer_size=1_000_000,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
            )
        else:
            raise ValueError(f"Unknown algo: {algo}")

        # Callbacks
        checkpoint_cb = CheckpointCallback(
            save_freq=max(int(args.save_freq // max(args.num_envs, 1)), 1),
            save_path="models",
            name_prefix=f"{algo}_model",
            save_replay_buffer=(algo == "sac"),
            save_vecnormalize=False,
        )

        eval_cb = EvalMetricsCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            max_steps=args.eval_max_steps,
            csv_path="eval_metrics.csv",
            json_path="eval_metrics.json",
            summary_csv_path="eval_summary.csv",
            summary_json_path="eval_summary.json",
            verbose=1,
        )

        callbacks = CallbackList([checkpoint_cb, eval_cb])

        # Train
        model.learn(
            total_timesteps=int(args.timesteps),
            tb_log_name=f"asv_{algo}",
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        model.save(model_path)
        print(f"Saved model -> {model_path}")

        vec_env.close()
        eval_env.close()

    elif args.mode == "test":
        # Import pygame only in test mode (keeps training lighter / headless-friendly)
        import pygame  # noqa: F401

        if algo == "ppo":
            model = PPO.load(model_path)
        elif algo == "sac":
            model = SAC.load(model_path)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        env = ASVLidarEnv(render_mode="human")
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)

        print(f"Test episode completed. Total reward: {total_reward:.2f}")

        # Save data
        result_data = {
            "heading": env.asv_h,
            "start": [env.start_x, env.start_y],
            "goal": [env.goal_x, env.goal_y],
            "obstacles": env.obstacles,
            "path": env.path.tolist() if hasattr(env, "path") else [],
            "asv_path": env.asv_path,
        }

        with open("asv_data.json", "w") as f:
            json.dump(result_data, f, indent=4)
            
        env.close()
