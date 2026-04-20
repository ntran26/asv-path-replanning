from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from rl_env import ASVLidarEnv, DEFAULT_EVAL_LAMBDA, RPM_MAX, RPM_MIN

# case 0 = centered no-obstacle path following, case 1 = centered single obstacle.
DEFAULT_EVAL_CASES = [0, 1, 2, 3, 6, 7]

def action_to_rpm(throttle_cmd: float) -> float:
    throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))
    return float(RPM_MIN + (throttle_cmd + 1.0) * 0.5 * (RPM_MAX - RPM_MIN))

def action_to_rudder_deg(rudder_cmd: float) -> float:
    rudder_cmd = float(np.clip(rudder_cmd, -1.0, 1.0))
    return float(rudder_cmd * 40.0)

def lidar_clearance_stats(env: ASVLidarEnv) -> Dict[str, float]:
    out = {"min_lidar_all": float("inf"), "p10_front": float("inf"), "p50_front": float("inf")}
    if not (hasattr(env, "lidar") and hasattr(env.lidar, "ranges") and hasattr(env.lidar, "angles")):
        return out

    r = np.array(env.lidar.ranges, dtype=np.float32)
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
    if hit_max_steps:
        return "timeout"
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

    collided = False
    if hasattr(env, "_check_collision_geom"):
        try:
            collided = bool(env._check_collision_geom())
        except Exception:
            collided = False
    if collided:
        return "obstacle"
    if done:
        return "goal"
    return "terminated" if done else "timeout"

def eval_one_episode(model, env: ASVLidarEnv, deterministic: bool = True, max_steps: int = 5000) -> Dict[str, Any]:
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    step_count = 0

    speed_list: List[float] = []
    u_list: List[float] = []
    v_list: List[float] = []
    rpm_list: List[float] = []
    rudder_deg_list: List[float] = []
    cte_list: List[float] = []
    course_error_list: List[float] = []
    lookahead_error_list: List[float] = []
    min_lidar_list: List[float] = []
    p10_front_list: List[float] = []
    lam_list: List[float] = []
    r_pf_list: List[float] = []
    r_oa_list: List[float] = []
    r_exist_list: List[float] = []
    min_sector_range_list: List[float] = []
    p10_sector_range_list: List[float] = []
    mean_sector_pen_list: List[float] = []
    collided_steps = 0

    d_start = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))

    last_info = {}
    last_truncated = False

    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.array(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        last_info = info if isinstance(info, dict) else {}
        last_truncated = bool(truncated)
        ep_reward += float(reward)
        step_count += 1

        speed_list.append(float(getattr(env, "speed_mps", 0.0)))
        u_list.append(float(getattr(env, "u_body", 0.0)))
        v_list.append(float(getattr(env, "v_body", 0.0)))
        rpm_list.append(action_to_rpm(float(action[1])))
        rudder_deg_list.append(action_to_rudder_deg(float(action[0])))
        cte_list.append(float(getattr(env, "cross_track_error", 0.0)))
        course_error_list.append(float(getattr(env, "course_error", 0.0)))
        lookahead_error_list.append(float(getattr(env, "lookahead_course_error", 0.0)))

        cs = lidar_clearance_stats(env)
        min_lidar_list.append(cs["min_lidar_all"])
        p10_front_list.append(cs["p10_front"])

        if isinstance(info, dict):
            if "lam" in info:
                lam_list.append(float(info["lam"]))
            if "r_pf" in info:
                r_pf_list.append(float(info["r_pf"]))
            if "r_oa" in info:
                r_oa_list.append(float(info["r_oa"]))
            if "r_exist" in info:
                r_exist_list.append(float(info["r_exist"]))
            if "min_sector_range" in info:
                min_sector_range_list.append(float(info["min_sector_range"]))
            if "p10_sector_range" in info:
                p10_sector_range_list.append(float(info["p10_sector_range"]))
            if "mean_sector_pen" in info:
                mean_sector_pen_list.append(float(info["mean_sector_pen"]))
            if bool(info.get("collided", False)):
                collided_steps += 1
            if bool(info.get("timeout", False)):
                reason = "timeout"

        if done:
            break

    hit_max_steps = (step_count >= max_steps and not done)
    d_end = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))
    prog_total = d_start - d_end
    prog_per_step = prog_total / float(step_count) if step_count > 0 else 0.0
    reached_goal = bool(last_info.get("reached_goal", False))
    env_timeout = bool(last_info.get("timeout", False)) or bool(last_truncated)

    if reached_goal:
        reason = "goal"
    elif env_timeout or hit_max_steps:
        reason = "timeout"
    else:
        reason = termination_reason(env, done=done, hit_max_steps=hit_max_steps)

    success = 1 if reached_goal else 0

    def safe_mean(x: Sequence[float]) -> float:
        return float(np.mean(x)) if len(x) else 0.0

    def safe_min(x: Sequence[float]) -> float:
        return float(np.min(x)) if len(x) else float("inf")

    def safe_max(x: Sequence[float]) -> float:
        return float(np.max(x)) if len(x) else 0.0

    return {
        "ep_reward": float(ep_reward),
        "ep_len": int(step_count),
        "success": int(success),
        "term_reason": str(reason),
        "d_start": float(d_start),
        "d_end": float(d_end),
        "progress_total": float(prog_total),
        "progress_per_step": float(prog_per_step),
        "mean_speed": safe_mean(speed_list),
        "mean_u": safe_mean(u_list),
        "mean_v": safe_mean(v_list),
        "mean_rpm": safe_mean(rpm_list),
        "min_rpm": safe_min(rpm_list),
        "max_rpm": safe_max(rpm_list),
        "mean_abs_rudder": safe_mean([abs(x) for x in rudder_deg_list]),
        "std_rudder": float(np.std(rudder_deg_list)) if len(rudder_deg_list) else 0.0,
        "mean_abs_cte": safe_mean([abs(x) for x in cte_list]),
        "max_abs_cte": safe_max([abs(x) for x in cte_list]),
        "mean_abs_course_error": safe_mean([abs(x) for x in course_error_list]),
        "max_abs_course_error": safe_max([abs(x) for x in course_error_list]),
        "mean_abs_lookahead_error": safe_mean([abs(x) for x in lookahead_error_list]),
        "max_abs_lookahead_error": safe_max([abs(x) for x in lookahead_error_list]),
        "min_lidar_all": safe_min(min_lidar_list),
        "p10_front": safe_min(p10_front_list),
        "mean_r_pf": safe_mean(r_pf_list),
        "mean_r_oa": safe_mean(r_oa_list),
        "mean_r_exist": safe_mean(r_exist_list),
        "mean_lambda": safe_mean(lam_list),
        "min_sector_range": safe_min(min_sector_range_list),
        "p10_sector_range": safe_min(p10_sector_range_list),
        "mean_sector_pen": safe_mean(mean_sector_pen_list),
        "has_reward_info": int(bool(len(r_pf_list) or len(r_oa_list) or len(r_exist_list) or len(lam_list))),
        "reward_per_step": float(ep_reward / float(step_count)) if step_count > 0 else 0.0,
        "collision_steps": int(collided_steps),
    }

class EvalMetricsCallback(BaseCallback):
    def __init__(
        self,
        eval_env: ASVLidarEnv,
        *,
        eval_cases: Sequence[int],
        eval_freq: int = 50_000,
        max_steps: int = 5_000,
        csv_path: str = "eval_metrics.csv",
        json_path: str = "eval_metrics.json",
        summary_csv_path: str = "eval_summary.csv",
        summary_json_path: str = "eval_summary.json",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_cases = list(eval_cases)
        self.eval_freq = int(eval_freq)
        self.max_steps = int(max_steps)
        self.csv_path = csv_path
        self.json_path = json_path
        self.summary_csv_path = summary_csv_path
        self.summary_json_path = summary_json_path
        self.rows: List[Dict[str, Any]] = []
        self.summary_rows: List[Dict[str, Any]] = []
        self.best_score = -np.inf
        self.header = [
            "timesteps", "test_case",
            "ep_reward", "ep_len", "success", "term_reason",
            "d_start", "d_end", "progress_total", "progress_per_step",
            "mean_speed", "mean_u", "mean_v",
            "mean_rpm", "min_rpm", "max_rpm",
            "mean_abs_rudder", "std_rudder",
            "mean_abs_cte", "max_abs_cte",
            "mean_abs_course_error", "max_abs_course_error",
            "mean_abs_lookahead_error", "max_abs_lookahead_error",
            "min_lidar_all", "p10_front",
            "mean_r_pf", "mean_r_oa", "mean_r_exist", "mean_lambda", "mean_sector_closeness",
            "reward_per_step", "collision_steps", "has_reward_info",
        ]
        self.summary_header = [
            "timesteps",
            "mean_ep_reward", "std_ep_reward", "mean_ep_len",
            "success_rate", "collision_rate", "border_rate", "obstacle_rate", "timeout_rate",
            "mean_progress_per_step", "mean_d_end", "mean_speed",
            "mean_abs_cte", "mean_abs_course_error", "mean_abs_lookahead_error",
            "min_min_lidar_all", "min_p10_front",
            "mean_r_pf", "mean_r_oa", "mean_r_exist", "mean_lambda", "mean_sector_closeness",
            "selection_score", "reward_info_rate",
        ]
        self._csv_inited = False
        self._summary_csv_inited = False

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
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        ep_metrics: List[Dict[str, Any]] = []
        prev_case = getattr(self.eval_env, "test_case", None)
        for case_id in self.eval_cases:
            self.eval_env.test_case = int(case_id)
            m = eval_one_episode(self.model, self.eval_env, deterministic=True, max_steps=self.max_steps)
            m["test_case"] = int(case_id)
            ep_metrics.append(m)
            row = [self.num_timesteps, int(case_id)] + [m.get(k) for k in self.header[2:]]
            self._append_row(row)
            self.rows.append({"timesteps": int(self.num_timesteps), **m})
            if self.verbose:
                print(
                    f"[EVAL @ {self.num_timesteps}] case#{case_id} "
                    f"succ={m['success']} reason={m['term_reason']} "
                    f"cte={m['mean_abs_cte']:.3f} ce={m['mean_abs_course_error']:.2f} la={m['mean_abs_lookahead_error']:.2f} "
                    f"R={m['ep_reward']:.1f}"
                )
        self.eval_env.test_case = prev_case

        def mean_of(key: str) -> float:
            vals = [float(x.get(key, 0.0)) for x in ep_metrics]
            return float(np.mean(vals)) if vals else 0.0

        def std_of(key: str) -> float:
            vals = [float(x.get(key, 0.0)) for x in ep_metrics]
            return float(np.std(vals)) if vals else 0.0

        reasons = [str(x.get("term_reason", "")) for x in ep_metrics]
        success_rate = float(np.mean([int(x.get("success", 0)) for x in ep_metrics])) if ep_metrics else 0.0
        collision_rate = float(np.mean([1 if r in ("obstacle", "border") else 0 for r in reasons])) if reasons else 0.0
        border_rate = float(np.mean([1 if r == "border" else 0 for r in reasons])) if reasons else 0.0
        obstacle_rate = float(np.mean([1 if r == "obstacle" else 0 for r in reasons])) if reasons else 0.0
        timeout_rate = float(np.mean([1 if r == "timeout" else 0 for r in reasons])) if reasons else 0.0

        selection_score = (
            5.0 * success_rate
            - 0.5 * mean_of("mean_abs_cte")
            - 0.05 * mean_of("mean_abs_course_error")
            - 0.02 * mean_of("mean_abs_lookahead_error")
            - 1.0 * border_rate
            - 0.5 * obstacle_rate
        )

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
            "mean_abs_cte": mean_of("mean_abs_cte"),
            "mean_abs_course_error": mean_of("mean_abs_course_error"),
            "mean_abs_lookahead_error": mean_of("mean_abs_lookahead_error"),
            "min_min_lidar_all": float(np.min([float(x.get("min_lidar_all", float("inf"))) for x in ep_metrics])) if ep_metrics else float("inf"),
            "min_p10_front": float(np.min([float(x.get("p10_front", float("inf"))) for x in ep_metrics])) if ep_metrics else float("inf"),
            "mean_r_pf": mean_of("mean_r_pf"),
            "mean_r_oa": mean_of("mean_r_oa"),
            "mean_r_exist": mean_of("mean_r_exist"),
            "mean_lambda": mean_of("mean_lambda"),
            "mean_sector_closeness": mean_of("mean_sector_closeness"),
            "selection_score": float(selection_score),
            "reward_info_rate": mean_of("has_reward_info"),
        }
        self.summary_rows.append(summary)
        self._append_summary_row([summary.get(k) for k in self.summary_header])

        with open(self.json_path, "w") as f:
            json.dump(self.rows, f, indent=2)
        with open(self.summary_json_path, "w") as f:
            json.dump(self.summary_rows, f, indent=2)

        if success_rate > 0.0 and selection_score > self.best_score:
            self.best_score = selection_score
            self.model.save("best_model.zip")
            self.model.save(f"best_model_{self.num_timesteps}.zip")
            if self.verbose:
                print(f"New BEST model saved! score={selection_score:.3f} success={success_rate:.3f}")

        self.logger.record("eval/mean_ep_reward", summary["mean_ep_reward"])
        self.logger.record("eval/std_ep_reward", summary["std_ep_reward"])
        self.logger.record("eval/success_rate", summary["success_rate"])
        self.logger.record("eval/collision_rate", summary["collision_rate"])
        self.logger.record("eval/border_rate", summary["border_rate"])
        self.logger.record("eval/obstacle_rate", summary["obstacle_rate"])
        self.logger.record("eval/mean_progress_per_step", summary["mean_progress_per_step"])
        self.logger.record("eval/mean_d_end", summary["mean_d_end"])
        self.logger.record("eval/mean_speed", summary["mean_speed"])
        self.logger.record("eval/mean_abs_cte", summary["mean_abs_cte"])
        self.logger.record("eval/mean_abs_course_error", summary["mean_abs_course_error"])
        self.logger.record("eval/mean_abs_lookahead_error", summary["mean_abs_lookahead_error"])
        self.logger.record("eval/min_min_lidar_all", summary["min_min_lidar_all"])
        self.logger.record("eval/min_p10_front", summary["min_p10_front"])
        self.logger.record("eval/mean_r_pf", summary["mean_r_pf"])
        self.logger.record("eval/mean_r_oa", summary["mean_r_oa"])
        self.logger.record("eval/mean_lambda", summary["mean_lambda"])
        self.logger.record("eval/selection_score", summary["selection_score"])
        return True

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test", "eval"], default="test")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--eval-max-steps", type=int, default=5_000)
    ap.add_argument("--save-freq", type=int, default=500_000)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--test-case", type=int, default=None)
    ap.add_argument("--eval-cases", type=int, nargs="+", default=DEFAULT_EVAL_CASES)
    ap.add_argument("--eval-lambda", type=float, default=DEFAULT_EVAL_LAMBDA)
    ap.add_argument("--test-lambda", type=float, default=DEFAULT_EVAL_LAMBDA)
    ap.add_argument("--train-map-width", type=float, default=25.0)
    ap.add_argument("--train-map-height", type=float, default=50.0)
    ap.add_argument("--eval-map-width", type=float, default=10.0)
    ap.add_argument("--eval-map-height", type=float, default=25.0)
    ap.add_argument("--train-path-mode", choices=["straight", "curve", "mixed"], default="mixed")
    ap.add_argument("--eval-path-mode", choices=["straight", "curve", "mixed"], default="straight")
    return ap.parse_args()

def make_env(seed: int, rank: int, *, map_width: float, map_height: float, path_mode: str):
    def _init():
        env = ASVLidarEnv(
            render_mode=None,
            map_width=map_width,
            map_height=map_height,
            path_mode=path_mode,
            lambda_override=None,
            test_case=None,
        )
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    algo = args.algo.lower()
    model_path = args.model_path or f"{algo}_asv_model.zip"

    if args.mode == "train":
        env_fns = [
            make_env(args.seed, i, map_width=args.train_map_width, map_height=args.train_map_height, path_mode=args.train_path_mode)
            for i in range(args.num_envs)
        ]
        vec_env = VecMonitor(SubprocVecEnv(env_fns), filename="train_monitor.csv")

        eval_env = ASVLidarEnv(
            render_mode=None,
            map_width=args.eval_map_width,
            map_height=args.eval_map_height,
            path_mode=args.eval_path_mode,
            lambda_override=args.eval_lambda,
        )
        eval_env.reset(seed=args.seed + 10_000)

        if algo == "ppo":
            policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=f"./{algo}_log/",
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
        else:
            model = SAC(
                "MultiInputPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=f"./{algo}_log/",
                learning_rate=1e-4,
                batch_size=512,
                gamma=0.99,
                buffer_size=1_000_000,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
            )

        checkpoint_cb = CheckpointCallback(
            save_freq=max(int(args.save_freq // max(args.num_envs, 1)), 1),
            save_path="models",
            name_prefix=f"{algo}_model",
            save_replay_buffer=(algo == "sac"),
            save_vecnormalize=False,
        )
        eval_cb = EvalMetricsCallback(
            eval_env=eval_env,
            eval_cases=args.eval_cases,
            eval_freq=args.eval_freq,
            max_steps=args.eval_max_steps,
            csv_path="eval_metrics.csv",
            json_path="eval_metrics.json",
            summary_csv_path="eval_summary.csv",
            summary_json_path="eval_summary.json",
            verbose=1,
        )
        callbacks = CallbackList([checkpoint_cb, eval_cb])

        model.learn(total_timesteps=int(args.timesteps), tb_log_name=f"asv_{algo}", callback=callbacks, progress_bar=True)
        model.save(model_path)
        print(f"Saved model -> {model_path}")
        vec_env.close()
        eval_env.close()

    elif args.mode == "test":
        if algo == "ppo":
            model = PPO.load(model_path)
        else:
            model = SAC.load(model_path)

        env = ASVLidarEnv(
            render_mode="human",
            map_width=args.eval_map_width,
            map_height=args.eval_map_height,
            path_mode=args.eval_path_mode,
            lambda_override=args.test_lambda,
            test_case=args.test_case,
            record_video=True,
        )
        obs, _ = env.reset(seed=args.seed + 123)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)

            print(action)

        print(f"Test episode completed. Total reward: {total_reward:.2f}")
        result_data = {
            "heading": env.asv_h,
            "start": [env.start_x, env.start_y],
            "goal": [env.goal_x, env.goal_y],
            "obstacles": env.obstacles,
            "path": env.path.tolist(),
            "asv_path": env.asv_path,
            "lambda": env.current_lambda,
        }
        with open("asv_data.json", "w") as f:
            json.dump(result_data, f, indent=4)
        env.close()

    elif args.mode == "eval":
        if algo == "ppo":
            model = PPO.load(model_path)
        else:
            model = SAC.load(model_path)
        eval_env = ASVLidarEnv(
            render_mode=None,
            map_width=args.eval_map_width,
            map_height=args.eval_map_height,
            path_mode=args.eval_path_mode,
            lambda_override=args.eval_lambda,
        )
        rows = []
        prev_case = getattr(eval_env, "test_case", None)
        for case_id in args.eval_cases:
            eval_env.test_case = int(case_id)
            m = eval_one_episode(model, eval_env, deterministic=True, max_steps=args.eval_max_steps)
            m["test_case"] = int(case_id)
            rows.append(m)
            print(
                f"[EVAL] case#{case_id} succ={m['success']} reason={m['term_reason']} "
                f"cte={m['mean_abs_cte']:.3f} ce={m['mean_abs_course_error']:.2f} la={m['mean_abs_lookahead_error']:.2f} R={m['ep_reward']:.1f}"
            )
        eval_env.test_case = prev_case
        with open("eval_only_metrics.json", "w") as f:
            json.dump(rows, f, indent=2)
        eval_env.close()