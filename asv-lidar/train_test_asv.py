import os
import csv
import time
import numpy as np
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from asv_lidar_rudder_speed_control import ASVLidarEnv, RPM_MAX, RPM_MIN
# from test_run import testEnv 
import json
import argparse
import sys
import multiprocessing

"""
Train: python train_test_asv.py --mode train --algo sac
Test: python train_test_asv.py --mode test --algo sac --case 0
"""

def action_to_rpm(throttle_cmd: float) -> float:
    """
    Map normalized throttle [-1,1] to rpm [RPM_MIN, RPM_MAX]
    """
    throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))
    return float(RPM_MIN + (throttle_cmd + 1.0) * 0.5 * (RPM_MAX - RPM_MIN))

def action_to_rudder_deg(rudder_cmd: float) -> float:
    """
    Map normalized rudder [-1,1] to degrees
    """
    rudder_cmd = float(np.clip(rudder_cmd, -1.0, 1.0))
    return float(rudder_cmd * 25.0)

def lidar_clearance_stats(env) -> dict:
    """Compute clearance stats from lidar beams (handles invalid/0)."""
    out = {
        "min_lidar_all": float("inf"),
        "p10_front": float("inf"),
        "p50_front": float("inf"),
    }
    if not (hasattr(env, "lidar") and hasattr(env.lidar, "ranges") and hasattr(env.lidar, "angles")):
        return out

    r = np.array(env.lidar.ranges, dtype=np.float32)
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

def termination_reason(env, done: bool, hit_max_steps: bool) -> str:
    """
    Infer termination reason without changing the env.
    Possible: goal / obstacle / border / timeout / terminated
    """
    if hit_max_steps:
        return "timeout"

    # Goal (uses same logic as env.check_done)
    goal_radius = float(getattr(env, "collision", 0.0)) + 30.0
    d_goal = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))
    if d_goal <= goal_radius:
        return "goal"

    # Border vs obstacle
    collided = False
    if hasattr(env, "_check_collision_geom"):
        try:
            collided = bool(env._check_collision_geom())
        except Exception:
            collided = False

    if collided and hasattr(env, "_hull_polygon_world"):
        try:
            hull = env._hull_polygon_world()
            xs = [p[0] for p in hull]
            ys = [p[1] for p in hull]
            if min(xs) < 0 or max(xs) > env.map_width or min(ys) < 0 or max(ys) > env.map_height:
                return "border"
        except Exception:
            pass
        return "obstacle"

    # Some other termination (rare)
    return "terminated" if done else "timeout"

def eval_one_episode(model, env, deterministic=True, max_steps=5000):
    obs, _ = env.reset()
    done = False
    terminated = False
    truncated = False

    ep_reward = 0.0
    step_count = 0

    # Episode-level lists for stats
    speed_list = []
    rpm_list = []
    rudder_deg_list = []
    tgt_list = []
    angle_diff_list = []
    dist_goal_list = []
    min_lidar_list = []
    p10_front_list = []

    d_start = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))

    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)

        # Ensure action is flat array-like
        action = np.array(action).reshape(-1)

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        ep_reward += float(reward)
        step_count += 1

        # Speed: use pose-based speed (env.speed_mps, also equals obs['speed'][0])
        spd = float(getattr(env, "speed_mps", float(obs["speed"][0]) if isinstance(obs, dict) and "speed" in obs else 0.0))
        speed_list.append(spd)

        # Decode commands
        rudder_deg = action_to_rudder_deg(float(action[0]))
        rpm = action_to_rpm(float(action[1]))
        rudder_deg_list.append(rudder_deg)
        rpm_list.append(rpm)

        # Task-state stats
        tgt_list.append(float(getattr(env, "tgt", 0.0)))
        angle_diff_list.append(float(getattr(env, "angle_diff", 0.0)))

        d_goal = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))
        dist_goal_list.append(d_goal)

        # Lidar clearance stats
        cs = lidar_clearance_stats(env)
        min_lidar_list.append(cs["min_lidar_all"])
        p10_front_list.append(cs["p10_front"])

        if done:
            break

    hit_max_steps = (step_count >= max_steps and not done)
    d_end = float(np.hypot(env.goal_x - env.asv_x, env.goal_y - env.asv_y))

    prog_total = d_start - d_end
    prog_per_step = prog_total / float(step_count) if step_count > 0 else 0.0

    reason = termination_reason(env, done=done, hit_max_steps=hit_max_steps)
    success = 1 if reason == "goal" else 0

    # Aggregate metrics
    def safe_mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    def safe_min(x):
        return float(np.min(x)) if len(x) else float("inf")

    def safe_max(x):
        return float(np.max(x)) if len(x) else 0.0

    metrics = {
        "ep_reward": ep_reward,
        "ep_len": int(step_count),
        "success": int(success),
        "term_reason": reason,
        "d_start": d_start,
        "d_end": d_end,
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
        "p10_front": safe_min(p10_front_list),  # worst-case front clearance proxy
    }
    return metrics

class CustomCallback(BaseCallback):
    def __init__(self, save_freq=500000, verbose=1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.model_save_counter = 0
        self.rewards = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/model_{self.num_timesteps}.zip"
            print(f"Saving model at {self.num_timesteps} timesteps -> {model_path}")
            self.model.save(model_path)
            self.model_save_counter += 1

        # SB3 stores recent episode info in ep_info_buffer
        if len(self.model.ep_info_buffer) > 0 and "r" in self.model.ep_info_buffer[0]:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
        return True


class EvalMetricsCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=3,
                 csv_path="eval_metrics.csv", json_path="eval_metrics.json",
                 verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.csv_path = csv_path
        self.json_path = json_path

        self.rows = []
        self._csv_inited = False

        self.header = [
            "timesteps", "episode",
            "ep_reward", "ep_len", "success", "term_reason",
            "d_start", "d_end", "progress_total", "progress_per_step",
            "mean_speed", "min_speed", "max_speed",
            "mean_rpm", "min_rpm", "max_rpm",
            "mean_abs_rudder", "std_rudder",
            "mean_abs_tgt", "max_abs_tgt",
            "mean_abs_angle_diff", "max_abs_angle_diff",
            "min_lidar_all", "p10_front"
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

    def _append_row(self, row):
        self._init_csv()
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        # Evaluate
        for ep_i in range(self.n_eval_episodes):
            m = eval_one_episode(self.model, self.eval_env, deterministic=True)

            if self.verbose:
                print(
                    f"[EVAL @ {self.num_timesteps}] ep#{ep_i} "
                    f"R={m['ep_reward']:.1f} len={m['ep_len']} "
                    f"succ={m['success']} reason={m['term_reason']} "
                    f"d_end={m['d_end']:.1f} prog/step={m['progress_per_step']:.3f} "
                    f"v_mean={m['mean_speed']:.2f} (min {m['min_speed']:.2f}) "
                    f"rpm_mean={m['mean_rpm']:.1f} "
                    f"p10_front={m['p10_front']:.1f}"
                )

            row = [self.num_timesteps, ep_i] + [m.get(k) for k in self.header[2:]]
            self._append_row(row)

            self.rows.append({"timesteps": self.num_timesteps, "episode": ep_i, **m})

        with open(self.json_path, "w") as f:
            json.dump(self.rows, f, indent=2)

        return True

if __name__=='__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','test'], required='False', help='Select train or test')
    parser.add_argument('--algo', choices=['ppo','sac'], required='False', help='Select PPO or SAC algorithm')
    parser.add_argument('--case', type=int, default=0, help='Select test case (only use in test mode)')

    if len(sys.argv) == 1:
        print("No arguments passed. Using default: mode=test, algo=sac, case=0\n")
        args = parser.parse_args(['--mode', 'test', '--algo', 'sac', '--case', '0'])
    else:
        args = parser.parse_args()

    algorithm = args.algo.upper()

    # Create the environment
    def make_env():
        return Monitor(ASVLidarEnv(render_mode=None))   # monitor/logging
    num_envs = 8
    env = SubprocVecEnv([make_env for _ in range(num_envs)])    # parallelize training
    eval_env = ASVLidarEnv(render_mode=None)

    # Hyperparamters
    learn_rate = 0.0001
    n_steps = num_envs*1024     # multiple of num_envs
    batch_size = 512
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.01
    vf_coef = 0.5

    MODEL_PATH = f"{algorithm.lower()}_asv_model.zip"

    #                       -------------- TRAINING --------------
    if args.mode == 'train':        
        if args.mode == "train":
            if algorithm == "PPO":
                model = PPO(
                    "MultiInputPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=f"./{algorithm.lower()}_log/",
                    learning_rate=learn_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                )
            elif algorithm == "SAC":
                model = SAC(
                    "MultiInputPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=f"./{algorithm.lower()}_log/",
                    learning_rate=learn_rate,
                    batch_size=batch_size,
                    gamma=gamma,
                    buffer_size=1_000_000,
                    train_freq=1,
                    gradient_steps=1,
                    ent_coef="auto",
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        
        timesteps = 1_000_000

        # Evaluation parameters
        custom_cb = CustomCallback(save_freq=500_000)
        eval_cb = EvalMetricsCallback(eval_env, eval_freq=50_000, n_eval_episodes=3)

        # Train model
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=f"asv_{algorithm.lower()}",
            callback=[custom_cb, eval_cb],
            progress_bar=True,
        )

        # Save the model
        model.save(f"{algorithm.lower()}_asv_model")
        print("Model saved!")

        # Quick evaluation (vector env)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

        # Plot rewards if available
        if len(custom_cb.rewards) > 0:
            fig = plt.figure(1)
            plt.plot(custom_cb.rewards, label="Rewards")
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            plt.title("Reward over Episodes")
            fig.savefig("reward_plot.png")
            plt.show()

        env.close()
        eval_env.close()


    #                       -------------- TESTING --------------
    elif args.mode == "test":
        if algorithm == "PPO":
            model = PPO.load(MODEL_PATH)
        elif algorithm == "SAC":
            model = SAC.load(MODEL_PATH)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

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

        # Save path taken as image
        path_surface = pygame.Surface((env.map_width, env.map_height))
        path_surface.fill((255, 255, 255))

        for i in range(1, len(env.asv_path)):
            pygame.draw.circle(path_surface, (0, 0, 200), env.asv_path[i], 3)

        # Draw obstacles
        for obs in env.obstacles:
            pygame.draw.polygon(path_surface, (200, 0, 0), obs)

        # Draw path
        env.draw_dashed_line(path_surface, (0, 200, 0), (env.start_x, env.start_y), (env.goal_x, env.goal_y), width=5)
        pygame.draw.circle(path_surface, (100, 0, 0), (env.tgt_x, env.tgt_y), 5)

        # Draw ship icon
        display = pygame.display.set_mode(env.screen_size)
        os_ = pygame.transform.rotozoom(env.icon, -env.asv_h, 1)
        path_surface.blit(os_, os_.get_rect(center=(env.asv_x, env.asv_y)))
        display.blit(path_surface, [0, 0])

        pygame.image.save(path_surface, "asv_path_result.png")