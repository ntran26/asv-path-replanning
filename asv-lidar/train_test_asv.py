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
from asv_lidar_rudder_speed_control import ASVLidarEnv
# from test_run import testEnv 
import json
import argparse
import sys
import multiprocessing

"""
Train: python train_test_asv.py --mode train --algo sac
Test: python train_test_asv.py --mode test --algo sac --case 0
"""

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

    # Create evaluation function
    def eval_one_episode(model, env, deterministic=True, max_steps=5000):
        obs, _ = env.reset()
        done = False

        ep_reward = 0
        step_count = 0

        # recorded metrics
        speed_list = []
        min_lidar_range = []

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += float(reward)
            step_count += 1

            # append speed
            spd = float(getattr(env.model, "_v", 0))
            speed_list.append(spd)

            # minimum lidar range
            if hasattr(env, "lidar") and hasattr(env.lidar, "ranges"):
                range = np.array(env.lidar.ranges, dtype=np.float32)
                range[range <= 0] = np.inf
                min_lidar_range.append(float(np.min(range)) if np.any(np.isfinite(range)) else float("inf"))
            
        # final distance to goal
        dx = float(env.goal_x - env.asv_x)
        dy = float(env.goal_y - env.asv_y)
        final_dist = float(np.hypot(dx, dy))

        # log metrics
        metrics = {
            "ep_reward": ep_reward,
            "ep_len": step_count,
            "final_dist": final_dist,
            "mean_speed": float(np.mean(speed_list)) if speed_list else 0,
            "min_lidar_range": float(np.min(min_lidar_range)) if min_lidar_range else float("inf")
        }
        return metrics

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

    class CustomCallback(BaseCallback):
        def __init__(self, save_freq=500000, verbose=1):
            super(CustomCallback, self).__init__(verbose)
            self.save_freq = save_freq
            self.model_save_counter = 0
            self.policy_loss = []
            self.value_loss = []
            self.rewards = []

        def _on_step(self) -> bool:
            # Save model at regular intervals
            if self.num_timesteps % self.save_freq == 0:
                model_path = f"models/model_{self.num_timesteps}.zip"
                print(f"Saving model at {self.num_timesteps} timesteps")
                self.model.save(model_path)
                self.model_save_counter += 1
            if len(self.model.ep_info_buffer) > 0:
                self.rewards.append(self.model.ep_info_buffer[0]["r"])
            return True

    class EvalMetricsCallback(BaseCallback):
        def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=3,
                    csv_path="eval_metrics.csv", json_path="eval_metrics.json",
                    verbose=1):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.csv_path = csv_path
            self.json_path = json_path

            self.rows = []
            self._csv_inited = False

        def _init_csv(self):
            if self._csv_inited:
                return
            header = ["timesteps", "episode", "ep_reward", "ep_len", "final_dist", "mean_speed", "min_lidar_range"]
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
            self._csv_inited = True

        def _append_row(self, row):
            self._init_csv()
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

        def _on_step(self) -> bool:
            # Run evaluation every eval_freq timesteps
            if self.num_timesteps % self.eval_freq != 0:
                return True

            results = []
            for i in range(self.n_eval_episodes):
                m = eval_one_episode(self.model, self.eval_env, deterministic=True)
                results.append(m)

                # print per episode
                if self.verbose:
                    print(
                        f"[EVAL @ {self.num_timesteps}] ep#{i} "
                        f"R={m['ep_reward']:.1f} len={m['ep_len']} "
                        f"d_goal={m['final_dist']:.2f}m "
                        f"v_mean={m['mean_speed']:.2f}m/s "
                        f"min_lidar={m['min_lidar_range']:.2f}m"
                    )

                # save each episode row
                row = [self.num_timesteps, i, m["ep_reward"], m["ep_len"],
                    m["final_dist"], m["mean_speed"], m["min_lidar_range"]]
                self._append_row(row)

            # also store JSON snapshot (accumulated)
            self.rows.extend([{"timesteps": self.num_timesteps, "episode": i, **r} for i, r in enumerate(results)])
            with open(self.json_path, "w") as f:
                json.dump(self.rows, f, indent=2)

            return True

    # Model save path
    MODEL_PATH = f"{algorithm.lower()}_asv_model.zip"

    #                       -------------- TRAINING --------------

    if args.mode == 'train':
        # Initialize PPO model
        if algorithm == 'PPO':
            model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./{algorithm.lower()}_log/",
                        learning_rate=learn_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                        gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, clip_range_vf=clip_range_vf,
                        ent_coef=ent_coef, vf_coef=vf_coef)
            # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_asv_tensorboard/")
        elif algorithm == 'SAC':
            model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./{algorithm.lower()}_log/",
                        learning_rate=learn_rate, batch_size=batch_size, gamma=gamma, buffer_size=1000000,
                        train_freq=1, gradient_steps=1, ent_coef='auto')
        
        # Training parameters
        timesteps = 1000000
        # callback = CustomCallback()
        callback = [CustomCallback(), EvalMetricsCallback(eval_env, eval_freq=50000, n_eval_episodes=3)]

        # Train the model
        model.learn(total_timesteps=timesteps, tb_log_name=f"asv_{algorithm.lower()}", callback=callback, progress_bar=True)

        # Save the model
        model.save(f"{algorithm.lower()}_asv_model")
        print("Model saved!")

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

        # Plot rewards/episodes
        fig = plt.figure(1)
        plt.plot(callback.rewards, label="Rewards")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward over Episodes')
        fig.savefig("reward_plot.png")

        # fig = plt.figure(2)
        # plt.subplot(221)
        # plt.plot(callback.policy_loss)
        # plt.title("Policy Loss")
        # plt.xlabel("Training Steps")

        # plt.subplot(222)
        # plt.plot(callback.value_loss)
        # plt.title("Value Loss")
        # plt.xlabel("Training Steps")

        plt.show()

        env.close()


    #                       -------------- TESTING --------------

    elif args.mode == 'test':
        # Load the trained model and test it
        if algorithm == 'PPO':
            model = PPO.load(MODEL_PATH)
            # model = PPO.load("models/ppo_asv_model_v1.zip")
            # model = PPO.load("models/ppo_asv_model_v2.zip")
            # model = PPO.load("models/ppo_asv_model_0_7.zip")
        elif algorithm == 'SAC':
            model = SAC.load(MODEL_PATH)
            # model = SAC.load("models/sac_asv_model_v1.zip")
            # model = SAC.load("models/sac_asv_model_v2.zip")
            # model = SAC.load("models/sac_asv_model_0_5.zip")

        env = ASVLidarEnv(render_mode="human")

        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True) 
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            # print(total_reward)

        print(f"Test episode completed. Total reward: {total_reward}")

        # Save data
        result_data = {
            "heading": env.asv_h,
            "start": [env.start_x, env.start_y],
            "goal": [env.goal_x, env.goal_y],
            "obstacles": env.obstacles,
            "path": env.path,
            "asv_path": env.asv_path
        }

        # i = 1
        # while os.path.exists(f"asv_data_{i}.json"):
        #     i += 1
        # filename = f"asv_data_{i}.json"

        with open("asv_data.json", "w") as f:
            json.dump(result_data, f, indent=4)

        # save data of random test cases
        if env.test_case == 0:
            random_data = {
                "start": [env.start_x, env.start_y],
                "goal": [env.goal_x, env.goal_y],
                "obstacles": env.obstacles,
                "path": env.path,
                "asv_start": env.asv_path[0]
            }
            with open("env_data.json","w") as f:
                json.dump(random_data, f, indent=4)

        # Save path taken as image
        path_surface = pygame.Surface((env.map_width, env.map_height))
        path_surface.fill((255,255,255))

        for i in range(1, len(env.asv_path)):
            pygame.draw.circle(path_surface, (0, 0, 200), env.asv_path[i], 3)

        # Draw obstacles
        for obs in env.obstacles:
            pygame.draw.polygon(path_surface, (200, 0, 0), obs)
        
        # Draw Path
        env.draw_dashed_line(path_surface,(0,200,0),(env.start_x,env.start_y),(env.goal_x,env.goal_y),width=5)
        pygame.draw.circle(path_surface,(100,0,0),(env.tgt_x,env.tgt_y),5)

        # Draw ship
        display = pygame.display.set_mode(env.screen_size)
        os_ = pygame.transform.rotozoom(env.icon,-env.asv_h,2)
        path_surface.blit(os_,os_.get_rect(center=(env.asv_x,env.asv_y)))
        display.blit(path_surface,[0,0])

        # # Draw map boundaries
        # pygame.draw.line(path_surface, (200, 0, 0), (0,0), (0,env.map_height), 5)
        # pygame.draw.line(path_surface, (200, 0, 0), (0,env.map_height), (env.map_width,env.map_height), 5)
        # pygame.draw.line(path_surface, (200, 0, 0), (env.map_width,0), (env.map_width,env.map_height), 5)
        # pygame.draw.line(path_surface, (200, 0, 0), (0,0), (env.map_width,0), 5)

        pygame.image.save(path_surface, "asv_path_result.png")