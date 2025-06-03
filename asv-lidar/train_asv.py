import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from asv_lidar_gym_continuous import ASVLidarEnv
from test_run import testEnv
import json
import os

import multiprocessing

if __name__=='__main__':
    multiprocessing.freeze_support()

    # Toggle between train and test
    TRAIN = 0

    TEST_CASE = 1

    # Choose algorithm
    # algorithm = 'PPO'
    # algorithm = 'TD3'
    algorithm = 'SAC'

    # Create the environment

    def make_env():
        return Monitor(ASVLidarEnv(render_mode=None))   # monitor/logging

    if algorithm == 'PPO' or 'SAC' or 'TD3':
        num_envs = 8
        env = SubprocVecEnv([make_env for _ in range(num_envs)])    # parallelize training
    # elif algorithm == 'TD3':
    #     num_envs = 1
    #     env = Monitor(ASVLidarEnv(render_mode=None))

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
        def __init__(self, save_freq=500000, verbose=0):
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

            if self.locals.get("loss", None) is not None:
                loss = self.locals["loss"]
                self.policy_loss.append(loss.get("policy_loss", 0))
                self.value_loss.append(loss.get("value_loss", 0))

            if len(self.model.ep_info_buffer) > 0:
                self.rewards.append(self.model.ep_info_buffer[0]["r"])
                if "loss" in self.model.ep_info_buffer[0]:
                    self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
                    self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])
            return True

    # Model save path
    MODEL_PATH = f"{algorithm.lower()}_asv_model"


    #                       -------------- TRAINING --------------

    if TRAIN == 1:
        # Initialize PPO model
        if algorithm == 'PPO':
            model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./{algorithm.lower()}_log/",
                        learning_rate=learn_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                        gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, clip_range_vf=clip_range_vf,
                        ent_coef=ent_coef, vf_coef=vf_coef)
        elif algorithm == 'SAC':
            model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./{algorithm.lower()}_log/",
                        learning_rate=learn_rate, batch_size=batch_size, gamma=gamma, buffer_size=1000000,
                        train_freq=1, gradient_steps=1, ent_coef='auto')
        elif algorithm == 'TD3':
            model = TD3("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./{algorithm.lower()}_log/",
                    learning_rate=learn_rate, batch_size=batch_size, gamma=gamma, buffer_size=1000000,
                    train_freq=(1, 'step'), gradient_steps=-1)
        
        # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_asv_tensorboard/")

        # Training parameters
        timesteps = 1000000
        callback = CustomCallback()

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

        fig = plt.figure(2)
        plt.subplot(221)
        plt.plot(callback.policy_loss)
        plt.title("Policy Loss")
        plt.xlabel("Training Steps")

        plt.subplot(222)
        plt.plot(callback.value_loss)
        plt.title("Value Loss")
        plt.xlabel("Training Steps")

        plt.show()

        env.close()


    #                       -------------- TESTING --------------

    elif TRAIN == 0:
        # Load the trained model and test it
        if algorithm == 'PPO':
            # model = PPO.load(MODEL_PATH)
            model = PPO.load("models/ppo_asv_model_continuous_1.zip")
            # model = PPO.load("models/ppo_asv_model_continuous_2.zip")
            # model = PPO.load("models/ppo_asv_model_continuous_3.zip")
        elif algorithm == 'SAC':
            model = SAC.load(MODEL_PATH)
            model = SAC.load("models/sac_asv_model_1.zip")
        elif algorithm == 'TD3':
            model = TD3.load(MODEL_PATH)

        env = testEnv(render_mode="human")
        env.test_case = TEST_CASE
        # env = ASVLidarEnv(render_mode="human")

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

    else:
        # define environment
        env = testEnv(render_mode="human")
        env.test_case = TEST_CASE

        # load data file
        with open("asv_data.json", "r") as f:
            data = json.load(f)
        
        # extract data
        start = data["start"]
        goal = data["goal"]
        obstacles = data["obstacles"]
        path = data["path"]
        asv_path = data["asv_path"]

        # plot data
        path_surface = pygame.Surface((env.map_width, env.map_height))
        path_surface.fill((255,255,255))

        for i in range(1, len(asv_path)):
            pygame.draw.circle(path_surface, (0, 0, 200), asv_path[i], 3)

        # Draw obstacles
        for obs in obstacles:
            pygame.draw.polygon(path_surface, (200, 0, 0), obs)
        
        # Draw Path
        env.draw_dashed_line(path_surface,(0,200,0),(start[0],start[1]),(goal[0],goal[1]),width=5)
        pygame.draw.circle(path_surface,(100,0,0),(goal[0],goal[1]),5)

        # # Draw ship
        # display = pygame.display.set_mode(env.screen_size)
        # os_ = pygame.transform.rotozoom(env.icon,-asv_path[-1][1],2)
        # path_surface.blit(os_,os_.get_rect(center=(asv_path[-1][0],asv_path[-1][1])))
        # display.blit(path_surface,[0,0])

        pygame.image.save(path_surface, "asv_data_result.png")

        # Plot with matplotlib

        plt.figure(figsize=(6,10))

        for i in range(1, len(env.asv_path)):
            pygame.draw.circle(path_surface, (0, 0, 200), env.asv_path[i], 3)
        plt.plot(*zip(*asv_path), label="ASV Path", color="blue")
        plt.plot(*zip(*path), label="Path", color="green", alpha=0.5, linestyle="--")
        plt.scatter(*start, color='green', label='Start')
        plt.scatter(*goal, color='yellow', label='Goal')
        for obs in obstacles:
            poly = plt.Polygon(obs, color='red')
            plt.gca().add_patch(poly)

        plt.gca().invert_yaxis()
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('ASV Path Visualization')
        plt.legend()
        # plt.axis('equal')
        plt.xlim((0,400))
        plt.ylim((600,0))
        plt.grid(False)
        plt.show()

