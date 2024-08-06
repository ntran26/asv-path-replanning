# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import CheckpointCallback
# from static_obs_env import ASVEnv
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from gymnasium import spaces

#                               -------- CONFIGURATION --------
# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Define map dimensions and start/goal points, number of static obstacles
WIDTH = 200
HEIGHT = 300
START = (100, 30)
GOAL = (100, 250)
NUM_STATIC_OBS = 5

# Define observation radius and grid size
RADIUS = 100
SQUARE_SIZE = 10
SPEED = 2
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define initial heading angle, turn rate and number of steps
INITIAL_HEADING = 90
TURN_RATE = 5

# Define states
FREE_STATE = 0          # free space
PATH_STATE = 1          # path
COLLISION_STATE = 2     # obstacle or border
GOAL_STATE = 3          # goal point


# if __name__ == '__main__':
#     # Create the environment
#     env = ASVEnv()

#     # Check the environment
#     check_env(env)

#     # Train the agent using PPO
#     model = PPO('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=int(1e6))

#     # Save the model
#     model.save("ppo_asv_model")

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from static_obs_env import ASVEnv

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.policy_loss = []
        self.value_loss = []
        self.rewards = []

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
            if "loss" in self.model.ep_info_buffer[0]:
                self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
                self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])
        return True

if __name__ == '__main__':
    # Create the environment
    env = ASVEnv()

    # Check the environment
    check_env(env)

    # Define the model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model with callback
    callback = CustomCallback()
    model.learn(total_timesteps=100000, callback=callback)

    # Save the model
    model.save("ppo_asv_model")

    # Plot the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(callback.rewards, label='Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward over Steps')
    plt.legend()
    plt.show()