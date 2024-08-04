import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from static_obs_env import ASVEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gymnasium import spaces

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


if __name__ == '__main__':
    # Create the environment
    env = ASVEnv()

    # Check the environment
    check_env(env)

    # Load the model
    model_path = "ppo_asv_model"
    model = PPO.load(model_path)

    # Test the trained model
    obs, info = env.reset()
    for _ in range(150):  # Run for 150 steps or until done
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        if done or truncated:
            break

    env.close()