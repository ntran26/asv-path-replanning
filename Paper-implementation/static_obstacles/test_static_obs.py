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


if __name__ == '__main__':
    # Create the environment
    env = ASVEnv()

    # Check the environment
    check_env(env)

    # Choose model version
    version = 0

    # Load the model
    if version == 1:
        model_path = "Paper-implementation/static_obstacles/ppo_static_obstacles_v1"
    elif version == 2:
        model_path = "Paper-implementation/static_obstacles/ppo_static_obstacles_v2"
    elif version == 3:
        model_path = "Paper-implementation/static_obstacles/ppo_static_obstacles_v3"
    else:
        model_path = "Paper-implementation/static_obstacles/ppo_static_obstacles"

    model = PPO.load(model_path)

    # Test the trained model
    obs, info = env.reset()
    cumulative_reward = 0
    for _ in range(env.max_num_step):  # Run for 500 steps or until done
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward
        env.render()

        if done or truncated:
            break
    
    print(f"Cumulative reward = {cumulative_reward}")

    # Plot the path taken
    env.display_path()

    env.close()