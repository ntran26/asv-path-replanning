import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from path_follow_env import ASVEnv
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

    # Load the model
    model_path = "ppo_path_follow"
    model = PPO.load(model_path)

    # Test the trained model
    obs, info = env.reset()
    cumulative_reward = 0
    for _ in range(50):  # Run for 500 steps or until done
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