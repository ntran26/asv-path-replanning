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

    # Load the model
    # model_path = "ppo_asv_model"
    model_path = "ppo_custom_policy"
    # model_path = "rl_model_100000_steps"
    # model_path = "rl_model_1000000_steps"
    model = PPO.load(model_path)

    # Test the trained model
    obs, info = env.reset()
    for _ in range(500):  # Run for 500 steps or until done
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        if done or truncated:
            break
    
    # Plot the path taken
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_aspect("equal")
    ax.set_title("Steps Taken")
    ax.set_xlim(-env.radius, env.width + env.radius)
    ax.set_ylim(-env.radius, env.height + env.radius)
    ax.plot(env.start[0], env.start[1], marker='o', color=BLUE)
    ax.plot(env.goal[0], env.goal[1], marker='o', color=YELLOW)
    for obj in env.boundary:
        boundary_line = plt.Rectangle((obj['x'], obj['y']), 1, 1, edgecolor=BLACK, facecolor=BLACK)
        ax.add_patch(boundary_line)
    path_x = [point['x'] for point in env.path]
    path_y = [point['y'] for point in env.path]
    ax.plot(path_x, path_y, '-', color=GREEN)
    for obj in env.obstacles:
        ax.plot(obj['x'], obj['y'], marker='o', color=RED)
    ax.plot(env.position[0], env.position[1], marker='^', color=BLUE)
    step_x = [point[0] for point in env.step_taken]
    step_y = [point[1] for point in env.step_taken]
    ax.plot(step_x, step_y, '-', color=BLUE)
    plt.show()

    env.close()