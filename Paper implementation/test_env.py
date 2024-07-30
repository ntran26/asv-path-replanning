import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from static_obstacle_env import ASVEnv

env = ASVEnv()
obs = env.reset()

# Display initial observation
env.render()

# Create the animation function
def animate(i):
    action = env.action_space.sample()  # Sample a random action
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()

# Create and start the animation
ani = FuncAnimation(plt.gcf(), animate, frames=200, interval=200, repeat=False)
plt.show()