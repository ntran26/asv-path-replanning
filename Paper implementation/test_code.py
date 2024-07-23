import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

class PathFollowingEnv(gym.Env):
    def __init__(self):
        self.map_width = 20
        self.map_height = 20
        self.agent_position = np.array([5, 5])
        self.goal_position = np.array([8, 8])
        self.heading = 90

        self.action_space = gym.spaces.Discrete(3)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,))

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.agent_marker = patches.Circle((5, 5), 0.3, color='red')
        self.goal_marker = patches.Circle((15, 15), 0.3, color='green')
        self.ax.add_patch(self.agent_marker)
        self.ax.add_patch(self.goal_marker)

    def reset(self):
        self.agent_position = np.array([5, 5])
        self.agent_marker.set_center((5, 5))
        return self.agent_position

    def step(self, action):
        # if action == 0:  # Up
        #     self.agent_position[1] = min(self.map_height, self.agent_position[1] + 1)
        # elif action == 1:  # Down
        #     self.agent_position[1] = max(0, self.agent_position[1] - 1)
        # elif action == 2:  # Left
        #     self.agent_position[0] = max(0, self.agent_position[0] - 1)
        # elif action == 3:  # Right
        #     self.agent_position[0] = min(self.map_width, self.agent_position[0] + 1)

        if action == 0: # straight
            self.heading = self.heading
        elif action == 1: # left
            self.heading += 5
        elif action == 2: # right
            self.heading -= 5
        self.agent_position[0] += 1 * np.cos(np.radians(self.heading))
        self.agent_position[1] += 1 * np.sin(np.radians(self.heading))

        self.agent_marker.set_center(self.agent_position)
        
        # Calculate reward
        if np.array_equal(self.agent_position, self.goal_position):
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return self.agent_position, reward, done, {}

    def render(self):
        plt.pause(0.1)

# Create the environment
env = PathFollowingEnv()

# Reset the environment
initial_observation = env.reset()


def update(frame):
    action = env.action_space.sample()  # Take a random action
    observation, reward, done, _ = env.step(action)
    return env.agent_marker,

ani = animation.FuncAnimation(env.fig, update, frames=50, blit=True)

plt.show()
