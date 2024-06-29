import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# Define colors
WHITE = 0
GREEN = 1
YELLOW = 2
RED = 3

# Define map dimensions
WIDTH = 200
HEIGHT = 300

class ASVEnvironment(gym.Env):
    def __init__(self):
        super(ASVEnvironment, self).__init__()
        
        # Define the action and observation space
        self.action_space = spaces.Discrete(4)  # Example actions: 0:move up, 1:move down, 2:move left, 3:move right
        
        # State with possible values: Free Space, Path, Goal, Obstacle/Boundary
        self.observation_space = spaces.Discrete(4)
        
        # Define the map grid with initial configuration
        self.map_grid = np.random.choice([WHITE, GREEN, YELLOW, RED], size=(WIDTH, HEIGHT))
        
        # Initialize the agent position randomly
        self.agent_position = (np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT))
        self.map_grid[self.agent_position] = WHITE  # Set initial agent position to white
        
        # Initialize the figure for rendering
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.map_grid.T, cmap='viridis')
    
    def step(self, action):
        # Execute the action and update the environment
        # This is where you would move the agent and update the state based on the action
        
        # For now, let's just move the agent randomly
        self.agent_position = (self.agent_position[0] + np.random.randint(-1, 2),
                               self.agent_position[1] + np.random.randint(-1, 2))
        
        # Clip the agent position within the map boundaries
        self.agent_position = (np.clip(self.agent_position[0], 0, WIDTH-1),
                               np.clip(self.agent_position[1], 0, HEIGHT-1))
        
        # Update the map grid with the new agent position
        self.map_grid[self.agent_position] = WHITE
        
        # Return the new observation, reward, done, info
        observation = self.map_grid[self.agent_position]
        reward = 0  # Placeholder reward
        done = False  # Placeholder for episode termination
        info = {}  # Additional information
        
        return observation, reward, done, info
    
    def reset(self):
        # Reset the environment to the initial state
        self.map_grid = np.random.choice([WHITE, GREEN, YELLOW, RED], size=(WIDTH, HEIGHT))
        self.agent_position = (np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT))
        self.map_grid[self.agent_position] = WHITE
        
        return self.map_grid[self.agent_position]
    
    def render(self, mode='human'):
        # Render the current state of the environment
        self.img.set_data(self.map_grid.T)
        plt.pause(0.1)  # Pause to show the updated image
        plt.draw()
    
    def close(self):
        plt.close()

# Create the ASV environment
env = ASVEnvironment()

# Random actions to test the environment
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    env.render()

# Close the environment after testing
env.close()
