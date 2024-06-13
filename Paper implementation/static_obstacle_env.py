import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Define map dimensions
WIDTH = 500
HEIGHT = 500

# Define ASV initial parameters
INITIAL_HEADING = 90
TURN_RATE = 5
SPEED = 1

# Define observation radius parameters
RADIUS = 100
SQUARE_SIZE = 10
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define start and goal position
START = (0, 0)
GOAL = (0, 300)

class StaticObsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}
    def __init__(self, render_mode='human'):
        self.render_mode = render_mode

        # Set width, height and boundary
        self.width = WIDTH
        self.height = HEIGHT
        self.boundary = [(0,0), (0,self.height), (self.width,self.height), (self.width,0), (0,0)]

        # ASV parameters
        self.position = START
        self.heading = INITIAL_HEADING
        self.speed = 1

        # Observation space and action space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=self.width, shape=(4,), dtype=np.float32)

        # Draw the path
    
    # Reset function
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # return observation, {}
        return 
    
    # Check if the ASV is on path
    def is_on_path(self, position):
        return
    
    # Check if the ASV is on the map
    def is_valid_pos(self, position):
        return
    
    # Check if the ASV reached the goal
    def is_goal(self, position):
        return
    
    # Calculate reward of the step
    def calculat_reward(self, position):
        return
    
    # Step function
    def step(self, action):
        return
    
    # Render function
    def render(self):
        return
    


