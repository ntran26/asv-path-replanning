import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

RADIUS = 100
SQUARE_SIZE = 10
SPEED = 2
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define map dimensions
WIDTH = 200
HEIGHT = 300
START = (0, 0)
GOAL = (0, 200)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 200/SPEED

# Map boundaries
X_LOW = -100
X_HIGH = 100
Y_LOW = -50
Y_HIGH = 250

# Define states of the grid cell
FREE_STATE = 0
PATH_STATE = 1
GOAL_STATE = 2
COLLISION_STATE = 3

class ASVEnv:
    metadata = {"render_modes": ["human"], "render_fps": 10}
    def __init__(self, render_mode = "human"):
        super(ASVEnv, self).__init__()
        self.render_mode = render_mode

        # Initialize main parameters
        self.width = WIDTH
        self.height = HEIGHT
        self.square_size = SQUARE_SIZE
        self.observation_radius = OBSTACLE_RADIUS

        self.boundary = [(0,0), (0,self.height), (self.width,self.height), (self.width,0), (0,0)]
        self.position = START
        self.goal = GOAL
        self.heading = INITIAL_HEADING
        self.speed = SPEED
        self.turn_rate = TURN_RATE

        # Action space and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=3, shape=(313,), dtype=np.int32)   
        # 3 possible actions: left, right, straight
        # 4 possible states for observation, and the shape = number of grids inside the observation radius
    
    # Generate grid function
    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i**2 + j**2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid
    
    # Generate static obstacles
    def generate_static_obstacles(self, num):
        obstacles = [np.array([100, 70]), np.array([100, 100])]         # Set 2 obstacles on the path
        for _ in range(num):
            pos = np.random.randint(0, [self.width, self.height])
            obstacles.append(pos)
        return obstacles

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize current position and heading angle
        self.position = START
        self.heading = INITIAL_HEADING

        self.generate_static_obstacles(3)

        observation = np.array([*self.position, self.heading, self.speed])
        
        return observation, {}
    
    def is_on_path(self, position):
        return
    
    def is_valid_pos(self, position):
        return
    
    def is_goal(self, position):
        return
    
    def is_collision(self, position):
        return
    
    def calculate_reward(self, position):
        return
    
    def step(self, action):
        return
    
    def render(self):
        return


