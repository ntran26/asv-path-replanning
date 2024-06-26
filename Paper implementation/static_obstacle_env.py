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

# Define state
FREE_STATE = 0
COLLISION_STATE = 1
PATH_STATE = 2
GOAL_STATE = 3

class StaticObsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}
    def __init__(self, render_mode='human'):
        self.render_mode = render_mode

        # Set width, height and boundary
        self.width = WIDTH
        self.height = HEIGHT
        self.boundary = [(0,0), (0,self.height), (self.width,self.height), (self.width,0), (0,0)]
        self.observation_radius = RADIUS
        self.square_size = SQUARE_SIZE

        # ASV parameters
        self.start_pos = np.array(START)
        self.goal_pos = np.array(GOAL)
        self.heading = INITIAL_HEADING
        self.turn_rate = TURN_RATE
        self.speed = SPEED
        self.step_count = 0

        # Observation space and action space
        self.action_space = spaces.Discrete(3)
        grid_shape = (2 * self.observation_radius // self.square_size,) * 2
        self.observation_space = spaces.Box(low=0, high=3, shape=grid_shape, dtype=np.int32)

        self.reset()
    
    def init_global_map(self):
        self.global_map = np.zeros((self.width, self.height), dtype=np.int32)

        # Create and add path to the global map
        for y in range(self.start_pos[1], self.goal_pos[1]):
            self.global_map[self.start_pos[0], y] = PATH_STATE
        
        # Add goal point to the global map
        self.global_map[self.goal_pos[0], self.goal_pos[1]] = GOAL_STATE

        # Create boundary
        self.boundary = []
        for x in range(-100, 100 + 1):
            self.boundary.append((x, -50))  # lower boundary
            self.boundary.append((x, 250))  # upper boundary 
        for y in range(-50, 250 + 1):
            self.boundary.append((-100, y))  # left boundary
            self.boundary.append((100, y))   # right boundary

    def generate_static_obstacles(self, num):
        obstacles = []
        for _ in range(num):
            pos = np.random.randint(-100, 100, size=2)
            obstacles.append(pos)
        return obstacles
    
    def get_observation(self):
        grid_size = 2 * self.observation_radius // self.square_size
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)

        # Populate the grid with static obstacles
        for obstacle in self.static_obstacles:
            if np.linalg.norm(obstacle - self.position) <= self.observation_radius:
                obs_pos = ((obstacle - self.position + self.observation_radius) // self.square_size).astype(int)
                grid[obs_pos[0], obs_pos[1]] = COLLISION_STATE
        
        # Populate the grid with path
        path = self._generate_path()
        for px, py in path:
            if 0 <= px < grid_size and 0 <= py < grid_size:
                if grid[px, py] == 0:   # Don't overwrite obstacles
                    grid[px, py] = PATH_STATE

        # Populate the grid with goal
        if np.linalg.norm(self.goal - self.position) <= self.observation_radius:
            goal_pos = ((self.goal - self.position + self.observation_radius) // self.square_size).astype(int)
            grid[goal_pos[0], goal_pos[1]] = GOAL_STATE
        
        return grid

    # Reset function
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.position = self.start_pos.copy()
        self.heading = INITIAL_HEADING
        self.speed = SPEED
        self.step_count = 0

        self.static_obstacles = self.generate_static_obstacles(num=4)

        # Add static obstacles to the global map
        for obstacle in self.static_obstacles:
            if 0 <= obstacle[0] < self.width and 0 <= obstacle[1] < self.height:
                self.global_map[obstacle[0], obstacle[1]] = COLLISION_STATE

        return self.get_observation(), {}
    
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
        if action == 0:     # Turn left
            self.heading += 5
        if action == 1:     # Turn right
            self.heading -= 5
        if action == 2:     # Go straight
            self.heading = self.heading
        # Update ASV position
        self.position[0] += self.speed * np.cos(np.radians(self.heading))
        self.position[1] += self.speed * np.sin(np.radians(self.heading))
        return
    
    # Render function
    def render(self):
        return
    


