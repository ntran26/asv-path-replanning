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

# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Constants
RADIUS = 100
SQUARE_SIZE = 10
SPEED = 2
OBSTACLE_RADIUS = SQUARE_SIZE / 3

# Map dimensions
WIDTH = 200
HEIGHT = 300
START = (0, 0)
GOAL = (0, 200)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 200 / SPEED

# Define states of the grid cell
FREE_STATE = 0
PATH_STATE = 1
GOAL_STATE = 2
COLLISION_STATE = 3

class ASVEnv(gym.Env):
    def __init__(self):
        super(ASVEnv, self).__init__()
        self.width = WIDTH
        self.height = HEIGHT
        self.heading = INITIAL_HEADING
        self.speed = SPEED
        self.turn_rate = TURN_RATE
        self.start_pos = np.array(START)
        self.goal = np.array(GOAL)
        self.observation_radius = RADIUS
        self.square_size = SQUARE_SIZE
        self.max_steps = int(STEP)
        self.step_count = 0
        self.position = self.start_pos.copy()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # turn left, turn right, go straight
        grid_shape = (2 * self.observation_radius // self.square_size,) * 2
        self.observation_space = spaces.Box(low=0, high=3, shape=grid_shape, dtype=np.int32)
        
        # Initialize global map and obstacles
        self.init_global_map()
        self.reset()
    
    def init_global_map(self):
        self.global_map = np.zeros((self.width, self.height), dtype=np.int32)
        
        # Add path to the global map
        for y in range(START[1], GOAL[1] + 1):
            if 0 <= y < self.height:
                self.global_map[START[0], y] = PATH_STATE

        # Add goal to the global map
        self.global_map[GOAL[0], GOAL[1]] = GOAL_STATE

        # # Create boundary
        # self.boundary = []
        # for x in range(-100, 100 + 1):
        #     self.boundary.append((x, -50))  # lower boundary
        #     self.boundary.append((x, 250))  # upper boundary 
        # for y in range(-50, 250 + 1):
        #     self.boundary.append((-100, y))  # left boundary
        #     self.boundary.append((100, y))   # right boundary

    def generate_static_obstacles(self, num):
        obstacles = []
        for _ in range(num):
            pos = np.random.randint(0, [self.width, self.height])
            obstacles.append(pos)
        return obstacles

    # Reset function
    def reset(self):
        self.position = self.start_pos.copy()
        self.heading = INITIAL_HEADING
        self.speed = SPEED
        self.step_count = 0
        self.taken_steps = [self.start_pos.tolist()]
        
        # Generate new obstacle positions
        self.static_obstacles = self.generate_static_obstacles(num=4)

        # Add static obstacles to the global map
        self.global_map.fill(FREE_STATE)
        for obstacle in self.static_obstacles:
            if 0 <= obstacle[0] < self.width and 0 <= obstacle[1] < self.height:
                self.global_map[obstacle[0], obstacle[1]] = COLLISION_STATE
        
        # Add path and goal back to the global map
        self.init_global_map()
        
        return self.get_observation()
    
    # Check if the ASV is on path
    def is_on_path(self, position):
        return
    
    # Check if the ASV is on the map
    def is_valid_pos(self, position):
        return
    
    # Check if the ASV reached the goal
    def is_goal(self, position):
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
        
        self.taken_steps.append(self.position.tolist())
        
        self.step_count += 1
        done, reward = self._check_done()
        
        obs = self._get_observation()
        return obs, reward, done, {}
    
    def check_done(self):
        if np.linalg.norm(self.goal - self.position) <= 13:
            return True, 0  # Goal reached
        for obstacle in self.static_obstacles:
            if np.linalg.norm(obstacle - self.position) <= 15:
                return True, 0  # Collision with static obstacle
        if self.step_count >= self.max_steps:
            return True, 0  # Max steps reached
        return False, 0  # Default

    def get_observation(self):
        grid_size = 2 * self.observation_radius // self.square_size
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Agent's position on the global map
        agent_pos_on_global = self.position // self.square_size

        for i in range(grid_size):
            for j in range(grid_size):
                global_i = int(agent_pos_on_global[0] - grid_size // 2 + i)
                global_j = int(agent_pos_on_global[1] - grid_size // 2 + j)
                if 0 <= global_i < self.width and 0 <= global_j < self.height:
                    grid[i, j] = self.global_map[global_i, global_j]
        
        return grid
    
    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(1, figsize=(6,8))
            self.ax.set_aspect('equal')

        self.ax.clear()

        # Plot global map
        for i in range(self.width):
            for j in range(self.height):
                color = 'white'
                if self.global_map[i, j] == COLLISION_STATE:
                    color = 'red'
                elif self.global_map[i, j] == PATH_STATE:
                    color = 'green'
                elif self.global_map[i, j] == GOAL_STATE:
                    color = 'yellow'
                self.ax.add_patch(plt.Rectangle((i, j), 1, 1, edgecolor='gray', facecolor=color, alpha=0.5))

        # Plot agent's taken steps on the global map
        for step in self.taken_steps:
            self.ax.add_patch(plt.Circle(np.array(step) / self.square_size, OBSTACLE_RADIUS, color='blue', alpha=0.3))

        # Plot current position of the agent on the global map
        self.ax.add_patch(plt.Circle(self.position / self.square_size, OBSTACLE_RADIUS, color='blue'))

        # Plot observation circle
        obs_circle = plt.Circle(self.position / self.square_size, self.observation_radius / self.square_size, color='blue', fill=False)
        self.ax.add_patch(obs_circle)

        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(-0.5, self.height - 0.5)

        plt.draw()
        plt.pause(0.001)
    


