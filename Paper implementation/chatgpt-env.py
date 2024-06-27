import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
RADIUS = 100
SQUARE_SIZE = 10
SPEED = 10
OBSTACLE_RADIUS = SQUARE_SIZE / 3
WIDTH = 50
HEIGHT = 250  # Adjusted height to accommodate the goal
START = (0, 0)
GOAL = (0, 200)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 200 / SPEED

# Define colors for visualization
COLOR_FREE = 0
COLOR_PATH = 1
COLOR_GOAL = 2
COLOR_OBSTACLE = 3

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
        self.taken_steps = []  # To store the agent's taken steps
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # accelerate, decelerate, turn left, turn right, do nothing
        grid_shape = (2 * self.observation_radius // self.square_size,) * 2
        self.observation_space = spaces.Box(low=0, high=3, shape=grid_shape, dtype=np.int32)
        
        # Initialize global map and obstacles
        self._initialize_global_map()
        self.reset()

    def _initialize_global_map(self):
        self.global_map = np.zeros((self.width, self.height), dtype=np.int32)
        
        # Add path to the global map
        for y in range(START[1], GOAL[1] + 1):
            if 0 <= y < self.height:
                self.global_map[START[0], y] = COLOR_PATH

        # Add goal to the global map
        self.global_map[GOAL[0], GOAL[1]] = COLOR_GOAL

    def _generate_static_obstacles(self, num):
        obstacles = []
        for _ in range(num):
            pos = np.random.randint(0, [self.width, self.height])
            obstacles.append(pos)
        return obstacles

    def _generate_dynamic_obstacles(self, num):
        obstacles = []
        for _ in range(num):
            pos = np.random.randint(0, [self.width, self.height])
            direction = np.random.randint(0, 360)
            speed = np.random.uniform(1, 3)
            obstacles.append([pos, direction, speed])
        return obstacles

    def reset(self):
        self.position = self.start_pos.copy()
        self.heading = INITIAL_HEADING
        self.speed = SPEED
        self.step_count = 0
        self.taken_steps = [self.start_pos.tolist()]
        
        # Generate new obstacle positions
        self.static_obstacles = self._generate_static_obstacles(num=4)
        self.dynamic_obstacles = self._generate_dynamic_obstacles(num=3)

        # Add static obstacles to the global map
        self.global_map.fill(COLOR_FREE)
        for obstacle in self.static_obstacles:
            if 0 <= obstacle[0] < self.width and 0 <= obstacle[1] < self.height:
                self.global_map[obstacle[0], obstacle[1]] = COLOR_OBSTACLE
        
        # Add path and goal back to the global map
        self._initialize_global_map()
        
        return self._get_observation()

    def step(self, action):
        if action == 0:  # Accelerate
            self.speed = min(self.speed + 0.5, 5)
        elif action == 1:  # Decelerate
            self.speed = max(self.speed - 0.5, 1)
        elif action == 2:  # Turn left
            self.heading = (self.heading - 5) % 360
        elif action == 3:  # Turn right
            self.heading = (self.heading + 5) % 360
        
        self.position[0] += self.speed * np.cos(np.radians(self.heading))
        self.position[1] += self.speed * np.sin(np.radians(self.heading))
        
        self._move_dynamic_obstacles()
        
        self.taken_steps.append(self.position.tolist())
        
        done, reward = self._check_done()
        self.step_count += 1
        
        obs = self._get_observation()
        return obs, reward, done, {}

    def _move_dynamic_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            pos, direction, speed = obstacle
            pos[0] += speed * np.cos(np.radians(direction))
            pos[1] += speed * np.sin(np.radians(direction))
            if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
                pos[:] = np.random.randint(0, [self.width, self.height])
                direction = np.random.randint(0, 360)

    def _check_done(self):
        if np.linalg.norm(self.goal - self.position) <= 13:
            return True, 50  # Goal reached
        for obstacle in self.static_obstacles:
            if np.linalg.norm(obstacle - self.position) <= 15:
                return True, -100  # Collision with static obstacle
        for pos, _, _ in self.dynamic_obstacles:
            if np.linalg.norm(pos - self.position) <= 15:
                return True, -100  # Collision with dynamic obstacle
        if self.step_count >= self.max_steps:
            return True, -10  # Max steps reached
        return False, -10  # Default penalty for each step

    def _get_observation(self):
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
            self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 8))
            self.ax1, self.ax2 = self.axs
            self.ax1.set_aspect('equal')
            self.ax2.set_aspect('equal')

        self.ax1.clear()
        self.ax2.clear()

        # Plot global map
        for i in range(self.width):
            for j in range(self.height):
                color = 'white'
                if self.global_map[i, j] == COLOR_OBSTACLE:
                    color = 'red'
                elif self.global_map[i, j] == COLOR_PATH:
                    color = 'green'
                elif self.global_map[i, j] == COLOR_GOAL:
                    color = 'yellow'
                self.ax1.add_patch(plt.Rectangle((i, j), 1, 1, edgecolor='gray', facecolor=color, alpha=0.5))

        # Plot agent's taken steps on the global map
        for step in self.taken_steps:
            self.ax1.add_patch(plt.Circle(np.array(step) / self.square_size, OBSTACLE_RADIUS, color='blue', alpha=0.3))

        # Plot current position of the agent on the global map
        self.ax1.add_patch(plt.Circle(self.position / self.square_size, OBSTACLE_RADIUS, color='blue'))

        # Plot dynamic obstacles on the global map
        for pos, _, _ in self.dynamic_obstacles:
            self.ax1.add_patch(plt.Circle(pos / self.square_size, OBSTACLE_RADIUS, color='red'))

        # Plot observation grid
        grid = self._get_observation()
        grid_size = grid.shape[0]
        for i in range(grid_size):
            for j in range(grid_size):
                color = 'white'
                if grid[i, j] == COLOR_OBSTACLE:
                    color = 'red'
                elif grid[i, j] == COLOR_PATH:
                    color = 'green'
                elif grid[i, j] == COLOR_GOAL:
                    color = 'yellow'
                self.ax2.add_patch(plt.Rectangle((i * self.square_size - self.observation_radius,
                                                  j * self.square_size - self.observation_radius),
                                                 self.square_size, self.square_size, edgecolor='gray', facecolor=color, alpha=0.5))

        self.ax1.set_xlim(-0.5, self.width - 0.5)
        self.ax1.set_ylim(-0.5, self.height - 0.5)
        self.ax2.set_xlim(-self.observation_radius // self.square_size, self.observation_radius // self.square_size)
        self.ax2.set_ylim(-self.observation_radius // self.square_size, self.observation_radius // self.square_size)

        plt.draw()
        plt.pause(0.001)

# Example usage
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
