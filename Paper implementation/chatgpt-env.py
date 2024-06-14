import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.animation import FuncAnimation

class ShipNavigationEnv:
    def __init__(self, grid_size=350, horizon=120, square_size=15):
        self.grid_size = grid_size
        self.horizon = horizon
        self.square_size = square_size
        
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.agent_dir = 0  # Direction in degrees
        self.agent_speed = 1  # Speed in m/s
        self.goal = np.random.randint(20, self.grid_size - 20, size=2)
        
        self.static_obstacles = self._generate_obstacles(num=3)
        self.moving_obstacles = self._generate_obstacles(num=3, moving=True)
        
    def _generate_obstacles(self, num, moving=False):
        obstacles = []
        for _ in range(num):
            pos = np.random.randint(20, self.grid_size - 20, size=2)
            if moving:
                direction = np.random.randint(0, 360)
                speed = np.random.uniform(0.5, 2)
                obstacles.append([pos, direction, speed])
            else:
                obstacles.append(pos)
        return obstacles

    def step(self, action):
        if action == 0:  # Accelerate
            self.agent_speed = min(self.agent_speed + 0.5, 5)
        elif action == 1:  # Decelerate
            self.agent_speed = max(self.agent_speed - 0.5, 1)
        elif action == 2:  # Turn left
            self.agent_dir = (self.agent_dir - 5) % 360
        elif action == 3:  # Turn right
            self.agent_dir = (self.agent_dir + 5) % 360
        
        self.agent_pos[0] += self.agent_speed * np.cos(np.deg2rad(self.agent_dir))
        self.agent_pos[1] += self.agent_speed * np.sin(np.deg2rad(self.agent_dir))
        
        self._move_obstacles()
        
        return self._check_collision()

    def _move_obstacles(self):
        for obstacle in self.moving_obstacles:
            pos, direction, speed = obstacle
            pos[0] += speed * np.cos(np.deg2rad(direction))
            pos[1] += speed * np.sin(np.deg2rad(direction))
            if pos[0] < 0 or pos[0] > self.grid_size or pos[1] < 0 or pos[1] > self.grid_size:
                pos[:] = np.random.randint(20, self.grid_size - 20, size=2)
                direction = np.random.randint(0, 360)

    def _check_collision(self):
        if np.linalg.norm(self.goal - self.agent_pos) <= 13:
            return True, "Goal"
        for obstacle in self.static_obstacles:
            if np.linalg.norm(obstacle - self.agent_pos) <= 15:
                return True, "Static Obstacle"
        for pos, _, _ in self.moving_obstacles:
            if np.linalg.norm(pos - self.agent_pos) <= 15:
                return True, "Moving Obstacle"
        return False, None

    def reset(self):
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.agent_dir = 0
        self.agent_speed = 1
        self.goal = np.random.randint(20, self.grid_size - 20, size=2)
        self.static_obstacles = self._generate_obstacles(num=3)
        self.moving_obstacles = self._generate_obstacles(num=3, moving=True)
        return self._get_observation()

    def _get_observation(self):
        # Generate a collision grid based on the current state
        grid = np.zeros((2 * self.horizon // self.square_size, 2 * self.horizon // self.square_size))
        for obstacle in self.static_obstacles:
            x, y = obstacle
            if self._within_horizon(x, y):
                grid[self._grid_index(x, y)] = 1
        for pos, dir, speed in self.moving_obstacles:
            x, y = pos
            if self._within_horizon(x, y):
                grid[self._grid_index(x, y)] = 1
        return grid.flatten()

    def _within_horizon(self, x, y):
        return np.linalg.norm([x - self.agent_pos[0], y - self.agent_pos[1]]) <= self.horizon

    def _grid_index(self, x, y):
        gx = (x - self.agent_pos[0] + self.horizon) // self.square_size
        gy = (y - self.agent_pos[1] + self.horizon) // self.square_size
        return int(gx), int(gy)

    def render(self, ax):
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        for obstacle in self.static_obstacles:
            ax.add_patch(Rectangle(obstacle - 5, 10, 10, color='blue'))
        
        for pos, _, _ in self.moving_obstacles:
            ax.add_patch(Rectangle(pos - 5, 10, 10, color='red'))
        
        ax.scatter(self.goal[0], self.goal[1], color='green', s=100)
        ax.scatter(self.agent_pos[0], self.agent_pos[1], color='black', s=100)
        ax.arrow(self.agent_pos[0], self.agent_pos[1], 10 * np.cos(np.deg2rad(self.agent_dir)), 10 * np.sin(np.deg2rad(self.agent_dir)), head_width=5, head_length=10, fc='black', ec='black')

# Initialize the environment
env = ShipNavigationEnv()
done = False

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Function to update the plot
def update(frame):
    global done
    if not done:
        action = np.random.choice([0, 1, 2, 3])  # Random action for demonstration
        done, reason = env.step(action)
        env.render(ax)
        ax.set_title(f"Episode in progress")
    else:
        ax.set_title(f"Episode finished")

# Create animation
ani = FuncAnimation(fig, update, frames=range(200), repeat=False)

# Display the animation
plt.show()
