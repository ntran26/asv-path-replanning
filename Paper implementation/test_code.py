import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import gym
from gym import spaces

# Define constants
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

WIDTH = 200
HEIGHT = 300
START = (100, 30)
GOAL = (100, 250)
NUM_STATIC_OBS = 5

RADIUS = 100
SQUARE_SIZE = 10
SPEED = 2
OBSTACLE_RADIUS = SQUARE_SIZE / 3

INITIAL_HEADING = 90
TURN_RATE = 5

FREE_STATE = 0
PATH_STATE = 1
COLLISION_STATE = 2
GOAL_STATE = 3

class ASVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ASVEnv, self).__init__()
        self.width = WIDTH
        self.height = HEIGHT
        self.heading = INITIAL_HEADING
        self.turn_rate = TURN_RATE
        self.speed = SPEED
        # self.num_step = STEP
        self.start = START
        self.goal = GOAL
        self.radius = RADIUS
        self.grid_size = SQUARE_SIZE
        self.center_point = (0, 0)

        self.obstacles = self.generate_static_obstacles(5, self.width, self.height)
        self.path = self.generate_path(self.start, self.goal)
        self.boundary = self.generate_border(self.width, self.height)
        self.goal_point = self.generate_goal(self.goal)
        self.objects_environment = self.obstacles + self.path + self.boundary + self.goal_point
        self.grid_dict = self.fill_grid(self.objects_environment, self.grid_size)

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Actions: 0 = go straight, 1 = turn left, 2 = turn right
        self.observation_space = spaces.Box(low=0, high=3, shape=(313,), dtype=np.int32)  # 313 grid points within the observation radius

        # Initialize other variables
        self.reset()

    # Helper functions (same as before)
    def get_priority_state(self, current_state, new_state):
        if new_state == COLLISION_STATE:
            return COLLISION_STATE
        elif new_state == GOAL_STATE and current_state != COLLISION_STATE:
            return GOAL_STATE
        elif new_state == PATH_STATE and current_state not in (COLLISION_STATE, GOAL_STATE):
            return PATH_STATE
        elif current_state not in (COLLISION_STATE, GOAL_STATE, PATH_STATE):
            return FREE_STATE
        return current_state
    
    def closest_multiple(self, n, mult):
        return int((n + mult / 2) // mult) * mult
    
    def fill_grid(self, objects, grid_size):
        grid_dict = {}
        for obj in objects:
            m = obj['x']
            n = obj['y']
            state = obj['state']

            m = self.closest_multiple(m, grid_size)
            n = self.closest_multiple(n, grid_size)

            if (m, n) not in grid_dict:
                grid_dict[(m, n)] = FREE_STATE
            
            grid_dict[(m, n)] = self.get_priority_state(grid_dict[(m, n)], state)
        return grid_dict

    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid

    def generate_border(self, map_width, map_height):
        boundary = []
        for x in range(0, map_width + 1):
            boundary.append({'x': x, 'y': 0, 'state': COLLISION_STATE})             # lower boundary
            boundary.append({'x': x, 'y': map_height, 'state': COLLISION_STATE})    # upper boundary
        for y in range(0, map_height + 1):
            boundary.append({'x': 0, 'y': y, 'state': COLLISION_STATE})             # left boundary
            boundary.append({'x': map_width, 'y': y, 'state': COLLISION_STATE})     # right boundary
        return boundary

    def generate_static_obstacles(self, num_obs, map_width, map_height):
        obstacles = []
        for _ in range(num_obs):
            x = np.random.randint(0, map_width)
            y = np.random.randint(0, map_height)
            obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        for _ in range(2):
            x = self.start[0]
            y = np.random.randint(self.start[1] + 20, self.goal[1] - 20)
            obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        return obstacles
    
    def generate_path(self, start_point, goal_point):
        path = []
        num_points = goal_point[1] - start_point[1]
        for i in range(num_points):
            y = start_point[1] + i
            path.append({'x': start_point[0], 'y': y, 'state': PATH_STATE})
        return path
    
    def generate_goal(self, goal_point):
        goal = []
        goal.append({'x': goal_point[0], 'y': goal_point[1], 'state': GOAL_STATE})
        return goal

    def reset(self):
        self.step_count = 0
        self.current_heading = self.heading
        self.position = self.start
        self.done = False
        self.grid = self.generate_grid(self.radius, self.grid_size, self.center_point)
        return self._get_observation()

    def _get_observation(self):
        current_pos = self.position
        new_grid = self.generate_grid(self.radius, self.grid_size, current_pos)
        observation = np.zeros(len(new_grid), dtype=np.int32)
        for idx, (cx, cy) in enumerate(new_grid):
            state = self.grid_dict.get((self.closest_multiple(cx, self.grid_size), self.closest_multiple(cy, self.grid_size)), FREE_STATE)
            observation[idx] = state
        return observation

    def step(self, action):
        if action == 0:
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                             self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 1:
            self.current_heading += self.turn_rate
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                             self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 2:
            self.current_heading -= self.turn_rate
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                             self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))

        self.step_count += 1
        reward = self._get_reward()
        self.done = self._check_done()
        observation = self._get_observation()
        return observation, reward, self.done, {}

    def _get_reward(self):
        current_pos = self.position
        cx, cy = current_pos
        state = self.grid_dict.get((self.closest_multiple(cx, self.grid_size), self.closest_multiple(cy, self.grid_size)), FREE_STATE)
        if state == COLLISION_STATE:
            return -100
        elif state == GOAL_STATE:
            return 100
        elif state == PATH_STATE:
            return 10
        else:
            return -1

    def _check_done(self):
        current_pos = self.position
        if self.position[1] >= self.goal[1] or self.grid_dict.get((self.closest_multiple(current_pos[0], self.grid_size), self.closest_multiple(current_pos[1], self.grid_size)), FREE_STATE) == COLLISION_STATE:
            return True
        return False

    def render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, 'fig'):
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 8))

                self.ax1.set_aspect('equal')
                self.ax1.set_title('MAP')
                self.ax1.set_xlim(-self.radius, self.width + self.radius)
                self.ax1.set_ylim(-self.radius, self.height + self.radius)

                self.ax2.set_aspect('equal')
                self.ax2.set_title('OBSERVATION')
                self.ax2.set_xlim(-self.radius, self.radius)
                self.ax2.set_ylim(-self.radius, self.radius)

                self.agent_1, = self.ax1.plot([], [], marker='^', color=BLUE)
                self.agent_2, = self.ax2.plot([], [], marker='^', color=BLUE)
                self.observation_horizon1 = plt.Circle(self.start, self.radius, color=RED, fill=False)
                self.observation_horizon2 = plt.Circle((0, 0), self.radius, color=RED, fill=False)
                self.ax1.add_patch(self.observation_horizon1)
                self.ax2.add_patch(self.observation_horizon2)

                self.ax1.plot(self.start[0], self.start[1], marker='o', color=BLUE)
                self.ax1.plot(self.goal[0], self.goal[1], marker='o', color=YELLOW)

                for obj in self.boundary:
                    boundary_line = plt.Rectangle((obj['x'], obj['y']), 1, 1, edgecolor=BLACK, facecolor=BLACK)
                    self.ax1.add_patch(boundary_line)
                
                path_x = [point['x'] for point in self.path]
                path_y = [point['y'] for point in self.path]
                self.ax1.plot(path_x, path_y, '-', color=GREEN)

                for obj in self.obstacles:
                    self.ax1.plot(obj['x'], obj['y'], marker='o', color=RED)

            self.agent_1.set_data(self.position[0], self.position[1])
            self.agent_1.set_marker((3, 0, self.current_heading - 90))

            self.observation_horizon1.center = self.position

            new_grid = self.generate_grid(self.radius, self.grid_size, self.position)
            for rect in getattr(self, 'grid_patches', []):
                rect.remove()
            self.grid_patches = []

            for (cx, cy) in new_grid:
                state = self.grid_dict.get((self.closest_multiple(cx, self.grid_size), self.closest_multiple(cy, self.grid_size)), FREE_STATE)
                color = 'white'
                if state == COLLISION_STATE:
                    color = 'red'
                elif state == PATH_STATE:
                    color = 'green'
                elif state == GOAL_STATE:
                    color = 'yellow'
                rect = plt.Rectangle((cx - self.grid_size / 2 - self.position[0], cy - self.grid_size / 2 - self.position[1]), self.grid_size, self.grid_size,
                                    edgecolor='gray', facecolor=color)
                rect.set_zorder(1)
                self.grid_patches.append(rect)
                self.ax2.add_patch(rect)

            self.agent_2.set_data(0, 0)
            self.agent_2.set_marker((3, 0, self.current_heading - 90))
            self.agent_2.set_zorder(3)

            self.observation_horizon2.center = (0, 0)
            self.observation_horizon2.set_zorder(2)

            plt.draw()
            plt.pause(0.01)

# # Test the environment with random actions
# if __name__ == '__main__':
#     env = ASVEnv()
#     obs = env.reset()

#     for _ in range(100):  # Run for 100 steps or until done
#         action = env.action_space.sample()  # Take a random action
#         obs, reward, done, info = env.step(action)
#         env.render()

#         if done:
#             break

#     env.close()

print(int(1e6))
