import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Define colors and states
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

FREE_STATE = 0
PATH_STATE = 1
COLLISION_STATE = 2
GOAL_STATE = 3

# Constants
RADIUS = 100
SQUARE_SIZE = 10
MAP_WIDTH = 200
MAP_HEIGHT = 300
INITIAL_CENTER_POINT = (0, 100)
NUM_OBSTACLES = 20
NUM_PATH_POINTS = 50
SPEED = 2
STEP = 200 / SPEED

def get_priority_state(current_state, new_state):
    if new_state == COLLISION_STATE:
        return COLLISION_STATE
    elif new_state == GOAL_STATE and current_state != COLLISION_STATE:
        return GOAL_STATE
    elif new_state == PATH_STATE and current_state not in (COLLISION_STATE, GOAL_STATE):
        return PATH_STATE
    elif current_state not in (COLLISION_STATE, GOAL_STATE, PATH_STATE):
        return FREE_STATE
    return current_state

def closest_multiple(n, mult):
    return int((n + mult / 2) // mult) * mult

def fill_grid(objects, dc):
    grid_dict = {}
    for obj in objects:
        m = obj['x']
        n = obj['y']
        state = obj['state']

        m = closest_multiple(m, dc)
        n = closest_multiple(n, dc)
        if (m, n) not in grid_dict:
            grid_dict[(m, n)] = FREE_STATE
        
        grid_dict[(m, n)] = get_priority_state(grid_dict[(m, n)], state)
    return grid_dict

def generate_grid(radius, square_size, center):
    x = np.arange(-radius + square_size, radius, square_size)
    y = np.arange(-radius + square_size, radius, square_size)
    grid = []
    for i in x:
        for j in y:
            if np.sqrt(i ** 2 + j ** 2) <= radius:
                grid.append((center[0] + i, center[1] + j))
    return grid

def generate_random_obstacles(num_obstacles, map_width, map_height):
    obstacles = []
    for _ in range(num_obstacles):
        x = np.random.randint(0, map_width)
        y = np.random.randint(0, map_height)
        obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
    return obstacles

def generate_path(initial_point, num_points, vertical_distance):
    path = []
    for i in range(num_points):
        y = initial_point[1] + i * vertical_distance
        path.append({'x': initial_point[0], 'y': y, 'state': PATH_STATE})
    return path

class ASVEnv:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.heading = 90
        self.speed = SPEED
        self.turn_rate = 5
        self.start_pos = INITIAL_CENTER_POINT
        self.step = STEP
        self.goal = (0, 200)
        
        self.obstacles = generate_random_obstacles(NUM_OBSTACLES, self.width, self.height)
        self.path = generate_path(self.start_pos, NUM_PATH_POINTS, SQUARE_SIZE)
        self.objects_environment = self.obstacles + self.path
        self.grid_dict = fill_grid(self.objects_environment, SQUARE_SIZE)

    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid

    def draw_path(self):
        # Initialize figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        self.agent_1, = ax1.plot([], [], marker='^', color=BLUE)
        self.agent_2, = ax2.plot([], [], marker='^', color=BLUE)
        observation_horizon1 = plt.Circle(self.start_pos, RADIUS, color='r', fill=False)
        observation_horizon2 = plt.Circle((0, 0), RADIUS, color='r', fill=False)
        ax1.add_patch(observation_horizon1)
        ax2.add_patch(observation_horizon2)

        # Plot goal point
        ax1.plot(self.goal[0], self.goal[1], marker='o', color=YELLOW)

        # Plot the boundary
        boundary = []
        for x in range(-100, 100 + 1):
            boundary.append((x, -50))  # lower boundary
            boundary.append((x, 250))  # upper boundary 
        for y in range(-50, 250 + 1):
            boundary.append((-100, y))  # left boundary
            boundary.append((100, y))   # right boundary 
        for (x, y) in boundary:
            boundary_line = plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black')
            ax1.add_patch(boundary_line)

        # Plot the path
        path_x = [point['x'] for point in self.path]
        path_y = [point['y'] for point in self.path]
        ax1.plot(path_x, path_y, '-', color=GREEN)

        # Plot obstacles
        for obj in self.obstacles:
            ax1.plot(obj['x'], obj['y'], marker='o', color=RED)

        # Initialize animation variables
        squares_ax2 = []

        def init():
            self.agent_1.set_data([], [])           # agent in the first plot
            self.agent_2.set_data([], [])           # agent in the second plot
            observation_horizon1.center = self.start_pos
            observation_horizon2.center = (0, 0)
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, self.start_pos)
            for (cx, cy) in grid:
                rect = plt.Rectangle((cx - SQUARE_SIZE/2, cy - SQUARE_SIZE/2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor='none')
                ax1.add_patch(rect)
                squares_ax2.append(rect)
            return self.agent_1, self.agent_2, observation_horizon1, observation_horizon2, *squares_ax2

        def reset():
            for rect in squares_ax2:
                rect.remove()
            squares_ax2.clear()

        def update(frame):
            agent_pos = (path_x[frame], path_y[frame])
            self.agent_1.set_data(agent_pos[0], agent_pos[1])
            self.agent_1.set_marker((3, 0, self.heading - 90))
            
            self.agent_2.set_data(0, 0)
            self.agent_2.set_marker((3, 0, self.heading - 90))

            observation_horizon1.center = agent_pos
            observation_horizon2.center = (0, 0)

            reset()

            grid = self.generate_grid(RADIUS, SQUARE_SIZE, agent_pos)
            for (cx, cy) in grid:
                state = self.grid_dict.get((closest_multiple(cx, SQUARE_SIZE), closest_multiple(cy, SQUARE_SIZE)), FREE_STATE)
                color = 'white'
                if state == COLLISION_STATE:
                    color = 'red'
                elif state == PATH_STATE:
                    color = 'green'
                elif state == GOAL_STATE:
                    color = 'yellow'
                rect = plt.Rectangle((cx - SQUARE_SIZE / 2 - agent_pos[0], cy - SQUARE_SIZE / 2 - agent_pos[1]),
                                     SQUARE_SIZE, SQUARE_SIZE, edgecolor='gray', facecolor=color)
                ax2.add_patch(rect)
                squares_ax2.append(rect)

            return self.agent_1, self.agent_2, observation_horizon1, observation_horizon2, *squares_ax2

        ani = FuncAnimation(fig, update, frames=len(path_x), init_func=init, blit=True, interval=200, repeat=False)
        ax1.set_xlim(-50, 250)
        ax1.set_ylim(-50, 250)
        ax2.set_xlim(-RADIUS, RADIUS)
        ax2.set_ylim(-RADIUS, RADIUS)

        plt.show()

# Create visualization
visualization = ASVEnv(MAP_WIDTH, MAP_HEIGHT)
visualization.draw_path()
