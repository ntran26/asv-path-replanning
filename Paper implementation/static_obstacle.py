import numpy as np
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
OBSTACLE_RADIUS = SQUARE_SIZE / 3

# Define map dimensions
WIDTH = 200
HEIGHT = 300
START = (50, 50)
GOAL = (50, 200)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 150 / SPEED

def closest_multiple(n, mult):
    return int((n + mult / 2) // mult) * mult

class asv_visualization:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.heading = INITIAL_HEADING
        self.speed = SPEED
        self.turn_rate = TURN_RATE
        self.start_pos = START
        self.step = STEP
        self.goal = GOAL
        self.grid_dict = {}  # Initialize grid dictionary

    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid

    def generate_static_obstacles(self, num):
        obstacles = [{'pos': np.array([50, 70]), 'weight': 0.9}, {'pos': np.array([50, 100]), 'weight': 0.9}]
        for _ in range(num):
            pos = np.random.randint(0, [100, 250])
            obstacles.append({'pos': pos, 'weight': 0.9})
        return obstacles

    def draw_path(self):
        self.step_count = 0
        self.speed = self.speed
        self.current_heading = self.heading
        self.left_path = [self.start_pos]
        self.left_heading = [self.heading]
        self.position = self.start_pos

        while self.step_count < self.step:
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                             self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
            self.left_path.append(self.position)
            self.left_heading.append(self.current_heading)
            self.current_heading = self.current_heading
            self.step_count += 1

        static_obstacles = self.generate_static_obstacles(3)

        boundary = []
        for x in range(0, 100 + 1):
            boundary.append((x, 0))
            boundary.append((x, 250))
        for y in range(0, 250 + 1):
            boundary.append((0, y))
            boundary.append((100, y))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        self.agent_1, = ax1.plot([], [], marker='^', color=BLUE)
        self.agent_2, = ax2.plot([], [], marker='^', color=BLUE)
        observation_horizon1 = plt.Circle(START, RADIUS, color='r', fill=False)
        observation_horizon2 = plt.Circle((0, 0), RADIUS, color='r', fill=False)  # Static center
        ax1.add_patch(observation_horizon1)
        ax2.add_patch(observation_horizon2)

        ax1.plot(GOAL[0], GOAL[1], marker='o', color=YELLOW)

        for (x, y) in boundary:
            boundary_line = plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black')
            ax1.add_patch(boundary_line)

        self.path = []
        for y in range(START[1], GOAL[1] + 1):
            self.path.append((START[0], y))

        path_array = np.array(self.path)
        ax1.plot(path_array[:, 0], path_array[:, 1], '-', color=GREEN)

        for obj in static_obstacles:
            x, y = obj['pos']
            ax1.plot(x, y, marker='o', color=RED)

        squares_ax2 = []

        def init():
            self.agent_1.set_data([], [])
            self.agent_2.set_data([], [])
            observation_horizon1.center = START
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, (0, 0))  # Static grid for second plot
            for (cx, cy) in grid:
                rect = plt.Rectangle((cx - SQUARE_SIZE / 2, cy - SQUARE_SIZE / 2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor='none')
                ax2.add_patch(rect)
                squares_ax2.append(rect)
            return self.agent_1, self.agent_2, observation_horizon1, observation_horizon2, *squares_ax2

        def calculate_weight(obj_type, distance):
            if obj_type == "goal":
                return 1
            elif obj_type == "wall" or obj_type == "static":
                return 0.9
            elif obj_type == "ship":
                return 0.8
            elif obj_type == "ship_prediction":
                return -0.8 * (0.85 ** distance)
            return 0

        def reset():
            for square in squares_ax2:
                square.remove()
            squares_ax2.clear()

        def update(frame):
            agent_pos = self.left_path[frame]
            self.agent_1.set_data(agent_pos[0], agent_pos[1])
            self.agent_2.set_data(0, 0)  # Static center

            # Update the observation circle in the first plot to move with the agent
            observation_horizon1.center = agent_pos

            # Reset gridDict for the new frame
            self.grid_dict.clear()

            # Draw new grid squares
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, (0, 0))  # Static grid for second plot
            for (cx, cy) in grid:
                self.grid_dict[(cx, cy)] = {'coordinates': (cx, cy), 'weight': 0}  # Initialize gridDict with default weights

            # Update gridDict based on the current environment state within observation range
            for obj in static_obstacles + [{'pos': self.goal, 'weight': 1}] + [{'pos': p, 'weight': 0.8} for p in self.path]:
                obj_pos = obj['pos']
                if np.linalg.norm(np.array(agent_pos) - np.array(obj_pos)) <= RADIUS:
                    m = obj_pos[0]
                    n = obj_pos[1]
                    m = np.sign(m) * closest_multiple(abs(m), SQUARE_SIZE)
                    n = np.sign(n) * closest_multiple(abs(n), SQUARE_SIZE)
                    if (m, n) in self.grid_dict:
                        self.grid_dict[(m, n)]['weight'] = obj['weight']

            # Update the static grid squares based on gridDict
            reset()  # Remove previous grid squares
            print(self.grid_dict)
            for (cx, cy), data in self.grid_dict.items():
                weight = data['weight']
                color = 'none'
                if weight == 1:
                    color = YELLOW
                elif weight == 0.9:
                    color = RED
                elif weight == 0.8:
                    color = GREEN

                rect = plt.Rectangle((cx - SQUARE_SIZE / 2, cy - SQUARE_SIZE / 2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor=color)
                ax2.add_patch(rect)
                squares_ax2.append(rect)

            return self.agent_1, self.agent_2, observation_horizon1, observation_horizon2, *squares_ax2

        ani = FuncAnimation(fig, update, frames=len(self.left_path), init_func=init, blit=True)
        plt.show()

        # Save animation
        # writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save("asv_navigation.mp4", writer=writer)

asv = asv_visualization(WIDTH, HEIGHT)
asv.draw_path()
