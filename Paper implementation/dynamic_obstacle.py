import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

#                               -------- CONFIGURATION --------
# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Define map dimensions and start/goal points, number of static obstacles
WIDTH = 200
HEIGHT = 300
START = (100, 30)
GOAL = (100, 250)
NUM_STATIC_OBS = 5

# Define observation radius and grid size
RADIUS = 100
SQUARE_SIZE = 10
SPEED = 2
OBSTACLE_RADIUS = SQUARE_SIZE/3
DYNAMIC_OBS_SPEED = 1

# Define initial heading angle, turn rate and number of steps
INITIAL_HEADING = 90
TURN_RATE = 5
STEP = (GOAL[1] - START[1]) / SPEED

# Define states
FREE_STATE = 0          # free space
PATH_STATE = 1          # path
COLLISION_STATE = 2     # obstacle or border
GOAL_STATE = 3          # goal point

class asv_visualisation:
    # Initialize environment
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.heading = INITIAL_HEADING
        self.turn_rate = TURN_RATE
        self.speed = SPEED
        self.step = STEP
        self.start = START
        self.goal = GOAL
        self.radius = RADIUS
        self.grid_size = SQUARE_SIZE
        self.center_point = (0, 0)
        self.dynamic_obstacle = {'x': 100, 'y': 250, 'speed': DYNAMIC_OBS_SPEED, 'heading': 270}  # Moving straight down

        self.obstacles = self.generate_static_obstacles(5, self.width, self.height)
        self.path = self.generate_path(self.start, self.goal)
        self.boundary = self.generate_border(self.width, self.height)
        self.goal_point = self.generate_goal(self.goal)
        self.objects_environment = self.obstacles + self.path + self.boundary + self.goal_point
        self.grid_dict = self.fill_grid(self.objects_environment, self.grid_size)

    #                           -------- HELPER FUNCTIONS --------

    # Create a function that sets the priority of each state in case they overlap: 
    # obstacle/collision > goal point > path > free space
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

    # Create a function that converts each point from the global map to a grid coordinate
    def closest_multiple(self, n, mult):
        return int((n + mult / 2) // mult) * mult

    # Create a function to generate a dictionary, storing the grid coordinates and state
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

    # Create a function that generate grid coordinates (x, y) from global map
    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid

    #                           -------- MAP GENERATION --------

    # Create a function to generate borders around the map
    def generate_border(self, map_width, map_height):
        boundary = []
        for x in range(0, map_width + 1):
            boundary.append({'x': x, 'y': 0, 'state': COLLISION_STATE})             # lower boundary
            boundary.append({'x': x, 'y': map_height, 'state': COLLISION_STATE})    # upper boundary
        for y in range(0, map_height + 1):
            boundary.append({'x': 0, 'y': y, 'state': COLLISION_STATE})             # left boundary
            boundary.append({'x': map_width, 'y': y, 'state': COLLISION_STATE})     # right boundary
        return boundary

    # Create a function to generate static obstacles
    def generate_static_obstacles(self, num_obs, map_width, map_height):
        obstacles = []
        # Generate random obstacles around the map
        for _ in range(num_obs):
            x = np.random.randint(0, map_width)
            y = np.random.randint(0, map_height)
            obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        # Generate 2 random obstacles along the path
        for _ in range(2):
            x = self.start[0]
            y = np.random.randint(self.start[1] + 20, self.goal[1] - 20)
            obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        return obstacles

    # Function that generate a path line (list/array of points)
    def generate_path(self, start_point, goal_point):
        path = []
        num_points = goal_point[1] - start_point[1]     # straight vertical line
        for i in range(num_points):
            y = start_point[1] + i
            path.append({'x': start_point[0], 'y': y, 'state': PATH_STATE})
        return path

    def generate_goal(self, goal_point):
        goal = []
        goal.append({'x': goal_point[0], 'y': goal_point[1], 'state': GOAL_STATE})
        return goal

    # Calculate relative bearing of the dynamic obstacle to the ASV
    def calculate_relative_bearing(self, asv_pos, obstacle_pos, asv_heading):
        dx = obstacle_pos[0] - asv_pos[0]
        dy = obstacle_pos[1] - asv_pos[1]
        angle_to_obstacle = np.degrees(np.arctan2(dy, dx))
        relative_bearing = (angle_to_obstacle - asv_heading + 360) % 360
        return relative_bearing

    #                           -------- MAIN LOOP --------
    def main(self):
        # Initialize figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        # First plot: whole map
        ax1.set_aspect('equal')
        ax1.set_title('MAP')
        ax1.set_xlim(-self.radius, self.width + self.radius)
        ax1.set_ylim(-self.radius, self.height + self.radius)

        # Second plot: observation per timestep
        ax2.set_aspect('equal')
        ax2.set_title('OBSERVATION')
        ax2.set_xlim(-self.radius, self.radius)
        ax2.set_ylim(-self.radius, self.radius)

        # Plot ASV and observation circle
        self.agent_1, = ax1.plot([], [], marker='^', color=BLUE)
        self.agent_2, = ax2.plot([], [], marker='^', color=BLUE)
        observation_horizon1 = plt.Circle(self.start, self.radius, color=RED, fill=False)
        observation_horizon2 = plt.Circle((0, 0), self.radius, color=RED, fill=False)
        ax1.add_patch(observation_horizon1)
        ax2.add_patch(observation_horizon2)

        # Plot start point
         # Plot start point
        ax1.plot(self.start[0], self.start[1], marker='o', color=GREEN)

        # Plot goal point
        ax1.plot(self.goal[0], self.goal[1], marker='o', color=YELLOW)

        # Plot static obstacles
        for obs in self.obstacles:
            circle = plt.Circle((obs['x'], obs['y']), OBSTACLE_RADIUS, color=BLACK)
            ax1.add_patch(circle)

        # Plot the path
        path_x = [point['x'] for point in self.path]
        path_y = [point['y'] for point in self.path]
        ax1.plot(path_x, path_y, color=WHITE)

        # Initialize dynamic obstacle
        self.dynamic_obs_plot, = ax1.plot([], [], marker='o', color=RED)

        # Animation function to update the plot
        def update(frame):
            # Update ASV position
            asv_x, asv_y = self.start
            asv_heading = self.heading
            self.agent_1.set_data(asv_x, asv_y)
            self.agent_2.set_data(0, 0)

            # Update dynamic obstacle position
            self.dynamic_obstacle['x'] += self.dynamic_obstacle['speed'] * np.cos(np.radians(self.dynamic_obstacle['heading']))
            self.dynamic_obstacle['y'] += self.dynamic_obstacle['speed'] * np.sin(np.radians(self.dynamic_obstacle['heading']))
            self.dynamic_obs_plot.set_data(self.dynamic_obstacle['x'], self.dynamic_obstacle['y'])

            # Check if dynamic obstacle is out of bounds and reset if necessary
            if self.dynamic_obstacle['x'] < 0 or self.dynamic_obstacle['x'] > self.width or \
               self.dynamic_obstacle['y'] < 0 or self.dynamic_obstacle['y'] > self.height:
                self.dynamic_obstacle['x'], self.dynamic_obstacle['y'] = 100, 250  # Reset to initial position

            # Calculate relative bearing
            rel_bearing = self.calculate_relative_bearing((asv_x, asv_y), (self.dynamic_obstacle['x'], self.dynamic_obstacle['y']), asv_heading)

            # Update observation grid
            grid = self.generate_grid(self.radius, self.grid_size, self.start)
            for g in grid:
                if g in self.grid_dict:
                    state = self.grid_dict[g]
                    color = {FREE_STATE: WHITE, PATH_STATE: GREEN, COLLISION_STATE: RED, GOAL_STATE: YELLOW}[state]
                    rect = plt.Rectangle((g[0] - self.grid_size / 2, g[1] - self.grid_size / 2), self.grid_size, self.grid_size, color=color)
                    ax2.add_patch(rect)

            return self.agent_1, self.agent_2, self.dynamic_obs_plot

        # Create animation
        ani = FuncAnimation(fig, update, frames=range(int(self.step)), blit=True, repeat=False)

        # # Save the animation
        # writer = FFMpegWriter(fps=10)
        # ani.save('asv_simulation.mp4', writer=writer)

        plt.show()

if __name__ == "__main__":
    vis = asv_visualisation()
    vis.main()
