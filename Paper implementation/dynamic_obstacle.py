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
OBSTACLE_RADIUS = SQUARE_SIZE / 3

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
    def __init__(self, case):
        self.case = case
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

        self.obstacles = self.generate_static_obstacles(5, self.width, self.height)
        self.dynamic_obstacles = self.generate_dynamic_obstacles()
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
            
            grid_dict[(m, n)] = self.get_priority_state(grid_dict[(m,n)], state)
        return grid_dict

    # Create a function that generate grid coordinates (x,y) from global map
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
    
    # Generate dynamic obstacles
    def generate_dynamic_obstacles(self):
        dynamic_obs = []
        if self.case == 1:
            # Crossing situation: Ship approaches from starboard
            dynamic_obs.append({'x': 120, 'y': 100, 'heading': 270, 'speed': 2, 'state': COLLISION_STATE})
        elif self.case == 2:
            # Head-on situation
            dynamic_obs.append({'x': 100, 'y': 200, 'heading': 270, 'speed': 2, 'state': COLLISION_STATE})
        elif self.case == 3:
            # Overtaking situation
            dynamic_obs.append({'x': 100, 'y': 50, 'heading': 90, 'speed': 1, 'state': COLLISION_STATE})
        elif self.case == 4:
            # Crossing situation where agent has priority
            dynamic_obs.append({'x': 80, 'y': 150, 'heading': 90, 'speed': 2, 'state': COLLISION_STATE})
        return dynamic_obs
    
    # Update dynamic obstacles
    def update_dynamic_obstacles(self):
        for obs in self.dynamic_obstacles:
            obs['x'] += obs['speed'] * np.cos(np.radians(obs['heading']))
            obs['y'] += obs['speed'] * np.sin(np.radians(obs['heading']))

    # Modify grid based on dynamic obstacles and COLREGs rules
    def update_grid_with_dynamic_obstacles(self):
        for obs in self.dynamic_obstacles:
            dx = obs['x'] - self.center_point[0]
            dy = obs['y'] - self.center_point[1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= self.radius:
                grid_x = self.closest_multiple(obs['x'], self.grid_size)
                grid_y = self.closest_multiple(obs['y'], self.grid_size)
                self.grid_dict[(grid_x, grid_y)] = COLLISION_STATE

    #                           -------- VISUALIZATION --------
    
    # Create an observation grid visualisation for ASV agent 
    def get_asv_observation_grid(self):
        center_x = self.center_point[0]
        center_y = self.center_point[1]
        grid = self.generate_grid(self.radius, self.grid_size, self.center_point)
        grid_states = [self.grid_dict.get((self.closest_multiple(p[0], self.grid_size), 
                                           self.closest_multiple(p[1], self.grid_size)), FREE_STATE) for p in grid]
        grid_relative = [(int((p[0] - center_x) / self.grid_size), int((p[1] - center_y) / self.grid_size)) for p in grid]
        return grid_relative, grid_states
    
    def display_map(self):
        self.grid_dict = self.fill_grid(self.objects_environment, self.grid_size)
        observation_grid, state = self.get_asv_observation_grid()
        x, y = zip(*observation_grid)
        colors = [BLACK if s == COLLISION_STATE else (GREEN if s == PATH_STATE else (YELLOW if s == GOAL_STATE else WHITE)) for s in state]

        fig, ax = plt.subplots()
        sc = ax.scatter(x, y, c=colors, s=500, marker='s', edgecolor='black')
        plt.xlim(-self.radius / self.grid_size, self.radius / self.grid_size)
        plt.ylim(-self.radius / self.grid_size, self.radius / self.grid_size)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.title('ASV Observation Grid')

        def update(frame):
            ax.cla()
            self.center_point = (self.start[0], self.start[1] + frame * self.speed)
            self.update_dynamic_obstacles()
            self.update_grid_with_dynamic_obstacles()
            observation_grid, state = self.get_asv_observation_grid()
            x, y = zip(*observation_grid)
            colors = [BLACK if s == COLLISION_STATE else (GREEN if s == PATH_STATE else (YELLOW if s == GOAL_STATE else WHITE)) for s in state]
            ax.scatter(x, y, c=colors, s=500, marker='s', edgecolor='black')
            plt.xlim(-self.radius / self.grid_size, self.radius / self.grid_size)
            plt.ylim(-self.radius / self.grid_size, self.radius / self.grid_size)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.title('ASV Observation Grid')

        ani = FuncAnimation(fig, update, frames=int(self.step), repeat=False, interval=50)
        plt.show()
        # # Export the animation
        # writer = FFMpegWriter(fps=10)
        # ani.save(f"asv_case_{self.case}.mp4", writer=writer)

#                           -------- TEST CASES --------

# Initialize environment with a specific case
case = 3
asv_env = asv_visualisation(case)
asv_env.display_map()
