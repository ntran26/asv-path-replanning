import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define states
FREE_STATE = 0
PATH_STATE = 1
COLLISION_STATE = 2
GOAL_STATE = 3

# Constants
RADIUS = 100
SQUARE_SIZE = 10
MAP_WIDTH = 200
MAP_HEIGHT = 200
INITIAL_CENTER_POINT = (100, 100)
NUM_OBSTACLES = 20
NUM_PATH_POINTS = 20

# Create a function that sets the priority of each state in case they overlap: 
# obstacle/collision > goal point > path > free space
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

# Create a function that converts each point from the global map to a grid coordinate
def closest_multiple(n, mult):
    return int((n + mult / 2) // mult) * mult

# Create a function to generate a dictionary, storing the grid coordinates and state
def fill_grid(objects, dc):
    grid_dict = {}      # initialize the dictionary grid_dict 
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

# Create a function that generate grid coordinates (x,y) from global map
def generate_grid(radius, square_size, center):
    x = np.arange(-radius + square_size, radius, square_size)
    y = np.arange(-radius + square_size, radius, square_size)
    grid = []
    for i in x:
        for j in y:
            if np.sqrt(i ** 2 + j ** 2) <= radius:
                grid.append((center[0] + i, center[1] + j))
    return grid

# Create a function to generate random obstacles
def generate_random_obstacles(num_obstacles, map_width, map_height):
    obstacles = []
    for _ in range(num_obstacles):
        x = np.random.randint(0, map_width)
        y = np.random.randint(0, map_height)
        obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
    return obstacles

# Create a function that generate the path to follow
def generate_path(initial_point, num_points, vertical_distance):
    path = []
    for i in range(num_points):
        y = initial_point[1] + i * vertical_distance
        path.append({'x': initial_point[0], 'y': y, 'state': PATH_STATE})
    return path

# Main plot
def plot_environment(objects, grid_dict, center_point, radius, square_size, map_width, map_height):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    # Plot the whole map on the left
    ax1.set_xlim(0, map_width)
    ax1.set_ylim(0, map_height)
    ax1.set_title('Whole Map')
    
    # Plot center point
    center_marker, = ax1.plot(center_point[0], center_point[1], marker='^', color='blue')
    
    # Plot obstacle
    for obj in objects:
        if obj['state'] == COLLISION_STATE:
            ax1.plot(obj['x'], obj['y'], 'ro')
    
    # Plot path
    path_x = [obj['x'] for obj in objects if obj['state'] == PATH_STATE]
    path_y = [obj['y'] for obj in objects if obj['state'] == PATH_STATE]
    path_line, = ax1.plot(path_x, path_y, 'g')
    
    # Plot observation circle
    obs_circle = plt.Circle(center_point, radius, color='red', fill=False)
    ax1.add_patch(obs_circle)
    
    grid_circle = plt.Circle((0,0), radius, color='red', fill=False)
    ax2.add_patch(grid_circle)
    
    # Plot the grid points on the right
    ax2.set_xlim(-radius, radius)
    ax2.set_ylim(-radius, radius)
    ax2.set_title('Grid Points')
    
    grid = generate_grid(radius, square_size, center_point)
    grid_patches = []
    for (cx, cy) in grid:
        state = grid_dict.get((closest_multiple(cx, square_size), closest_multiple(cy, square_size)), FREE_STATE)
        color = 'white'
        if state == COLLISION_STATE:
            color = 'red'
        elif state == PATH_STATE:
            color = 'green'
        elif state == GOAL_STATE:
            color = 'yellow'
        rect = plt.Rectangle((cx - square_size / 2 - center_point[0], cy - square_size / 2 - center_point[1]), square_size, square_size,
                             edgecolor='gray', facecolor=color)
        grid_patches.append(rect)
        ax2.add_patch(rect)
    
    def update(frame):
        # Move center point along the path
        if frame < len(path_x):
            new_center = (path_x[frame], path_y[frame])
            center_marker.set_data(new_center)
            obs_circle.center = new_center

            # Update grid_dict with the new center point
            new_grid = generate_grid(radius, square_size, new_center)
            for rect in grid_patches:
                rect.remove()
            grid_patches.clear()
            for (cx, cy) in new_grid:
                state = grid_dict.get((closest_multiple(cx, square_size), closest_multiple(cy, square_size)), FREE_STATE)
                color = 'white'
                if state == COLLISION_STATE:
                    color = 'red'
                elif state == PATH_STATE:
                    color = 'green'
                elif state == GOAL_STATE:
                    color = 'yellow'
                rect = plt.Rectangle((cx - square_size / 2 - new_center[0], cy - square_size / 2 - new_center[1]), square_size, square_size,
                                     edgecolor='gray', facecolor=color)
                grid_patches.append(rect)
                ax2.add_patch(rect)

        return center_marker, obs_circle, *grid_patches

    ani = FuncAnimation(fig, update, frames=len(path_x), blit=True, interval=200, repeat=False)
    ax1.set_xlim(-50, 250)
    ax1.set_ylim(0, 400)
    
    plt.show()

# Define objects in the environment
obstacles = generate_random_obstacles(NUM_OBSTACLES, MAP_WIDTH, MAP_HEIGHT)
path = generate_path(INITIAL_CENTER_POINT, NUM_PATH_POINTS, 5)
objects_environment = obstacles + path

dc = SQUARE_SIZE
grid_dict = fill_grid(objects_environment, dc)

# Plot the environment with animation
plot_environment(objects_environment, grid_dict, INITIAL_CENTER_POINT, RADIUS, SQUARE_SIZE, MAP_WIDTH, MAP_HEIGHT)
