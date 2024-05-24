import numpy as np
import matplotlib.pyplot as plt

RADIUS = 120
SQUARE_SIZE = 12

def generate_grid(radius, square_size):
    half_size = square_size / 2
    x = np.arange(-radius, radius, square_size)
    y = np.arange(-radius, radius, square_size)
    grid = []
    for i in x:
        for j in y:
            if np.sqrt(i**2 + j**2) <= radius:
                grid.append((i + half_size, j + half_size))
    return grid

def plot_grid(grid, radius, agent_pos, goal_pos, static_obstacles, moving_obstacles):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')      # set equal aspect ratio
    
    square_size = SQUARE_SIZE
    # Draw grid
    for (cx, cy) in grid:
        rect = plt.Rectangle((cx - square_size / 2, cy - square_size / 2), square_size, square_size,
                             edgecolor='gray', facecolor='none')
        ax.add_patch(rect)
    
    # Draw observation horizon
    circle = plt.Circle((0, 0), radius, color='r', fill=False)
    ax.add_patch(circle)
    
    # Draw agent
    ax.plot(agent_pos[0], agent_pos[1], 'bo')

    # Draw goal
    ax.plot(goal_pos[0], goal_pos[1], 'go')
    
    # Draw static obstacles
    for (x, y) in static_obstacles:
        ax.plot(x, y, 'yo')
    
    # Draw moving obstacles
    for (x, y) in moving_obstacles:
        ax.plot(x, y, 'ro')
    
    # ax.legend()
    plt.xlim(-radius-100, radius+100)
    plt.ylim(-radius-100, radius+100)
    # plt.grid(True)
    plt.show()

# Generate grid
grid = generate_grid(RADIUS, SQUARE_SIZE)

# Agent position (center of the grid)
agent_pos = (0, 0)

# Example positions 
goal_pos = (50, 50)
static_obstacles = [(-30, -40), (70, -60)]
moving_obstacles = [(10, 20), (40, 80)]

# Plot the grid and objects
plot_grid(grid, RADIUS, agent_pos, goal_pos, static_obstacles, moving_obstacles)
