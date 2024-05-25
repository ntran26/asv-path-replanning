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

RADIUS = 120
SQUARE_SIZE = 12
SPEED = 1

OBSTACLE_RADIUS = SQUARE_SIZE
# OBSTACLE_RADIUS = int(SQUARE_SIZE/2)

def generate_grid(radius, square_size, center):
    half_size = square_size / 2
    x = np.arange(-radius, radius, square_size)
    y = np.arange(-radius, radius, square_size)
    grid = []
    for i in x:
        for j in y:
            if np.sqrt(i**2 + j**2) <= radius:
                grid.append((center[0] + i + half_size, center[1] + j + half_size))
    return grid

def plot_grid(radius, start_pos, goal_pos, static_obstacles, moving_obstacles):
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_aspect('equal')  # set equal aspect ratio
    ax2.set_aspect('equal')

    asv1, = ax1.plot([], [], '^', color=BLUE)
    asv2, = ax2.plot([], [], '^', color=BLUE)
    observation_horizon1 = plt.Circle(start_pos, radius, color='r', fill=False)
    observation_horizon2 = plt.Circle(start_pos, radius, color='r', fill=False)
    ax1.add_patch(observation_horizon1)
    ax2.add_patch(observation_horizon2)

    # Draw goal
    ax1.plot(goal_pos[0], goal_pos[1], 'o', color=GREEN)
    ax2.plot(goal_pos[0], goal_pos[1], 'o', color=GREEN)
    
    # Draw static obstacles
    for (x, y) in static_obstacles:
        # ax1.plot(x, y, 'yo')
        ax2.plot(x, y, 'yo')
    
    # Draw moving obstacles
    for (x, y) in moving_obstacles:
        # ax1.plot(x, y, 'ro')
        ax2.plot(x, y, 'ro')
    
    # Create path from start to goal
    path = []
    for x in range(start_pos[0], goal_pos[0]+1, SPEED):
        for y in range(start_pos[1], goal_pos[1]+1, SPEED):
            path.append((x,y))
    squares = []

    # Initialize variables for animation
    def init():
        asv1.set_data([], [])
        asv2.set_data([], [])
        observation_horizon1.center = start_pos
        observation_horizon2.center = start_pos
        grid = generate_grid(radius, SQUARE_SIZE, start_pos)
        for (cx, cy) in grid:
            rect = plt.Rectangle((cx - SQUARE_SIZE / 2, cy - SQUARE_SIZE / 2), SQUARE_SIZE, SQUARE_SIZE,
                                 edgecolor='gray', facecolor='none')
            ax1.add_patch(rect)
            squares.append(rect)
        return asv1, asv2, observation_horizon1, observation_horizon2, *squares
    
    # Reset locations of the grid squares
    def reset():
        for rect in squares:
            rect.remove()
        squares.clear()

    # Main animation loop to update the frame
    def update(frame):
        pos = path[frame]
        # x = agent_pos[0] + (goal_pos[0] - agent_pos[0]) * t
        # y = agent_pos[1] + (goal_pos[1] - agent_pos[1]) * t
        asv1.set_data(pos[0], pos[1])
        asv2.set_data(pos[0], pos[1])
        observation_horizon1.center = (pos[0], pos[1])
        observation_horizon2.center = (pos[0], pos[1])
        reset()     # remove previous grid squares

        # Draw new grid squares
        grid = generate_grid(radius, SQUARE_SIZE, (pos[0], pos[1]))
        for (cx, cy) in grid:
            # Check for collision with obstacles
            is_collision = any(np.sqrt((cx - ox)**2 + (cy - oy)**2) < (SQUARE_SIZE/2 + OBSTACLE_RADIUS)
                                for ox, oy in static_obstacles + moving_obstacles)
            color = 'red' if is_collision else 'none'
            rect = plt.Rectangle((cx - SQUARE_SIZE / 2, cy - SQUARE_SIZE / 2), SQUARE_SIZE, SQUARE_SIZE,
                                edgecolor='gray', facecolor=color)
            ax1.add_patch(rect)
            squares.append(rect)

        return asv1, asv2, observation_horizon1, observation_horizon2, *squares

    ani = FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, interval=100, repeat=False)
    ax1.set_xlim(-radius - 100, radius + 100)
    ax1.set_ylim(-radius - 100, radius + 200)
    ax2.set_xlim(-radius - 100, radius + 100)
    ax2.set_ylim(-radius - 100, radius + 200)

    # # Write to mp4 file
    # FFwriter = FFMpegWriter(fps=10)
    # ani.save("Environment construction/Paper implementation/animation.mp4", writer=FFwriter)

    # Show plot
    plt.show()

# Agent position (center of the grid)
start_pos = (0, 0)

# Example positions 
goal_pos = (0, 150)
static_obstacles = [(-30, -40), (70, -60)]
moving_obstacles = [(10, 20), (40, 80), (50,150)]

# Plot the grid and objects with animation
plot_grid(RADIUS, start_pos, goal_pos, static_obstacles, moving_obstacles)

