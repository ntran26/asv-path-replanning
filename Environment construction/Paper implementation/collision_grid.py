import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RADIUS = 120
SQUARE_SIZE = 12

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

def plot_grid(radius, agent_pos, goal_pos, static_obstacles, moving_obstacles):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')  # set equal aspect ratio

    agent_dot, = ax.plot([], [], 'bo')
    observation_horizon = plt.Circle(agent_pos, radius, color='r', fill=False)
    ax.add_patch(observation_horizon)

    # Draw goal
    ax.plot(goal_pos[0], goal_pos[1], 'go')
    
    # Draw static obstacles
    for (x, y) in static_obstacles:
        ax.plot(x, y, 'yo')
    
    # Draw moving obstacles
    for (x, y) in moving_obstacles:
        ax.plot(x, y, 'ro')

    squares = []

    def init():
        agent_dot.set_data([], [])
        observation_horizon.center = agent_pos
        grid = generate_grid(radius, SQUARE_SIZE, agent_pos)
        for (cx, cy) in grid:
            rect = plt.Rectangle((cx - SQUARE_SIZE / 2, cy - SQUARE_SIZE / 2), SQUARE_SIZE, SQUARE_SIZE,
                                 edgecolor='gray', facecolor='none')
            ax.add_patch(rect)
            squares.append(rect)
        return agent_dot, observation_horizon, *squares

    def update(frame):
        t = frame/100  # Moving speed
        x = agent_pos[0] + (goal_pos[0] - agent_pos[0]) * t
        y = agent_pos[1] + (goal_pos[1] - agent_pos[1]) * t
        agent_dot.set_data(x, y)
        observation_horizon.center = (x, y)

        # Remove previous grid squares
        for rect in squares:
            rect.remove()
        squares.clear()

        # Draw new grid squares
        grid = generate_grid(radius, SQUARE_SIZE, (x, y))
        for (cx, cy) in grid:
            rect = plt.Rectangle((cx - SQUARE_SIZE / 2, cy - SQUARE_SIZE / 2), SQUARE_SIZE, SQUARE_SIZE,
                                 edgecolor='gray', facecolor='none')
            ax.add_patch(rect)
            squares.append(rect)

        return agent_dot, observation_horizon, *squares

    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=100, repeat=False)
    plt.xlim(-radius - 100, radius + 100)
    plt.ylim(-radius - 100, radius + 100)
    plt.show()

# Agent position (center of the grid)
agent_pos = (0, 0)

# Example positions 
goal_pos = (0, 70)
static_obstacles = [(-30, -40), (70, -60)]
moving_obstacles = [(10, 20), (40, 80)]

# Plot the grid and objects with animation
plot_grid(RADIUS, agent_pos, goal_pos, static_obstacles, moving_obstacles)

