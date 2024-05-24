import numpy as np
import matplotlib.pyplot as plt

# Define Object class
class Object:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight

# Naive algorithm for filling the grid
def naive_fill_grid(grid, objects, dc):
    fill_grid = []
    for i in range(len(grid)):
        fill_grid.append([])
        for j in range(len(objects)):
            x = grid[i][0]
            y = grid[i][1]
            m = objects[j].x
            n = objects[j].y
            if x - dc / 2 <= m <= x + dc / 2 and y - dc / 2 <= n <= y + dc / 2:
                fill_grid[i].append(objects[j].weight)
    return fill_grid

# Optimized algorithm for filling the grid
def closest_multiple(n, mult):
    return int((n + mult / 2) / mult) * mult

def optimized_fill_grid(grid_dict, objects, dc):
    for i in range(len(objects)):
        m = objects[i].x
        n = objects[i].y
        m = closest_multiple(abs(m), dc) * (1 if m >= 0 else -1)
        n = closest_multiple(abs(n), dc) * (1 if n >= 0 else -1)
        if (m, n) not in grid_dict:
            grid_dict[(m, n)] = []
        grid_dict[(m, n)].append(objects[i].weight)
    return grid_dict

# Generate the grid
def generate_grid(radius, square_size):
    half_size = square_size / 2
    x = np.arange(-radius, radius + square_size, square_size)
    y = np.arange(-radius, radius + square_size, square_size)
    grid = []
    for i in x:
        for j in y:
            if np.sqrt(i**2 + j**2) <= radius:
                grid.append((i + half_size, j + half_size))
    return grid

# Plot the grid and objects
def plot_grid(grid, radius, agent_pos, goal_pos, static_obstacles, moving_obstacles, square_size):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    for ax in axs:
        ax.set_aspect('equal')  # set equal aspect ratio

        # Draw grid
        for (cx, cy) in grid:
            rect = plt.Rectangle((cx - square_size / 2, cy - square_size / 2), square_size, square_size,
                                 edgecolor='gray', facecolor='none')
            ax.add_patch(rect)

        # Draw observation horizon
        circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--')
        ax.add_patch(circle)

        # Draw agent
        ax.plot(agent_pos[0], agent_pos[1], 'bo', label='Agent')

        # Draw goal
        ax.plot(goal_pos[0], goal_pos[1], 'go', label='Goal')

        # Draw static obstacles
        for (x, y) in static_obstacles:
            ax.plot(x, y, 'yo', label='Static Obstacle')

        # Draw moving obstacles
        for (x, y) in moving_obstacles:
            ax.plot(x, y, 'ro', label='Moving Obstacle')

    axs[0].set_title('Plot 1')
    axs[1].set_title('Plot 2')

    for ax in axs:
        ax.legend()
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)

    plt.show()


# Parameters
radius = 120  # Observation horizon radius
square_size = 15  # Size of each grid square

# Generate grid
grid = generate_grid(radius, square_size)

# Agent position (center of the grid)
agent_pos = (0, 0)

# Example positions
goal_pos = (50, 50)
static_obstacles = [Object(-30, -40, 1), Object(70, -60, 1)]
moving_obstacles = [Object(10, 20, 1), Object(40, 80, 1)]

# Using naive algorithm
naive_result = naive_fill_grid(grid, static_obstacles + moving_obstacles, square_size)
print("Naive Algorithm Result:")
for row in naive_result:
    print(row)

# Using optimized algorithm
optimized_result = optimized_fill_grid({}, static_obstacles + moving_obstacles, square_size)
print("\nOptimized Algorithm Result:")
for key, value in optimized_result.items():
    print(f"Grid {key}: {value}")

# Plot the grid and objects
plot_grid(grid, radius, agent_pos, goal_pos, [(o.x, o.y) for o in static_obstacles], [(o.x, o.y) for o in moving_obstacles], square_size)
