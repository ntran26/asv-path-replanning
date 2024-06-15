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
SPEED = 10
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define map dimensions
WIDTH = 50
HEIGHT = 50
START = (0, 0)
GOAL = (0, 200)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 200/SPEED

class asv_visualization:
    # Initialize environment
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.heading = INITIAL_HEADING
        self.speed = SPEED
        self.turn_rate = TURN_RATE
        self.start_pos = START
        self.step = STEP
        self.goal = GOAL

    # Generate grid function
    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i**2 + j**2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid

    # Main function to create and draw ASV trajectory
    def draw_path(self):
        # Initialize/Reset the variables
        self.step_count = 0
        self.speed = self.speed
        self.current_heading = self.heading
        self.left_path = [self.start_pos]
        self.left_heading = [self.heading]
        self.position = self.start_pos

        # Go straight
        while self.step_count < self.step:
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                             self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
            # Append new position and heading angle to list
            self.left_path.append(self.position)
            self.left_heading.append(self.current_heading)
            # Update new heading angle and step count
            self.current_heading = self.current_heading
            self.step_count += 1

        # Define and obstacles
        static_obstacles = [(-30, -40), (70, -60), (70, 70), (0, 150)]

        # Define boundary
        boundary = []
        for x in range(-100, 100 + 1):
            boundary.append((x, -50))  # lower boundary
            boundary.append((x, 250))  # upper boundary 
        for y in range(-50, 250 + 1):
            boundary.append((-100, y))  # left boundary
            boundary.append((100, y))   # right boundary 

        # Initialize figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        self.agent_1, = ax1.plot([], [], marker='^', color=BLUE)
        self.agent_2, = ax2.plot([], [], marker='^', color=BLUE)
        observation_horizon1 = plt.Circle(START, RADIUS, color='r', fill=False)
        observation_horizon2 = plt.Circle(START, RADIUS, color='r', fill=False)
        ax1.add_patch(observation_horizon1)
        ax2.add_patch(observation_horizon2)

        # Plot goal point
        ax1.plot(0, 200, marker='o', color=YELLOW)

        # Plot the boundary
        for (x, y) in boundary:
            boundary_line = plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black')
            ax1.add_patch(boundary_line)

        # Generate the path
        self.path = []
        for y in range(START[1], GOAL[1]+1):
            self.path.append((START[0], y))

        # Plot the path
        path_array = np.array(self.path)
        ax1.plot(path_array[:,0], path_array[:,1], '-', color=GREEN)
        
        # Plot obstacles
        for (x, y) in static_obstacles:
            ax1.plot(x, y, marker='o', color=RED)

        # Empty list to store the collision grid coordinates
        squares_ax2 = []

        # Initialize animation variables
        def init():
            self.agent_1.set_data([], [])           # agent in the first plot
            self.agent_2.set_data([], [])           # agent in the second plot
            observation_horizon1.center = START
            observation_horizon2.center = START
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, START)
            for (cx, cy) in grid:
                rect = plt.Rectangle((cx - SQUARE_SIZE/2, cy - SQUARE_SIZE/2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor='none')
                ax1.add_patch(rect)
                squares_ax2.append(rect)
            return self.agent_1, self.agent_2, observation_horizon1, observation_horizon2, *squares_ax2

        # Reset locations of the grid squares
        def reset():
            for rect in squares_ax2:
                rect.remove()
            squares_ax2.clear()

        # Main animation loop to update the frame
        def update(frame):
            agent_pos = self.left_path[frame]
            heading = self.left_heading[frame]

            # Update the line segment as part of the plot
            self.agent_1.set_data(agent_pos[0], agent_pos[1])
            self.agent_1.set_marker((3, 0, heading - INITIAL_HEADING))
            
            self.agent_2.set_data(agent_pos[0], agent_pos[1])
            self.agent_2.set_marker((3, 0, heading - INITIAL_HEADING))

            # Check if the static obstacle is within the radius

            observation_horizon1.center = (agent_pos[0], agent_pos[1])
            observation_horizon2.center = (agent_pos[0], agent_pos[1])

            reset()  # remove previous grid squares

            # Draw new grid squares
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, (agent_pos[0], agent_pos[1]))
            for (cx, cy) in grid:
                # Check for obstacles, path and goal in the second plot
                is_collision = any(np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2) < (SQUARE_SIZE / 2 + OBSTACLE_RADIUS)
                                   for ox, oy in static_obstacles + boundary)
                is_path = any(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < (SQUARE_SIZE / 2 + OBSTACLE_RADIUS)
                              for px, py in self.path)
                is_goal = np.sqrt((cx - self.goal[0]) ** 2 + (cy - self.goal[1]) ** 2) < (SQUARE_SIZE / 2 + OBSTACLE_RADIUS)
                # Change the color of the grid if there is an obstacle or path                
                if is_collision:
                    color = RED
                elif is_goal:
                    color = YELLOW
                elif is_path:
                    color = GREEN
                else:
                    color = 'none'
                rect = plt.Rectangle((cx - SQUARE_SIZE/2, cy - SQUARE_SIZE/2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor=color)
                # Update the collision grid on the second plot
                ax2.add_patch(rect)
                squares_ax2.append(rect)

            return self.agent_1, self.agent_2, observation_horizon1, observation_horizon2, *squares_ax2

        ani = FuncAnimation(fig, update, frames=len(self.left_path), init_func=init, blit=True, interval=200, repeat=False)
        ax1.set_xlim(-RADIUS - 50, RADIUS + 50)
        ax1.set_ylim(-RADIUS - 50, RADIUS + 200)
        ax2.set_xlim(-RADIUS - 50, RADIUS + 50)
        ax2.set_ylim(-RADIUS - 50, RADIUS + 200)

        # # Write to mp4 file
        # FFwriter = FFMpegWriter(fps=5)
        # ani.save("Environment construction/Paper implementation/animation.mp4", writer=FFwriter)

        # Show plot
        ax1.grid(True)
        plt.show()

# Create visualization
visualization = asv_visualization(WIDTH, HEIGHT)
visualization.draw_path()
