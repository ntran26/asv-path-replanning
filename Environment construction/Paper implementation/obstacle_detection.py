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
SPEED = 2
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define map dimensions
WIDTH = 50
HEIGHT = 50
START = (0, 0)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 60

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

    # Generate grid function
    def generate_grid(self, radius, square_size, center):
        half_size = square_size / 2
        x = np.arange(-radius, radius, square_size)
        y = np.arange(-radius, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i**2 + j**2) <= radius:
                    grid.append((center[0] + i + half_size, center[1] + j + half_size))
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

        # Turn left
        while self.step_count < self.step:
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                             self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
            # Append new position and heading angle to list
            self.left_path.append(self.position)
            self.left_heading.append(self.current_heading)
            # Update new heading angle and step count
            self.current_heading += self.turn_rate
            self.step_count += 1

        # Define and obstacles
        static_obstacles = [(-30, -40), (70, -60)]
        # moving_obstacles = [(10, 20), (50, 150)]
        moving_obstacle = [(40, 80)]

        # Create moving osbtacle trajectory
        self.step_count = 0
        self.speed = self.speed+5
        self.current_heading = self.heading
        self.obstacle_path = [(moving_obstacle[0][0], moving_obstacle[0][1])]
        self.obstacle_heading = [self.heading]
        self.obstacle_position = (moving_obstacle[0][0], moving_obstacle[0][1])

        while self.step_count < self.step:
            self.obstacle_position = (self.obstacle_position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                             self.obstacle_position[1] + self.speed * np.sin(np.radians(self.current_heading)))
            # Append new position and heading angle to list
            self.obstacle_path.append(self.obstacle_position)
            self.obstacle_heading.append(self.current_heading)
            # Update new heading angle and step count
            self.current_heading += self.turn_rate
            self.step_count += 1

        # Initialize figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        self.agent_1, = ax1.plot([], [], marker='^', color=BLUE)
        self.agent_2, = ax2.plot([], [], marker='^', color=BLUE)
        observation_horizon1 = plt.Circle(START, RADIUS, color='r', fill=False)
        observation_horizon2 = plt.Circle(START, RADIUS, color='r', fill=False)
        ax1.add_patch(observation_horizon1)
        ax2.add_patch(observation_horizon2)

        # Plot the path
        path_array = np.array(self.left_path)
        ax1.plot(path_array[:,0], path_array[:,1], '-', color=RED)

        # Plot obstacles
        for (x, y) in static_obstacles:
            ax1.plot(x, y, marker='o', color=RED)
        self.moving_obstacle, = ax1.plot([], [], marker='o', color=RED)

        # Empty list to store the collision grid coordinates
        squares = []

        # Initialize animation variables
        def init():
            self.agent_1.set_data([], [])           # agent in the first plot
            self.agent_2.set_data([], [])           # agent in the second plot
            self.moving_obstacle.set_data([], [])   # moving obstacle
            observation_horizon1.center = START
            observation_horizon2.center = START
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, START)
            for (cx, cy) in grid:
                rect = plt.Rectangle((cx - SQUARE_SIZE/2, cy - SQUARE_SIZE/2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor='none')
                ax1.add_patch(rect)
                squares.append(rect)
            return self.agent_1, self.agent_2, self.moving_obstacle, observation_horizon1, observation_horizon2, *squares

        # Reset locations of the grid squares
        def reset():
            for rect in squares:
                rect.remove()
            squares.clear()

        # Main animation loop to update the frame
        def update(frame):
            agent_pos = self.left_path[frame]
            heading = self.left_heading[frame]
            obstacle_pos = self.obstacle_path[frame]

            # Update the line segment as part of the plot
            self.agent_1.set_data(agent_pos[0], agent_pos[1])
            self.agent_1.set_marker((3, 0, heading - INITIAL_HEADING))
            
            self.agent_2.set_data(agent_pos[0], agent_pos[1])
            self.agent_2.set_marker((3, 0, heading - INITIAL_HEADING))

            self.moving_obstacle.set_data(obstacle_pos[0], obstacle_pos[1])

            observation_horizon1.center = (agent_pos[0], agent_pos[1])
            observation_horizon2.center = (agent_pos[0], agent_pos[1])

            reset()  # remove previous grid squares

            # Draw new grid squares
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, (agent_pos[0], agent_pos[1]))
            for (cx, cy) in grid:
                # Check for obstacles in the second plot
                is_collision = any(np.sqrt((cx - ox)**2 + (cy - oy)**2) < (SQUARE_SIZE/2 + OBSTACLE_RADIUS)
                                   for ox, oy in static_obstacles)
                if not is_collision:
                    is_collision = np.sqrt((cx - obstacle_pos[0])**2 + (cy - obstacle_pos[1])**2) \
                                    < (SQUARE_SIZE/2 + OBSTACLE_RADIUS)
                # Change the color of the grid if there is obstacle
                color = 'red' if is_collision else 'none'
                rect = plt.Rectangle((cx - SQUARE_SIZE/2, cy - SQUARE_SIZE/2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor=color)
                # Update the collision grid on the second plot
                ax2.add_patch(rect)
                squares.append(rect)

            return self.agent_1, self.agent_2, self.moving_obstacle, observation_horizon1, observation_horizon2, *squares

        ani = FuncAnimation(fig, update, frames=len(self.left_path), init_func=init, blit=True, interval=200, repeat=False)
        ax1.set_xlim(-RADIUS - 50, RADIUS + 50)
        ax1.set_ylim(-RADIUS - 50, RADIUS + 50)
        ax2.set_xlim(-RADIUS - 50, RADIUS + 50)
        ax2.set_ylim(-RADIUS - 50, RADIUS + 50)

        # # Write to mp4 file
        # FFwriter = FFMpegWriter(fps=5)
        # ani.save("Environment construction/Paper implementation/animation.mp4", writer=FFwriter)

        # Show plot
        ax1.grid(True)
        plt.show()

# Create visualization
visualization = asv_visualization(WIDTH, HEIGHT)
visualization.draw_path()
