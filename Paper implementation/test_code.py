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
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define map dimensions
WIDTH = 200
HEIGHT = 300
START = (0, 0)
GOAL = (0, 200)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 200/SPEED

# Map boundaries
X_LOW = -100
X_HIGH = 100
Y_LOW = -50
Y_HIGH = 250

# Define states of the grid cell
FREE_STATE = 0
PATH_STATE = 1
GOAL_STATE = 2
COLLISION_STATE = 3

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
        print(len(grid))
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
        static_obstacles = [(-30, -40), (70, -60), (70, 70), (0, 150), (0, 30), (0, 70)]

        # Define boundary
        boundary = []
        for x in range(X_LOW, X_HIGH + 1):
            boundary.append((x, Y_LOW))  # lower boundary
            boundary.append((x, Y_HIGH))  # upper boundary 
        for y in range(Y_LOW, Y_HIGH + 1):
            boundary.append((X_LOW, y))  # left boundary
            boundary.append((X_HIGH, y))   # right boundary

        # Initialize figure and axes
        fig, ax = plt.subplots(1, figsize=(6,8))
        ax.set_aspect('equal')

        # Plot agent and observation radius
        self.agent, = ax.plot([], [], marker='^', color=BLUE)
        observation_horizon = plt.Circle(START, RADIUS, color='r', fill=False)
        ax.add_patch(observation_horizon)

        # Plot goal point
        ax.plot(0,200, marker='o', color=YELLOW)

        # Plot boundary
        for (x,y) in boundary:
            boundary_line = plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='black')
            ax.add_patch(boundary_line)

        # Generate the path
        self.path = []
        for y in range(START[1], GOAL[1]+1):
            self.path.append((START[0], y))

        # Plot the path
        path_array = np.array(self.path)
        ax.plot(path_array[:,0], path_array[:,1], '-', color=GREEN)
        
        # Plot obstacles
        for (x, y) in static_obstacles:
            ax.plot(x, y, marker='o', color=RED)

        # Empty list to store the collision grid coordinates
        squares_ax = []

        # Initialize animation variables
        def init():
            self.agent.set_data([], [])      
            observation_horizon.center = START
            grid = self.generate_grid(RADIUS, SQUARE_SIZE, START)
            for (cx, cy) in grid:
                rect = plt.Rectangle((cx - SQUARE_SIZE/2, cy - SQUARE_SIZE/2), SQUARE_SIZE, SQUARE_SIZE,
                                     edgecolor='gray', facecolor='none')
                ax.add_patch(rect)
                squares_ax.append(rect)
            return self.agent, observation_horizon, *squares_ax

        # Reset locations of the grid squares
        def reset():
            for rect in squares_ax:
                rect.remove()
            squares_ax.clear()

        # Main animation loop to update the frame
        def update(frame):
            agent_pos = self.left_path[frame]
            heading = self.left_heading[frame]

            # Update the line segment as part of the plot
            self.agent.set_data(agent_pos[0], agent_pos[1])
            self.agent.set_marker((3, 0, heading - INITIAL_HEADING))

            # Check if the static obstacle is within the radius
            observation_horizon.center = (agent_pos[0], agent_pos[1])

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
                
                # Update the collision grid
                ax.add_patch(rect)
                squares_ax.append(rect)

                # Update the state of the ASV based on color of the cell
                if (agent_pos[0] - SQUARE_SIZE/2 <= cx <= agent_pos[0] + SQUARE_SIZE/2) and (agent_pos[1] - SQUARE_SIZE/2 <= cy <= agent_pos[1] + SQUARE_SIZE/2):
                    if color == RED:
                        STATE = COLLISION_STATE
                    elif color == YELLOW:
                        STATE = GOAL_STATE
                    elif color == GREEN:
                        STATE = PATH_STATE
                    elif color == WHITE:
                        STATE = FREE_STATE
                    
                    if STATE == COLLISION_STATE:
                        print("Collide")
                    elif STATE == GOAL_STATE:
                        print("Reached Goal")
                    elif STATE == PATH_STATE:
                        print("On Path")
                    elif STATE == FREE_STATE:
                        print("Free Space")

            return self.agent, observation_horizon, *squares_ax

        ani = FuncAnimation(fig, update, frames=len(self.left_path), init_func=init, blit=True, interval=200, repeat=False)
        ax.set_xlim(-RADIUS - 50, RADIUS + 50)
        ax.set_ylim(-RADIUS - 50, RADIUS + 200)

        # # Write to mp4 file
        # FFwriter = FFMpegWriter(fps=5)
        # ani.save("Paper implementation/static_obstacle.mp4", writer=FFwriter)

        # Show plot
        ax.grid(False)
        plt.show()

# Create visualization
visualization = asv_visualization(WIDTH, HEIGHT)
visualization.draw_path()

# num = 3
# obstacles = [np.array([100, 70]), np.array([100, 100])]         # Set 2 obstacles on the path
# for _ in range(num):
#     pos = np.random.randint(0, [WIDTH, HEIGHT])
#     obstacles.append(pos)
# print(obstacles)

