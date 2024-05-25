import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Define map dimensions
WIDTH = 50
HEIGHT = 50
START = (0,0)
TURN_RATE = 5
INITIAL_HEADING = 90
STEP = 20

class asv_visualization:
    # Initialize environment
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.start_pos = START

    # Main function to create and draw asv trajectory
    def draw_path(self):
        # Turn left
        self.step = 0
        self.heading = INITIAL_HEADING
        self.speed = 1
        self.left_path = [START]
        self.left_heading = [INITIAL_HEADING]
        self.position = START

        while self.step < STEP:
            self.position = (self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                            self.position[1] + self.speed*np.sin(np.radians(self.heading)))
            self.left_path.append(self.position)
            self.left_heading.append(self.heading)
            self.step += 1
            self.heading += TURN_RATE

        # Turn right
        self.step = 0
        self.heading = INITIAL_HEADING
        self.speed = 1
        self.right_path = [START]
        self.right_heading = [INITIAL_HEADING]
        self.position = START

        while self.step < STEP:
            self.position = (self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                            self.position[1] + self.speed*np.sin(np.radians(self.heading)))
            self.right_path.append(self.position)
            self.right_heading.append(self.heading)
            self.step += 1
            self.heading -= TURN_RATE
        
        # Go straight
        self.step = 0
        self.heading = INITIAL_HEADING
        self.speed = 1
        self.straight_path = [START]
        self.straight_heading = [INITIAL_HEADING]
        self.position = START

        while self.step < STEP:
            self.position = (self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                            self.position[1] + self.speed*np.sin(np.radians(self.heading)))
            self.straight_path.append(self.position)
            self.straight_heading.append(self.heading)
            self.heading = self.heading
            self.step += 1

        fig, ax = plt.subplots()
        ax.set_xlim(START[0]-30, START[0]+30)
        ax.set_ylim(START[1]-5, START[1]+40)
        ax.plot(START[0], START[1], marker='^', markersize=5, color=BLACK)
        
        path_array = np.array(self.left_path)
        ax.plot(path_array[:,0], path_array[:,1], '-', color=RED)

        path_array = np.array(self.right_path)
        ax.plot(path_array[:,0], path_array[:,1], '-', color=BLUE)

        path_array = np.array(self.straight_path)
        ax.plot(path_array[:,0], path_array[:,1], '-', color=GREEN)

        self.agent_left, = ax.plot([], [], '^', color=BLACK)
        self.agent_right, = ax.plot([], [], '^', color=BLACK)
        self.agent_straight, = ax.plot([], [], '^', color=BLACK)
            
        def init():
            self.agent_left.set_data([], [])
            self.agent_right.set_data([], [])
            self.agent_straight.set_data([], [])
            return self.agent_left, self.agent_right, self.agent_straight
        
        def update(frame):
            pos = self.left_path[frame]
            heading = self.left_heading[frame]
            self.agent_left.set_data(pos[0], pos[1])
            self.agent_left.set_marker((3, 0, heading - INITIAL_HEADING))

            pos = self.right_path[frame]
            heading = self.right_heading[frame]
            self.agent_right.set_data(pos[0], pos[1])
            self.agent_right.set_marker((3, 0, heading - INITIAL_HEADING))

            pos = self.straight_path[frame]
            heading = self.straight_heading[frame]
            self.agent_straight.set_data(pos[0], pos[1])
            self.agent_straight.set_marker((3, 0, heading - INITIAL_HEADING))

            return self.agent_left, self.agent_right, self.agent_straight,

        ani = FuncAnimation(fig, update, frames=STEP+1, init_func=init, blit=True, interval=200, repeat=True)

        # plt.grid(True)
        plt.show()

visualization = asv_visualization(WIDTH, HEIGHT)
visualization.draw_path()

        
    
