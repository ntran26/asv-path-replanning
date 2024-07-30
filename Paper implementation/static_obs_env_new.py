import numpy as np
import gymnasium as gym
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

class ASVEnv:
    def __init__(self):
        super(ASVEnv, self).__init__()
    
    def reset(self):
        return
    
    def is_on_path(self, position):
        return
    
    def is_valid_pos(self, position):
        return
    
    def is_goal(self, position):
        return
    
    def is_collision(self, position):
        return
    
    def calculate_reward(self, position):
        return
    
    def step(self, action):
        return
    
    def render(self):
        return


