import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class MazeGameEnv(gym.Env):
    def __init__(self, maze):
        super(MazeGameEnv, self).__innit__()
        self.maze = np.array(maze)      # represent as 2D numpy array
        self.start_pos = np.where(self.maze == 'S')     # set start position
        self.goal_pos = np.where(self.maze == 'G')      # set goal position
        self.current_pos = self.start_pos               # initialize current position as start position
        self.num_rows, self.num_cols = self.maze.shape  # set number of rows and columns

        # Define action space: 4 discrete actions 0-up, 1-down, 2-left, 3-right
        self.action_space = spaces.Discrete(4)

        # Define observation space: grid of size row x col
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Initialize pygame
        pygame.init()
        self.cell_size = 125

        # Set display style
        self.screen = pygame.display.set_mode((self.num_cols*self.cell_size), (self.num_rows*self.cell_size))
    
    def reset(self):
        self.current_pos = self.start_pos   # when reset, initialize current position
        return self.current_pos
    
    def step(self, action):
        # Move the agent based on the selected action
        new_pos = np.array(self.current_pos)    # update current position
        if action == 0:     # move up
            new_pos[0] -= 1
        if action == 1:     # move down
            new_pos[0] += 1
        if action == 2:     # move left
            new_pos[1] -= 1
        if action == 3:     # move right
            new_pos[1] += 1
        
        # Check if the new position is valid
        if self.is_valid_pos(new_pos):
            self.current_pos = new_pos
        
        # Reward function: Goal = 1, others = 0
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        
        info = {}
        return self.current_pos, reward, done, info
    
    def is_valid_pos(self, pos):
        row, col = pos

        # If the agent gets out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False
        
        # If the agent hits an obstacle
        if self.maze[row, col] == '#':
            return False
        
        return True
    
    def render(self):
        # Clear the screen
        self.screen.fill((255,255,255))

        # Draw elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                try:
                    print(np.array(self.current_pos) == np.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial state')
            
