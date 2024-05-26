import gym
import numpy as np
import os
from gym import spaces
from gym_image_maze.image_maze.maze_drawer import MazeDrawer
from gym_image_maze.image_maze.models import *
from gym_image_maze.image_maze.maze import Maze
from pyglet.window import Window

from gym_image_maze.image_maze.renderer import Renderer

SCREEN_SIZE = 300

class ImageMazeEnv(gym.Env):
    
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    
    ALL_ACTIONS = [Action.Left, Action.Right, Action.Up, Action.Down]
    
    def __init__(self, config_file, time_limit, visualize=True):
        self.config_file = config_file
        self.time_limit = time_limit
        self.timestep = 0
        self.done = False
        self.visualize = visualize
        
        self.maze = Maze.from_config(config_file)
        self.action_space = spaces.Discrete(len(self.ALL_ACTIONS))
        low = np.zeros((self.maze.size, self.maze.size), dtype=np.uint8)
        high = np.ones((self.maze.size, self.maze.size), dtype=np.uint8) * 255
        self.observation_space = spaces.Box(low, high)
        if visualize:
            self.renderer = Renderer()
        
    def step(self, action):
        action = self.ALL_ACTIONS[action]
        self.timestep += 1
        
        current_dist_to_goal = self.maze.dist_to_goal()
        self.maze.move_robot(action)
        new_dist_to_goal = self.maze.dist_to_goal()
        is_collide = current_dist_to_goal == new_dist_to_goal
        
        self.done = self.maze.is_robot_reach_goal() or (self.timestep == self.time_limit)
        
        reward = 0.
        if self.maze.is_robot_reach_goal():
            reward = Reward.Goal
        elif is_collide:
            reward = Reward.Collide
        elif new_dist_to_goal < current_dist_to_goal:
            reward = Reward.Closer
        else:
            reward = Reward.Further
        
        return MazeDrawer.draw_maze(self.maze, 500, 500), reward.value, self.done, {}
    
    def reset(self):
        self.maze = Maze.from_config(self.config_file)
        self.done = False
        self.timestep = 0 
        if self.visualize:
            self.renderer.render(MazeDrawer.draw_maze(self.maze, 500, 500))
        
        return MazeDrawer.draw_maze(self.maze, 500, 500)
    
    def render(self, mode='human'):
        self.renderer.render(MazeDrawer.draw_maze(self.maze, 500, 500))
    
    def close(self):
        if self.visualize:
            self.renderer.close()
    
def get_config_path():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(dir_path, 'configs')
    return config_path

class ImageMazeV0(ImageMazeEnv):
    def __init__(self):
        config_file = 'image_maze_v0.json'
        super().__init__(config_file=os.path.join(get_config_path(), config_file), time_limit=200)
        
class ImageMazeV1(ImageMazeEnv):
    def __init__(self):
        config_file = 'image_maze_v1.json'
        super().__init__(config_file=os.path.join(get_config_path(), config_file), time_limit=200)
        
class ImageMazeV2(ImageMazeEnv):
    def __init__(self):
        config_file = 'image_maze_v2.json'
        super().__init__(config_file=os.path.join(get_config_path(), config_file), time_limit=200)