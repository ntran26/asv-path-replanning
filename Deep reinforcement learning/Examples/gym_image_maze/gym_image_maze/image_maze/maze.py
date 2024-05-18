import json
from typing import List
from gym_image_maze.image_maze.models import Robot, Wall
from gym_image_maze.image_maze.utilities import distance, dist_from_point_to_line
from operator import *

class Maze(object):
    
    @classmethod
    def from_config(cls, config_file):
        maze_config = json.load(open(config_file))
        maze_size = maze_config['size']
        robot_pos = maze_config['robot']['x'], maze_config['robot']['y']
        robot_size = maze_config['robot']['size']
        goal_pos = maze_config['goal']['x'], maze_config['goal']['y']
        walls = [Wall((config['ax'], config['ay']), (config['bx'], config['by'])) for config in maze_config['walls']]
        maze = Maze(maze_size, robot_pos, robot_size, goal_pos, walls)
        return maze

    def __init__(self, size, robot_pos, robot_size, goal_pos, walls: List[Wall] = []):
        self.size = size
        self.initial_pos = robot_pos
        self.robot = Robot(robot_pos, robot_size)
        self.goal = goal_pos
        # Add borders
        height, width = (size, size)
        self.walls = [
            Wall((0, 0), (width - 1, 0)),
            Wall((0, 0), (0, height - 1)),
            Wall((0, height - 1), (height - 1, width - 1)),
            Wall((width - 1, height - 1), (width - 1,0))
        ]
        self.walls.extend(walls)
        
    def move_robot(self, action):
        new_pos = self.robot.calculate_new_pos(action)
        if not self.collide(new_pos): self.robot.move(action)
    
    def collide(self, pos):
        for wall in self.walls:
            if dist_from_point_to_line(pos, wall.a, wall.b) <= self.robot.size:
                return True
        return False
    
    def is_robot_reach_goal(self):
        return self.robot.position == self.goal
    
    def dist_to_goal(self):
        return distance(self.robot.position, self.goal)
    
    def dist_from_start(self):
        return distance(self.initial_pos, self.robot.position)