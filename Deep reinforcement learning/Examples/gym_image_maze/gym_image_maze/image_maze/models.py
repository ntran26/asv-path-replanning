from enum import Enum
from gym_image_maze.image_maze.utilities import distance, map_tuple
from operator import *

class Action(Enum):
    Left    = (-1, 0)
    Right   = (1, 0)
    Up      = (0, -1)
    Down    = (0, 1)
    
class Reward(Enum):
    Closer  = 1
    Further = -1
    Same    = 0
    Collide = -10
    Goal    = 1000
        
class Robot(object):
    def __init__(self, position, size):
        self.position = position
        self.size = size
        
    def move(self, action):
        self.position = self.calculate_new_pos(action)
        
    def calculate_new_pos(self, action):
        d = action.value
        new_pos = map_tuple(add, self.position, d)
        return new_pos

class Wall(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def length(self):
        return distance(self.a, self.b)