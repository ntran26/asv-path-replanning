import gymnasium as gym
from gymnasium import spaces
import numpy as np

START = (5,5)
GOAL = (90,90)

class PathFollowEnv(gym.Env):
    def __init__(self):
        self.width = 100
        self.height = 100
        self.boundary = [(0,0), (0,self.height), (self.width,self.height), (self.width,0)]
        self.position = START
        self.heading = 0
        self.speed = 1

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
    
    def reset(self):
        self.position = GOAL
        self.heading = 0
        self.speed = 1

        return np.array([*self.position, self.heading, self.speed])
    
    def step(self, action):
        new_pos = np.array(self.position)
        # 0: left 1m/s | 1: straight 1m/s | 2: right 1m/s
        # 3: left 2m/s | 4: straight 2m/s | 5: right 2m/s
        
