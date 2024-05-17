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
    
    def is_on_path(self, position):
        path = []
        for x in range(START[0],self.height-9):
            path.append(x, START[1])
        for y in range(START[1],self.width-9):
            path.append(self.height-10, y)
        
        tolerance = 0.1

        for point in path:
            if abs(position[0] - point[0]) < tolerance and abs(position[1] - point[1]) < tolerance:
                return True
        return False
    
    def is_goal(self, position):
        if position[0] == GOAL[0] and position[1] == GOAL[1]:
            return True
        return False

    def calculate_reward(self, new_position):
        if new_position in self.boundary:
            return -50
        if self.is_on_path(self.position):
            return -1
        if not self.is_on_path(self.position):
            return -5
        if self.is_goal(self.position):
            return 0
    
    def step(self, action):
        new_pos = np.array(self.position)
        # 0: left 1m/s | 1: straight 1m/s | 2: right 1m/s
        # 3: left 2m/s | 4: straight 2m/s | 5: right 2m/s
        if action == 0:     # heading -5, speed = 1
            self.speed = 1
            self.heading -= 5
        elif action == 1:   # keep current heading, speed = 1
            self.speed = 1
            self.heading = self.heading
        elif action == 2:   # heading +5, speed = 1
            self.speed = 1
            self.heading += 5
        elif action == 3:   # heading -5, speed = 2
            self.speed = 2
            self.heading -= 5
        elif action == 4:   # keep current heading, speed = 2
            self.speed = 2
            self.heading = self.heading
        elif action == 5:   # heading +5, speed = 2
            self.speed = 2
            self.heading += 5

        # Update ASV position
        self.position = (self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                         self.position[1] + self.speed*np.sin(np.radians(self.heading)))

        # Check boundary
        in_boundary = self.position[0] > 0 and self.position[0] < self.width and \
                      self.position[1] > 0 and self.position[1] < self.height
        
        # Calculate reward
        reward = self.calculate_reward(self.position)

        # Set terminal condition: when the ASV hits the boundary or reach goal
        done = not in_boundary or self.is_goal(self.position)

        # Update state observation
        observation = np.array([*self.position, self.heading, self.speed])

        return observation, reward, done, {}
    
    def render():
        
        return
