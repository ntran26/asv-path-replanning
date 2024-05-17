import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import cv2

START = (5,5)
GOAL = (90,90)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('training_video.avi', fourcc, 20.0, (600, 600))

class PathFollowEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self,render_mode='human'):
        self.render_mode = render_mode
        self.width = 100
        self.height = 100
        # self.boundary = [(0,0), (0,self.height), (self.width,self.height), (self.width,0)]
        self.boundary = [(0,0), (0,self.height), (self.width,self.height), (self.width,0), (0,0)]
        self.position = START
        self.heading = 0
        self.speed = 1

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, self.width, shape=(4,), dtype=np.float32)

        # Define the path
        self.path = []
        for x in range(START[0], self.width-9):
            self.path.append((START[1], x))
        for y in range(START[1], self.height-9):
            self.path.append((y, self.width-10))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.position = GOAL
        self.heading = 0
        self.speed = 1

        observation = np.array([*self.position, self.heading, self.speed])
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32)
        # observation = np.clip(observation, 0.0, 100.0)

        return observation, {}
    
    def is_on_path(self, position):
        tolerance = 0.1
        for point in self.path:
            if abs(position[0] - point[0]) < tolerance and abs(position[1] - point[1]) < tolerance:
                return True
        return False
    
    def is_valid_pos(self, position):
        row, col = position
        if row > 0 or row < self.width or col > 0 or col < self.height:
            return False
        return True
    
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
        new_pos = (self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                   self.position[1] + self.speed*np.sin(np.radians(self.heading)))
        
        if self.is_valid_pos(new_pos):
            self.position = new_pos

        # Check boundary
        in_boundary = self.position[0] > 0 and self.position[0] < self.width and \
                      self.position[1] > 0 and self.position[1] < self.height
        
        # Calculate reward
        reward = self.calculate_reward(self.position)

        # Set terminal condition: when the ASV hits the boundary or reach goal
        done = not in_boundary or self.is_goal(self.position)

        # Update state observation
        observation = np.array([*self.position, self.heading, self.speed])
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32)
        # observation = np.clip(observation, 0.0, 100.0)
        
        return observation, reward, done, False, {}
    
    def render(self):
        fig, ax = plt.subplots()

        # Plot map boundaries
        boundary_pts = np.array(self.boundary)
        ax.plot(boundary_pts[:,0], boundary_pts[:,1], color='black')

        # Plot the path
        path_pts = np.array(self.path)
        ax.plot(path_pts[:, 0], path_pts[:, 1], color='blue')
        
        # Plot the ASV
        asv = patches.Circle((self.position[0], self.position[1]), radius=0.5, fc='blue', ec='black')
        # asv = patches.Polygon([[self.position[0], self.position[1]-0.25],
        #                        [self.position[0]-0.25, self.position[1]+0.25],
        #                        [self.position[0]+0.25, self.position[1]+0.25]], closed=True)
        ax.add_patch(asv)

        # Plot the heading angle line
        angle_rad = np.radians(self.heading)
        heading_line = ([self.position[0],self.position[0]+0.5*np.sin(angle_rad),],
                        [self.position[1],self.position[1]+0.5*np.sin(angle_rad),])
        ax.plot(heading_line[0],heading_line[1],color='red')

        plt.show()



env = PathFollowEnv(render_mode='human')

# Wrap the environment for vectorized training
env = DummyVecEnv([lambda: env])
total_episodes = 100

# Instantiate the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
for episode in range(total_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # Render the environment
        img = env.render(mode='human')
        if img is not None:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img_resized = cv2.resize(img_bgr, (600, 600))
            out.write(img_bgr)

out.release()
cv2.destroyAllWindows()
env.close()

# Save the model
model.save("path_following_model")

