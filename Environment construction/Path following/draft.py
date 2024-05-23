import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Define map dimensions
WIDTH = 900
HEIGHT = 500
START = (50, 50)
STEP = 100

class ASVEnv(gymnasium.Env):
    metadata = {'render_modes': ['human', "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode = None):
        super(ASVEnv, self).__init__()
        self.width = WIDTH
        self.height = HEIGHT
        self.start = np.array(START, dtype=float)
        self.position = self.start.copy()
        self.heading = 0
        self.speed = 2
        self.path = self._generate_path()
        self.action_space = spaces.Discrete(3)  # 0: left, 1: straight, 2: right
        self.observation_space = spaces.Box(low=0, high=max(WIDTH, HEIGHT), shape=(2,), dtype=np.float64)
        self.screen = None
        self.clock = None
        self.done = False
        self.step_count = 0
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start.copy()
        self.heading = 90
        self.done = False
        self.step_count = 0
        return self.position, {}

    def step(self, action):
        if action == 0:  # left
            self.heading -= 5
        elif action == 1: # straight
            self.heading = self.heading
        elif action == 2:  # right
            self.heading += 5

        # Update ASV position
        self.position[0] += self.speed * np.cos(np.radians(self.heading))
        self.position[1] += self.speed * np.sin(np.radians(self.heading))

        # Check if out of bounds
        if not (0 <= self.position[0] <= self.width and 0 <= self.position[1] <= self.height):
            self.done = True

        # Check if maximum steps are reached
        self.step_count += 1
        if self.step_count >= 30000:
            self.done = True

        reward = self._compute_reward()
        return self.position, reward, self.done, False, {}

    def render(self, mode='human'):
        if self.render_mode is None or mode != self.render_mode:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        
        self.screen.fill(BLACK)

        # Draw the path
        for point in self.path:
            pygame.draw.circle(self.screen, GREEN, (int(point[0]), int(point[1])), 1)

        # Draw the ASV
        pygame.draw.circle(self.screen, BLUE, (int(self.position[0]), int(self.position[1])), 5)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _generate_path(self):
        pts = []
        i = int(self.width / 100)
        for step in range(0, i):
            if step % 2 == 0:  # go top down
                p = (START[0] + step * 100, START[1])
                pts.append(p)
                p = (START[0] + step * 100, self.height - 50)
                pts.append(p)
            else:  # go bottom up
                p = (START[0] + step * 100, self.height - 50)
                pts.append(p)
                p = (START[0] + step * 100, START[1])
                pts.append(p)

        path = np.empty((0, 2), int)
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            if p1[0] == p2[0]:  # vertical line
                y_range = range(p1[1], p2[1]) if p1[1] < p2[1] else range(p1[1], p2[1], -1)
                for y in y_range:
                    path = np.append(path, [[p1[0], y]], axis=0)
            else:  # horizontal line
                x_range = range(p1[0], p2[0]) if p1[0] < p2[0] else range(p1[0], p2[0], -1)
                for x in x_range:
                    path = np.append(path, [[x, p1[1]]], axis=0)

        return path

    def _compute_reward(self):
        # Calculate the distance to the nearest point in the path
        distances = np.linalg.norm(self.path - self.position, axis=1)
        min_distance = np.min(distances)
        
        # Check if the ASV is on the green path
        on_green_path = min_distance < 5  # Adjust the threshold as needed
        
        if on_green_path:
            reward = -1  # Small negative reward for staying on path
        else:
            reward = -50  # Larger negative reward for being off path

        # Check if the ASV has reached the end of the path (goal)
        if self.position[0] >= self.path[-1, 0]:
            reward = 0  # Zero reward upon reaching the end
        
        return reward

# Register the environment
gymnasium.envs.registration.register(id='ASVEnv-v0', entry_point=ASVEnv, kwargs={'render_mode': 'human'})

# # Create the environment
# env = gymnasium.make('ASVEnv-v0', render_mode='human')

# # Check if the environment follows the gym interface
# check_env(env, warn=True)

# # Train with PPO
# env = DummyVecEnv([lambda: env])  # Wrap the environment
# # Create a callback for saving models
# checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/', name_prefix='ppo_asv_model')

# # Create the PPO model
# model = PPO('MlpPolicy', env, verbose=1)

# # Train the model
# model.learn(total_timesteps=1000000, callback=checkpoint_callback)

# # Save the final model
# model.save("ppo_asv_model_final")

# # Load the model for evaluation
# model = PPO.load("ppo_asv_model_final")

# # Evaluate the trained model
# obs = env.reset()
# for _ in range(30000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


