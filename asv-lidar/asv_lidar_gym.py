import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
import pygame
import pygame.freetype
from ship_model import ShipModel
from asv_lidar import Lidar, LIDAR_RANGE, LIDAR_BEAMS
from images import BOAT_ICON

UPDATE_RATE = 0.5
RENDER_FPS = 10

# Actions
PORT = 0
CENTER = 1
STBD = 2
rudder_action = {
    PORT: -25,    # rudder left command (in percentage)
    CENTER: 0,    # centered
    STBD: 25      # rudder right command
}

class ASVLidarEnv(gym.Env):
    """Gymnasium environment for an Autonomous Surface Vessel with LIDAR."""
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = 'human'):
        super().__init__()
        pygame.init()
        self.render_mode = render_mode
        self.screen_size = (400, 600)
        self.icon = None
        self.fps_clock = pygame.time.Clock()

        self.display = None
        self.surface = None
        self.status = None
        if self.render_mode in self.metadata["render_modes"]:
            self.display = pygame.display.set_mode(self.screen_size)
            self.surface = pygame.Surface(self.screen_size)
            self.status = pygame.freetype.SysFont(pygame.font.get_default_font(), size=10)
        
        # State variables (positions, heading, etc.)
        self.elapsed_time = 0.
        # Define a vertical “path” by a fixed x coordinate.
        self.tgt_x = 150
        self.asv_x = 200
        self.asv_y = 550
        self.asv_h = 0      # heading in degrees
        self.asv_w = 0      # angular velocity in degrees/s
        self.tgt = 0        # horizontal offset from desired path

        # Ship dynamics model; set an initial forward speed.
        self.model = ShipModel()
        self.model._v = 4.5

        # Define observation space (the LIDAR, the ship's position, heading, heading rate, and target offset).
        self.observation_space = Dict({
            "lidar": Box(low=0, high=LIDAR_RANGE, shape=(LIDAR_BEAMS,), dtype=np.int16),
            "pos": Box(low=np.array([0, 0]), high=np.array(self.screen_size), shape=(2,), dtype=np.int16),
            "hdg": Box(low=0, high=360, shape=(1,), dtype=np.int16),
            "dhdg": Box(low=-180, high=180, shape=(1,), dtype=np.int16),
            "tgt": Box(low=-200, high=200, shape=(1,), dtype=np.int16)
        })
        self.action_space = Discrete(3)
        
        self.lidar = Lidar()
        self.boat_icon = BOAT_ICON

    def _get_obs(self):
        return {
            "lidar": self.lidar.ranges.astype(np.int16),
            "pos": np.array([self.asv_x, self.asv_y], dtype=np.int16),
            "hdg": np.array([int(self.asv_h) % 360], dtype=np.int16),
            "dhdg": np.array([int(self.asv_w)], dtype=np.int16),
            "tgt": np.array([int(self.tgt)], dtype=np.int16)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed_time = 0.
        self.tgt_x = 150
        self.asv_x = 200
        self.asv_y = 550
        self.asv_h = 0
        self.asv_w = 0
        self.tgt = 0
        self.model = ShipModel()
        self.model._v = 4.5
        self.lidar.reset()
        if self.render_mode in self.metadata["render_modes"]:
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        # Use a fixed engine rpm (e.g. 100) and adjust rudder based on action.
        dx, dy, h, w = self.model.update(100, rudder_action[action], UPDATE_RATE)
        self.asv_x += dx
        # In our coordinate system, increasing y is downward. We subtract dy.
        self.asv_y -= dy
        self.asv_h = h
        self.asv_w = w
        # The target is defined as the horizontal (x-axis) difference between the ASV and the path.
        self.tgt = self.tgt_x - self.asv_x

        # Update the LIDAR scan (currently no obstacles)
        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h)

        # Check for termination:
        terminated = False
        # If the ship reaches the top of the screen (or goes off screen vertically) terminate.
        if self.asv_y <= 0:
            terminated = True
        # Also terminate if the ship strays too far horizontally from the path.
        if abs(self.tgt) > 100:
            terminated = True

        # Define reward:
        #   - Always penalize a small amount per step.
        #   - If near the desired path (target offset within ±10 pixels), give no penalty.
        #   - Otherwise, if too far off the path, give a heavy penalty.
        reward = -1.0
        if abs(self.tgt) <= 10:
            reward = 0.0
        elif abs(self.tgt) > 50:
            reward = -10.0
        # if the ship moves backwards (dy < 0), add a penalty.
        if dy < 0:
            reward -= 10.0

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

        return self._get_obs(), reward, terminated, {}, {}

    def render(self):
        if self.render_mode not in self.metadata["render_modes"]:
            return
        # Create display
        if self.display is None:
            self.display = pygame.display.set_mode(self.screen_size)
        self.surface.fill((0, 0, 0))
        
        # Draw the LIDAR beams
        self.lidar.render(self.surface)
        
        # Draw the desired path as a vertical green line.
        pygame.draw.line(self.surface, (0, 200, 0), (self.tgt_x, 0), (self.tgt_x, self.screen_size[1]), 5)
        # Draw a marker on the path at a fixed distance ahead (for example purposes).
        pygame.draw.circle(self.surface, (100, 0, 0), (self.tgt_x, max(self.asv_y - 50, 0)), 5)
        
        # Draw ownship
        if self.boat_icon is None:
            self.boat_icon = pygame.image.frombytes(BOAT_ICON['bytes'],BOAT_ICON['size'],BOAT_ICON['format'])
        
        # Draw status text (time, heading, etc.)
        if self.status is not None:
            status_str = (f"Time: {self.elapsed_time:05.1f}s  "
                          f"HDG: {self.asv_h:+06.1f}°  "
                          f"dHDG: {self.asv_w:+05.1f}°/s  "
                          f"Offset: {self.tgt:+04.1f}")
            text_surf, _ = self.status.render(status_str, (255, 255, 255), (0, 0, 0))
            self.surface.blit(text_surf, (10, self.screen_size[1] - 20))
        
        self.display.blit(self.surface, (0, 0))
        pygame.display.update()
        self.fps_clock.tick(RENDER_FPS)

    def close(self):
        pygame.quit()

# For manual testing via keyboard control.
if __name__ == '__main__':
    env = ASVLidarEnv(render_mode="human")
    obs, _ = env.reset()
    action = CENTER
    total_reward = 0
    running = True
    while running:
        # # Random actions
        # action = env.action_space.sample()
        # obs, reward, done, _, _ = env.step(action)
        # total_reward += reward

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         done = True

        # Manual control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYUP:
                action = CENTER
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action = STBD
                elif event.key == pygame.K_LEFT:
                    action = PORT

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode finished. Time: {env.elapsed_time:.1f}s, Total reward: {total_reward}")
            running = False
    env.close()
