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
MAP_WIDTH = 400
MAP_HEIGHT = 600

# Actions
PORT = 0
CENTER = 1
STBD = 2
rudder_action = {
    PORT: -25,
    CENTER: 0,
    STBD: 25
}

class ASVLidarEnv(gym.Env):
    """ Autonomous Surface Vessel w/ LIDAR Gymnasium environment

        Args:
            render_mode (str): If/How to render the environment
                "human" will render a pygame windows, episodes are run in real-time
                None will not render, episodes run as fast as possible
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(
            self, 
            render_mode:str = 'human'
            ) -> None:
        
        self.map_width = MAP_WIDTH
        self.map_height = MAP_HEIGHT

        self.path_range = 20
        self.collision = 20

        pygame.init()
        self.render_mode = render_mode
        self.screen_size = (self.map_width,self.map_height)

        self.icon = None
        self.fps_clock = pygame.time.Clock()

        self.display = None
        self.surface = None
        self.status = None
        if render_mode in self.metadata['render_modes']:
            self.surface = pygame.Surface(self.screen_size)
            self.status = pygame.freetype.SysFont(pygame.font.get_default_font(),size=10)
        # State
        self.elapsed_time = 0.
        self.tgt_x = 0
        self.tgt_y = 0
        self.tgt = 0
        self.asv_y = 0
        self.asv_x = 0
        self.asv_h = 0
        self.asv_w = 0

        self.model = ShipModel()
        self.model._v = 4.5

        """
        Observation space:
            lidar: an array of lidar range: [21 values]
            pos: (x,y) coordinate of asv
            hdg: heading/yaw of the asv
            dhdg: rate of change of heading
            tgt: horizontal offset of the asv from the path
        """
        self.observation_space = Dict(
            {
                "lidar": Box(low=0,high=LIDAR_RANGE,shape=(LIDAR_BEAMS,),dtype=np.int16),
                "pos"  : Box(low=np.array([0,0]),high=np.array(self.screen_size),shape=(2,),dtype=np.int16),
                "hdg"  : Box(low=0,high=360,shape=(1,),dtype=np.int16),
                "dhdg" : Box(low=0,high=36,shape=(1,),dtype=np.int16),
                "tgt"  : Box(low=-50,high=50,shape=(1,),dtype=np.int16)
            }
        )

        self.action_space = Discrete(3)
        
        # LIDAR
        self.lidar = Lidar()

        # Initialize obstacles
        self.obstacles = []


    def _get_obs(self):
        return {
            'lidar': self.lidar.ranges.astype(np.int16),
            'pos': np.array([self.asv_x, self.asv_y],dtype=np.int16),
            'hdg': np.array([self.asv_h],dtype=np.int16),
            'dhdg': np.array([self.asv_w],dtype=np.int16),
            'tgt': np.array([self.tgt],dtype=np.int16)
        }

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.tgt_x = 150
        self.asv_y = 550
        self.asv_x = 200

        # Generate static obstacles
        self.obstacles = []
        self.obstacles.append(pygame.Rect(np.random.randint(50,300), 50, 60, 60))
        self.obstacles.append(pygame.Rect(np.random.randint(50,300), 300, 40, 40))
        # self.obstacles.append(pygame.Rect(200, 400, 40, 40))

        if self.render_mode in self.metadata['render_modes']:
            self.render()
        return self._get_obs(), {}

    # Configure terminal condition
    def check_done(self, position):
        # check if asv goes outside of the map
        # top or bottom
        if position[1] <= 0 or position[1] >= self.map_height:
            return True
        # left or right
        if position[0] <= 0 or position[0] >= self.map_width:
            return True

        # collide with an obstacle
        lidar_list = self.lidar.ranges.astype(np.int64)
        if np.any(lidar_list <= self.collision):
            return True

        # for obs in self.obstacles:
        #     if obs.collidepoint(position[0], position[1]):
        #         return True

        return False

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        dx,dy,h,w = self.model.update(100,rudder_action[int(action)],UPDATE_RATE)#pygame.time.get_ticks() / 1000.)
        self.asv_x += dx
        self.asv_y -= dy
        self.asv_h = h
        self.asv_w = w
        self.tgt_y = self.asv_y-50
        self.tgt = self.tgt_x - self.asv_x

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles)

        if self.render_mode in self.metadata['render_modes']:
            self.render()

        """
        Reward function:
            For each step taken: -1
            Stay on or near the path: 0
            Go outside of the map: -10
            Move in reverse: -10
            Collide with an obstacle: -10
        """
        # step loss
        reward = -1
        if self.tgt < self.path_range and self.tgt > -self.path_range:
            # on or near line
            reward = 0
        if dy < 0:
            # moving in reverse
            reward = -10
        # collision
        lidar_list = self.lidar.ranges.astype(np.int64)
        if np.any(lidar_list <= 30):
            reward = -10
        # off border
        if self.asv_x <= 0 or self.asv_x >= self.map_width or self.asv_y >= self.map_height:
            reward = -10

        terminated = self.check_done((self.asv_x, self.asv_y))
        return self._get_obs(), reward, terminated, {}, {}

    def render(self):
        if self.render_mode != 'human':
            return        
        if self.display is None:
            self.display = pygame.display.set_mode(self.screen_size)

        self.surface.fill((0, 0, 0))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.surface, (200, 0, 0), obs)

        # Draw LIDAR scan
        self.lidar.render(self.surface)

        # Draw Path
        pygame.draw.line(self.surface,(0,200,0),(self.tgt_x,0),(self.tgt_x,self.screen_size[1]),5)
        pygame.draw.circle(self.surface,(100,0,0),(self.tgt_x,self.tgt_y),5)

        # Draw ownship
        if self.icon is None:
            self.icon = pygame.image.frombytes(BOAT_ICON['bytes'],BOAT_ICON['size'],BOAT_ICON['format'])

        # Draw status
        lidar = self.lidar.ranges.astype(np.int16)
        if self.status is not None:
            status, rect = self.status.render(f"{self.elapsed_time:005.1f}s  HDG:{self.asv_h:+004.0f}({self.asv_w:+03.0f})  TGT:{self.tgt:+004.0f}",(255,255,255),(0,0,0))
            self.surface.blit(status, [10,550])
            # lidar_status, rect = self.status.render(f"{lidar}",(255,255,255),(0,0,0))
            # self.surface.blit(lidar_status, [5,575])
        os = pygame.transform.rotozoom(self.icon,-self.asv_h,2)
        self.surface.blit(os,os.get_rect(center=(self.asv_x,self.asv_y)))
        self.display.blit(self.surface,[0,0])
        pygame.display.update()
        self.fps_clock.tick(RENDER_FPS)


if __name__ == '__main__':
    env = ASVLidarEnv(render_mode='human')
    env.reset()
    pygame.event.set_allowed((pygame.QUIT,pygame.KEYDOWN,pygame.KEYUP))
    action = CENTER
    total_reward = 0
    while True:
        # # Manual control
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.display.quit()
        #         pygame.quit()
        #         exit()
        #     elif event.type == pygame.KEYUP:
        #         action = CENTER
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_RIGHT:
        #             action = STBD
        #         elif event.key == pygame.K_LEFT:
        #             action = PORT
        # obs,rew,term,_,_ = env.step(action)
        print(total_reward)
        # lidar_list = obs['lidar']
        # print(lidar_list)
        
        # Random actions
        action = env.action_space.sample()
        obs,rew,term,_,_ = env.step(action)


        total_reward += rew
        if term:
            print(f"Elapsed time: {env.elapsed_time}, Reward: {total_reward}")
            pygame.display.quit()
            pygame.quit()
            exit()