import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
import pygame
import pygame.freetype
from ship_model import ShipModel
from asv_lidar import Lidar, LIDAR_RANGE, LIDAR_BEAMS
from images import BOAT_ICON
import cv2

UPDATE_RATE = 0.5
RENDER_FPS = 10
MAP_WIDTH = 400
MAP_HEIGHT = 600
NUM_OBS = 5
COLLISION_RANGE = 10

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

        self.collision = COLLISION_RANGE

        # Path that ASV taken
        self.asv_path = []

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
        self.angle_diff = 0

        self.model = ShipModel()
        self.model._v = 4.5

        """
        Observation space:
            lidar: an array of lidar range: [63 values]
            pos: (x,y) coordinate of asv
            hdg: heading/yaw of the asv
            dhdg: rate of change of heading
            tgt: horizontal offset of the asv from the path
            target_heading: heading error with respect to the destination point
        """
        self.observation_space = Dict(
            {
                "lidar": Box(low=0,high=LIDAR_RANGE,shape=(LIDAR_BEAMS,),dtype=np.int16),
                "pos"  : Box(low=np.array([0,0]),high=np.array(self.screen_size),shape=(2,),dtype=np.int16),
                "hdg"  : Box(low=0,high=360,shape=(1,),dtype=np.int16),
                "dhdg" : Box(low=0,high=36,shape=(1,),dtype=np.int16),
                "tgt"  : Box(low=-50,high=50,shape=(1,),dtype=np.int16),
                "target_heading": Box(low=-180,high=180,shape=(1,),dtype=np.int16)
            }
        )

        self.action_space = Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        
        # LIDAR
        self.lidar = Lidar()

        # Initialize number of obstacles
        self.num_obs = NUM_OBS

        # Initialize map borders
        # self.map_border = [(0,0), (0,self.map_height), (self.map_width,self.map_height), (self.map_width,0)]
        self.map_border = [
                            [(0, 0), (0, self.map_height),(0,0),(0, self.map_height)],  
                            [(0, self.map_height), (self.map_width, self.map_height),(0, self.map_height),(self.map_width, self.map_height)],
                            [(self.map_width, self.map_height), (self.map_width, 0),(self.map_width, self.map_height),(self.map_width, 0)],
                            [(0, 0), (self.map_width, 0),(0,0),(self.map_width, 0)]
                        ]

        # Initialize video recorder
        self.record_video = True
        self.video_writer = None
        self.frame_size = (self.map_width, self.map_height)
        self.video_fps = RENDER_FPS

    def _get_obs(self):
        return {
            'lidar': self.lidar.ranges.astype(np.int16),
            'pos': np.array([self.asv_x, self.asv_y],dtype=np.int16),
            'hdg': np.array([self.asv_h],dtype=np.int16),
            'dhdg': np.array([self.asv_w],dtype=np.int16),
            'tgt': np.array([self.tgt],dtype=np.int16),
            'target_heading': np.array([self.angle_diff],dtype=np.int16)
        }

    def generate_path(self, start_x, start_y, goal_x, goal_y):
        path_length = max(2, int(np.hypot(abs(goal_x - start_x), abs(goal_y - start_y))))

        # record path coordinates
        path_x = np.round(np.linspace(start_x, goal_x, path_length)).astype(int)
        path_y = np.round(np.linspace(start_y, goal_y, path_length)).astype(int)

        # store path coordinates
        path = np.column_stack((path_x, path_y))

        return path
    
    def generate_obstacles(self, num_obs):
        obstacles = []
        for _ in range(num_obs):
            x = np.random.randint(50, self.map_width - 50)
            y = np.random.randint(50, self.map_height - 150)

            # ensure the obstacle is not close to start/goal 
            if np.linalg.norm([x - self.start_x, y - self.start_y]) > 100 and \
                np.linalg.norm([x - self.goal_x, y - self.goal_y]) > 100:
                obstacles.append([(x, y), (x+50, y), (x+50, y+50), (x, y+50)])

        return obstacles

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        # Randomize start position
        self.start_y = self.map_height - 50
        self.start_x = np.random.randint(50, self.map_width - 50)
        # self.start_x = 100

        # # Initialize asv position (fixed)
        # self.asv_x = self.map_width/2
        # self.asv_y = self.start_y

        # Initialize asv position (random)
        if self.start_x > 100 and self.start_x < self.map_width - 100:
            self.asv_x = np.random.randint(self.start_x - 50, self.start_x + 50)
        elif self.start_x <= 100:
            self.asv_x = self.start_x + 50
        elif self.start_x >= self.map_width - 100:
            self.asv_x = self.start_x - 50
        
        self.asv_y = self.start_y

        # Randomize goal position
        self.goal_y = 50
        self.goal_x = np.random.randint(50, self.map_width - 50)
        # self.goal_x = self.start_x

        # Generate the path
        self.path = self.generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        # Generate static obstacles
        self.num_obs = np.random.randint(0, NUM_OBS)
        self.obstacles = self.generate_obstacles(self.num_obs)

        # Initialize the ASV path list
        self.asv_path = [(self.asv_x, self.asv_y)]

        if self.render_mode in self.metadata['render_modes']:
            self.render()
        return self._get_obs(), {}

    # Configure terminal condition
    def check_done(self, position):
        # # check if asv goes outside of the map
        # # top or bottom
        if position[1] >= self.map_height:
            return True
        # # left or right
        # if position[0] <= 0 or position[0] >= self.map_width:
        #     return True

        # collide with an obstacle
        lidar_list = self.lidar.ranges.astype(np.int64)
        if np.any(lidar_list <= self.collision):
            return True
        
        # the agent reaches goal
        if self.distance_to_goal <= self.collision+30:
            return True

        return False
    
    # Calculate the relative angle between current heading and goal
    def calculate_angle(self, asv_x, asv_y, heading, goal_x, goal_y):
        dx = goal_x - asv_x
        dy = goal_y - asv_y

        target_angle = np.degrees(np.arctan2(dx, -dy))       # pygame invert y-axis
        angle_diff = (target_angle - heading + 180) % 360 - 180    # normalize to [-180,180]

        return angle_diff

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder = float(np.clip(action[0], -1, 1))
        dx,dy,h,w = self.model.update(100,rudder*25,UPDATE_RATE)#pygame.time.get_ticks() / 1000.)
        self.asv_x += dx
        self.asv_y -= dy
        self.asv_h = h
        self.asv_w = w

        # closest perpendicular distance from asv to path
        asv_pos = np.array([self.asv_x, self.asv_y])
        distance = np.linalg.norm(self.path - asv_pos, axis=1)
        self.tgt = np.min(distance)

        # extract (x,y) target
        closest_idx = np.argmin(distance)
        self.tgt_x, self.tgt_y = self.path[closest_idx]

        # self.tgt_y = self.asv_y-50
        # self.tgt_x = self.goal_x
        # self.tgt = self.tgt_x - self.asv_x

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)

        self.angle_diff = self.calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y)
        
        if self.render_mode in self.metadata['render_modes']:
            self.render()
        
        # append new coordinate of asv
        self.asv_path.append((self.asv_x, self.asv_y))

        """
        Reward function
        """

        # penatly for each step taken
        # if dy < 0:
        #     r_exist = -10
        # else:
        r_exist = -1

        # heading alignment reward (reward = 1 if aligned, -1 if opposite)
        angle_diff_rad = np.radians(self.angle_diff)
        r_heading = np.cos(angle_diff_rad)

        # path following reward
        r_pf = np.exp(-0.05 * abs(self.tgt))

        # obstacle avoidance reward
        lidar_list = self.lidar.ranges.astype(np.float32)
        r_oa = 0
        for i, dist in enumerate(lidar_list):
            theta = self.lidar.angles[i]    # angle of lidar beam
            weight = 1 / (1 + abs(theta))   # prioritize beams closer to center/front
            r_oa += weight / max(dist, 1)
        r_oa = -r_oa / len(lidar_list)

        # if the agent reaches goal
        self.distance_to_goal = np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y])
        if self.distance_to_goal <= self.collision+30:
            r_goal = 50
        else:
            r_goal = 0

        # Combined rewards
        lambda_ = 0.9       # weighting factor
        # reward = lambda_ * r_pf + (1 - lambda_) * r_oa + r_exist + r_goal + r_heading

        if np.any(self.lidar.ranges.astype(np.int64) <= self.collision):
            reward = -1000
        else:
            reward = lambda_ * r_pf + (1 - lambda_) * r_oa + r_heading + r_exist + r_goal

        terminated = self.check_done((self.asv_x, self.asv_y))
        return self._get_obs(), reward, terminated, False, {}
    
    def draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=10, exclude_corner=True):
        # convert to numpy array
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        # get distance between start and end pos
        length = np.linalg.norm(end_pos - start_pos)
        dash_amount = int(length/dash_length)

        dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()
        
        return [pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width) for n in range(int(exclude_corner), dash_amount - int(exclude_corner), 2)]

    def render(self):
        if self.render_mode != 'human':
            return        
        if self.display is None:
            self.display = pygame.display.set_mode(self.screen_size)

        self.surface.fill((0, 0, 0))

        # Draw map boundaries
        line = self.map_border
        pygame.draw.line(self.surface, (200, 0, 0), (0,0), (0,self.map_height), 5)
        pygame.draw.line(self.surface, (200, 0, 0), (0,self.map_height), (self.map_width,self.map_height), 5)
        pygame.draw.line(self.surface, (200, 0, 0), (self.map_width,0), (self.map_width,self.map_height), 5)
        # pygame.draw.line(self.surface, (200, 0, 0), (0,0), (self.map_width,0), 5)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.polygon(self.surface, (200, 0, 0), obs)

        # Draw LIDAR scan
        self.lidar.render(self.surface)

        # Draw Path
        self.draw_dashed_line(self.surface,(0,200,0),(self.start_x,self.start_y),(self.goal_x,self.goal_y),width=5)
        pygame.draw.circle(self.surface,(100,0,0),(self.tgt_x,self.tgt_y),5)

        # Draw destination
        pygame.draw.circle(self.surface,(200,0,200),(self.goal_x,self.goal_y),10)

        # Draw ownship
        if self.icon is None:
            self.icon = pygame.image.frombytes(BOAT_ICON['bytes'],BOAT_ICON['size'],BOAT_ICON['format'])

        # Draw status
        lidar = self.lidar.ranges.astype(np.int16)
        if self.status is not None:
            status, rect = self.status.render(f"{self.elapsed_time:005.1f}s  HDG:{self.asv_h:+004.0f}({self.asv_w:+03.0f})  TGT:{self.tgt:+004.0f}  TGT_HDG:{self.angle_diff:.2f}",(255,255,255),(0,0,0))
            self.surface.blit(status, [10,550])
            # lidar_status, rect = self.status.render(f"{lidar}",(255,255,255),(0,0,0))
            # self.surface.blit(lidar_status, [5,575])

        os = pygame.transform.rotozoom(self.icon,-self.asv_h,2)
        self.surface.blit(os,os.get_rect(center=(self.asv_x,self.asv_y)))
        self.display.blit(self.surface,[0,0])
        pygame.display.update()
        self.fps_clock.tick(RENDER_FPS)

        # Capture frame and save to video
        if self.record_video:
            frame = pygame.surfarray.array3d(self.surface)  # convert pygame surface to numpy array
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # rotate for correct orientation
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert RGB to BGR (opencv)
            
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
                self.video_writer = cv2.VideoWriter('asv_lidar.mp4', fourcc, self.video_fps, self.frame_size)

            self.video_writer.write(frame)

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
        # lidar_list = obs['lidar']
        
        # Random actions
        action = env.action_space.sample()
        obs,rew,term,_,_ = env.step(action)

        # print(lidar_list)
        # print(total_reward)
        # print(obs)
        total_reward += rew
        if term:
            print(f"Elapsed time: {env.elapsed_time}, Reward: {total_reward:0.2f}")

            # Save path taken as image
            path_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
            path_surface.fill((255,255,255))

            for i in range(1, len(env.asv_path)):
                pygame.draw.circle(path_surface, (0, 0, 200), env.asv_path[i], 5)

            # Draw obstacles
            for obs in env.obstacles:
                pygame.draw.polygon(path_surface, (200, 0, 0), obs)
            
            # Draw Path
            pygame.draw.line(path_surface,(0,200,0),(env.start_x,env.start_y),(env.goal_x,env.goal_y),5)
            pygame.draw.circle(path_surface,(100,0,0),(env.tgt_x,env.tgt_y),5)

            # Draw map boundaries
            pygame.draw.line(path_surface, (200, 0, 0), (0,0), (0,env.map_height), 5)
            pygame.draw.line(path_surface, (200, 0, 0), (0,env.map_height), (env.map_width,env.map_height), 5)
            pygame.draw.line(path_surface, (200, 0, 0), (env.map_width,0), (env.map_width,env.map_height), 5)
            pygame.draw.line(path_surface, (200, 0, 0), (0,0), (env.map_width,0), 5)

            pygame.image.save(path_surface, "asv_path_result.png")          

            pygame.display.quit()
            pygame.quit()
            exit()