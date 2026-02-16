import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import pygame
import pygame.freetype
from ship_model import ShipModel
from asv_lidar import Lidar, LIDAR_RANGE, LIDAR_BEAMS
from images import BOAT_ICON
import json
import cv2

"""
Test case 0: random start & goal points, random obstacles

Test case 1: start & goal at center, no obstacles (path following)
Test case 2: start & goal at center, single obstacle on path
Test case 3: start & goal at center, single obstacle to the left of path
Test case 4: start & goal at center, single obstacle to the right of path
Test case 5: start & goal at center, 3 obstacles scatter along the path
Test case 6: start & goal at each corner, 4 obstacles cover the path, leaving a single blank space
Test case 7: no obstacles

Else: take the setup from recorded data of a random obstacles scenario (test case 0)
"""

TEST_CASE = 0
ENV_DATA = "data/env_setup/env_4.json"

UPDATE_RATE = 0.5
RENDER_FPS = 10
MAP_WIDTH = 400
MAP_HEIGHT = 600
NUM_OBS = 10
COLLISION_RANGE = 10
LIDAR_PARTITION = 15

class testEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
            self, 
            render_mode:str = 'human'
            ) -> None:
        self.test_case = TEST_CASE

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
        self.model._v = 0.0

        self.observation_space = Dict(
            {
                # "lidar": Box(low=0,high=LIDAR_RANGE,shape=(LIDAR_BEAMS,),dtype=np.int16),
                "lidar": Box(low=0,high=1,shape=(LIDAR_PARTITION,),dtype=np.float32),
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

        self.env_data = ENV_DATA

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
        # calculate number of waypoints
        path_length = max(2, int(np.hypot(abs(goal_x - start_x), abs(goal_y - start_y))))

        # record path coordinates
        path_x = np.round(np.linspace(start_x, goal_x, path_length)).astype(int)
        path_y = np.round(np.linspace(start_y, goal_y, path_length)).astype(int)

        # store path coordinates
        path = np.column_stack((path_x, path_y))
        converted_path = [tuple(map(int, point)) for point in path]

        return converted_path
    
    def generate_obstacles(self, test_case):
        obstacles = []
        if test_case == 0:          # 10 random obstacles
            for _ in range(NUM_OBS):
                x = np.random.randint(50, self.map_width - 50)
                y = np.random.randint(50, self.map_height - 150)

                # ensure the obstacle is not close to start/goal 
                if np.linalg.norm([x - self.start_x, y - self.start_y]) > 100 and \
                    np.linalg.norm([x - self.goal_x, y - self.goal_y]) > 100:
                    # obstacles.append([(x, y), (x+50, y), (x+50, y+50), (x, y+50)])
                    obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])

        elif test_case == 7:
            obstacles = []
        elif test_case == 1:
            x = 150
            y = 300
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
        elif test_case == 2:
            x = 125
            y = 300
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
        elif test_case == 3:
            x = 175
            y = 300
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
        elif test_case == 4:
            x = 125
            y = 125
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
            x = 275
            y = 225
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
            x = 200
            y = 325
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
        elif test_case == 5:
            x = 50
            y = 325
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
            x = 110
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
            x = 170
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
            x = 320
            obstacles.append([(x-25,y-25), (x+25,y-25), (x+25,y+25), (x-25,y+25)])
        elif test_case == 6:
            x = 90
            y = 300
            obstacles.append([(x-80, y-150), (x+80, y-150), (x+80, y+150), (x-80, y+150)])
            x = 310
            obstacles.append([(x-60, y-100), (x+60, y-100), (x+60, y+100), (x-60, y+100)])
            x = 200
            y = 50
            obstacles.append([(x-60, y-30), (x+90, y-30), (x+90, y+30), (x-60, y+30)])

        else:
            # load data file
            with open(self.env_data, "r") as f:
                data = json.load(f)
            obstacles = data["obstacles"]

        return obstacles

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        if self.test_case == 0:
            # Randomize start position
            self.start_y = self.map_height - 50
            self.start_x = np.random.randint(50, self.map_width - 50)
            # Randomize goal position
            self.goal_y = 50
            self.goal_x = np.random.randint(50, self.map_width - 50)
            # Initialize asv position (random)
            if self.start_x > 100 and self.start_x < self.map_width - 100:
                self.asv_x = np.random.randint(self.start_x - 50, self.start_x + 50)
            elif self.start_x <= 100:
                self.asv_x = self.start_x + 50
            elif self.start_x >= self.map_width - 100:
                self.asv_x = self.start_x - 50
        
        elif self.test_case == 7:
            self.start_x = 150
            self.start_y = 550

            self.goal_x = 150
            self.goal_y = 50

            self.asv_x = 150

        elif self.test_case >= 1 and self.test_case <= 3:
            self.start_x = 150
            self.start_y = 550

            self.goal_x = 150
            self.goal_y = 50

            self.asv_x = 150
        
        elif self.test_case == 4:
            self.start_x = 200
            self.start_y = 550
            
            self.goal_x = 200
            self.goal_y = 50

            self.asv_x = 200
        
        elif self.test_case == 5:
            self.start_x = 50
            self.start_y = 550

            self.goal_x = 250
            self.goal_y = 50

            self.asv_x = 50

        elif self.test_case == 6:
            self.start_x = 350
            self.start_y = 550
            self.goal_x = 50
            self.goal_y = 50
            self.asv_x = 300
        
        else:
            # load data file
            with open(self.env_data, "r") as f:
                data = json.load(f)
            self.start_x = data["start"][0]
            self.start_y = data["start"][1]
            self.goal_x = data["goal"][0]
            self.goal_y = data["goal"][1]
            self.asv_x = data["asv_start"][0]

        self.asv_y = self.start_y

        # Generate the path
        self.path = self.generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        # Generate static obstacles
        self.obstacles = self.generate_obstacles(self.test_case)

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

        # penatly for each step taken
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
        lambda_ = 0.5       # weighting factor
        reward = lambda_ * r_pf + (1 - lambda_) * r_oa + r_exist + r_goal + r_heading

        if np.any(self.lidar.ranges.astype(np.int64) <= self.collision):
            reward = -1000
        else:
            reward = lambda_ * r_pf + (1 - lambda_) * r_oa + r_heading + r_exist + r_goal

        terminated = self.check_done((self.asv_x, self.asv_y))
        return self._get_obs(), reward, terminated, False, {}

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
        pygame.draw.line(self.surface, (200, 0, 0), (0,0), (self.map_width,0), 5)

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
    
    def draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=10, exclude_corner=True):
        # convert to numpy array
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        # get distance between start and end pos
        length = np.linalg.norm(end_pos - start_pos)
        dash_amount = int(length/dash_length)

        dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()
        
        return [pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width) for n in range(int(exclude_corner), dash_amount - int(exclude_corner), 2)]