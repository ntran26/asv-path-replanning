import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
import pygame
import pygame.freetype
from ship_model import ShipModel, THRUST_COEF, DRAG_COEF, VESSEL_LENGTH, VESSEL_WIDTH, HULL_MARGIN, HULL_FORWARD_SHIFT
from asv_lidar import Lidar, LIDAR_RANGE, LIDAR_BEAMS
from test_run import TestCase
from images import BOAT_ICON
import cv2

RENDER_SCALE = 25
TEST_CASE = None

# System parameters
UPDATE_RATE = 0.1   # 10 Hz
RENDER_FPS = 10
MAP_WIDTH = 10
MAP_HEIGHT = 25
MAX_OBS = 6

# Reward shaping parameters
GAMMA_E      = 0.05
GAMMA_THETA  = 4.0
GAMMA_X      = 0.005
EPSILON_X    = 1.0
ALPHA_R      = 0.1
R_COLLISION  = -2000.0

# Speed control (rpm)
RPM_MIN = 16
RPM_MAX = 32
U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * RPM_MAX)
MAX_IN = 1
MIN_IN = -1

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

        # Path that ASV taken
        self.asv_path = []

        pygame.init()
        self.render_mode = render_mode
        self.world_size = (self.map_width,self.map_height)
        self.render_scale = RENDER_SCALE
        self.window_size = (int(self.map_width*self.render_scale), 
                            int(self.map_height*self.render_scale))

        self.icon = None
        self.fps_clock = pygame.time.Clock()

        self.display = None
        self.surface = None
        self.status = None
        if render_mode in self.metadata['render_modes']:
            self.surface = pygame.Surface(self.window_size)
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
        self.prev_x = None
        self.prev_y = None
        self.speed_mps = 0.0

        self.model = ShipModel()
        self.scenario = TestCase()

        """
        Observation space:
            lidar: an array of lidar range: [63 values]
            pos: (x,y) coordinate of asv
            hdg: heading/yaw of the asv
            dhdg: rate of change of heading
            speed: velocity of the vessel (m/s)
            tgt: horizontal offset of the asv from the path
            target_heading: heading error with respect to the destination point
        """
        self.observation_space = Dict(
            {
                "lidar": Box(low=0, high=LIDAR_RANGE, shape=(LIDAR_BEAMS,), dtype=np.float32),
                "pos"  : Box(low=np.array([0,0]),high=np.array(self.world_size),shape=(2,),dtype=np.int16),
                "hdg"  : Box(low=0,high=360,shape=(1,),dtype=np.int16),
                "dhdg" : Box(low=0,high=36,shape=(1,),dtype=np.int16),
                "speed"  : Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "tgt"  : Box(low=-50,high=50,shape=(1,),dtype=np.int16),
                "target_heading": Box(low=-180,high=180,shape=(1,),dtype=np.int16)
            }
        )
        """
        Action space:
            action = [rudder, throttle] within normalized range [-1,1]
            rudder command: rudder angle percentage for ShipModel.update()
            throttle command: RPM for [RPM_MIN, RPM_MAX]
        """
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # LIDAR
        self.lidar = Lidar()

        # Initialize number of obstacles
        self.max_obs = MAX_OBS

        # Initialize map borders
        self.map_border = [
                            [(0, 0), (0, self.map_height),(0,0),(0, self.map_height)],  
                            [(0, self.map_height), (self.map_width, self.map_height),(0, self.map_height),(self.map_width, self.map_height)],
                            [(self.map_width, self.map_height), (self.map_width, 0),(self.map_width, self.map_height),(self.map_width, 0)],
                            [(0, 0), (self.map_width, 0),(0,0),(self.map_width, 0)]
                        ]

        # Initialize video recorder
        self.record_video = True
        self.video_writer = None
        self.frame_size = self.window_size
        self.video_fps = RENDER_FPS

        self.test_case = TEST_CASE

    def _get_obs(self):
        return {
            'lidar': self.lidar.ranges.astype(np.float32),
            'pos': np.array([self.asv_x, self.asv_y],dtype=np.int16),
            'hdg': np.array([self.asv_h],dtype=np.int16),
            'dhdg': np.array([self.asv_w],dtype=np.int16),
            'speed': np.array([self.speed_mps], dtype=np.float32),
            'tgt': np.array([self.tgt],dtype=np.int16),
            'target_heading': np.array([self.angle_diff],dtype=np.int16)
        }

    def _hull_polygon_world(self):
        """
        Returns 4 points (x,y) of the vessel hull rectangle
        Assumes self.asv_x, self.asv_y is vessel center
        Heading self.asv_h is degrees, where 0 points "up" (negative y)
        """
        L = VESSEL_LENGTH + 2*HULL_MARGIN
        W = VESSEL_WIDTH + 2*HULL_MARGIN

        # optional: if sensor position not at center
        shift = HULL_FORWARD_SHIFT

        half_L = 0.5*L
        half_W = 0.5*W

        h = np.radians(float(self.asv_h))
        sin_h = np.sin(h)
        cos_h = np.cos(h)

        # four corners of the vessel
        local = [(+half_L + shift, +half_W),
                 (+half_L + shift, -half_W),
                 (-half_L + shift, -half_W),
                 (-half_L + shift, +half_W)]
        
        cx = float(self.asv_x)
        cy = float(self.asv_y)

        # Convert (forward, left) -> world (x right, y down)
        # forward vector = (sin(h), -cos(h))
        # left vector    = (-cos(h), -sin(h))
        poly = []
        for x_forward, y_left in local:
            x = cx + x_forward * sin_h - y_left * cos_h
            y = cy - x_forward * cos_h - y_left * sin_h
            poly.append((x,y))
        return poly

    def _polys_intersect_sat(self, polyA, polyB):
        """
        Separating Axis Theorem for convex polygons (works for rectangles).
        polyA, polyB: list of (x,y)
        """
        def project(poly, ax, ay):
            dots = [p[0]*ax + p[1]*ay for p in poly]
            return min(dots), max(dots)
        
        for poly in (polyA, polyB):
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i+1) % n]
                # edge normal (axis)
                ax = -(y2 - y1)
                ay = (x2 - x1)

                minA, maxA = project(polyA, ax, ay)
                minB, maxB = project(polyB, ax, ay)

                if maxA < minB or maxB < minA:
                    return False
        return True
    
    def _check_collision_geom(self):
        """
        True collision if hull intersects any obstacle OR crosses map boundary
        Independent of LiDAR collision_range
        """
        hull = self._hull_polygon_world()

        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]

        # border collision: any corner outside boundary
        if min(xs) < 0 or max(xs) > self.map_width or min(ys) < 0 or max(ys) > self.map_height:
            return True
        
        hx0, hx1 = min(xs), max(xs)
        hy0, hy1 = min(ys), max(ys)

        # Obstacle collision
        for obs in self.obstacles:
            # obs is polygon list [(x,y),...]
            oxs = [p[0] for p in obs]
            oys = [p[1] for p in obs]
            ox0, ox1 = min(oxs), max(oxs)
            oy0, oy1 = min(oys), max(oys)

            if hx1 < ox0 or ox1 < hx0 or hy1 < oy0 or oy1 < hy0:
                continue
            if self._polys_intersect_sat(hull, obs):
                return True
            
        return False

    def _generate_path(self, start_x, start_y, goal_x, goal_y):
        path_length = max(2, int(np.hypot(abs(goal_x - start_x), abs(goal_y - start_y))))

        # record path coordinates
        path_x = np.round(np.linspace(start_x, goal_x, path_length)).astype(int)
        path_y = np.round(np.linspace(start_y, goal_y, path_length)).astype(int)

        # store path coordinates
        path = np.column_stack((path_x, path_y))

        return path
    
    def _generate_obstacles(self, num_obs, test_case=None):
        obstacles = []

        if test_case is None:
            for _ in range(num_obs):
                x = np.random.randint(1, self.map_width - 1)
                y = np.random.randint(1, self.map_height - 1)

                # ensure the obstacle is not close to start/goal 
                if np.linalg.norm([x - self.start_x, y - self.start_y]) > 1 and \
                    np.linalg.norm([x - self.goal_x, y - self.goal_y]) > 1:
                    obstacles.append([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])

        else:
            obstacles = self.scenario.obstacles(test_case=test_case)

        return obstacles
    
    # Calculate the relative angle between current heading and goal
    def _calculate_angle(self, asv_x, asv_y, heading, goal_x, goal_y):
        dx = goal_x - asv_x
        dy = goal_y - asv_y

        target_angle = np.degrees(np.arctan2(dx, -dy))       # pygame invert y-axis
        angle_diff = (target_angle - heading + 180) % 360 - 180    # normalize to [-180,180]

        return angle_diff
    
    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=10, exclude_corner=True):
        # convert to numpy array
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        # get distance between start and end pos
        length = np.linalg.norm(end_pos - start_pos)
        dash_amount = int(length/dash_length)

        dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()
        
        return [pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width) for n in range(int(exclude_corner), dash_amount - int(exclude_corner), 2)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset episode-level state
        self.elapsed_time = 0.0
        self.asv_h = 0.0
        self.asv_w = 0.0
        self.tgt = 0.0
        self.angle_diff = 0.0

        # Reset dynamics + sensors
        self.model = ShipModel()
        self.model._v = 0
        self.lidar.reset()

        # Randomize start and goal positions
        if self.test_case is None:
            self.start_x = np.random.randint(1, self.map_width - 1)
            self.start_y = self.map_height - 1
            self.goal_x = np.random.randint(1, self.map_width - 1)
            self.goal_y = 1    
        else:
            self.start_x, self.start_y, self.goal_x, self.goal_y = self.scenario.position(test_case=self.test_case)

        self.asv_x = self.start_x
        self.asv_y = self.start_y

        self.prev_x = float(self.asv_x)
        self.prev_y = float(self.asv_y)
        self.speed_mps = 0.0

        # Generate the path
        self.path = self._generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        # Generate static obstacles
        self.num_obs = np.random.randint(0, self.max_obs)
        self.obstacles = self._generate_obstacles(self.num_obs, self.test_case)

        # Initialize the ASV path list
        self.asv_path = [(self.asv_x, self.asv_y)]

        # Initialize distance to goal
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))

        self.reward = 0

        if self.render_mode in self.metadata['render_modes']:
            self.render()
        return self._get_obs(), {}

    # Configure terminal condition
    def check_done(self, position):
        # collide with an obstacle or out of bounds
        if self._check_collision_geom():
            return True
        
        # the agent reaches goal
        if self.distance_to_goal <= HULL_MARGIN:
            return True

        return False

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], MIN_IN, MAX_IN))
        throttle_cmd = float(np.clip(action[1], MIN_IN, MAX_IN))

        # Map rudder_cmd [-1,1] -> rudder [-25, 25]
        rudder = rudder_cmd * 25

        # Map throttle_cmd [-1,1] -> rpm [RPM_MIN, RPM_MAX]
        rpm = (throttle_cmd - MIN_IN) * ((RPM_MAX - RPM_MIN)/(MAX_IN - MIN_IN)) + RPM_MIN

        # Store current position
        x_prev = float(self.asv_x)
        y_prev = float(self.asv_y)

        dx,dy,h,w = self.model.update(rpm, rudder, UPDATE_RATE)  # ShipModel expects rudder in [-100,100]
        self.asv_x += dx
        self.asv_y -= dy
        self.asv_h = h
        self.asv_w = w

        # calculate speed
        dx_pos = float(self.asv_x) - x_prev
        dy_pos = float(self.asv_y) - y_prev
        speed_units_per_s = np.sqrt((dx_pos*dx_pos + dy_pos*dy_pos)) / float(UPDATE_RATE)
        self.speed_mps = float(speed_units_per_s)

        # closest perpendicular distance from asv to path
        asv_pos = np.array([self.asv_x, self.asv_y])
        distance = np.linalg.norm(self.path - asv_pos, axis=1)
        self.tgt = np.min(distance)

        # extract (x,y) target
        closest_idx = np.argmin(distance)
        self.tgt_x, self.tgt_y = self.path[closest_idx]

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)

        self.angle_diff = self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y)
        
        if self.render_mode in self.metadata['render_modes']:
            self.render()
        
        # append new coordinate of asv
        self.asv_path.append((self.asv_x, self.asv_y))

        # update distance to goal
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))

        # Define terminal flags
        collided = bool(self._check_collision_geom())
        reached_goal = bool(self.distance_to_goal <= VESSEL_LENGTH/2)

        # Speed
        U = float(self.speed_mps)
        U_max = float(max(U_MAX, 1e-6))
        U_norm = float(U / U_max)

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
        if reached_goal:
            r_goal = 50
        else:
            r_goal = 0

        # Combined rewards
        lambda_ = 0.5       # weighting factor
        # reward = lambda_ * r_pf + (1 - lambda_) * r_oa + r_exist + r_goal + r_heading

        if collided:
            reward = -1000
        else:
            reward = lambda_ * r_pf + (1 - lambda_) * r_oa + r_heading + r_exist + r_goal

        terminated = self.check_done((self.asv_x, self.asv_y))

        info = {
            # reward mix + components
            "lam": float(lambda_),
            "r_pf": float(r_pf),
            "r_oa": float(r_oa),
            "r_exist": float(r_exist),
            "r_heading": float(r_heading),

            # speed diagnostics
            "U": float(U),
            "U_norm": float(U_norm),
            "speed_mps": float(self.speed_mps),

            # actions in real units (useful for debugging)
            "rpm": float(rpm),
            "rudder_deg": float(rudder_cmd * 25.0),

            # navigation + safety
            "distance_to_goal": float(self.distance_to_goal),
            "tgt": float(self.tgt),
            "angle_diff": float(self.angle_diff),
            "collision": bool(self._check_collision_geom()),
        }

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode != 'human':
            return        
        if self.display is None:
            self.display = pygame.display.set_mode(self.window_size)
        
        scale = float(self.render_scale)

        def scale_point(xy):    # scale point from world -> pixel
            return (int(round(xy[0] * scale)), int(round(xy[1] * scale)))
        def scale_scalar(v):    # scale scalar (radius, width, dash length) -> pixels
            return max(1, int(round(v * scale)))

        self.surface.fill((0, 0, 0))

        # Draw map boundaries
        bw = max(2, int(round(2)))  # keep border thickness readable in pixels
        W = self.window_size[0] - 1
        H = self.window_size[1] - 1
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (0, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, H), (W, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (W, 0), (W, H), bw)
        pygame.draw.line(self.surface, (200, 0, 0), (0, 0), (W, 0), bw)

        # Draw obstacles
        for obs in self.obstacles:
            obs_px = [scale_point(p) for p in obs]
            pygame.draw.polygon(self.surface, (200, 0, 0), obs_px)

        # Draw LIDAR scan
        self.lidar.render(self.surface, scale=scale)

        # Draw Path
        self._draw_dashed_line(
            self.surface,
            (0,200,0),
            scale_point((self.start_x,self.start_y)),
            scale_point((self.goal_x,self.goal_y)),
            width=2,
            dash_length=int(np.clip(scale, 8, 30))
        )
        pygame.draw.circle(self.surface,(100,0,0),
                           scale_point((self.tgt_x,self.tgt_y)),
                           radius=3)

        # Draw destination
        pygame.draw.circle(self.surface, (200, 0, 200), 
                           scale_point((self.goal_x, self.goal_y)), 
                           max(4, int(round(6))))

        # Draw ownship
        if self.icon is None:
            self.icon = pygame.image.frombytes(BOAT_ICON['bytes'],BOAT_ICON['size'],BOAT_ICON['format'])
            self.icon_scaled = None
            self._icon_scaled_size = None

        icon_width = max(1, int(round(VESSEL_WIDTH * scale)))
        icon_length = max(1, int(round(VESSEL_LENGTH * scale)))
        icon_size = (icon_width, icon_length)

        if self.icon_scaled is None or self._icon_scaled_size != icon_size:
            self.icon_scaled = pygame.transform.smoothscale(self.icon, icon_size)
            self._icon_scaled_size = icon_size

        # Draw status
        os = pygame.transform.rotozoom(self.icon_scaled, -self.asv_h, 1)
        self.surface.blit(os, os.get_rect(center=scale_point((self.asv_x, self.asv_y))))
        ship_outline = self._hull_polygon_world()
        ship_outline_px = [scale_point(p) for p in ship_outline]
        pygame.draw.polygon(self.surface, (255, 0, 0), ship_outline_px, width=max(2, int(round(2))))

        if self.status is not None:
            status_surf_1, rect = self.status.render(
                f"{self.elapsed_time:005.1f}s  V:{self.speed_mps:0.2f}m/s  "
                f"HDG:{self.asv_h:+004.0f}({self.asv_w:+03.0f})  "
                f"TGT:{self.tgt:+004.0f}    ",
                (255, 255, 255),
                (0, 0, 0)
            )
            status_surf_2, rect = self.status.render(
                f"TGT_HDG:{self.angle_diff:.2f}    "
                f"GOAL:{self.distance_to_goal:.2f}  ",
                (255, 255, 255),
                (0, 0, 0)
            )
            self.surface.blit(status_surf_1, (10, self.window_size[1] - 30))
            self.surface.blit(status_surf_2, (10, self.window_size[1] - 15))

        self.display.blit(self.surface, (0, 0))
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
        # Random actions
        action = env.action_space.sample()
        obs,rew,term,_,_ = env.step(action)
        total_reward += rew
        # print(f"Action: {action}    Reward: {rew}")
        if term:
            print(f"Elapsed time: {env.elapsed_time}, Reward: {total_reward:0.2f}")         
            pygame.display.quit()
            pygame.quit()
            exit()