import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import pygame
import pygame.freetype
from ship_model import ShipModel, THRUST_COEF, DRAG_COEF, VESSEL_LENGTH, VESSEL_WIDTH, HULL_MARGIN, HULL_FORWARD_SHIFT, MAX_RUD_ANGLE
from asv_lidar import Lidar, LIDAR_MIN_RANGE, LIDAR_RANGE, LIDAR_BEAMS
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
MAX_OBS = 8
DHDG_MAX_DPS = 180

# Reward shaping parameters
GAMMA_E = 0.12
ALPHA_R = 0.1
R_COLLISION = -2000.0
R_GOAL = 300.0

# Threat-adaptive blend between path following and obstacle avoidance.
LAMBDA_CLEAR = 0.70
LAMBDA_THREAT = 0.25

OA_FRONT_HALF_ANGLE_DEG = 45
OA_FRONT_PERCENTILE = 10
OA_CENTER_HALF_ANGLE_DEG = 20
OA_WARN_CLEARANCE = 4.5
OA_CRIT_CLEARANCE = 1.5
OA_NEAR_CLEARANCE = 1.8
OA_CENTER_GAIN = 2.5
OA_DIR_GAIN = 0.8
OA_DIR_SCALE = 1.5
OA_NEAR_GAIN = 2.0
OA_SPEED_GAIN = 1.0

# LiDAR-sector observation hysteresis.
SECTOR_CLEAR_RECOVERY_M = 0.20

# Speed control (rpm)
RPM_MIN = 0
RPM_MAX = 24
U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * RPM_MAX)
MAX_IN = 1
MIN_IN = -1
GOAL_DIST_MAX = float(np.hypot(MAP_WIDTH, MAP_HEIGHT))

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
        self.speed_mps = 0.
        self.goal_dist_norm = 1.0
        self.left_clear_norm = 1.0
        self.center_clear_norm = 1.0
        self.right_clear_norm = 1.0
        self.left_clear_instant_norm = 1.0
        self.center_clear_instant_norm = 1.0
        self.right_clear_instant_norm = 1.0
        self.gap_asymmetry = 0.0
        self.rudder_state_norm = 0.0
        self.rpm_state_norm = 0.0
        self.left_clear_lidar_m = float(LIDAR_RANGE)
        self.center_clear_lidar_m = float(LIDAR_RANGE)
        self.right_clear_lidar_m = float(LIDAR_RANGE)

        self.model = ShipModel()
        self.scenario = TestCase()

        """
        Observation space:
            lidar: an array of lidar range: [63 values]
            pos: (x,y) coordinate of asv
            hdg: heading/yaw of the asv 
            dhdg: rate of change of heading (normalized)
            speed: velocity of the vessel (m/s)
            tgt: horizontal offset of the asv from the path
            target_heading: heading error with respect to the destination point
            goal_dist: normalized distance-to-goal
            left/center/right_clear: compact LiDAR-derived sector clearances with hysteresis
            left/center/right_clear_instant: instantaneous LiDAR sector clearances
            gap_asymmetry: normalized right-vs-left clearance bias
            rudder_state: current actual rudder state (normalized)
            rpm_state: current rpm command state (normalized)
        """
        self.observation_space = Dict(
            {
                "lidar": Box(low=LIDAR_MIN_RANGE, high=LIDAR_RANGE, shape=(LIDAR_BEAMS,), dtype=np.float32),
                "pos"  : Box(low=np.array([0,0]),high=np.array(self.world_size),shape=(2,),dtype=np.float32),
                "hdg"  : Box(low=0,high=360,shape=(1,),dtype=np.float32),
                "dhdg" : Box(low=-1.0,high=1.0,shape=(1,),dtype=np.float32),
                "speed"  : Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "tgt"  : Box(low=-50,high=50,shape=(1,),dtype=np.float32),
                "target_heading": Box(low=-180,high=180,shape=(1,),dtype=np.float32),
                "goal_dist": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "left_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "center_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "right_clear": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "left_clear_instant": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "center_clear_instant": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "right_clear_instant": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "gap_asymmetry": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "rudder_state": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "rpm_state": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
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
            'pos': np.array([self.asv_x, self.asv_y],dtype=np.float32),
            'hdg': np.array([self.asv_h],dtype=np.float32),
            'dhdg': np.array([np.clip(self.asv_w / 180, -1.0, 1.0)],dtype=np.float32),
            'speed': np.array([self.speed_mps], dtype=np.float32),
            'tgt': np.array([self.tgt],dtype=np.float32),
            'target_heading': np.array([self.angle_diff],dtype=np.float32),
            'goal_dist': np.array([self.goal_dist_norm], dtype=np.float32),
            'left_clear': np.array([self.left_clear_norm], dtype=np.float32),
            'center_clear': np.array([self.center_clear_norm], dtype=np.float32),
            'right_clear': np.array([self.right_clear_norm], dtype=np.float32),
            'left_clear_instant': np.array([self.left_clear_instant_norm], dtype=np.float32),
            'center_clear_instant': np.array([self.center_clear_instant_norm], dtype=np.float32),
            'right_clear_instant': np.array([self.right_clear_instant_norm], dtype=np.float32),
            'gap_asymmetry': np.array([self.gap_asymmetry], dtype=np.float32),
            'rudder_state': np.array([self.rudder_state_norm], dtype=np.float32),
            'rpm_state': np.array([self.rpm_state_norm], dtype=np.float32),
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

        # Convert (forward, left) -> world (x right, y up)
        # forward vector = (+sin(h), +cos(h))
        # left vector    = (-cos(h), +sin(h))
        poly = []
        for x_forward, y_left in local:
            x = cx + x_forward * sin_h - y_left * cos_h
            y = cy + x_forward * cos_h + y_left * sin_h
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

        target_angle = np.degrees(np.arctan2(dx, dy))    
        angle_diff = (target_angle - heading + 180) % 360 - 180    # normalize to [-180,180]

        return angle_diff

    def _signed_cross_track_error(self, asv_x, asv_y):
        path_dx = float(self.goal_x - self.start_x)
        path_dy = float(self.goal_y - self.start_y)
        path_norm = float(np.hypot(path_dx, path_dy))
        if path_norm < 1e-6:
            return 0.0

        rel_x = float(asv_x - self.start_x)
        rel_y = float(asv_y - self.start_y)
        cross = path_dx * rel_y - path_dy * rel_x
        return float(cross / path_norm)

    def _border_clearance(self):
        hull = self._hull_polygon_world()
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        return float(min(min(xs), self.map_width - max(xs), min(ys), self.map_height - max(ys)))

    def _geometry_ranges_for_angles(self, beam_angles_deg):
        """True geometry clearances without LiDAR blind-zone saturation."""
        if beam_angles_deg.size == 0:
            return np.empty(0, dtype=np.float32)

        origin = (float(self.lidar._pos_x), float(self.lidar._pos_y))
        edges = []

        for obs in self.obstacles:
            for i in range(len(obs)):
                edges.append((obs[i], obs[(i + 1) % len(obs)]))

        for border in self.map_border:
            for i in range(len(border)):
                edges.append((border[i], border[(i + 1) % len(border)]))

        geom_ranges = np.full(beam_angles_deg.shape, LIDAR_RANGE, dtype=np.float32)
        for idx, angle in enumerate(beam_angles_deg):
            absolute_angle = np.radians(float(self.asv_h) + float(angle))
            end_x = origin[0] + LIDAR_RANGE * np.sin(absolute_angle)
            end_y = origin[1] + LIDAR_RANGE * np.cos(absolute_angle)

            closest_distance = float(LIDAR_RANGE)
            for edge in edges:
                intersection = self.lidar.line_intersection(origin, (end_x, end_y), edge[0], edge[1])
                if intersection is not None:
                    dist = float(np.hypot(intersection[0] - origin[0], intersection[1] - origin[1]))
                    closest_distance = min(closest_distance, dist)

            geom_ranges[idx] = closest_distance

        return geom_ranges

    def _compute_clearance_features(self):
        theta_deg = self.lidar.angles.astype(np.float32)
        lidar_d = self.lidar.ranges.astype(np.float32)

        raw_front_mask = np.abs(theta_deg) <= OA_FRONT_HALF_ANGLE_DEG
        raw_front_ranges = lidar_d[raw_front_mask] if np.any(raw_front_mask) else lidar_d
        raw_front_ranges = raw_front_ranges[np.isfinite(raw_front_ranges)]
        if raw_front_ranges.size == 0:
            raw_front_ranges = np.array([LIDAR_RANGE], dtype=np.float32)
        raw_front_clear = float(np.percentile(raw_front_ranges, OA_FRONT_PERCENTILE))

        geom_theta_deg = theta_deg[raw_front_mask] if np.any(raw_front_mask) else theta_deg
        geom_ranges = self._geometry_ranges_for_angles(geom_theta_deg)
        if geom_ranges.size == 0:
            geom_theta_deg = np.array([0.0], dtype=np.float32)
            geom_ranges = np.array([LIDAR_RANGE], dtype=np.float32)

        left_mask = (geom_theta_deg >= -OA_FRONT_HALF_ANGLE_DEG) & (geom_theta_deg < -OA_CENTER_HALF_ANGLE_DEG)
        center_mask = np.abs(geom_theta_deg) <= OA_CENTER_HALF_ANGLE_DEG
        right_mask = (geom_theta_deg > OA_CENTER_HALF_ANGLE_DEG) & (geom_theta_deg <= OA_FRONT_HALF_ANGLE_DEG)

        left_ranges = geom_ranges[left_mask] if np.any(left_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        center_ranges = geom_ranges[center_mask] if np.any(center_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        right_ranges = geom_ranges[right_mask] if np.any(right_mask) else np.array([LIDAR_RANGE], dtype=np.float32)

        return {
            "raw_front_clear": raw_front_clear,
            "geom_front_clear": float(np.percentile(geom_ranges, OA_FRONT_PERCENTILE)),
            "left_p10": float(np.percentile(left_ranges, OA_FRONT_PERCENTILE)),
            "center_p10": float(np.percentile(center_ranges, OA_FRONT_PERCENTILE)),
            "right_p10": float(np.percentile(right_ranges, OA_FRONT_PERCENTILE)),
        }

    def _lidar_sector_clear_with_memory(self, sector_ranges, prev_clear_m):
        sector_ranges = sector_ranges[np.isfinite(sector_ranges)]
        valid = sector_ranges[(sector_ranges >= LIDAR_MIN_RANGE) & (sector_ranges < LIDAR_RANGE)]

        if valid.size:
            instant_clear = float(np.percentile(valid, OA_FRONT_PERCENTILE))
        else:
            instant_clear = float(LIDAR_RANGE)

        smoothed_clear = float(min(instant_clear, prev_clear_m + SECTOR_CLEAR_RECOVERY_M))
        return instant_clear, smoothed_clear

    def _compute_lidar_sector_observation_features(self):
        theta_deg = self.lidar.angles.astype(np.float32)
        lidar_d = self.lidar.ranges.astype(np.float32)

        front_mask = np.abs(theta_deg) <= OA_FRONT_HALF_ANGLE_DEG
        front_theta = theta_deg[front_mask] if np.any(front_mask) else theta_deg
        front_ranges = lidar_d[front_mask] if np.any(front_mask) else lidar_d

        left_mask = (front_theta >= -OA_FRONT_HALF_ANGLE_DEG) & (front_theta < -OA_CENTER_HALF_ANGLE_DEG)
        center_mask = np.abs(front_theta) <= OA_CENTER_HALF_ANGLE_DEG
        right_mask = (front_theta > OA_CENTER_HALF_ANGLE_DEG) & (front_theta <= OA_FRONT_HALF_ANGLE_DEG)

        left_ranges = front_ranges[left_mask] if np.any(left_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        center_ranges = front_ranges[center_mask] if np.any(center_mask) else np.array([LIDAR_RANGE], dtype=np.float32)
        right_ranges = front_ranges[right_mask] if np.any(right_mask) else np.array([LIDAR_RANGE], dtype=np.float32)

        left_instant, left_smoothed = self._lidar_sector_clear_with_memory(left_ranges, self.left_clear_lidar_m)
        center_instant, center_smoothed = self._lidar_sector_clear_with_memory(center_ranges, self.center_clear_lidar_m)
        right_instant, right_smoothed = self._lidar_sector_clear_with_memory(right_ranges, self.right_clear_lidar_m)

        self.left_clear_lidar_m = left_smoothed
        self.center_clear_lidar_m = center_smoothed
        self.right_clear_lidar_m = right_smoothed

        return {
            "left_lidar_clear_m": left_smoothed,
            "center_lidar_clear_m": center_smoothed,
            "right_lidar_clear_m": right_smoothed,
            "left_lidar_instant_m": left_instant,
            "center_lidar_instant_m": center_instant,
            "right_lidar_instant_m": right_instant,
        }

    def _update_compact_obs_state(self, lidar_sector_features, rpm_norm):
        self.goal_dist_norm = float(np.clip(self.distance_to_goal / GOAL_DIST_MAX, 0.0, 1.0))
        self.left_clear_norm = float(np.clip(lidar_sector_features["left_lidar_clear_m"] / OA_WARN_CLEARANCE, 0.0, 1.0))
        self.center_clear_norm = float(np.clip(lidar_sector_features["center_lidar_clear_m"] / OA_WARN_CLEARANCE, 0.0, 1.0))
        self.right_clear_norm = float(np.clip(lidar_sector_features["right_lidar_clear_m"] / OA_WARN_CLEARANCE, 0.0, 1.0))
        self.left_clear_instant_norm = float(np.clip(lidar_sector_features["left_lidar_instant_m"] / OA_WARN_CLEARANCE, 0.0, 1.0))
        self.center_clear_instant_norm = float(np.clip(lidar_sector_features["center_lidar_instant_m"] / OA_WARN_CLEARANCE, 0.0, 1.0))
        self.right_clear_instant_norm = float(np.clip(lidar_sector_features["right_lidar_instant_m"] / OA_WARN_CLEARANCE, 0.0, 1.0))
        self.gap_asymmetry = float(np.clip((lidar_sector_features["right_lidar_clear_m"] - lidar_sector_features["left_lidar_clear_m"]) / OA_WARN_CLEARANCE, -1.0, 1.0))
        rudder_deg = float(np.degrees(self.model._delta))
        self.rudder_state_norm = float(np.clip(rudder_deg / MAX_RUD_ANGLE, -1.0, 1.0))
        self.rpm_state_norm = float(np.clip(rpm_norm, 0.0, 1.0))
    
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
            self.start_x = np.random.randint(2, self.map_width - 2)
            self.start_y = 2
            self.goal_x = np.random.randint(2, self.map_width - 2)
            self.goal_y = self.map_height - 5
            # self.start_x, self.start_y, self.goal_x, self.goal_y = self.scenario.position(test_case=1)
        else:
            self.start_x, self.start_y, self.goal_x, self.goal_y = self.scenario.position(test_case=self.test_case)

        self.asv_x = self.start_x
        self.asv_y = self.start_y

        self.prev_x = float(self.asv_x)
        self.prev_y = float(self.asv_y)
        self.speed_mps = 0.0
        self.left_clear_lidar_m = float(LIDAR_RANGE)
        self.center_clear_lidar_m = float(LIDAR_RANGE)
        self.right_clear_lidar_m = float(LIDAR_RANGE)

        # Generate the path
        self.path = self._generate_path(self.start_x, self.start_y, self.goal_x, self.goal_y)

        # Generate static obstacles
        self.num_obs = np.random.randint(0, self.max_obs + 1)
        self.obstacles = self._generate_obstacles(self.num_obs, self.test_case)

        # Initialize the ASV path list
        self.asv_path = [(self.asv_x, self.asv_y)]

        # Initialize distance to goal
        self.distance_to_goal = float(np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y]))
        self.angle_diff = float(self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y))
        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)
        lidar_sector_features = self._compute_lidar_sector_observation_features()
        self._update_compact_obs_state(lidar_sector_features, rpm_norm=0.0)

        self.reward = 0

        if self.render_mode in self.metadata['render_modes']:
            self.render()
        return self._get_obs(), {}

    # Configure terminal condition
    def check_done(self, position):
        # collide with an obstacle or out of bounds
        if self._check_collision_geom():
            return True
        
        # # lidar < 1m
        # lidar_list = self.lidar.ranges.astype(np.int64)
        # if np.any(lidar_list <= 1.0):
        #     return True
        
        # the agent reaches goal
        if self.distance_to_goal <= VESSEL_LENGTH:
            return True

        return False

    def step(self, action):
        self.elapsed_time += UPDATE_RATE
        rudder_cmd = float(np.clip(action[0], MIN_IN, MAX_IN))
        throttle_cmd = float(np.clip(action[1], MIN_IN, MAX_IN))

        # Map rudder_cmd [-1,1] -> rudder [-25, 25]
        rudder = rudder_cmd * 100

        # Map throttle_cmd [-1,1] -> rpm [RPM_MIN, RPM_MAX]
        rpm = (throttle_cmd - MIN_IN) * ((RPM_MAX - RPM_MIN)/(MAX_IN - MIN_IN)) + RPM_MIN

        # Store current position
        x_prev = float(self.asv_x)
        y_prev = float(self.asv_y)

        dx,dy,h,w = self.model.update(rpm, rudder, UPDATE_RATE)
        self.asv_x += dx
        self.asv_y += dy
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
        self.tgt = self._signed_cross_track_error(self.asv_x, self.asv_y)

        # extract (x,y) target
        closest_idx = np.argmin(distance)
        self.tgt_x, self.tgt_y = self.path[closest_idx]

        self.lidar.scan((self.asv_x, self.asv_y), self.asv_h, obstacles=self.obstacles, map_border=self.map_border)
        
        if self.render_mode in self.metadata['render_modes']:
            self.render()
        
        # append new coordinate of asv
        self.asv_path.append((self.asv_x, self.asv_y))

        # update distance to goal
        self.distance_to_goal = np.linalg.norm([self.asv_x - self.goal_x, self.asv_y - self.goal_y])
        self.angle_diff = self._calculate_angle(self.asv_x, self.asv_y, self.asv_h, self.goal_x, self.goal_y)

        # Define terminal flags
        collided = bool(self._check_collision_geom())
        reached_goal = bool(self.distance_to_goal <= VESSEL_LENGTH)

        # lam = LAMBDA_REWARD

        # cross-track error
        ye = abs(self.tgt)

        # pose-base speed
        U = self.speed_mps
        U_norm = U / U_MAX
        rpm_norm = float(rpm / RPM_MAX) if RPM_MAX > 0 else 0.0

        # Course error relative to the path direction (better than using heading-only)
        if U > 1e-6:
            course_deg = np.degrees(np.arctan2(dx_pos, dy_pos))
        else:
            course_deg = self.asv_h

        path_dx = float(self.goal_x - self.start_x)
        path_dy = float(self.goal_y - self.start_y)
        path_course_deg = float(np.degrees(np.arctan2(path_dx, path_dy)))

        chi_tilde_deg = (course_deg - path_course_deg + 180) % 360 - 180
        chi_tilde = float(np.radians(chi_tilde_deg))
        cos_chi = float(np.cos(chi_tilde))

        r_pf = float(-1 + (U_norm * cos_chi + 1) * (np.exp(-GAMMA_E * ye) + 1))

        clearance_features = self._compute_clearance_features()
        lidar_sector_features = self._compute_lidar_sector_observation_features()
        raw_front_clear = clearance_features["raw_front_clear"]
        geom_front_clear = clearance_features["geom_front_clear"]
        left_p10 = clearance_features["left_p10"]
        center_p10 = clearance_features["center_p10"]
        right_p10 = clearance_features["right_p10"]

        center_norm = (OA_WARN_CLEARANCE - center_p10) / max(OA_WARN_CLEARANCE - OA_CRIT_CLEARANCE, 1e-6)
        center_norm = float(np.clip(center_norm, 0.0, 1.0))
        r_center = -OA_CENTER_GAIN * (center_norm ** 2)

        gap_bias = float(np.tanh((right_p10 - left_p10) / OA_DIR_SCALE))
        # ShipModel turns right for negative rudder and left for positive rudder.
        turn_cmd = float(-rudder_cmd)
        rudder_align = float(np.tanh(2.0 * turn_cmd))
        r_dir = OA_DIR_GAIN * center_norm * gap_bias * rudder_align

        if center_p10 < OA_NEAR_CLEARANCE:
            near_norm = float(np.clip((OA_NEAR_CLEARANCE - center_p10) / OA_NEAR_CLEARANCE, 0.0, 1.0))
            r_near = -OA_NEAR_GAIN * (near_norm ** 2)
        else:
            near_norm = 0.0
            r_near = 0.0

        threat = max(center_norm, near_norm)
        r_speed_threat = -OA_SPEED_GAIN * threat * (U_norm ** 2)
        r_oa = r_center + r_dir + r_near + r_speed_threat
        lam = LAMBDA_CLEAR - (LAMBDA_CLEAR - LAMBDA_THREAT) * threat
        self._update_compact_obs_state(lidar_sector_features, rpm_norm)

        r_exist = -ALPHA_R

        if collided:
            reward = float(R_COLLISION)
        elif reached_goal:
            reward = float(R_GOAL)
        else:
            reward = float(lam * r_pf + (1.0 - lam) * r_oa + r_exist)

        terminated = self.check_done((self.asv_x, self.asv_y))

        info = {
            "reward_pf_contrib": float(lam * r_pf),
            "reward_oa_contrib": float((1.0 - lam) * r_oa),
            "x": float(self.asv_x),
            "y": float(self.asv_y),
            "heading_deg": float(self.asv_h),
            "dhdg_raw": float(self.asv_w),

            "front_clear": float(geom_front_clear),
            "oa_active": bool(center_norm > 0),
            "left_p10": float(left_p10),
            "center_p10": float(center_p10),
            "right_p10": float(right_p10),
            "raw_front_clear": float(raw_front_clear),
            "geom_front_clear": float(geom_front_clear),
            "center_norm": float(center_norm),
            "gap_bias": float(gap_bias),
            "rudder_align": float(rudder_align),
            "near_norm": float(near_norm),
            "r_center": float(r_center),
            "r_dir": float(r_dir),
            "r_near": float(r_near),
            "r_speed_threat": float(r_speed_threat),
            "r_pf": float(r_pf),
            "r_oa": float(r_oa),
            "threat": float(threat),
            "lam": float(lam),
            "goal_dist_norm": float(self.goal_dist_norm),
            "left_clear": float(self.left_clear_norm),
            "center_clear": float(self.center_clear_norm),
            "right_clear": float(self.right_clear_norm),
            "left_clear_instant": float(self.left_clear_instant_norm),
            "center_clear_instant": float(self.center_clear_instant_norm),
            "right_clear_instant": float(self.right_clear_instant_norm),
            "gap_asymmetry": float(self.gap_asymmetry),
            "lidar_left_clear_m": float(lidar_sector_features["left_lidar_clear_m"]),
            "lidar_center_clear_m": float(lidar_sector_features["center_lidar_clear_m"]),
            "lidar_right_clear_m": float(lidar_sector_features["right_lidar_clear_m"]),
            "lidar_left_instant_m": float(lidar_sector_features["left_lidar_instant_m"]),
            "lidar_center_instant_m": float(lidar_sector_features["center_lidar_instant_m"]),
            "lidar_right_instant_m": float(lidar_sector_features["right_lidar_instant_m"]),
            "rudder_state": float(self.rudder_state_norm),
            "rpm_state": float(self.rpm_state_norm),
        }

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode != 'human':
            return        
        if self.display is None:
            self.display = pygame.display.set_mode(self.window_size)
        
        scale = float(self.render_scale)

        def scale_point(xy):    # scale point from world -> pixel
            x, y = float(xy[0]), float(xy[1])
            px = int(round(x * scale))
            py = int(round((self.map_height - y) * scale))
            py = max(0, min(self.window_size[1] - 1, py))   # clamp border
            return (px,py)

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
        self.lidar.render(self.surface, scale_point)

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
        action = [-1, 1]
        obs,rew,term,_,_ = env.step(action)
        print(action)
        total_reward += rew
        # print(f"Action: {action}    Reward: {rew}")
        if term:
            print(f"Elapsed time: {env.elapsed_time}, Reward: {total_reward:0.2f}")         
            pygame.display.quit()
            pygame.quit()
            exit()
