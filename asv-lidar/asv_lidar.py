import pygame
import numpy as np

LIDAR_RANGE = 150       # Range of lidar beams
LIDAR_SWATH = 90        # Angle of lidar view
LIDAR_BEAMS = 21        # Total number of lidar beams

class Lidar:
    """ Basic LIDAR simulator.

        Utilises pygame rects to determine array of ranges
    """
    def __init__(self):
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0
        self._angles = None
        self._ranges = None
        self.reset()


    def reset(self):
        """ Reset LIDAR 
        """
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0
        self._angles = np.linspace(-LIDAR_SWATH/2,LIDAR_SWATH/2,LIDAR_BEAMS,dtype=np.int16)
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE


    @property
    def angles(self):
        """ array of sensor angles """
        return self._angles.copy()
    
    @property
    def ranges(self):
        """ array of ranges from most recent scan """
        return self._ranges.copy()
    
    
    def scan(self, pos, hdg, obstacles=None) -> np.ndarray:
        """ Perform a LIDAR scan

            Args:
                pos (tuple): x,y position of sensor
                hdg (float): orientation of sensor in degrees
                obstacles (list): list of obstacle rects
            Returns:
                array of ranges from sensor to obstacles. 
                If no obstacle reads LIDAR_RANGE
        """
        self._pos_x = pos[0]
        self._pos_y = pos[1]
        self._hdg = hdg

        # Update self.ranges
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE     # set to max range

        if obstacles is None:
            return self._ranges.copy()
        
        for idx, angle in enumerate(self._angles):
            ray_angle = np.radians(self._hdg + angle)
            end_x = self._pos_x + LIDAR_RANGE * np.sin(ray_angle)
            end_y = self._pos_x - LIDAR_RANGE * np.cos(ray_angle)

            min_distance = LIDAR_RANGE

            for obs in obstacles:
                if isinstance(obs, pygame.Rect):
                    line_start = (self._pos_x, self._pos_y)
                    line_end = (end_x, end_y)

                    # Check for intersection with obstacle
                    clipped_line = obs.clipline(line_start, line_end)
                    if clipped_line:
                        for point in clipped_line:
                            dist = np.hypot(point[0] - self._pos_x, point[1] - self._pos_y)
                            min_distance = min(min_distance, dist)

            self._ranges[idx] = min_distance

        return self._ranges.copy()
        
        
    def render(self, surface:pygame.Surface):
        """ Render the LIDAR as a series of lines

            Args:
                surface (pygame.Surface): surface to render to
        """
        for idx,a in enumerate(self._angles):
            r = np.radians(self._hdg + a)
            x = self._pos_x + self._ranges[idx] * np.sin(r)
            y = self._pos_y - self._ranges[idx] * np.cos(r)
            pygame.draw.aaline(surface,(90,90,200),(self._pos_x,self._pos_y),(x,y))