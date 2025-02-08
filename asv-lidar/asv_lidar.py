# asv_lidar.py
import pygame
import numpy as np

LIDAR_RANGE = 150
LIDAR_SWATH = 90
LIDAR_BEAMS = 21

class Lidar:
    """Basic LIDAR simulator that provides an array of distance measurements."""
    def __init__(self):
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0  # heading in degrees
        self._angles = np.linspace(-LIDAR_SWATH / 2, LIDAR_SWATH / 2, LIDAR_BEAMS, dtype=np.int16)
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE

    def reset(self):
        """Reset LIDAR measurements."""
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE

    @property
    def angles(self):
        return self._angles.copy()
    
    @property
    def ranges(self):
        return self._ranges.copy()
    
    def scan(self, pos, hdg, obstacles=None) -> np.ndarray:
        """
        Simulate a LIDAR scan.
        For now, without obstacles, simply return the max range for all beams.
        (Later you can add obstacle intersection logic.)
        """
        self._pos_x, self._pos_y = pos
        self._hdg = hdg
        # In a full implementation, you would check for intersections between each beam and obstacles.
        # Here we simply return the maximum range.
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE
        return self._ranges.copy()
        
    def render(self, surface: pygame.Surface):
        """Render the LIDAR beams as lines on the given surface."""
        for idx, a in enumerate(self._angles):
            total_angle = self._hdg + a
            r = np.radians(total_angle)
            x = self._pos_x + self._ranges[idx] * np.sin(r)
            y = self._pos_y - self._ranges[idx] * np.cos(r)
            pygame.draw.aaline(surface, (90, 90, 200), (self._pos_x, self._pos_y), (x, y))
