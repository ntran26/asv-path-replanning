import pygame
import numpy as np

LIDAR_RANGE = 150
LIDAR_SWATH = 90
LIDAR_BEAMS = 21

class Lidar:
    """Basic LIDAR simulator using pygame rects to determine sensor ranges."""
    def __init__(self):
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0
        self._angles = None
        self._ranges = None
        self.reset()

    def reset(self):
        """Reset LIDAR to initial state."""
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0
        # EDIT: Changed dtype to float for more precision and set up beam angles evenly across the swath.
        self._angles = np.linspace(-LIDAR_SWATH/2, LIDAR_SWATH/2, LIDAR_BEAMS, dtype=np.float64)
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE

    @property
    def angles(self):
        """Return a copy of sensor angles."""
        return self._angles.copy()
    
    @property
    def ranges(self):
        """Return a copy of sensor range readings."""
        return self._ranges.copy()
    
    def scan(self, pos, hdg, obstacles=None) -> np.ndarray:
        """
        Perform a LIDAR scan.

        Args:
            pos (tuple): (x, y) position of the sensor.
            hdg (float): heading of sensor in degrees.
            obstacles (list): list of pygame.Rect obstacles.
        Returns:
            numpy.ndarray: array of ranges from sensor to obstacles.
                If no obstacle is detected, the range remains LIDAR_RANGE.
        """
        self._pos_x = pos[0]
        self._pos_y = pos[1]
        self._hdg = hdg
        
        # ADD: Loop over each beam angle to compute collision distances.
        for idx, angle in enumerate(self._angles):
            # Calculate the absolute angle (sensor heading + beam angle) in radians.
            absolute_angle = np.radians(self._hdg + angle)
            # Compute the endpoint of the beam at maximum range (if no obstacle is hit).
            # NOTE: Using sin for x and cos for y, with y subtracted to account for pygame's coordinate system.
            end_x = self._pos_x + LIDAR_RANGE * np.sin(absolute_angle)
            end_y = self._pos_y - LIDAR_RANGE * np.cos(absolute_angle)
            # Define the ray as a line tuple: (start_x, start_y, end_x, end_y)
            ray_line = (self._pos_x, self._pos_y, end_x, end_y)
            
            # Initialize the closest distance to the maximum range.
            closest_distance = LIDAR_RANGE
            
            # Check for collision with each obstacle.
            if obstacles:
                for obs in obstacles:
                    collision = obs.clipline(ray_line)
                    if collision:
                        # returns two points that define the intersecting segment
                        # compute the distance from the sensor to both points and take the smaller one
                        p1 = collision[0]
                        p2 = collision[1]
                        d1 = np.hypot(p1[0] - self._pos_x, p1[1] - self._pos_y)
                        d2 = np.hypot(p2[0] - self._pos_x, p2[1] - self._pos_y)
                        collision_distance = min(d1, d2)
                        if collision_distance < closest_distance:
                            closest_distance = collision_distance
            # Update the range reading for this beam.
            self._ranges[idx] = closest_distance
        
        return self._ranges.copy()
        
    def render(self, surface: pygame.Surface):
        """
        Render the LIDAR beams as lines on the given surface.

        Args:
            surface (pygame.Surface): The surface on which to render the beams.
        """
        for idx, angle in enumerate(self._angles):
            # Calculate the absolute angle in radians.
            absolute_angle = np.radians(self._hdg + angle)
            # Compute the endpoint for the current beam using its range reading.
            x = self._pos_x + self._ranges[idx] * np.sin(absolute_angle)
            y = self._pos_y - self._ranges[idx] * np.cos(absolute_angle)
            pygame.draw.aaline(surface, (90, 90, 200), (self._pos_x, self._pos_y), (x, y))
