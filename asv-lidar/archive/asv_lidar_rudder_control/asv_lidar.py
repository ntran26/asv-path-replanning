import pygame
import numpy as np

LIDAR_RANGE = 150
LIDAR_SWATH = 270
LIDAR_BEAMS = 63

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
        # EDIT: Changed dtype to float for more precision and set up beam angles evenly across the swath
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
    
    def scan(self, pos, hdg, obstacles=None, map_border=None) -> np.ndarray:
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
        # self._pos_x = pos[0]
        # self._pos_y = pos[1]
        self._hdg = hdg

        # Set the lidar (x,y) to be in front of the asv
        lidar_offset = 30
        self._pos_x = pos[0] + lidar_offset * np.sin(np.radians(self._hdg))
        self._pos_y = pos[1] - lidar_offset * np.cos(np.radians(self._hdg))
        
        # ADD: Loop over each beam angle to compute collision distances
        for idx, angle in enumerate(self._angles):
            # Calculate the absolute angle (sensor heading + beam angle) in radians
            absolute_angle = np.radians(self._hdg + angle)
            # Compute the endpoint of the beam at maximum range (if no obstacle)
            end_x = self._pos_x + LIDAR_RANGE * np.sin(absolute_angle)
            end_y = self._pos_y - LIDAR_RANGE * np.cos(absolute_angle)
            
            # Initialize the closest distance to the maximum range
            closest_distance = LIDAR_RANGE

            obstacle_edges = []

            # Define the ray as a line tuple: (start_x, start_y, end_x, end_y)
            # ray_line = (self._pos_x, self._pos_y, end_x, end_y)
            
            # # Check for collision with each obstacle
            # if obstacles:
            #     for obs in obstacles:
            #         collision = obs.clipline(ray_line)
            #         if collision:
            #             # returns two points that define the intersecting segment
            #             # compute the distance from the sensor to both points and take the smaller one
            #             p1 = collision[0]
            #             p2 = collision[1]
            #             d1 = np.hypot(p1[0] - self._pos_x, p1[1] - self._pos_y)
            #             d2 = np.hypot(p2[0] - self._pos_x, p2[1] - self._pos_y)
            #             collision_distance = min(d1, d2)
            #             if collision_distance < closest_distance:
            #                 closest_distance = collision_distance

            if obstacles:
                for obs in obstacles:
                    for i in range(len(obs)):
                        v1 = obs[i]
                        v2 = obs[(i + 1) % len(obs)]
                        obstacle_edges.append((v1, v2))

            if map_border:
                for border in map_border:
                    for i in range(len(border)):
                        v1 = border[i]
                        v2 = border[(i + 1) % len(border)]
                        obstacle_edges.append((v1, v2))

            for edge in obstacle_edges:
                intersection = self.line_intersection((self._pos_x, self._pos_y), (end_x, end_y), edge[0], edge[1])
                if intersection:
                    dist = np.hypot(intersection[0] - self._pos_x, intersection[1] - self._pos_y)
                    closest_distance = min(closest_distance, dist)

            # Update the range reading for this beam.
            self._ranges[idx] = closest_distance
        
        return self._ranges.copy()

    def line_intersection(self, a1, a2, b1, b2):
        """
        Compute the intersection between 2 line segments
        Returns the intersection points (x, y) and None if there's no intersection
        """
        def cross_product(a, b):
            return a[0] * b[1] - a[1] * b[0]
        
        # vectors from p1 to p2, q1 to q2
        a = (a2[0] - a1[0], a2[1] - a1[1])
        b = (b2[0] - b1[0], b2[1] - b1[1])

        # cross product between 2 vectors
        a_cross_b = cross_product(a, b)

        # vector from a1 to b1
        a_b = (b1[0] - a1[0], b1[1] - a1[1])
        a_b_cross_a = cross_product(a_b, a)

        # check if the lines are parallel or collinear
        if a_cross_b == 0 and a_b_cross_a == 0:     # collinear
            return None
        if a_cross_b == 0:      # parallel
            return None

        # compute scaler_a, where intersection occurs along vector a
        scalar_a = cross_product(a_b, b) / a_cross_b
        # compute scalar_b, where intersection occurs along vector b
        scalar_b = a_b_cross_a / a_cross_b

        # check for intersection
        if 0 <= scalar_a <= 1 and 0 <= scalar_b <= 1:
            intersection_x = a1[0] + scalar_a * a[0]
            intersection_y = a1[1] + scalar_a * a[1]
            return (intersection_x, intersection_y)

        return None
        
    def render(self, surface: pygame.Surface):
        """
        Render the LIDAR beams as lines on the given surface

        Args:
            surface (pygame.Surface): The surface on which to render the beams
        """
        for idx, angle in enumerate(self._angles):
            # Calculate the absolute angle in radians.
            absolute_angle = np.radians(self._hdg + angle)
            # Compute the endpoint for the current beam using its range reading.
            x = self._pos_x + self._ranges[idx] * np.sin(absolute_angle)
            y = self._pos_y - self._ranges[idx] * np.cos(absolute_angle)
            pygame.draw.aaline(surface, (90, 90, 200), (self._pos_x, self._pos_y), (x, y))
