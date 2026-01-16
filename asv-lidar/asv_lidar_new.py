import pygame
import numpy as np

LIDAR_RANGE = 150
LIDAR_SWATH = 180
LIDAR_BEAMS = 90
LIDAR_PARTITION = 15
BEAM_PER_SECTOR = LIDAR_BEAMS // LIDAR_PARTITION
VESSEL_WIDTH = 1

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
        # self._angles = np.linspace(-LIDAR_SWATH/2, LIDAR_SWATH/2, LIDAR_BEAMS, dtype=np.float64)
        self._angles = np.linspace(0, 360-(LIDAR_SWATH/LIDAR_BEAMS), LIDAR_BEAMS, dtype=np.float64)
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE

    @property
    def angles(self):
        """Return a copy of sensor angles."""
        return self._angles.copy()
    
    @property
    def ranges(self):
        """Return a copy of sensor range readings."""
        return self._ranges.copy()
    
    def pooling(self, sector_ranges, vessel_width, sector_angle_span):
        theta = sector_angle_span / (len(sector_ranges) - 1)
        I = np.argsort(sector_ranges)   # sort indices from smallest to largest
        for idx in I:
            xi = sector_ranges[idx]
            arc_length = theta * xi
            opening_width = arc_length / 2
            found = False

            for j in range(len(sector_ranges)):
                if sector_ranges[j] > xi:
                    opening_width += arc_length
                    if opening_width > vessel_width:
                        found = True
                        break
                else:
                    opening_width += arc_length / 2
                    if opening_width > vessel_width:
                        found = True
                        break
            if not found:
                return xi
        return max(sector_ranges)
    
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
        
        # Loop over each beam angle to compute collision distances
        for idx, angle in enumerate(self._angles):
            # Calculate the absolute angle (sensor heading + beam angle) in radians
            absolute_angle = np.radians(self._hdg + angle)
            # Compute the endpoint of the beam at maximum range (if no obstacle)
            end_x = self._pos_x + LIDAR_RANGE * np.sin(absolute_angle)
            end_y = self._pos_y - LIDAR_RANGE * np.cos(absolute_angle)
            
            # Initialize the closest distance to the maximum range
            closest_distance = LIDAR_RANGE

            edges = []

            if obstacles:
                for obs in obstacles:
                    for i in range(len(obs)):
                        # v1 = obs[i]
                        # v2 = obs[(i + 1) % len(obs)]
                        # edges.append((v1, v2))
                        edges.append((obs[i], obs[(i + 1) % len(obs)]))

            if map_border:
                for border in map_border:
                    for i in range(len(border)):
                        # v1 = border[i]
                        # v2 = border[(i + 1) % len(border)]
                        # edges.append((v1, v2))
                        edges.append((border[i], border[(i + 1) % len(border)]))

            for e in edges:
                intersect = self.line_intersection((self._pos_x, self._pos_y), (end_x, end_y), e[0], e[1])
                if intersect:
                    d = np.hypot(intersect[0] - self._pos_x, intersect[1] - self._pos_y)
                    closest = min(closest, d)

            self._ranges[idx] = closest

        # Sector pooling
        sector_size = BEAM_PER_SECTOR
        sector_angle = np.radians(LIDAR_SWATH / LIDAR_PARTITION)

        self._sector_ranges = np.zeros(LIDAR_PARTITION, dtype=np.float32)
        self._sector_feasible = np.zeros(LIDAR_PARTITION, dtype=bool)

        # Threshold for visualization 
        min_clearance = getattr(self, "min_clearance", 10.0)

        for i in range(LIDAR_PARTITION):
            start = i * sector_size
            end = (i + 1) * sector_size
            sector_ranges = self._ranges[start:end]

            pooled = self.pooling(sector_ranges, VESSEL_WIDTH, sector_angle)
            self._sector_ranges[i] = pooled

            # For rendering: "blocked" if pooled distance is very close.
            self._sector_feasible[i] = pooled > min_clearance

        return self._sector_ranges.copy()

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
        Render:
        1) Raw beams faintly (optional)
        2) Sector pooled output as thick mid-ray + arc per sector
        """
        origin = (int(self._pos_x), int(self._pos_y))

        # Draw onto an overlay for transparency 
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        # --------------------
        # 1) Raw beams (faint)
        # --------------------
        raw_col = (90, 90, 200, 50)  # RGBA with alpha
        for idx, angle in enumerate(self._angles):
            absolute_angle = np.radians(self._hdg + angle)
            x = self._pos_x + self._ranges[idx] * np.sin(absolute_angle)
            y = self._pos_y - self._ranges[idx] * np.cos(absolute_angle)
            pygame.draw.aaline(overlay, raw_col, origin, (int(x), int(y)))

        # ----------------------------------------
        # 2) Sector pooled results (thick + arc)
        # ----------------------------------------
        if hasattr(self, "_sector_ranges") and self._sector_ranges is not None:
            sector_size = BEAM_PER_SECTOR

            free_col = (90, 90, 200, 180)     # bluish
            blocked_col = (255, 60, 60, 200)  # reddish
            boundary_col = (200, 200, 200, 60)

            for i in range(LIDAR_PARTITION):
                start_idx = i * sector_size
                end_idx = min((i + 1) * sector_size - 1, len(self._angles) - 1)

                a0 = np.radians(self._hdg + self._angles[start_idx])
                a1 = np.radians(self._hdg + self._angles[end_idx])
                r = float(self._sector_ranges[i])

                feasible = True
                if hasattr(self, "_sector_feasible") and self._sector_feasible is not None:
                    feasible = bool(self._sector_feasible[i])

                col = free_col if feasible else blocked_col

                # Draw sector boundary rays out to max range 
                b0 = (int(self._pos_x + LIDAR_RANGE * np.sin(a0)),
                    int(self._pos_y - LIDAR_RANGE * np.cos(a0)))
                b1 = (int(self._pos_x + LIDAR_RANGE * np.sin(a1)),
                    int(self._pos_y - LIDAR_RANGE * np.cos(a1)))
                pygame.draw.aaline(overlay, boundary_col, origin, b0)
                pygame.draw.aaline(overlay, boundary_col, origin, b1)

                # Draw the pooled arc at radius r 
                pts = []
                for t in np.linspace(a0, a1, 10):
                    px = self._pos_x + r * np.sin(t)
                    py = self._pos_y - r * np.cos(t)
                    pts.append((int(px), int(py)))
                if len(pts) >= 2:
                    pygame.draw.aalines(overlay, col, False, pts)

                # Draw thick mid-ray at pooled distance
                amid = 0.5 * (a0 + a1)
                mx = self._pos_x + r * np.sin(amid)
                my = self._pos_y - r * np.cos(amid)
                pygame.draw.line(overlay, col, origin, (int(mx), int(my)), 4)

        # Composite overlay onto the main surface
        surface.blit(overlay, (0, 0))
