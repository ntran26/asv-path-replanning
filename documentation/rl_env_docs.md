# RL Gym Environment Simulation
### Overview
The simulation consists of 4 core scripts:
1. **ship_model.py** describes vessel dynamics
2. **asv_lidar.py** simulates a 2D LiDAR system
3. **asv_lidar_rudder_speed_control.py** contains the simulated gym environment
4. **train_test_asv.py** is responsible for training and testing/evaluating the RL agent
(Optional) **test_run.py** contains the preset scenarios to test the performance
(Optional) **plot_data.py** constructs a plot of a completed run

```text
train_test_asv.py 
   │   (Stable-Baselines3 PPO/SAC)
   │   obs_t ───────────────▶ policy πθ ────────▶ action_t
   │                               ▲                   |
   │                               |                   ▼
   └────────────── call ──────── ASVLidarEnv.step(action_t) ───────┐
                                                                   |
                               ┌───────────────────────────────────|
                               |                                   |
                               ▼                                   ▼
                        ShipModel.update                      Lidar.scan
                        (rpm, δ, Δt)                    (pose, obstacles, border)                         
                               │                                   │
                               └────────── update ─────────────────┘
                                             │
                                             ▼
                                      obs_{t+1}, reward_t, done_t
```

### Coordinate Conventions
The simulation uses pygame-style 2D world coordinates:
- Position: x to the right, y downward
- Heading (degrees):
    - $\psi$ = 0° points up (negative y direction)
    - $\psi$ = 90° points right (positive x direction)

This convention affects:
- LiDAR ray endpoint equations (note the **-cos()** on **y**)
- How **dy** from **ShipModel** is applied: **y ← y - dy**
- Heading-to-goal bearing computation: **atan2(dx, -dy)**

### Glossary
Control/Action
- $\boldsymbol{a_t}$ ∈ [-1,1] : normalized rudder command (policy output)
- $\boldsymbol{a_t}$ ∈ [-1,1] : normalized throttle command (policy output)
- $\boldsymbol{\delta}$: rudder angle command (deg or rad)
- **RPM**: propeller speed command

Time and pose
- $\boldsymbol{\Delta t}$: simulation step (s) (UPDATE_RATE)
- **(x,y)**: vessel position in world coordinate
- $\boldsymbol{\omega}$: yaw rate (deg/s or rad/s)

Ship dynamics
- **v**: forward speed (m/s)
- **a**: forward acceleration (m/s^2)
- $\boldsymbol\alpha$: yaw acceleration (rad/s^2)
- **T**: thrust force
- **F**: net forward force
- **M**: yaw moment
- **m**: mass
- **I**: moment of inertia
- $\boldsymbol{k_T}$: thrust coefficient
- $\boldsymbol{k_D}$: drag coefficient

LiDAR
- **N**: number of beams
- **S**: lidar swath (deg)
- **R**: max scanning range (m)
- $\boldsymbol{\theta_i}$: relative beam angle (deg)
- $\boldsymbol\phi_i=\psi+\theta_i$: absolute beam direction
- **n**: measured range of beam i (m)

Navigation/Reward
- **tgt**: path tracking error (min distance to reference path)
- **e** $\boldsymbol\psi$: heading error to goal, wrapped to [-180,180]
- $\boldsymbol{r_{pf}, r_{oa}, r_{head}, r_{exist}, r_{goal}}$: reward components
- $\boldsymbol\lambda$: weighting factor between path following/obstacle avoidance

## 1. Ship Dynamic Model

### File: ship_model.py

This script provides a simplified planar vessel dynamics model 


## 2. Simulate LiDAR

### File: asv_lidar.py

This code implements a 2D LiDAR simulator for the ASV gym environment. It casts LiDAR beams at a set of angles and measures distance to obstacles and map borders. For this simulated LiDAR to work, the coordinates of obstacles and map boundaries are known and fed as inputs.

### Workflow:

Input: Lidar.scan = ( pos(x,y), $\boldsymbol\psi$, obstacle, map_border ) <br>
Output: Lidar.ranges = ranges[i]  of length N
1. Offset sensor origin to the front of vessel by L/2
2. Convert all polygons (obstacles + border) to a list of edge segments
3. For i = (1, N) beams:
    - $\phi_i=\psi+\theta_i$
    - Compute endpoint: <br>
        $x_{end}=x_s + R \cdot sin(\phi_i)$ <br>
        $y_{end}=y_s - R \cdot cos(\phi_i)$
    - closest = R
    - For each segment:
        - if intersection: <br>
            d = distance ( intersection, (x<sub>s</sub> , y<sub>s</sub>) ) <br>
            closest = min (closest , d)
    - ranges[i] = closest
4. return ranges[i]

### Imports and LiDAR constants:

```python
import pygame
import numpy as np
from ship_model import VESSEL_LENGTH

LIDAR_RANGE = 16
LIDAR_SWATH = 270
LIDAR_BEAMS = 90
```
Imports
- **pygame**: for rendering and geometry helper functions
- **numpy**: processing angle grids and vector math

Constants
- **VESSEL_LENGTH**: length of vessel from **ship_model.py** (m)
- **LIDAR_RANGE**: maximum range (m)
- **LIDAR_SWATH**: angular field of view (degrees)
- **LIDAR_BEAMS**: number of LiDAR beams

### Lidar class
```python
class Lidar:
    def __init__(self):
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0
        self._angles = None
        self._ranges = None
        self.reset()

    def reset(self):
        self._pos_x = 0
        self._pos_y = 0
        self._hdg = 0
        self._angles = np.linspace(-LIDAR_SWATH/2, LIDAR_SWATH/2, LIDAR_BEAMS, dtype=np.float64)
        self._ranges = np.ones_like(self._angles) * LIDAR_RANGE
```
- Initialize lidar class: (x, y), heading
- **reset()** initializes lidar **angles** and **ranges**

```python
@property
def angles(self):
    return self._angles.copy()

@property
def ranges(self):
    return self._ranges.copy()
```
- Make a copy of lidar **angles** and **ranges** for outputs.

### Main Function
```python
def scan(self, pos, hdg, obstacles=None, map_border=None) -> np.ndarray:
    self._hdg = hdg
    lidar_offset = VESSEL_LENGTH/2
    self._pos_x = pos[0] + lidar_offset * np.sin(np.radians(self._hdg))
    self._pos_y = pos[1] - lidar_offset * np.cos(np.radians(self._hdg))
```
- Initialize lidar scan function
- Pass current heading angle
- Offset lidar position to the front <br>
$x_s = x + (L/2)sin(\psi)$ <br> $y_s = y - (L/2)cos(\psi)$ <br>

```python
    for idx, angle in enumerate(self._angles):
        absolute_angle = np.radians(self._hdg + angle)
        end_x = self._pos_x + LIDAR_RANGE * np.sin(absolute_angle)
        end_y = self._pos_y - LIDAR_RANGE * np.cos(absolute_angle)
        closest_distance = LIDAR_RANGE

        obstacle_edges = []
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

        self._ranges[idx] = closest_distance
    
    return self._ranges.copy()
```
- Iterate through indices in **self._angles** and find **absolute_angle** with respect to the current heading
- Initialize **closest_distance** as maximum LiDAR range (no obstacles)
- Obstacles are passed in as polygons: list of vertex tuples
- Convert polygons to a flat list of edges
- Map boundaries are also treated as obstacles
- Append to **obstacle_edges** and run through **line_intersection** function to check for intersection
- If intersection occurs, compute distance and save the **closest_distance**, store the result for each beam index
- Return the LiDAR measurement array **self._ranges**

### Obstacle Detection
- Input: segment A (a1, a2) and segment B (b1, b2)
- Output: if intersect, return intersection point
```python
def line_intersection(self, a1, a2, b1, b2):
    def cross_product(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    a = (a2[0] - a1[0], a2[1] - a1[1])
    b = (b2[0] - b1[0], b2[1] - b1[1])

    a_cross_b = cross_product(a, b)

    a_b = (b1[0] - a1[0], b1[1] - a1[1])
    a_b_cross_a = cross_product(a_b, a)

    if a_cross_b == 0 and a_b_cross_a == 0:     
        return None
    if a_cross_b == 0:      
        return None

    scalar_a = cross_product(a_b, b) / a_cross_b
    scalar_b = a_b_cross_a / a_cross_b

    if 0 <= scalar_a <= 1 and 0 <= scalar_b <= 1:
        intersection_x = a1[0] + scalar_a * a[0]
        intersection_y = a1[1] + scalar_a * a[1]
        return (intersection_x, intersection_y)

    return None
```
- Each point has a coordinate on the 2D plane
```
(a1[0], a1[1]): LiDAR origin
(a2[0], a2[1]): beam endpoint
(b1[0], b1[1]): obstacle vertex
(b2[0], b2[1]): obstacle vertex
```
- The **cross_product()** function computes the 2D vector cross product:
$$ a \times b = a_x b_y - a_y b_x $$
- If $a\times b$ > 0 or < 0, b is on the left or right of a
- If $a\times b$ = 0, a and b vectors are parallel
- First, the segments are converted into vectors **a** and **b** that represents the *direction* vectors
$$ \boldsymbol{a} = a_2 - a_1 $$
$$ \boldsymbol{b} = b_2 - b_1 $$
- **a_cross_b** checks if the lines are parallel
- **a_b** is the vector between the 2 segment origins: $ b_1-a_1 $
```
                   a1 -------- a2
                    \
                     \ a_b
                      \
                       b1 -------- b2
```
- Check the cross product between **a_b** and **a**, stored as **a_b_cross_a**
- Collinearity check: if both **a_cross_b** and **a_b_cross_a** = 0, the segments are collinear (lie on top of each other)
- Parallel check: if **a_cross_b** = 0, the segments are parallel
- Solve parametric intersection:
    - The intersection point is represented as:
$$P = a_1 + ta$$ $$Q=b_1+ub$$
    - Intersection occurs when $$ a_1 + ta = b_1 + ub $$
    - where $t$ = scalar_a, $u$ = scalar_b
    - Calculate **scalar_a** and **scalar_b**
    $$ t = \frac{ab \times b}{a \times b}  $$
    $$ u = \frac{ab \times a}{a \times b} $$
- If **scalar_a** and **scalar_b** in range of [0, 1], intersection occurs inside the segment
- Compute the intersection point $P$ (**intersection_x**) and $Q$ (**intersection_y**)
- Return the coordinates of intersection point as output if the segments intersect
```
function line_intersection(a1,a2,b1,b2):

    a = a2 - a1
    b = b2 - b1

    cross_ab = cross(a,b)

    if cross_ab == 0
        return None

    t = cross(b1-a1, b) / cross_ab
    u = cross(b1-a1, a) / cross_ab

    if 0 ≤ t ≤ 1 and 0 ≤ u ≤ 1
        return a1 + t*a

    return None
```

### Rendering
```python
def render(self, surface: pygame.Surface, scale: float=1.0):
    for idx, angle in enumerate(self._angles):
        absolute_angle = np.radians(self._hdg + angle)
        x = self._pos_x + self._ranges[idx] * np.sin(absolute_angle)
        y = self._pos_y - self._ranges[idx] * np.cos(absolute_angle)
        pygame.draw.aaline(
            surface,
            (90, 90, 200),
            (self._pos_x * scale, self._pos_y * scale),
            (x * scale, y * scale))
```
- Iterate through each indices in **self._angles** (LIDAR_SWATH) and find 
- Calculate **absolute_angle** based on current heading 
- Compute the corresponding **self._ranges** incorporated with **absolute_angle** and store as as (x, y) coordinates
- Draw a line from lidar to the calculated (x, y) point

## 3. ASV Gym Environment

## File: asv_lidar_rudder_speed_control.py

## 4. Train/Test/Eval Agent with Stable-Baselines3

### File: train_test_asv.py