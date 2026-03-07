# RL Gym Environment Simulation
### Overview
The simulation consists of 4 core scripts:
1. **ship_model.py** describes vessel dynamics
2. **asv_lidar.py** simulates a 2D LiDAR system
3. **asv_lidar_rudder_speed_control.py** contains the simulated gym environment
4. **train_test_asv.py** is responsible for training and testing/evaluating the RL agent
(Optional) **test_run.py** contains the preset scenarios to test the performance
(Optional) **plot_data.py** constructs a plot of a completed run

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
- $\boldsymbol{\delta}$ (deg or rad): rudder angle command
- **RPM**: propeller speed command

Time and pose
- $\boldsymbol{\Delta t}$ (s): simulation step (UPDATE_RATE)
- **(x,y)**: vessel position in world coordinate
- $\boldsymbol{\omega}$ (deg/s or rad/s): yaw rate

Dynamics
- 


## 2. Simulate LiDAR

### File: asv_lidar.py

This code implements a 2D LiDAR simulator for the ASV gym environment. It casts LiDAR beams at a set of angles and measures distance to obstacles and map borders. For this simulated LiDAR to work, the coordinates of obstacles and map boundaries are known and fed as inputs.

Workflow: LidarScan =(pos(x,y), heading)
1. Offset sensor origin forward by vessel_length/2 along heading
2. Convert each obstacle polygon into a list of edge segments
3. For each beam angle θ_i:
    - Compute ray end at max range R
    - For each edge segment:
        - Compute intersection with ray segment
        - Keep the shortest distance
    - Store the shortest distance as **ranges[i]**
4. Return **ranges**



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


```python
    @property
    def angles(self):
        return self._angles.copy()

    @property
    def ranges(self):
        return self._ranges.copy()
```
Make a copy of lidar **angles** and **ranges** for outputs.