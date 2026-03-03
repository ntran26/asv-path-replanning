## Simulate LiDAR

### File: asv_lidar.py

### Overview

This code implements a 2D LiDAR simulator for the ASV gym environment. It casts LiDAR beams at a set of angles and measures distance to obstacles and map borders.


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