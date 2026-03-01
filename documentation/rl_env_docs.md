## Simulate LiDAR

### File: asv_lidar.py

### Overview

This code implements a 2D LiDAR simulator for the ASV gym environment. It casts LiDAR beams at a set of angles and measures distance to obstacles and map borders.

Key constants define the sensor configuration:
- **LIDAR_RANGE**: maximum range (m)
- **LIDAR_SWATH**: angular field of view (degrees)
- **LIDAR_BEAMS**: number of LiDAR beams

