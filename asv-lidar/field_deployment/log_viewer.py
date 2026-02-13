"""
Replay a Bluefin log file and visualize decoded values 
in a simple pygame window.

Parameters:
- Timestamp + t_sec
- Pose: x, y, yaw
- Derived velocity: vx, vy, speed
- LiDAR stats + (optionally) the full list, scrollable

Controls:
- SPACE: pause/resume
- UP/DOWN: scroll LiDAR output
- F: toggle LiDAR list on/off
- R: restart
- ESC: quit

Run:
    python log_viewer.py test_1.log
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Optional, List
import numpy as np
import pygame
from log_parser import BluefinFrame, BluefinStreamDecoder

class FrameStream:
    """
    Incremental decoder for a log file
    """

    def __init__(self, filepath:str, decoder:Optional[BluefinStreamDecoder] = None):
        self.filepath = filepath
        self.decoder = decoder
        self._fh = open(filepath, 'r', errors='ignore')
        self.frame_index = 0

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def restart(self) -> None:
        self.close()
        self._fh = open(self.filepath, 'r', errors='ignore')
        self.frame_index = 0
        self.decoder = BluefinStreamDecoder(
            lidar_out_beams=self.decoder.lidar_out_beams,
            lidar_angle_offset_deg=self.decoder.lidar_angle_offset_deg,
            lidar_max_m=self.decoder.lidar_max_m,
            lidar_unit_scale=self.decoder.lidar_unit_scale,
            lidar_out_of_range=self.decoder.lidar_out_of_range,
        )
    
    def next_frame(self) -> Optional[BluefinFrame]:
        """
        Return the next decoded frame, or None at end of file
        """
        while True:
            line = self._fh.readline()
            if line == "":
                return None
            frame = self.decoder.feed(line)
            if frame is not None:
                self.frame_index += 1
                return frame
    
    def format_lidar_lines(lidar_m, per_line=12, precision=1):
        """
        Turn lidar array into a list of strings
        each string contains a number of "per_line" values for clear observation
        """
        lidar_m = np.asarray(lidar_m).ravel()
        format = f"{{:.{precision}f}}"
        for x in lidar_m:
            output = [format.format(float(x))]
        lines = []
        for i in range(0,len(output),per_line):
            chunk = output[i:i+per_line]
            lines.append(", ".join(chunk))
        return lines
    
    def world_to_screen(xy_world, view_center_world, view_center_px, px_per_m):
        """
        xy_world: (x,y) in meters
        view_center_world: (cx,cy) in meters 
        view_center_px: (cx,cy) in pixels
        
        Pygame coordination:
            +x to the right
            +y from top down => invert y when drawing
        """

        x, y = xy_world
        cx_w, cy_w = view_center_world
        cx_px, cy_px = view_center_px

        sx = cx_px + (x - cx_w) * px_per_m
        sy = cy_px - (y - cy_w) * px_per_m  # invert y

        return int(round(sx)), int(round(sy))