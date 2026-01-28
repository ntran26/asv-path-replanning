"""
Extract data from the test logs

Log file format:
- HDG: global heading, can be ignored
- Position: X, Y, Yaw (m)
- Lidar values: 10 - 160 (dm), else 0
- (Manual control): S1 - rudder, S2 - thruster

Parse values: Position and Lidar

Outputs:
- Lidar ranges (full and snipped)
- Position (X,Y)
- Heading angle
- Heading rate
- Speed (m/s)
- Target (implement at later stage)
- Heading relative to target (implement at later stage)


"""

import re
import numpy as np
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any

# Data Container for a single timestep
@dataclass
class BluefinFrame:
    """
    A single frame for Bluefin is 10 Hz

    Time
        t_sec: seconds since the start of the run
        ts_str: original HH:MM:SS.microseconds string from the log

    Pose
        x_m, y_m: position (m)
        yaw_deg: yaw/heading normalized to [0,360]

    Velocity (derived from position and time)
        vx_mps, vy_mps: velocity in world frame
        speed_mps: U = sqrt(vx^2 + vy^2)
    
    Range sensor
        lidar_m: LiDAR ranges (m), shape=(N,)
    
    """

    t_sec: float
    ts_str: str

    x_m: float
    y_m: float
    yaw_deg: float

    vx_mps: float
    vy_mps: float
    speed_mps: float

    lidar_m: np.ndarray

# Helper Functions

def _ts_to_seconds(ts_str: str) -> float:
    """
    Convert HH:MM:SS.microsec to seconds (float)

    Example: 13:32:07.817313 = 13*3600 + 32*60
    """
    hh, mm, rest = ts_str.split(":")
    ss, micros = rest.split(".")
    return int(hh)*3600 + int(mm)*60 + int(ss) + int(micros) / 1e6

def _wrap_360(deg: float) -> float:
    """
    Normalize angle to [0,360]
    """
    return (deg % 360 + 360) % 360

def _parse_int_list_csv(text: str) -> np.ndarray:
    """
    Parse a list of integers into a numpy array
    Use for LiDAR list inside [...]
    """
    parts = text.split(",")
    out = np.fromiter((int(p) for p in (x.strip() for x in parts) if p != ""), dtype=np.int32)
    return out

def _rotate_lidar_by_degrees(lidar_m: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotate a 360 degree LiDAR scan

    720 beams over 360 deg, each beam = 0.5 deg
    shift_beams = degrees/0.5 = degrees*2

    Use when LiDAR's initial index differs from simulation
    """
    if degrees == 0:
        return lidar_m
    shift = int(round(degrees*2))
    return np.roll(lidar_m, shift)

def _downsample_stride(arr: np.ndarray, out_n: int) -> np.ndarray:
    """
    Stride downsampling using evenly spaced indices
    Works even if out_n doesn't divide len(arr)
    """
    n = len(arr)
    if out_n == n:
        return arr
    if out_n <= 0:
        raise ValueError("out_n must be > 0")
    step = n / out_n
    idx = (np.arrange(out_n) * step).astype(int)
    return arr[idx]

# Streaming Decoder
class BluefinStreamDecoder:
    """
    A decoder that takes lines and outputs BluefinFrame objects
    Output a frame when there is a LiDAR line
    """

    # Regex patterns matching the line formats

    _re_hdg = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<seq>\d+)\]\s*HDG:(?P<hdg>[-+]?\d+(?:\.\d+)?)\s*$"
    )

    _re_pose = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]"
        r"(?P<x>[-+]\d+\.\d+),(?P<y>[-+]\d+\.\d+),(?P<yaw>[-+]\d+\.\d+)\s*$"
    )

    _re_rc = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<seq>\d+)\]\s*"
        r"S1:(?P<s1>\d+)\s*S2:(?P<s2>\d+)\s*RC\s*$"
    )

    _re_lidar = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<body>.*)\]\s*$"
    )

    def __init__(
            self,
            *,
            lidar_out_beams: int = 720,
            lidar_angle_offset_deg: float = 0,
            lidar_max_m: float = 16,
            lidar_unit_scale: float = 0.1,
            lidar_out_of_range: bool = True) -> None:
        
        self.lidar_out_beams = int(lidar_out_beams)