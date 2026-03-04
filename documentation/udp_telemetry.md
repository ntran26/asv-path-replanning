## 1. Decode Log File

### File: log_parser.py
This file containts helper functions that parse and decode lines from the log file.

### Imports
```python
from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any
import time
```

- **re**: regex parsing log lines
- **numpy**: for lidar arrays and math
- **dataclass**: define frame object
- **typing**: hints for data types (for readability and debugging)

### Bluefin frame data container
```python
@dataclass
class BluefinFrame:
    t_sec: float
    ts_str: str

    x_m: float
    y_m: float
    yaw_deg: float

    vx_mps: float
    vy_mps: float
    speed_mps: float

    lidar_m: np.ndarray

    hdg_ref_deg: Optional[float] = None
    s1: Optional[int] = None
    s2: Optional[int] = None
    seq: Optional[int] = None
```
A Bluefin frame is the decoded state snapshot produced each time a LiDAR scan line arrives.

It includes:
- Time:
    - **ts_str**: original timestamp string from the log (13:32:07.817313)
    - **t_sec**: time since start of run (relative time, seconds)
- Pose:
    - **x_m, y_m, yaw_deg**: pose in meters and heading/yaw angle in degrees
- Derived velocity:
    - **vx_mps, vy_mps**: computed from pose vs time
    - **speed_mps**: speed magnitude (hypot)
- Sensor:
    - **lidar_m**: LiDAR ranges in meters (numpy array)
- Latched metadata (optional):
    - **hdg_ref_deg**: reference heading (HDG)
    - **s1, s2**: rudder, thruster values
    - **seq**: sequence number if present

### Helper functions
```python
def _ts_to_seconds(ts_str: str) -> float:
    hh, mm, rest = ts_str.split(":")
    ss, micros = rest.split(".")
    seconds = int(hh)*3600 + int(mm)*60 + int(ss) + int(micros)/1e6
    return seconds
```
Converts a log timestamp string (HH:MM:SS.microseconds) into seconds.
```math
seconds = hours*3600 + minutes*60 + seconds + \frac{microseconds}{10^6}
```

```python
def _wrap_360(deg: float) -> float:
    return (deg % 360 + 360) % 360
```
Map any angle into range of [0, 360]

```python
def _parse_int_list_csv(text: str) -> np.ndarray:
    parts = text.split(",")
    out = np.fromiter((int(p) for p in (x.strip() for x in parts) if p != ""), dtype=np.int32)
    return out
```
Converts the inner CSV string from LiDAR line into a numpy array of int32.

```python
def _rotate_lidar_by_degrees(lidar_m: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0:
        return lidar_m
    shift = int(round(degrees*2))
    return np.roll(lidar_m, shift)
```
The real LiDAR may have a different initial index (beam 0) direction than the simulator. Since the current LiDAR has 720 beams across 360° -> 1 beam every 0.5° -> shifting by degrees*2 beams.

```python
def _downsample_stride(arr: np.ndarray, out_n: int) -> np.ndarray:
    n = len(arr)
    if out_n == n:
        return arr
    if out_n <= 0:
        raise ValueError("out_n must be > 0")
    step = n / out_n
    idx = (np.arange(out_n) * step).astype(int)
    return arr[idx]
```
If downsample to fewer beams (90 or 180), this function selects evenly spaced indices. This is important for matching RL training vs deployment as the policy expects a certain lidar shape for observation.

### Decode lines by regex patterns
```python
class BluefinStreamDecoder:
    _re_hdg = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<seq>\d+)\]\s*HDG:(?P<hdg>[-+]?\d+(?:\.\d+)?)\s*$"
    )

    _re_pose = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]"
        r"(?P<y>[-+]\d+\.\d+),(?P<x>[-+]\d+\.\d+),(?P<yaw>[-+]\d+\.\d+)\s*$"
    )

    _re_rc = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<seq>\d+)\]\s*"
        r"S1:(?P<s1>\d+)\s*S2:(?P<s2>\d+)\s*RC\s*$"
    )

    _re_lidar = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<body>.*)\]\s*$"
    )
```
This class is built for streaming/incremental decoding:
- One raw log line is fed at a time.
- It remembers (latches) the most recent pose/RC/HDG lines.
- When it finally receives a LiDAR line, it constructs and returns a complete BluefinFrame.

The regex patterns describe the different line formats:
- **_re_hdg**: timestamp + sequence + HDG:...
- **_re_pose**: timestamp + (Y, X, Yaw)
- **_re_rc**: timestamp + sequence + S1:.., S2:.. RC
- **_re_lidar**: timestamp + [comma-separated list]

Note: Since the coordinate system of the vessel is shifted, the correct order is (-Y, X, -Yaw) instead of (X, Y, Yaw) as convention.

### Decoder initialization
```python
def __init__(
        self,
        *,
        lidar_out_beams: int = 720,
        lidar_angle_offset_deg: float = 0,
        lidar_max_m: float = 16,
        lidar_unit_scale: float = 0.1,
        lidar_out_of_range: bool = True) -> None:

    self.lidar_out_beams = int(lidar_out_beams)
    self.lidar_angle_offset_deg = float(lidar_angle_offset_deg)
    self.lidar_max_m = float(lidar_max_m)
    self.lidar_unit_scale = float(lidar_unit_scale)
    self.lidar_out_of_range = bool(lidar_out_of_range)

    self._t0: Optional[float] = None

    self._last_hdg_ref: Optional[float] = None
    self._last_seq: Optional[int] = None
    self._last_s1: Optional[int] = 0
    self._last_s2: Optional[int] = 0

    self._last_pose: Optional[Tuple[float, float, float]] = None
    self._last_pose_t: Optional[float] = None
    self._last_vel: Tuple[float, float, float] = (0, 0, 0)
```
- **lidar_out_beams**: output beam count after optional downsampling
- **lidar_angle_offset_deg**: circular shift to align LiDAR index 0
- **lidar_max_m**: sensor maximum range
- **lidar_unit_scale**: lidar unit conversion (dm -> m)
- **lidar_out_of_range**: if True, replace with **lidar_max_m**

```python
def _real_time(self, ts_str: str) -> float:
    sec = _ts_to_seconds(ts_str)
    if self._t0 is None:
        self._t0 = sec
    return sec - self._t0
```

Calculate time since start in seconds.

### Core streaming decode function
```python
def feed(self, line: str) -> Optional[BluefinFrame]:
    line = line.strip()
    if not line:
        return None
```
Strips whitespace and ignore empty lines
```python
m = self._re_rc.match(line)
if m:
    self._last_seq = int(m.group("seq"))
    self._last_s1 = int(m.group("s1"))
    self._last_s2 = int(m.group("s2"))
    return None

m = self._re_hdg.match(line)
    if m:
        self._last_seq = int(m.group("seq"))
        self._last_hdg_ref = float(m.group("hdg"))
        return None
```
Latch RC state (seq, s1, s2) and reference heading (seq, hdg) if available. No frame is produced yet.

```python
m = self._re_pose.match(line)
    if m:
        ts_str = m.group("ts")
        t = self._real_time(ts_str)

        x = float(m.group("x"))
        y = -float(m.group("y"))
        yaw_deg = -float(m.group("yaw"))

        if self._last_pose is not None and self._last_pose_t is not None:
            dt = t - self._last_pose_t
            if dt > 1e-6:
                prev_x, prev_y, _ = self._last_pose
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                spd = float(np.hypot(vx, vy))
                self._last_vel = (float(vx), float(vy), spd)
        
        self._last_pose = (x, y, yaw_deg)
        self._last_pose_t = t
        return None
```
Pose line handling and velocity estimation:
- Coordinate conversion: Assign and convert the coordinates so that it matches the form of (-Y, X, -Yaw).
- Velocity estimation:
    - Calculate the position differences (vx, vy) over time dt.
    - Stores last computed velocity in **_last_vel** for later use when the LiDAR line arrives.

No frame produced at this step.

```python
m = self._re_lidar.match(line)
if m:
    ts_str = m.group("ts")
    t = self._real_time(ts_str)

    lidar_int = _parse_int_list_csv(m.group("body"))
    lidar_m = lidar_int.astype(np.float32) * self.lidar_unit_scale

    if self.lidar_out_of_range:
        lidar_m = np.where(lidar_m <= 0, self.lidar_max_m, lidar_m)

    lidar_m = np.clip(lidar_m, 0, self.lidar_max_m)
    lidar_m = _rotate_lidar_by_degrees(lidar_m, self.lidar_angle_offset_deg)
    lidar_m = _downsample_stride(lidar_m, self.lidar_out_beams)

    if self._last_pose is None:
        return None

    x, y, yaw_deg = self._last_pose
    vx, vy, spd = self._last_vel

    return BluefinFrame(
        t_sec=float(t),
        ts_str=ts_str,
        x_m=float(x),
        y_m=float(y),
        yaw_deg=float(yaw_deg),
        vx_mps=float(vx),
        vy_mps=float(vy),
        speed_mps=float(spd),
        lidar_m=lidar_m,
        hdg_ref_deg=self._last_hdg_ref,
        s1=self._last_s1,
        s2=self._last_s2,
        seq=self._last_seq
    )
```
LiDAR line handling, which produces a BluefinFrame. The procedure goes as follows:
- Parse lidar list into meters, stored as **lidar_m**.
- Replace zeros with max range with **np_clip** (can be reconfigured).
- Rotate and downsample if necessary.
- Combine with **_last_pose** and **_last_vel** (and **_last_hdg_ref**, **_last_s1**, **_last_s2**, **_last_seq** if available).
- Return BluefinFrame

```python
def frames_from_file(filepath: str, decoder: Optional[BluefinStreamDecoder] = None):
    if decoder is None:
        decoder = BluefinStreamDecoder()
    
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            frame = decoder.feed(line)
            if frame is not None:
                yield frame
```
Offiline decoder: read frames from a log file and decode to produce BluefinFrame.

```python
def frame_to_gym_obs(frame: BluefinFrame,
                    *,
                    origin_xyh: Optional[Tuple[float, float, float]] = None) -> 
                    Dict[str, Any]:
    x = frame.x_m
    y = frame.y_m
    yaw = frame.yaw_deg

    if origin_xyh is not None:
        x0, y0, yaw0 = origin_xyh
        x -= x0
        y -= y0
        yaw = _wrap_360(yaw - yaw0)
    
    obs: Dict[str, Any] = {
        "lidar": frame.lidar_m.astype(np.float32),
        "pos": np.array([x,y], dtype=np.float32),
        "hdg": np.array([yaw], dtype=np.float32),
        "dhdg": np.array([0.0], dtype=np.float32),
        "speed": np.array([frame.speed_mps], dtype=np.float32),
        "tgt": np.array([0.0], dtype=np.float32),
        "target_heading": np.array([0.0], dtype=np.float32),
    }        

    return obs
```
Convert a BluefinFrame into RL environment observation dict.
- This function creates a MultiInputPolicy dictionary observation.
- It supports "relative coordinates" by substracting an origin pose (x0, y0, yaw0) if **origin_xyh** is specified.

```python
if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python log_parser.py test_1.log")
        raise SystemExit(1)
    
    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    origin = None
    count = 0

    test_dir = 'data'
    test_file = sys.argv[1]
    filename = os.path.join(test_dir, test_file)

    for frame in frames_from_file(filename, decoder):
        if origin is None:
            origin = (frame.x_m, frame.y_m, frame.yaw_deg)
        
        obs = frame_to_gym_obs(frame, origin_xyh=origin, include_velocity=True)

        if count:
            print(f"Frame {count}: t={frame.t_sec:.3f}s pos={obs['pos']} yaw={obs['hdg']} spd={obs['spd']}")
            print(f" lidar shape: {obs['lidar'].shape}, min/max: {obs['lidar'].min():.2f}/{obs['lidar'].max():.2f}")
            time.sleep(0.1)
        count += 1
    
    print(f"Decoded {count} frames.")
```
Main loop upon running the script. Used for testing if it can open the log file, decode frames, print pose and lidar status.

## 2. Render Data: 

### File: log_viewer.py

```python
python log_viewer.py data/test_1.log
```
This script uses the data decoded from a log file and renders "real-time" status and trajectory of the vessel. The purpose of this is to check if the decoder receives and output the data correctly through visualization. The rendered window is interactive: user can pause, expand/minimize and scroll through LiDAR list, track/untrack vessel position, etc. The pygame window renders:
- Text status for each frame
- Full or processed LiDAR list
- Map panel with trajectory, heading arrow, LiDAR beams
- (Optional) Recording and trajectory plot

**log_parser.py** is considered as backend and **log_viewer.py** is the frontend (UI).

```python
LIDAR_FULL_BEAMS = 720
LIDAR_FULL_STEP = 360 / LIDAR_FULL_BEAMS

LIDAR_SWATH = 270
LIDAR_BEAMS = 90
LIDAR_MAX = 16
LIDAR_INDEX_DEG = 0

VESSEL_LENGTH = 1.7
VESSEL_WIDTH = 0.5
LIDAR_OFFSET_M = VESSEL_LENGTH/2
```
### Define LiDAR and vessel constants
- The original LiDAR format from log is 720 beams over 360° -> Step = 360/720 = 0.5
- For display and to match RL observation space, the number of LiDAR beams is set to 90 over LiDAR swath of 270° (downsampled)
- **LIDAR_INDEX_DEG** defines the angle offset if the initial LiDAR index needs to be calibrated.
- **LIDAR_OFFSET_M** places the LiDAR in front of the vessel when rendering.

```python
class FrameStream:
    def __init__(self, filepath: str, decoder: Optional[BluefinStreamDecoder] = None):
        self.filepath = filepath
        self.decoder = decoder or BluefinStreamDecoder(lidar_out_beams=720)
        self._fh = open(filepath, "r", errors="ignore")
        self.frame_index = 0

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def restart(self) -> None:
        self.close()
        self._fh = open(self.filepath, "r", errors="ignore")
        self.frame_index = 0

        # Recreate a fresh decoder with the same settings
        self.decoder = BluefinStreamDecoder(
            lidar_out_beams=self.decoder.lidar_out_beams,
            lidar_angle_offset_deg=self.decoder.lidar_angle_offset_deg,
            lidar_max_m=self.decoder.lidar_max_m,
            lidar_unit_scale=self.decoder.lidar_unit_scale,
            lidar_out_of_range=self.decoder.lidar_out_of_range,
        )

    def next_frame(self) -> Optional[BluefinFrame]:
        while True:
            line = self._fh.readline()
            if line == "":
                return None  # EOF
            frame = self.decoder.feed(line)
            if frame is not None:
                self.frame_index += 1
                return frame
```





## Simulate UDP Server

### fake_vessel_replay.py

```
python fake_vessel_replay.py --log data/test_1.log --bind-ip 0.0.0.0 --port 5050
```

This script simulates the vessel telemetry server by replaying a recorded log file over UDP. It waits for a handshake message to start communicating with the client, then streams log lines.

### Imports

```python
import socket
import time
import re
import argparse
```

- **socket**: UDP networking
- **time**: sleep between lines to match timing
- **re**: parse data from log
- **argparse**: command line flags

### 



