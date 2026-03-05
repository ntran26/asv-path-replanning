## 1. Decode Log File

### File: log_parser.py
This file contains helper functions that parse and decode lines from the log file.

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

### Format to RL Dict
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

### Main Loop
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

### Define LiDAR and Vessel Constants
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
- The original LiDAR format from log is 720 beams over 360° -> Step = 360/720 = 0.5
- For display and to match RL observation space, the number of LiDAR beams is set to 90 over LiDAR swath of 270° (downsampled)
- **LIDAR_INDEX_DEG** defines the angle offset if the initial LiDAR index needs to be calibrated.
- **LIDAR_OFFSET_M** places the LiDAR in front of the vessel when rendering.

### FrameStream Class
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
                return None
            frame = self.decoder.feed(line)
            if frame is not None:
                self.frame_index += 1
                return frame
```
This class iterates over the decoded frames. It wraps the file handle and decoder state into an object that can return the next decoded LiDAR frame each time **next_frame()** is called. 

**next_frame()** reads as many lines as needed until it reaches end of file or it gets a full BluefinFrame from the decoder.

### Format LiDAR
```python
def format_lidar_lines(lidar_m: np.ndarray, 
                        *, 
                        per_line: int = 12, 
                        precision: int = 1) -> List[str]:
    if lidar_m.ndim != 1:
        lidar_m = np.asarray(lidar_m).ravel()

    fmt = f"{{:.{precision}f}}"
    tokens = [fmt.format(float(x)) for x in lidar_m]

    lines: List[str] = []
    for i in range(0, len(tokens), per_line):
        chunk = tokens[i : i + per_line]
        lines.append(", ".join(chunk))
    return lines

def pick_lidar_swath(full_ranges_m: np.ndarray, 
                    angles_deg: np.ndarray, 
                    *, 
                    index0_deg: float) -> np.ndarray:
    full_ranges_m = np.asarray(full_ranges_m).ravel()
    n = full_ranges_m.size
    if n == 0:
        return full_ranges_m
    
    step = 360/n
    idx = np.round((angles_deg - index0_deg)/step).astype(int) % n

    return full_ranges_m[idx]
```
- **format_lidar_lines()** makes a long vector printable inside the UI, wrapped across multiple lines.
- **pick_lidar_swath()** is the "view" function that downscale to a desirable number of beams and swath.

In the viewer, the user can toggle between full LiDAR list of 720 values or a processed LiDAR list of 90 beams.

### Coordinate Transform
```python
def world_to_screen(xy_world: Tuple[float, float],
                    *,
                    view_center_world: Tuple[float, float],
                    view_center_px: Tuple[int, int],
                    px_per_m: float) -> Tuple[int, int]:
    x, y = xy_world
    cx_w, cy_w = view_center_world
    cx_px, cy_px = view_center_px

    sx = cx_px + (x - cx_w) * px_per_m
    sy = cy_px - (y - cy_w) * px_per_m
    return sx, sy
```
This function **world_to_screen()** converts world coordinate (meters) to pygame coordinate (pixels). An important convention in pygame is that the y-axis is inverted -> the code substracts **(y - cy_w)** to invert Y.

The "camera" is defined by:
- **view_center_world**: world point at the center of view
- **view_center_pix**: pixel center of the map panel
- **px_per_m**: zoom scale

### Map Rendering Function
```python
def draw_map_panel(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    path_world: List[Tuple[float, float]],
    current_world: Optional[Tuple[float, float]] = None,
    yaw_deg: Optional[float] = None,
    view_center_world: Tuple[float, float],
    px_per_m: float,
    show_axes: bool = True,
    lidar_angles_deg: Optional[np.ndarray] = None,
    lidar_ranges_m: Optional[np.ndarray] = None,
    lidar_offset_m: float = LIDAR_OFFSET_M,
    lidar_index0_deg: float = 0,
    lidar_index0_range_m: Optional[float] = None,
    mark_index0: bool = True
    ) -> None:

    pygame.draw.rect(surface, (10, 10, 12), map_rect)
    pygame.draw.rect(surface, (80, 80, 90), map_rect, width=2)

    view_center_px = map_rect.center
```
The first step is to clear and frame the map panel.

```python
    if len(path_world) >= 2:
        pts = [
            world_to_screen(p, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
            for p in path_world
        ]
        prev_clip = surface.get_clip()
        surface.set_clip(map_rect)
        try:
            pygame.draw.lines(surface, (80, 180, 255), False, pts, 2)
        finally:
            surface.set_clip(prev_clip)
```
Draw the trajectory polyline:
- Each stored point is converted into pixels.
- Use **set_clip(map_rect)** to avoid drawing outside the panel.

```python
    if current_world is not None:
        p = world_to_screen(current_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
        pygame.draw.circle(surface, (255, 255, 255), p, 5)
        pygame.draw.circle(surface, (0, 0, 0), p, 5, 1)

        if yaw_deg is not None:
            yaw_rad = float(np.deg2rad(yaw_deg))
            arrow_len_m = 1.2
            tip_world = (
                float(current_world[0] + arrow_len_m * np.sin(yaw_rad)),
                float(current_world[1] + arrow_len_m * np.cos(yaw_rad)),
            )
            tip = world_to_screen(tip_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
            pygame.draw.line(surface, (255, 200, 80), p, tip, 3)
            pygame.draw.circle(surface, (255, 200, 80), tip, 4)
```
Draw current position and heading arrow:
- Draws the vessel as a dot and heading direction as a yellow arrow.
- Assumes yaw=0 points to positive Y and yaw=90 towards positive X.

```python
    if (current_world is not None) and (yaw_deg is not None) and (lidar_angles_deg is not None) and (lidar_ranges_m is not None):
            
        h = float(np.deg2rad(yaw_deg))

        sensor_world = (float(current_world[0] + lidar_offset_m * np.sin(h)),
                        float(current_world[1] + lidar_offset_m * np.cos(h)))
        s_px = world_to_screen(sensor_world, 
                                view_center_world=view_center_world,
                                view_center_px=view_center_px,
                                px_per_m=px_per_m)
        if mark_index0:
            a0 = float(np.deg2rad(yaw_deg + lidar_index0_deg))
            r0 = float(lidar_index0_range_m) if lidar_index0_range_m is not None else LIDAR_MAX
            r0 = float(np.clip(r0, 0, LIDAR_MAX))

            end0_world = (sensor_world[0] + r0*np.sin(a0), sensor_world[1] + r0*np.cos(a0))
            end0_px = world_to_screen(end0_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)

            pygame.draw.aaline(surface, (255,50,50), s_px, end0_px)
            pygame.draw.circle(surface, (255,50,50), end0_px, 4)

        prev_clip = surface.get_clip()
        surface.set_clip(map_rect)
        try:
            for angle, range in zip(lidar_angles_deg, lidar_ranges_m):
                r = float(np.clip(range, 0, LIDAR_MAX))
                a = float(np.deg2rad(yaw_deg + angle))

                end_world = (sensor_world[0] + r*np.sin(a), sensor_world[1] + r*np.cos(a))
                e_px = world_to_screen(end_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
                pygame.draw.aaline(surface, (90,90,200), s_px, e_px)
        finally:
            surface.set_clip(prev_clip)
```
Draw LiDAR beamss:
- Converts to radians
- Position the LiDAR start point in front of the vessel
- Draw each beam as a line of length = range
- (Optional) highlights the initial index in red

### Functions to Record Video and Plot Trajectory
```python
def surface_to_bgr(screen: pygame.Surface) -> np.ndarray:
    frame_rgb = pygame.surfarray.array3d(screen)            
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))   
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) 
    return frame_bgr

def plot_trajectory(traj_xy: List[Tuple[float, float]], traj_yaw_deg: List[float], out_png: str) -> None:
    import matplotlib.pyplot as plt

    xs = np.array([p[0] for p in traj_xy], dtype=float)
    ys = np.array([p[1] for p in traj_xy], dtype=float)

    plt.figure(figsize=(6,6))
    plt.plot(xs,ys)
    plt.scatter([xs[-1]], [ys[-1]])

    h = np.deg2rad(traj_yaw_deg[-1])
    arrow_len = 1
    dx = arrow_len * np.sin(h)
    dy = arrow_len * np.cos(h)
    plt.arrow(xs[-1], ys[-1], dx, dy, length_includes_head=True)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {out_png}")
```
- **surface_to_bgr()** converts pygame frames into a form of OpenCV so that it can be encoded into MP4.
- **plot_trajectory()** produces a clean final figure.

### Main loop
```python
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to test_*.log")
    ap.add_argument("--rate", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime)")
    ap.add_argument("--fps", type=int, default=60, help="UI frame rate cap")
    ap.add_argument("--full", action="store_true", help="Start with full LiDAR list enabled")
    ap.add_argument("--no-map", action="store_true", help="Start with the map panel hidden")
    ap.add_argument("--zoom", type=float, default=20, help="Initial zoom in pixels per meter")
    ap.add_argument("--record", action="store_true", help="Record an MP4 of the pygame window")
    ap.add_argument("--out-video", default="bluefin_replay.mp4", help="Output video filename")
    ap.add_argument("--out-image", default="bluefin_final.png", help="Output final screenshot filename")
    ap.add_argument("--video-fps", type=float, default=None, help="Video FPS. If not set, defaults to --fps (UI rate).")
    ap.add_argument("--plot", default="trajectory_plot.png", help="Matplotlib trajectory plot output")

    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        raise SystemExit(f"File not found: {args.logfile}")
    if args.rate <= 0:
        raise SystemExit("--rate must be > 0")
```
Argument parsing setup:
- **--rate** playback rate
- **--fps** UI frame rate
- **--full** start with full lidar
- **--no-map** enable/disable map
- **--zoom** zoom scale
- **--record** screen recording

### Initialization
```python
    pygame.init()
    pygame.display.set_caption("Bluefin log viewer + trajectory")
    video_fps = float(args.video_fps) if args.video_fps is not None else float(args.fps)

    win_w, win_h = 1200, 600
    text_w = 800
    map_w = win_w - text_w

    screen = pygame.display.set_mode((win_w, win_h))

    video_writer = None
    capture_period = 1.0 / max(video_fps, 1e-9)
    next_capture_due = time.perf_counter()

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.out_video, fourcc, video_fps, (win_w, win_h))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer")
        print(f"[REC] Recording to {args.out_video} at {video_fps:.1f} fps, size={win_w}x{win_h}")

    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18) or pygame.font.Font(None, 18)
    small = pygame.font.SysFont("consolas", 15) or pygame.font.Font(None, 15)

    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    stream = FrameStream(args.logfile, decoder)
```
- Initialize pygame window, resolution of text/map panels
- Initialize MP4 video recording
- (Optional) Define video framerate if specified
- Define start time, font sizes, and assign decoder

```python
    paused = False
    show_full_lidar = bool(args.full)
    lidar_scroll = 0

    show_map = not bool(args.no_map)
    follow_mode = True

    px_per_m = args.zoom

    origin_world: Optional[Tuple[float, float]] = None
    path_world: List[Tuple[float, float]] = []  
    view_center_world = (0.0, 0.0)

    frame: Optional[BluefinFrame] = None
    prev_t_sec: Optional[float] = None
    next_due = time.perf_counter()
    dt_last = 0.1

    cached_lidar_lines: List[str] = []
    cached_lidar_key = None

    lidar_draw_angles = np.linspace(-LIDAR_SWATH/2, LIDAR_SWATH/2, LIDAR_BEAMS, dtype=np.float64)

    traj_xy: List[Tuple[float, float]] = []
    traj_yaw: List[float] = []
```
- Reset variables to default values
- Initialize variables for displaying text and map panels

### Keyboard Configuration
```python
    running = True
    while running:
        now = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    show_full_lidar = not show_full_lidar
                    lidar_scroll = 0
                elif event.key == pygame.K_r:
                    stream.restart()
                    frame = None
                    prev_t_sec = None
                    cached_lidar_lines = []
                    cached_lidar_key = None
                    lidar_scroll = 0
                    next_due = time.perf_counter()
                elif event.key == pygame.K_p:
                    pygame.image.save(screen, args.out_image)
                    print(f"[IMG] Saved: {args.out_image}")
```
- **ESC** or close pygame window to end stream
- **SPACE** to pause/unpause
- **F** to toggle LiDAR list between full and scaled
- **R** to restart stream
- **P** to take snapshot

```python
                elif event.key == pygame.K_UP:
                    if show_full_lidar:
                        lidar_scroll = max(0, lidar_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    if show_full_lidar:
                        lidar_scroll = lidar_scroll + 1
```
- In full LiDAR list mode, press **UP** and **DOWN** keys to scroll through the list.

```python
                elif event.key == pygame.K_m:
                    show_map = not show_map
                elif event.key == pygame.K_g:
                    follow_mode = not follow_mode
                elif event.key == pygame.K_c:
                    path_world = []
                elif event.key == pygame.K_o:
                    if frame is not None:
                        origin_world = (float(frame.x_m), float(frame.y_m))
                        path_world = [(0.0, 0.0)]
                        view_center_world = (0.0, 0.0)
```
- **M** to toggle map panel on/off
- **G** to follow/unfollow vessel
- **C** to clear the plotting of path taken
- **O** to reset to world origin coordinate (0, 0)

```python
            elif not follow_mode:
                pan_step_m = 20.0 / px_per_m 
                if event.key == pygame.K_w:
                    view_center_world = (view_center_world[0], view_center_world[1] + pan_step_m)
                elif event.key == pygame.K_s:
                    view_center_world = (view_center_world[0], view_center_world[1] - pan_step_m)
                elif event.key == pygame.K_a:
                    view_center_world = (view_center_world[0] - pan_step_m, view_center_world[1])
                elif event.key == pygame.K_d:
                    view_center_world = (view_center_world[0] + pan_step_m, view_center_world[1])
```
- For manual tracking mode, use **W,A,S,D** keys to navigate the map.

```python
        while not paused and now >= next_due:
            next_frame = stream.next_frame()
            if next_frame is None:
                paused = True 
                break
            else:
                if prev_t_sec is None:
                    dt_last = 0.1
                else:
                    dt = float(next_frame.t_sec - prev_t_sec)
                    if dt <= 0 or dt > 5:
                        dt = 0.1
                    dt_last = dt

                frame = next_frame
                prev_t_sec = float(next_frame.t_sec)
                next_due += (dt_last / float(args.rate))

                cached_lidar_key = None

                if origin_world is None:
                    origin_world = (float(frame.x_m), float(frame.y_m))
                    path_world = [(0.0, 0.0)]

                rel = (
                    float(frame.x_m - origin_world[0]),
                    float(frame.y_m - origin_world[1]),
                )
                path_world.append(rel)

                traj_xy.append((float(frame.x_m), float(frame.y_m)))
                traj_yaw.append(float(frame.yaw_deg))

                if follow_mode:
                    view_center_world = rel
```

### Display Text Panel
```python
        screen.fill((20, 20, 25))
        text_rect = pygame.Rect(0, 0, text_w, win_h)
        map_rect = pygame.Rect(text_w, 0, map_w, win_h)

        y = 10
        line_h = 22

        lidar_raw = None
        lidar_view = None

        if frame is not None:
            lidar_raw = frame.lidar_m
            lidar_view = pick_lidar_swath(lidar_raw, lidar_draw_angles, index0_deg=LIDAR_INDEX_DEG)

        header_lines = [
            f"File: {os.path.basename(args.logfile)}",
            f"Playback: {'PAUSED' if paused else 'RUNNING'}   speed={args.rate:.2f}x   (Space=pause, F=full lidar, R=restart)",
            f"Map: {'ON' if show_map else 'OFF'}  follow={'ON' if follow_mode else 'OFF'}  zoom={px_per_m:0.1f}px/m",
        ]

        if frame is None:
            next_due = now
            header_lines.append("Waiting for first LiDAR frame...")
        else:
            lidar = lidar_view if lidar_view is not None else lidar_raw
            lidar_min = float(np.min(lidar)) if lidar.size else float("nan")
            lidar_max = float(np.max(lidar)) if lidar.size else float("nan")
            lidar_mean = float(np.mean(lidar)) if lidar.size else float("nan")

            if origin_world is None:
                rel_x, rel_y = 0.0, 0.0
            else:
                rel_x = float(frame.x_m - origin_world[0])
                rel_y = float(frame.y_m - origin_world[1])

            header_lines += [
                f"Frame #{stream.frame_index:06d}    ts={frame.ts_str}    t_sec={frame.t_sec:9.3f}    dt~{dt_last:0.3f}s (~{(1.0/dt_last if dt_last>1e-6 else 0):0.1f} Hz)",
                f"Pose(SLAM):  x={frame.x_m:+0.3f} m   y={frame.y_m:+0.3f} m   yaw={frame.yaw_deg:0.2f} deg   (hdg_ref={frame.hdg_ref_deg})",
                f"Control: rudder: {frame.s1:0.2f}, thruster: {frame.s2:0.2f}",
                f"Vel(derived): vx={frame.vx_mps:+0.3f} m/s   vy={frame.vy_mps:+0.3f} m/s   speed={frame.speed_mps:0.3f} m/s",
                f"LiDAR: beams={lidar.size}   units=m (dm*0.1)   min/mean/max={lidar_min:0.2f}/{lidar_mean:0.2f}/{lidar_max:0.2f}",
            ]

        for s in header_lines:
            screen.blit(font.render(s, True, (235, 235, 245)), (10, y))
            y += line_h

        y += 10
```

### Display LiDAR Text
```python
        if frame is not None:
            if show_full_lidar:
                lidar_src = lidar_raw
                title = "LiDAR full list (F)"
            else:
                lidar_src = lidar_view
                title = "Processed LiDAR list (F)"
            cached_key = (stream.frame_index, show_full_lidar)
            if cached_key != cached_lidar_key:
                cached_lidar_lines = format_lidar_lines(lidar_src, per_line=15, precision=1)
                cached_lidar_key = cached_key
            max_lines_on_screen = max(1, (win_h-y-20)//18)
            max_scroll = max(0, len(cached_lidar_lines) - max_lines_on_screen)
            lidar_scroll = min(lidar_scroll, max_scroll)
            
            if show_full_lidar:
                info = f"{title} (scroll {lidar_scroll}/{max_scroll})"
            else:
                info = title
            
            screen.blit(font.render(info, True, (200,200,210)), (10,y))
            y += 22

            for s in cached_lidar_lines[lidar_scroll : lidar_scroll + max_lines_on_screen]:
                screen.blit(small.render(s, True, (210,210,220)), (10,y))
                y += 18
```
This block renders LiDAR information on the text panel
- By default, the processed (or downscaled) LiDAR list will be displayed
- Press F to toggle the original (full) LiDAR list on/off
- **cached_key** is used to avoid formatting the LiDAR vector every UI frame
- Only updates when a new BluefinFrame arrives

### Display Map Panel
```python
        if show_map:
            lidar_ranges_draw = pick_lidar_swath(frame.lidar_m, lidar_draw_angles, index0_deg=LIDAR_INDEX_DEG)
            if frame is None or origin_world is None:
                # Show an empty map
                draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=None,
                    yaw_deg=None,
                    view_center_world=view_center_world,
                    px_per_m=px_per_m
                )
            else:
                current_rel = (
                    float(frame.x_m - origin_world[0]),
                    float(frame.y_m - origin_world[1]),
                )
                draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=current_rel,
                    yaw_deg=float(frame.yaw_deg),
                    view_center_world=view_center_world,
                    px_per_m=px_per_m,
                    lidar_angles_deg=lidar_draw_angles,
                    lidar_ranges_m=lidar_ranges_draw,
                    lidar_index0_deg=LIDAR_INDEX_DEG,
                    lidar_index0_range_m=frame.lidar_m[0] if frame.lidar_m.size > 0 else None,
                    mark_index0=True
                )

                label = f"points={len(path_world)}"
                screen.blit(small.render(label, True, (210, 210, 220)), (map_rect.left + 8, map_rect.top + 8))
```
- **lidar_ranges_draw** takes the original LiDAR list and transform it to a desired scale (see **Format LiDAR**)
- If no frame is produced, an empty map will be displayed on the right panel as a placeholder.
- Starts drawing elements along with LiDAR beams as BluefinFrame arrives.

### Write Video Frame
```python
        if video_writer is not None and not paused:
            while now >= next_capture_due:
                video_writer.write(surface_to_bgr(screen))
                next_capture_due += capture_period
        pygame.display.flip()
        clock.tick(args.fps)
```

### End Stream
```python
    stream.close()

    pygame.image.save(screen, args.out_image)
    print(f"[IMG] Saved final screenshot: {args.out_image}")

    if video_writer is not None:
        video_writer.release()
        print(f"[REC] Video saved: {args.out_video}")

    plot_trajectory(traj_xy, traj_yaw, args.plot)

    pygame.quit()
```
- Close the stream
- Save final screenshot
- Release video recording
- Finalize trajectory plot


## 3. Simulate UDP Server

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

### Timestamp Regex and Conversion
```python
TS_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{6})\]")

def ts_to_seconds(line: str):
    m = TS_RE.match(line)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))
    ms = int(m.group(4))
    
    return hh*3600 + mm*60 + ss + ms/1e6
```
- This section is similar to the timestamp regex and convert function from **log_parser.py**.
- Timestamp is decoded in this script to match the timestamp in recorded log with real time.

### Argument Parsing
```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0")     # IP to bind
    ap.add_argument("--port", type=int, default=5050)     # UDP port to listen on
    ap.add_argument("--log", required=True)     # Path to log file to replay
    ap.add_argument("--speed", default=1.0)     # Replay speed: 1x = real time
    ap.add_argument("--ignore-rc", action="store_true")     # ignore RC line
    ap.add_argument("--loop", action="store_true")  # loop log forever
    args = ap.parse_args()
```

### UDP Socket Setup
```python
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.port))

    print("[FAKE VESSEL] Listening on {}:{}", args.bind_ip, args.port)
    print("[FAKE VESSEL] Waiting for START...")

    while True:
        data, addr = sock.recvfrom(4096)
        if data.strip() == b"START":
            client_addr = addr
            print("[FAKE VESSEL] Got START from {}, streaming will start", client_addr)
            break
        else:
            print("[FAKE VESSEL] Ignoring packet from {}: {}", addr, data)
```
- The socket communicates using IPv4 address with UDP protocol (**AF_INET**, **SOCK_DGRAM**)
- **sock.setsockopt** allows the port to be reused immediately, convenient when restarting program
- The socket binds to local address **bind_ip** and **port**
- The server waits for a handshake message (START) to start streaming

### Replay Log
```python
    def replay_once():
        t0 = None
        last_t = None 

        with open(args.log, "r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                if args.ignore_rc and " RC" in line and "S1:" in line and "S2:" in line:
                    continue

                t = ts_to_seconds(line)
                if t is None:
                    continue

                if t0 is None:
                    t0 = t
                    last_t = t
                
                dt = (t - last_t) / max(args.speed, 1e-9)
                if dt > 0:
                    time.sleep(dt)
                
                payload = (line + "\n").encode("utf-8", errors="replace")
                sock.sendto(payload, client_addr)
                last_t = t
```
- Read the log line-by-line
- Use timestamp to sleep between each line (can be modified by **--speed**)
- Send each line as a UDP datagram

```python
    while True:
        replay_once()
        if not args.loop:
            print("[FAKE VESSEL] End of log reached. Exiting.")
            break
        print("[FAKE VESSEL] Looping log...")
        time.sleep(1.0)
```
- Replay log repeatedly if **--loop** is set
- Otherwise, exit when reaches end of file

## 4. Live Telemetry Viewer

For live deployment
```
python udp_listener.py --server-ip 10.201.208.224 --record-log sea_trial.log
```
For simulated UDP server
```
python udp_listener.py
```

This script is the main script for telemetry test with the vessel. It binds a local UDP port, send **START** as a handshake to receive streaming data, then decodes using **BluefinStreamDecoder** and outputs **BluefinFrame**. It renders the decoded data to a pygame window and write the received messages to a log file **sea_trial.log**.

### Imports
```python
import socket
import argparse
import pygame
import numpy as np
from typing import Optional, List, Tuple
from log_parser import BluefinFrame, BluefinStreamDecoder
import log_viewer
```

### Argument Parsing
```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0", help="local bind IP")
    ap.add_argument("--local-port", type=int, default=5000, help="local UDP port to listen on")
    ap.add_argument("--server-ip", default="127.0.0.1", help="vessel IP address")
    ap.add_argument("--server-port", type=int, default=5050, help="vessel port")
    ap.add_argument("--print-raw", action="store_true", help="print raw lines")
    ap.add_argument("--record-log", default=None, help="write to log file")
    # UI viewer
    ap.add_argument("--fps", type=int, default=60, help="UI FPS cap")
    ap.add_argument("--full", action="store_true", help="Start with full LiDAR text enabled")
    ap.add_argument("--no-map", action="store_true", help="Start with map panel hidden")
    ap.add_argument("--zoom", type=float, default=30.0, help="Initial zoom (pixels per meter)")
    ap.add_argument("--max-path", type=int, default=20000, help="Limit stored path points (avoid RAM blowup)")
    args = ap.parse_args()
```

### UDP Socket Creation and Bind
```python
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.local_port))

    sock.sendto(b"START\n", (args.server_ip, args.server_port))
    print("[LISTENER] Sent START to {}, {}", args.server_ip, args.server_port)
    print("[LISTENER] Listening on {}, {}", args.bind_ip, args.local_port)
```
- Creates a UDP socket and binds it to **(bind_ip, local_port)**
- Send handshake message **START** to receive telemetry datagrams

### Initialize Log File
```python
    log = None
    if args.record_log:
        log = open(args.record_log, "a", encoding="utf-8", buffering=1)
        print(f"[LISTENER] Recording received lines to: {args.record_log}")
```
If **--log** is specified, open a file in append mode and write each received line to it

### Decoder and Counters
```python
    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    rx_lines = 0
    rx_frames = 0
```
- **decoder** takes each line and outputs a BluefinFrame
- **rx_lines** counts raw lines received
- **rx_frames** counts decoded LiDAR frames

### Initialize Pygame UI
```python
    pygame.init()
    pygame.display.set_caption("Bluefin UDP live viewer")
    win_w, win_h = 1200, 600
    text_w = 800
    map_w = win_w - text_w
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18) or pygame.font.Font(None, 18)
    small = pygame.font.SysFont("consolas", 15) or pygame.font.Font(None, 15)
```
Initialize pygame, creates window, splits text and map panels, specifies fonts and framerate clock

```python
    paused = False
    show_full_lidar = bool(args.full)
    lidar_scroll = 0

    show_map = not bool(args.no_map)
    follow_mode = True

    px_per_m = args.zoom
    view_center_world = (0,0)
    origin_world: Optional[Tuple[float, float]] = None
    path_world: List[Tuple[float, float]] = []

    frame: Optional[BluefinFrame] = None
    cached_lidar_lines: List[str] = []
    cached_lidar_key = None

    lidar_draw_angles = np.linspace(-log_viewer.LIDAR_SWATH/2, log_viewer.LIDAR_SWATH/2, log_viewer.LIDAR_BEAMS, dtype=np.float64)
```
Set up variables and process LiDAR angles. This part is similar to **log_viewer.py**

```python
    running = True
    while running:
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    show_full_lidar = not show_full_lidar
                    lidar_scroll = 0
                elif event.key == pygame.K_UP:
                    if show_full_lidar:
                        lidar_scroll = max(0, lidar_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    if show_full_lidar:
                        lidar_scroll = lidar_scroll + 1
                elif event.key == pygame.K_m:
                    show_map = not show_map
                elif event.key == pygame.K_g:
                    follow_mode = not follow_mode
                elif event.key == pygame.K_c:
                    path_world = []
                elif event.key == pygame.K_o:
                    if frame is not None:
                        origin_world = (frame.x_m, frame.y_m)
                        path_world = [(0,0)]
                        view_center_world = (0,0)
                elif event.key == pygame.K_p:
                    out = "snapshot.png"
                    pygame.image.save(screen, out)
                    print(f"[LISTENER] Saved snapshot: {out}")
                
                if not follow_mode:
                    pan_step_m = 20 / max(px_per_m, 1e-9)
                    if event.key == pygame.K_w:
                        view_center_world = (view_center_world[0], view_center_world[1] + pan_step_m)
                    elif event.key == pygame.K_s:
                        view_center_world = (view_center_world[0], view_center_world[1] - pan_step_m)
                    elif event.key == pygame.K_a:
                        view_center_world = (view_center_world[0] - pan_step_m, view_center_world[1])
                    elif event.key == pygame.K_d:
                        view_center_world = (view_center_world[0] + pan_step_m, view_center_world[1])
```
The event handling keys are also the same as **log_viewer.py**

### Receive and Decode UDP Packet
```python
        msg, addr = sock.recvfrom(65535)
        line = msg.decode("utf-8", errors="replace")

        rx_lines += 1

        if args.print_raw:
            print(line)

        if log is not None:
            log.write(line + "\n")
        
        decoded = decoder.feed(line)
        if decoded is not None:
            rx_frames += 1
            if not paused:
                frame = decoded
                cached_lidar_key = None
                if origin_world is None:
                    origin_world = (frame.x_m, frame.y_m)
                    path_world = [(0,0)]
                rel = (frame.x_m - origin_world[0], frame.y_m - origin_world[1])
                path_world.append(rel)

                if len(path_world) > args.max_path:
                    path_world = path_world[-args.max_path:]
                if follow_mode:
                    view_center_world = rel
```
- Receives a UDP datagram **recvfrom** and decodes it as UTF-8 text
- The vessel sends 1 telemetry line per UDP datagram, it is then fed directly into the decoder
- When a LiDAR frame is decoded, it updates the current frame and appends to the trajectory list

### Draw UI: Text and Map Panels
```python
        screen.fill((20,20,25))
        map_rect = pygame.Rect(text_w, 0, map_w, win_h)
        y = 10
        line_h = 22
        lidar_raw = None
        lidar_view = None

        if frame is not None:
            lidar_raw = frame.lidar_m
            lidar_view = log_viewer.pick_lidar_swath(lidar_raw, lidar_draw_angles, index0_deg=log_viewer.LIDAR_INDEX_DEG)
        header_lines = [f"UDP: local={args.bind_ip}:{args.local_port}  server={args.server_ip}:{args.server_port}",
                        f"RX: lines={rx_lines}  frames={rx_frames}   {'PAUSED' if paused else 'RUNNING'} (Space)   full_lidar={'ON' if show_full_lidar else 'OFF'} (F)",
                        f"Map: {'ON' if show_map else 'OFF'} (M)   follow={'ON' if follow_mode else 'OFF'} (G)   zoom={px_per_m:0.1f}px/m  origin={'SET' if origin_world else 'NONE'} (O)"]
        if frame is None:
            header_lines.append("Waiting for first decoded LiDAR frame...")
        else:
            lidar = frame.lidar_m
            header_lines += [
                f"ts={frame.ts_str}   t={frame.t_sec:0.3f}s   seq={frame.seq}   hdg_ref={frame.hdg_ref_deg}",
                f"Pose: x={frame.x_m:+0.3f} m  y={frame.y_m:+0.3f} m  yaw={frame.yaw_deg:+0.2f} deg",
                f"Vel:  vx={frame.vx_mps:+0.3f} m/s  vy={frame.vy_mps:+0.3f} m/s  spd={frame.speed_mps:0.3f} m/s",
                f"RC:   S1={frame.s1}   S2={frame.s2}",
                f"LiDAR: N={lidar.size}  min/mean/max={float(lidar.min()):0.2f}/{float(lidar.mean()):0.2f}/{float(lidar.max()):0.2f}",
            ]
        for s in header_lines:
            screen.blit(font.render(s, True, (235, 235, 245)), (10, y))
            y += line_h
        y += 10
        # lidar text area
        if frame is not None:
            if show_full_lidar:
                lidar_src = lidar_raw
                title = "LiDAR full list (F)"
            else:
                lidar_src = lidar_view
                title = "Processed LiDAR list (F)"
            cached_key = (rx_frames, show_full_lidar)
            if cached_key != cached_lidar_key:
                cached_lidar_lines = log_viewer.format_lidar_lines(lidar_src, per_line=15, precision=1)
                cached_lidar_key = cached_key
            max_lines_on_screen = max(1, (win_h-y-20)//18)
            max_scroll = max(0, len(cached_lidar_lines) - max_lines_on_screen)
            lidar_scroll = min(lidar_scroll, max_scroll)
            if show_full_lidar:
                info = f"{title} (scroll {lidar_scroll}/{max_scroll})"
            else:
                info = title
            screen.blit(font.render(info, True, (200,200,210)), (10,y))
            y += 22
            for s in cached_lidar_lines[lidar_scroll : lidar_scroll + max_lines_on_screen]:
                screen.blit(small.render(s, True, (210,210,220)), (10,y))
                y += 18
        # map panel
        if show_map:
            if frame is None or origin_world is None:
                log_viewer.draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=None,
                    yaw_deg=None,
                    view_center_world=view_center_world,
                    px_per_m=px_per_m)
            else:
                current_rel = (frame.x_m - origin_world[0], frame.y_m - origin_world[1])
                lidar_ranges_draw = log_viewer.pick_lidar_swath(frame.lidar_m, lidar_draw_angles, index0_deg=log_viewer.LIDAR_INDEX_DEG)

                log_viewer.draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=current_rel,
                    yaw_deg=frame.yaw_deg,
                    view_center_world=view_center_world,
                    px_per_m=px_per_m,
                    lidar_angles_deg=lidar_draw_angles,
                    lidar_ranges_m=lidar_ranges_draw,
                    lidar_index0_deg=log_viewer.LIDAR_INDEX_DEG,
                    lidar_index0_range_m=float(frame.lidar_m[0]) if frame.lidar_m.size > 0 else None,
                    mark_index0=True)

        pygame.display.flip()
        clock.tick(args.fps)
```
The rendering block is similar to **log_viewer.py**, with the addition of writing to log file and frame counter.

### Cleanup and Exit
```python
    if log is not None:
        log.close()
    
    sock.close()
    pygame.quit()
    print("[LISTENER] Exit")
```
