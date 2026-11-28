"""
Extract data from the test logs

Log file format:
- HDG: global heading, can be ignored
- Position: X, Y, Yaw (m)
- Lidar values: 10 - 160 (dm), else 0
- (Manual control): S1 - rudder, S2 - thruster

Parse values: Position and Lidar

The vessel's own log (RAW_LOG, datalogs/laser_*.log; checked on the D0 logs of 2026-11-26) adds
IMU and servo lines, `[HH:MM:SS.ffffff]<counter ms>,<KIND>,<values>`:
- GYRO,x,y,z   angular rate, deg/s (z: yaw, counter-clockwise positive; 1/16 deg/s steps)
- ACCEL,x,y,z  linear acceleration, m/s^2, gravity removed (0.01 steps)
- QUAT,x,y,z,w orientation quaternion, unit norm (its heading drifts; not a heading reference)
- SRVO,s1,s2,f1,f2  servo values (about 1000-2000) and two flags
about 9.5 Hz each, with pose lines at up to about 85 Hz (repeated in bursts) and LiDAR at 10 Hz.
BluefinStreamDecoder.raw_format() reads it: the IMU and servo values ride on each frame, the pose is
read in the November 2026 layout, and speeds come from poses about 0.5 s apart.  The defaults keep
the July 2026 behaviour (July layout, consecutive-pose speeds, IMU lines skipped).

Outputs:
- Lidar ranges (full and snipped)
- Position (X,Y)
- Heading angle
- Heading rate
- Speed (m/s)
- Target (implement at later stage)
- Heading relative to target (implement at later stage)
"""

from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any
import time

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
    yaw_rate: float

    # body frame
    u_body_mps: float
    v_body_mps: float

    lidar_m: np.ndarray

    hdg_ref_deg: Optional[float] = None
    s1: Optional[int] = None
    s2: Optional[int] = None
    seq: Optional[int] = None

    # The vessel's raw-log lines (None when the stream carries none): the latest value of each
    gyro_dps: Optional[Tuple[float, float, float]] = None
    accel_mps2: Optional[Tuple[float, float, float]] = None
    quat_xyzw: Optional[Tuple[float, float, float, float]] = None
    srvo: Optional[Tuple[int, int, int, int]] = None
    imu_t_sec: Optional[float] = None
    imu_counter_ms: Optional[int] = None

# Helper Functions

def _ts_to_seconds(ts_str: str) -> float:
    """
    Convert HH:MM:SS.microsec to seconds (float)

    Example: 13:32:07.817313 = 13*3600 + 32*60
    """
    hh, mm, rest = ts_str.split(":")
    ss, micros = rest.split(".")
    seconds = int(hh)*3600 + int(mm)*60 + int(ss) + int(micros) / 1e6
    return seconds

def _wrap_360(deg: float) -> float:
    """
    Normalize angle to [0,360]
    """
    return (deg % 360 + 360) % 360

def _wrap_180(deg: float) -> float:
    """
    Normalize angle difference to [-180, 180]
    Useful for shortest signed angular difference
    """
    return (deg + 180.0) % 360.0 - 180.0

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
    idx = (np.arange(out_n) * step).astype(int)
    return arr[idx]

# Streaming Decoder
# Pose layouts: a pose line [ts]A,B,C as (x, y, heading deg clockwise from +y).
#   july    x = B,  y = -A, heading = -C   July 2026 and February 2026 logs (the default here)
#   nov2026 x = -B, y = A,  heading = -C   the vessel's output since November 2026 (D0 logs and
#                                          videos of 2026-11-26; as --pose-map in the field tools)
POSE_MAPS = {
    "july": lambda a, b, c: (b, -a, -c),
    "nov2026": lambda a, b, c: (-b, a, -c),
}


class BluefinStreamDecoder:
    """
    A decoder that takes lines and outputs BluefinFrame objects
    Output a frame when there is a LiDAR line
    """

    r"""
    Regex patterns matching the line formats
        re.compile(pattern): pre-compile regex to run faster
        r"..": raw string
        ^ and $: start and end of line
        \[ and \]: match the bracket characters
        \d: one digit
        \d{2}: exactly two digits (HH:MM:SS)
        \d{6}: exactly six digits (microseconds)
            \d{2}:\d{2}:\d{2}\.\d{6} matches 13:32:07.817313
        (?P<name>...): named capture groups, capture whatever matches inside and store it under "name"
            (?P<seq>\d+) captures the sequence number
            (?P<body>.*) captures the LiDAR list
            (?P<x>...),(?P<y>...),(?P<yaw>...) captures x, y, yaw
        \s*: allow any amount of spaces
        .*: match anything (except newline)
            *: repeat 0 or more times
            \[(?P<body>.*)\]: capture all text inside bracket (lidar ranges)
    """

    _re_hdg = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<seq>\d+)\]\s*HDG:(?P<hdg>[-+]?\d+(?:\.\d+)?)\s*$"
    )

    # Re-check sensor orientation (-Y, X, -Yaw)
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

    # The vessel's raw log: [ts]<counter ms>,<GYRO|ACCEL|QUAT|SRVO>,<values>
    _re_vessel = re.compile(
        r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\](?P<counter>\d+),(?P<kind>GYRO|ACCEL|QUAT|SRVO),"
        r"(?P<vals>[-+0-9.,]+)\s*$"
    )

    def __init__(
            self,
            *,
            lidar_out_beams: int = 720,
            lidar_angle_offset_deg: float = 0,
            lidar_max_m: float = 16,
            lidar_unit_scale: float = 0.1,
            lidar_out_of_range: bool = True,
            pose_map: str = "july",
            velocity_window_s: float = 0.0) -> None:
        """
        lidar_out_beams:
            Output beam count after optional downsampling
        lidar_angle_offset_deg:
            Circular shift to align LiDAR index convention with simulation
        lidar_max_m:
            Sensor max range
        lidar_unit_scale:
            Convert integers to meters (dm -> m)
        lidar_out_of_range:
            if True, replace zeros with lidar_max_m
        pose_map:
            "july" (default) or "nov2026": how a pose line [ts]A,B,C is read (POSE_MAPS)
        velocity_window_s:
            0 (default): velocity and yaw rate from consecutive pose lines, as in July;
            > 0: from the pose about this long before (the vessel's raw log repeats pose lines
            in bursts about 1 ms apart, with 0.1 m steps, so consecutive differences are noise)
        """
        if pose_map not in POSE_MAPS:
            raise ValueError("pose_map must be one of %s" % sorted(POSE_MAPS))
        self.pose_map = pose_map
        self._pose_fn = POSE_MAPS[pose_map]
        self.velocity_window_s = float(velocity_window_s)
        self._pose_hist = []            # (t, x, y, unwrapped yaw) within the window
        self._yaw_unwrapped = None
        # The vessel's raw-log values, latched
        self._gyro = self._accel = self._quat = self._srvo = None
        self._imu_t = None
        self._imu_counter = None
        self.lidar_out_beams = int(lidar_out_beams)
        self.lidar_angle_offset_deg = float(lidar_angle_offset_deg)
        self.lidar_max_m = float(lidar_max_m)
        self.lidar_unit_scale = float(lidar_unit_scale)
        self.lidar_out_of_range = bool(lidar_out_of_range)

        # Time origin
        self._t0: Optional[float] = None

        # Latched values
        self._last_hdg_ref: Optional[float] = None
        self._last_seq: Optional[int] = None
        self._last_s1: Optional[int] = 0
        self._last_s2: Optional[int] = 0

        # Pose + velocity state
        self._last_pose: Optional[Tuple[float, float, float]] = None
        self._last_pose_t: Optional[float] = None
        # self._last_vel: Tuple[float, float, float] = (0, 0, 0)                # (x, y, yaw)
        # self._last_vel: Tuple[float, float, float, float] = (0, 0, 0, 0)      # add yaw_rate
        self._last_vel: Tuple[float, float, float, float] = (0, 0, 0, 0, 0, 0)  # add u and v
        self._max_spd: Optional[float] = 0

    @classmethod
    def raw_format(cls, **kw) -> "BluefinStreamDecoder":
        """A decoder for the vessel's raw log (and a stream in its format): November 2026 pose
        layout, speeds over 0.5 s.  Any argument may still be given."""
        kw.setdefault("pose_map", "nov2026")
        kw.setdefault("velocity_window_s", 0.5)
        return cls(**kw)

    # Pose + velocity state
    def _real_time(self, ts_str: str) -> float:
        """
        Convert timestamp to seconds since start of the run
        """
        sec = _ts_to_seconds(ts_str)
        if self._t0 is None:
            self._t0 = sec
        return sec - self._t0
    
    def feed(self, line: str) -> Optional[BluefinFrame]:
        """
        Feed one complete line
        Return:
            None if no frame completed
            BluefinFrame if a LiDAR scan appears
        """
        line = line.strip()
        if not line:
            return None
        
        # 0a. The vessel's IMU and servo lines (raw log)
        m = self._re_vessel.match(line)
        if m:
            try:
                vals = [float(v) for v in m.group("vals").split(",") if v != ""]
            except ValueError:
                return None
            kind = m.group("kind")
            if kind == "GYRO" and len(vals) == 3:
                self._gyro = tuple(vals)
            elif kind == "ACCEL" and len(vals) == 3:
                self._accel = tuple(vals)
            elif kind == "QUAT" and len(vals) == 4:
                self._quat = tuple(vals)
            elif kind == "SRVO" and len(vals) == 4:
                self._srvo = tuple(int(v) for v in vals)
            else:
                return None
            # Time since the run's origin, without setting it: the origin stays the first pose,
            # LiDAR, RC or HDG line, as before IMU lines were read
            sec = _ts_to_seconds(m.group("ts"))
            self._imu_t = None if self._t0 is None else sec - self._t0
            self._imu_counter = int(m.group("counter"))
            return None

        # 0. RC line
        m = self._re_rc.match(line)
        if m:
            self._last_seq = int(m.group("seq"))
            self._last_s1 = int(m.group("s1"))
            self._last_s2 = int(m.group("s2"))
            return None
        
        # 1. HDG line
        m = self._re_hdg.match(line)
        if m:
            self._last_seq = int(m.group("seq"))
            self._last_hdg_ref = float(m.group("hdg"))
            return None
        
        # 2. Pose line (updates pose + velocity)
        m = self._re_pose.match(line)
        if m:
            ts_str = m.group("ts")
            t = self._real_time(ts_str)

            x, y, yaw_deg = self._pose_fn(float(m.group("y")), float(m.group("x")), float(m.group("yaw")))

            # Velocity over a window (raw-log streams): from the pose about velocity_window_s ago
            if self.velocity_window_s > 0:
                if self._yaw_unwrapped is None:
                    self._yaw_unwrapped = yaw_deg
                else:
                    self._yaw_unwrapped += _wrap_180(yaw_deg - self._last_pose[2])
                self._pose_hist.append((t, x, y, self._yaw_unwrapped))
                while len(self._pose_hist) > 2 and t - self._pose_hist[1][0] >= self.velocity_window_s:
                    self._pose_hist.pop(0)
                t0, x0, y0, h0 = self._pose_hist[0]
                dt = t - t0
                if dt >= 0.5 * self.velocity_window_s:
                    vx = (x - x0) / dt
                    vy = (y - y0) / dt
                    spd = np.hypot(vx, vy)
                    yaw_rate = (self._yaw_unwrapped - h0) / dt
                    yaw_rad = np.deg2rad(yaw_deg)
                    u_body = vx * np.sin(yaw_rad) + vy * np.cos(yaw_rad)
                    v_body = -vx * np.cos(yaw_rad) + vy * np.sin(yaw_rad)
                    self._last_vel = (vx, vy, spd, u_body, v_body, yaw_rate)
                    if spd > self._max_spd:
                        self._max_spd = spd
                        print(f"Max velocity: {spd:.3f} m/s")
                self._last_pose = (x, y, yaw_deg)
                self._last_pose_t = t
                return None

            # Velocity and heading rate
            if self._last_pose is not None and self._last_pose_t is not None:
                dt = t - self._last_pose_t
                if dt > 1e-6:
                    prev_x, prev_y, prev_yaw = self._last_pose
                    vx = (x - prev_x) / dt
                    vy = (y - prev_y) / dt
                    spd = np.hypot(vx, vy)
                    yaw_rate = (_wrap_180(yaw_deg - prev_yaw)) / dt
                    yaw_rad = np.deg2rad(yaw_deg)

                    # body frame velocity
                    u_body = vx * np.sin(yaw_rad) + vy * np.cos(yaw_rad)
                    v_body = -vx * np.cos(yaw_rad) + vy * np.sin(yaw_rad)

                    self._last_vel = (vx, vy, spd, u_body, v_body, yaw_rate)                  

                    # show max speed
                    if spd > self._max_spd:
                        self._max_spd = spd
                        print(f"Max velocity: {spd:.3f} m/s")
            
            self._last_pose = (x, y, yaw_deg)
            self._last_pose_t = t
            return None
        
        # 3. LiDAR line
        m = self._re_lidar.match(line)
        if m:
            ts_str = m.group("ts")
            t = self._real_time(ts_str)

            lidar_int = _parse_int_list_csv(m.group("body"))
            lidar_m = lidar_int.astype(np.float32) * self.lidar_unit_scale
            lidar_m = lidar_m[::-1]     # reverse lidar scan direction (clockwise)

            if self.lidar_out_of_range:
                lidar_m = np.where(lidar_m <= 0, self.lidar_max_m, lidar_m)
            
            lidar_m = np.clip(lidar_m, 0, self.lidar_max_m)
            lidar_m = _rotate_lidar_by_degrees(lidar_m, self.lidar_angle_offset_deg)
            lidar_m = _downsample_stride(lidar_m, self.lidar_out_beams)

            if self._last_pose is None:
                return None
            
            x, y, yaw_deg = self._last_pose
            vx, vy, spd, u_body, v_body, yaw_rate = self._last_vel

            return BluefinFrame(
                t_sec = t,
                ts_str = ts_str,
                x_m = x,
                y_m = y,
                yaw_deg = yaw_deg,
                vx_mps = vx,
                vy_mps = vy,
                speed_mps = spd,
                yaw_rate = yaw_rate,
                u_body_mps = u_body,
                v_body_mps = v_body,
                lidar_m = lidar_m,
                hdg_ref_deg = self._last_hdg_ref,
                s1 = self._last_s1,
                s2 = self._last_s2,
                seq = self._last_seq,
                gyro_dps = self._gyro,
                accel_mps2 = self._accel,
                quat_xyzw = self._quat,
                srvo = self._srvo,
                imu_t_sec = self._imu_t,
                imu_counter_ms = self._imu_counter
            )
        return None
        
# Offline decoder from a log file
def frames_from_file(filepath: str, decoder: Optional[BluefinStreamDecoder] = None):
    if decoder is None:
        decoder = BluefinStreamDecoder()
    
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            frame = decoder.feed(line)
            if frame is not None:
                yield frame

# Convert to RL observation dict format
def frame_to_gym_obs(
        frame: BluefinFrame,
        *,
        origin_xyh: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
    """
    Convert frame -> Gym MultiInput dict
    If origin_xyh is provided, set (x, y, yaw) as (0,0,0)
    """
    x = frame.x_m
    y = frame.y_m
    yaw = frame.yaw_deg
    dhdg = frame.yaw_rate

    if origin_xyh is not None:
        x0, y0, yaw0 = origin_xyh
        x -= x0
        y -= y0
        yaw = _wrap_360(yaw - yaw0)
    
    obs: Dict[str, Any] = {
        "lidar": frame.lidar_m.astype(np.float32),
        "pos": np.array([x,y], dtype=np.float32),
        "hdg": np.array([yaw], dtype=np.float32),
        "dhdg": np.array([dhdg], dtype=np.float32),
        "speed": np.array([frame.speed_mps], dtype=np.float32),
        "tgt": np.array([0.0], dtype=np.float32),
        "target_heading": np.array([0.0], dtype=np.float32),
    }        

    return obs

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
        
        obs = frame_to_gym_obs(frame, origin_xyh=origin)

        if count:
            print(f"Frame {count}: t={frame.t_sec:.3f}s pos={obs['pos']} yaw={obs['hdg']} dhdg={obs['dhdg']} spd={obs['speed']}")
            print(f" lidar shape: {obs['lidar'].shape}, min/max: {obs['lidar'].min():.2f}/{obs['lidar'].max():.2f}")
            time.sleep(0.1)
        count += 1
    
    print(f"Decoded {count} frames.")