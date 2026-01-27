"""
Bluefin / model-scale ASV log parser (SLAM + 360° LiDAR) → Gym-style observations.

This module is designed for BOTH:
  1) Offline parsing of .log files (like test_1.log)
  2) Real-time line-by-line decoding, where the vessel sends the same text lines.

Log format (as observed in test_1.log):
  [HH:MM:SS.microseconds][xxxxx] HDG:350.75
  [HH:MM:SS.microseconds]-000.1,+000.0,+014.7
  [HH:MM:SS.microseconds][  0,  0, ... ,  0]      # 720 ints, decimeters (0 = out-of-range)
  [HH:MM:SS.microseconds][xxxxx] S1:1542 S2:1000 RC

Notes:
  - During autonomous deployment you likely won't receive the RC lines. This parser ignores them if absent.
  - A 'frame' is emitted on each LiDAR scan line (LiDAR is treated as the main observation clock).
  - Pose + HDG are fused as "latest values seen before the scan".

Author: generated for Nam_PhD / Bluefin deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, IO, Iterator, Optional, Tuple, Union

import numpy as np
import re


_HDG_RE = re.compile(r'^\[(?P<t>\d\d:\d\d:\d\d\.\d+)\]\[(?P<pkt>\s*\d+)\]\s*HDG:(?P<hdg>-?\d+(?:\.\d+)?)\s*$')
_POSE_RE = re.compile(
    r'^\[(?P<t>\d\d:\d\d:\d\d\.\d+)\]'
    r'(?P<x>[+-]?\d+(?:\.\d+)?),'
    r'(?P<y>[+-]?\d+(?:\.\d+)?),'
    r'(?P<yaw>[+-]?\d+(?:\.\d+)?)\s*$'
)
_LIDAR_RE = re.compile(r'^\[(?P<t>\d\d:\d\d:\d\d\.\d+)\]\[(?P<arr>.*)\]\s*$')
_RC_RE = re.compile(
    r'^\[(?P<t>\d\d:\d\d:\d\d\.\d+)\]\[(?P<pkt>\s*\d+)\]\s*S1:(?P<s1>\d+)\s+S2:(?P<s2>\d+)\s+RC\s*$'
)


def _parse_time(t_str: str) -> datetime:
    return datetime.strptime(t_str, "%H:%M:%S.%f")


def wrap_deg(angle_deg: float) -> float:
    """Wrap angle to [0, 360)."""
    return (angle_deg % 360.0 + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference a-b in degrees, result in [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


@dataclass
class BluefinFrame:
    """One fused timestep (LiDAR + latest pose + latest hdg + optional RC)."""
    t_str: str
    t_sec: float
    pose_xyh: Tuple[float, float, float]   # (x, y, yaw_deg_raw) from SLAM
    hdg_ref_deg: Optional[float]
    lidar_m: np.ndarray
    s1: Optional[int] = None
    s2: Optional[int] = None
    pose_t_sec: Optional[float] = None
    hdg_t_sec: Optional[float] = None
    rc_t_sec: Optional[float] = None


class BluefinStreamDecoder:
    """
    Stateful, line-by-line decoder.

    Use this for real-time deployment: call feed(line) for each received text line.
    When a LiDAR line arrives, feed() returns a BluefinFrame; otherwise returns None.
    """

    def __init__(
        self,
        *,
        lidar_max_m: float = 16.0,
        lidar_zero_is_out_of_range: bool = True,
        lidar_angle_step_deg: float = 0.5,
        lidar_swath_deg: float = 360.0,
        lidar_front_angle_deg: float = 0.0,
        lidar_angle_offset_deg: float = 0.0,
        lidar_out_beams: int = 720,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.lidar_max_m = float(lidar_max_m)
        self.lidar_zero_is_out_of_range = bool(lidar_zero_is_out_of_range)
        self.lidar_angle_step_deg = float(lidar_angle_step_deg)
        self.lidar_swath_deg = float(lidar_swath_deg)
        self.lidar_front_angle_deg = float(lidar_front_angle_deg)
        self.lidar_angle_offset_deg = float(lidar_angle_offset_deg)
        self.lidar_out_beams = int(lidar_out_beams)
        self.dtype = dtype

        self.raw_beams = int(round(360.0 / self.lidar_angle_step_deg))
        if self.raw_beams <= 0:
            raise ValueError("Invalid lidar_angle_step_deg; produced non-positive raw_beams")

        self._t0_dt: Optional[datetime] = None

        self._last_pose: Optional[Tuple[float, float, float]] = None
        self._last_pose_t: Optional[float] = None

        self._last_hdg: Optional[float] = None
        self._last_hdg_t: Optional[float] = None

        self._last_s1: Optional[int] = None
        self._last_s2: Optional[int] = None
        self._last_rc_t: Optional[float] = None

    def _to_t_sec(self, t_str: str) -> float:
        dt = _parse_time(t_str)
        if self._t0_dt is None:
            self._t0_dt = dt
        return (dt - self._t0_dt).total_seconds()

    def feed(self, line: str) -> Optional[BluefinFrame]:
        """Feed one line. Returns a BluefinFrame only when a LiDAR scan line arrives."""
        line = line.strip()
        if not line:
            return None

        m = _HDG_RE.match(line)
        if m:
            t_str = m.group("t")
            self._last_hdg = float(m.group("hdg"))
            self._last_hdg_t = self._to_t_sec(t_str)
            return None

        m = _RC_RE.match(line)
        if m:
            t_str = m.group("t")
            self._last_s1 = int(m.group("s1"))
            self._last_s2 = int(m.group("s2"))
            self._last_rc_t = self._to_t_sec(t_str)
            return None

        m = _POSE_RE.match(line)
        if m:
            t_str = m.group("t")
            self._last_pose = (float(m.group("x")), float(m.group("y")), float(m.group("yaw")))
            self._last_pose_t = self._to_t_sec(t_str)
            return None

        m = _LIDAR_RE.match(line)
        if m:
            t_str = m.group("t")
            t_sec = self._to_t_sec(t_str)

            if self._last_pose is None:
                return None

            arr_str = m.group("arr")
            parts = [p.strip() for p in arr_str.split(",") if p.strip() != ""]
            raw = np.fromiter((int(p) for p in parts), dtype=np.int32, count=len(parts))
            if raw.size != self.raw_beams:
                raise ValueError(f"Expected {self.raw_beams} lidar beams, got {raw.size} at t={t_str}")

            lidar_m = raw.astype(np.float32) / 10.0
            if self.lidar_zero_is_out_of_range:
                lidar_m = np.where(lidar_m == 0.0, self.lidar_max_m, lidar_m)
            lidar_m = np.clip(lidar_m, 0.0, self.lidar_max_m)

            # Rotate scan if needed
            if abs(self.lidar_angle_offset_deg) > 1e-9:
                shift = int(round(self.lidar_angle_offset_deg / self.lidar_angle_step_deg)) % self.raw_beams
                lidar_m = np.roll(lidar_m, shift)

            # Swath selection
            if self.lidar_swath_deg < 360.0 - 1e-9:
                half = self.lidar_swath_deg / 2.0
                angles = (np.arange(self.raw_beams) * self.lidar_angle_step_deg)
                rel = (angles - self.lidar_front_angle_deg + 360.0) % 360.0
                mask = (rel <= half) | (rel >= 360.0 - half)
                selected = lidar_m[mask]
            else:
                selected = lidar_m

            # Downsample
            if selected.size == self.lidar_out_beams:
                lidar_out = selected.astype(self.dtype, copy=False)
            else:
                idx = np.linspace(0, selected.size - 1, self.lidar_out_beams).round().astype(int)
                lidar_out = selected[idx].astype(self.dtype, copy=False)

            return BluefinFrame(
                t_str=t_str,
                t_sec=t_sec,
                pose_xyh=self._last_pose,
                hdg_ref_deg=self._last_hdg,
                lidar_m=lidar_out,
                s1=self._last_s1,
                s2=self._last_s2,
                pose_t_sec=self._last_pose_t,
                hdg_t_sec=self._last_hdg_t,
                rc_t_sec=self._last_rc_t,
            )

        # Unknown line → ignore
        return None


class BluefinLogParser(BluefinStreamDecoder):
    """Thin wrapper around BluefinStreamDecoder to provide file iteration helpers."""

    def iter_frames(self, fp: IO[str]) -> Iterator[BluefinFrame]:
        for raw_line in fp:
            fr = self.feed(raw_line)
            if fr is not None:
                yield fr

    def frames_from_file(self, path: Union[str, "os.PathLike[str]"]) -> Iterator[BluefinFrame]:
        with open(path, "r", errors="ignore") as f:
            yield from self.iter_frames(f)


def frame_to_gym_obs(
    frame: BluefinFrame,
    *,
    origin_xyh: Optional[Tuple[float, float, float]] = None,
    prev_yaw_deg: Optional[float] = None,
    prev_t_sec: Optional[float] = None,
    path_xy: Optional[np.ndarray] = None,
    goal_xy: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert a frame to a Gym-style observation dict.

    Returns keys:
      lidar: (N,)
      pos: (2,)
      hdg: (1,)   [0, 360)
      dhdg: (1,)  deg/s
      tgt: (1,)   distance-to-path (m) OR 0 if no path provided
      target_heading: (1,) heading error to goal (deg) OR 0 if no goal provided
    """
    x, y, yaw_raw = frame.pose_xyh
    yaw = wrap_deg(yaw_raw)

    # Origin shift
    if origin_xyh is None:
        ox, oy, oyaw = 0.0, 0.0, 0.0
    else:
        ox, oy, oyaw = origin_xyh
    x_rel = x - ox
    y_rel = y - oy
    yaw_rel = wrap_deg(yaw - oyaw)

    # Yaw rate (deg/s)
    dhdg = 0.0
    if (prev_yaw_deg is not None) and (prev_t_sec is not None) and (frame.t_sec > prev_t_sec):
        dyaw = angle_diff_deg(yaw_rel, prev_yaw_deg)
        dhdg = dyaw / (frame.t_sec - prev_t_sec)

    # Distance to path
    tgt = 0.0
    if path_xy is not None and len(path_xy) > 0:
        p = np.array([x_rel, y_rel], dtype=np.float64)
        d = np.linalg.norm(path_xy - p[None, :], axis=1)
        tgt = float(np.min(d))

    # Target heading error to goal
    target_heading = 0.0
    if goal_xy is not None:
        gx, gy = goal_xy
        dx = gx - x_rel
        dy = gy - y_rel
        # This assumes X points East and Y points North (ENU).
        # If your SLAM uses a different axis convention, adjust the atan2 arguments here.
        target_angle = np.degrees(np.arctan2(dx, dy))
        target_heading = float(angle_diff_deg(target_angle, yaw_rel))

    return {
        "lidar": frame.lidar_m.astype(np.float32, copy=False),
        "pos": np.array([x_rel, y_rel], dtype=np.float32),
        "hdg": np.array([yaw_rel], dtype=np.float32),
        "dhdg": np.array([dhdg], dtype=np.float32),
        "tgt": np.array([tgt], dtype=np.float32),
        "target_heading": np.array([target_heading], dtype=np.float32),
    }
