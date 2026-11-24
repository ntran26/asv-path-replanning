"""Reader for the February 2026 radio-control logs (`field_deployment/data/test_1.log` .. `test_4.log`).

Line kinds, as recorded by the vessel's telemetry stream (vessel time stamps, not arrival times):

    [HH:MM:SS.ffffff]f1,f2,f3                  pose, 10 Hz; x = f2, y = -f1, heading = -f3 (deg),
                                               the convention of `log_parser` and `build_dataset`
    [HH:MM:SS.ffffff][seq] HDG:350.75          a second heading channel, 10 Hz (not used here)
    [HH:MM:SS.ffffff][seq] S1:1511 S2:1467 RC  radio-control receiver values, about 1 Hz
    [HH:MM:SS.ffffff][seq] S1:1500 S2:1500 WD  the same line while no transmitter is bound
    [HH:MM:SS.ffffff][0,0,...]                 LiDAR scan, 720 integers (dm)

Radio-control channels, established from the logs (`feb_analysis.py` section 2), against the
`log_parser.py` docstring, which names S1 the rudder:

    S1  throttle, pulse width in microseconds. Neutral 1511 (at rest), forward above, reverse below
        (1007-1012 at full reverse); 1500 in WD lines.
    S2  rudder. Neutral 1467 (straight running), extremes 999-1000 and 1966-1967. S2 = 1000 gives a
        negative yaw rate in the pipeline frame, the sign a positive transmitted `$CMD` rudder gives
        in the July logs; S2 = 1967 a positive one.

`rudder_cmd_from_s2` maps S2 linearly to the transmitted `$CMD` convention under the assumption
that the receiver's end points are the `$CMD` end points (+100 at 1000, -100 at 1967). That
assumption is not established by these logs; see `REPORT_FEB2026.md`.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, Optional

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
BLUEFIN = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(BLUEFIN, "..", ".."))
DATA_DIR = os.path.join(REPO, "field_deployment", "data")
if BLUEFIN not in sys.path:
    sys.path.insert(0, BLUEFIN)

S1_NEUTRAL = 1511.0
S2_NEUTRAL = 1467.0
S2_END_NEG_YAW = 1000.0     # gives r < 0 in the pipeline frame ("port" below)
S2_END_POS_YAW = 1967.0     # gives r > 0 ("starboard")

_RE_POSE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{6})\]([-+]\d+\.\d+),([-+]\d+\.\d+),([-+]\d+\.\d+)\s*$")
_RE_RC = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{6})\]\[(\d+)\]\s*S1:(\d+)\s*S2:(\d+)\s*(RC|WD)\s*$")
_RE_HDG = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{6})\]\[(\d+)\]\s*HDG:([-+]?\d+(?:\.\d+)?)\s*$")
_RE_LIDAR = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{6})\]\[\s*\d")


def _sec(m) -> float:
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1e6


def load(k: int, data_dir: str = DATA_DIR) -> Dict:
    """Parse `test_<k>.log`. Times are seconds from the first pose line."""
    path = os.path.join(data_dir, f"test_{k}.log")
    pose, rc, wd, hdg = [], [], [], []
    n_lidar, other = 0, []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _RE_POSE.match(line)
            if m:
                pose.append((_sec(m), float(m.group(6)), -float(m.group(5)), -float(m.group(7))))
                continue
            m = _RE_RC.match(line)
            if m:
                (rc if m.group(8) == "RC" else wd).append((_sec(m), int(m.group(6)), int(m.group(7))))
                continue
            m = _RE_HDG.match(line)
            if m:
                hdg.append((_sec(m), float(m.group(6))))
                continue
            if _RE_LIDAR.match(line):
                n_lidar += 1
                continue
            other.append(line[:80])
    pose = np.array(pose)
    t0 = pose[0, 0]
    rc = np.array(rc) if rc else np.zeros((0, 3))
    wd = np.array(wd) if wd else np.zeros((0, 3))
    hdg = np.array(hdg) if hdg else np.zeros((0, 2))
    return {
        "name": f"test_{k}", "path": path,
        "t": pose[:, 0] - t0, "x": pose[:, 1], "y": pose[:, 2],
        "yaw": np.rad2deg(np.unwrap(np.deg2rad(pose[:, 3]))),
        "trc": rc[:, 0] - t0, "s1": rc[:, 1].astype(float), "s2": rc[:, 2].astype(float),
        "n_wd": len(wd), "n_lidar": n_lidar, "n_other": len(other),
        "thdg": hdg[:, 0] - t0 if len(hdg) else hdg[:, 0], "hdg": hdg[:, 1] if len(hdg) else hdg[:, 0],
    }


def kinematics(d: Dict, dt: float = 0.1, win: int = 11) -> Dict:
    """Uniform grid and Savitzky-Golay derivatives (quadratic, `win` samples).

    Body-frame velocities follow `dynamics.py`: u forward, v to starboard,
    dx = u sin(psi) + v cos(psi), dy = u cos(psi) - v sin(psi).
    """
    tg = np.arange(0.0, d["t"][-1], dt)
    x = np.interp(tg, d["t"], d["x"])
    y = np.interp(tg, d["t"], d["y"])
    yaw = np.interp(tg, d["t"], d["yaw"])
    vx = savgol_filter(x, win, 2, deriv=1, delta=dt)
    vy = savgol_filter(y, win, 2, deriv=1, delta=dt)
    r = savgol_filter(yaw, win, 2, deriv=1, delta=dt)
    psi = np.deg2rad(savgol_filter(yaw, win, 2))
    u = vx * np.sin(psi) + vy * np.cos(psi)
    v = vx * np.cos(psi) - vy * np.sin(psi)
    course = np.rad2deg(np.arctan2(vx, vy))
    drift = (course - np.rad2deg(psi) + 180.0) % 360.0 - 180.0
    out = dict(t=tg, x=x, y=y, yaw=yaw, u=u, v=v, r=r, U=np.hypot(vx, vy), drift=drift)
    out["s1"] = zoh(d["trc"], d["s1"], tg)
    out["s2"] = zoh(d["trc"], d["s2"], tg)
    return out


def zoh(t_src: np.ndarray, val: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Zero-order hold of a sampled channel; NaN before its first sample."""
    if len(t_src) == 0:
        return np.full_like(t, np.nan, dtype=float)
    idx = np.clip(np.searchsorted(t_src, t, side="right") - 1, 0, len(t_src) - 1)
    out = val[idx].astype(float)
    out[t < t_src[0]] = np.nan
    return out


def rudder_cmd_from_s2(s2) -> np.ndarray:
    """S2 to the transmitted `$CMD` rudder percent, assuming receiver end points = +/-100 %."""
    s2 = np.asarray(s2, dtype=float)
    neg = (S2_NEUTRAL - s2) / (S2_NEUTRAL - S2_END_NEG_YAW) * 100.0     # S2 < neutral -> +cmd
    pos = -(s2 - S2_NEUTRAL) / (S2_END_POS_YAW - S2_NEUTRAL) * 100.0    # S2 > neutral -> -cmd
    return np.clip(np.where(s2 <= S2_NEUTRAL, neg, pos), -100.0, 100.0)


class ThrottleMap:
    """S1 to an equivalent rpm-unit, through the measured straight-line speed.

    `points` are (S1, settled straight speed) pairs from `feb_analysis.py` section 4; the speed is
    interpolated linearly and converted to rpm-units with the identified surge balance
    (steady speed = 0.0930 m/s per rpm-unit). Below neutral the result is negative: reverse.
    """

    def __init__(self, points, speed_per_rpm: float):
        pts = sorted(points)
        self.s1 = np.array([p[0] for p in pts], dtype=float)
        self.u = np.array([p[1] for p in pts], dtype=float)
        self.k = float(speed_per_rpm)

    def speed(self, s1) -> np.ndarray:
        return np.interp(np.asarray(s1, dtype=float), self.s1, self.u)

    def rpm(self, s1) -> np.ndarray:
        s1 = np.asarray(s1, dtype=float)
        fwd = self.speed(np.maximum(s1, S1_NEUTRAL)) / self.k
        return np.where(s1 < S1_NEUTRAL - 50.0, -24.0, fwd)


def as_run(d: Dict, t_start: float, t_end: float, throttle: ThrottleMap, rate: float = 2.0,
           name: Optional[str] = None):
    """A `build_dataset.Run` over [t_start, t_end], pose decimated to `rate` Hz (July rate).

    Commands: S2 through `rudder_cmd_from_s2`, S1 through `throttle`, both as the 1 Hz samples
    (the transmitter's position between samples is unknown).
    """
    from build_dataset import Run
    m = (d["t"] >= t_start) & (d["t"] <= t_end)
    t = d["t"][m]
    keep = np.r_[0, np.where(np.diff(np.floor((t - t[0]) * rate + 1e-6)) > 0)[0] + 1]
    t, x, y, yaw = t[keep], d["x"][m][keep], d["y"][m][keep], d["yaw"][m][keep]
    mc = (d["trc"] >= t_start - 1.5) & (d["trc"] <= t_end)
    tc = d["trc"][mc]
    return Run(name=name or f"{d['name']}[{t_start:.1f}-{t_end:.1f}]", session="2026-02",
               t=t - t[0], x=x, y=y, yaw=yaw, tc=tc - t[0],
               rudder_cmd=rudder_cmd_from_s2(d["s2"][mc]), rpm_cmd=throttle.rpm(d["s1"][mc]),
               meta={"path": d["path"], "t_abs_start": float(t_start)})
