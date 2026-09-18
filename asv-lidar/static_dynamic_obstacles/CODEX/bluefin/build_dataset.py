"""Build an identification dataset from the 2026-07-02 / 2026-07-03 field logs.

Only the July logs are used: they carry the extended `#ACTION` record, which
pairs the commanded rudder and RPM with the LiDAR frame timestamp.

Facts established by inspecting the logs together with `udp_live_rl.py`:

* Telemetry and control both run at 2 Hz (dt = 0.500 s), not the 10 Hz claimed
  in the `log_parser.py` docstring.
* Pose is quantised to exactly 0.1 m in x/y and 0.1 deg in yaw.
* The command actually transmitted is recoverable exactly: the `#ACTION` line
  carries the literal `$CMD,<rudder>,<thrust>` string. Checked against every
  July run, the transmitted rudder equals `limited_rudder` in 100% of samples
  and equals `raw_rudder` in only some runs — `udp_live_rl.py` carries a line
  (`rudder_cmd = float(raw_rudder_cmd)`) that disables the rate limiter, and
  the logs span versions with and without it active. So the rate limiter was
  live for most runs and the vessel received a ramped command, not the
  bang-bang policy output. The `$CMD` string is parsed directly and is the
  authoritative input; `limited_rudder` is the fallback.
* The `x_real`/`y_real`/`yaw_real` fields inside `#ACTION` are the decoder's
  *latched* pose and repeat whenever a LiDAR line arrives before the matching
  pose line. The raw pose lines themselves update cleanly at 2 Hz, so pose is
  taken from the raw stream with its own timestamps.
* `udp_live_rl.py` opens the record file in append mode, so one file can hold
  several bridge launches. Each launch restarts the vessel pose at the origin,
  so files are split at clock gaps.

Convention: x/y/yaw follow `log_parser.BluefinStreamDecoder` (y and yaw negated
relative to the raw telemetry fields).
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

_SPLIT = re.compile(r",(?=[A-Za-z_][A-Za-z_0-9]*=)")

_RE_CMD = re.compile(r"'\$CMD,(?P<rud>[-+0-9.]+),(?P<thr>[-+0-9.]+)'")

_RE_POSE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]"
    r"(?P<y>[-+]\d+\.\d+),(?P<x>[-+]\d+\.\d+),(?P<yaw>[-+]\d+\.\d+)\s*$"
)

POSE_QUANT_M = 0.1
POSE_QUANT_DEG = 0.1
NOMINAL_DT = 0.5
SPLIT_GAP_S = 5.0          # clock gap that indicates a new bridge launch
MIN_SAMPLES = 20


@dataclass
class Run:
    """One continuous field run."""
    name: str
    session: str
    # pose stream (2 Hz, own timestamps)
    t: np.ndarray            # s from run start
    x: np.ndarray            # m
    y: np.ndarray            # m
    yaw: np.ndarray          # deg, unwrapped
    # command stream (2 Hz, own timestamps)
    tc: np.ndarray           # s from run start
    rudder_cmd: np.ndarray   # percent, transmitted ($CMD field 1)
    rpm_cmd: np.ndarray      # commanded propeller rpm
    meta: Dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def rudder_at(self, t) -> np.ndarray:
        """Zero-order hold of the commanded rudder at arbitrary times."""
        idx = np.searchsorted(self.tc, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.tc) - 1)
        return self.rudder_cmd[idx]

    def rpm_at(self, t) -> np.ndarray:
        idx = np.searchsorted(self.tc, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.tc) - 1)
        return self.rpm_cmd[idx]

    def path_length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))


def _ts_to_seconds(ts: str) -> float:
    hh, mm, rest = ts.split(":")
    ss, us = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(us) / 1e6


def _parse_kv(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in _SPLIT.split(body.strip()):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_segments(path: str) -> List[Run]:
    pose: List[tuple] = []      # (t_abs, x, y, yaw)
    cmd: List[tuple] = []       # (t_abs, rudder, rpm)
    config: Dict[str, str] = {}

    for line in open(path, errors="ignore"):
        line = line.rstrip("\n")
        if line.startswith("#CONFIG"):
            config = _parse_kv(line[len("#CONFIG,"):])
            continue
        if line.startswith("#ACTION"):
            d = _parse_kv(line[len("#ACTION,"):])
            if "ts" not in d:
                continue
            rud = None
            m = _RE_CMD.match(d.get("cmd", ""))
            if m:
                rud = float(m.group("rud"))
            elif "limited_rudder" in d:
                rud = float(d["limited_rudder"])
            elif "rudder" in d:
                rud = float(d["rudder"])
            if rud is None:
                continue
            try:
                cmd.append((_ts_to_seconds(d["ts"]), rud, float(d.get("rpm", "nan"))))
            except ValueError:
                continue
            continue
        m = _RE_POSE.match(line.strip())
        if m:
            pose.append((_ts_to_seconds(m.group("ts")),
                         float(m.group("x")),
                         -float(m.group("y")),
                         -float(m.group("yaw"))))

    if len(pose) < MIN_SAMPLES or not cmd:
        return []

    pose_a = np.array(pose)
    cmd_a = np.array(cmd)

    tp = pose_a[:, 0]
    cuts = [0] + [i for i in range(1, len(tp))
                  if (tp[i] - tp[i - 1]) > SPLIT_GAP_S or (tp[i] - tp[i - 1]) <= 0] + [len(tp)]

    runs: List[Run] = []
    multi = (len(cuts) - 1) > 1
    for k in range(len(cuts) - 1):
        seg = pose_a[cuts[k]:cuts[k + 1]]
        if len(seg) < MIN_SAMPLES:
            continue
        t0, t1 = seg[0, 0], seg[-1, 0]
        cm = cmd_a[(cmd_a[:, 0] >= t0 - 1.0) & (cmd_a[:, 0] <= t1 + 1.0)]
        if len(cm) < MIN_SAMPLES:
            continue

        session = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        name = f"{session}/{stem}" + (f"#{k+1}" if multi else "")

        yaw = np.rad2deg(np.unwrap(np.deg2rad(seg[:, 3])))
        runs.append(Run(
            name=name, session=session,
            t=seg[:, 0] - t0, x=seg[:, 1], y=seg[:, 2], yaw=yaw,
            tc=cm[:, 0] - t0, rudder_cmd=cm[:, 1], rpm_cmd=cm[:, 2],
            meta={"path": path, "config": config,
                  "t_abs_start": float(t0),
                  "dt_med": float(np.median(np.diff(seg[:, 0])))},
        ))
    return runs


def load_all(data_dir: str = "data", sessions=("2026-07-02", "2026-07-03")) -> List[Run]:
    runs: List[Run] = []
    for s in sessions:
        for p in sorted(glob.glob(os.path.join(data_dir, s, "*.log"))):
            runs.extend(load_segments(p))
    return runs


def classify(run: Run) -> str:
    """Label a run 'moving' or 'static'.

    A static run never leaves a 1 m disc: the calibration file contains a long
    held/bench period at 12 rpm with the rudder pinned, which carries no
    dynamics but is useful for drift characterisation.
    """
    span = max(run.x.max() - run.x.min(), run.y.max() - run.y.min())
    return "moving" if (span > 1.0 and run.path_length() > 2.0) else "static"


if __name__ == "__main__":
    runs = load_all()
    print(f"{'run':26s} {'kind':7s} {'n':>4} {'dur':>6} {'dt':>5} {'rpm':>10} "
          f"{'rudRMS':>7} {'path_m':>7} {'yawspan':>8}")
    tot = 0
    for r in runs:
        print(f"{r.name:26s} {classify(r):7s} {r.n:4d} {r.duration:6.1f} "
              f"{r.meta['dt_med']:5.2f} {np.nanmin(r.rpm_cmd):4.1f}-{np.nanmax(r.rpm_cmd):4.1f} "
              f"{np.sqrt(np.mean(r.rudder_cmd**2)):7.1f} {r.path_length():7.1f} "
              f"{r.yaw.max()-r.yaw.min():8.1f}")
        tot += r.n
    mov = [r for r in runs if classify(r) == "moving"]
    print(f"\n{len(runs)} segments ({len(mov)} moving), {tot} pose samples, "
          f"{sum(r.duration for r in mov)/60:.1f} min of vessel motion")
