"""Loaders for the trial analysis: bridge run logs, the run sheet and the ground-truth tracks.

Design: `static_dynamic_obstacles/planning/FIELD_TRIALS_PLAN.md` sections 5-7 and
`PAPER3_CLAIMS_AND_TABLES.md` (claims C-15 to C-20, tables F1-F5).  The analysis reads files
only; it opens no socket.

Session folder
--------------
    <session>/run_sheet.csv
    <session>/runs/<run_id>/decisions.jsonl      written by udp_live_p3.py
    <session>/runs/<run_id>/shadow.jsonl         the shadow layer's decisions (D-4)
    <session>/ground_truth/<run_id>_target.csv   target ground truth (decision D-2)
    <session>/ground_truth/<run_id>_own.csv      optional; else the registered pose of the log

Run sheet columns (one row per run, invalid runs kept and flagged, FIELD_TRIALS_PLAN.md
section 8): run_id, session, scenario_id, family (A, B, C), encounter (none, head_on,
crossing_starboard, crossing_port, overtaking, being_overtaken), layout (L1, L2, L3, open),
repeat, valid (1 or 0), outcome (goal, contact_panel, contact_boundary, contact_target,
operator_stop, timeout), notes.  The outcome is the operator's record; the geometric checks of
`trial_metrics.py` are reported beside it, not in its place.

Ground-truth track columns: one time column, `t_run` (s from the run's first decision) or
`t_unix` (UTC seconds, matched through the log's clock), then `x`, `y` (m, the registered map
frame) and optionally `heading_deg` (compass).  Without a heading column the heading is the
direction of motion, held while the speed is below 0.05 m/s.  An optional `speed` column (m/s)
replaces the speed from position differences.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
FIELD_DIR = HERE.parent
SIM_ROOT = FIELD_DIR.parent / "static_dynamic_obstacles"
SIM_SRC = SIM_ROOT / "src"

# The simulator's sources first (field_deployment carries Paper 2 modules of the same names);
# the diagnostics, the identified model and the bridge after them.
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))
for _extra in (SIM_ROOT / "tools" / "diagnostics", SIM_ROOT / "bluefin", FIELD_DIR):
    if str(_extra) not in sys.path:
        sys.path.append(str(_extra))

import constants as cfg  # noqa: E402

DT = float(cfg.UPDATE_RATE)

FAMILIES = ("A", "B", "C")
ENCOUNTERS = ("none", "head_on", "crossing_starboard", "crossing_port", "overtaking", "being_overtaken")
CORE_ENCOUNTERS = ("head_on", "crossing_starboard", "crossing_port", "overtaking")   # C-16
OUTCOMES = ("goal", "contact_panel", "contact_boundary", "contact_target", "operator_stop", "timeout")
LAYOUTS = ("L1", "L2", "L3", "open")
# encounter -> (simulator class, crossing side), as `tools/diagnostics/colregs_compliance.py` reads them
ENCOUNTER_CLASS = {"none": ("no_target", ""), "head_on": ("head_on", ""),
                   "crossing_starboard": ("crossing", "starboard"), "crossing_port": ("crossing", "port"),
                   "overtaking": ("overtaking", ""), "being_overtaken": ("being_overtaken", "")}
# encounter -> test set v4 cell code (`L1-HO`, ...)
CELL_CODE = {"head_on": "HO", "crossing_starboard": "CRS", "crossing_port": "CRP",
             "overtaking": "OT", "being_overtaken": "BO", "none": "NT"}
# the operator's outcome -> the simulator's outcome string
SIM_OUTCOME = {"goal": "goal", "contact_panel": "collision:obstacle", "contact_boundary": "collision:boundary",
               "contact_target": "collision:target", "operator_stop": "operator_stop", "timeout": "timeout"}


def load_module(name: str, path: Path):
    """Import a file under a unique name (the tiers' `common.py` would clash otherwise)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wrap180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Run logs
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class FieldRun:
    """One bridge run, at its decisions."""

    run_id: str
    folder: Path
    header: Dict
    path: np.ndarray            # (M, 2) the reference path
    boundary: np.ndarray        # (B, 2) the map boundary polygon
    panels: List[np.ndarray]    # panel polygons
    start: np.ndarray
    goal: np.ndarray
    t: np.ndarray               # s from the first decision (telemetry frame time)
    t_unix: np.ndarray          # UTC s of each decision's release; NaN without the clock record
    pose: np.ndarray            # (n, 3) x, y, heading deg: the estimate perception read
    ego: np.ndarray             # (n, 3) measured surge, sway (+ starboard), yaw rate deg/s
    action: np.ndarray          # (n, 2) the policy's action
    rudder_pct: np.ndarray      # issued rudder, simulator convention, percent
    vessel_rudder: np.ndarray   # issued rudder, the vessel's convention (`$CMD` field)
    rpm: np.ndarray             # issued propulsion, rpm-units
    s2: np.ndarray
    source: List[str]           # "policy", "layer" (active mode) or the operator's stop
    operator_stop: List[str]
    tracks: List[List[Dict]]
    contexts: List[Dict[str, Dict]]
    shadow: List[Optional[Dict]]  # the layer's record per decision, None where absent
    observation: List[Dict]       # the 70-value observation by branch
    command_latency: np.ndarray   # s from the LiDAR line to the sent command (NaN if not logged)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def safety_mode(self) -> str:
        return str((self.header.get("safety") or {}).get("mode", ""))

    def policy_span(self) -> int:
        """Decisions before the first command not from the policy (an operator stop)."""
        for k, src in enumerate(self.source):
            if src not in ("policy", "layer"):
                return k
        return self.n


def load_run(folder: Path) -> FieldRun:
    folder = Path(folder)
    rows = read_jsonl(folder / "decisions.jsonl")
    header = next(r for r in rows if r.get("type") == "header")
    dec = sorted((r for r in rows if r.get("type") == "decision"), key=lambda r: r["k"])
    if not dec:
        raise ValueError(f"{folder}: no decisions")
    shadow_by_k: Dict[int, Dict] = {}
    if (folder / "shadow.jsonl").exists():
        for r in read_jsonl(folder / "shadow.jsonl"):
            if r.get("type") == "shadow":
                shadow_by_k[int(r["k"])] = r.get("safety")
    geo = header["geometry"]
    t_frame = np.array([float(r["t_frame"]) for r in dec])
    wall0, mono0 = header.get("wall_utc_at_start"), header.get("monotonic_at_start")
    t_unix = np.array([(float(wall0) + float(r["clock"]["t_wall_released"]) - float(mono0))
                       if (wall0 is not None and mono0 is not None and r.get("clock")) else np.nan
                       for r in dec])

    def pose_of(r):
        p = (r.get("perception") or {}).get("estimated_pose")
        return p if p is not None else r["sensor"]["pose"]

    def ego_of(r):
        e = (r.get("perception") or {}).get("measured_ego")
        return e if e is not None else r["sensor"]["ego"]

    return FieldRun(
        run_id=str(header.get("run_id", folder.name)), folder=folder, header=header,
        path=np.asarray(geo["path"], dtype=float), boundary=np.asarray(geo["boundary_polygon"], dtype=float),
        panels=[np.asarray(p, dtype=float) for p in (geo.get("panels") or [])],
        start=np.asarray(geo["start"], dtype=float), goal=np.asarray(geo["goal"], dtype=float),
        t=t_frame - t_frame[0], t_unix=t_unix,
        pose=np.array([pose_of(r) for r in dec], dtype=float),
        ego=np.array([ego_of(r) for r in dec], dtype=float),
        action=np.array([r["action"] for r in dec], dtype=float),
        rudder_pct=np.array([r["command"]["rudder_pct"] for r in dec], dtype=float),
        vessel_rudder=np.array([r["command"]["vessel_rudder"] for r in dec], dtype=float),
        rpm=np.array([r["command"]["rpm"] for r in dec], dtype=float),
        s2=np.array([r["command"]["s2"] for r in dec], dtype=float),
        source=[str(r["command"].get("source", "policy")) for r in dec],
        operator_stop=[str((r.get("operator_stop") or {}).get("state", "")) for r in dec],
        tracks=[list((r.get("perception") or {}).get("tracks") or []) for r in dec],
        contexts=[dict((r.get("perception") or {}).get("contexts") or {}) for r in dec],
        shadow=[r.get("safety") if r.get("safety") is not None else shadow_by_k.get(int(r["k"])) for r in dec],
        observation=[r.get("observation") or {} for r in dec],
        command_latency=np.array([float((r.get("seconds") or {}).get("lidar_line_to_sent", np.nan) or np.nan)
                                  for r in dec]),
    )


# ---------------------------------------------------------------------------
# Run sheet
# ---------------------------------------------------------------------------
def load_run_sheet(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for r in rows:
        r["family"] = r["family"].strip().upper()
        r["encounter"] = r["encounter"].strip().lower()
        r["layout"] = r["layout"].strip()
        r["outcome"] = r["outcome"].strip().lower()
        r["valid"] = str(r.get("valid", "1")).strip() not in ("0", "false", "False", "no", "")
        allowed = {"family": FAMILIES, "encounter": ENCOUNTERS, "outcome": OUTCOMES, "layout": LAYOUTS}
        problems = [f"{key} {r[key]!r}" for key, values in allowed.items() if r[key] not in values]
        if problems:
            raise ValueError(f"run sheet {path}, run {r.get('run_id')}: unknown " + ", ".join(problems))
    return rows


def sim_cell(sheet_row: Dict) -> Optional[str]:
    """The test set v4 cell of a family C run (`L1-HO`, ...); None for the other families."""
    if sheet_row["family"] != "C" or sheet_row["layout"] not in ("L1", "L2", "L3"):
        return None
    return f"{sheet_row['layout']}-{CELL_CODE[sheet_row['encounter']]}"


# ---------------------------------------------------------------------------
# Ground-truth tracks
# ---------------------------------------------------------------------------
@dataclass
class Track:
    """A track on the run's time base (s from the first decision), interpolated linearly."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading_unwrapped: np.ndarray   # deg, compass, unwrapped for interpolation
    speed: np.ndarray               # m/s, from the positions

    def at(self, times) -> tuple:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        x = np.interp(times, self.t, self.x)
        y = np.interp(times, self.t, self.y)
        h = np.interp(times, self.t, self.heading_unwrapped) % 360.0
        s = np.interp(times, self.t, self.speed)
        return x, y, h, s

    def covers(self, times) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return (times >= self.t[0] - 1e-9) & (times <= self.t[-1] + 1e-9)


def ground_speed(t: np.ndarray, x: np.ndarray, y: np.ndarray, half_span_s: float = 0.5) -> np.ndarray:
    """Speed over the track by centred differences spanning about 2 x `half_span_s`."""
    n = len(t)
    if n < 2:
        return np.zeros(n)
    out = np.empty(n)
    for k in range(n):
        lo = int(np.searchsorted(t, t[k] - half_span_s, side="left"))
        hi = int(np.searchsorted(t, t[k] + half_span_s, side="right")) - 1
        lo, hi = max(0, min(lo, k)), min(n - 1, max(hi, k))
        if hi == lo:
            lo, hi = (k - 1, k) if k > 0 else (k, k + 1)
        dt = t[hi] - t[lo]
        out[k] = math.hypot(x[hi] - x[lo], y[hi] - y[lo]) / dt if dt > 0 else 0.0
    return out


def make_track(t, x, y, heading_deg=None, speed=None) -> Track:
    t, x, y = (np.asarray(v, dtype=float) for v in (t, x, y))
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]
    speed = ground_speed(t, x, y) if speed is None else np.asarray(speed, dtype=float)[order]
    if heading_deg is None:
        h = np.full(len(t), np.nan)
        for k in range(len(t)):
            lo, hi = max(0, k - 1), min(len(t) - 1, k + 1)
            if speed[k] >= 0.05 and hi > lo:
                h[k] = math.degrees(math.atan2(x[hi] - x[lo], y[hi] - y[lo])) % 360.0
        valid = np.flatnonzero(np.isfinite(h))
        if len(valid) == 0:
            h[:] = 0.0
        else:
            for k in range(len(t)):                   # hold the last known heading (first one before it)
                if not np.isfinite(h[k]):
                    prev = valid[valid < k]
                    h[k] = h[prev[-1]] if len(prev) else h[valid[0]]
    else:
        h = np.asarray(heading_deg, dtype=float)[order]
    return Track(t=t, x=x, y=y, heading_unwrapped=np.degrees(np.unwrap(np.radians(h))), speed=speed)


def load_track(path: Path, run: FieldRun) -> Track:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: empty track")
    cols = rows[0].keys()
    if "t_run" in cols:
        t = np.array([float(r["t_run"]) for r in rows])
    elif "t_unix" in cols:
        if not np.isfinite(run.t_unix[0]):
            raise ValueError(f"{path}: t_unix given but run {run.run_id} has no clock record")
        t = np.array([float(r["t_unix"]) for r in rows]) - float(run.t_unix[0])
    else:
        raise ValueError(f"{path}: needs a t_run or t_unix column")
    x = [float(r["x"]) for r in rows]
    y = [float(r["y"]) for r in rows]
    h = [float(r["heading_deg"]) for r in rows] if "heading_deg" in cols else None
    speed = [float(r["speed"]) for r in rows] if "speed" in cols else None
    return make_track(t, x, y, h, speed)


def own_track(run: FieldRun, gt: Optional[Track] = None) -> Track:
    """The own ship's track: its ground truth if given, else the registered pose of the log."""
    if gt is not None:
        return gt
    return make_track(run.t, run.pose[:, 0], run.pose[:, 1], run.pose[:, 2])


def uniform_grid(run: FieldRun, end: Optional[float] = None) -> np.ndarray:
    """The decision interval as a uniform grid from the first decision (0.5 s)."""
    end = float(run.t[-1] if end is None else end)
    return np.arange(0.0, end + 1e-9, DT)


def nearest_decision(run: FieldRun, times: Sequence[float]) -> np.ndarray:
    """Index of the last decision at or before each time (the state the vessel acted on)."""
    idx = np.searchsorted(run.t, np.asarray(times, dtype=float) + 1e-9, side="right") - 1
    return np.clip(idx, 0, run.n - 1)


@dataclass
class TrialInputs:
    sheet: Dict
    run: FieldRun
    target: Optional[Track]
    own_gt: Optional[Track]


def load_session(session_dir: Path, *, require_truth: bool = False) -> List[TrialInputs]:
    """Every run of a session's run sheet with its logs and ground truth."""
    session_dir = Path(session_dir)
    out = []
    for row in load_run_sheet(session_dir / "run_sheet.csv"):
        folder = session_dir / "runs" / row["run_id"]
        if not (folder / "decisions.jsonl").exists():
            raise FileNotFoundError(f"run {row['run_id']}: no decisions.jsonl under {folder}")
        run = load_run(folder)
        gt_dir = session_dir / "ground_truth"
        target = own = None
        if (gt_dir / f"{row['run_id']}_target.csv").exists():
            target = load_track(gt_dir / f"{row['run_id']}_target.csv", run)
        elif require_truth and row["encounter"] != "none":
            raise FileNotFoundError(f"run {row['run_id']}: no target ground truth")
        if (gt_dir / f"{row['run_id']}_own.csv").exists():
            own = load_track(gt_dir / f"{row['run_id']}_own.csv", run)
        out.append(TrialInputs(sheet=row, run=run, target=target, own_gt=own))
    return out
