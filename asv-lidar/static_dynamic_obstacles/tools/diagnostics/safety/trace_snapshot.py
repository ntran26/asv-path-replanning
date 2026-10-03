"""Read-only decision snapshots for offline safety diagnostics.

Call ``capture_before(env)`` immediately before step and ``capture_decision(env)``
after it. The latter reads the filter's already-built decision snapshot, never
runs perception a second time, and never reads the next frame as the decision
input. Truth geometry is explicitly separated for offline scoring only.
No reset, physics update, sensor draw, association, or controller call occurs.
Headings are labelled with their units; nonfinite floats become JSON strings.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
import sys


def json_safe(value):
    """Detach arrays/containers into strict-JSON-compatible Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if is_dataclass(value):
        return {f.name: json_safe(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)) or type(value).__name__ == "deque":
        return [json_safe(v) for v in value]
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def _attributes(obj, names):
    return {name: json_safe(getattr(obj, name, None)) for name in names}


def _track(track):
    row = _attributes(track, (
        "id", "position", "velocity", "state", "cov", "hits", "misses",
        "age", "max_coast", "is_dynamic", "confirmed", "evidence_run",
        "quiet_run", "last_fit_centre", "last_fit_heading_deg",
        "fit_offset", "fit_offset_known", "_pending", "_pending_steps",
    ))
    evidence = getattr(track, "last_evidence", None)
    row["evidence"] = (None if evidence is None else _attributes(evidence, (
        "appear", "vacate", "compared", "violations", "moving")))
    row["history"] = [
        {"scan_serial": int(serial), "points": json_safe(points)}
        for serial, points in getattr(track, "history", ())
    ]
    return row


def _scan(scan):
    return _attributes(scan, (
        "serial", "origin", "heading_deg", "ranges", "points", "max_range"))


def pre_env_snapshot(env):
    """Detach the held onboard inputs and separately labelled pre-step truth."""
    tracker = getattr(env, "tracker", None)
    lidar = getattr(env, "lidar", None)
    cfg = sys.modules.get("constants")
    truth_targets = []
    for index, target in enumerate(getattr(env, "targets", ())):
        truth_targets.append(dict(
            index=index,
            **_attributes(target, ("x", "y", "heading", "speed", "velocity")),
            hull=json_safe(target.hull()),
        ))
    return {
        "schema": 1,
        "timing": "immediately before env.step; held observation used by next decision",
        "elapsed_time": json_safe(getattr(env, "elapsed_time", None)),
        "onboard": {
            "pose_hold_xy_heading_deg": json_safe(getattr(env, "_pose_hold", None)),
            "ego_hold_u_v_yaw_deg_s": json_safe(getattr(env, "_ego_hold", None)),
            "pose_stale": bool(getattr(env, "pose_stale", False)),
            "tracker_accumulated_dt": json_safe(getattr(env, "_tracker_dt", None)),
            "scan": {
                "bearings_deg": json_safe(getattr(lidar, "bearings", None)),
                "raw_ranges": json_safe(getattr(env, "raw_ranges", None)),
                "gated_ranges": json_safe(getattr(env, "gated_ranges", None)),
            },
            "published_track_ids": [int(t.id) for t in getattr(env, "tracks", ())],
            "raw_tracks": [_track(t) for t in getattr(tracker, "tracks", ())],
            "tracker_scans": [_scan(s) for s in getattr(tracker, "_scans", ())],
            "tracker_settings": _attributes(tracker, (
                "classifier", "measurement", "max_misses", "dt")),
            "thresholds": _attributes(cfg, (
                "UPDATE_RATE", "PHYSICS_DT", "TRACK_MIN_HITS", "TRACK_MAX_MISSES",
                "TRACK_FIT_PRIOR_SPEED", "TRACK_FIT_MIN_POINTS",
                "TRACK_FIT_FULL_EXTENT_TOL_M", "MOTION_MIN_POINTS", "MOTION_MIN_FRAC",
                "DYNAMIC_PROMOTE_STEPS", "DYNAMIC_DEMOTE_STEPS",
                "LIDAR_MIN_RANGE", "LIDAR_RANGE", "LOA", "BREADTH")),
        },
        "truth_scoring_only": {
            "own": _attributes(env, (
                "asv_x", "asv_y", "asv_h", "u_body", "v_body", "asv_w")),
            "own_units": {"asv_h": "deg", "asv_w": "deg/s"},
            "own_collision_hull": json_safe(env.hull_polygon()),
            "targets": truth_targets,
            "static_polygons": json_safe(getattr(env, "obstacles", ())),
            "boundary_polygon": json_safe(getattr(env, "boundary_polygon", None)),
        },
    }


def _perception_chain(perception):
    """Inspect stored adapter diagnostics/caches, without delegated lookups."""
    chain, seen = [], set()
    while perception is not None and id(perception) not in seen:
        seen.add(id(perception))
        data = vars(perception)
        row = {"class": type(perception).__name__}
        for name in ("frames", "_frames", "_frame", "memory_frames",
                     "require_motion_evidence", "last_provisional_stats",
                     "last_track_history_stats", "last_stats", "_hypotheses", "_anchors"):
            if name in data:
                row[name] = json_safe(data[name])
        chain.append(row)
        perception = data.get("base_perception")
    return chain


def post_filter_snapshot(filt):
    """Read the previous decision's saved snapshot even after env perception advances."""
    if filt is None:
        return None
    snap = getattr(filt, "_observer_snapshot", None)
    snapshot = None
    if snap is not None:
        snapshot = _attributes(snap, (
            "x", "y", "heading", "u", "v", "r", "tangent", "right", "centre",
            "base_heading", "lateral", "remaining", "points", "edges_a", "edges_b"))
        snapshot["units"] = {"heading": "rad", "r": "rad/s", "velocity": "m/s"}
        snapshot["tracks"] = [_attributes(t, (
            "id", "position", "velocity", "heading")) for t in snap.tracks]
    return {
        "schema": 1, "filter_class": type(filt).__name__,
        "timing": "stored filter decision snapshot; not post-physics observation",
        "snapshot": snapshot,
        "raw_ego_u_v_yaw_rad_s": json_safe(getattr(filt, "_raw_ego", None)),
        "actuators_before_decision": _attributes(getattr(filt, "_observer_actuators", None),
                                                  ("servo", "buffer", "executed")),
        "actuators_after_decision": _attributes(getattr(filt, "actuators", None),
                                                 ("servo", "buffer", "executed")),
        "perception_chain": _perception_chain(getattr(filt, "perception", None)),
        "last": json_safe(getattr(filt, "last", {})),
        "plan": json_safe(getattr(filt, "plan", None)),
    }


def capture_before(env):
    return pre_env_snapshot(env)


def capture_decision(env):
    return post_filter_snapshot(getattr(env, "_safety_v2", None))
