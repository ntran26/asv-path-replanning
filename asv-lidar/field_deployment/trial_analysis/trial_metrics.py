"""Per-run field metrics, the C-16 rule checks, perception (F5) and the shadow layer (C-20).

Definitions follow the simulation's, so that field and simulation are compared like for like
(FIELD_TRIALS_PLAN.md section 6):

* clearances, path efficiency, rudder rate, saturation and control effort as
  `tools/tiers/common.run_episode` computes them (`src/metrics.py`): the own ship as the
  environment's metric hull (`ASVLidarEnv.hull_polygon`, length and beam plus the hull margin),
  the target as its traced outline, over the states the simulator scores, from the first after
  the start to the goal capture (or the last decision under the policy);
* the rule conditions of Section 3.4.3 by `tools/diagnostics/colregs_compliance.episode`
  itself, fed the field tracks on a uniform 0.5 s grid (claim C-16);
* the goal test of `ASVLidarEnv._reached_goal`, reported beside the operator's outcome.

Field-only choices (registered with the trial pre-registration):

* speed is the ground speed of the own track by centred differences over 1 s (the simulator
  reports the true speed);
* a rudder reversal is a sign change of the issued rudder between consecutive decisions with
  both commands at least 5 % of full scale;
* a target track is the tracker's dynamic track nearest the target's true position within the
  simulator's pairing radius (`TARGET_MATCH_RADIUS`); any other dynamic track is a phantom;
* the true encounter class is the simulator's `cls_true` (`EncounterContexts.attach_truth`):
  the classifier on the true positions and motion, with the own ship's course taken as the
  reference path's at its position (decision A19), compared with the perceived class of the
  target track's engaged context where the true geometry has a class (the simulator's own
  measure); engaged decisions whose true geometry has none (a context held after the passing)
  are counted apart;
* the target is in the scan when its true position is inside the map boundary and within the
  LiDAR range of the own ship;
* the closest approach is reached when the true range has a minimum before the end of the
  log and before any operator stop.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

import trial_io as tio
from trial_io import DT, TrialInputs, cfg

import boundary_raycast as br  # noqa: E402
import colregs_compliance as cc  # noqa: E402
import encounter as enc  # noqa: E402
import metrics as pm  # noqa: E402
import targets as tgt  # noqa: E402
from ship import HULL_MARGIN, VESSEL_LENGTH, VESSEL_WIDTH  # noqa: E402

ASSOC_GATE_M = float(cfg.TARGET_MATCH_RADIUS)   # the simulator's track-to-target pairing radius
REVERSAL_MIN_PCT = 5.0
TRACK_HEADING_MIN_SPEED = 0.10


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def own_hull(x: float, y: float, heading_deg: float):
    """`ASVLidarEnv.hull_polygon`: the rectangle the simulator's metrics and contacts use."""
    half_l = 0.5 * (VESSEL_LENGTH + 2.0 * HULL_MARGIN)
    half_w = 0.5 * (VESSEL_WIDTH + 2.0 * HULL_MARGIN)
    h = math.radians(heading_deg)
    sin_h, cos_h = math.sin(h), math.cos(h)
    return [(x + fwd * sin_h - lat * cos_h, y + fwd * cos_h + lat * sin_h)
            for fwd, lat in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]


def boundary_clearance(hull, boundary: np.ndarray) -> float:
    """`ASVLidarEnv._hull_boundary_distance`: hull to the boundary polygon, 0 once a corner is out."""
    xs = np.asarray([p[0] for p in hull])
    ys = np.asarray([p[1] for p in hull])
    poly = [tuple(map(float, p)) for p in boundary]
    if not bool(np.all(br.points_in_polygon(xs, ys, poly))):
        return 0.0
    return float(np.min(br.points_boundary_distance(xs, ys, poly)))


def project(path: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Arc length of the nearest path point for each position, and the path length."""
    a, ab = path[:-1], path[1:] - path[:-1]
    seg = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    p = np.stack([x, y], axis=1)[:, None, :]
    t = np.clip(((p - a) * ab).sum(axis=2) / np.maximum(seg ** 2, 1e-12), 0.0, 1.0)
    rel = p - (a + t[..., None] * ab)
    k = np.argmin((rel * rel).sum(axis=2), axis=1)
    s = cum[k] + t[np.arange(len(k)), k] * seg[k]
    return s, float(cum[-1])


def path_course(path: np.ndarray, x: float, y: float) -> float:
    """Compass course of the path segment nearest a position (A19's classification heading)."""
    a, ab = path[:-1], path[1:] - path[:-1]
    t = np.clip(((np.array([x, y]) - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    k = int(np.argmin(np.linalg.norm(np.array([x, y]) - (a + t[:, None] * ab), axis=1)))
    return float(np.degrees(np.arctan2(ab[k, 0], ab[k, 1])) % 360.0)


def reached_goal(path: np.ndarray, goal: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`ASVLidarEnv._reached_goal` at each position."""
    cte, _ = cc.path_frame(path, x, y)
    s, length = project(path, x, y)
    tangent = path[-1] - path[-2]
    tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
    out = np.zeros(len(x), dtype=bool)
    for k in range(len(x)):
        if math.hypot(x[k] - goal[0], y[k] - goal[1]) <= cfg.GOAL_RADIUS:
            out[k] = True
            continue
        delta = np.array([x[k], y[k]]) - path[-1]
        if float(delta @ tangent) > 0.0 and float(np.linalg.norm(delta)) > cfg.GOAL_CTE_RADIUS:
            continue
        out[k] = (length - s[k]) <= cfg.GOAL_ALONG_DIST and abs(cte[k]) <= cfg.GOAL_CTE_RADIUS
    return out


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def scored_window(run, goal_flags: np.ndarray) -> tuple:
    """States the simulator scores (after the start, to the goal capture or the last decision
    under the policy) and the decisions whose actions led there."""
    span = run.policy_span()
    k_goal = int(np.argmax(goal_flags[:span])) if goal_flags[:span].any() else None
    end = k_goal if k_goal is not None else span - 1
    states = np.arange(1, end + 1) if end >= 1 else np.array([0])
    actions = np.arange(0, end if k_goal is not None else span)
    return states, actions, k_goal


def run_metrics(inp: TrialInputs) -> Dict:
    run, sheet = inp.run, inp.sheet
    own = tio.own_track(run, inp.own_gt)
    times = run.t
    ox, oy, oh, ou = own.at(times)
    goal_flags = reached_goal(run.path, run.goal, ox, oy)
    states, acts, k_goal = scored_window(run, goal_flags)
    hulls = {k: own_hull(ox[k], oy[k], oh[k]) for k in states}
    panels = [[tuple(map(float, p)) for p in poly] for poly in run.panels]
    obs_clear = [pm.min_obstacle_clearance(hulls[k], panels) for k in states] if panels else []
    bnd_clear = [boundary_clearance(hulls[k], run.boundary) for k in states]
    cte, _ = cc.path_frame(run.path, ox[states], oy[states])
    span = run.policy_span()
    a0 = np.clip(run.action[acts, 0], -1.0, 1.0)
    rud = run.rudder_pct[acts]
    big = np.abs(rud) >= REVERSAL_MIN_PCT
    reversals = int(np.sum(big[1:] & big[:-1] & (np.sign(rud[1:]) != np.sign(rud[:-1]))))
    last = int(states[-1])
    length = pm._polyline_length(np.stack([ox[:last + 1], oy[:last + 1]], axis=1))
    path_len = pm._polyline_length(run.path)
    row = {
        "run_id": run.run_id, "scenario_id": sheet.get("scenario_id", ""), "family": sheet["family"],
        "encounter": sheet["encounter"], "layout": sheet["layout"], "repeat": sheet.get("repeat", ""),
        "valid": bool(sheet["valid"]), "outcome": sheet["outcome"], "decisions": run.n,
        "policy_decisions": span, "operator_stop": span < run.n, "scored_states": len(states),
        "duration_s": float(times[-1]),
        "goal_geometric": k_goal is not None,
        "completion_time_s": float(times[k_goal]) if k_goal is not None else float("nan"),
        "min_obstacle_clearance": float(np.min(obs_clear)) if obs_clear else float("nan"),
        "min_boundary_clearance": float(np.min(bnd_clear)),
        "rms_cte": float(np.sqrt(np.mean(np.square(cte)))),
        "path_length_m": length,
        "path_efficiency": length / path_len if path_len > 1e-6 else float("nan"),
        "mean_speed": float(np.mean(ou[states])),
        "min_speed": float(np.min(ou[states])),
        "mean_abs_rudder_rate": (float(np.mean(np.abs(np.diff(a0 * pm.MAX_RUD_ANGLE)))) / DT
                                 if len(a0) > 1 else 0.0),
        "rudder_saturation_fraction": float(np.mean(np.abs(a0) >= pm.SATURATION_THRESHOLD)) if len(a0) else float("nan"),
        "control_effort": float(np.sum(np.square(a0)) * DT),
        "rudder_reversals": reversals,
        "own_track_source": "ground truth" if inp.own_gt is not None else "registered pose",
    }
    row["contact_panel_geometric"] = bool(obs_clear) and row["min_obstacle_clearance"] <= 0.0
    row["contact_boundary_geometric"] = row["min_boundary_clearance"] <= 0.0
    row.update(target_metrics(inp, ox, oy, oh, states))
    row.update(shadow_metrics(inp))
    return row


def target_metrics(inp: TrialInputs, ox, oy, oh, states) -> Dict:
    nan = float("nan")
    out = {"min_target_range": nan, "t_closest_s": nan, "min_target_clearance": nan,
           "contact_target_geometric": False, "target_truth_coverage": nan}
    if inp.target is None:
        return out
    run = inp.run
    tx, ty, th, _ = inp.target.at(run.t)
    rng = np.hypot(tx - ox, ty - oy)
    k = int(states[int(np.argmin(rng[states]))])
    clear = [pm.polygon_distance(own_hull(ox[i], oy[i], oh[i]), tgt.hull_polygon(tx[i], ty[i], th[i]))
             for i in states]
    out.update(min_target_range=float(rng[k]), t_closest_s=float(run.t[k]),
               min_target_clearance=float(np.min(clear)), contact_target_geometric=bool(np.min(clear) <= 0.0),
               target_truth_coverage=float(np.mean(inp.target.covers(run.t))))
    return out


def shadow_metrics(inp: TrialInputs) -> Dict:
    """The shadow layer's would-be interventions (C-20), reported apart from the policy."""
    run = inp.run
    recs = [(k, r) for k, r in enumerate(run.shadow) if r]
    whys = Counter(str(r.get("why", "")) for _, r in recs if r.get("changed"))
    committed = next((k for k, r in recs if r.get("committed") is not None), None)
    changed = sum(bool(r.get("changed")) for _, r in recs)
    return {"shadow_decisions": len(recs), "shadow_would_change": changed,
            "shadow_would_change_rate": changed / len(recs) if recs else float("nan"),
            "shadow_brake_requests": sum(bool(r.get("brake")) for _, r in recs),
            "shadow_stop_committed": committed is not None,
            "shadow_stop_time_s": float(run.t[committed]) if committed is not None else float("nan"),
            "shadow_change_reasons": "; ".join(f"{w} {n}" for w, n in whys.most_common())}


# ---------------------------------------------------------------------------
# Encounter behaviour and the rule conditions (F2, C-16)
# ---------------------------------------------------------------------------
def encounter_behaviour(inp: TrialInputs) -> Optional[Dict]:
    run, sheet = inp.run, inp.sheet
    cls, side = tio.ENCOUNTER_CLASS[sheet["encounter"]]
    if cls == "no_target" or inp.target is None:
        return None
    span = run.policy_span()
    grid = tio.uniform_grid(run, end=run.t[span - 1])
    own = tio.own_track(run, inp.own_gt)
    ox, oy, oh, ou = own.at(grid)
    tx, ty, th, _ = inp.target.at(grid)
    dec = tio.nearest_decision(run, grid)
    engaged = np.array([any(bool(c.get("engaged")) for c in run.contexts[i].values()) for i in dec])
    tr = {"own_x": ox, "own_y": oy, "own_h": oh, "own_u": ou, "tgt_x": tx, "tgt_y": ty, "tgt_h": th,
          "engaged": engaged, "path": run.path}
    row = {"class": cls, "crossing_side": side, "outcome": tio.SIM_OUTCOME[sheet["outcome"]],
           "variant": "FIX", "test_id": run.run_id}
    out = dict(cc.episode(row, tr))
    out["run_id"] = out.pop("test_id")

    # Extras for F2: when and how far the first alteration went, and the speed reduction.
    rng = np.hypot(tx - ox, ty - oy)
    k_cpa = int(np.argmin(rng))
    k0 = min(int(np.argmax(engaged)) if engaged.any() else 0, k_cpa)
    y_own, bear = cc.path_frame(run.path, ox, oy)
    hi, lo = max(y_own[k0], 0.0), min(y_own[k0], 0.0)
    depart = np.maximum(np.maximum(y_own - hi, lo - y_own), 0.0)
    first = next((k for k in range(k0, k_cpa + 1) if depart[k] >= cc.LATERAL_M), None)
    cog_dev = []
    for k in range(max(k0, cc.COG_STEPS), k_cpa + 1):
        dx, dy = ox[k] - ox[k - cc.COG_STEPS], oy[k] - oy[k - cc.COG_STEPS]
        if math.hypot(dx, dy) > 0.05:
            cog_dev.append(abs(float(tio.wrap180(math.degrees(math.atan2(dx, dy)) - bear[k]))))
    # Reached without an operator stop: the minimum over the whole logged run (target truth
    # after a stop included) lies inside the policy's span and before the end of the log.
    full = tio.own_track(run, inp.own_gt)
    gx, gy, _, _ = full.at(run.t)
    fx, fy, _, _ = inp.target.at(run.t)
    k_min_full = int(np.argmin(np.hypot(fx - gx, fy - gy)))
    reached = (k_min_full < run.n - 1) and (k_min_full < span)
    out.update(engaged_ever=bool(engaged.any()), engagement_time_s=float(k0 * DT) if engaged.any() else float("nan"),
               first_alteration_after_engagement_s=float((first - k0) * DT) if first is not None else float("nan"),
               max_course_alteration_deg=float(max(cog_dev)) if cog_dev else float("nan"),
               speed_at_engagement=float(ou[k0]),
               speed_reduction_mps=float(ou[k0] - np.min(ou[k0:k_cpa + 1])),
               reached_cpa_without_stop=bool(reached), encounter=sheet["encounter"], family=sheet["family"],
               layout=sheet["layout"], valid=bool(sheet["valid"]), outcome=sheet["outcome"])
    return out


def c16_verdict(behaviour_rows: List[Dict]) -> Dict:
    """C-16: in each core encounter type of family B, at least 2 of the runs that reach the
    closest approach without an operator stop pass the rule conditions (registered with 3 runs
    per type)."""
    per = {}
    for enc_type in tio.CORE_ENCOUNTERS:
        rows = [r for r in behaviour_rows if r["family"] == "B" and r["encounter"] == enc_type and r["valid"]]
        eligible = [r for r in rows if r["reached_cpa_without_stop"]]
        passes = sum(1 for r in eligible if r.get("rule_compliant") == 1.0)
        per[enc_type] = {"runs": len(rows), "eligible": len(eligible), "pass": passes, "met": passes >= 2}
    return {"per_type": per, "met": all(v["met"] for v in per.values()),
            "rule": "at least 2 eligible runs pass in each of the four core types"}


# ---------------------------------------------------------------------------
# Perception on the water (F5)
# ---------------------------------------------------------------------------
def perception(inp: TrialInputs) -> Dict:
    run, sheet = inp.run, inp.sheet
    own = tio.own_track(run, inp.own_gt)
    ox, oy, oh, ou = own.at(run.t)
    poly = [tuple(map(float, p)) for p in run.boundary]
    has_target = inp.target is not None and sheet["encounter"] != "none"
    if has_target:
        tx, ty, th, ts = inp.target.at(run.t)
        covered = inp.target.covers(run.t)
        inside = br.points_in_polygon(np.asarray(tx), np.asarray(ty), poly)
        in_scan = covered & inside & (np.hypot(tx - ox, ty - oy) <= float(cfg.LIDAR_RANGE))
        k_enter = int(np.argmax(in_scan)) if in_scan.any() else None
    else:
        in_scan, k_enter = np.zeros(run.n, dtype=bool), None
    phantom, eligible, assoc_k, pos_err, head_err, agree, truth_none = [], [], [], [], [], [], 0
    for k in range(run.n):
        moving = [t for t in run.tracks[k] if t.get("dynamic")]
        target_track = None
        if has_target and in_scan[k] and moving:
            d = [math.hypot(t["position"][0] - tx[k], t["position"][1] - ty[k]) for t in moving]
            j = int(np.argmin(d))
            if d[j] <= ASSOC_GATE_M:
                target_track = moving[j]
        others = [t for t in moving if t is not target_track]
        is_eligible = sheet["family"] == "A" or not has_target or (k_enter is None or k < k_enter)
        if is_eligible:
            eligible.append(k)
            phantom.append(bool(others))
        if target_track is None:
            continue
        if target_track.get("confirmed"):
            assoc_k.append(k)
        pos_err.append(math.hypot(target_track["position"][0] - tx[k], target_track["position"][1] - ty[k]))
        if ts[k] >= TRACK_HEADING_MIN_SPEED and float(target_track.get("speed") or 0.0) >= TRACK_HEADING_MIN_SPEED:
            head_err.append(abs(float(tio.wrap180(float(target_track["course_deg"]) - th[k]))))
        ctx = run.contexts[k].get(str(target_track["id"]))
        if ctx and ctx.get("engaged"):
            truth = enc.classify((ox[k], oy[k]), path_course(run.path, ox[k], oy[k]), ou[k],
                                 (tx[k], ty[k]), th[k], ts[k])
            if truth == enc.NONE:
                truth_none += 1
            else:
                agree.append(str(ctx.get("cls")) == str(truth))
    nan = float("nan")
    delay = (float(run.t[assoc_k[0]] - run.t[k_enter]) if (k_enter is not None and assoc_k) else nan)
    return {"run_id": run.run_id, "family": sheet["family"], "encounter": sheet["encounter"],
            "layout": sheet["layout"], "valid": bool(sheet["valid"]),
            "target_entered_scan_s": float(run.t[k_enter]) if k_enter is not None else nan,
            "track_confirmation_delay_s": delay,
            "target_tracked_share": (len(assoc_k) / int(in_scan.sum())) if in_scan.any() else nan,
            "track_position_error_median_m": float(np.median(pos_err)) if pos_err else nan,
            "track_position_error_p90_m": float(np.percentile(pos_err, 90)) if pos_err else nan,
            "track_heading_error_median_deg": float(np.median(head_err)) if head_err else nan,
            "classification_agreement": float(np.mean(agree)) if agree else nan,
            "classification_decisions": len(agree),
            "engaged_truth_none_decisions": truth_none,
            "phantom_eligible_decisions": len(eligible),
            "phantom_share": float(np.mean(phantom)) if phantom else nan,
            "decisions_with_moving_track": float(np.mean([any(t.get("dynamic") for t in tr) for tr in run.tracks]))}
