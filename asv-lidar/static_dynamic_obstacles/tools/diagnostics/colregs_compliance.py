"""Per-rule COLREGs compliance from the recorded true trajectories (2026-10-08).

Reads a test-set run's `episodes.csv` and `trajectories_off.npz` (written by
`tools/tiers/test_set.py`) and writes `compliance.csv` beside them, one row per episode
with a target encounter. Every encounter in test set v4 carries a risk of collision
(drawn DCPA below d_req = 2.5 m), so every rule check applies.

Manoeuvres are judged on the track, not the instantaneous heading, so path following and
heading oscillation do not count as alterations. The lateral departure is measured in the
reference-path frame (cross-track position, positive to starboard) beyond the band between
the path and the offset held at the encounter's onset (the first engaged step, else the
start): an encounter that begins off the path and returns towards it has not altered
course; one that moves past the path, or further out, has. About 5-8 % of encounters
begin 0.5 m or more off the path (seed-0 SAC and PPO).

* first manoeuvre: the side of the first lateral departure of 0.5 m or more before the
  CPA ("none" if there is none);
* Rule 14 (head-on): no collision, port-to-port passing (target on own ship's port side at
  the CPA, in the path frame), first manoeuvre to starboard or none; against a
  non-compliant target (variant NC, which does not keep to Rule 14 itself) port-to-port
  passing may be neither possible nor safe, so those episodes are judged on Rules 2(b) and
  8 instead: no collision and CPA >= d_req;
* Rules 15-16 (crossing from starboard): no collision, does not cross ahead of the target,
  first manoeuvre to starboard or none;
* crossing from port: no collision and does not cross ahead (the turn direction differs
  between the paper's Rule 9(b) reading, give way and pass astern, and the open-water
  reading, stand on with no port turn, Rule 17(c));
* Rule 13 (overtaking): no collision and CPA >= d_req (keeps clear, either side, so the
  apparent-action and first-action checks below accept either side);
* Rule 17 (being overtaken): no collision, lateral departure within 1.0 m and speed
  within 0.15 m/s of the onset speed until the target is within d_req or the CPA;
* Rule 8 (give-way classes): safe passing distance (CPA >= d_req); readily apparent action
  (course made good, over 2 s, at least 20 deg off the path in the compliant sense while
  on that side of the band, or a 30 % slowdown); the true TCPA when the first compliant
  manoeuvre (0.5 m lateral departure or 0.1 m/s slower) starts.

    python tools/diagnostics/colregs_compliance.py sacs0_bl3_metrics los_dwa ...
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import constants as cfg  # noqa: E402

D_REQ = 2.0 * float(cfg.DOMAIN_LATERAL)
DT = float(cfg.UPDATE_RATE)
LATERAL_M = 0.5           # a manoeuvre: this much lateral departure beyond the band
HOLD_LATERAL_M, HOLD_U = 1.0, 0.15
APPARENT_DEG, APPARENT_SLOW = 20.0, 0.3 * float(cfg.U_REF)
SLOW_MPS = 0.1
COG_STEPS = 4             # course made good over 2 s


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def path_frame(path: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Signed cross-track offset (+ starboard) and path bearing at each point."""
    a, ab = path[:-1], path[1:] - path[:-1]                         # segments (M, 2)
    p = np.stack([x, y], axis=1)[:, None, :]                        # points (N, 1, 2)
    L2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((p - a) * ab).sum(axis=2) / L2, 0.0, 1.0)          # (N, M)
    rel = p - (a + t[..., None] * ab)
    k = np.argmin((rel * rel).sum(axis=2), axis=1)                  # nearest segment per point
    h = np.arctan2(ab[k, 0], ab[k, 1])                              # compass bearing (+y north, clockwise)
    r = rel[np.arange(len(k)), k]
    return r[:, 0] * np.cos(h) - r[:, 1] * np.sin(h), np.degrees(h)


def episode(row, tr) -> dict:
    cls, side = str(row["class"]), str(row["crossing_side"])
    collided = str(row["outcome"]).startswith("collision")
    ox, oy, ou = tr["own_x"].astype(float), tr["own_y"].astype(float), tr["own_u"].astype(float)
    tx, ty, th = tr["tgt_x"].astype(float), tr["tgt_y"].astype(float), tr["tgt_h"].astype(float)
    eng, path = tr["engaged"], tr["path"].astype(float)
    n = len(ox)
    rng = np.hypot(tx - ox, ty - oy)
    k_cpa = int(np.argmin(rng)); cpa = float(rng[k_cpa])
    k0 = int(np.argmax(eng)) if eng.any() else 0
    k0 = min(k0, k_cpa)
    y_own, bear = path_frame(path, ox, oy)
    y_tgt, _ = path_frame(path, tx, ty)
    hi, lo = max(y_own[k0], 0.0), min(y_own[k0], 0.0)              # band: path to onset offset
    stbd, port = y_own - hi, lo - y_own                             # departure beyond the band
    depart = np.maximum(np.maximum(stbd, port), 0.0)
    win = range(k0, k_cpa + 1)
    first = next((k for k in win if depart[k] >= LATERAL_M), None)
    first_sense = "none" if first is None else ("starboard" if stbd[first] > 0 else "port")
    target_side = "starboard" if y_tgt[k_cpa] - y_own[k_cpa] > 0 else "port"
    # crossing ahead: own ship crosses the target's track line while in front of the target
    tdir = np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], axis=1)
    d = np.stack([ox - tx, oy - ty], axis=1)
    lateral = tdir[:, 0] * d[:, 1] - tdir[:, 1] * d[:, 0]
    along = (d * tdir).sum(axis=1)
    flips = np.flatnonzero(np.sign(lateral[1:]) * np.sign(lateral[:-1]) < 0) + 1
    crossed_ahead = any(along[k] > 0.0 for k in flips if k <= k_cpa + 1)
    nc = str(row.get("variant", "")) == "NC"
    out = {"test_id": row["test_id"], "rule": "", "target_behaviour": str(row.get("variant", "")),
           "rule_compliant": np.nan, "first_manoeuvre": first_sense,
           "target_side_at_cpa": target_side, "crossed_ahead": float(crossed_ahead),
           "max_lateral_m": float(depart[k0:k_cpa + 1].max()),
           "cpa_m": cpa, "safe_passing": float(cpa >= D_REQ and not collided),
           "apparent_action": np.nan, "tcpa_first_action_s": np.nan}
    ok = not collided
    if cls == "head_on" and nc:
        out["rule"] = "head-on, non-compliant target"
        ok = ok and cpa >= D_REQ
    elif cls == "head_on":
        out["rule"] = "Rule 14"
        ok = ok and target_side == "port" and first_sense in ("starboard", "none")
    elif cls == "crossing":
        out["rule"] = "Rules 15-16" if side == "starboard" else "crossing from port"
        ok = ok and not crossed_ahead
        if side == "starboard":
            ok = ok and first_sense in ("starboard", "none")
    elif cls == "overtaking":
        out["rule"] = "Rule 13"
        ok = ok and cpa >= D_REQ
    elif cls == "being_overtaken":
        out["rule"] = "Rule 17"
        stop = next((k for k in range(k0, n) if rng[k] < D_REQ), k_cpa)
        seg = slice(k0, max(stop, k0) + 1)
        ok = ok and depart[seg].max() <= HOLD_LATERAL_M and np.abs(ou[seg] - ou[k0]).max() <= HOLD_U
    else:
        return out
    out["rule_compliant"] = float(bool(ok))
    if cls in ("head_on", "crossing", "overtaking"):
        # compliant sense: starboard (head-on, crossing from starboard), port (crossing from
        # port, passing astern), either (overtaking; the head-on with a non-compliant target)
        sense = (0.0 if cls == "overtaking" or nc else
                 1.0 if (cls == "head_on" or side == "starboard") else -1.0)
        cog = np.full(n, np.nan)
        for k in range(COG_STEPS, n):
            ddx, ddy = ox[k] - ox[k - COG_STEPS], oy[k] - oy[k - COG_STEPS]
            if math.hypot(ddx, ddy) > 0.05:
                cog[k] = math.degrees(math.atan2(ddx, ddy))
        dev = wrap180(cog - bear)
        on_s, on_p = y_own >= hi, y_own <= lo                       # not returning from the other side
        alter = (np.where(on_s, dev, -np.inf) if sense > 0 else np.where(on_p, -dev, -np.inf) if sense < 0
                 else np.maximum(np.where(on_s, dev, -np.inf), np.where(on_p, -dev, -np.inf)))
        w = slice(k0, k_cpa + 1)
        slow = ou[k0] - ou[w]
        out["apparent_action"] = float(np.nanmax(np.r_[alter[w], -np.inf]) >= APPARENT_DEG or slow.max() >= APPARENT_SLOW)
        lat = depart if sense == 0.0 else stbd if sense > 0 else port
        acted = next((k for k in win if lat[k] >= LATERAL_M or ou[k0] - ou[k] >= SLOW_MPS), None)
        if acted is not None and acted > 0:
            v = np.array([tx[acted] - tx[acted - 1] - (ox[acted] - ox[acted - 1]),
                          ty[acted] - ty[acted - 1] - (oy[acted] - oy[acted - 1])]) / DT
            p = np.array([tx[acted] - ox[acted], ty[acted] - oy[acted]])
            vv = float(v @ v)
            out["tcpa_first_action_s"] = float(-(p @ v) / vv) if vv > 1e-9 else np.nan
    return out


def run(tag: str) -> Path:
    folder = ROOT / "results" / "test_set" / "v4" / tag
    eps = pd.read_csv(folder / "episodes.csv", keep_default_na=False)
    eps = eps[eps.safety == "off"]
    eps.loc[(eps["class"] == "") & eps.cell.str.contains("null"), "class"] = "null"
    npz = np.load(folder / "trajectories_off.npz")
    trajs: dict = {}
    for key in npz.files:
        tid, field = key.split("|", 1)
        trajs.setdefault(tid, {})[field] = npz[key]
    rows = [episode(r, trajs[r["test_id"]]) for _, r in eps.iterrows()
            if r["test_id"] in trajs and "tgt_x" in trajs[r["test_id"]]
            and r["class"] in ("head_on", "crossing", "overtaking", "being_overtaken")]
    out = folder / "compliance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        print(tag, "->", run(tag), flush=True)
