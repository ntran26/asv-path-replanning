"""
Compare a raw log written by ros2_relay.py with the vessel's own log of the same run.

    python3 ros2_relay.py --bag ~/rosbag2_2026_11_26-11_08_31 --log relay_from_bag.log --lidar-hz 0
    python ros2_relay_check.py relay_from_bag.log 2026-11-26/RAW_LOG.log --out p3_bench/relay_check/summary.json

Pose and LiDAR lines: for each line of the vessel's log inside the relay log's time span, whether a
relay line with the same text (after the stamp) lies within --window-s of its stamp (pose 0.15 s,
LiDAR 0.3 s by default); the stamp offset of the matches.  IMU: the relay's IMU (ROS /imu/data_raw)
against the vessel's GYRO lines (its on-board IMU), nearest in time: correlation, slope and rms
difference per axis; the vessel's SRVO lines have no counterpart.  Standard library and numpy only.
"""

import argparse
import bisect
import json
import re
import sys

import numpy as np

RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d\.\d{6})\](.*)$")
RE_POSE = re.compile(r"^[-+]\d+\.\d+,[-+]\d+\.\d+,[-+]\d+\.\d+$")
RE_IMU = re.compile(r"^(\d+),(GYRO|ACCEL|QUAT|SRVO),(.*)$")


def read(path):
    out = {"pose": [], "LiDAR": [], "GYRO": [], "ACCEL": [], "QUAT": [], "SRVO": []}
    last = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE.match(line.rstrip("\r\n"))
            if not m:
                continue
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            if last is not None and t < last - 43200:
                t += 86400
            last = t
            body = m.group(4).strip()
            if body.startswith("["):
                out["LiDAR"].append((t, body))
            elif RE_POSE.match(body):
                out["pose"].append((t, body))
            else:
                mi = RE_IMU.match(body)
                if mi:
                    try:
                        out[mi.group(2)].append((t, [float(v) for v in mi.group(3).split(",")]))
                    except ValueError:
                        pass
    for k in out:
        out[k].sort(key=lambda r: r[0])
    return out


def match_text(relay, vessel, window):
    """Vessel lines in the relay's span with an identical relay line within the window."""
    if not relay:
        return {"vessel_lines_in_span": 0}
    by_text = {}
    for t, body in relay:
        by_text.setdefault(body, []).append(t)
    t0, t1 = relay[0][0], relay[-1][0]
    n = ok = 0
    offsets, misses = [], []
    for t, body in vessel:
        if t < t0 or t > t1:
            continue
        n += 1
        ts = by_text.get(body)
        hit = None
        if ts:
            i = bisect.bisect_left(ts, t - window)
            cands = [u for u in ts[i:i + 8] if abs(u - t) <= window]
            if cands:
                hit = min(cands, key=lambda u: abs(u - t))
        if hit is None:
            misses.append(t)
        else:
            ok += 1
            offsets.append(t - hit)
    res = {"vessel_lines_in_span": n, "identical": ok, "share": ok / n if n else None}
    if offsets:
        o = np.array(offsets)
        res["vessel_minus_relay_stamp_s"] = {"median": float(np.median(o)), "p5": float(np.percentile(o, 5)),
                                             "p95": float(np.percentile(o, 95))}
    res["first_misses_s_of_day"] = [round(t, 3) for t in misses[:10]]
    return res


def compare_imu(relay, vessel):
    if not relay or not vessel:
        return {"relay_samples": len(relay), "vessel_samples": len(vessel)}
    tr = np.array([r[0] for r in relay])
    vr = np.array([r[1] for r in relay])
    sel = [(t, v) for t, v in vessel if tr[0] <= t <= tr[-1]]
    if len(sel) < 10:
        return {"relay_samples": len(relay), "vessel_samples_in_span": len(sel)}
    tv = np.array([s[0] for s in sel])
    vv = np.array([s[1] for s in sel])
    j = np.clip(np.searchsorted(tr, tv), 0, len(tr) - 1)
    out = {"relay_samples": len(relay), "vessel_samples_in_span": len(sel)}
    for k, ax in enumerate("xyz"):
        a, b = vv[:, k], vr[j, k]
        c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
        out[ax] = {"corr": c, "slope_vessel_on_relay": float(np.polyfit(b, a, 1)[0]) if b.std() > 0 else None,
                   "rms_diff": float(np.sqrt(np.mean((a - b) ** 2)))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("relay_log")
    ap.add_argument("vessel_log")
    ap.add_argument("--pose-window-s", type=float, default=0.15)
    ap.add_argument("--lidar-window-s", type=float, default=0.3)
    ap.add_argument("--out", default=None, help="write the summary to this JSON file")
    args = ap.parse_args()
    R, V = read(args.relay_log), read(args.vessel_log)
    summary = {
        "relay_log": args.relay_log, "vessel_log": args.vessel_log,
        "relay_lines": {k: len(v) for k, v in R.items()}, "vessel_lines": {k: len(v) for k, v in V.items()},
        "pose": match_text(R["pose"], V["pose"], args.pose_window_s),
        "lidar": match_text(R["LiDAR"], V["LiDAR"], args.lidar_window_s),
        "gyro_deg_s": compare_imu(R["GYRO"], V["GYRO"]),
        "accel_m_s2": compare_imu(R["ACCEL"], V["ACCEL"]),
        "note": "IMU: relay = ROS /imu/data_raw, vessel = its on-board IMU (no ROS topic); SRVO has no ROS source",
    }
    for k in ("pose", "lidar"):
        s = summary[k]
        if s.get("vessel_lines_in_span"):
            print("%-5s %6d of %6d vessel lines identical (%.2f %%), stamp offset median %+.3f s"
                  % (k, s["identical"], s["vessel_lines_in_span"], 100 * s["share"],
                     s.get("vessel_minus_relay_stamp_s", {}).get("median", float("nan"))))
    for k in ("gyro_deg_s", "accel_m_s2"):
        s = summary[k]
        if "z" in s:
            print("%-10s corr x %+.3f  y %+.3f  z %+.3f; z slope %.3f, rms diff %.3f" % (
                k, s["x"]["corr"] or 0, s["y"]["corr"] or 0, s["z"]["corr"] or 0,
                s["z"]["slope_vessel_on_relay"] or 0, s["z"]["rms_diff"]))
    print("vessel SRVO lines (no ROS source): %d" % len(V["SRVO"]))
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)
        print("written", args.out)


if __name__ == "__main__":
    sys.exit(main())
