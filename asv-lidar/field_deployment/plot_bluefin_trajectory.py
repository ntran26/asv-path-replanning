"""
Plot trajectory from Bluefin/model-scale ASV log.

Default behavior (recommended for checking SLAM consistency):
  - Plot the SLAM trajectory (x,y) after applying an origin offset.
  - Optionally overlay RC inputs as color-coded segments or separate subplots.

Usage examples:
  python plot_bluefin_trajectory.py --log test_1.log
  python plot_bluefin_trajectory.py --log test_1.log --origin first
  python plot_bluefin_trajectory.py --log test_1.log --origin "0,0,0"
  python plot_bluefin_trajectory.py --log test_1.log --show_controls

Outputs:
  - trajectory.png
  - (optional) controls.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from bluefin_log_parser import BluefinLogParser


def parse_origin(origin_arg: str) -> Optional[Tuple[float, float, float]]:
    if origin_arg.lower() == "none":
        return None
    if origin_arg.lower() == "first":
        return "first"  # sentinel
    parts = [p.strip() for p in origin_arg.split(",")]
    if len(parts) != 3:
        raise ValueError('origin must be "first", "none", or "x,y,yaw"')
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to log file")
    ap.add_argument("--origin", default="first",
                    help='Origin definition: "first" (default), "none", or "x,y,yaw" in SLAM units')
    ap.add_argument("--out", default="trajectory.png", help="Output PNG path")
    ap.add_argument("--show_controls", action="store_true",
                    help="Also generate a controls plot (S1/S2 over time) if RC lines exist")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    parser = BluefinLogParser()
    frames = list(parser.frames_from_file(log_path))

    if len(frames) == 0:
        raise RuntimeError("No frames found (no LiDAR lines parsed).")

    # Extract pose series aligned to frames
    t = np.array([fr.t_sec for fr in frames], dtype=np.float64)
    x = np.array([fr.pose_xyh[0] for fr in frames], dtype=np.float64)
    y = np.array([fr.pose_xyh[1] for fr in frames], dtype=np.float64)
    yaw = np.array([fr.pose_xyh[2] for fr in frames], dtype=np.float64)

    origin = parse_origin(args.origin)
    if origin == "first":
        ox, oy, oyaw = x[0], y[0], yaw[0]
    elif origin is None:
        ox, oy, oyaw = 0.0, 0.0, 0.0
    else:
        ox, oy, oyaw = origin

    x_rel = x - ox
    y_rel = y - oy

    fig = plt.figure()
    plt.plot(x_rel, y_rel, linewidth=1)
    plt.scatter([0], [0], s=30, marker="x")
    plt.title("Trajectory (origin-shifted)")
    plt.xlabel("X (m, relative)")
    plt.ylabel("Y (m, relative)")
    plt.axis("equal")
    plt.grid(True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.out}")

    if args.show_controls:
        # Note: RC lines are not at every timestep. We just plot available samples.
        rc_t = np.array([fr.rc_t_sec for fr in frames if fr.rc_t_sec is not None], dtype=np.float64)
        s1 = np.array([fr.s1 for fr in frames if fr.rc_t_sec is not None], dtype=np.float64)
        s2 = np.array([fr.s2 for fr in frames if fr.rc_t_sec is not None], dtype=np.float64)
        if rc_t.size > 0:
            fig2 = plt.figure()
            plt.plot(rc_t, s1, label="S1 (rudder)")
            plt.plot(rc_t, s2, label="S2 (thruster)")
            plt.title("RC inputs over time")
            plt.xlabel("t (s)")
            plt.ylabel("PWM / raw units")
            plt.grid(True)
            plt.legend()
            out2 = Path(args.out).with_name("controls.png")
            fig2.savefig(out2, dpi=200, bbox_inches="tight")
            print(f"Saved: {out2}")
        else:
            print("No RC samples found to plot (S1/S2 lines absent).")


if __name__ == "__main__":
    main()
