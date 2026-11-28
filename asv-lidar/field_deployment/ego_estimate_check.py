"""
Own-motion estimates for the bridge, against a reference, on a full-precision raw log.

The policy's ego input is surge u, sway v and yaw rate r at each 0.5 s decision.  The simulator
gave it the true values with noise of 0.05 m/s and 1.0 deg/s (F46), never a delay or an offset.
This compares causal estimates of u, v and r, as a bridge could form them at a decision, with a
non-causal reference from the same log: full-precision poses differenced over +-0.25 s around
the instant (no lag; smoothed over 0.5 s).

    python3 ros2_relay.py --bag ~/rosbag2_2026_11_26-11_08_31 --log full.log --pose-decimals 4 --imu-hz 0 --stamp header
    python ego_estimate_check.py full.log --out p3_bench/ego_estimate_check

Estimates:
    pose 0.1 m, 0.5 s   the bridge today: the vessel's pose lines (0.1 m, 0.1 deg), newest pose
                         against the one nearest 0.5 s older (udp_live_p3.EgoWindow)
    pose full, 0.5 s    the same from full-precision poses (the relay's --pose-decimals 3)
    pose full, 0.25 s   full precision, a 0.25 s window
    gyro                (r only) GYRO z minus its at-rest bias, mean of the samples in the last
                         0.1 s (also: the last sample, the last 0.25 s)

For each: error against the reference at the same instant (mean, sd, rms, 95th percentile of
|error|), while moving and at rest, and the delay that best aligns it with the reference.  The
log's pose and IMU stamps are the messages' own (--stamp header).  Needs numpy; matplotlib for
the figure.  The frame is the log's: A, B = map x, y (m), C = yaw (deg, counter-clockwise).
"""

import argparse
import json
import os
import re
import sys

import numpy as np

SIM_NOISE = {"u": 0.05, "v": 0.05, "r": 1.0}          # m/s, m/s, deg/s (constants, F46)
RE_LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d\.\d+)\](.*)$")
RE_POSE = re.compile(r"^([-+]\d+\.\d+),([-+]\d+\.\d+),([-+]\d+\.\d+)$")
RE_GYRO = re.compile(r"^\d+,GYRO,([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)$")


def read(path):
    poses, gyro = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_LINE.match(line.strip())
            if not m:
                continue
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            body = m.group(4)
            mp = RE_POSE.match(body)
            if mp:
                poses.append((t, float(mp.group(1)), float(mp.group(2)), float(mp.group(3))))
                continue
            mg = RE_GYRO.match(body)
            if mg:
                gyro.append((t, float(mg.group(3))))
    P = np.array(sorted(poses))
    G = np.array(sorted(gyro)) if gyro else np.zeros((0, 2))
    return P, G


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def body(vx, vy, yaw_deg):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return vx * c + vy * s, -vx * s + vy * c                 # u forward, v to port


def reference(P, T, half=0.25):
    """Centred differences of full-precision poses over +-half s (linear interpolation)."""
    t, x, y, h = P[:, 0], P[:, 1], P[:, 2], unwrap_deg(P[:, 3])
    xa, xb = np.interp(T - half, t, x), np.interp(T + half, t, x)
    ya, yb = np.interp(T - half, t, y), np.interp(T + half, t, y)
    ha, hb = np.interp(T - half, t, h), np.interp(T + half, t, h)
    vx, vy, r = (xb - xa) / (2 * half), (yb - ya) / (2 * half), (hb - ha) / (2 * half)
    u, v = body(vx, vy, np.interp(T, t, h))
    return u, v, r


def pose_window(P, T, window, decimals=None):
    """Newest pose at or before T against the one whose age is nearest `window` (EgoWindow)."""
    t, x, y, h = P[:, 0], P[:, 1].copy(), P[:, 2].copy(), P[:, 3].copy()
    if decimals is not None:
        x, y, h = (np.array([float("%.*f" % (decimals, a)) for a in arr]) for arr in (x, y, h))
    hu = unwrap_deg(h)
    i = np.searchsorted(t, T, side="right") - 1
    j = np.clip(np.searchsorted(t, t[i] - window), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - (t[i] - window)) < np.abs(t[j] - (t[i] - window)), j - 1, j)
    dt = t[i] - t[j]
    ok = dt > 1e-6
    vx = np.where(ok, (x[i] - x[j]) / np.where(ok, dt, 1), np.nan)
    vy = np.where(ok, (y[i] - y[j]) / np.where(ok, dt, 1), np.nan)
    r = np.where(ok, (hu[i] - hu[j]) / np.where(ok, dt, 1), np.nan)
    u, v = body(vx, vy, h[i])
    return u, v, r


def gyro_mean(G, T, span, bias):
    """Mean of the GYRO z samples in (T - span, T], minus the bias; span 0: the last sample."""
    t, z = G[:, 0], G[:, 1] - bias
    i = np.searchsorted(t, T, side="right") - 1
    if span <= 0:
        return z[i]
    c = np.concatenate([[0.0], np.cumsum(z)])
    j = np.searchsorted(t, T - span, side="right")
    n = (i + 1) - j
    return np.where(n > 0, (c[i + 1] - c[j]) / np.maximum(n, 1), z[i])


def stats(err):
    e = err[np.isfinite(err)]
    if len(e) == 0:
        return None
    return {"n": int(len(e)), "mean": float(e.mean()), "sd": float(e.std()),
            "rms": float(np.sqrt(np.mean(e ** 2))), "median": float(np.median(e)),
            "median_abs": float(np.median(np.abs(e))), "p95_abs": float(np.percentile(np.abs(e), 95)),
            "p99_abs": float(np.percentile(np.abs(e), 99))}


def best_delay(est, ref, T, step=0.1, max_s=1.0):
    """Delay d (s) minimising rms(est(T) - ref(T - d)): > 0 means the estimate is late."""
    best = None
    for k in range(-int(max_s / step), int(max_s / step) + 1):
        d = k * step
        ok = np.isfinite(est) & (T - d >= T[0]) & (T - d <= T[-1])
        rr = np.interp(T[ok] - d, T, ref)
        e = np.median(np.abs(est[ok] - rr))          # robust to the SLAM jumps
        if best is None or e < best[1]:
            best = (d, e)
    return {"delay_s": round(best[0], 2), "median_abs_at_delay": float(best[1])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="raw log with full-precision poses and GYRO lines (ros2_relay.py --bag)")
    ap.add_argument("--step-s", type=float, default=0.1, help="evaluation instants every this many s")
    ap.add_argument("--out", default=None, help="folder for summary.json and ego_estimates.png")
    args = ap.parse_args()

    P, G = read(args.log)
    if len(P) < 100:
        sys.exit("too few pose lines in %s" % args.log)
    t0, t1 = P[0, 0] + 1.0, P[-1, 0] - 1.0
    if len(G):
        t0, t1 = max(t0, G[0, 0] + 1.0), min(t1, G[-1, 0] - 1.0)
    T = np.arange(t0, t1, args.step_s)
    u_ref, v_ref, r_ref = reference(P, T)
    speed_ref = np.hypot(u_ref, v_ref)
    rest = (speed_ref < 0.02) & (np.abs(r_ref) < 1.0)
    moving = (speed_ref > 0.1) | (np.abs(r_ref) > 3.0)

    est = {
        "pose 0.1 m, 0.5 s (today)": pose_window(P, T, 0.5, decimals=1),
        "pose full, 0.5 s": pose_window(P, T, 0.5),
        "pose full, 0.25 s": pose_window(P, T, 0.25),
    }
    gyro_info = None
    if len(G):
        g_rest = gyro_mean(G, T, 0.1, 0.0)[rest]
        bias = float(np.median(g_rest)) if rest.sum() > 50 else 0.0
        gyro_info = {"bias_deg_s": bias, "rest_instants": int(rest.sum()),
                     "noise_sd_one_sample_at_rest": float(np.std(gyro_mean(G, T, 0.0, bias)[rest])),
                     "rate_hz": float(len(G) / (G[-1, 0] - G[0, 0]))}
        for label, span in (("gyro, last sample", 0.0), ("gyro, mean 0.1 s", 0.1), ("gyro, mean 0.25 s", 0.25)):
            est[label] = (None, None, gyro_mean(G, T, span, bias))
        est["gyro, mean 0.1 s, no bias removed"] = (None, None, gyro_mean(G, T, 0.1, 0.0))
        ok = moving & np.isfinite(est["gyro, mean 0.1 s"][2])
        gyro_info["scale_vs_reference"] = float(np.polyfit(est["gyro, mean 0.1 s"][2][ok], r_ref[ok], 1)[0])

    refs = {"u": u_ref, "v": v_ref, "r": r_ref}
    # SLAM jumps: consecutive poses implying more than 3 m/s or 90 deg/s
    dt = np.diff(P[:, 0])
    okd = dt > 1e-4
    step = np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2]))[okd]
    dh = np.abs((np.diff(P[:, 3]) + 180) % 360 - 180)[okd]
    jump = (step / dt[okd] > 3.0) | (dh / dt[okd] > 90.0)
    t_jump = P[1:, 0][okd][jump]
    near = np.zeros(len(T), bool)                  # reference +-0.25 s, an estimate's window 0.5 s
    for tj in t_jump:
        near |= (T > tj - 0.3) & (T < tj + 0.8)
    clean = moving & ~near
    jumps = {"count": int(jump.sum()), "per_minute": float(jump.sum() / ((P[-1, 0] - P[0, 0]) / 60)),
             "moving_instants_near_a_jump": int((moving & near).sum()),
             "step_m_median": float(np.median(step[jump])) if jump.any() else 0.0,
             "step_m_max": float(step[jump].max()) if jump.any() else 0.0,
             "heading_deg_median": float(np.median(dh[jump])) if jump.any() else 0.0}
    out = {"log": os.path.basename(args.log), "instants": int(len(T)), "span_s": float(T[-1] - T[0]),
           "moving_instants": int(moving.sum()), "rest_instants": int(rest.sum()),
           "sim_noise": SIM_NOISE, "gyro": gyro_info, "slam_jumps": jumps, "estimates": {}}
    print("%d instants over %.0f s (moving %d, at rest %d); simulator noise: u, v %.2f m/s, r %.1f deg/s"
          % (len(T), T[-1] - T[0], moving.sum(), rest.sum(), SIM_NOISE["u"], SIM_NOISE["r"]))
    if gyro_info:
        print("gyro: %.0f Hz, bias %+.3f deg/s, one-sample sd at rest %.3f deg/s, scale against reference %.3f"
              % (gyro_info["rate_hz"], gyro_info["bias_deg_s"], gyro_info["noise_sd_one_sample_at_rest"],
                 gyro_info["scale_vs_reference"]))
    print("SLAM pose jumps (> 3 m/s or 90 deg/s between consecutive poses): %d (%.1f per minute), median step "
          "%.2f m, max %.2f m, median heading step %.1f deg" % (jumps["count"], jumps["per_minute"],
                                                             jumps["step_m_median"], jumps["step_m_max"],
                                                             jumps["heading_deg_median"]))
    print("\n%-34s %-2s %7s %7s %7s %7s %7s | %7s %7s %7s | %6s" % ("estimate", "", "moving", "", "", "", "",
                                                                  "rest", "", "", "delay"))
    print("%-34s %-2s %7s %7s %7s %7s %7s | %7s %7s %7s | %6s" % ("", "", "median", "med|e|", "p95|e|", "p99|e|",
                                                                  ">noise", "median", "sd", "p95|e|", "s"))
    for name, (u, v, r) in est.items():
        out["estimates"][name] = {}
        for q, e in (("u", u), ("v", v), ("r", r)):
            if e is None:
                continue
            err = e - refs[q]
            sm, sr = stats(err[moving]), stats(err[rest])
            dl = best_delay(e, refs[q], T) if q in ("u", "r") else None
            over = float(np.mean(np.abs(err[moving & np.isfinite(err)]) > SIM_NOISE[q]))
            sm["share_over_sim_noise"] = over
            sc = stats(err[clean])
            sc["share_over_sim_noise"] = float(np.mean(np.abs(err[clean & np.isfinite(err)]) > SIM_NOISE[q]))
            out["estimates"][name][q] = {"moving": sm, "moving_away_from_jumps": sc, "rest": sr, "delay": dl}
            print("%-34s %-2s %7.3f %7.3f %7.3f %7.3f %6.0f%% | %7.3f %7.3f %7.3f | %6s" % (
                name, q, sm["median"], sm["median_abs"], sm["p95_abs"], sm["p99_abs"], 100 * over,
                sr["median"], sr["sd"], sr["p95_abs"], "%+.1f" % dl["delay_s"] if dl else ""))
    print("\nMoving, away from SLAM jumps (%d of %d moving instants):" % (clean.sum(), moving.sum()))
    print("%-34s %-2s %7s %7s %7s %7s" % ("estimate", "", "median", "med|e|", "p95|e|", ">noise"))
    for name, d in out["estimates"].items():
        for q, s in d.items():
            c = s["moving_away_from_jumps"]
            print("%-34s %-2s %7.3f %7.3f %7.3f %6.0f%%" % (name, q, c["median"], c["median_abs"], c["p95_abs"],
                                                           100 * c["share_over_sim_noise"]))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json.dump(out, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
        try:
            figure(T, refs, est, moving, os.path.join(args.out, "ego_estimates.png"))
        except ImportError:
            pass
        print("\nwritten", args.out)


def figure(T, refs, est, moving, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3dd"
    c_ref, c_now, c_new = "#6b6a63", "#2a78d6", "#eb6834"
    # a 40 s window with the most turning
    w = int(round(40 / (T[1] - T[0])))
    score = np.convolve(np.abs(refs["r"]), np.ones(w), "valid")
    k = int(np.argmax(score))
    sl = slice(k, k + w)
    tt = T[sl] - T[sl][0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    panels = (("r", "yaw rate r (deg/s)", "pose 0.1 m, 0.5 s (today)", "gyro, mean 0.1 s"),
              ("u", "surge u (m/s)", "pose 0.1 m, 0.5 s (today)", "pose full, 0.5 s"))
    for ax, (q, ylab, now, new) in zip(axes, panels):
        ax.plot(tt, refs[q][sl], color=c_ref, lw=2.4, label="reference (full-precision poses, centred)")
        ax.plot(tt, est[now][{"u": 0, "v": 1, "r": 2}[q]][sl], color=c_now, lw=1.6, label=now)
        ax.plot(tt, est[new][{"u": 0, "v": 1, "r": 2}[q]][sl], color=c_new, lw=1.6, label=new)
        ax.set_ylabel(ylab, color=ink)
        ax.grid(True, color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8.5, labelcolor=ink)
    axes[-1].set_xlabel("time in the 40 s window with the most turning (s)", color=ink)
    fig.suptitle("Own-motion estimates against the reference, recording of 2026-11-26", color=ink, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
