"""
Replay a saved Bluefin log in the live display (telemetry_view.py), offline: no network, no command.

Reads any log with the vessel's [HH:MM:SS.ffffff] stamps: the shore computer's logs
(udp_raw_logger.py, udp_manoeuvre.py, the bridges' telemetry.log) and the vessel's own logs
(RAW_LOG, datalogs/laser_*.log, D0-n-RAW.log, with their IMU and SRVO lines, which are counted and
skipped).  Lines are played at the pace of their stamps (--rate 2: twice as fast; --rate 0: as fast
as the window draws), and drawn as in the live window: the track, the hull and heading, the latest
LiDAR returns and, with --fence, the fence.

    python log_replay_view.py 2026-11-26/D0-5.log
    python log_replay_view.py 2026-11-26/D0-5-RAW.log --rate 4 --full-track
    python log_replay_view.py 2026-11-26/D0-4.log --rate 0 --full-track --save-png d0_4_track.png --exit-at-end
    python log_replay_view.py 2026-07-03/1.log --pose-map july

--full-track: when the log ends, the whole trajectory is drawn and the view fitted to it, and the
window stays open until it is closed (or, with --exit-at-end, the program ends after --save-png).
--start-s / --end-s replay part of a log (seconds from its first stamp).

Pose layout (--pose-map): nov2026 (default) x = -B, y = A, heading = -C, the vessel's output since
November 2026; july x = B, y = -A, heading = -C, for the July and February logs.  Keys in the window:
Space pauses, Esc or closing the window quits.
"""

import argparse
import os
import re
import sys
import time

import telemetry_view as tv

RE_STAMP = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2}\.\d+)\]")


def stamp(line):
    m = RE_STAMP.match(line)
    return None if m is None else int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def kind(line):
    s = line.strip()
    if tv.RE_POSE.match(s):
        return "pose"
    if tv.RE_LIDAR.match(s):
        return "LiDAR"
    m = re.match(r"^\[[\d:.]+\]\d+,([A-Z]+),", s)
    if m:
        return m.group(1)                       # GYRO, ACCEL, QUAT, SRVO (vessel logs)
    if s.lower() == "hey":
        return "hey"
    if s.startswith("#"):
        return s[:4]                            # #CMD, #MAN (manoeuvre logs)
    return "other"


def load(path):
    """(stamp, line) for every stamped line, in stamp order (vessel logs are not strictly ordered)."""
    rows, last = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            t = stamp(line)
            if t is None:
                continue
            if last is not None and t < last - 43200:          # past midnight
                t += 86400
            rows.append((t, line))
            last = t
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("log")
    ap.add_argument("--rate", type=float, default=1.0, help="replay speed; 0: as fast as the window draws")
    ap.add_argument("--pose-map", choices=("nov2026", "july"), default="nov2026")
    ap.add_argument("--start-s", type=float, default=0.0, help="start this far into the log")
    ap.add_argument("--end-s", type=float, default=0.0, help="stop this far into the log; 0: the end")
    ap.add_argument("--full-track", action="store_true",
                    help="at the end, draw the whole trajectory fitted to the window and keep it open")
    ap.add_argument("--fence", default=None, metavar="XMIN,XMAX,YMIN,YMAX",
                    help="draw this fence (pose frame of --pose-map); write it with '='")
    ap.add_argument("--fence-margin-m", type=float, default=1.5)
    ap.add_argument("--render-fps", type=float, default=20.0)
    ap.add_argument("--save-png", default=None, help="save the final view to this PNG")
    ap.add_argument("--exit-at-end", action="store_true", help="end the program when the log ends")
    args = ap.parse_args()

    rows = load(args.log)
    if not rows:
        sys.exit("no stamped line in %s" % args.log)
    t0 = rows[0][0]
    rows = [r for r in rows if r[0] - t0 >= args.start_s and (args.end_s <= 0 or r[0] - t0 <= args.end_s)]
    fence = tuple(float(v) for v in args.fence.split(",")) if args.fence else None
    view, why = tv.TelemetryView.open("Replay  " + os.path.basename(args.log), fps=args.render_fps,
                                      fence=fence, fence_margin_m=args.fence_margin_m, pose_map=args.pose_map)
    if view is None:
        sys.exit("no display: %s" % why)
    pg = view.pg
    counts, last_pose = {}, None
    track = []                                  # the whole trajectory, for --full-track
    for t, line in rows:
        p = tv.read_pose(line, args.pose_map)
        if p is not None:
            track.append(p[:2])
    duration = rows[-1][0] - rows[0][0]
    print("[REPLAY] %s: %d stamped lines, %.1f s, pose layout %s" % (args.log, len(rows), duration, args.pose_map))

    def status(t_now):
        lines = [os.path.basename(args.log), "log time %.1f / %.1f s   rate %s" % (t_now - rows[0][0], duration,
                                                                                    args.rate or "max")]
        if last_pose is not None:
            lines.append("pose x %+.1f  y %+.1f  heading %.1f" % (last_pose[0], last_pose[1], last_pose[2] % 360))
        lines.append("line kinds:")
        for k, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
            lines.append("  %-10s %7d" % (k, n))
        lines.append(("Space pauses, Esc quits", (150, 156, 170)))
        return lines

    wall0, paused, quit_ = time.monotonic(), False, False
    log0 = rows[0][0]
    i = 0
    while i < len(rows) and not quit_:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                quit_ = True
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                paused = not paused
                if not paused:                  # resume the clock where it stopped
                    wall0 = time.monotonic() - (rows[i][0] - log0) / args.rate if args.rate > 0 else wall0
        if paused:
            view.tick(status(rows[i][0]), ("PAUSED", (255, 200, 60)), force=True)
            time.sleep(0.05)
            continue
        now = time.monotonic()
        due = rows[-1][0] if args.rate <= 0 else log0 + (now - wall0) * args.rate
        batch_end = time.monotonic() + 0.02
        while i < len(rows) and rows[i][0] <= due and time.monotonic() < batch_end:
            t, line = rows[i]
            k = kind(line)
            counts[k] = counts.get(k, 0) + 1
            view.feed(line, now=now)
            if k == "pose":
                last_pose = view.pose
            i += 1
        view.t_last_line = time.monotonic()     # replay: no NO TELEMETRY banner
        view.tick(status(rows[min(i, len(rows) - 1)][0]))
        if args.rate > 0 and i < len(rows) and rows[i][0] > due:
            time.sleep(min(0.01, rows[i][0] - due))
        if view.closed:
            quit_ = True

    if not view.closed:
        if args.full_track and track:
            view.trail.clear()
            view.trail.extend(track)
            view.bounds = None                  # fit the view to the whole trajectory
            banner = ("END OF LOG: full trajectory (%d poses)" % len(track), (120, 230, 140))
        else:
            banner = ("END OF LOG", (120, 230, 140))
        view.tick(status(rows[-1][0]), banner, force=True)
        if args.save_png:
            pg.image.save(view.screen, args.save_png)
            print("[REPLAY] saved %s" % os.path.abspath(args.save_png))
        while not args.exit_at_end and not view.closed:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    view.close()
            time.sleep(0.05)
        view.close()
    print("[REPLAY] done: %s" % ", ".join("%s %d" % kv for kv in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
