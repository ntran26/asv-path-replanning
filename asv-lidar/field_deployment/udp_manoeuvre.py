"""
Scripted manoeuvres for the Bluefin calibration sessions: exact rudder and propulsion commands
sent over UDP at exact times, with the telemetry logged raw on the same clock.

The same link as every bridge here (START to the vessel-control laptop on 5050, telemetry back to
port 5000, commands as `$CMD,<vessel rudder>,<S2>`), so it replaces radio control for the
identification blocks: every command is known, timed and repeatable, and the log holds it.

    python udp_manoeuvre.py --server-ip 10.201.205.110 --out S1b-B-01.log steps --rpm 0 --rudder 25 50 100
    python udp_manoeuvre.py --server-ip 10.201.205.110 --out S1b-D-01.log turn --rpm 12 --rudder 100 --heading-change 720
    python udp_manoeuvre.py --server-ip 10.201.205.110 --out S1b-E-01.log zigzag --rpm 12 --delta 20 --psi 20
    python udp_manoeuvre.py --server-ip 10.201.205.110 --out S1b-C2-01.log stop --rpm 12 --allow-astern
    python udp_manoeuvre.py --selftest

Manoeuvres (each starts when Enter is pressed, after the telemetry is flowing):
  hold      a constant command for a time (straight runs, speed points, steady turns)
  coast     run-up at an RPM, then zero propulsion (coast-down)
  steps     rudder steps 0 -> +a -> 0 -> -a -> 0 for each amplitude, optionally +a -> -a
  ramp      rudder ramp at a set rate to +a, to -a, back to 0
  turn      run-up, then a fixed rudder until a heading change (e.g. 720 deg) or a time
  zigzag    run-up, then delta/psi zig-zag switching on the measured heading
  stop      run-up (optionally into a steady turn), then full astern, rudder centred, until the
            measured speed reads stopped, then zero; needs --allow-astern (bench test P-8 first)
  sequence  a fixed command sequence from a CSV file (t_s, rudder_pct, rpm), zero-order hold

Conventions:
  rudder    percent of full rudder in the simulator's convention: positive turns to starboard
            (heading increasing). The vessel receives --rudder-sign times it (-1, as the bridges).
  rpm       the simulator's propulsion unit (0-24; cruise 6); S2 = rpm / --rpm-max * 100.
            Negative rpm (astern) only with --allow-astern; S2 = -100 is full astern.

Safety: Ctrl+C at any time sends zero rudder and zero propulsion three times and stops. Telemetry
lost for --watchdog-s, a manoeuvre running past --max-s, or the pose coming within
--fence-margin-m of the --fence box (the basin, assumed 10 m x 25 m, in the pose frame) does the
same. The radio-control override stays the physical safety.

Files (an existing name is never overwritten: _2, _3, ... is added):
  <out>                every datagram as received, with `#CMD` lines (every command sent) and
                       `#MAN` lines (phases and events) between them, on the same clock
  <out>.arrivals.csv   arrival number, wall-clock time, time since start, size of each datagram
  <out>.commands.csv   every command: times, phase, reason, rudder, rpm, S2, the text sent
  <out>.meta.json      options, manoeuvre, events (switch times, overshoots, stop), how it ended

Needs only Python 3.6 or later and no other file of this repository.
"""

import argparse
import csv
import json
import math
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone

HELLO = "START"
ZERO = (0.0, 0.0)

# The pose format of log_parser.BluefinStreamDecoder (copied: standard library only); the decoder
# reads x as given, y and yaw negated, and so does this program.
RE_POSE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]"
    r"(?P<y>[-+]\d+\.\d+),(?P<x>[-+]\d+\.\d+),(?P<yaw>[-+]\d+\.\d+)\s*$")
RE_STAMP = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d+\]")        # a telemetry line carries the vessel's stamp


def stamp_seconds(ts):
    hh, mm, rest = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(rest)


def wrap180(deg):
    return (deg + 540.0) % 360.0 - 180.0


def free_name(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 2
    while os.path.exists("%s_%d%s" % (stem, k, ext)):
        k += 1
    return "%s_%d%s" % (stem, k, ext)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Abort(Exception):
    pass


class FenceStop(Abort):
    pass


# ---------------------------------------------------------------------------
# Pose tracking from the telemetry
# ---------------------------------------------------------------------------
class PoseTrack:
    """Heading change (unwrapped) and speed over a window, from the vessel's pose lines and stamps."""

    def __init__(self, speed_window_s):
        self.window = float(speed_window_s)
        self.history = []               # (t_vessel, x, y)
        self.heading = None             # deg, decoder convention
        self.unwrapped = 0.0
        self.t_last_mono = None
        self.count = 0

    def feed(self, line, t_mono):
        m = RE_POSE.match(line.strip())
        if m is None:
            return False
        t = stamp_seconds(m.group("ts"))
        x, y, hdg = float(m.group("x")), -float(m.group("y")), -float(m.group("yaw"))
        if self.heading is not None:
            self.unwrapped += wrap180(hdg - self.heading)
        self.heading = hdg
        if self.history and t < self.history[-1][0] - 43200:     # past midnight
            t += 86400
        self.history.append((t, x, y))
        while len(self.history) > 2 and t - self.history[0][0] > max(5.0, 3 * self.window):
            self.history.pop(0)
        self.t_last_mono = t_mono
        self.count += 1
        return True

    def speed(self):
        """Speed over the window (m/s), or None until the window is covered."""
        if len(self.history) < 2:
            return None
        t1, x1, y1 = self.history[-1]
        best = None
        for t0, x0, y0 in reversed(self.history[:-1]):
            if t1 - t0 >= self.window:
                best = (t0, x0, y0)
                break
        if best is None:
            return None
        t0, x0, y0 = best
        return math.hypot(x1 - x0, y1 - y0) / (t1 - t0)


# ---------------------------------------------------------------------------
# Manoeuvres: generators that yield (rudder_pct, rpm, phase, reason) and read the pose track
# ---------------------------------------------------------------------------
class Context:
    def __init__(self, runner):
        self.r = runner

    def now(self):
        return time.monotonic() - self.r.t_manoeuvre

    def wait(self, seconds):
        """Yield control until `seconds` have passed (the runner keeps receiving and sending)."""
        end = self.now() + float(seconds)
        while self.now() < end:
            yield None

    def event(self, kind, **data):
        self.r.event(kind, **data)


def m_hold(ctx, a):
    yield (a.rudder, a.rpm, "hold", "start")
    for _ in ctx.wait(a.duration_s):
        yield None


def m_coast(ctx, a):
    yield (0.0, a.rpm, "runup", "start")
    for _ in ctx.wait(a.runup_s):
        yield None
    yield (0.0, 0.0, "coast", "propulsion off")
    for _ in ctx.wait(a.coast_s):
        yield None


def m_steps(ctx, a):
    yield (0.0, a.rpm, "steps", "start")
    for _ in ctx.wait(a.hold_s):
        yield None
    for amp in a.rudder:
        for k in range(a.repeats):
            for target in (amp, 0.0, -amp, 0.0):
                yield (target, a.rpm, "steps", "step to %+.0f (amplitude %.0f, repeat %d)" % (target, amp, k + 1))
                for _ in ctx.wait(a.hold_s):
                    yield None
    if a.reversal:
        for amp in a.rudder:
            for k in range(a.repeats):
                for target in (amp, -amp, amp, 0.0):
                    yield (target, a.rpm, "reversal", "step to %+.0f (amplitude %.0f, repeat %d)" % (target, amp, k + 1))
                    for _ in ctx.wait(a.hold_s):
                        yield None


def m_ramp(ctx, a):
    yield (0.0, a.rpm, "ramp", "start")
    for _ in ctx.wait(a.hold_s):
        yield None
    for start, end in ((0.0, a.to), (a.to, -a.to), (-a.to, 0.0)):
        span = abs(end - start) / a.rate
        t0 = ctx.now()
        while True:
            frac = min(1.0, (ctx.now() - t0) / span) if span > 0 else 1.0
            yield (start + (end - start) * frac, a.rpm, "ramp", "ramp")
            if frac >= 1.0:
                break
            for _ in ctx.wait(0.05):
                yield None
        for _ in ctx.wait(a.hold_s):
            yield None


def runup(ctx, a):
    yield (0.0, a.rpm, "runup", "start")
    for _ in ctx.wait(a.runup_s):
        yield None


def m_turn(ctx, a, track):
    for c in runup(ctx, a):
        yield c
    psi0 = track.unwrapped
    yield (a.rudder, a.rpm, "turn", "helm %+.0f" % a.rudder)
    ctx.event("helm", heading_change=0.0)
    t0 = ctx.now()
    told = False
    while True:
        signed = track.unwrapped - psi0
        change = abs(signed)
        if not told and change >= 10.0:
            told = True
            side = "starboard (heading increasing)" if signed > 0 else "port (heading decreasing)"
            expected = (signed > 0) == (a.rudder > 0)
            ctx.event("turning", side=side, as_expected=expected,
                      note="positive rudder should turn to starboard" if not expected else "")
        if a.heading_change and change >= a.heading_change:
            ctx.event("heading change reached", heading_change=round(signed, 1), seconds=round(ctx.now() - t0, 2))
            break
        if a.duration_s and ctx.now() - t0 >= a.duration_s:
            ctx.event("duration reached", heading_change=round(signed, 1))
            break
        yield None
    yield (0.0, a.rpm, "straighten", "rudder centred")
    for _ in ctx.wait(a.after_s):
        yield None


def m_zigzag(ctx, a, track):
    for c in runup(ctx, a):
        yield c
    psi0 = track.unwrapped
    rudder = a.delta_pct
    yield (rudder, a.rpm, "zigzag", "first helm %+.0f" % rudder)
    ctx.event("first helm", rudder=rudder)
    sense = None                    # heading change sign for a positive rudder, from the response
    switches = 0
    peak, peaks = 0.0, []
    while switches < a.switches:
        psi = track.unwrapped - psi0
        if sense is None:
            if abs(psi) >= min(5.0, a.psi):
                sense = 1.0 if psi > 0 else -1.0
                ctx.event("response sense", positive_rudder_turns="heading increasing" if sense > 0 else "heading decreasing")
            yield None
            continue
        along = psi * sense * (1.0 if rudder > 0 else -1.0)      # change in the commanded direction
        if switches > 0:
            peak = max(peak, -psi * sense * (1.0 if rudder > 0 else -1.0))  # overshoot past the last switch
        if along >= a.psi:
            if switches > 0:
                peaks.append(round(peak - a.psi, 2))
            rudder = -rudder
            switches += 1
            peak = -1e9
            ctx.event("switch", n=switches, heading_change=round(psi, 1), rudder=rudder)
            yield (rudder, a.rpm, "zigzag", "switch %d to %+.0f" % (switches, rudder))
            continue
        yield None
    # the overshoot after the last switch
    t_end = ctx.now() + a.after_s
    while ctx.now() < t_end:
        psi = track.unwrapped - psi0
        peak = max(peak, -psi * sense * (1.0 if rudder > 0 else -1.0))
        yield None
    peaks.append(round(peak - a.psi, 2))
    ctx.event("overshoots", degrees=peaks, note="heading past +/-psi after each switch, from the pose")
    yield (0.0, a.rpm, "straighten", "rudder centred")
    for _ in ctx.wait(2.0):
        yield None


def m_stop(ctx, a, track):
    for c in runup(ctx, a):
        yield c
    if a.rudder:
        yield (a.rudder, a.rpm, "turn", "settle into the turn at %+.0f" % a.rudder)
        for _ in ctx.wait(a.turn_s):
            yield None
    ctx.event("stop ordered", speed=track.speed(), heading_change=round(track.unwrapped, 1))
    yield (0.0, a.astern_rpm, "astern", "full astern, rudder centred")
    t0 = ctx.now()
    while True:
        u = track.speed()
        if ctx.now() - t0 >= a.min_astern_s and u is not None and u <= a.stop_speed:
            ctx.event("stopped", seconds=round(ctx.now() - t0, 2), speed=round(u, 3))
            break
        if ctx.now() - t0 >= a.max_astern_s:
            ctx.event("astern time limit", seconds=round(ctx.now() - t0, 2), speed=u)
            break
        yield None
    yield (0.0, 0.0, "rest", "propulsion off")
    for _ in ctx.wait(a.after_s):
        yield None


def m_sequence(ctx, a):
    rows = a.rows
    for i, (t, rud, rpm) in enumerate(rows):
        while ctx.now() < t:
            yield None
        yield (rud, rpm, "sequence", "row %d" % (i + 1))
    for _ in ctx.wait(a.after_s):
        yield None


# ---------------------------------------------------------------------------
# The runner: link, logging, command sending, watchdog
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, args, out=sys.stdout):
        self.a = args
        self.out = out
        self.track = PoseTrack(args.speed_window_s)
        self.sock = None
        self.command = None             # (rudder_pct, rpm)
        self.phase = "waiting"
        self.t_start = time.monotonic()
        self.t_manoeuvre = None
        self.t_last_sent = None
        self.t_last_rx = None           # last time-stamped telemetry line
        self.t_first_rx = None          # first datagram of any kind
        self.n_rx = 0
        self.n_cmd = 0
        self.events = []
        self.ended = None

    def say(self, text):
        try:
            self.out.write(text + "\n")
            self.out.flush()
        except Exception:
            pass

    def event(self, kind, **data):
        t = None if self.t_manoeuvre is None else round(time.monotonic() - self.t_manoeuvre, 3)
        item = dict(kind=kind, t_manoeuvre_s=t, t_mono_s=round(time.monotonic() - self.t_start, 6), **data)
        self.events.append(item)
        self.log.write(("#MAN," + json.dumps(item, separators=(",", ":")) + "\n").encode())
        self.say("[MAN] %s %s" % (kind, ", ".join("%s=%s" % (k, v) for k, v in data.items())))

    # -- commands --------------------------------------------------------------------------
    def message(self, rudder, rpm):
        s2 = rpm / self.a.rpm_max * 100.0
        s2 = max(-100.0 if self.a.allow_astern else 0.0, min(100.0, s2))
        rud = max(-100.0, min(100.0, rudder))
        return "$CMD,%.2f,%.2f" % (self.a.rudder_sign * rud + 0.0, s2 + 0.0), s2     # + 0.0: no "-0.00"

    def send(self, rudder, rpm, phase, reason):
        text, s2 = self.message(rudder, rpm)
        sent = False
        if not self.a.dry_run:
            try:
                self.sock.sendto(text.encode(), (self.a.server_ip, self.a.server_port))
                sent = True
            except OSError as exc:
                self.say("[CMD] not sent (%s)" % exc)
        now = time.monotonic()
        t_rel = now - self.t_start
        self.log.write(("#CMD,t_mono=%.6f,t_unix=%.6f,phase=%s,reason=%s,rudder_pct=%.2f,rpm=%.3f,s2=%.2f,cmd=%s,sent=%d\n"
                        % (t_rel, time.time(), phase, reason.replace(",", ";"), rudder, rpm, s2, text, int(sent))).encode())
        self.cmd_csv.writerow(["%.6f" % t_rel, "%.6f" % time.time(), phase, reason, "%.2f" % rudder, "%.3f" % rpm,
                               "%.2f" % s2, text, int(sent)])
        self.cmd_fh.flush()
        self.t_last_sent = now
        self.n_cmd += 1
        if reason != "keepalive":
            self.say("[CMD] %6.2f s  %-10s %s  (%s)" % (0.0 if self.t_manoeuvre is None else now - self.t_manoeuvre,
                                                     phase, text, reason))

    def inside_fence(self):
        """True or False against --fence (shrunk by the margin), None without a fence or a pose."""
        box = getattr(self.a, "fence_box", None)
        if box is None or not self.track.history:
            return None
        _, x, y = self.track.history[-1]
        return box[0] <= x <= box[1] and box[2] <= y <= box[3]

    def zero(self, reason):
        for _ in range(3):
            self.send(0.0, 0.0, "end", reason)
            time.sleep(0.1)

    # -- telemetry -------------------------------------------------------------------------
    def receive(self, timeout):
        self.sock.settimeout(timeout)
        try:
            data, _ = self.sock.recvfrom(65535)
        except socket.timeout:
            return False
        except ConnectionResetError:
            return False
        except OSError as exc:
            self.say("[RAW] receive error (%s)" % exc)
            return False
        now = time.monotonic()
        self.log.write(data if data.endswith(b"\n") else data + b"\n")
        self.n_rx += 1
        self.arrivals.write("%d,%.6f,%.6f,%d\n" % (self.n_rx, time.time(), now - self.t_start, len(data)))
        if self.t_first_rx is None:
            self.t_first_rx = now
        try:
            for line in data.decode("utf-8", errors="replace").splitlines():
                if RE_STAMP.match(line.strip()):
                    self.t_last_rx = now        # telemetry; the `hey` answers to commands do not count
                self.track.feed(line, now)
        except Exception:
            pass
        return True

    def drain(self, until):
        """Receive until the monotonic time `until` (at least one short wait)."""
        while True:
            left = until - time.monotonic()
            if left <= 0:
                self.receive(0.0005)
                return
            self.receive(min(left, 0.02))

    # -- run -------------------------------------------------------------------------------
    def open_files(self):
        path = free_name(self.a.out)
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        self.path = path
        self.log = open(path, "wb", buffering=0)
        self.arrivals = open(path + ".arrivals.csv", "w", encoding="utf-8", buffering=1)
        self.arrivals.write("n,t_unix_s,t_since_start_s,bytes\n")
        self.cmd_fh = open(path + ".commands.csv", "w", newline="", encoding="utf-8")
        self.cmd_csv = csv.writer(self.cmd_fh)
        self.cmd_csv.writerow(["t_since_start_s", "t_unix_s", "phase", "reason", "rudder_pct", "rpm", "s2", "message", "sent"])

    def close_files(self, meta):
        for fh in (self.log, self.arrivals, self.cmd_fh):
            try:
                fh.close()
            except Exception:
                pass
        try:
            with open(self.path + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1, default=str)
        except Exception as exc:
            self.say("[MAN] meta file not written: %s" % exc)

    def generator(self):
        a, ctx = self.a, Context(self)
        name = a.manoeuvre
        if name == "hold":
            return m_hold(ctx, a)
        if name == "coast":
            return m_coast(ctx, a)
        if name == "steps":
            return m_steps(ctx, a)
        if name == "ramp":
            return m_ramp(ctx, a)
        if name == "turn":
            return m_turn(ctx, a, self.track)
        if name == "zigzag":
            return m_zigzag(ctx, a, self.track)
        if name == "stop":
            return m_stop(ctx, a, self.track)
        if name == "sequence":
            return m_sequence(ctx, a)
        raise SystemExit("unknown manoeuvre %s" % name)

    def run(self, start_event=None):
        a = self.a
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        try:
            self.sock.bind((a.bind_ip, a.local_port))
        except OSError as exc:
            self.say("[MAN] cannot listen on %s:%d (%s): close the other program or give --local-port"
                     % (a.bind_ip, a.local_port, exc))
            return None
        self.local_port = self.sock.getsockname()[1]
        self.open_files()
        meta = {"program": "udp_manoeuvre.py", "log": os.path.abspath(self.path), "started_utc": utc_now(),
                "options": {k: v for k, v in vars(a).items() if k != "rows"}, "python": sys.version.split()[0]}
        self.say("[MAN] logging to %s" % os.path.abspath(self.path))
        self.say("[MAN] %s; vessel %s:%d; %s" % (describe(a), a.server_ip, a.server_port,
                                                 "DRY RUN: no command is sent" if a.dry_run else "commands are sent"))
        try:
            # 1. Handshake, until telemetry flows.
            hellos, t_hello = 0, -1e9
            while self.t_first_rx is None or self.t_last_rx is None:
                if time.monotonic() - t_hello >= a.hello_every_s:
                    try:
                        self.sock.sendto((a.hello + "\n").encode(), (a.server_ip, a.server_port))
                    except OSError as exc:
                        self.say("[MAN] handshake not sent (%s)" % exc)
                    hellos += 1
                    t_hello = time.monotonic()
                    if hellos > 1:
                        self.say("[MAN] nothing received yet; handshake sent again (%d)" % hellos)
                self.receive(0.2)
            self.say("[MAN] telemetry flowing")
            # 2. Wait for the start (Enter), logging all the while.
            needs_pose = a.manoeuvre in ("turn", "zigzag", "stop")
            t_wait = time.monotonic()
            while time.monotonic() - t_wait < 2.0:
                self.receive(0.05)
            if needs_pose and self.track.count == 0:
                self.say("[MAN] no pose line recognised in 2 s: this manoeuvre needs the heading or speed. "
                         "Check the telemetry (udp_raw_logger.py --check), or use hold / steps / sequence.")
                self.ended = "no pose"
                return self.finish(meta)
            if start_event is None:
                go = threading.Event()

                def wait_enter():
                    try:
                        input()
                    except Exception:
                        pass
                    go.set()

                if getattr(a, "fence_box", None) is not None:
                    if self.track.history:
                        _, x, y = self.track.history[-1]
                        self.say("[MAN] pose (%.1f, %.1f) is %s the fence less its %.1f m margin"
                                 % (x, y, "inside" if self.inside_fence() else "OUTSIDE", a.fence_margin_m))
                    else:
                        self.say("[MAN] no pose yet: the fence cannot be checked")
                self.say("[MAN] pose lines %d, last heading %s. Press Enter to start, Ctrl+C to quit."
                         % (self.track.count, self.track.heading))
                threading.Thread(target=wait_enter, daemon=True).start()
            else:
                go = start_event
            last_status = time.monotonic()
            while not go.is_set():
                self.receive(0.1)
                if time.monotonic() - last_status >= 5.0:
                    last_status = time.monotonic()
                    self.say("[MAN] waiting for Enter; datagrams %d, pose lines %d" % (self.n_rx, self.track.count))
            # 3. The manoeuvre.
            self.t_manoeuvre = time.monotonic()
            self.event("manoeuvre start", manoeuvre=a.manoeuvre)
            steps = self.generator()
            last_status = time.monotonic()
            for item in steps:
                now = time.monotonic()
                if item is not None:
                    rudder, rpm, phase, reason = item
                    self.command, self.phase = (rudder, rpm), phase
                    self.send(rudder, rpm, phase, reason)
                elif self.command is not None and now - self.t_last_sent >= 1.0 / a.cmd_rate:
                    self.send(self.command[0], self.command[1], self.phase, "keepalive")
                if a.watchdog_s > 0 and now - (self.t_last_rx or now) > a.watchdog_s:
                    raise Abort("telemetry lost for %.1f s" % (now - self.t_last_rx))
                if needs_pose and a.watchdog_s > 0 and now - (self.track.t_last_mono or now) > a.watchdog_s:
                    raise Abort("no pose line for %.1f s" % (now - self.track.t_last_mono))
                if now - self.t_manoeuvre > a.max_s:
                    raise Abort("manoeuvre longer than --max-s %.0f s" % a.max_s)
                inside = self.inside_fence()
                if inside is False:
                    _, x, y = self.track.history[-1]
                    raise FenceStop("fence: pose (%.1f, %.1f) within %.1f m of the basin edge" % (x, y, a.fence_margin_m))
                if now - last_status >= a.status_s:
                    last_status = now
                    u = self.track.speed()
                    self.say("[MAN] %6.1f s  %-10s heading change %+7.1f deg  speed %s  telemetry age %.2f s"
                             % (now - self.t_manoeuvre, self.phase, self.track.unwrapped,
                                "-" if u is None else "%.2f m/s" % u, now - (self.t_last_rx or now)))
                self.drain(now + 0.01)
            self.zero("manoeuvre complete")
            self.ended = "complete"
            tail_end = time.monotonic() + a.tail_s
            self.drain(tail_end)
        except FenceStop as exc:
            # With astern permitted the vessel is braked (full astern, rudder centred) until the
            # measured speed reads stopped; otherwise propulsion is cut and the radio-control
            # override must stop it: cut, it coasts (about 13 m from RPM 12 in the model).
            if a.allow_astern:
                self.event("fence", reason=str(exc), action="full astern until stopped")
                t0 = time.monotonic()
                self.send(0.0, -a.rpm_max, "fence", "full astern")
                while time.monotonic() - t0 < 8.0:
                    self.drain(time.monotonic() + 0.05)
                    if time.monotonic() - self.t_last_sent >= 1.0 / a.cmd_rate:
                        self.send(0.0, -a.rpm_max, "fence", "keepalive")
                    u = self.track.speed()
                    if time.monotonic() - t0 >= 1.0 and u is not None and u <= 0.08:
                        break
            else:
                self.event("fence", reason=str(exc), action="propulsion cut: TAKE OVER BY RADIO CONTROL")
                self.say("[MAN] *** FENCE: propulsion cut; the vessel coasts. TAKE OVER BY RADIO CONTROL ***")
            self.zero("abort: %s" % exc)
            self.ended = "abort: %s" % exc
        except Abort as exc:
            self.zero("abort: %s" % exc)
            self.ended = "abort: %s" % exc
            self.event("abort", reason=str(exc))
        except KeyboardInterrupt:
            self.zero("abort: Ctrl+C")
            self.ended = "abort: Ctrl+C"
            self.event("abort", reason="Ctrl+C")
        except Exception as exc:
            try:
                self.zero("abort: error %s" % type(exc).__name__)
            except Exception:
                pass
            self.ended = "error: %r" % (exc,)
            self.say("[MAN] error: %r" % (exc,))
        return self.finish(meta)

    def finish(self, meta):
        try:
            self.sock.close()
        except Exception:
            pass
        meta.update(finished_utc=utc_now(), ended=self.ended, datagrams=self.n_rx, commands=self.n_cmd,
                    pose_lines=self.track.count, events=self.events)
        self.close_files(meta)
        self.say("[MAN] ended (%s): %d datagrams, %d commands; %s" % (self.ended, self.n_rx, self.n_cmd,
                                                                     os.path.abspath(self.path)))
        return meta


def describe(a):
    n = a.manoeuvre
    if n == "hold":
        return "hold rudder %+.0f %%, rpm %.1f for %.0f s" % (a.rudder, a.rpm, a.duration_s)
    if n == "coast":
        return "run-up %.0f s at rpm %.1f, then coast %.0f s" % (a.runup_s, a.rpm, a.coast_s)
    if n == "steps":
        return "rudder steps %s %%, %d repeats, %.1f s holds, rpm %.1f%s" % (
            a.rudder, a.repeats, a.hold_s, a.rpm, ", with reversals" if a.reversal else "")
    if n == "ramp":
        return "rudder ramp to +/-%.0f %% at %.0f %%/s, rpm %.1f" % (a.to, a.rate, a.rpm)
    if n == "turn":
        return "run-up %.0f s at rpm %.1f, then rudder %+.0f %% until %s" % (
            a.runup_s, a.rpm, a.rudder, ("%.0f deg" % a.heading_change) if a.heading_change else ("%.0f s" % a.duration_s))
    if n == "zigzag":
        return "run-up %.0f s at rpm %.1f, then zig-zag %.0f/%.0f (rudder %.0f %%), %d switches" % (
            a.runup_s, a.rpm, a.delta, a.psi, a.delta_pct, a.switches)
    if n == "stop":
        return "run-up %.0f s at rpm %.1f%s, then astern S2 %.0f until %.2f m/s (at most %.0f s)" % (
            a.runup_s, a.rpm, (", turn at %+.0f %% for %.0f s" % (a.rudder, a.turn_s)) if a.rudder else "",
            a.astern_rpm / a.rpm_max * 100.0, a.stop_speed, a.max_astern_s)
    if n == "sequence":
        return "sequence of %d rows over %.1f s" % (len(a.rows), a.rows[-1][0] if a.rows else 0.0)
    return n


# ---------------------------------------------------------------------------
# --selftest: a stand-in vessel on this machine
# ---------------------------------------------------------------------------
class ToyVessel(threading.Thread):
    """Answers START, streams pose (10 Hz) and LiDAR (2 Hz) lines in the vessel's format, obeys $CMD.

    Fast kinematics, so a whole manoeuvre takes seconds: speed tends to 0.2 m/s per rpm unit
    (time constant 0.5 s), yaw rate to 3 deg/s per rudder percent at 1 m/s, astern brakes."""

    def __init__(self, silent_after_s=None):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.received = []
        self.stop_flag = threading.Event()
        self.silent_after_s = silent_after_s

    def run(self):
        self.sock.settimeout(10.0)
        data, client = self.sock.recvfrom(4096)
        self.sock.settimeout(0.0)
        x = y = hdg = u = 0.0
        rud_vessel, s2 = 0.0, 0.0
        t0 = time.monotonic()
        last, k = t0, 0
        while not self.stop_flag.is_set():
            try:
                while True:
                    d, _ = self.sock.recvfrom(4096)
                    text = d.decode()
                    self.received.append((time.monotonic() - t0, text))
                    if text.startswith("$CMD,"):
                        _, r, s = text.strip().split(",")
                        rud_vessel, s2 = float(r), float(s)
                        self.sock.sendto(b"hey\n", client)
            except (BlockingIOError, socket.timeout):
                pass
            except OSError:
                pass
            now = time.monotonic()
            dt = now - last
            last = now
            target = 0.2 * (s2 / 100.0 * 24.0) if s2 >= 0 else 0.0
            if s2 < 0:
                u = max(0.0, u - 2.0 * dt)
            else:
                u += (target - u) * min(1.0, dt / 0.5)
            hdg += 3.0 * (-rud_vessel) * u * dt          # vessel rudder = -simulator rudder
            x += u * math.sin(math.radians(hdg)) * dt
            y += u * math.cos(math.radians(hdg)) * dt
            if self.silent_after_s is None or now - t0 < self.silent_after_s:
                ts = time.strftime("%H:%M:%S", time.localtime()) + ".%06d" % int((time.time() % 1) * 1e6)
                yaw_field = wrap180(-hdg)
                self.sock.sendto(("[%s]%+06.1f,%+06.1f,%+06.1f\n" % (ts, -y, x, yaw_field)).encode(), client)
                if k % 5 == 0:
                    self.sock.sendto(("[%s][%s]\n" % (ts, ",".join(["100"] * 720))).encode(), client)
            k += 1
            time.sleep(0.1)


def selftest():
    import tempfile
    problems = []
    folder = tempfile.mkdtemp(prefix="udp_manoeuvre_selftest_")

    class Quiet:
        def write(self, _):
            pass

        def flush(self):
            pass

    def run_case(argv, silent_after_s=None, name=None):
        vessel = ToyVessel(silent_after_s)
        vessel.start()
        args = build_parser().parse_args(["--server-ip", "127.0.0.1", "--server-port", str(vessel.port),
                                          "--local-port", "0", "--out", os.path.join(folder, (name or argv[0]) + ".log")] + argv)
        finalise(args)
        go = threading.Event()
        go.set()
        meta = Runner(args, out=Quiet()).run(start_event=go)
        vessel.stop_flag.set()
        vessel.join(2.0)
        return meta, vessel

    # steps: every command reaches the vessel and is logged
    meta, vessel = run_case(["steps", "--rpm", "0", "--rudder", "25", "50", "--repeats", "1", "--hold-s", "0.3"])
    sent = [t for _, t in vessel.received if t.startswith("$CMD")]
    with open(meta["log"] + ".commands.csv", encoding="utf-8") as fh:
        logged = [row["message"] for row in csv.DictReader(fh)]
    if meta["ended"] != "complete" or sent != logged:
        problems.append("steps: ended %s, %d sent, %d logged" % (meta["ended"], len(sent), len(logged)))
    if "$CMD,-25.00,0.00" not in sent or "$CMD,25.00,0.00" not in sent:
        problems.append("steps: rudder sign or format wrong: %s" % sent[:4])

    # zigzag: switches alternate and overshoots are reported
    meta, _ = run_case(["zigzag", "--rpm", "5", "--delta", "20", "--psi", "20", "--switches", "4",
                        "--runup-s", "1.0", "--after-s", "1.5"])
    sw = [e for e in meta["events"] if e["kind"] == "switch"]
    over = [e for e in meta["events"] if e["kind"] == "overshoots"]
    if meta["ended"] != "complete" or len(sw) != 4 or [e["rudder"] for e in sw] != [-50.0, 50.0, -50.0, 50.0] or not over:
        problems.append("zigzag: ended %s, switches %s" % (meta["ended"], [(e["rudder"], e["heading_change"]) for e in sw]))

    # stop without --allow-astern is refused; with it, astern until stopped
    try:
        args = build_parser().parse_args(["--server-ip", "127.0.0.1", "stop", "--rpm", "6"])
        finalise(args)
        problems.append("stop: accepted without --allow-astern")
    except SystemExit:
        pass
    meta, vessel = run_case(["stop", "--rpm", "5", "--runup-s", "1.5", "--allow-astern", "--after-s", "0.5",
                             "--min-astern-s", "0.3"])
    kinds = [e["kind"] for e in meta["events"]]
    sent = [t for _, t in vessel.received if t.startswith("$CMD")]
    if meta["ended"] != "complete" or "stopped" not in kinds or "$CMD,0.00,-100.00" not in sent:
        problems.append("stop: ended %s, events %s" % (meta["ended"], kinds))

    # watchdog: telemetry stops, zero commands go out
    meta, vessel = run_case(["hold", "--rpm", "6", "--duration-s", "10", "--watchdog-s", "0.8"], silent_after_s=3.0)
    sent = [t for _, t in vessel.received if t.startswith("$CMD")]
    if not str(meta["ended"]).startswith("abort: telemetry lost") or sent[-1] != "$CMD,0.00,0.00":
        problems.append("watchdog: ended %s, last command %s" % (meta["ended"], sent[-1:]))

    # fence: the vessel runs north out of a 3 m box (1 m margin), zero commands go out
    meta, vessel = run_case(["--fence=-5,5,-5,3", "--fence-margin-m", "1", "hold", "--rpm", "6",
                             "--duration-s", "20"], name="fence")
    sent = [t for _, t in vessel.received if t.startswith("$CMD")]
    if not str(meta["ended"]).startswith("abort: fence") or sent[-1] != "$CMD,0.00,0.00":
        problems.append("fence: ended %s, last command %s" % (meta["ended"], sent[-1:]))

    # sequence from a file
    seq = os.path.join(folder, "seq.csv")
    with open(seq, "w", encoding="utf-8") as fh:
        fh.write("t_s,rudder_pct,rpm\n0,0,6\n0.5,30,6\n1.0,-30,6\n1.5,0,0\n")
    meta, vessel = run_case(["sequence", "--file", seq, "--after-s", "0.3"])
    sent = [t for _, t in vessel.received if t.startswith("$CMD") and not t.startswith("$CMD,0.00,0.00")]
    if meta["ended"] != "complete" or "$CMD,-30.00,25.00" not in sent or "$CMD,30.00,25.00" not in sent:
        problems.append("sequence: ended %s, sent %s" % (meta["ended"], sent[:6]))

    print("[SELFTEST] files in %s" % folder)
    if problems:
        print("[SELFTEST] FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("[SELFTEST] PASSED: steps (sign and format), zig-zag switching on the heading, stop with astern "
          "and its refusal without --allow-astern, telemetry watchdog, basin fence, sequence file")
    return 0


# ---------------------------------------------------------------------------
def common_args(p, suppress):
    """The link and safety options, accepted before or after the manoeuvre name."""
    def d(value):
        return argparse.SUPPRESS if suppress else value
    p.add_argument("--bind-ip", default=d("0.0.0.0"))
    p.add_argument("--local-port", type=int, default=d(5000))
    p.add_argument("--server-ip", default=d("127.0.0.1"), help="the vessel-control computer")
    p.add_argument("--server-port", type=int, default=d(5050))
    p.add_argument("--hello", default=d(HELLO))
    p.add_argument("--hello-every-s", type=float, default=d(2.0), help="handshake repeated until telemetry arrives")
    p.add_argument("--out", default=d(None), help="log file (default manoeuvre_logs/<manoeuvre>_<date>_<time>.log)")
    p.add_argument("--rudder-sign", type=float, default=d(-1.0), help="vessel rudder = sign x rudder (as the bridges)")
    p.add_argument("--rpm-max", type=float, default=d(24.0), help="S2 = rpm / rpm-max x 100 (as the bridges)")
    p.add_argument("--cmd-rate", type=float, default=d(2.0), help="Hz; the current command is repeated this often "
                                                                   "(changes go out at once)")
    p.add_argument("--watchdog-s", type=float, default=d(1.5), help="abort when telemetry stops this long; 0: off")
    p.add_argument("--max-s", type=float, default=d(180.0), help="abort a manoeuvre running longer than this")
    p.add_argument("--speed-window-s", type=float, default=d(1.0), help="speed from pose lines this far apart")
    p.add_argument("--status-s", type=float, default=d(2.0))
    p.add_argument("--tail-s", type=float, default=d(3.0), help="telemetry kept after the final zero command")
    p.add_argument("--fence", default=d(None), metavar="XMIN,XMAX,YMIN,YMAX",
                   help="the basin in the pose frame (x as sent, y negated, as log_parser reads it), "
                        "e.g. the assumed 10 m x 25 m; a manoeuvre is stopped when the pose comes within "
                        "--fence-margin-m of an edge. Write it with '=' (--fence=-10,0,0,25), since a "
                        "negative first value would otherwise read as an option")
    p.add_argument("--fence-margin-m", type=float, default=d(1.5),
                   help="clearance kept from the fence (the pose is the hull reference point; the bow is "
                        "0.86 m ahead of it)")
    p.add_argument("--allow-astern", action="store_true", default=d(False),
                   help="permit negative S2 (only after the ESC test P-8)")
    p.add_argument("--dry-run", action="store_true", default=d(False), help="log the commands but send none")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    common_args(ap, suppress=False)
    ap.add_argument("--selftest", action="store_true", help="loopback test with a stand-in vessel, then exit")
    shared = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common_args(shared, suppress=True)
    sub = ap.add_subparsers(dest="manoeuvre")
    _add_parser = sub.add_parser

    def add_parser(name, **kw):
        return _add_parser(name, parents=[shared], allow_abbrev=False, **kw)

    sub.add_parser = add_parser

    def runup_args(p, runup=15.0):
        p.add_argument("--rpm", type=float, required=True)
        p.add_argument("--runup-s", type=float, default=runup, help="straight run before the manoeuvre "
                                                                    "(90 %% of steady speed: about 14 s at rpm 6, 7 s at rpm 12)")

    p = sub.add_parser("hold", help="a constant command for a time")
    p.add_argument("--rpm", type=float, required=True)
    p.add_argument("--rudder", type=float, default=0.0)
    p.add_argument("--duration-s", type=float, required=True)
    p = sub.add_parser("coast", help="run-up, then zero propulsion")
    runup_args(p)
    p.add_argument("--coast-s", type=float, default=20.0)
    p = sub.add_parser("steps", help="rudder steps")
    p.add_argument("--rpm", type=float, default=0.0)
    p.add_argument("--rudder", type=float, nargs="+", required=True, help="amplitudes, percent")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--hold-s", type=float, default=4.0)
    p.add_argument("--reversal", action="store_true", help="also +a -> -a -> +a -> 0 for each amplitude")
    p = sub.add_parser("ramp", help="rudder ramp")
    p.add_argument("--rpm", type=float, default=0.0)
    p.add_argument("--rate", type=float, default=10.0, help="percent per second")
    p.add_argument("--to", type=float, default=100.0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p = sub.add_parser("turn", help="run-up, then a fixed rudder")
    runup_args(p)
    p.add_argument("--rudder", type=float, required=True)
    p.add_argument("--heading-change", type=float, default=0.0, help="deg; end the turn here (e.g. 720)")
    p.add_argument("--duration-s", type=float, default=0.0, help="or end after this long")
    p.add_argument("--after-s", type=float, default=3.0, help="straight running after the turn")
    p = sub.add_parser("zigzag", help="delta/psi zig-zag on the measured heading")
    runup_args(p)
    p.add_argument("--delta", type=float, required=True, help="rudder angle, deg (of the 40 deg maximum)")
    p.add_argument("--psi", type=float, required=True, help="heading change that triggers each switch, deg")
    p.add_argument("--max-rudder-deg", type=float, default=40.0)
    p.add_argument("--switches", type=int, default=4)
    p.add_argument("--after-s", type=float, default=6.0, help="logging after the last switch, for its overshoot")
    p = sub.add_parser("stop", help="crash stop (needs --allow-astern)")
    runup_args(p)
    p.add_argument("--rudder", type=float, default=0.0, help="settle into a turn at this rudder first")
    p.add_argument("--turn-s", type=float, default=15.0)
    p.add_argument("--astern-s2", type=float, default=-100.0)
    p.add_argument("--stop-speed", type=float, default=0.08, help="m/s, from the pose over --speed-window-s")
    p.add_argument("--min-astern-s", type=float, default=1.0)
    p.add_argument("--max-astern-s", type=float, default=8.0)
    p.add_argument("--after-s", type=float, default=10.0, help="logging at rest after the stop")
    p = sub.add_parser("sequence", help="commands from a CSV file: t_s, rudder_pct, rpm")
    p.add_argument("--file", required=True)
    p.add_argument("--after-s", type=float, default=3.0)
    return ap


def finalise(args):
    """Derived values and checks; raises SystemExit with a plain message."""
    if args.manoeuvre is None:
        raise SystemExit("give a manoeuvre: hold, coast, steps, ramp, turn, zigzag, stop or sequence (-h for help)")
    if args.manoeuvre == "zigzag":
        args.delta_pct = 100.0 * args.delta / args.max_rudder_deg
    if args.manoeuvre == "turn" and not (args.heading_change or args.duration_s):
        raise SystemExit("turn: give --heading-change or --duration-s")
    if args.manoeuvre == "stop":
        if not args.allow_astern:
            raise SystemExit("stop: full astern needs --allow-astern (only after the ESC reverse test, P-8)")
        args.astern_rpm = args.astern_s2 / 100.0 * args.rpm_max
    if args.manoeuvre == "sequence":
        rows = []
        with open(args.file, encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if not row or row[0].strip().startswith("#"):
                    continue
                try:
                    rows.append((float(row[0]), float(row[1]), float(row[2])))
                except ValueError:
                    continue            # a header line
        if not rows:
            raise SystemExit("sequence: no rows in %s" % args.file)
        if any(r[2] < 0 for r in rows) and not args.allow_astern:
            raise SystemExit("sequence: negative rpm needs --allow-astern")
        args.rows = rows
    rpm = getattr(args, "rpm", 0.0)
    if rpm < 0 or rpm > args.rpm_max:
        raise SystemExit("rpm must be within 0 and --rpm-max")
    args.fence_box = None
    if args.fence:
        try:
            x0, x1, y0, y1 = (float(v) for v in args.fence.split(","))
        except ValueError:
            raise SystemExit("--fence takes XMIN,XMAX,YMIN,YMAX")
        m = args.fence_margin_m
        if x1 - x0 <= 2 * m or y1 - y0 <= 2 * m:
            raise SystemExit("--fence is smaller than twice --fence-margin-m")
        args.fence_box = (x0 + m, x1 - m, y0 + m, y1 - m)
    if args.out is None:
        args.out = os.path.join("manoeuvre_logs", "%s_%s.log" % (args.manoeuvre, time.strftime("%Y%m%d_%H%M%S")))
    return args


def main():
    args = build_parser().parse_args()
    if args.selftest:
        return selftest()
    finalise(args)
    meta = Runner(args).run()
    if meta is None:
        return 2
    return 0 if meta.get("ended") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
