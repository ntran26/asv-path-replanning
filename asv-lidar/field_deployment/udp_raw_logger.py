"""
Raw telemetry logger for the Bluefin: the July handshake and a raw log, with no decoder.

Sends the handshake (START, as every bridge here does; the vessel answers with `hey` lines), then
writes every datagram it receives, unchanged, to a log file.  Nothing is decoded on the way to the
file and no command is ever sent, so a new line type (an IMU line, a longer pose line, binary
data) cannot stop the recording.

    python3 udp_raw_logger.py --server-ip 10.201.205.110 --out imu_test_1.log
    python3 udp_raw_logger.py --server-ip 10.201.205.110 --out imu_test_1.log --no-summary
    python3 udp_raw_logger.py --check imu_test_1.log        (line kinds of a saved log; no network)
    python3 udp_raw_logger.py --selftest                    (loopback test of this machine; no vessel)

Needs only Python 3.6 or later: no numpy, pygame, model or simulator, and no other file of this
repository, so the file can be copied on its own to any machine.  Stop with Ctrl+C; each
datagram is on disk as soon as it arrives.

Files (an existing name is never overwritten: _2, _3, ... is added):
  <out>                every datagram as received, one per line (the format of the bridges'
                       telemetry logs and of the July logs, without their #CONFIG line)
  <out>.arrivals.csv   arrival number, wall-clock time, time since start and size of each datagram
  <out>.meta.json      the options, the start and end times and the count of each line kind

Console: a status line every few seconds (datagrams, rate, time since the last one) and a table
of line kinds.  The first line of each new kind is printed in full when it arrives, so a new IMU
line shows its format at once.  Beside each kind is what the bridges' decoder
(`log_parser.BluefinStreamDecoder`) would do with it:
  read                  a LiDAR scan (720 integers), pose, HDG or RC line, as in July
  ignored               the decoder skips the line (a `hey` line, a labelled IMU line, ...)
  taken as a scan       a bracketed list of integers that is not 720 long: the decoder would
                        read it as a LiDAR scan
  decoder error         bracketed values that are not integers: the decoder raises, which stops
                        the July bridge (udp_live_rl.py); udp_live_p3.py logs the line and skips it
The table is computed after the datagram is on disk, inside a guard; --no-summary turns it off
and leaves only the recording and the status line.
"""

import argparse
import collections
import json
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone

HELLO = "START"
LIDAR_RAW_BEAMS = 720

# The decoder's four line formats, as in log_parser.BluefinStreamDecoder (copied so that this
# program needs nothing outside the standard library; --selftest checks they are still equal
# when log_parser can be imported).
RE_HDG = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<seq>\d+)\]\s*HDG:(?P<hdg>[-+]?\d+(?:\.\d+)?)\s*$")
RE_POSE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]"
    r"(?P<y>[-+]\d+\.\d+),(?P<x>[-+]\d+\.\d+),(?P<yaw>[-+]\d+\.\d+)\s*$")
RE_RC = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<seq>\d+)\]\s*"
    r"S1:(?P<s1>\d+)\s*S2:(?P<s2>\d+)\s*RC\s*$")
RE_LIDAR = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<body>.*)\]\s*$")

RE_TS = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d+)\]")
RE_INT = re.compile(r"[-+]?\d+")
RE_NUMBERS = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)+\s*$")
RE_SEQ_WORD = re.compile(r"^\[\d+\]\s*([A-Za-z_][A-Za-z0-9_]*)")
RE_WORD = re.compile(r"^\s*#?\s*([A-Za-z_][A-Za-z0-9_]*)")
MAX_KINDS = 40


# ---------------------------------------------------------------------------
# Line kinds (screen and meta file only; the recording never depends on them)
# ---------------------------------------------------------------------------
def vessel_seconds(line):
    """Seconds of day from the vessel's [HH:MM:SS.ffffff] stamp, or None."""
    m = RE_TS.match(line)
    if m is None:
        return None
    hh, mm, ss, frac = m.groups()
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(frac) / 10.0 ** len(frac)


def classify(line):
    """(kind, what the bridges' decoder does with the line)."""
    s = line.strip()
    if not s:
        return "empty line", "ignored"
    if RE_RC.match(s):
        return "RC", "read"
    if RE_HDG.match(s):
        return "HDG", "read"
    if RE_POSE.match(s):
        return "pose (3 values)", "read"
    m = RE_LIDAR.match(s)
    if m:
        values = [p.strip() for p in m.group("body").split(",")]
        values = [p for p in values if p]
        if not values:
            return "empty bracketed list", "decoder error"
        if all(RE_INT.fullmatch(p) for p in values):
            if len(values) == LIDAR_RAW_BEAMS:
                return "LiDAR scan (720)", "read"
            return "bracketed list of %d integers" % len(values), "taken as a scan"
        word = RE_SEQ_WORD.match(m.group(0)[m.end("ts") + 1:])
        label = word.group(1) if word else "values"
        return "bracketed %s line" % label, "decoder error"
    t = RE_TS.match(s)
    if t:
        rest = s[t.end():]
        word = RE_SEQ_WORD.match(rest)
        if word:
            return "[seq] %s" % word.group(1), "ignored"
        if RE_NUMBERS.match(rest):
            n = len(rest.split(","))
            return "numbers (%d values)" % n, "ignored"
        word = RE_WORD.match(rest)
        return ("stamped %s" % word.group(1)) if word else "stamped, other", "ignored"
    if s.lower() == "hey":
        return "hey", "ignored"
    word = RE_WORD.match(s)
    if word:
        return ("#%s" % word.group(1)) if s.startswith("#") else word.group(1), "ignored"
    return "other", "ignored"


class Kinds:
    """Counts, first example and rate of each line kind."""

    def __init__(self):
        self.rows = collections.OrderedDict()

    def add(self, line, t_arrival):
        kind, verdict = classify(line)
        if kind not in self.rows and len(self.rows) >= MAX_KINDS:
            kind, verdict = "other (kind limit reached)", "?"
        row = self.rows.get(kind)
        new = row is None
        if new:
            row = self.rows[kind] = {"verdict": verdict, "count": 0, "example": line.strip()[:300],
                                     "first_arrival": t_arrival, "last_arrival": t_arrival,
                                     "first_vessel": None, "last_vessel": None}
        row["count"] += 1
        row["last_arrival"] = t_arrival
        tv = vessel_seconds(line.strip())
        if tv is not None:
            if row["first_vessel"] is None:
                row["first_vessel"] = tv
            row["last_vessel"] = tv
        return (kind, row) if new else None

    @staticmethod
    def rate(row):
        """Lines per second, by the vessel's stamps when the kind carries them."""
        n = row["count"]
        if n < 2:
            return None
        if row["first_vessel"] is not None and row["last_vessel"] is not None:
            span = row["last_vessel"] - row["first_vessel"]
        else:
            span = row["last_arrival"] - row["first_arrival"]
        return (n - 1) / span if span > 0 else None

    def table(self):
        out = ["  %-34s %8s %9s  %s" % ("line kind", "count", "rate /s", "decoder")]
        for kind, row in self.rows.items():
            r = self.rate(row)
            out.append("  %-34s %8d %9s  %s" % (kind[:34], row["count"], "-" if r is None else "%.1f" % r,
                                                row["verdict"]))
        return "\n".join(out)

    def as_dict(self):
        return {k: dict(v, rate_per_s=self.rate(v)) for k, v in self.rows.items()}


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def free_name(path):
    """`path`, or `path` with _2, _3, ... before the extension if it exists."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 2
    while os.path.exists("%s_%d%s" % (stem, k, ext)):
        k += 1
    return "%s_%d%s" % (stem, k, ext)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def record(args, *, on_bound=None, stop_event=None, out=sys.stdout):
    """Run the handshake and record until Ctrl+C, --duration-s or `stop_event`.  Returns the meta dict."""
    def say(text):
        try:
            out.write(text + "\n")
            out.flush()
        except Exception:
            pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:                                    # room for bursts (the system may cap it)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except OSError:
        pass
    try:
        sock.bind((args.bind_ip, args.local_port))
    except OSError as exc:
        say("[RAW] cannot listen on %s:%d (%s).  Another program holds the port: close the other bridge "
            "or viewer, or give --local-port." % (args.bind_ip, args.local_port, exc))
        sock.close()
        return None
    local_port = sock.getsockname()[1]
    if on_bound is not None:
        on_bound(local_port)
    server = (args.server_ip, args.server_port)

    out_path = free_name(args.out)
    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)
    log = open(out_path, "wb", buffering=0)
    arrivals = open(out_path + ".arrivals.csv", "w", encoding="utf-8", buffering=1)
    arrivals.write("n,t_unix_s,t_since_start_s,bytes\n")
    meta_path = out_path + ".meta.json"
    meta = {"program": "udp_raw_logger.py", "log": os.path.abspath(out_path), "started_utc": utc_now(),
            "options": vars(args), "python": sys.version.split()[0], "finished_utc": None}

    def write_meta():
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)
        except Exception as exc:
            say("[RAW] could not write %s: %s" % (meta_path, exc))

    write_meta()
    say("[RAW] recording to %s" % os.path.abspath(out_path))
    say("[RAW] listening on %s:%d; handshake '%s' to %s:%d; no command is ever sent; Ctrl+C stops"
        % (args.bind_ip, local_port, args.hello, server[0], server[1]))

    kinds = Kinds() if not args.no_summary else None
    summary_errors = 0
    n = 0
    n_bytes = 0
    hellos = 0
    hey_seen = False
    t_start = time.monotonic()
    t_first = None
    t_last = None
    t_hello = None
    t_status = t_start
    t_table = t_start
    deadline = t_start + args.duration_s if args.duration_s > 0 else None
    sock.settimeout(0.2)
    stopped_by = "ctrl+c"

    def hello():
        nonlocal hellos, t_hello
        try:
            sock.sendto((args.hello + "\n").encode(), server)
            hellos += 1
        except OSError as exc:
            say("[RAW] handshake not sent (%s): check --server-ip and the network" % exc)
        t_hello = time.monotonic()

    hello()
    try:
        while True:
            now = time.monotonic()
            if stop_event is not None and stop_event.is_set():
                stopped_by = "stop event"
                break
            if deadline is not None and now >= deadline:
                stopped_by = "--duration-s"
                break
            if t_first is None and args.hello_every_s > 0 and now - t_hello >= args.hello_every_s:
                hello()
                say("[RAW] nothing received yet; handshake sent again (%d so far)" % hellos)
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                data = None
            except ConnectionResetError:        # Windows: an earlier datagram found no listener
                data = None
            except OSError as exc:
                say("[RAW] receive error (%s); still listening" % exc)
                data = None

            now = time.monotonic()
            if data is not None:
                # 1. On disk first, exactly as received.
                log.write(data if data.endswith(b"\n") else data + b"\n")
                n += 1
                n_bytes += len(data)
                arrivals.write("%d,%.6f,%.6f,%d\n" % (n, time.time(), now - t_start, len(data)))
                if t_first is None:
                    t_first = now
                    say("[RAW] first datagram after %.1f s" % (now - t_start))
                t_last = now
                # 2. Then, for the screen only, the line kinds.
                if kinds is not None:
                    try:
                        text = data.decode("utf-8", errors="replace")
                        for line in text.splitlines() or [""]:
                            if not hey_seen and line.strip().lower() == "hey":
                                hey_seen = True
                                say("[RAW] the vessel answered: hey")
                            new = kinds.add(line, now)
                            if new is not None:
                                kind, row = new
                                say("[RAW] new line kind: %s  (decoder: %s)\n      %s"
                                    % (kind, row["verdict"], row["example"][:200]))
                    except Exception as exc:
                        summary_errors += 1
                        if summary_errors == 1:
                            say("[RAW] summary error (the recording continues): %r" % (exc,))

            if now - t_status >= args.status_s:
                t_status = now
                if t_first is None:
                    say("[RAW] %5.1f s  waiting for the vessel: nothing received" % (now - t_start))
                else:
                    span = (t_last - t_first) if t_last is not None else 0.0
                    rate = (n - 1) / span if n > 1 and span > 0 else 0.0
                    quiet = now - t_last
                    say("[RAW] %5.1f s  datagrams %d (%.1f/s)  %.1f kB  last %.1f s ago%s"
                        % (now - t_start, n, rate, n_bytes / 1e3, quiet,
                           "   NO DATA" if quiet > 2.0 else ""))
            if kinds is not None and args.table_s > 0 and now - t_table >= args.table_s and n:
                t_table = now
                say(kinds.table())
    except KeyboardInterrupt:
        stopped_by = "ctrl+c"
    finally:
        sock.close()
        log.close()
        arrivals.close()
        meta.update(finished_utc=utc_now(), stopped_by=stopped_by, datagrams=n, bytes=n_bytes,
                    handshakes_sent=hellos, vessel_answered_hey=hey_seen,
                    seconds_to_first_datagram=None if t_first is None else t_first - t_start,
                    seconds_recorded=time.monotonic() - t_start, summary_errors=summary_errors,
                    line_kinds=kinds.as_dict() if kinds is not None else None)
        write_meta()
        if kinds is not None and n:
            say(kinds.table())
        say("[RAW] stopped (%s): %d datagrams, %.1f kB in %s" % (stopped_by, n, n_bytes / 1e3,
                                                                os.path.abspath(out_path)))
    return meta


# ---------------------------------------------------------------------------
# --check: the line kinds of a saved log
# ---------------------------------------------------------------------------
def check_file(path):
    kinds = Kinds()
    t = 0.0                                 # a line without a stamp takes the last stamp seen
    with open(path, "rb") as fh:            # (or the first one, before any)
        for raw in fh:
            tv = vessel_seconds(raw.decode("utf-8", errors="replace").strip())
            if tv is not None:
                t = tv
                break
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            tv = vessel_seconds(line.strip())
            t = tv if tv is not None else t
            kinds.add(line, t)
    print("%s: %d lines" % (path, sum(r["count"] for r in kinds.rows.values())))
    print(kinds.table())
    for kind, row in kinds.rows.items():
        print("  %s: %s" % (kind, row["example"][:160]))
    return 0


# ---------------------------------------------------------------------------
# --selftest: loopback, with a stand-in vessel on this machine
# ---------------------------------------------------------------------------
SAMPLE_LINES = [
    "hey",
    "[10:00:00.000000]-000.0,+000.1,-007.6",
    "[10:00:00.010000][" + ",".join(["%3d" % (100 + i % 50) for i in range(LIDAR_RAW_BEAMS)]) + "]",
    "[10:00:00.020000][00123] IMU:0.012,-0.034,9.810,0.10,-0.20,0.30",
    "[10:00:00.030000][0.012,-0.034,9.810,0.10,-0.20,0.30]",
    "[10:00:00.040000][12,-3,981,1,2,3]",
    "[10:00:00.050000][00124] IMU:[0.012,-0.034]",
    "[10:00:00.060000]-000.0,+000.1,-007.6,+0.012,-0.034",
    "[10:00:00.070000][]",
    "IMU,0.012,-0.034,9.810",
    "[10:00:00.080000][01935] HDG:350.75",
    "[10:00:00.090000][01936] S1:1500 S2:1500 RC",
]
SAMPLE_VERDICTS = ["ignored", "read", "read", "ignored", "decoder error", "taken as a scan", "decoder error",
                   "ignored", "decoder error", "ignored", "read", "read"]


def selftest():
    import tempfile
    problems = []

    # 1. The copied formats against log_parser's, where it can be imported.
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        from log_parser import BluefinStreamDecoder as D
        for name, mine in (("_re_hdg", RE_HDG), ("_re_pose", RE_POSE), ("_re_rc", RE_RC), ("_re_lidar", RE_LIDAR)):
            if getattr(D, name).pattern != mine.pattern:
                problems.append("format %s differs from log_parser's" % name)
        print("[SELFTEST] formats equal to log_parser's" if not problems else "[SELFTEST] " + "; ".join(problems))
    except Exception as exc:
        print("[SELFTEST] log_parser not importable here (%s): format comparison skipped" % type(exc).__name__)

    # 2. The verdicts on the sample lines.
    for line, want in zip(SAMPLE_LINES, SAMPLE_VERDICTS):
        kind, got = classify(line)
        if got != want:
            problems.append("classify(%r) -> %s, expected %s" % (line[:40], got, want))

    # 3. Loopback: a stand-in vessel answers the handshake and streams the samples, a binary
    #    datagram and a two-line datagram; the log must hold every byte sent.
    vessel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    vessel.bind(("127.0.0.1", 0))
    vessel.settimeout(10.0)
    sent = []
    stop = threading.Event()

    def run_vessel():
        try:
            data, client = vessel.recvfrom(4096)
            if data.strip() != HELLO.encode():
                problems.append("vessel got %r, not the handshake" % data)
            payloads = [(line + "\n").encode() for line in SAMPLE_LINES]
            payloads += [b"\x00\xff\xfe binary \x81\n", b"[10:00:00.100000][00125] IMU:1,2,3\n[10:00:00.100100]-000.0,+000.1,-007.6\n"]
            for p in payloads:
                vessel.sendto(p, client)
                sent.append(p)
                time.sleep(0.01)
            time.sleep(0.3)
        except Exception as exc:
            problems.append("vessel: %r" % (exc,))
        finally:
            stop.set()

    folder = tempfile.mkdtemp(prefix="udp_raw_logger_selftest_")
    args = argparse.Namespace(bind_ip="127.0.0.1", local_port=0, server_ip="127.0.0.1",
                              server_port=vessel.getsockname()[1], hello=HELLO,
                              out=os.path.join(folder, "selftest.log"), hello_every_s=1.0, duration_s=15.0,
                              no_summary=False, status_s=60.0, table_s=0.0)
    thread = threading.Thread(target=run_vessel, daemon=True)
    thread.start()

    class Quiet:
        def write(self, _):
            pass

        def flush(self):
            pass

    meta = record(args, stop_event=stop, out=Quiet())
    thread.join(5.0)
    vessel.close()
    if meta is None:
        problems.append("the logger could not listen on a loopback port")
    else:
        with open(meta["log"], "rb") as fh:
            got = fh.read()
        if got != b"".join(sent):
            problems.append("the log differs from the bytes sent")
        if meta["datagrams"] != len(sent):
            problems.append("%d datagrams logged, %d sent" % (meta["datagrams"], len(sent)))
        if not meta["vessel_answered_hey"]:
            problems.append("the 'hey' answer was not noticed")
        with open(meta["log"] + ".arrivals.csv", encoding="utf-8") as fh:
            if len(fh.read().splitlines()) != len(sent) + 1:
                problems.append("arrivals.csv has the wrong number of rows")
    print("[SELFTEST] files in %s" % folder)
    if problems:
        print("[SELFTEST] FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("[SELFTEST] PASSED: handshake, %d datagrams logged byte for byte (LiDAR, pose, IMU variants, "
          "binary, two lines in one datagram), line kinds as expected" % len(sent))
    return 0


# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--local-port", type=int, default=5000)
    ap.add_argument("--server-ip", default="127.0.0.1", help="the vessel-control computer")
    ap.add_argument("--server-port", type=int, default=5050)
    ap.add_argument("--hello", default=HELLO, help="handshake text (default START, as in July)")
    ap.add_argument("--hello-every-s", type=float, default=2.0,
                    help="send the handshake again this often until the first datagram arrives; 0: once")
    ap.add_argument("--out", default=None,
                    help="log file (default raw_logs/raw_<date>_<time>.log); an existing name gets _2, _3, ...")
    ap.add_argument("--duration-s", type=float, default=0.0, help="stop after this long; 0: until Ctrl+C")
    ap.add_argument("--no-summary", action="store_true", help="no line kinds: record and status line only")
    ap.add_argument("--status-s", type=float, default=2.0, help="seconds between status lines")
    ap.add_argument("--table-s", type=float, default=10.0, help="seconds between line-kind tables; 0: at the end only")
    ap.add_argument("--check", metavar="LOG", default=None, help="print the line kinds of a saved log and exit")
    ap.add_argument("--selftest", action="store_true", help="loopback test with a stand-in vessel, then exit")
    return ap


def main():
    args = build_parser().parse_args()
    if args.selftest:
        return selftest()
    if args.check:
        return check_file(args.check)
    if args.out is None:
        args.out = os.path.join("raw_logs", time.strftime("raw_%Y%m%d_%H%M%S.log"))
    meta = record(args)
    return 0 if meta is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
