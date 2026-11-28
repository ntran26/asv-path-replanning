"""
Relay the vessel's ROS 2 topics to the field tools as lines of the vessel's raw log, over UDP.

Runs on the shore laptop with ROS 2 Jazzy (Python 3.12, after `source /opt/ros/jazzy/setup.bash`),
on the vessel's network and with the vessel's ROS_DOMAIN_ID.  It subscribes to the vessel's topics
and, towards the field tools, behaves as the vessel's own UDP link does: a tool sends START to the
relay's port (127.0.0.1:5051 by default), the relay answers "hey" and streams lines to it, one line
per datagram.  Several tools can listen at once (a logger and the bridge), each on its own port.

    [HH:MM:SS.ffffff]A,B,C                   /map_odom: x, y (m) and yaw (deg, counter-clockwise),
                                              base_link in the map frame, 0.1 steps
    [HH:MM:SS.ffffff][ r0, r1, ..., r719]     /scan: range x 10, truncated (0.1 m steps); no return
                                              and returns of 1.0 m or less are 0
    [HH:MM:SS.ffffff]<ms>,GYRO,x,y,z          /imu/data_raw: angular rate, deg/s
    [HH:MM:SS.ffffff]<ms>,ACCEL,x,y,z         /imu/data_raw: linear acceleration, m/s^2
    [HH:MM:SS.ffffff]<ms>,QUAT,x,y,z,w        /imu/data_raw: orientation

Pose and LiDAR lines are the vessel's own: on the recording of 2026-11-26 (rosbag2_2026_11_26-
11_08_31 against RAW_LOG.log) every pose line and 99.9 % of the LiDAR lines are reproduced
character for character (ros2_relay_check.py).  The IMU lines carry /imu/data_raw, which is not the
source of the vessel's GYRO/ACCEL/QUAT lines: those come from an on-board IMU with no ROS topic in
the recording (nor do the SRVO lines, which the relay cannot give).  The two agree on yaw rate
(correlation 0.97), not on acceleration or the quaternion's heading.  log_parser reads every line.

Rates.  Poses: every /map_odom message (about 86 Hz).  LiDAR: --lidar-hz 2 (default), one scan per
0.5 s, the bridge's decision interval (the bridge decides on every LiDAR line); 0 gives every scan
(10 Hz) for logging.  IMU: --imu-hz 10 (default, as the vessel's log); 0 gives every message (100 Hz).
Line stamps are this laptop's clock when the message arrives (--stamp receive, as the vessel's log
stamps its own arrival times) or the message's own stamp (--stamp header).

Pose layout.  --pose-layout vessel (default) writes A, B, C as the vessel does since November 2026
(read with --pose-map nov2026).  --pose-layout july writes -A, -B, C, which a reader in the July
layout (the bridge's decoder before its switch) decodes to the same position and heading.

Commands.  A `$CMD` datagram from a tool goes on to --vessel IP:PORT (the vessel's command port) if
given; otherwise it is counted and dropped.  The relay itself never sends a command.

    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=<the vessel's>
    python3 ros2_relay.py --log 2026-12-03/relay-01.log
    python3 ros2_relay.py --vessel 192.168.8.197:5050 --log 2026-12-03/relay-02.log
    python udp_raw_logger.py --server-ip 127.0.0.1 --server-port 5051 --out 2026-12-03/S1a-01.log

A recorded bag becomes a raw log, with the bag's arrival times as stamps (no network, no ROS graph):

    python3 ros2_relay.py --bag ~/rosbag2_2026_11_26-11_08_31 --log relay_from_bag.log --lidar-hz 0

--selftest checks the line formats and the UDP link (handshake, streaming to two tools, command
forwarding) on loopback ports, without ROS.
"""

import argparse
import datetime
import math
import os
import socket
import sys
import time

import numpy as np

LISTEN_PORT = 5051
LIDAR_MIN_M = 1.0              # the vessel writes 0 for returns of this or less (RAW_LOG, 2026-11-26)
IMU_COUNTER_MOD = 1_000_000_000


# ----------------------------------------------------------------------------- line formats
def stamp_text(t: float) -> str:
    """[HH:MM:SS.ffffff] in this computer's local time, as the vessel stamps its lines."""
    return datetime.datetime.fromtimestamp(t).strftime("[%H:%M:%S.%f]")


def yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def pose_line(t: float, x: float, y: float, yaw: float, layout: str = "vessel", decimals: int = 1) -> str:
    """A pose line from the map-frame position (m) and yaw (deg, counter-clockwise)."""
    if layout == "july":
        x, y = -x, -y
    f = "%+0{}.{}f".format(5 + decimals, decimals)
    return stamp_text(t) + ",".join(f % v for v in (x, y, yaw))


def lidar_values(ranges) -> np.ndarray:
    """The vessel's LiDAR values: range x 10 truncated; no return (inf, nan) and <= 1.0 m give 0."""
    r = np.asarray(ranges, dtype=np.float32)
    r = np.where(np.isfinite(r) & (r > np.float32(LIDAR_MIN_M)), r, np.float32(0.0))
    return np.floor(r * np.float32(10.0)).astype(np.int64)


def lidar_line(t: float, ranges) -> str:
    return stamp_text(t) + "[" + ",".join("%3d" % v for v in lidar_values(ranges)) + "]"


def imu_lines(t: float, stamp_s: float, gyro_rad_s, accel_m_s2, quat_xyzw):
    """GYRO (deg/s), ACCEL (m/s^2) and QUAT lines with a millisecond counter from the message stamp."""
    counter = "%09d" % (int(round(stamp_s * 1000.0)) % IMU_COUNTER_MOD)
    v = lambda vals: ",".join("%+011.6f" % float(x) for x in vals)
    ts = stamp_text(t)
    return [ts + counter + ",GYRO," + v(math.degrees(g) for g in gyro_rad_s),
            ts + counter + ",ACCEL," + v(accel_m_s2),
            ts + counter + ",QUAT," + v(quat_xyzw)]


class RateGate:
    """Passes one message per 1/hz s on average, by the messages' own stamps (0: every message),
    never two closer than 0.85/hz: at 2 Hz from scans 0.098 s apart, intervals of 0.49 or 0.59 s
    (the bridge skips a LiDAR line less than 0.4 s after the last decision's)."""

    MIN_SPACING = 0.85

    def __init__(self, hz: float) -> None:
        self.period = 1.0 / hz if hz > 0 else 0.0
        self.due = None
        self.last = None

    def __call__(self, t: float) -> bool:
        if self.period <= 0:
            return True
        if self.due is None or t - self.due > 2 * self.period or t < self.due - 2 * self.period:
            self.due, self.last = t, None        # start, a gap, or the clock jumped back
        if t + 1e-6 < self.due:
            return False
        if self.last is not None and t - self.last < self.MIN_SPACING * self.period:
            return False
        self.due += self.period
        self.last = t
        return True


# ----------------------------------------------------------------------------- UDP link
class UdpLink:
    """The vessel's side of the UDP link, towards the field tools."""

    def __init__(self, listen_ip: str, listen_port: int, vessel=None, hey_hz: float = 2.0, log_path=None,
                 say=print) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((listen_ip, listen_port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.vessel = vessel
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if vessel else None
        self.clients = []
        self.hey_period = 1.0 / hey_hz if hey_hz > 0 else 0.0
        self.t_hey = 0.0
        self.say = say
        self.log = open(log_path, "a", buffering=1) if log_path else None
        self.counts = {"pose": 0, "LiDAR": 0, "IMU": 0, "sent": 0, "cmd_forwarded": 0, "cmd_dropped": 0,
                       "send_errors": 0}

    def poll(self) -> None:
        """Handshakes and commands from the tools; the periodic "hey"."""
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                break
            except ConnectionResetError:
                continue
            text = data.decode("utf-8", errors="replace").strip()
            if text.startswith("$"):
                if self.cmd_sock is not None:
                    self.cmd_sock.sendto(data, self.vessel)
                    self.counts["cmd_forwarded"] += 1
                else:
                    self.counts["cmd_dropped"] += 1
                    if self.counts["cmd_dropped"] == 1:
                        self.say("[RELAY] a command arrived and was dropped (no --vessel): %s" % text[:60])
                continue
            if addr not in self.clients:
                self.clients.append(addr)
                self.say("[RELAY] %s from %s:%d; streaming to it (%d listening)" % (text[:20] or "datagram",
                                                                                   addr[0], addr[1], len(self.clients)))
            self._send("hey", addr)
        now = time.monotonic()
        if self.hey_period and now - self.t_hey >= self.hey_period:
            self.t_hey = now
            for addr in list(self.clients):
                self._send("hey", addr)

    def _send(self, line: str, addr) -> None:
        try:
            self.sock.sendto((line + "\n").encode(), addr)
            self.counts["sent"] += 1
        except OSError:
            self.counts["send_errors"] += 1

    def emit(self, kind: str, line: str) -> None:
        self.counts[kind] += 1
        if self.log is not None:
            self.log.write(line + "\n")
        for addr in list(self.clients):
            self._send(line, addr)

    def close(self) -> None:
        if self.log is not None:
            self.log.close()
        self.sock.close()
        if self.cmd_sock is not None:
            self.cmd_sock.close()


# ----------------------------------------------------------------------------- formatting a message
class Formatter:
    def __init__(self, args) -> None:
        self.args = args
        self.lidar_gate = RateGate(args.lidar_hz)
        self.imu_gate = RateGate(args.imu_hz)

    @staticmethod
    def msg_stamp(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def line_time(self, msg, t_arrival: float) -> float:
        return self.msg_stamp(msg) if self.args.stamp == "header" else t_arrival

    def pose(self, msg, t_arrival):
        p = msg.pose.pose
        q = p.orientation
        return [("pose", pose_line(self.line_time(msg, t_arrival), p.position.x, p.position.y,
                                   yaw_deg(q.x, q.y, q.z, q.w), self.args.pose_layout, self.args.pose_decimals))]

    def scan(self, msg, t_arrival):
        if not self.lidar_gate(self.msg_stamp(msg)):
            return []
        return [("LiDAR", lidar_line(self.line_time(msg, t_arrival), msg.ranges))]

    def imu(self, msg, t_arrival):
        if self.args.imu_hz < 0 or not self.imu_gate(self.msg_stamp(msg)):
            return []
        g, a, q = msg.angular_velocity, msg.linear_acceleration, msg.orientation
        return [("IMU", s) for s in imu_lines(self.line_time(msg, t_arrival), self.msg_stamp(msg),
                                              (g.x, g.y, g.z), (a.x, a.y, a.z), (q.x, q.y, q.z, q.w))]


# ----------------------------------------------------------------------------- live: ROS 2 subscriptions
def run_live(args) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, LaserScan
    except ImportError as exc:
        sys.exit("[RELAY] ROS 2 is not loaded (%s): run `source /opt/ros/jazzy/setup.bash` first, with python3" % exc)
    vessel = parse_addr(args.vessel) if args.vessel else None
    link = UdpLink(args.listen_ip, args.listen_port, vessel, args.hey_hz, args.log)
    fmt = Formatter(args)
    rclpy.init()
    node = Node("asv_shore_relay")
    if args.reliable:
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    else:
        qos = qos_profile_sensor_data

    def cb(kind_fn):
        def f(msg):
            for kind, line in kind_fn(msg, time.time()):
                link.emit(kind, line)
        return f

    node.create_subscription(Odometry, args.pose_topic, cb(fmt.pose), qos)
    node.create_subscription(LaserScan, args.scan_topic, cb(fmt.scan), qos)
    if args.imu_hz >= 0:
        node.create_subscription(Imu, args.imu_topic, cb(fmt.imu), qos)
    node.create_timer(0.01, link.poll)
    last = {"t": time.monotonic(), "c": dict(link.counts)}

    def report():
        now, c = time.monotonic(), dict(link.counts)
        dt = now - last["t"]
        rates = ", ".join("%s %.1f/s" % (k, (c[k] - last["c"][k]) / dt) for k in ("pose", "LiDAR", "IMU"))
        print("[RELAY] %s; %d listening; commands forwarded %d, dropped %d" % (rates, len(link.clients),
                                                                          c["cmd_forwarded"], c["cmd_dropped"]))
        if c["pose"] == last["c"]["pose"]:
            print("[RELAY] no %s message in the last %.0f s: check ROS_DOMAIN_ID (%s), the network and "
                  "`ros2 topic list`" % (args.pose_topic, dt, os.environ.get("ROS_DOMAIN_ID", "0, not set")))
        last["t"], last["c"] = now, c

    node.create_timer(args.report_s, report)
    print("[RELAY] listening for START on %s:%d; topics %s, %s, %s; LiDAR %s, IMU %s; pose layout %s; "
          "ROS_DOMAIN_ID %s; commands %s" % (args.listen_ip, link.port, args.pose_topic, args.scan_topic,
                                             args.imu_topic, rate_text(args.lidar_hz), rate_text(args.imu_hz),
                                             args.pose_layout, os.environ.get("ROS_DOMAIN_ID", "0 (not set)"),
                                             "to %s:%d" % vessel if vessel else "dropped (no --vessel)"))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print("[RELAY] stopped: %s" % ", ".join("%s %d" % kv for kv in link.counts.items()))
        link.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ----------------------------------------------------------------------------- offline: a bag to a raw log
def run_bag(args) -> None:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        sys.exit("[RELAY] ROS 2 is not loaded (%s): run `source /opt/ros/jazzy/setup.bash` first, with python3" % exc)
    if not args.log:
        sys.exit("[RELAY] --bag needs --log")
    meta = os.path.join(os.path.expanduser(args.bag), "metadata.yaml")
    storage = "mcap" if os.path.exists(meta) and "mcap" in open(meta).read() else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=os.path.expanduser(args.bag), storage_id=storage),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    fmt = Formatter(args)
    handlers = {args.pose_topic: fmt.pose, args.scan_topic: fmt.scan}
    if args.imu_hz >= 0:
        handlers[args.imu_topic] = fmt.imu
    missing = [t for t in handlers if t not in types]
    if missing:
        sys.exit("[RELAY] not in the bag: %s" % ", ".join(missing))
    classes = {t: get_message(types[t]) for t in handlers}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(handlers)))
    counts = {"pose": 0, "LiDAR": 0, "IMU": 0}
    with open(args.log, "w") as out:
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            for kind, line in handlers[topic](deserialize_message(data, classes[topic]), t_ns * 1e-9):
                out.write(line + "\n")
                counts[kind] += 1
    print("[RELAY] %s -> %s: %s" % (args.bag, args.log, ", ".join("%s %d" % kv for kv in counts.items())))


# ----------------------------------------------------------------------------- self-test (no ROS)
def selftest() -> int:
    from types import SimpleNamespace as NS
    problems = []
    t = datetime.datetime(2026, 11, 26, 11, 8, 36, 543843).timestamp()
    # Formats, against lines of the vessel's own log
    if pose_line(t, -0.04, 0.0, -5.6) != "[11:08:36.543843]-000.0,+000.0,-005.6":
        problems.append("pose line: %s" % pose_line(t, -0.04, 0.0, -5.6))
    if pose_line(t, 8.94, -3.81, 21.94, "july") != "[11:08:36.543843]-008.9,+003.8,+021.9":
        problems.append("pose line, July layout")
    v = lidar_values([float("inf"), 0.99, 1.0, 1.0005, 9.57, 15.29, float("nan"), 0.3])
    if v.tolist() != [0, 0, 0, 10, 95, 152, 0, 0]:
        problems.append("LiDAR values %s" % v.tolist())
    if not lidar_line(t, [9.5, 0.2, 10.0]).endswith("[ 95,  0,100]"):
        problems.append("LiDAR line %s" % lidar_line(t, [9.5, 0.2, 10.0]))
    g = imu_lines(t, 6685.016, (0.0, 0.0, math.radians(-0.0625)), (0.03, 0.03, -0.01), (-0.037537, -0.017395, 0.737122, 0.674438))
    want = ["[11:08:36.543843]006685016,GYRO,+000.000000,+000.000000,-000.062500",
            "[11:08:36.543843]006685016,ACCEL,+000.030000,+000.030000,-000.010000",
            "[11:08:36.543843]006685016,QUAT,-000.037537,-000.017395,+000.737122,+000.674438"]
    if g != want:
        problems.append("IMU lines %s" % g)
    gate = RateGate(2.0)
    passed = [round(k * 0.0977, 4) for k in range(60) if gate(k * 0.0977)]
    if len(passed) != 12 or max(np.diff(passed)) > 0.6 or min(np.diff(passed)) < 0.425:
        problems.append("LiDAR gate %s" % passed)
    # The parser reads every kind
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import io, contextlib
    import log_parser
    dec = log_parser.BluefinStreamDecoder.raw_format()
    lines = [pose_line(t, 1.0, 2.0, 30.0)] + g + [lidar_line(t + 0.05, [5.0] * 720)]
    frames = []
    with contextlib.redirect_stdout(io.StringIO()):
        for line in lines:
            fr = dec.feed(line)
            if fr is not None:
                frames.append(fr)
    if len(frames) != 1 or frames[0].gyro_dps is None or abs(frames[0].gyro_dps[2] + 0.0625) > 1e-9:
        problems.append("log_parser did not read the relay's lines as expected")
    else:
        x, y, h = frames[0].x_m, frames[0].y_m, frames[0].yaw_deg
        if (round(x, 3), round(y, 3), round(h % 360, 3)) != (-2.0, 1.0, 330.0):
            problems.append("log_parser nov2026 reading of the pose: %s" % ((x, y, h),))
    # UDP: two tools, handshake, streaming, command forwarding
    vessel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    vessel.bind(("127.0.0.1", 0))
    vessel.settimeout(1.0)
    link = UdpLink("127.0.0.1", 0, vessel.getsockname(), hey_hz=0, say=lambda s: None)
    tools = []
    for _ in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        s.settimeout(1.0)
        s.sendto(b"START\n", ("127.0.0.1", link.port))
        tools.append(s)
    time.sleep(0.05)
    link.poll()
    for line in lines:
        link.emit("pose", line)
    for k, s in enumerate(tools):
        got = []
        try:
            while len(got) < 1 + len(lines):
                got.append(s.recvfrom(65535)[0].decode().rstrip("\n"))
        except socket.timeout:
            pass
        if got != ["hey"] + lines:
            problems.append("tool %d received %d of %d lines" % (k, len(got), 1 + len(lines)))
    tools[0].sendto(b"$CMD,-0.10,50", ("127.0.0.1", link.port))
    time.sleep(0.05)
    link.poll()
    try:
        fwd = vessel.recvfrom(65535)[0]
    except socket.timeout:
        fwd = b""
    if fwd != b"$CMD,-0.10,50":
        problems.append("command forwarded as %r" % fwd)
    link.close()
    vessel.close()
    for s in tools:
        s.close()
    if problems:
        print("[SELFTEST] FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("[SELFTEST] PASSED: pose, LiDAR and IMU lines as the vessel's log writes them (and the July "
          "layout), log_parser reads them, LiDAR rate gate, handshake and streaming to two tools, "
          "command forwarding")
    return 0


# ----------------------------------------------------------------------------- arguments
def parse_addr(text: str):
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit("[RELAY] give the vessel as IP:PORT, e.g. 192.168.8.197:5050")
    return host, int(port)


def rate_text(hz: float) -> str:
    return "off" if hz < 0 else ("every message" if hz == 0 else "%g Hz" % hz)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--listen-ip", default="127.0.0.1", help="where the tools send START (0.0.0.0: other computers too)")
    ap.add_argument("--listen-port", type=int, default=LISTEN_PORT)
    ap.add_argument("--vessel", default=None, metavar="IP:PORT", help="forward the tools' $CMD datagrams here")
    ap.add_argument("--log", default=None, help="also write every line to this file (appended; with --bag: written)")
    ap.add_argument("--lidar-hz", type=float, default=2.0, help="LiDAR lines per second (0: every scan)")
    ap.add_argument("--imu-hz", type=float, default=10.0, help="IMU samples per second (0: every message, -1: none)")
    ap.add_argument("--pose-layout", choices=("vessel", "july"), default="vessel")
    ap.add_argument("--pose-decimals", type=int, default=1, help="1: as the vessel; more for finer positions")
    ap.add_argument("--stamp", choices=("receive", "header"), default="receive")
    ap.add_argument("--hey-hz", type=float, default=2.0, help="'hey' lines per second to each tool (0: only in reply)")
    ap.add_argument("--reliable", action="store_true", help="reliable QoS (default: best effort, as sensor data)")
    ap.add_argument("--pose-topic", default="/map_odom")
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--imu-topic", default="/imu/data_raw")
    ap.add_argument("--report-s", type=float, default=10.0, help="seconds between rate reports")
    ap.add_argument("--bag", default=None, help="write this recorded bag as a raw log (--log) and exit")
    ap.add_argument("--selftest", action="store_true", help="check formats and the UDP link without ROS, then exit")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.listen_port in (5000, 5050):
        ap.error("--listen-port %d is a field tool's or the vessel's port; use another (default %d)"
                 % (args.listen_port, LISTEN_PORT))
    if args.pose_decimals < 1:
        ap.error("--pose-decimals must be at least 1")
    if args.bag:
        run_bag(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
