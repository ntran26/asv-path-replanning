
"""
bluefin_realtime_observer.py

Connect to a Bluefin (model-scale ASV) sensor text stream and decode frames in real time.

Supports:
  1) TCP client (connect to vessel's IP:port)
  2) UDP listener (bind local port, receive datagrams)
  3) Serial port (USB/UART)

Assumptions about incoming stream:
  - Each record is ASCII/UTF-8 text
  - Lines are newline-terminated (best) OR arrive in chunks that can be split by '\n'
  - Log line formats match your existing decoder:
      [HH:MM:SS.microsec][seq] HDG:...
      [HH:MM:SS.microsec]+X,+Y,+Yaw
      [HH:MM:SS.microsec][<comma-separated lidar ints>]
    (Manual control lines S1/S2 may be absent in field deployment.)

This script is READ-ONLY: it does not send commands to the vessel.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Iterable, Iterator, Optional, Tuple

# Import your existing parser module (put this file in the same folder as log_parser.py)
# If your file is named differently, update the import below.
try:
    from log_parser import BluefinStreamDecoder, frame_to_gym_obs, BluefinFrame
except Exception as e:
    print("ERROR: Could not import from 'log_parser.py'.")
    print("Put bluefin_realtime_observer.py in the SAME folder as your working parser script,")
    print("and ensure your parser file is named 'log_parser.py' (or change the import in this script).")
    print(f"Import error: {e}")
    raise


def _iter_lines_from_tcp(host: str, port: int, *, timeout_s: float = 5.0) -> Iterator[str]:
    """
    TCP client: connect to (host, port) and yield newline-delimited lines.
    """
    sock = socket.create_connection((host, port), timeout=timeout_s)
    # If you want lower latency, disable Nagle
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Wrap in a file-like object so we can iterate by lines easily.
    # This assumes the sender ends each message with '\n'.
    f = sock.makefile("r", encoding="utf-8", errors="ignore", newline="\n")
    try:
        for line in f:
            yield line
    finally:
        try:
            f.close()
        finally:
            sock.close()


def _iter_lines_from_udp(bind_port: int, *, bind_host: str = "0.0.0.0", bufsize: int = 65535) -> Iterator[str]:
    """
    UDP listener: bind to (bind_host, bind_port), receive datagrams, yield lines.
    Handles the case where a datagram contains multiple lines or partial lines.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_host, bind_port))

    buffer = ""
    while True:
        data, _addr = sock.recvfrom(bufsize)
        chunk = data.decode("utf-8", errors="ignore")
        buffer += chunk

        while True:
            nl = buffer.find("\n")
            if nl < 0:
                break
            line = buffer[:nl]
            buffer = buffer[nl + 1 :]
            yield line + "\n"


def _iter_lines_from_serial(port: str, baud: int, *, timeout_s: float = 0.5) -> Iterator[str]:
    """
    Serial reader: yields lines read from a serial port.

    Requires: pip install pyserial
    """
    try:
        import serial  # type: ignore
    except Exception:
        print("ERROR: pyserial not installed. Install with: pip install pyserial")
        raise

    ser = serial.Serial(port=port, baudrate=baud, timeout=timeout_s)
    try:
        while True:
            b = ser.readline()  # reads until '\n' or timeout
            if not b:
                continue
            yield b.decode("utf-8", errors="ignore")
    finally:
        ser.close()


def _parse_host_port(s: str) -> Tuple[str, int]:
    """
    Parse 'HOST:PORT' into (HOST, PORT).
    """
    if ":" not in s:
        raise ValueError("Expected HOST:PORT")
    host, port_str = s.rsplit(":", 1)
    return host, int(port_str)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Real-time Bluefin sensor stream observer (decode + print).")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tcp", help="Connect to TCP stream, format: HOST:PORT")
    src.add_argument("--udp", type=int, help="Listen on UDP port (bind 0.0.0.0:PORT)")
    src.add_argument("--serial", help="Read from serial port, e.g. COM5 (Windows) or /dev/ttyUSB0 (Linux/macOS)")

    ap.add_argument("--baud", type=int, default=115200, help="Serial baud rate (only for --serial)")
    ap.add_argument("--record", default=None, help="Optional: write raw incoming lines to this file")
    ap.add_argument("--print-every", type=int, default=1, help="Print one summary every N decoded frames")
    ap.add_argument("--lidar-out-beams", type=int, default=720, help="Output LiDAR beams after downsampling")
    ap.add_argument("--lidar-angle-offset-deg", type=float, default=0.0, help="Circular shift for LiDAR alignment")
    ap.add_argument("--origin", action="store_true", help="Zero the first pose to (0,0,0) in printed obs")

    args = ap.parse_args(argv)

    decoder = BluefinStreamDecoder(
        lidar_out_beams=args.lidar_out_beams,
        lidar_angle_offset_deg=args.lidar_angle_offset_deg,
    )

    # Pick line source
    if args.tcp:
        host, port = _parse_host_port(args.tcp)
        line_iter = _iter_lines_from_tcp(host, port)
        print(f"[observer] TCP connect -> {host}:{port}")
    elif args.udp is not None:
        line_iter = _iter_lines_from_udp(args.udp)
        print(f"[observer] UDP listen -> 0.0.0.0:{args.udp}")
    else:
        line_iter = _iter_lines_from_serial(args.serial, args.baud)
        print(f"[observer] Serial read -> {args.serial} @ {args.baud} baud")

    # Optional raw recorder
    rec_f = open(args.record, "a", encoding="utf-8") if args.record else None
    if rec_f:
        print(f"[observer] Recording raw stream to: {args.record}")

    origin_xyh = None
    n_frames = 0

    # For measuring effective frame rate (based on LiDAR frames)
    t_wall_start = time.time()
    t_wall_last = t_wall_start

    try:
        for line in line_iter:
            if rec_f:
                rec_f.write(line)
                rec_f.flush()

            frame = decoder.feed(line)
            if frame is None:
                continue

            # Latch origin on first decoded frame
            if args.origin and origin_xyh is None:
                origin_xyh = (frame.x_m, frame.y_m, frame.yaw_deg)

            obs = frame_to_gym_obs(frame, origin_xyh=origin_xyh, include_velocity=True)

            # Update rate estimate
            n_frames += 1
            now = time.time()
            dt_wall = now - t_wall_last
            t_wall_last = now
            hz_inst = (1.0 / dt_wall) if dt_wall > 1e-9 else float("inf")
            hz_avg = n_frames / max(now - t_wall_start, 1e-9)

            if (n_frames % args.print_every) == 0:
                pos = obs["pos"]
                yaw = float(obs["hdg"][0])
                spd = float(obs["spd"][0]) if "spd" in obs else float("nan")
                lidar = obs["lidar"]

                print(
                    f"[{n_frames:06d}] t={frame.t_sec:8.3f}s  "
                    f"pos=({pos[0]:8.2f},{pos[1]:8.2f})  yaw={yaw:7.2f}deg  "
                    f"spd={spd:6.2f}m/s  "
                    f"lidar[min/max]={float(lidar.min()):.2f}/{float(lidar.max()):.2f}m  "
                    f"rate(inst/avg)={hz_inst:5.1f}/{hz_avg:5.1f} Hz"
                )

    except KeyboardInterrupt:
        print("\n[observer] Stopped by user (Ctrl+C).")
        return 0
    finally:
        if rec_f:
            rec_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
