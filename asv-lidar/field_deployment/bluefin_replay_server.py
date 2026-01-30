
"""
bluefin_replay_server.py

Replay a saved log file as a real-time stream so you can test your TCP/UDP receiver
without needing the vessel.

Two modes:
  1) TCP server: listens on PORT, sends lines to ONE client (simple for testing)
  2) UDP sender: sends datagrams to HOST:PORT

Replay timing:
  - If --rate is provided (Hz), sleeps 1/rate between lines that "matter".
  - If --use-timestamps is set, it will try to sleep according to the timestamp
    in each line (requires lines to start with [HH:MM:SS.microsec]).

Note: This is only for debugging your pipeline.
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Optional

import re

_TS_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2}\.\d{6})\]")

def _ts_to_seconds(ts_str: str) -> float:
    hh, mm, rest = ts_str.split(":")
    ss, micros = rest.split(".")
    return int(hh)*3600 + int(mm)*60 + int(ss) + int(micros) / 1e6

def _parse_host_port(s: str):
    host, port_s = s.rsplit(":", 1)
    return host, int(port_s)

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Replay Bluefin log file over TCP or UDP.")
    ap.add_argument("logfile", help="Path to .log file")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tcp-server", type=int, help="Run as TCP server on PORT")
    mode.add_argument("--udp", help="Send UDP to HOST:PORT")

    ap.add_argument("--rate", type=float, default=10.0, help="Replay rate in Hz (used if not --use-timestamps)")
    ap.add_argument("--use-timestamps", action="store_true", help="Sleep according to timestamps in the log")
    ap.add_argument("--loop", action="store_true", help="Loop the file forever")
    args = ap.parse_args(argv)

    # Setup output
    tcp_conn = None
    udp_sock = None
    udp_addr = None

    if args.tcp_server is not None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", args.tcp_server))
        srv.listen(1)
        print(f"[replay] TCP server listening on 0.0.0.0:{args.tcp_server}")
        tcp_conn, addr = srv.accept()
        print(f"[replay] Client connected from {addr}")
    else:
        host, port = _parse_host_port(args.udp)
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr = (host, port)
        print(f"[replay] UDP sending to {host}:{port}")

    dt = 1.0 / max(args.rate, 1e-9)

    def send_line(line: str):
        data = line.encode("utf-8", errors="ignore")
        if tcp_conn is not None:
            tcp_conn.sendall(data)
        else:
            udp_sock.sendto(data, udp_addr)

    try:
        while True:
            with open(args.logfile, "r", encoding="utf-8", errors="ignore") as f:
                last_ts = None
                for line in f:
                    if not line.endswith("\n"):
                        line = line + "\n"

                    # Compute sleep
                    if args.use_timestamps:
                        m = _TS_RE.match(line)
                        if m:
                            ts = _ts_to_seconds(m.group(1))
                            if last_ts is None:
                                last_ts = ts
                            else:
                                sleep_s = max(0.0, ts - last_ts)
                                time.sleep(sleep_s)
                                last_ts = ts
                        # If no timestamp, just send immediately
                    else:
                        time.sleep(dt)

                    send_line(line)

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\n[replay] Stopped by user.")
        return 0
    finally:
        if tcp_conn is not None:
            tcp_conn.close()
        if udp_sock is not None:
            udp_sock.close()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
