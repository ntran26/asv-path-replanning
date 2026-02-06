import socket
import time
import re
import argparse

# Timestamp
TS_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{6})\]")

def ts_to_seconds(line: str):
    """
    Extract HH:MM:SS.microseconds from the start of a log line and convert to seconds
    Return None if the line does not start with a timestamp in brackets
    """
    m = TS_RE.match(line)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))
    ms = int(m.group(4))
    
    return hh*3600 + mm*60 + ss + ms/1e6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0")     # IP to bind
    ap.add_argument("--port", default=5050)     # UDP port to listen on
    ap.add_argument("--log", required=True)     # Path to log file to replay
    ap.add_argument("--speed", default=1.0)     # Replay speed: 1x = real time
    ap.add_argument("--ignore-rc", action="store_true")     # ignore RC line
    ap.add_argument("--loop", action="store_true")  # loop log forever
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.port))

    print("[FAKE VESSEL] Listening on {}:{}", args.bind_ip, args.port)
    print("[FAKE VESSEL] Waiting for HEY...")

    # Wait for handshake
    while True:
        data, addr = sock.recvfrom(4096)
        if data.strip() == b"HEY":
            client_addr = addr
            print("[FAKE VESSEL] Got HEY from {}, streaming will start", client_addr)
            break
        else:
            print("[FAKE VESSEL] Ignoring packet from {}: {}", addr, data)
    
    def replay_once():
        t0 = None
        last_t = None 

        with open(args.log, "r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                if args.ignore_rc and " RC" in line and "S1:" in line and "S2:" in line:
                    continue

                t = ts_to_seconds(line)
                if t is None:
                    continue

                if t0 is None:
                    t0 = t
                    last_t = t
                
                dt = (t - last_t) / max(args.speed, 1e-9)
                if dt > 0:
                    time.sleep(dt)
                
                payload = (line + "\n").encode("utf-8", errors="replace")
                sock.sendto(payload, client_addr)
                last_t = t
    
    while True:
        replay_once()
        if not args.loop:
            print("[FAKE VESSEL] End of log reached. Exiting.")
            break
        print("[FAKE VESSEL] Looping log...")
        time.sleep(1.0)

if __name__ == "__main__":
    main()