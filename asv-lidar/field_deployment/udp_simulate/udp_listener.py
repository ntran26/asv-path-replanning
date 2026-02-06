import socket
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0")     # local bind IP
    ap.add_argument("--local-port", default=5000)       # local UDP port to listen on
    ap.add_argument("--server-ip", default="127.0.0.1") # vessel IP
    ap.add_argument("--server-port", default=5050)      # vessel port
    ap.add_argument("--print-raw", action="store_true") # print raw lines
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.local_port))

    # Send HEY to start stream
    sock.sendto(b"HEY\n", (args.server_ip, args.server_port))
    print("[LISTENER] Sent HEY to {}, {}", args.server_ip, args.server_port)
    print("[LISTENER] Listening on {}, {}", args.bind_ip, args.local_port)

    # decoder = BluefinStreamDecoder(lidar_out_beams=720)
    # origin = None

    buf = ""    # in case packets contain multiple lines

    while True:
        msg, addr = sock.recvfrom(65535)
        chunk = msg.decode("utf-8", errors="replace")
        buf += chunk

        # split into lines safely
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            if args.print_raw:
                print(line)

            # # optional: parse it
            # frame = decoder.feed(line)
            # if frame is not None:
            #     if origin is None:
            #         origin = (frame.x_m, frame.y_m, frame.yaw_deg)
            #     obs = frame_to_gym_obs(frame, origin_xyh=origin, include_velocity=True)
            #     print(f"[FRAME] t={frame.t_sec:.2f}   pos={obs['pos']}  yaw={obs['hdg']}   spd={obs.get('spd')}")

if __name__ == "__main__":
    main()