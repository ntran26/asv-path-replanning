"""
Run: python udp_listener.py --record-log sea_trial.log
"""

import socket
import argparse
import pygame
import numpy as np
from typing import Optional, List, Tuple
from log_parser import BluefinFrame, BluefinStreamDecoder
import log_viewer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0", help="local bind IP")
    ap.add_argument("--local-port", type=int, default=5000, help="local UDP port to listen on")
    ap.add_argument("--server-ip", default="127.0.0.1", help="vessel IP address")
    ap.add_argument("--server-port", type=int, default=5050, help="vessel port")
    ap.add_argument("--print-raw", action="store_true", help="print raw lines")
    ap.add_argument("--record-log", default=None, help="write to log file")
    
    # UI viewer
    ap.add_argument("--fps", type=int, default=60, help="UI FPS cap")
    ap.add_argument("--full", action="store_true", help="Start with full LiDAR text enabled")
    ap.add_argument("--no-map", action="store_true", help="Start with map panel hidden")
    ap.add_argument("--zoom", type=float, default=30.0, help="Initial zoom (pixels per meter)")
    ap.add_argument("--max-path", type=int, default=20000, help="Limit stored path points (avoid RAM blowup)")

    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.local_port))

    # Send HEY to start stream
    sock.sendto(b"START\n", (args.server_ip, args.server_port))
    print("[LISTENER] Sent START to {}, {}", args.server_ip, args.server_port)
    print("[LISTENER] Listening on {}, {}", args.bind_ip, args.local_port)

    log = None
    if args.record_log:
        log = open(args.record_log, "a", encoding="utf-8", buffering=1)
        print(f"[LISTENER] Recording received lines to: {args.record_log}")

    # Decoder state
    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    buf = ""    # in case packets contain multiple lines
    rx_lines = 0
    rx_frames = 0

    # UI state
    pygame.init()
    pygame.display.set_caption("Bluefin UDP live viewer")
    win_w, win_h = 1200, 600
    text_w = 800
    map_w = win_w - text_w
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18) or pygame.font.Font(None, 18)
    small = pygame.font.SysFont("consolas", 15) or pygame.font.Font(None, 15)

    paused = False
    show_full_lidar = bool(args.full)
    lidar_scroll = 0

    show_map = not bool(args.no_map)
    follow_mode = True

    px_per_m = args.zoom
    view_center_world = (0,0)
    origin_world: Optional[Tuple[float, float]] = None
    path_world: List[Tuple[float, float]] = []

    frame: Optional[BluefinFrame] = None
    cached_lidar_lines: List[str] = []
    cached_lidar_key = None

    lidar_draw_angles = np.linspace(-log_viewer.LIDAR_SWATH/2, log_viewer.LIDAR_SWATH/2, log_viewer.LIDAR_BEAMS, dtype=np.float64)

    running = True
    while running:
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    show_full_lidar = not show_full_lidar
                    lidar_scroll = 0
                elif event.key == pygame.K_UP:
                    if show_full_lidar:
                        lidar_scroll = max(0, lidar_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    if show_full_lidar:
                        lidar_scroll = lidar_scroll + 1
                elif event.key == pygame.K_m:
                    show_map = not show_map
                elif event.key == pygame.K_g:
                    follow_mode = not follow_mode
                elif event.key == pygame.K_c:
                    path_world = []
                elif event.key == pygame.K_o:
                    if frame is not None:
                        origin_world = (frame.x_m, frame.y_m)
                        path_world = [(0,0)]
                        view_center_world = (0,0)
                elif event.key == pygame.K_p:
                    out = "snapshot.png"
                    pygame.image.save(screen, out)
                    print(f"[LISTENER] Saved snapshot: {out}")
                
                if not follow_mode:
                    pan_step_m = 20 / max(px_per_m, 1e-9)
                    if event.key == pygame.K_w:
                        view_center_world = (view_center_world[0], view_center_world[1] + pan_step_m)
                    elif event.key == pygame.K_s:
                        view_center_world = (view_center_world[0], view_center_world[1] - pan_step_m)
                    elif event.key == pygame.K_a:
                        view_center_world = (view_center_world[0] - pan_step_m, view_center_world[1])
                    elif event.key == pygame.K_d:
                        view_center_world = (view_center_world[0] + pan_step_m, view_center_world[1])       

        # Receive and decode
        msg, addr = sock.recvfrom(65535)
        chunk = msg.decode("utf-8", errors="replace")
        buf += chunk

        # split into lines safely
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            rx_lines += 1

            if args.print_raw:
                print(line)

            if log is not None:
                log.write(line + "\n")
            
            decoded = decoder.feed(line)
            if decoded is not None:
                rx_frames += 1
                if not paused:
                    frame = decoded
                    cached_lidar_key = None
                    if origin_world is None:
                        origin_world = (frame.x_m, frame.y_m)
                        path_world = [(0,0)]
                    rel = (frame.x_m - origin_world[0], frame.y_m - origin_world[1])
                    path_world.append(rel)

                    if len(path_world) > args.max_path:
                        path_world = path_world[-args.max_path:]
                    if follow_mode:
                        view_center_world = rel

        # Draw UI
        screen.fill((20,20,25))

        map_rect = pygame.Rect(text_w, 0, map_w, win_h)

        y = 10
        line_h = 22

        lidar_raw = None
        lidar_view = None

        if frame is not None:
            lidar_raw = frame.lidar_m
            lidar_view = log_viewer.pick_lidar_swath(lidar_raw, lidar_draw_angles, index0_deg=log_viewer.LIDAR_INDEX_DEG)
        
        header_lines = [f"UDP: local={args.bind_ip}:{args.local_port}  server={args.server_ip}:{args.server_port}",
                        f"RX: lines={rx_lines}  frames={rx_frames}   {'PAUSED' if paused else 'RUNNING'} (Space)   full_lidar={'ON' if show_full_lidar else 'OFF'} (F)",
                        f"Map: {'ON' if show_map else 'OFF'} (M)   follow={'ON' if follow_mode else 'OFF'} (G)   zoom={px_per_m:0.1f}px/m  origin={'SET' if origin_world else 'NONE'} (O)"]
        if frame is None:
            header_lines.append("Waiting for first decoded LiDAR frame...")
        else:
            lidar = frame.lidar_m
            header_lines += [
                f"ts={frame.ts_str}   t={frame.t_sec:0.3f}s   seq={frame.seq}   hdg_ref={frame.hdg_ref_deg}",
                f"Pose: x={frame.x_m:+0.3f} m  y={frame.y_m:+0.3f} m  yaw={frame.yaw_deg:+0.2f} deg",
                f"Vel:  vx={frame.vx_mps:+0.3f} m/s  vy={frame.vy_mps:+0.3f} m/s  spd={frame.speed_mps:0.3f} m/s",
                f"RC:   S1={frame.s1}   S2={frame.s2}",
                f"LiDAR: N={lidar.size}  min/mean/max={float(lidar.min()):0.2f}/{float(lidar.mean()):0.2f}/{float(lidar.max()):0.2f}",
            ]
        for s in header_lines:
            screen.blit(font.render(s, True, (235, 235, 245)), (10, y))
            y += line_h

        y += 10
    
        # lidar text area
        if frame is not None:
            if show_full_lidar:
                lidar_src = lidar_raw
                title = "LiDAR full list"
            else:
                lidar_src = lidar_view
                title = "Processed LiDAR list"
            cached_key = (rx_frames, show_full_lidar)
            if cached_key != cached_lidar_key:
                cached_lidar_lines = log_viewer.format_lidar_lines(lidar_src, per_line=15, precision=1)
                cached_lidar_key = cached_key
            max_lines_on_screen = max(1, (win_h-y-20)//18)
            max_scroll = max(0, len(cached_lidar_lines) - max_lines_on_screen)
            lidar_scroll = min(lidar_scroll, max_scroll)
            
            if show_full_lidar:
                info = f"{title} (scroll {lidar_scroll}/{max_scroll})"
            else:
                info = title
            
            screen.blit(font.render(info, True, (200,200,210)), (10,y))
            y += 22

            for s in cached_lidar_lines[lidar_scroll : lidar_scroll + max_lines_on_screen]:
                screen.blit(small.render(s, True, (210,210,220)), (10,y))
                y += 18
        
        # map panel
        if show_map:
            if frame is None or origin_world is None:
                log_viewer.draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=None,
                    yaw_deg=None,
                    view_center_world=view_center_world,
                    px_per_m=px_per_m,
                )
            else:
                current_rel = (frame.x_m - origin_world[0], frame.y_m - origin_world[1])
                lidar_ranges_draw = log_viewer.pick_lidar_swath(frame.lidar_m, lidar_draw_angles, index0_deg=log_viewer.LIDAR_INDEX_DEG)

                log_viewer.draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=current_rel,
                    yaw_deg=frame.yaw_deg,
                    view_center_world=view_center_world,
                    px_per_m=px_per_m,
                    lidar_angles_deg=lidar_draw_angles,
                    lidar_ranges_m=lidar_ranges_draw,
                    lidar_index0_deg=log_viewer.LIDAR_INDEX_DEG,
                    lidar_index0_range_m=float(frame.lidar_m[0]) if frame.lidar_m.size > 0 else None,
                    mark_index0=True,
                )

        pygame.display.flip()
        clock.tick(args.fps)

    if log is not None:
        log.close()
    
    sock.close()
    pygame.quit()
    print("[LISTENER] Exit")

if __name__ == "__main__":
    main()