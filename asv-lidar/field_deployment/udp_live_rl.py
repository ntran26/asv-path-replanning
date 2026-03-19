"""
Run: python udp_live_rl.py --server-ip "10.201.208.224" --test-case 0
"""
import time
import socket
import argparse
import pygame
import numpy as np
import json
from typing import Optional, List, Tuple, Dict, Any
from log_parser import BluefinFrame, BluefinStreamDecoder
import log_viewer
from stable_baselines3 import PPO, SAC
from test_run import TestCase

LIDAR_RANGE = 16
LIDAR_SWATH = 270
LIDAR_BEAMS = 90

MAP_WIDTH = 10
MAP_HEIGHT = 25
OBS_LENGTH = 1

SCALE = 1

def generate_path(start_x, start_y, goal_x, goal_y):
    path_length = max(2, int(np.hypot(abs(goal_x - start_x), abs(goal_y - start_y))))

    # record path coordinates
    path_x = np.round(np.linspace(start_x, goal_x, path_length)).astype(int)
    path_y = np.round(np.linspace(start_y, goal_y, path_length)).astype(int)

    # store path coordinates
    path = np.column_stack((path_x, path_y))

    return path

def calculate_angle(asv_x, asv_y, heading, goal_x, goal_y):
    dx = goal_x - asv_x
    dy = goal_y - asv_y

    target_angle = np.degrees(np.arctan2(dx, dy))    
    angle_diff = (target_angle - heading + 180) % 360 - 180    # normalize to [-180,180]

    return angle_diff

def draw_static_ref(surface: pygame.Surface,
                    map_rect: pygame.Rect,
                    *,
                    map_width: float,
                    map_height: float,
                    reference_path: np.ndarray,
                    obstacles: List[List[Tuple[float, float]]],
                    start_xy: Tuple[float, float],
                    goal_xy: Tuple[float, float],
                    view_center_world: Tuple[float, float],
                    px_per_m: float) -> None:
    """
    Draw static map content in the same world frame as the RL observation:
      - border
      - obstacles
      - path
      - start / goal
    """
    vc_px = map_rect.center
    def w2s(pt):
        return log_viewer.world_to_screen(pt,
                                          view_center_world=view_center_world,
                                          view_center_px=vc_px,
                                          px_per_m=px_per_m)
    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        border_world = [(0.0, 0.0), (map_width, 0.0), (map_width, map_height), (0.0, map_height)]
        border_px = [w2s(p) for p in border_world]
        pygame.draw.polygon(surface, (160, 80, 80), border_px, width=2)

        for poly in obstacles:
            poly_px = [w2s(p) for p in poly]
            pygame.draw.polygon(surface, (200, 60, 60), poly_px, width=0)
            pygame.draw.polygon(surface, (240, 180, 180), poly_px, width=1)

        if reference_path is not None and len(reference_path) >= 2:
            pts = [w2s((float(x), float(y))) for x, y in reference_path]
            pygame.draw.lines(surface, (80, 220, 120), False, pts, 2)
        
        start_px = w2s(start_xy)
        goal_px = w2s(goal_xy)
        pygame.draw.circle(surface, (80, 220, 120), start_px, 6)
        pygame.draw.circle(surface, (220, 80, 220), goal_px, 6)
        pygame.draw.circle(surface, (0, 0, 0), start_px, 6, 1)
        pygame.draw.circle(surface, (0, 0, 0), goal_px, 6, 1)
    
    finally:
        surface.set_clip(prev_clip)

# RL observation adapter
def frame_to_rl_obs(
        frame: BluefinFrame,
        *,
        model_obs_space,
        real_origin_xy: Optional[Tuple[float, float]],
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        reference_path: np.ndarray,
        pos_scale: float,
        speed_scale: float,
        lidar_index0_deg: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Tuple[float, float], float, float]:
    """
    Convert a decoded telemetry frame into the exact observation format used by the updated y-up env.

    Observation format to match:
        {
            "lidar": shape (LIDAR_BEAMS,), float32
            "pos": shape (2,)
            "hdg": shape (1,)
            "dhdg": shape (1,)
            "speed": shape (1,)
            "tgt": shape (1,)
            "target_heading": shape (1,)
        }

    Mapping strategy:
    - Keep deployment processing in y-up.
    - Use the first real telemetry position as an origin anchor.
    - Place that origin at the scenario start point.
    """
    spaces = getattr(model_obs_space, "spaces", {})
    
    # position mapping: first telemetry sample maps to scenario start
    if real_origin_xy is None:
        rx0, ry0 = float(frame.x_m), float(frame.y_m)
    else:
        rx0, ry0 = real_origin_xy

    mx = start_xy[0] + (frame.x_m - rx0) * pos_scale
    my = start_xy[1] + (frame.y_m - ry0) * pos_scale

    # mx = frame.x_m * pos_scale
    # my = frame.y_m * pos_scale

    # heading and heading rate
    yaw_deg = frame.yaw_deg
    yaw_rate = frame.yaw_rate
    
    # speed
    speed = frame.speed_mps * speed_scale

    # path features
    pos_arr = np.array([mx, my], dtype=np.float32)
    distance = np.linalg.norm(reference_path - pos_arr, axis=1)
    tgt = np.min(distance)
    target_heading = calculate_angle(mx, my, yaw_deg, goal_xy[0], goal_xy[1])

    # lidar processing to env beam count / swath
    lidar_angles = np.linspace(-LIDAR_SWATH/2, LIDAR_SWATH/2, LIDAR_BEAMS, dtype=np.float64)
    lidar_proc = log_viewer.pick_lidar_swath(
        np.asarray(frame.lidar_m, dtype=np.float32),
        lidar_angles,
        index0_deg=lidar_index0_deg,
    ).astype(np.float32)
    lidar_proc = np.clip(lidar_proc, 0, LIDAR_RANGE)

    obs_values = {
        "lidar": lidar_proc.astype(np.float32),
        "pos": np.array([mx, my], dtype=np.float32),
        "hdg": np.array([yaw_deg], dtype=np.float32),
        "dhdg": np.array([yaw_rate], dtype=np.float32),
        "speed": np.array([speed], dtype=np.float32),
        "tgt": np.array([tgt], dtype=np.float32),
        "target_heading": np.array([target_heading], dtype=np.float32),
    }

    # enforce exact keys/shapes expected by loaded model
    obs: Dict[str, np.ndarray] = {}
    for key, sp in spaces.items():
        if key in obs_values:
            arr = np.asarray(obs_values[key], dtype=np.float32).reshape(sp.shape)
            obs[key] = arr
        else:
            obs[key] = np.zeros(sp.shape, dtype=np.float32)
    
    aux = {
        "x": mx,
        "y": my,
        "yaw_deg": yaw_deg,
        "yaw_rate": yaw_rate,
        "speed": speed,
        "tgt": tgt,
        "target_heading": target_heading,
    }
    return obs, aux, (mx, my), yaw_deg, yaw_rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0", help="local bind IP")
    ap.add_argument("--local-port", type=int, default=5000, help="local UDP port to listen on")
    ap.add_argument("--server-ip", default="127.0.0.1", help="vessel IP address")
    ap.add_argument("--server-port", type=int, default=5050, help="vessel port")
    ap.add_argument("--record-log", default=None, help="write to log file")
    
    # UI viewer
    ap.add_argument("--fps", type=int, default=60, help="UI FPS cap")
    ap.add_argument("--full", action="store_true", help="Start with full LiDAR text enabled")
    ap.add_argument("--no-map", action="store_true", help="Start with map panel hidden")
    ap.add_argument("--zoom", type=float, default=30.0, help="Initial zoom (pixels per meter)")
    ap.add_argument("--max-path", type=int, default=20000, help="Limit stored path points (avoid RAM blowup)")

    # Test case
    ap.add_argument("--test-case", type=int, default=1, help="Select test scenario")

    args = ap.parse_args()

    # Load RL policy
    model = PPO.load("ppo_asv_model.zip")
    model_obs_space = model.observation_space

    # Load reference map elements from TestCase class
    test_scenario = TestCase()
    sx, sy, gx, gy = test_scenario.position(test_case=args.test_case)
    start_xy = (sx, sy)
    goal_xy = (gx, gy)
    obstacles = test_scenario.obstacles(test_case=args.test_case)
    reference_path = generate_path(sx, sy, gx, gy)

    # Socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.local_port))
    sock.settimeout(0.05)

    # Send to start stream
    sock.sendto(b"START\n", (args.server_ip, args.server_port))
    print("[LISTENER] Sent START to {}, {}", args.server_ip, args.server_port)
    print("[LISTENER] Listening on {}, {}", args.bind_ip, args.local_port)

    log = None
    if args.record_log:
        log = open(args.record_log, "a", encoding="utf-8", buffering=1)
        print(f"[LISTENER] Recording received lines to: {args.record_log}")

    # Decoder state
    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    rx_lines = 0
    rx_frames = 0

    real_origin_xy: Optional[Tuple[float, float]] = None
    last_yaw_deg: Optional[float] = None
    last_t_sec: Optional[float] = None

    latest_obs: Optional[Dict[str, np.ndarray]] = None
    latest_action: Optional[np.ndarray] = None
    latest_aux: Optional[Dict[str, float]] = None

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

    lidar_draw_angles = np.linspace(-LIDAR_SWATH/2, LIDAR_SWATH/2, LIDAR_BEAMS, dtype=np.float64)

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

        # Receive and decode (timeout keeps viewer responsive)
        try:
            msg, addr = sock.recvfrom(65535)
            line = msg.decode("utf-8", errors="replace")
        except socket.timeout:
            line = None

        if line is not None:    
            rx_lines += 1

            if log is not None:
                log.write(line + "\n")

            decoded = decoder.feed(line)
            if decoded is not None:
                rx_frames += 1

                if not paused:
                    frame = decoded
                    cached_lidar_key = None

                    if real_origin_xy is None:
                        real_origin_xy = (float(frame.x_m), float(frame.y_m))

                    if model is not None and model_obs_space is not None:
                        latest_obs, latest_aux, mapped_xy, yaw_deg, yaw_rate = frame_to_rl_obs(
                            frame,
                            model_obs_space=model_obs_space,
                            real_origin_xy=real_origin_xy,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            reference_path=reference_path,
                            pos_scale=SCALE,
                            speed_scale=SCALE,
                            lidar_index0_deg=log_viewer.LIDAR_INDEX_DEG,
                        )

    #-----------------Send action commands----------------------------
                        action, _ = model.predict(latest_obs)
                        latest_action = np.asarray(action, dtype=np.float32).reshape(-1)
                        # latest_cmd = latest_action*100

                        def rudder_to_cmd(a):
                            return 80 * a - 20      # [-100, 60]
                        def thrust_to_cmd(b):
                            return np.clip(100*b, 50, 100)  # [50, 100]

                        rudder_cmd = rudder_to_cmd(latest_action[0])
                        thrust_cmd = thrust_to_cmd(latest_action[1])

                        command = f"$CMD,{rudder_cmd},{thrust_cmd}"
                        sock.sendto(command.encode(), (args.server_ip, args.server_port))
    #-----------------------------------------------------------------
                    else:
                        # viewer-only fallback mapping
                        mapped_xy = (
                            float(start_xy[0] + (frame.x_m - real_origin_xy[0]) * args.pos_scale),
                            float(start_xy[1] + (frame.y_m - real_origin_xy[1]) * args.pos_scale),
                        )
                        latest_aux = {
                            "x": mapped_xy[0],
                            "y": mapped_xy[1],
                            "yaw_deg": frame.yaw_deg,
                            "yaw_rate": frame.yaw_rate,
                            "speed": float(frame.speed_mps),
                            "tgt": 0.0,
                            "target_heading": 0.0,
                        }

                    path_world.append(mapped_xy)
                    if len(path_world) > args.max_path:
                        path_world = path_world[-args.max_path:]

                    if follow_mode:
                        view_center_world = mapped_xy

        # Draw UI
        screen.fill((20,20,25))
        map_rect = pygame.Rect(text_w, 0, map_w, win_h)

        y = 10
        line_h = 22
        lidar_raw = None
        lidar_view = None

        if frame is not None:
            lidar_raw = frame.lidar_m
            lidar_view = log_viewer.pick_lidar_swath(
                lidar_raw, 
                lidar_draw_angles, 
                index0_deg=log_viewer.LIDAR_INDEX_DEG
            )
        
        header_lines = [
            f"UDP: local={args.bind_ip}:{args.local_port}  server={args.server_ip}:{args.server_port}",
            f"RX: lines={rx_lines}  frames={rx_frames}   {'PAUSED' if paused else 'RUNNING'} (Space)   full_lidar={'ON' if show_full_lidar else 'OFF'} (F)",
            f"Map: {'ON' if show_map else 'OFF'} (M)   follow={'ON' if follow_mode else 'OFF'} (G)   zoom={px_per_m:0.1f}px/m  origin={'SET' if origin_world else 'NONE'} (O)"
        ]
        if frame is None:
            header_lines.append("Waiting for first decoded LiDAR frame...")
        else:
            header_lines += [
                f"ts={frame.ts_str}   t={frame.t_sec:0.3f}s   seq={frame.seq}   hdg_ref={frame.hdg_ref_deg}",
                "  ",
                f"Real frame: x={frame.x_m:+0.3f}  y={frame.y_m:+0.3f}  hdg={frame.yaw_deg:+0.2f}   dhdg={frame.yaw_rate:+0.2f}  spd={frame.speed_mps:0.3f}",
            ]
            header_lines += ["    "]
            if latest_aux is not None:
                header_lines += [
                    f"RL obs: pos=({latest_aux['x']:+0.3f},{latest_aux['y']:+0.3f})  hdg={latest_aux['yaw_deg']:+0.2f}  dhdg={latest_aux['yaw_rate']:+0.2f}",
                    f"RL obs: speed={latest_aux['speed']:.3f}  tgt={latest_aux['tgt']:.3f}  target_heading={latest_aux['target_heading']:+0.2f}",
                ]
        
        header_lines += ["    "]
        
        if latest_action is not None:
            header_lines += [
                f"Policy action: [{float(latest_action[0]):+.3f}, {float(latest_action[1]):+.3f}]",
                f"Command: [{float(rudder_cmd):+.1f}, {float(thrust_cmd):+.1f}]",
            ]

        for s in header_lines:
            screen.blit(font.render(s, True, (235, 235, 245)), (10, y))
            y += line_h

        y += 10
    
        # lidar text area
        if frame is not None:
            if show_full_lidar:
                lidar_src = lidar_raw
                title = "LiDAR full list (F)"
            else:
                lidar_src = lidar_view
                title = "Processed LiDAR list (F)"

            cached_key = (rx_frames, show_full_lidar)
            if cached_key != cached_lidar_key:
                cached_lidar_lines = log_viewer.format_lidar_lines(lidar_src, per_line=15, precision=1)
                cached_lidar_key = cached_key

            max_lines_on_screen = max(1, (win_h - y - 20) // 18)
            max_scroll = max(0, len(cached_lidar_lines) - max_lines_on_screen)
            lidar_scroll = min(lidar_scroll, max_scroll)

            info = f"{title} (scroll {lidar_scroll}/{max_scroll})" if show_full_lidar else title
            screen.blit(font.render(info, True, (200, 200, 210)), (10, y))
            y += 22

            for s in cached_lidar_lines[lidar_scroll : lidar_scroll + max_lines_on_screen]:
                screen.blit(small.render(s, True, (210, 210, 220)), (10, y))
                y += 18
        
        # map panel
        if show_map:
            current_world = None
            yaw_for_draw = None
            lidar_ranges_draw = None

            if latest_aux is not None:
                current_world = (latest_aux["x"], latest_aux["y"])
                yaw_for_draw = latest_aux["yaw_deg"]

            if frame is not None:
                lidar_ranges_draw = log_viewer.pick_lidar_swath(
                    frame.lidar_m,
                    lidar_draw_angles,
                    index0_deg=log_viewer.LIDAR_INDEX_DEG,
                )

            log_viewer.draw_map_panel(
                screen,
                map_rect,
                path_world=path_world,
                current_world=current_world,
                yaw_deg=yaw_for_draw,
                view_center_world=view_center_world,
                px_per_m=px_per_m,
                lidar_angles_deg=lidar_draw_angles if lidar_ranges_draw is not None else None,
                lidar_ranges_m=lidar_ranges_draw,
                lidar_index0_deg=log_viewer.LIDAR_INDEX_DEG,
                lidar_index0_range_m=float(frame.lidar_m[0]) if frame is not None and frame.lidar_m.size > 0 else None,
                mark_index0=True,
            )

            draw_static_ref(
                screen,
                map_rect,
                map_width=MAP_WIDTH,
                map_height=MAP_HEIGHT,
                reference_path=reference_path,
                obstacles=obstacles,
                start_xy=start_xy,
                goal_xy=goal_xy,
                view_center_world=view_center_world,
                px_per_m=px_per_m,
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


# t_now = time.perf_counter()

# if last_proc_time is not None:
#     print(f"wall dt = {t_now - last_proc_time:.4f}s")

# t0 = time.perf_counter()
# action, _ = model.predict(latest_obs)
# t1 = time.perf_counter()

# print(f"predict time = {t1 - t0:.4f}s, frame_t = {frame.t_sec:.3f}")
# last_proc_time = t_now