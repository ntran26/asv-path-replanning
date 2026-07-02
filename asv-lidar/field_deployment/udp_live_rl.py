"""
Live UDP bridge between Bluefin telemetry and the current SAC ASV local-planner policy.

shadow test with fake replay:
  python fake_vessel_replay.py --log trial.log --loop
  python udp_live_rl.py --test-case 1 --shadow
  
live control:
  python udp_live_rl.py --server-ip 10.201.205.110 --record-video --record-log 2026-07-02/trial.log --test-case 1 --fixed-rpm

Notes:
- SAC is hardcoded. PPO support was removed intentionally.
- The observation adapter builds the same dict-style observation as the current rl_env.py:
    lidar, u, v, yaw_rate, cross_track_error, course_error,
    lookahead_course_error, front_clearance, side_clearance_diff,
    local_target_cte.
- Rudder sign is reversed by default because the real vessel rudder command uses
  the opposite sign to the simulation convention.
"""

from __future__ import annotations

import argparse
import os
import math
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

try:
    import cv2
except Exception:
    cv2 = None
from stable_baselines3 import SAC

from log_parser import BluefinFrame, BluefinStreamDecoder
import log_viewer
import policy_render as pr
from test_run import TestCase
from lidar_pooling import pool_lidar_to_sectors_shared, normalise_pooling_mode

try:
    from asv_lidar import LIDAR_RANGE, LIDAR_SWATH, LIDAR_BEAMS, LIDAR_SECTORS
except Exception:
    LIDAR_RANGE = 16.0
    LIDAR_SWATH = 270.0
    LIDAR_BEAMS = 225
    LIDAR_SECTORS = 25

try:
    from ship_model import VESSEL_WIDTH, HULL_MARGIN
except Exception:
    VESSEL_WIDTH = 0.50
    HULL_MARGIN = 0.15

# ---------------------------------------------------------------------------
# LiDAR pooling mode
# ---------------------------------------------------------------------------
LIDAR_POOLING_MODE = normalise_pooling_mode("paper")
FEASIBILITY_SAFE_WIDTH = float(VESSEL_WIDTH + 2.0 * HULL_MARGIN)

# ---------------------------------------------------------------------------
# Deployment defaults: stage 3 speed-control policy
# ---------------------------------------------------------------------------
MAP_WIDTH = 10.0
MAP_HEIGHT = 25.0
POS_SCALE = 1.0
LOOKAHEAD_FRACTION = 0.25
DEFAULT_LAMBDA = 0.5

RPM_STAGE = 1

if RPM_STAGE == 1:
    RPM_DELTA = 3.0       
    RPM_FLOOR = 9.0
    RPM_CEIL = 15.0
elif RPM_STAGE == 2:
    RPM_DELTA = 4.0       
    RPM_FLOOR = 8.0
    RPM_CEIL = 16.0
elif RPM_STAGE == 3:
    RPM_DELTA = 6.0      
    RPM_FLOOR = 6.0
    RPM_CEIL = 18.0
elif RPM_STAGE == 4:
    RPM_DELTA = 12.0      
    RPM_FLOOR = 0.0
    RPM_CEIL = 24.0

RPM_MAX = 24.0
CRUISE_RPM = 12.0
S2_MAX_CMD = 100.0    # vessel command S2 range: 0-100
RUDDER_SCALE = 100.0
RUDDER_SIGN = -1.0    # real vessel sign is reversed vs sim

BLOCK_D_SAFE = 4.5
BLOCK_D_CRIT = 2.0
BLOCK_FRONT_DEG = 25.0
SIDE_ARC_MIN_DEG = 15.0
SIDE_ARC_MAX_DEG = 100.0
SIDE_CLEAR_TIE = 0.15
BYPASS_CTE = 0.8

MAX_RUDDER_DEG_FOR_RATE = 40.0
MAX_RUDDER_RATE_DPS = 20.0
MAX_CMD_DT = 0.5
MIN_CMD_DT = 1e-3

# ---------------------------------------------------------------------------
# Geometry/path helpers
# ---------------------------------------------------------------------------

def wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def generate_reference_path(start_x: float, start_y: float, goal_x: float, goal_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dense straight reference path and cumulative arc-length."""
    path_length = max(40, int(np.hypot(goal_x - start_x, goal_y - start_y) * 5.0))
    path_x = np.linspace(start_x, goal_x, path_length, dtype=np.float32)
    path_y = np.linspace(start_y, goal_y, path_length, dtype=np.float32)
    path = np.column_stack((path_x, path_y)).astype(np.float32)

    diffs = np.diff(path, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    path_s = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)
    return path, path_s


def path_tangent(path: np.ndarray, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, len(path) - 1))
    if len(path) < 2:
        return np.array([0.0, 1.0], dtype=np.float32)
    if idx == 0:
        vec = path[1] - path[0]
    elif idx == len(path) - 1:
        vec = path[-1] - path[-2]
    else:
        vec = path[idx + 1] - path[idx - 1]
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float32)
    return (vec / n).astype(np.float32)

def bearing_deg(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    dx = float(to_xy[0] - from_xy[0])
    dy = float(to_xy[1] - from_xy[1])
    return float(np.degrees(np.arctan2(dx, dy)))

# ---------------------------------------------------------------------------
# LiDAR pooling helpers
# ---------------------------------------------------------------------------

def pool_lidar_to_sectors(
    full_lidar_m: np.ndarray,
    *,
    lidar_index0_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw policy swath + 25 sector ranges/closeness.

    This uses the same shared pooling helper as the simulator. The pooling
    mode is hardcoded for this branch so it matches the trained policy.
    """
    raw_angles = np.linspace(
        -float(LIDAR_SWATH) / 2.0,
        float(LIDAR_SWATH) / 2.0,
        int(LIDAR_BEAMS),
        dtype=np.float32,
    )
    raw = log_viewer.pick_lidar_swath(
        np.asarray(full_lidar_m, dtype=np.float32),
        raw_angles,
        index0_deg=lidar_index0_deg,
    ).astype(np.float32)
    raw = np.clip(raw, 0.0, float(LIDAR_RANGE))

    sector_ranges, sector_closeness, sector_angles = pool_lidar_to_sectors_shared(
        raw,
        raw_angles_deg=raw_angles,
        lidar_range=float(LIDAR_RANGE),
        lidar_swath_deg=float(LIDAR_SWATH),
        n_sectors=int(LIDAR_SECTORS),
        safe_width_m=float(FEASIBILITY_SAFE_WIDTH),
        mode=str(LIDAR_POOLING_MODE),
    )
    return raw, raw_angles, sector_ranges, sector_closeness, sector_angles

# ---------------------------------------------------------------------------
# RL observation adapter
# ---------------------------------------------------------------------------

def frame_to_rl_obs(
    frame: BluefinFrame,
    *,
    model_obs_space,
    real_origin_xy: Optional[Tuple[float, float]],
    start_xy: Tuple[float, float],
    reference_path: np.ndarray,
    reference_path_s: np.ndarray,
    lookahead_fraction: float,
    pos_scale: float,
    lidar_index0_deg: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Tuple[float, float], float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spaces = getattr(model_obs_space, "spaces", {})

    # Map first real telemetry sample to scenario start in RL map coordinates.
    if real_origin_xy is None:
        rx0, ry0 = float(frame.x_m), float(frame.y_m)
    else:
        rx0, ry0 = real_origin_xy

    mx = float(start_xy[0] + (frame.x_m - rx0) * pos_scale)
    my = float(start_xy[1] + (frame.y_m - ry0) * pos_scale)
    asv_pos = np.array([mx, my], dtype=np.float32)

    yaw_deg = float(frame.yaw_deg)
    yaw_rate = float(frame.yaw_rate)
    u_body = float(frame.u_body_mps)
    v_body = float(frame.v_body_mps)

    # Course from measured world velocity if moving; otherwise yaw.
    if float(frame.speed_mps) > 1e-4:
        course_deg = float(np.degrees(np.arctan2(float(frame.vx_mps), float(frame.vy_mps))))
    else:
        course_deg = yaw_deg

    # Path-relative features.
    d = np.linalg.norm(reference_path - asv_pos, axis=1)
    closest_idx = int(np.argmin(d))
    cte_abs = float(d[closest_idx])
    closest_pt = reference_path[closest_idx]
    tangent = path_tangent(reference_path, closest_idx)

    rel = asv_pos - closest_pt
    cross_z = float(tangent[0] * rel[1] - tangent[1] * rel[0])
    sign = 1.0 if cross_z > 0.0 else (-1.0 if cross_z < 0.0 else 0.0)
    cross_track_error = sign * cte_abs

    path_course_deg = float(np.degrees(np.arctan2(float(tangent[0]), float(tangent[1]))))
    course_error = wrap180(path_course_deg - course_deg)

    total_len = float(reference_path_s[-1]) if len(reference_path_s) else 1.0
    lookahead_distance = max(2.0, float(lookahead_fraction) * total_len)
    s_here = float(reference_path_s[closest_idx])
    s_target = min(total_len, s_here + lookahead_distance)
    lookahead_idx = int(np.searchsorted(reference_path_s, s_target, side="left"))
    lookahead_idx = int(np.clip(lookahead_idx, 0, len(reference_path) - 1))
    lookahead_pt = reference_path[lookahead_idx]
    lookahead_course_error = wrap180(bearing_deg(asv_pos, lookahead_pt) - course_deg)

    raw_lidar, raw_angles, sector_ranges, sector_closeness, sector_angles = pool_lidar_to_sectors(
        frame.lidar_m,
        lidar_index0_deg=lidar_index0_deg,
    )

    front_mask = np.abs(sector_angles) <= BLOCK_FRONT_DEG
    left_mask = (sector_angles <= -SIDE_ARC_MIN_DEG) & (sector_angles >= -SIDE_ARC_MAX_DEG)
    right_mask = (sector_angles >= SIDE_ARC_MIN_DEG) & (sector_angles <= SIDE_ARC_MAX_DEG)

    def pctl(mask, p=20.0):
        vals = sector_ranges[mask]
        return float(np.percentile(vals, p)) if vals.size else float(LIDAR_RANGE)

    front_clearance = pctl(front_mask, 10.0)
    left_clearance = pctl(left_mask, 20.0)
    right_clearance = pctl(right_mask, 20.0)

    block_alpha = float(np.clip((BLOCK_D_SAFE - front_clearance) / (BLOCK_D_SAFE - BLOCK_D_CRIT), 0.0, 1.0))
    if block_alpha <= 1e-6:
        local_target_cte = 0.0
    else:
        # In the simulator convention, starboard/right of path has negative CTE.
        if abs(right_clearance - left_clearance) < SIDE_CLEAR_TIE:
            side_cte_sign = -1.0
        elif right_clearance > left_clearance:
            side_cte_sign = -1.0
        else:
            side_cte_sign = +1.0
        local_target_cte = float(side_cte_sign * BYPASS_CTE * block_alpha)

    obs_values: Dict[str, np.ndarray] = {
        "lidar": sector_closeness.astype(np.float32),
        "u": np.array([u_body], dtype=np.float32),
        "v": np.array([v_body], dtype=np.float32),
        "yaw_rate": np.array([yaw_rate], dtype=np.float32),
        "cross_track_error": np.array([cross_track_error], dtype=np.float32),
        "course_error": np.array([course_error], dtype=np.float32),
        "lookahead_course_error": np.array([lookahead_course_error], dtype=np.float32),
        "front_clearance": np.array([front_clearance], dtype=np.float32),
        "side_clearance_diff": np.array([right_clearance - left_clearance], dtype=np.float32),
        "local_target_cte": np.array([local_target_cte], dtype=np.float32),
    }

    obs: Dict[str, np.ndarray] = {}
    for key, sp in spaces.items():
        if key not in obs_values:
            raise KeyError(
                f"Model expects observation key {key!r}, but udp_live_rl.py does not build it. "
                f"Available keys: {sorted(obs_values.keys())}"
            )
        obs[key] = np.asarray(obs_values[key], dtype=np.float32).reshape(sp.shape)

    aux = {
        "x": mx,
        "y": my,
        "yaw_deg": yaw_deg,
        "yaw_rate": yaw_rate,
        "u": u_body,
        "v": v_body,
        "speed": float(frame.speed_mps),
        "cross_track_error": float(cross_track_error),
        "course_error": float(course_error),
        "lookahead_course_error": float(lookahead_course_error),
        "front_clearance": float(front_clearance),
        "left_clearance": float(left_clearance),
        "right_clearance": float(right_clearance),
        "local_target_cte": float(local_target_cte),
    }

    return obs, aux, (mx, my), yaw_deg, yaw_rate, raw_lidar, raw_angles, sector_ranges, sector_closeness, sector_angles

# ---------------------------------------------------------------------------
# Command mapping
# ---------------------------------------------------------------------------

def action_to_rpm(a1: float, *, fixed_rpm: bool, cruise_rpm: float, rpm_delta: float, rpm_floor: float, rpm_ceil: float) -> float:
    if fixed_rpm:
        return float(cruise_rpm)
    return float(np.clip(cruise_rpm + rpm_delta * float(np.clip(a1, -1.0, 1.0)), rpm_floor, rpm_ceil))

def rpm_to_s2_cmd(rpm: float, *, rpm_max: float, s2_max_cmd: float) -> float:
    return np.clip((float(rpm) / max(float(rpm_max), 1e-6)) * float(s2_max_cmd), 0.0, float(s2_max_cmd))

def rudder_to_cmd(a0: float, *, sign: float, scale: float) -> float:
    return float(sign * scale * float(np.clip(a0, -1.0, 1.0)))

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_static_ref(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    map_width: float,
    map_height: float,
    reference_path: np.ndarray,
    obstacles: List[List[Tuple[float, float]]],
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    view_center_world: Tuple[float, float],
    px_per_m: float,
) -> None:
    vc_px = map_rect.center

    def w2s(pt):
        return log_viewer.world_to_screen(
            pt,
            view_center_world=view_center_world,
            view_center_px=vc_px,
            px_per_m=px_per_m,
        )

    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        border_world = [(0.0, 0.0), (map_width, 0.0), (map_width, map_height), (0.0, map_height)]
        pygame.draw.polygon(surface, (160, 80, 80), [w2s(p) for p in border_world], width=2)

        for poly in obstacles:
            poly_px = [w2s(p) for p in poly]
            pygame.draw.polygon(surface, (200, 60, 60), poly_px, width=0)
            pygame.draw.polygon(surface, (240, 180, 180), poly_px, width=1)

        ref = np.asarray(reference_path, dtype=np.float32)
        if ref.ndim == 2 and ref.shape[0] >= 2:
            pts = [w2s((float(p[0]), float(p[1]))) for p in ref]
            pygame.draw.lines(surface, (80, 220, 120), False, pts, 2)

        start_px = w2s(start_xy)
        goal_px = w2s(goal_xy)
        pygame.draw.circle(surface, (80, 220, 120), start_px, 6)
        pygame.draw.circle(surface, (220, 80, 220), goal_px, 6)
        pygame.draw.circle(surface, (0, 0, 0), start_px, 6, 1)
        pygame.draw.circle(surface, (0, 0, 0), goal_px, 6, 1)
    finally:
        surface.set_clip(prev_clip)


def format_sector_lines(sector_closeness: Optional[np.ndarray], sector_ranges: Optional[np.ndarray], *, per_line: int = 5) -> List[str]:
    if sector_closeness is None or sector_ranges is None:
        return []
    lines: List[str] = []
    n = len(sector_closeness)
    for i in range(0, n, per_line):
        parts = []
        for j in range(i, min(i + per_line, n)):
            parts.append(f"{j:02d}:c{float(sector_closeness[j]):.2f}/d{float(sector_ranges[j]):.1f}")
        lines.append("  ".join(parts))
    return lines


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    global LIDAR_POOLING_MODE, FEASIBILITY_SAFE_WIDTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--local-port", type=int, default=5000)
    ap.add_argument("--server-ip", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=5050)
    ap.add_argument("--record-log", default=None, help="Append received UDP lines and RL action lines to this log file")
    ap.add_argument("--record-video", action="store_true", help="Record the live pygame display to MP4")
    ap.add_argument("--out-video", default="udp_live_rl_record.mp4", help="Output MP4 file when --record-video is enabled")
    ap.add_argument("--video-fps", type=float, default=10.0, help="Video recording FPS")

    ap.add_argument("--model-path", default="best_model.zip")
    ap.add_argument("--test-case", type=int, default=0)
    ap.add_argument("--lookahead-fraction", type=float, default=LOOKAHEAD_FRACTION)
    ap.add_argument("--pos-scale", type=float, default=POS_SCALE)
    ap.add_argument("--lidar-index0-deg", type=float, default=getattr(log_viewer, "LIDAR_INDEX_DEG", 0.0))
    ap.add_argument(
        "--feasibility-safe-width",
        type=float,
        default=FEASIBILITY_SAFE_WIDTH,
        help="Safety-adjusted vessel width used by feasibility pooling.",
    )

    ap.add_argument("--fixed-rpm", action="store_true")
    ap.add_argument("--cruise-rpm", type=float, default=CRUISE_RPM)
    ap.add_argument("--rpm-delta", type=float, default=RPM_DELTA)
    ap.add_argument("--rpm-floor", type=float, default=RPM_FLOOR)
    ap.add_argument("--rpm-ceil", type=float, default=RPM_CEIL)
    ap.add_argument("--rpm-max", type=float, default=RPM_MAX)
    ap.add_argument("--s2-max-cmd", type=float, default=S2_MAX_CMD)
    ap.add_argument("--rudder-sign", type=float, default=RUDDER_SIGN)
    ap.add_argument("--rudder-scale", type=float, default=RUDDER_SCALE)
    ap.add_argument("--shadow", action="store_true", help="compute actions but do not send $CMD")

    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--zoom", type=float, default=30.0)
    ap.add_argument("--max-path", type=int, default=20000)
    args = ap.parse_args()
    FEASIBILITY_SAFE_WIDTH = float(args.feasibility_safe_width)

    model = SAC.load(args.model_path)
    model_obs_space = model.observation_space
    print(f"[RL] Loaded SAC model: {args.model_path}")
    print(f"[RL] Observation space: {model_obs_space}")
    print(f"[LiDAR] pooling={LIDAR_POOLING_MODE} safe_width={FEASIBILITY_SAFE_WIDTH:.3f} m")

    scenario = TestCase()
    sx, sy, gx, gy = scenario.position(test_case=args.test_case)
    start_xy = (float(sx), float(sy))
    goal_xy = (float(gx), float(gy))
    obstacles = scenario.obstacles(test_case=args.test_case)
    reference_path, reference_path_s = generate_reference_path(float(sx), float(sy), float(gx), float(gy))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind_ip, args.local_port))
    sock.settimeout(0.05)

    sock.sendto(b"START\n", (args.server_ip, args.server_port))
    print(f"[UDP] Sent START to {args.server_ip}:{args.server_port}")
    print(f"[UDP] Listening on {args.bind_ip}:{args.local_port}")
    if args.shadow:
        print("[SAFETY] Shadow mode enabled: actions are computed but $CMD is NOT sent.")

    log = None
    if args.record_log:
        log = open(args.record_log, "a", encoding="utf-8", buffering=1)
        log.write(
            f"#CONFIG,"
            f"model_path={args.model_path},"
            f"test_case={args.test_case},"
            f"fixed_rpm={int(args.fixed_rpm)},"
            f"cruise_rpm={args.cruise_rpm},"
            f"rpm_delta={args.rpm_delta},"
            f"rpm_floor={args.rpm_floor},"
            f"rpm_ceil={args.rpm_ceil},"
            f"rudder_sign={args.rudder_sign},"
            f"rudder_scale={args.rudder_scale},"
            f"max_rudder_rate_dps={MAX_RUDDER_RATE_DPS},"
            f"max_rudder_deg={MAX_RUDDER_DEG_FOR_RATE},"
            f"lidar_index0_deg={args.lidar_index0_deg},"
            f"lidar_pooling={LIDAR_POOLING_MODE},"
            f"feasibility_safe_width={FEASIBILITY_SAFE_WIDTH},"
            f"lookahead_fraction={args.lookahead_fraction},"
            f"pos_scale={args.pos_scale},"
            f"BLOCK_D_SAFE={BLOCK_D_SAFE},"
            f"BLOCK_D_CRIT={BLOCK_D_CRIT},"
            f"SIDE_CLEAR_TIE={SIDE_CLEAR_TIE},"
            f"BYPASS_CTE={BYPASS_CTE}\n"
        )
        print(f"[UDP] Recording received lines to: {args.record_log}")

    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    rx_lines = 0
    rx_frames = 0

    real_origin_xy: Optional[Tuple[float, float]] = None
    latest_obs: Optional[Dict[str, np.ndarray]] = None
    latest_action: Optional[np.ndarray] = None
    latest_aux: Optional[Dict[str, float]] = None
    latest_sector_ranges: Optional[np.ndarray] = None
    latest_sector_closeness: Optional[np.ndarray] = None
    latest_sector_angles: Optional[np.ndarray] = None
    latest_raw_lidar: Optional[np.ndarray] = None
    latest_raw_angles: Optional[np.ndarray] = None
    rudder_cmd = 0.0
    raw_rudder_cmd = 0.0
    prev_rudder_cmd = 0.0
    prev_rudder_t = None
    rpm_cmd = args.cruise_rpm
    thrust_cmd = rpm_to_s2_cmd(rpm_cmd, rpm_max=args.rpm_max, s2_max_cmd=args.s2_max_cmd)

    pygame.init()
    pygame.display.set_caption(f"Bluefin UDP SAC live RL bridge [{LIDAR_POOLING_MODE} pooling]")
    win_w, win_h = 1280, 720
    text_w = 840
    map_w = win_w - text_w
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    video_writer = None
    cv2 = None
    if args.record_video:
        try:
            import cv2 as _cv2  # imported lazily so normal use does not require OpenCV
            cv2 = _cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                args.out_video,
                fourcc,
                float(args.video_fps),
                (int(win_w), int(win_h)),
            )
            if not video_writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            print(f"[VIDEO] Recording display to: {args.out_video} @ {args.video_fps:g} FPS")
        except Exception as e:
            print(f"[VIDEO] Could not start recording: {type(e).__name__}: {e}")
            video_writer = None
            cv2 = None

    font = pygame.font.SysFont("consolas", 17) or pygame.font.Font(None, 17)
    small = pygame.font.SysFont("consolas", 14) or pygame.font.Font(None, 14)

    paused = False
    show_full_lidar = bool(args.full)
    show_map = not bool(args.no_map)
    follow_mode = True
    lidar_scroll = 0
    px_per_m = float(args.zoom)
    view_center_world = (0.0, 0.0)
    path_world: List[Tuple[float, float]] = []
    frame: Optional[BluefinFrame] = None
    cached_lidar_lines: List[str] = []
    cached_lidar_key = None

    lidar_draw_angles = np.linspace(-float(LIDAR_SWATH) / 2.0, float(LIDAR_SWATH) / 2.0, int(LIDAR_BEAMS), dtype=np.float32)

    running = True
    while running:
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
                    lidar_scroll = max(0, lidar_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    lidar_scroll += 1
                elif event.key == pygame.K_m:
                    show_map = not show_map
                elif event.key == pygame.K_g:
                    follow_mode = not follow_mode
                elif event.key == pygame.K_c:
                    path_world = []
                elif event.key == pygame.K_o:
                    if frame is not None:
                        real_origin_xy = (float(frame.x_m), float(frame.y_m))
                        path_world = [(float(start_xy[0]), float(start_xy[1]))]
                        view_center_world = start_xy
                        print("[VIEW] Origin reset: current real pose maps to scenario start")
                elif event.key == pygame.K_p:
                    out = "snapshot.png"
                    pygame.image.save(screen, out)
                    print(f"[VIEW] Saved snapshot: {out}")

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

        try:
            msg, _addr = sock.recvfrom(65535)
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

                    try:
                        (
                            latest_obs,
                            latest_aux,
                            mapped_xy,
                            _yaw_deg,
                            _yaw_rate,
                            raw_lidar,
                            raw_lidar_angles,
                            latest_sector_ranges,
                            latest_sector_closeness,
                            latest_sector_angles,
                        ) = frame_to_rl_obs(
                            frame,
                            model_obs_space=model_obs_space,
                            real_origin_xy=real_origin_xy,
                            start_xy=start_xy,
                            reference_path=reference_path,
                            reference_path_s=reference_path_s,
                            lookahead_fraction=args.lookahead_fraction,
                            pos_scale=args.pos_scale,
                            lidar_index0_deg=args.lidar_index0_deg,
                        )
                        latest_raw_lidar = raw_lidar
                        latest_raw_angles = raw_lidar_angles
                    except Exception as e:
                        print(f"[OBS ERROR] {type(e).__name__}: {e}")
                        latest_obs = None
                        latest_aux = None
                        mapped_xy = (0.0, 0.0)

# ---------------------------------------------------------------------------------------------------
#                               ACTION
# ---------------------------------------------------------------------------------------------------

                    if latest_obs is not None:
                        action, _ = model.predict(latest_obs, deterministic=True)
                        latest_action = np.asarray(action, dtype=np.float32).reshape(-1)

                        raw_rudder_cmd = rudder_to_cmd(
                            latest_action[0],
                            sign=args.rudder_sign,
                            scale=args.rudder_scale,
                        )

                        # Store previous limited command for logging and rate calculation.
                        prev_rudder_cmd_for_log = prev_rudder_cmd

                        # Use telemetry frame time, not pygame FPS.
                        t_now = float(frame.t_sec)
                        if prev_rudder_t is None:
                            dt_cmd = 0.1
                        else:
                            dt_cmd = float(np.clip(t_now - prev_rudder_t, MIN_CMD_DT, MAX_CMD_DT))

                        # Simulation mapping:
                        # ±100 command percent corresponds approximately to ±40 deg rudder.
                        # 20 deg/s therefore corresponds to 50 command-percent/s.
                        max_cmd_rate_per_s = (
                            abs(float(args.rudder_scale))
                            * MAX_RUDDER_RATE_DPS
                            / MAX_RUDDER_DEG_FOR_RATE
                        )

                        # This matches the ship_model.py style:
                        # delta_dot = clip(delta_cmd - delta, ±max_rate)
                        rudder_error = raw_rudder_cmd - prev_rudder_cmd_for_log

                        rudder_cmd_dot = float(np.clip(
                            rudder_error,
                            -max_cmd_rate_per_s,
                            +max_cmd_rate_per_s,
                        ))

                        rudder_cmd = float(prev_rudder_cmd_for_log + rudder_cmd_dot * dt_cmd)

                        rudder_cmd = float(np.clip(
                            rudder_cmd,
                            -abs(float(args.rudder_scale)),
                            +abs(float(args.rudder_scale)),
                        ))

                        # Actual realised rate of the limited command that will be sent.
                        # Units: command-percent per second.
                        rudder_rate_cmd_per_s = float(
                            (rudder_cmd - prev_rudder_cmd_for_log) / max(dt_cmd, 1e-6)
                        )

                        prev_rudder_cmd = rudder_cmd
                        prev_rudder_t = t_now

                        rpm_cmd = action_to_rpm(
                            latest_action[1],
                            fixed_rpm=bool(args.fixed_rpm),
                            cruise_rpm=args.cruise_rpm,
                            rpm_delta=args.rpm_delta,
                            rpm_floor=args.rpm_floor,
                            rpm_ceil=args.rpm_ceil,
                        )
                        thrust_cmd = rpm_to_s2_cmd(rpm_cmd, rpm_max=args.rpm_max, s2_max_cmd=args.s2_max_cmd)

                        # # manual rudder/throttle test
                        # rudder_cmd = -100
                        # thrust_cmd = 0

                        # # turn rate limiter off
                        # rudder_cmd = float(raw_rudder_cmd)

                        command = f"$CMD,{rudder_cmd:.2f},{thrust_cmd:.2f}"
                        command = f"$CMD,{0.0},{0.0}"
                        
                        sent_command = False
                        if not args.shadow:
                            sock.sendto(command.encode(), (args.server_ip, args.server_port))
                            sent_command = True

                        # If recording the raw UDP log, also record the action/command
                        # associated with each decoded LiDAR frame. Lines beginning
                        # with '#ACTION' are ignored by log_parser.py, but preserve
                        # enough information to replay/debug the control decision.

                        # if log is not None:
                        #     log.write(
                        #         f"#ACTION,t={frame.t_sec:.6f},ts={frame.ts_str},seq={frame.seq},"
                        #         f"a0={float(latest_action[0]):+.6f},a1={float(latest_action[1]):+.6f},"
                        #         f"rudder={rudder_cmd:+.3f},rpm={rpm_cmd:.3f},S2={thrust_cmd:.3f},"
                        #         f"shadow={int(args.shadow)},sent={int(sent_command)}\n"
                        #     )

                        if log is not None:
                            log.write(
                                f"#ACTION,"
                                f"t_frame={frame.t_sec:.6f},"
                                f"t_wall={time.monotonic():.6f},"
                                f"ts={frame.ts_str},seq={frame.seq},"
                                f"test_case={args.test_case},"
                                f"shadow={int(args.shadow)},sent={int(sent_command)},"
                                f"a0={float(latest_action[0]):+.6f},"
                                f"a1={float(latest_action[1]):+.6f},"
                                f"raw_rudder={raw_rudder_cmd:+.3f},"
                                f"limited_rudder={rudder_cmd:+.3f},"
                                f"dt_cmd={dt_cmd:.4f},"
                                f"rudder_rate={rudder_rate_cmd_per_s:+.3f},"
                                f"max_rudder_rate={max_cmd_rate_per_s:.3f},"
                                f"rpm={rpm_cmd:.3f},"
                                f"S2={thrust_cmd:.3f},"
                                f"S1_telem={frame.s1},S2_telem={frame.s2},"
                                f"x_real={frame.x_m:+.3f},y_real={frame.y_m:+.3f},yaw_real={frame.yaw_deg:+.2f},"
                                f"speed={frame.speed_mps:.3f},u={frame.u_body_mps:+.3f},v={frame.v_body_mps:+.3f},yaw_rate={frame.yaw_rate:+.3f},"
                                f"x_rl={latest_aux['x']:+.3f},y_rl={latest_aux['y']:+.3f},"
                                f"cte={latest_aux['cross_track_error']:+.3f},"
                                f"course={latest_aux['course_error']:+.2f},"
                                f"lookahead={latest_aux['lookahead_course_error']:+.2f},"
                                f"front={latest_aux['front_clearance']:.3f},"
                                f"left={latest_aux['left_clearance']:.3f},"
                                f"right={latest_aux['right_clearance']:.3f},"
                                f"local_cte={latest_aux['local_target_cte']:+.3f},"
                                f"cmd='{command}'\n"
                            )

                        path_world.append(mapped_xy)
                        if len(path_world) > args.max_path:
                            path_world = path_world[-args.max_path:]
                        if follow_mode:
                            view_center_world = mapped_xy

        # ------------------------------------------------------------------
        # Draw UI
        # ------------------------------------------------------------------
        screen.fill((20, 20, 25))
        map_rect = pygame.Rect(text_w, 0, map_w, win_h)

        y = 10
        line_h = 21
        header_lines = [
            f"UDP local={args.bind_ip}:{args.local_port}  server={args.server_ip}:{args.server_port}  shadow={args.shadow}",
            f"RX lines={rx_lines} frames={rx_frames} {'PAUSED' if paused else 'RUNNING'}  lidar_list={'225 beams' if show_full_lidar else '25 sectors'}(F)",
            f"Map={'ON' if show_map else 'OFF'}(M) follow={'ON' if follow_mode else 'OFF'}(G) zoom={px_per_m:.1f}px/m",
        ]
        if frame is None:
            header_lines.append("Waiting for first decoded LiDAR frame...")
        else:
            header_lines += [
                f"ts={frame.ts_str} t={frame.t_sec:.3f}s seq={frame.seq} hdg_ref={frame.hdg_ref_deg}",
                f" ",
                f"Real:",
                f"x={frame.x_m:+.3f} y={frame.y_m:+.3f} hdg={frame.yaw_deg:+.2f} yaw_rate={frame.yaw_rate:+.2f} speed={frame.speed_mps:.3f}",
                f" ",
            ]
            if latest_aux is not None:
                header_lines += [
                    f"RL map:",
                    f"x={latest_aux['x']:+.3f} y={latest_aux['y']:+.3f} hdg={latest_aux['yaw_deg']:+.2f}",
                    f"Obs: u={latest_aux['u']:+.3f} v={latest_aux['v']:+.3f} cte={latest_aux['cross_track_error']:+.3f}",
                    f"Obs: course={latest_aux['course_error']:+.2f} lookahead={latest_aux['lookahead_course_error']:+.2f}",
                    f"Obs: front={latest_aux['front_clearance']:.2f} L={latest_aux['left_clearance']:.2f} R={latest_aux['right_clearance']:.2f} local_cte={latest_aux['local_target_cte']:+.3f}",
                ]
        if latest_action is not None:
            header_lines += [
                f"Action: rud={float(latest_action[0]):+.3f}    thr={float(latest_action[1]):+.3f}",
                f"Command: rudder={rudder_cmd:+.1f}   throttle={thrust_cmd:.1f}(rpm={rpm_cmd:.2f})",
                f"a0={float(latest_action[0]):+.6f}    a1={float(latest_action[1]):+.6f}",
                f"raw_rudder={raw_rudder_cmd:+.3f}    rudder={rudder_cmd:+.3f}",
                f"rpm={rpm_cmd:.3f}    S2={thrust_cmd:.3f}",
            ]

        for s in header_lines:
            screen.blit(font.render(s, True, (235, 235, 245)), (10, y))
            y += line_h
        y += 8

        if frame is not None:
            if show_full_lidar:
                title = "Policy raw LiDAR swath: 225 beam distances (F)"
                cached_key = (rx_frames, show_full_lidar, "raw225")
                if cached_key != cached_lidar_key:
                    cached_lidar_lines = pr.format_range_lines(latest_raw_lidar, per_line=10, precision=1)
                    cached_lidar_key = cached_key
            else:
                title = "Policy 25-sector LiDAR: sector distances (F)"
                cached_key = (rx_frames, show_full_lidar, "sector")
                if cached_key != cached_lidar_key:
                    cached_lidar_lines = pr.format_sector_lines(latest_sector_closeness, latest_sector_ranges, per_line=5)
                    cached_lidar_key = cached_key

            max_lines = max(1, (win_h - y - 20) // 18)
            max_scroll = max(0, len(cached_lidar_lines) - max_lines)
            lidar_scroll = min(lidar_scroll, max_scroll)
            screen.blit(font.render(f"{title} scroll {lidar_scroll}/{max_scroll}", True, (200, 200, 210)), (10, y))
            y += 22
            for s in cached_lidar_lines[lidar_scroll: lidar_scroll + max_lines]:
                screen.blit(small.render(s, True, (210, 210, 220)), (10, y))
                y += 18

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
                    index0_deg=args.lidar_index0_deg,
                )

            # Right panel intentionally has no text overlay.  All policy/command
            # information is shown in the left panel.
            pr.draw_policy_map(
                screen,
                map_rect,
                map_width=MAP_WIDTH,
                map_height=MAP_HEIGHT,
                reference_path=reference_path,
                obstacles=obstacles,
                start_xy=start_xy,
                goal_xy=goal_xy,
                trajectory=path_world,
                current_xy=current_world,
                heading_deg=yaw_for_draw,
                raw_lidar_ranges=latest_raw_lidar if current_world is not None else None,
                raw_lidar_angles=latest_raw_angles if current_world is not None else None,
                sector_ranges=latest_sector_ranges if current_world is not None else None,
                sector_closeness=latest_sector_closeness if current_world is not None else None,
                sector_angles=latest_sector_angles if current_world is not None else None,
                view_center_world=view_center_world,
                px_per_m=px_per_m,
                # F toggles the visual LiDAR mode too: sector view by default,
                # raw 225-beam swath only when requested. This makes the sector
                # representation much easier to see during field testing.
                show_raw_lidar=show_full_lidar,
                show_sector_lidar=not show_full_lidar,
                show_icon=True,
                show_hull=True,
                status_lines=None,
                font=None,
            )

        pygame.display.flip()

        if video_writer is not None and cv2 is not None:
            try:
                frame_rgb = pygame.surfarray.array3d(screen)
                frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            except Exception as e:
                print(f"[VIDEO] Write failed: {type(e).__name__}: {e}")
                video_writer.release()
                video_writer = None

        clock.tick(args.fps)

    if video_writer is not None:
        video_writer.release()
        print(f"[VIDEO] Saved {args.out_video}")
    if log is not None:
        log.close()
    sock.close()
    pygame.quit()
    print("[UDP] Exit")


if __name__ == "__main__":
    main()
