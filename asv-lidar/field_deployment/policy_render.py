"""Shared policy/simulation-style rendering helpers for ASV field tools.

This module is intentionally independent from Gymnasium/SB3.  It draws the
same visual elements used by the current RL environment:

- true map/pool boundary
- reference path, start/goal, obstacles
- ASV trajectory
- raw LiDAR beams
- pooled LiDAR sector rays/end-points
- hull collision polygon and optional boat icon

Both ``udp_live_rl.py`` and ``log_viewer.py`` can import these helpers so the
live and offline displays do not drift apart.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pygame

try:
    from ship_model import VESSEL_LENGTH, VESSEL_WIDTH, HULL_MARGIN, HULL_FORWARD_SHIFT, LIDAR_OFFSET_M
except Exception:  # fallback for viewer-only use
    VESSEL_LENGTH = 1.725
    VESSEL_WIDTH = 0.50
    HULL_MARGIN = 0.15
    HULL_FORWARD_SHIFT = 0.0
    LIDAR_OFFSET_M = VESSEL_LENGTH / 2.0

try:
    from images import BOAT_ICON
except Exception:
    BOAT_ICON = None

try:
    from asv_lidar import LIDAR_RANGE, LIDAR_SWATH, LIDAR_SECTORS
except Exception:
    LIDAR_RANGE = 16.0
    LIDAR_SWATH = 240.0
    LIDAR_SECTORS = 25

Color = Tuple[int, int, int]
Point = Tuple[float, float]

def world_to_screen(
    xy_world: Point,
    *,
    view_center_world: Point,
    view_center_px: Tuple[int, int],
    px_per_m: float,
) -> Tuple[int, int]:
    """Convert simulator/world coordinate to pygame pixels.

    Convention: world +x is right, +y is up; pygame +y is down.
    """
    x, y = xy_world
    cx_w, cy_w = view_center_world
    cx_px, cy_px = view_center_px
    sx = cx_px + (float(x) - cx_w) * px_per_m
    sy = cy_px - (float(y) - cy_w) * px_per_m
    return int(round(sx)), int(round(sy))

def _w2s_factory(map_rect: pygame.Rect, view_center_world: Point, px_per_m: float):
    vc_px = map_rect.center

    def w2s(pt: Point) -> Tuple[int, int]:
        return world_to_screen(pt, view_center_world=view_center_world, view_center_px=vc_px, px_per_m=px_per_m)

    return w2s

def _as_path_array(path: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.asarray(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return None
    return arr[:, :2]

def draw_reference_scene(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    map_width: Optional[float] = None,
    map_height: Optional[float] = None,
    reference_path: Optional[Sequence[Sequence[float]]] = None,
    obstacles: Optional[Sequence[Sequence[Point]]] = None,
    start_xy: Optional[Point] = None,
    goal_xy: Optional[Point] = None,
    view_center_world: Point = (0.0, 0.0),
    px_per_m: float = 25.0,
    show_axes: bool = True,
    show_boundary: bool = True,
) -> None:
    """Draw static background, reference path, obstacles, and start/goal."""
    pygame.draw.rect(surface, (10, 10, 12), map_rect)
    pygame.draw.rect(surface, (80, 80, 90), map_rect, width=2)
    w2s = _w2s_factory(map_rect, view_center_world, px_per_m)

    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        if show_axes:
            cx, cy = map_rect.center
            pygame.draw.line(surface, (40, 40, 45), (map_rect.left, cy), (map_rect.right, cy), 1)
            pygame.draw.line(surface, (40, 40, 45), (cx, map_rect.top), (cx, map_rect.bottom), 1)
            # 1 m scale bar
            bar_len_px = int(round(px_per_m))
            bar_x0 = map_rect.left + 16
            bar_y0 = map_rect.bottom - 24
            pygame.draw.line(surface, (180, 180, 190), (bar_x0, bar_y0), (bar_x0 + bar_len_px, bar_y0), 3)

        if show_boundary and map_width is not None and map_height is not None:
            border_world = [(0.0, 0.0), (float(map_width), 0.0), (float(map_width), float(map_height)), (0.0, float(map_height))]
            pygame.draw.polygon(surface, (155, 100, 100), [w2s(p) for p in border_world], width=2)

        if obstacles:
            for poly in obstacles:
                pts = [w2s((float(p[0]), float(p[1]))) for p in poly]
                if len(pts) >= 3:
                    pygame.draw.polygon(surface, (190, 55, 55), pts, width=0)
                    pygame.draw.polygon(surface, (245, 185, 185), pts, width=1)

        ref = _as_path_array(reference_path)
        if ref is not None:
            pts = [w2s((float(p[0]), float(p[1]))) for p in ref]
            pygame.draw.lines(surface, (80, 220, 120), False, pts, 2)

        if start_xy is not None:
            p = w2s(start_xy)
            pygame.draw.circle(surface, (80, 220, 120), p, 6)
            pygame.draw.circle(surface, (0, 0, 0), p, 6, 1)
        if goal_xy is not None:
            p = w2s(goal_xy)
            pygame.draw.circle(surface, (220, 80, 220), p, 6)
            pygame.draw.circle(surface, (0, 0, 0), p, 6, 1)
    finally:
        surface.set_clip(prev_clip)

def draw_trajectory(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    trajectory: Optional[Sequence[Point]],
    view_center_world: Point,
    px_per_m: float,
    color: Color = (120, 220, 255),
    width: int = 2,
) -> None:
    if not trajectory or len(trajectory) < 2:
        return
    w2s = _w2s_factory(map_rect, view_center_world, px_per_m)
    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        pts = [w2s((float(x), float(y))) for x, y in trajectory]
        pygame.draw.lines(surface, color, False, pts, width)
    finally:
        surface.set_clip(prev_clip)


def vessel_hull_polygon_world(x: float, y: float, heading_deg: float) -> List[Point]:
    """Return collision hull polygon in world coordinates.

    Matches the rl_env.py convention: heading 0 points along +y.
    """
    L = float(VESSEL_LENGTH) + 2.0 * float(HULL_MARGIN)
    W = float(VESSEL_WIDTH) + 2.0 * float(HULL_MARGIN)
    half_L = 0.5 * L
    half_W = 0.5 * W
    shift = float(HULL_FORWARD_SHIFT)

    h = math.radians(float(heading_deg))
    sin_h = math.sin(h)
    cos_h = math.cos(h)

    local = [
        (+half_L + shift, +half_W),
        (+half_L + shift, -half_W),
        (-half_L + shift, -half_W),
        (-half_L + shift, +half_W),
    ]

    poly: List[Point] = []
    for x_forward, y_left in local:
        wx = float(x) + x_forward * sin_h - y_left * cos_h
        wy = float(y) + x_forward * cos_h + y_left * sin_h
        poly.append((wx, wy))
    return poly


def _load_boat_icon() -> Optional[pygame.Surface]:
    if BOAT_ICON is None:
        return None
    try:
        surf = pygame.image.frombuffer(BOAT_ICON["bytes"], BOAT_ICON["size"], BOAT_ICON["format"])
        return surf.convert_alpha()
    except Exception:
        return None

_BOAT_ICON_SURFACE: Optional[pygame.Surface] = None


def draw_vessel(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    x: float,
    y: float,
    heading_deg: float,
    view_center_world: Point,
    px_per_m: float,
    draw_icon: bool = True,
    draw_hull: bool = True,
) -> None:
    """Draw the boat icon and the actual collision hull.

    The hull is drawn on TOP of the icon with a thick high-contrast outline so
    it is easy to compare with the collision geometry used by rl_env.py.
    """
    w2s = _w2s_factory(map_rect, view_center_world, px_per_m)
    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        # Draw icon first so the collision hull remains visible on top.
        if draw_icon:
            global _BOAT_ICON_SURFACE
            if _BOAT_ICON_SURFACE is None:
                _BOAT_ICON_SURFACE = _load_boat_icon()
            if _BOAT_ICON_SURFACE is not None:
                icon = pygame.transform.rotozoom(_BOAT_ICON_SURFACE, -float(heading_deg), 1.0)
                rect = icon.get_rect(center=w2s((x, y)))
                surface.blit(icon, rect)

        if draw_hull:
            poly = vessel_hull_polygon_world(x, y, heading_deg)
            pts = [w2s(p) for p in poly]
            if len(pts) >= 3:
                # Match rl_env.py: draw the collision hull as a red outline on
                # top of the vessel icon. No fill, so the icon remains visible.
                pygame.draw.polygon(surface, (255, 0, 0), pts, width=3)
                pygame.draw.polygon(surface, (255, 230, 230), pts, width=1)
                for pt in pts:
                    pygame.draw.circle(surface, (255, 0, 0), pt, 3)

        # Heading/nose line.
        h = math.radians(float(heading_deg))
        nose = (
            float(x) + float(VESSEL_LENGTH) * 0.85 * math.sin(h),
            float(y) + float(VESSEL_LENGTH) * 0.85 * math.cos(h),
        )
        pygame.draw.line(surface, (255, 255, 255), w2s((x, y)), w2s(nose), 2)
        pygame.draw.circle(surface, (255, 255, 255), w2s((x, y)), 3)
    finally:
        surface.set_clip(prev_clip)

def draw_raw_lidar(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    x: float,
    y: float,
    heading_deg: float,
    raw_ranges: Optional[Sequence[float]],
    raw_angles: Optional[Sequence[float]],
    view_center_world: Point,
    px_per_m: float,
    max_range: float = LIDAR_RANGE,
    lidar_offset_m: float = LIDAR_OFFSET_M,
    every: int = 1,
) -> None:
    if raw_ranges is None or raw_angles is None:
        return
    ranges = np.asarray(raw_ranges, dtype=np.float32).ravel()
    angles = np.asarray(raw_angles, dtype=np.float32).ravel()
    if ranges.size == 0 or angles.size == 0:
        return
    n = min(ranges.size, angles.size)
    ranges = ranges[:n]
    angles = angles[:n]
    w2s = _w2s_factory(map_rect, view_center_world, px_per_m)
    h0 = math.radians(float(heading_deg))
    ox = float(x) + float(lidar_offset_m) * math.sin(h0)
    oy = float(y) + float(lidar_offset_m) * math.cos(h0)
    origin = w2s((ox, oy))

    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        step = max(1, int(every))
        for dist, rel_angle in zip(ranges[::step], angles[::step]):
            d = float(np.clip(dist, 0.0, max_range))
            a = math.radians(float(heading_deg) + float(rel_angle))
            end = (ox + d * math.sin(a), oy + d * math.cos(a))
            color = (55, 55, 120) if d >= 0.98 * float(max_range) else (120, 120, 255)
            pygame.draw.line(surface, color, origin, w2s(end), 1)
    finally:
        surface.set_clip(prev_clip)

def closeness_color(c: float) -> Color:
    c = float(np.clip(c, 0.0, 1.0))
    if c < 0.35:
        return (60, 160, 255)
    if c < 0.70:
        return (240, 210, 60)
    return (255, 80, 60)

def draw_lidar_sectors(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    x: float,
    y: float,
    heading_deg: float,
    sector_ranges: Optional[Sequence[float]],
    sector_closeness: Optional[Sequence[float]],
    sector_angles: Optional[Sequence[float]] = None,
    view_center_world: Point,
    px_per_m: float,
    max_range: float = LIDAR_RANGE,
    lidar_offset_m: float = 0.0,
) -> None:
    """Draw the policy's pooled LiDAR sectors.

    This intentionally avoids drawing filled wedges/overlays. Each sector is a
    thick center ray whose endpoint is the pooled sector distance. This makes
    the policy observation much easier to read in the live display.
    """
    if sector_ranges is None or sector_closeness is None:
        return
    ranges = np.asarray(sector_ranges, dtype=np.float32).ravel()
    close = np.asarray(sector_closeness, dtype=np.float32).ravel()
    if ranges.size == 0 or close.size == 0:
        return
    n = min(ranges.size, close.size)
    ranges = ranges[:n]
    close = close[:n]
    if sector_angles is None:
        angles = np.linspace(-float(LIDAR_SWATH) / 2.0, float(LIDAR_SWATH) / 2.0, n, dtype=np.float32)
    else:
        angles = np.asarray(sector_angles, dtype=np.float32).ravel()[:n]

    w2s = _w2s_factory(map_rect, view_center_world, px_per_m)
    h0 = math.radians(float(heading_deg))
    ox = float(x) + float(lidar_offset_m) * math.sin(h0)
    oy = float(y) + float(lidar_offset_m) * math.cos(h0)
    origin = w2s((ox, oy))

    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        # Draw a faint max-range fan so the sector FOV is still clear.
        for rel_angle in angles:
            a = math.radians(float(heading_deg) + float(rel_angle))
            end = (ox + 0.98 * max_range * math.sin(a), oy + 0.98 * max_range * math.cos(a))
            pygame.draw.line(surface, (35, 35, 50), origin, w2s(end), 1)

        for d, c, rel_angle in zip(ranges, close, angles):
            dist = float(np.clip(d, 0.0, max_range))
            a = math.radians(float(heading_deg) + float(rel_angle))
            end = (ox + dist * math.sin(a), oy + dist * math.cos(a))
            col = closeness_color(float(c))
            pygame.draw.line(surface, col, origin, w2s(end), 4)
            pygame.draw.circle(surface, col, w2s(end), 5)
    finally:
        surface.set_clip(prev_clip)

def draw_policy_scene(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    map_width: Optional[float] = None,
    map_height: Optional[float] = None,
    reference_path: Optional[Sequence[Sequence[float]]] = None,
    obstacles: Optional[Sequence[Sequence[Point]]] = None,
    start_xy: Optional[Point] = None,
    goal_xy: Optional[Point] = None,
    trajectory: Optional[Sequence[Point]] = None,
    current_xy: Optional[Point] = None,
    heading_deg: Optional[float] = None,
    raw_lidar_ranges: Optional[Sequence[float]] = None,
    raw_lidar_angles: Optional[Sequence[float]] = None,
    sector_ranges: Optional[Sequence[float]] = None,
    sector_closeness: Optional[Sequence[float]] = None,
    sector_angles: Optional[Sequence[float]] = None,
    view_center_world: Point = (0.0, 0.0),
    px_per_m: float = 25.0,
    lidar_offset_m: float = LIDAR_OFFSET_M,
    show_raw_lidar: bool = True,
    show_sector_lidar: bool = True,
    show_icon: bool = True,
    show_hull: bool = True,
) -> None:
    """Draw a full policy-view scene in one call."""
    draw_reference_scene(
        surface,
        map_rect,
        map_width=map_width,
        map_height=map_height,
        reference_path=reference_path,
        obstacles=obstacles,
        start_xy=start_xy,
        goal_xy=goal_xy,
        view_center_world=view_center_world,
        px_per_m=px_per_m,
    )

    if trajectory:
        draw_trajectory(surface, map_rect, trajectory=trajectory, view_center_world=view_center_world, px_per_m=px_per_m)

    if current_xy is not None and heading_deg is not None:
        if show_raw_lidar:
            draw_raw_lidar(
                surface,
                map_rect,
                x=float(current_xy[0]),
                y=float(current_xy[1]),
                heading_deg=float(heading_deg),
                raw_ranges=raw_lidar_ranges,
                raw_angles=raw_lidar_angles,
                view_center_world=view_center_world,
                px_per_m=px_per_m,
                lidar_offset_m=lidar_offset_m,
                every=2,
            )
        if show_sector_lidar:
            draw_lidar_sectors(
                surface,
                map_rect,
                x=float(current_xy[0]),
                y=float(current_xy[1]),
                heading_deg=float(heading_deg),
                sector_ranges=sector_ranges,
                sector_closeness=sector_closeness,
                sector_angles=sector_angles,
                view_center_world=view_center_world,
                px_per_m=px_per_m,
                lidar_offset_m=lidar_offset_m,
            )
        draw_vessel(
            surface,
            map_rect,
            x=float(current_xy[0]),
            y=float(current_xy[1]),
            heading_deg=float(heading_deg),
            view_center_world=view_center_world,
            px_per_m=px_per_m,
            draw_icon=show_icon,
            draw_hull=show_hull,
        )

def format_sector_lines(sector_closeness: Optional[np.ndarray], sector_ranges: Optional[np.ndarray], *, per_line: int = 5) -> List[str]:
    """Format the 25 policy sectors as distances only.

    The left text panel already has enough high-level observation info, so for
    the LiDAR list we keep it simple: sector index and pooled distance.
    """
    if sector_ranges is None:
        return []
    r = np.asarray(sector_ranges, dtype=np.float32).ravel()
    n = r.size
    lines: List[str] = []
    for i in range(0, n, per_line):
        parts = []
        for j in range(i, min(i + per_line, n)):
            parts.append(f"{j:02d}:{float(r[j]):4.1f}m")
        lines.append("  ".join(parts))
    return lines

def format_raw_lidar_lines(raw_ranges: Optional[np.ndarray], raw_angles: Optional[np.ndarray] = None, *, per_line: int = 9) -> List[str]:
    """Format the raw policy LiDAR swath as 225 beam distances."""
    if raw_ranges is None:
        return []
    r = np.asarray(raw_ranges, dtype=np.float32).ravel()
    if raw_angles is None:
        a = np.arange(r.size, dtype=np.float32)
    else:
        a = np.asarray(raw_angles, dtype=np.float32).ravel()[: r.size]
    n = r.size
    lines: List[str] = []
    for i in range(0, n, per_line):
        parts = []
        for j in range(i, min(i + per_line, n)):
            parts.append(f"{j:03d}:{float(r[j]):4.1f}")
        lines.append("  ".join(parts))
    return lines

def format_range_lines(raw_ranges: Optional[np.ndarray], *, per_line: int = 9, precision: int = 1) -> List[str]:
    """Compatibility wrapper: format raw LiDAR swath distances."""
    return format_raw_lidar_lines(raw_ranges, raw_angles=None, per_line=per_line)

def pool_raw_lidar_to_sectors(raw_ranges: Sequence[float], *, n_sectors: int = LIDAR_SECTORS, max_range: float = LIDAR_RANGE):
    """Simple min-pooling helper for viewers when no policy pooling is available."""
    raw = np.asarray(raw_ranges, dtype=np.float32).ravel()
    if raw.size == 0:
        ranges = np.ones(int(n_sectors), dtype=np.float32) * float(max_range)
    else:
        sectors = np.array_split(np.clip(raw, 0.0, float(max_range)), int(n_sectors))
        ranges = np.array([float(np.min(s)) if len(s) else float(max_range) for s in sectors], dtype=np.float32)
    closeness = np.clip(1.0 - ranges / float(max_range), 0.0, 1.0).astype(np.float32)
    angles = np.linspace(-float(LIDAR_SWATH) / 2.0, float(LIDAR_SWATH) / 2.0, int(n_sectors), dtype=np.float32)
    return ranges, closeness, angles

def draw_policy_info_overlay(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: Iterable[str],
    *,
    pos: Tuple[int, int] = (8, 8),
) -> None:
    """Draw small translucent text overlay inside a map panel."""
    x, y = int(pos[0]), int(pos[1])
    for line in lines:
        txt = font.render(str(line), True, (235, 235, 245))
        bg = pygame.Surface((txt.get_width() + 6, txt.get_height() + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 135))
        surface.blit(bg, (x - 3, y - 1))
        surface.blit(txt, (x, y))
        y += txt.get_height() + 3

def draw_policy_map(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    view_center_world: Point,
    px_per_m: float,
    map_width: Optional[float] = None,
    map_height: Optional[float] = None,
    reference_path: Optional[Sequence[Sequence[float]]] = None,
    obstacles: Optional[Sequence[Sequence[Point]]] = None,
    start_xy: Optional[Point] = None,
    goal_xy: Optional[Point] = None,
    trajectory: Optional[Sequence[Point]] = None,
    current_xy: Optional[Point] = None,
    yaw_deg: Optional[float] = None,
    heading_deg: Optional[float] = None,
    raw_lidar_ranges: Optional[Sequence[float]] = None,
    raw_lidar_angles: Optional[Sequence[float]] = None,
    sector_ranges: Optional[Sequence[float]] = None,
    sector_closeness: Optional[Sequence[float]] = None,
    sector_angles: Optional[Sequence[float]] = None,
    lidar_max: float = LIDAR_RANGE,
    show_raw_lidar: bool = True,
    show_sector_lidar: bool = True,
    show_icon: bool = True,
    show_hull: bool = True,
    status_lines: Optional[Iterable[str]] = None,
    font: Optional[pygame.font.Font] = None,
    lidar_offset_m: float = LIDAR_OFFSET_M,
) -> None:
    """Compatibility wrapper used by both live and offline viewers."""
    hdg = heading_deg if heading_deg is not None else yaw_deg
    draw_policy_scene(
        surface,
        map_rect,
        map_width=map_width,
        map_height=map_height,
        reference_path=reference_path,
        obstacles=obstacles,
        start_xy=start_xy,
        goal_xy=goal_xy,
        trajectory=trajectory,
        current_xy=current_xy,
        heading_deg=hdg,
        raw_lidar_ranges=raw_lidar_ranges,
        raw_lidar_angles=raw_lidar_angles,
        sector_ranges=sector_ranges,
        sector_closeness=sector_closeness,
        sector_angles=sector_angles,
        view_center_world=view_center_world,
        px_per_m=px_per_m,
        show_raw_lidar=show_raw_lidar,
        show_sector_lidar=show_sector_lidar,
        show_icon=show_icon,
        show_hull=show_hull,
        lidar_offset_m=lidar_offset_m,
    )
    # No right-panel text overlay by default; the live/log viewers show
    # policy diagnostics in their left information panel.
