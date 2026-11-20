"""Memory-bounded evaluation of the existing hull-to-point clearance formula.

This is a computational tiling of classical.common.point_clearance, not a new
safety margin or collision model. Global reach filtering and each point's
arithmetic are identical; minima are accumulated over small point blocks.
"""
from __future__ import annotations

import numpy as np
from classical import common as cc

MAX_BLOCK_ELEMENTS = 1_000_000


def point_clearance(positions, headings, points, reach=None):
    k, n = headings.shape
    if len(points) == 0:
        return np.full((k, n), np.inf)
    if reach is not None:
        centre = positions.reshape(-1, 2)
        lo, hi = centre.min(axis=0) - reach, centre.max(axis=0) + reach
        points = points[np.all((points >= lo) & (points <= hi), axis=1)]
        if len(points) == 0:
            return np.full((k, n), np.inf)
    block = max(1, MAX_BLOCK_ELEMENTS // max(1, k*n))
    s, c = np.sin(headings)[..., None], np.cos(headings)[..., None]
    result = np.full((k, n), np.inf)
    for start in range(0, len(points), block):
        rel = points[None, None, start:start+block, :] - positions[:, :, None, :]
        lon = np.abs(rel[..., 0]*s + rel[..., 1]*c) - cc.HALF_L
        lat = np.abs(rel[..., 0]*c - rel[..., 1]*s) - cc.HALF_W
        outside = np.hypot(np.maximum(lon, 0.), np.maximum(lat, 0.))
        inside = np.minimum(np.maximum(lon, lat), 0.)
        result = np.minimum(result, np.min(outside + inside, axis=-1))
    return result


def evaluate(snap, ro):
    """Exact V2 checks with only the temporary point-array allocation tiled."""
    import safety_v2 as v2
    clear = point_clearance(ro.positions, ro.headings, snap.points,
                            reach=cc.HALF_L+1.) - v2.GAP_STATIC_M
    clear = np.minimum(clear, cc.boundary_clearance(ro.positions,ro.headings,
                                                   snap.edges_a,snap.edges_b)-v2.GAP_BOUNDARY_M)
    for track in snap.tracks:
        clear = np.minimum(clear, cc.target_gap(ro.positions,ro.headings,ro.times,track)-v2.GAP_TARGET_M)
    bad = clear < 0.
    first = np.where(bad.any(axis=0),ro.times[np.argmax(bad,axis=0)],np.inf)
    minimum = clear.min(axis=0)
    moving = ro.speeds[-1] > v2.TERMINAL_STOP_SPEED
    if moving.any():
        run = np.minimum(v2.TERMINAL_M,v2.TERMINAL_S*ro.speeds[-1][moving])
        d = np.linspace(1/8,1.,8)[:,None,None]*run[None,:,None]
        heading = ro.headings[-1][moving]
        extension = ro.positions[-1][moving][None]+d*np.stack([np.sin(heading),np.cos(heading)],axis=1)[None]
        hk = np.broadcast_to(heading,extension.shape[:2])
        terminal = point_clearance(extension,hk,snap.points,reach=cc.HALF_L+1.)-v2.GAP_STATIC_M
        if v2.TERMINAL_BOUNDARY:
            terminal = np.minimum(terminal,cc.boundary_clearance(extension,hk,snap.edges_a,snap.edges_b)-v2.GAP_BOUNDARY_M)
        tm = np.full(len(minimum),np.inf)
        tm[moving] = terminal.min(axis=0)
        first = np.where(np.isinf(first)&(tm<0.),ro.times[-1],first)
        minimum = np.minimum(minimum,tm)
    return first, minimum
