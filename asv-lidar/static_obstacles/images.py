"""Minimal boat icon used by rl_env.py rendering.

The training environment only needs this file when render_mode='human'.  The
icon is generated from a small RGBA array so the package is self-contained.
"""
from __future__ import annotations
import numpy as np

_w, _h = 32, 64
_img = np.zeros((_h, _w, 4), dtype=np.uint8)
# hull body
_img[8:58, 10:22, :] = [210, 210, 225, 255]
# bow point
for y in range(0, 12):
    left = max(0, 16 - y // 2)
    right = min(_w, 16 + y // 2 + 1)
    _img[y, left:right, :] = [230, 230, 245, 255]
# center line
_img[8:58, 15:17, :] = [70, 70, 90, 255]
# border
_img[8:58, 9:10, :] = [40, 40, 55, 255]
_img[8:58, 22:23, :] = [40, 40, 55, 255]
_img[57:59, 10:22, :] = [40, 40, 55, 255]

BOAT_ICON = {
    "bytes": _img.tobytes(),
    "size": (_w, _h),
    "format": "RGBA",
}
