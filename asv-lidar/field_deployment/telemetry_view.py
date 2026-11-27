"""
Live display for the lab tools without the policy (udp_raw_logger.py, udp_manoeuvre.py).

A pygame window beside the terminal: a status panel (the tool's own state, its last console
messages, a NO TELEMETRY banner) and a map in the pose frame (the track, the hull and its heading,
the latest LiDAR returns and, for the manoeuvre sender, the fence and its margin).  It only reads
the lines the tool has already written to disk; it never decides anything and never sends.

The display is optional by design.  The tools import this module inside a guard, so a tool copied
on its own, a computer without pygame or without a screen, or any error while drawing leaves the
recording untouched: the tool prints one warning and goes on without the window.  Drawing is
rate-limited (--render-fps, 10 by default), so it adds at most a few milliseconds to a loop that
already waits on the socket.  Closing the window closes the display only; the recording goes on
until Ctrl+C in the terminal.

Conventions (log_parser.BluefinStreamDecoder): a pose line `[ts]A,B,C` gives x = B, y = -A and
heading -C deg, clockwise from +y; a LiDAR line holds 720 ranges in decimetres (0: no return),
reversed so that beam i lies i x 0.5 deg clockwise from the bow; a return at range r on bearing
a is drawn at (x + r sin a, y + r cos a), from the pose point (the LiDAR offset is not applied,
O-9).
"""

import collections
import math
import re
import time

RE_POSE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]"
    r"(?P<y>[-+]\d+\.\d+),(?P<x>[-+]\d+\.\d+),(?P<yaw>[-+]\d+\.\d+)\s*$")
RE_LIDAR = re.compile(r"^\[(?P<ts>\d{2}:\d{2}:\d{2}\.\d{6})\]\[(?P<body>[\d,\s]*)\]\s*$")
LIDAR_UNIT_M = 0.1
LIDAR_BEAMS = 720
LOA, BEAM = 1.725, 0.50
TRAIL_MAX = 6000

BG, PANEL, INK, DIM = (18, 20, 26), (28, 31, 40), (232, 234, 240), (150, 156, 170)
GRID, GRID_TXT = (40, 44, 55), (110, 116, 130)
TRAIL, HULL, SCAN = (90, 160, 255), (255, 255, 255), (235, 220, 110)
FENCE, MARGIN, WARN, ALERT = (220, 90, 90), (150, 80, 80), (255, 200, 60), (255, 90, 90)


class TelemetryView:
    """The window.  Create with `TelemetryView.open(...)`; returns None (with a reason) if it cannot."""

    def __init__(self, pg, screen, title, fps, fence, fence_margin_m):
        self.pg, self.screen, self.title = pg, screen, title
        self.period = 1.0 / max(1.0, float(fps))
        self.fence = fence                    # (xmin, xmax, ymin, ymax) in the pose frame, or None
        self.margin = float(fence_margin_m or 0.0)
        self.font = pg.font.SysFont("dejavusansmono,liberationmono,monospace", 15)
        self.small = pg.font.SysFont("dejavusansmono,liberationmono,monospace", 12)
        self.big = pg.font.SysFont("dejavusans,liberationsans,sans", 19, bold=True)
        self.trail = collections.deque(maxlen=TRAIL_MAX)
        self.pose = None                      # (x, y, heading_deg)
        self.scan = []                        # [(bearing_deg_from_bow, range_m)]
        self.messages = collections.deque(maxlen=8)
        self.t_next = 0.0
        self.t_last_line = None
        self.bounds = None                    # expand-only view box (xmin, xmax, ymin, ymax)
        self.closed = False
        self.frames = 0

    @classmethod
    def open(cls, title, fps=10.0, fence=None, fence_margin_m=0.0, size=(1180, 720)):
        try:
            import os
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.display.init()
            pygame.font.init()
            screen = pygame.display.set_mode(size)
            pygame.display.set_caption(title)
            return cls(pygame, screen, title, fps, fence, fence_margin_m), None
        except Exception as exc:              # no pygame, no screen (SSH, no DISPLAY), any other failure
            return None, "%s: %s" % (type(exc).__name__, exc)

    # -- input --------------------------------------------------------------------------------
    def note(self, text):
        for line in str(text).splitlines():          # tables and continuation lines stay in the terminal
            if line.startswith("["):
                self.messages.append(line.strip()[:110])

    def feed(self, line, now=None):
        s = line.strip()
        if not s:
            return
        if s.startswith("["):
            self.t_last_line = time.monotonic() if now is None else now
        m = RE_POSE.match(s)
        if m:
            x, y, h = float(m.group("x")), -float(m.group("y")), -float(m.group("yaw"))
            self.pose = (x, y, h)
            if not self.trail or (x - self.trail[-1][0]) ** 2 + (y - self.trail[-1][1]) ** 2 > 1e-4:
                self.trail.append((x, y))
            return
        m = RE_LIDAR.match(s)
        if m:
            vals = [int(v) for v in m.group("body").replace(" ", "").split(",") if v]
            if len(vals) == LIDAR_BEAMS:
                vals.reverse()
                step = 360.0 / LIDAR_BEAMS
                self.scan = [(i * step, v * LIDAR_UNIT_M) for i, v in enumerate(vals) if v > 0]

    # -- drawing ------------------------------------------------------------------------------
    def tick(self, status_lines, banner=None, force=False):
        """Draw if a frame is due.  `status_lines` is a list, or a function returning one."""
        if self.closed:
            return
        now = time.monotonic()
        if not force and now < self.t_next:
            return
        self.t_next = now + self.period
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.close()
                return
        lines = status_lines() if callable(status_lines) else status_lines
        if banner is None and self.t_last_line is not None and now - self.t_last_line > 1.5:
            banner = ("NO TELEMETRY FOR %.1f s" % (now - self.t_last_line), WARN)
        self._draw(lines, banner)
        self.frames += 1

    def _view(self, rect):
        pts = list(self.trail)
        if self.pose is not None:
            x, y, _ = self.pose
            pts += [(x - 5.0, y - 5.0), (x + 5.0, y + 5.0)]
        if self.fence is not None:
            pts += [(self.fence[0], self.fence[2]), (self.fence[1], self.fence[3])]
        if not pts:
            pts = [(-5.0, -5.0), (5.0, 5.0)]
        b = (min(p[0] for p in pts) - 1.0, max(p[0] for p in pts) + 1.0,
             min(p[1] for p in pts) - 1.0, max(p[1] for p in pts) + 1.0)
        if self.bounds is not None:                      # expand only, so the map does not jitter
            b = (min(b[0], self.bounds[0]), max(b[1], self.bounds[1]),
                 min(b[2], self.bounds[2]), max(b[3], self.bounds[3]))
        self.bounds = b
        scale = min(rect.width / (b[1] - b[0]), rect.height / (b[3] - b[2]))
        cx, cy = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0

        def w2s(p):
            return (int(rect.centerx + (p[0] - cx) * scale), int(rect.centery - (p[1] - cy) * scale))
        return w2s, scale, b

    def _draw(self, lines, banner):
        pg, scr = self.pg, self.screen
        W, H = scr.get_size()
        scr.fill(BG)
        panel = pg.Rect(0, 0, 450, H)
        pg.draw.rect(scr, PANEL, panel)
        y = 12
        scr.blit(self.big.render(self.title, True, INK), (14, y))
        y += 34
        if banner is not None:
            scr.blit(self.big.render(banner[0], True, banner[1]), (14, y))
            y += 30
        for s in lines:
            colour = INK
            if isinstance(s, tuple):
                s, colour = s
            scr.blit(self.font.render(str(s)[:52], True, colour), (14, y))
            y += 21
        y = max(y + 10, H - 22 - 18 * len(self.messages))
        if self.messages:
            scr.blit(self.small.render("console:", True, DIM), (14, y - 18))
        for s in self.messages:
            scr.blit(self.small.render(s[:68], True, DIM), (14, y))
            y += 18

        rect = pg.Rect(panel.right + 20, 20, W - panel.right - 40, H - 50)
        w2s, scale, b = self._view(rect)
        step = 1.0 if (b[1] - b[0]) < 30 else 5.0
        gx = math.floor(b[0] / step) * step
        while gx <= b[1]:
            p0, p1 = w2s((gx, b[2])), w2s((gx, b[3]))
            pg.draw.line(scr, GRID, p0, p1, 1)
            if abs(gx / 5.0 - round(gx / 5.0)) < 1e-6:
                scr.blit(self.small.render("%g" % gx, True, GRID_TXT), (p0[0] - 6, rect.bottom + 4))
            gx += step
        gy = math.floor(b[2] / step) * step
        while gy <= b[3]:
            p0, p1 = w2s((b[0], gy)), w2s((b[1], gy))
            pg.draw.line(scr, GRID, p0, p1, 1)
            if abs(gy / 5.0 - round(gy / 5.0)) < 1e-6:
                scr.blit(self.small.render("%g" % gy, True, GRID_TXT), (rect.left - 30, p0[1] - 7))
            gy += step
        scr.blit(self.small.render("pose frame, m (x right, y up)", True, GRID_TXT), (rect.left, rect.bottom + 20))

        if self.fence is not None:
            x0, x1, y0, y1 = self.fence
            pg.draw.rect(scr, FENCE, pg.Rect(w2s((x0, y1)), (int((x1 - x0) * scale), int((y1 - y0) * scale))), 2)
            m = self.margin
            if m > 0 and x1 - x0 > 2 * m and y1 - y0 > 2 * m:
                pg.draw.rect(scr, MARGIN, pg.Rect(w2s((x0 + m, y1 - m)),
                                                  (int((x1 - x0 - 2 * m) * scale), int((y1 - y0 - 2 * m) * scale))), 1)
        if len(self.trail) > 1:
            pg.draw.lines(scr, TRAIL, False, [w2s(p) for p in self.trail], 2)
        if self.pose is not None:
            x, y, h = self.pose
            for bearing, r in self.scan:
                a = math.radians(h + bearing)
                q = w2s((x + r * math.sin(a), y + r * math.cos(a)))
                if rect.collidepoint(q):
                    scr.set_at(q, SCAN)
            hr = math.radians(h)
            fwd, side = (math.sin(hr), math.cos(hr)), (math.cos(hr), -math.sin(hr))
            hull = []
            for a_, s_ in ((-0.5 * LOA, -0.5 * BEAM), (0.32 * LOA, -0.5 * BEAM), (0.5 * LOA, 0.0),
                           (0.32 * LOA, 0.5 * BEAM), (-0.5 * LOA, 0.5 * BEAM)):
                hull.append(w2s((x + a_ * fwd[0] + s_ * side[0], y + a_ * fwd[1] + s_ * side[1])))
            pg.draw.polygon(scr, HULL, hull, 2)
            pg.draw.circle(scr, HULL, w2s((x, y)), 3)
        else:
            scr.blit(self.font.render("waiting for a pose line", True, DIM), (rect.left + 10, rect.top + 10))
        pg.display.flip()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.pg.display.quit()
            except Exception:
                pass


class SafeView:
    """Wraps a TelemetryView so that no drawing error can reach the tool: the first error closes the
    window with one message; feed/tick/note/close are then no-ops."""

    def __init__(self, view, say):
        self.view, self.say = view, say

    def _guard(self, fn, *a, **k):
        if self.view is None:
            return
        try:
            fn(*a, **k)
            if self.view is not None and self.view.closed:
                self.view = None
                self.say("[VIEW] window closed; recording continues (Ctrl+C in the terminal stops)")
        except Exception as exc:
            view, self.view = self.view, None          # off first, so the message below cannot re-enter
            try:
                view.close()
            except Exception:
                pass
            self.say("[VIEW] display stopped after an error, recording continues: %s: %s" % (type(exc).__name__, exc))

    def feed(self, line, now=None):
        self._guard(lambda: self.view.feed(line, now))

    def note(self, text):
        self._guard(lambda: self.view.note(text))

    def tick(self, status_lines, banner=None, force=False):
        self._guard(lambda: self.view.tick(status_lines, banner, force))

    def close(self):
        if self.view is not None:
            try:
                self.view.close()
            except Exception:
                pass
            self.view = None

    @property
    def frames(self):
        return 0 if self.view is None else self.view.frames


def open_view(title, say, fps=10.0, fence=None, fence_margin_m=0.0):
    """A SafeView, or None with the reason printed once."""
    view, why = TelemetryView.open(title, fps=fps, fence=fence, fence_margin_m=fence_margin_m)
    if view is None:
        say("[VIEW] no display (%s); recording continues without the window" % why)
        return None
    say("[VIEW] display open (closing the window does not stop the recording)")
    return SafeView(view, say)
