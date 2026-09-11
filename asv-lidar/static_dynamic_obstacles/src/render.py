"""Pygame view of the environment: the field on the right, telemetry on the left.

Only imported when `render_mode="human"`, so headless training never touches
pygame or OpenCV.

The panel
---------
Per `RENDER_PANEL_SPEC`.  Seven blocks, toggled with the number keys, defaulting
to `[4]` COLREGS and `[5]` REWARD -- the two that are the reward's instrument
while it is being built, and the two without which the scale audit is the only
feedback and it runs after the fact.

**The panel is a view on `info`, never a separate computation.**  Every field it
draws is a key `env.step()` emitted, assembled once in `env._panel_view()`.  If
the panel needs a number `info` does not carry, the number goes into `info` --
same principle as the single `EncounterContext`, and the same reason: two
consumers, one source, or they diverge.

The one idea it is built around is **show the gate, not just the value**.  When
`v_side` reads `0.000` the number alone cannot tell you whether that is correct
(wrong class for this term) or a bug (gate stuck, `rho_t` collapsed, class never
latched).  Every COLREGs sub-term prints its value *and* the reason it holds it,
which costs one string per term and is the difference between a panel you glance
at and a panel you debug with.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np
import pygame
import pygame.freetype

import constants as cfg
from constants import LIDAR_RANGE
from ship import VESSEL_LENGTH, VESSEL_WIDTH

VIDEO_PATH = "asv_lidar.mp4"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
MAGENTA = (200, 0, 200)
CYAN = (0, 220, 220)
AMBER = (255, 180, 0)
DARK_RED = (100, 0, 0)
ORANGE = (255, 120, 0)      # target ships
STEEL = (70, 130, 200)      # boundary rays
BEAM_CLEAR = (55, 55, 95)
BEAM_HIT = (90, 90, 160)

# Panel palette.  Near-monochrome, like the field, with **four** colour rules and
# no others (RENDER_PANEL_SPEC §5): red for a class mismatch, red for a clip rate
# above the threshold, amber for an engaged encounter, amber for a COLREGs
# sub-term above 0.5.  Colour that means something is worth more than colour that
# looks organised.
PANEL_BG = (12, 12, 16)
PANEL_FG = (198, 202, 210)
PANEL_DIM = (110, 114, 124)
PANEL_HEAD = (120, 170, 220)
PANEL_OFF = (64, 66, 74)

PANEL_WIDTH = 430
PANEL_MIN_HEIGHT = 800      # all seven blocks, expanded, plus the footer
PANEL_PAD = 8
LINE_H = 12
FONT_SIZE = 10.5

CLIP_WARN = 0.10            # a branch above this is losing gradient
TERM_WARN = 0.50            # a COLREGs sub-term above this is doing real work

BLOCKS = ("run", "ego", "perception", "colregs", "reward", "obs_health", "clearance")
DEFAULT_BLOCKS = frozenset({"colregs", "reward"})
HISTORY = 200               # steps held for the scrub keys


def boat_icon() -> pygame.Surface:
    """A small top-down hull sprite, drawn from an RGBA array."""
    w, h = 32, 64
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[8:58, 10:22] = (210, 210, 225, 255)      # hull
    for y in range(12):                          # bow taper
        img[y, max(0, 16 - y // 2):min(w, 16 + y // 2 + 1)] = (230, 230, 245, 255)
    img[8:58, 15:17] = (70, 70, 90, 255)         # centre line
    img[8:58, 9:10] = (40, 40, 55, 255)          # outline
    img[8:58, 22:23] = (40, 40, 55, 255)
    img[57:59, 10:22] = (40, 40, 55, 255)
    return pygame.image.frombytes(img.tobytes(), (w, h), "RGBA")


class Renderer:
    def __init__(self, map_width: float, map_height: float, *,
                 record_video: bool = False, panel: bool = True) -> None:
        pygame.init()
        self.map_height = float(map_height)
        self.scale = float(cfg.RENDER_SCALE)
        self.field_size = (int(round(map_width * self.scale)),
                           int(round(map_height * self.scale)))

        self.panel_width = PANEL_WIDTH if panel else 0
        # Tall enough for every block at once.  The default two fit in far
        # less, but a panel that silently truncates when you press [7] is worse
        # than a tall window: the block you just asked for is the one that
        # disappears off the bottom.
        self.window_size = (self.panel_width + self.field_size[0],
                            max(self.field_size[1], PANEL_MIN_HEIGHT))

        self.surface = pygame.Surface(self.window_size)
        self.font = pygame.freetype.SysFont(
            "consolas,dejavusansmono,couriernew,monospace", size=FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.display = None

        self.icon = pygame.transform.smoothscale(boat_icon(), (
            max(1, int(round(VESSEL_WIDTH * self.scale))),
            max(1, int(round(VESSEL_LENGTH * self.scale))),
        ))

        self.record_video = bool(record_video)
        self.video_writer = None

        # Which blocks are shown, and the scrub position.  `scrub = 0` is live;
        # a positive value steps back through the ring buffer.
        self.blocks = set(DEFAULT_BLOCKS)
        self.history: deque = deque(maxlen=HISTORY)
        self.scrub = 0

        # Extra lines drawn at the foot of the panel, for whatever the caller
        # wants to show: key bindings in manual play, scenario labels in 04's
        # evaluation runs.
        self.overlay: list = []

    # ------------------------------------------------------------------
    def toggle(self, name: str) -> None:
        self.blocks.symmetric_difference_update({name})

    def toggle_index(self, index: int) -> None:
        """Keys 1-7, matching the block numbers in `RENDER_PANEL_SPEC` §2."""
        if 1 <= index <= len(BLOCKS):
            self.toggle(BLOCKS[index - 1])

    def scrub_by(self, steps: int) -> None:
        """Step the panel back and forward through the ring buffer.

        Nearly every question worth asking about a COLREGs encounter is "what
        was the state four seconds ago", and re-running with a breakpoint to
        find out is the slow way.
        """
        self.scrub = int(np.clip(self.scrub + int(steps), 0, max(0, len(self.history) - 1)))

    def to_screen(self, xy):
        """World metres -> screen pixels, with y flipped and the panel offset."""
        px = self.panel_width + int(round(float(xy[0]) * self.scale))
        py = int(round((self.map_height - float(xy[1])) * self.scale))
        return px, max(0, min(self.field_size[1] - 1, py))

    # ------------------------------------------------------------------
    def draw(self, env) -> None:
        if self.display is None:
            self.display = pygame.display.set_mode(self.window_size)

        # Without this the OS never gets its events back and the window is
        # reported as unresponsive.  Callers that read the keyboard pump the
        # queue themselves; this covers everyone who does not.
        pygame.event.pump()

        panel = getattr(env, "last_panel", None)
        if panel is not None and (not self.history or self.history[-1] is not panel):
            self.history.append(panel)

        self.surface.fill(BLACK)
        pygame.draw.rect(self.surface, RED,
                         pygame.Rect(self.panel_width, 0,
                                     self.field_size[0] - 1, self.field_size[1] - 1),
                         width=2)

        for obs in env.obstacles:
            pygame.draw.polygon(self.surface, RED, [self.to_screen(p) for p in obs])

        # Target ships, and the tracker's estimate of each one.
        for target in getattr(env, "targets", ()):
            pygame.draw.polygon(self.surface, ORANGE,
                                [self.to_screen(p) for p in target.hull()])
        for track in getattr(env, "tracks", ()):
            centre = self.to_screen(tuple(track.position))
            pygame.draw.circle(self.surface, WHITE, centre, 4, width=1)
            tip = track.position + track.velocity * 3.0
            pygame.draw.line(self.surface, WHITE, centre, self.to_screen(tuple(tip)), 1)

        self._draw_lidar(env.lidar)
        self._draw_boundary(env)

        path_px = [self.to_screen(p) for p in env.path.points]
        if len(path_px) >= 2:
            pygame.draw.lines(self.surface, GREEN, False, path_px, 2)

        pygame.draw.circle(self.surface, DARK_RED, self.to_screen((env.tgt_x, env.tgt_y)), 3)
        pygame.draw.circle(self.surface, CYAN, self.to_screen((env.lookahead_x, env.lookahead_y)), 3)
        pygame.draw.circle(self.surface, MAGENTA, self.to_screen((env.goal_x, env.goal_y)), 6)

        rotated = pygame.transform.rotozoom(self.icon, -env.asv_h, 1)
        self.surface.blit(rotated, rotated.get_rect(center=self.to_screen((env.asv_x, env.asv_y))))
        pygame.draw.polygon(self.surface, (255, 0, 0),
                            [self.to_screen(p) for p in env.hull_polygon()], width=2)

        if self.panel_width:
            self._draw_panel()

        self.display.blit(self.surface, (0, 0))
        pygame.display.update()
        self.clock.tick(cfg.RENDER_FPS)

        if self.record_video:
            self._write_frame()

    # ------------------------------------------------------------------
    # Field
    # ------------------------------------------------------------------
    def _draw_lidar(self, lidar) -> None:
        origin = self.to_screen(lidar.pos)

        # Pooled sectors: what the policy actually sees.  Blue = clear, red = close.
        for angle, dist, close in zip(lidar.sector_angles, lidar.sector_ranges, lidar.sector_closeness):
            colour = (int(80 + 175 * close), int(180 * (1.0 - close)), int(220 * (1.0 - close)))
            end = self._beam_end(lidar, angle, dist)
            pygame.draw.line(self.surface, colour, origin, end, 2)
            pygame.draw.circle(self.surface, colour, end, 2)

        # Every fourth raw beam, dim, to show the underlying geometry.
        for angle, dist in zip(lidar.bearings[::4], lidar.ranges[::4]):
            colour = BEAM_CLEAR if dist >= LIDAR_RANGE - 1e-6 else BEAM_HIT
            pygame.draw.line(self.surface, colour, origin, self._beam_end(lidar, angle, dist), 1)

    def _beam_end(self, lidar, angle_deg, dist):
        bearing = np.radians(lidar.heading + float(angle_deg))
        return self.to_screen((lidar.pos[0] + float(dist) * np.sin(bearing),
                               lidar.pos[1] + float(dist) * np.cos(bearing)))

    def _draw_boundary(self, env) -> None:
        """The 7 virtual boundary rays -- from the map, not from the sensor."""
        import boundary_raycast as br

        ranges = br.boundary_ranges(env.asv_x, env.asv_y, env.asv_h, env.boundary_polygon)
        origin = self.to_screen((env.asv_x, env.asv_y))
        for bearing_deg, dist in zip(cfg.BOUNDARY_BEARINGS_DEG, ranges):
            a = np.radians(env.asv_h + float(bearing_deg))
            end = (env.asv_x + float(dist) * np.sin(a), env.asv_y + float(dist) * np.cos(a))
            pygame.draw.line(self.surface, STEEL, origin, self.to_screen(end), 1)

    # ------------------------------------------------------------------
    # Panel
    # ------------------------------------------------------------------
    def _draw_panel(self) -> None:
        pygame.draw.rect(self.surface, PANEL_BG,
                         pygame.Rect(0, 0, self.panel_width, self.window_size[1]))
        self._y = PANEL_PAD

        if not self.history:
            self._line("no telemetry yet", PANEL_DIM)
            return

        index = max(0, len(self.history) - 1 - self.scrub)
        panel = self.history[index]

        if self.scrub:
            self._line(f"-- SCRUBBED {self.scrub} steps back "
                       f"({index + 1}/{len(self.history)}) --", AMBER)

        drawers = {
            "run": self._block_run,
            "ego": self._block_ego,
            "perception": self._block_perception,
            "colregs": self._block_colregs,
            "reward": self._block_reward,
            "obs_health": self._block_obs_health,
            "clearance": self._block_clearance,
        }
        for i, name in enumerate(BLOCKS, start=1):
            if name in self.blocks:
                drawers[name](panel, i)

        self._footer()

    def _line(self, text: str, colour=PANEL_FG) -> None:
        if self._y > self.window_size[1] - LINE_H:
            return
        surface, _ = self.font.render(str(text), colour, PANEL_BG)
        self.surface.blit(surface, (PANEL_PAD, self._y))
        self._y += LINE_H

    def _rule(self, title: str, index: int) -> None:
        width = 46 - len(title)
        self._line(f"--- {title} " + "-" * max(1, width) + f"  [{index}]", PANEL_HEAD)

    def _block_run(self, panel, index) -> None:
        """[1] RUN -- provenance.

        `spawn=COMPLIANT|DISPLACED` is the head-on regime from 02a §2.4.  If
        every frame ever seen says `DISPLACED`, `TARGET_COMPLIANT_SPAWN_PROB` is
        not wired, and `v_r8`'s zero branch -- the whole point of the 02 §3.2
        head-on rationale -- is never being exercised.
        """
        run = panel["run"]
        seed = "none" if run["seed"] is None else run["seed"]
        self._rule("RUN", index)
        self._line(f"seed={seed}   scen={run['scenario']}   "
                   f"step={run['step']}/{run['max_steps']}  t={run['t']:.1f}s")
        water = "OPEN WATER" if run["open_water"] else (
            f"W={run['corridor_w']:.1f}m ({run['corridor_b']:.0f} B)"
            f"  W_local={run['w_local']:.2f}")
        self._line(f"{water}  targets={run['targets']}  spawn={run['spawn']}")

    def _block_ego(self, panel, index) -> None:
        """[2] EGO -- the speed gate and `r_path`.

        `g_u` with a `[SAT]` flag catches a dead gate, which is the shape of the
        F19 bug that pinned the `ego` surge feature at 1.0 for 45% of a run.
        `Dact` against the rate limit shows whether `kappa_delta` is calibrated:
        if it never approaches the limit, `r_smooth` is inert.
        """
        ego = panel["ego"]
        self._rule("EGO (truth)", index)
        self._line(f"x={ego['x']:+.2f} y={ego['y']:+.2f}  hdg={ego['hdg']:+06.1f}   "
                   f"u={ego['u']:+.2f} v={ego['v']:+.2f} r={ego['r_dps']:+.1f} d/s")

        sat = " [SAT]" if ego["g_u_sat"] else ""
        rule = f"  <{ego['u_ref_rule']}>" if ego["u_ref_rule"] else ""
        self._line(f"U_ref={ego['u_ref']:.2f}  U_ref_eff={ego['u_ref_eff']:.2f}   "
                   f"g_u={ego['g_u']:.2f}{sat}{rule}",
                   AMBER if (ego["g_u_sat"] or ego["u_ref_rule"]) else PANEL_FG)
        if ego["u_ref_rule"]:
            self._line(f"  {ego['u_ref_reason']}", AMBER)

        turn = "STBD" if ego["r_err"] > 0 else ("PORT" if ego["r_err"] < 0 else "----")
        self._line(f"r_path={ego['r_path']:+.3f} rad/s   "
                   f"r-r_path={ego['r_err']:+.3f} ({turn})")
        self._line(f"act rud={ego['rudder']:+.2f} thr={ego['throttle']:+.2f} | "
                   f"cmd rudder={ego['rudder_deg']:+.1f} rpm={ego['rpm']:.2f}")
        frac = abs(ego["d_rudder"]) / max(ego["kappa_delta"], 1e-9)
        self._line(f"Dact rud={ego['d_rudder']:+.3f} ({frac:.0%} of the rate limit "
                   f"{ego['kappa_delta']:.3f})  sigma={ego['sigma']:.2f}",
                   PANEL_DIM if frac < 0.05 else PANEL_FG)

    def _block_perception(self, panel, index) -> None:
        """[3] PERCEPTION vs TRUTH -- the N1 block.

        Under `R-1` the safety terms read truth and the COLREGs gating reads the
        estimate.  The panel must show both or that decision is invisible.  The
        class row goes red on a mismatch, because a misclassification is the
        failure 04 §6 names as the one that matters -- the agent turns the wrong
        way -- and it is otherwise almost impossible to spot in a replay.
        """
        block = panel.get("perception")
        self._rule("PERCEPTION vs TRUTH", index)
        if block is None:
            self._line("no target tracked", PANEL_DIM)
            return

        est, true = block["est"], block["true"]
        self._line("                  est      true       err", PANEL_DIM)
        # `angle` rows are differenced on the circle.  A bearing estimate of
        # -7.2 against a truth of 330.7 is a 22 degree error, not a 338 degree
        # one, and the naive subtraction made the tracker look broken on the
        # first frame it was ever looked at.
        rows = (("tgt range", "range", "{:9.3f}", False),
                ("tgt bearing", "bearing", "{:9.1f}", True),
                ("tgt speed", "speed", "{:9.3f}", False),
                ("tgt heading", "heading", "{:9.1f}", True),
                ("DCPA", "dcpa", "{:9.2f}", False),
                ("TCPA", "tcpa", "{:9.1f}", False))
        for label, key, fmt, angular in rows:
            e = est.get(key, float("nan"))
            t = true.get(key)
            if t is None:
                self._line(f" {label:<12s}{fmt.format(e)}        --        --")
            else:
                err = _wrap180(e - t) if angular else e - t
                self._line(f" {label:<12s}{fmt.format(e)}{fmt.format(t)}"
                           f"{fmt.format(err)}")

        mismatch = true.get("cls") is not None and est["cls"] != true["cls"]
        self._line(f" {'class':<12s}{est['cls']:>8s}{true.get('cls', '--'):>9s}"
                   f"{'MISMATCH' if mismatch else 'ok':>10s}",
                   RED if mismatch else PANEL_FG)
        track = block["track"]
        self._line(f" track: age={track['age']} hits={track['hits']} "
                   f"misses={track['misses']} coast={track['coast']:.1f}s  "
                   f"gate: {block['dropped']} dropped")
        if block["mismatched_steps"]:
            self._line(f" misclassified on {block['mismatched_steps']} steps "
                       f"this episode", RED)

    def _block_colregs(self, panel, index) -> None:
        """[4] COLREGS -- the state machine and the admissibility predicate.

        The three numbers behind `A_stbd` (`Dy_req`, `r_stbd`, `r_port`) explain
        *why* it flipped, which is what is actually wanted when the agent does
        something odd near a wall.  `compliant_sense` printed explicitly is the
        guard against the 02 §4.2 trap: `sense=PORT` on an overtaking encounter,
        next to a starboard turn and `v_port` rising, is the whole bug in one
        frame.
        """
        block = panel.get("colregs")
        self._rule("COLREGS", index)
        if block is None:
            self._line("no target tracked", PANEL_DIM)
            return

        engaged = block["state"] == "engaged"
        since = (f"  since t={block['engaged_at']}  (+{block['engaged_for']:.1f}s)"
                 if block["engaged_at"] >= 0 else "")
        self._line(f"class={block['cls'].upper()}(est)  state={block['state'].upper()}{since}",
                   AMBER if engaged else PANEL_FG)

        known = "" if block["known"] else "  [no map: permissive]"
        self._line(f"compliant_sense={block['sense']}    "
                   f"A_stbd={'YES' if block['a_stbd'] else 'NO '}    "
                   f"A_port={'YES' if block['a_port'] else 'NO '}{known}",
                   PANEL_FG if block["known"] else PANEL_DIM)
        self._line(f"Dy_req={block['dy_req']:.2f}  r_stbd={_fin(block['r_stbd'])}  "
                   f"r_port={_fin(block['r_port'])}   d_req={block['d_req']:.2f}")
        extremis = "  IN EXTREMIS" if block["in_extremis"] else ""
        self._line(f"rho={block['rho']:.2f}   A_req={block['a_req']:.2f}  "
                   f"A_t={block['a_t']:.2f}  urgency={block['urgency']:.2f}{extremis}",
                   AMBER if block["in_extremis"] else PANEL_FG)

        for name in ("port", "bow", "side", "hold", "r8"):
            value = block["terms"].get(name, 0.0)
            why = block["why"].get(name, "")
            colour = AMBER if value > TERM_WARN else (
                PANEL_FG if value > 0.0 else PANEL_DIM)
            self._line(f" v_{name:<6s}{value:6.3f}  [{why}]", colour)
        self._line(f" group  {block['group']:6.3f}  "
                   f"(pre-clip {block['pre_clip']:.3f} / 1.0)",
                   AMBER if block["group"] > TERM_WARN else PANEL_FG)

    def _block_reward(self, panel, index) -> None:
        """[5] REWARD -- four columns, and the fourth is the important one.

        `inst` (pre-weight) / `xw` (post-weight) / `Sep` (episode integral) /
        **`range(ep)`**.  The range column is the direct detector for the Paper 2
        scale bug: a term whose episode range is `[-0.44, -0.41]` varies by less
        than 10% of its own value -- a constant offset wearing a shaping term's
        costume, and invisible in the other three columns.
        """
        block = panel["reward"]
        self._rule("REWARD", index)
        self._line("              inst      xw      Sep   range(ep)", PANEL_DIM)
        for row in block["rows"]:
            span = "[flat]" if row["flat"] else f"[{row['lo']:+.2f},{row['hi']:+.2f}]"
            colour = AMBER if row["flat"] else PANEL_FG
            self._line(f" {row['name']:<9s}{row['inst']:+8.3f}{row['xw']:+8.3f}"
                       f"{row['sum']:+9.1f}   {span}", colour)
        self._line(" " + "-" * 50, PANEL_DIM)
        self._line(f" step total        {block['step_total']:+9.3f}      "
                   f"dominant: {block['dominant']}")
        terminal = (f"  terminal {block['terminal']:+.0f}"
                    if block["terminal"] else "")
        self._line(f" episode           {block['episode_total']:+9.1f}      "
                   f"dominant: {block['episode_dominant']}{terminal}")
        for problem in block["hierarchy"]:
            self._line(f" HIERARCHY: {problem}", AMBER)

    def _block_obs_health(self, panel, index) -> None:
        """[6] OBS HEALTH -- `clip%`, the F19 detector.

        Fraction of episode steps at the normaliser's clip, per branch.  An
        `ego` branch reading 45% is exactly the bug where the surge feature was
        pinned at 1.0 and carried no gradient.
        """
        self._rule("OBS HEALTH", index)
        self._line(" branch       dim     min     max   clip%", PANEL_DIM)
        for row in panel["obs_health"]:
            hot = row["clip"] > CLIP_WARN
            flag = " [!]" if hot else ""
            self._line(f" {row['name']:<12s}{row['dim']:4d}{row['lo']:8.2f}"
                       f"{row['hi']:8.2f}{row['clip'] * 100:7.0f}%{flag}",
                       RED if hot else PANEL_FG)
            if hot and row.get("worst"):
                self._line(f"   worst: {row['worst']} at "
                           f"{row['worst_clip'] * 100:.0f}%", RED)

    def _block_clearance(self, panel, index) -> None:
        """[7] CLEARANCE -- how close to which terminal.

        `domain margin` is signed, so negative means intruding.  Four numbers
        that say which termination is nearest, which is the context wanted when
        a run ends abruptly.
        """
        c = panel["clearance"]
        self._rule("CLEARANCE", index)
        self._line(f" boundary {c['boundary']:.2f}   static obs {_fin(c['obstacle'])}"
                   f"   target hull {_fin(c.get('target', float('inf')))}")
        margin = c.get("domain_margin")
        line = f" goal in {c['goal']:.1f} m   steps left {c['steps_left']}"
        if margin is not None and np.isfinite(margin):
            self._line(f" domain margin {margin:+.2f}" + line,
                       RED if margin < 0 else PANEL_FG)
        else:
            self._line(line)

    def _footer(self) -> None:
        self._y = max(self._y, self.window_size[1] - LINE_H * (len(self.overlay) + 2))
        shown = "".join(str(i) if BLOCKS[i - 1] in self.blocks else "-"
                        for i in range(1, len(BLOCKS) + 1))
        self._line(f"blocks {shown}   1-7 toggle   , . scrub", PANEL_OFF)
        for text in self.overlay:
            self._line(str(text), PANEL_DIM)

    # ------------------------------------------------------------------
    def _write_frame(self) -> None:
        frame = pygame.surfarray.array3d(self.surface)
        frame = cv2.cvtColor(cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1), cv2.COLOR_RGB2BGR)
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(
                VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), cfg.RENDER_FPS, self.window_size)
        self.video_writer.write(frame)

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.display is not None:
            pygame.display.quit()
            self.display = None


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _fin(value: float, fmt: str = "{:.2f}") -> str:
    """Format a distance that may legitimately be infinite."""
    return "  inf" if not np.isfinite(value) else fmt.format(value)
