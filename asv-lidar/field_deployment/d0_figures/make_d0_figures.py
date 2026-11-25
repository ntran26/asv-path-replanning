"""
Figures for D0_OPERATIONS_2026-11-26.md: where the vessel starts and what is done in each D0 test.

Frame as in static_dynamic_obstacles/planning/field_sessions_figures/: the planned basin, 10 m
across (x, from the left wall) by 25 m long (y, from the near end), seen from above with the near
end (the launch end, where the shore computer stands) at the bottom.  The real basin is wider than
10 m; the planned box sits centred across it, so x = 5 m is the real basin's centreline.

    python d0_figures/make_d0_figures.py        (from field_deployment/; writes the PNGs beside this file)
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W, L = 10.0, 25.0                 # planned basin
LOA, BEAM = 1.725, 0.50           # Bluefin hull; the pose reference point is mid-hull
MARGIN = 1.5                      # udp_manoeuvre.py fence margin
OWN, INK, INK2, ACT, PUSH, BAD = "#1f6fd1", "#222222", "#555555", "#d9480f", "#c2255c", "#868e96"
START = (5.0, 2.0)                # S: centreline, 2 m from the near end (the field-suite start)
P1 = (5.0, 12.5)                  # mid-basin (rest position P1 of the plan)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def hull(ax, x, y, heading_deg, colour=OWN, label=None, lw=1.4, fill=False, dx=0.35, dy=0.0):
    """The Bluefin outline at (x, y), heading in degrees clockwise from +y (up the basin)."""
    h = math.radians(heading_deg)
    fwd, side = (math.sin(h), math.cos(h)), (math.cos(h), -math.sin(h))
    pts = []
    for a, s in ((-0.5 * LOA, -0.5 * BEAM), (0.32 * LOA, -0.5 * BEAM), (0.5 * LOA, 0.0),
                 (0.32 * LOA, 0.5 * BEAM), (-0.5 * LOA, 0.5 * BEAM)):
        pts.append((x + a * fwd[0] + s * side[0], y + a * fwd[1] + s * side[1]))
    ax.add_patch(Polygon(pts, closed=True, fill=fill, facecolor=colour, edgecolor=colour, lw=lw, alpha=0.9))
    ax.plot([x], [y], "o", color=colour, ms=3)
    if label:
        ax.text(x + dx, y + dy, label, color=colour, fontsize=8, fontweight="bold", va="center")


def basin(ax, title, fence=True, note_walls=True):
    ax.add_patch(Rectangle((0, 0), W, L, fill=False, lw=2.0, ec=INK))
    if fence:
        ax.add_patch(Rectangle((MARGIN, MARGIN), W - 2 * MARGIN, L - 2 * MARGIN, fill=False, lw=0.8,
                               ec=INK2, ls="--"))
    ax.plot([W / 2, W / 2], [0, L], color="#adb5bd", lw=0.6, ls=":")
    if note_walls:
        ax.text(W / 2, -1.0, "NEAR END (launch, shore computer)", ha="center", va="top", fontsize=7, color=INK2)
        ax.text(W / 2, L + 0.4, "far end", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.text(-0.35, L / 2, "left wall", rotation=90, ha="right", va="center", fontsize=7, color=INK2)
        ax.text(W + 0.35, L / 2, "right wall", rotation=270, ha="left", va="center", fontsize=7, color=INK2)
    ax.set_xlim(-1.6, W + 1.6)
    ax.set_ylim(-2.6, L + 1.8)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 26, 5))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (m), across", fontsize=7)
    ax.set_ylabel("y (m), along", fontsize=7)
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")


def arrow(ax, p, q, colour=ACT, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=colour, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# 1. Overview: frame, start mark, rest mark, fence --------------------------------------------
def fig_overview():
    fig, ax = plt.subplots(figsize=(4.4, 7.6))
    basin(ax, "D0: frame, marks and fence")
    hull(ax, *START, 0, label="S  start mark\n(5, 2): centreline,\n2 m from near end", dx=0.5, dy=0.4)
    ax.text(0.2, 15.0, "The real basin is wider than\n10 m: the planned box sits\ncentred across it, so x = 5\nis the real centreline.",
            fontsize=6.8, color=INK2)
    ax.text(MARGIN + 0.15, L - MARGIN - 0.5, "fence 1.5 m inside the\nplanned 10 m x 25 m box",
            fontsize=6.8, color=INK2, va="top")
    ax.annotate("", xy=(5.0, 0.0), xytext=(5.0, START[1] - 0.86), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
    ax.text(5.25, 0.55, "stern 1.1 m\nfrom wall", fontsize=6.3, color=INK2)
    ax.text(0.2, 6.5, "Bow points up the basin\n(+y) in every rest test.\nThe dot is the pose\nreference point (mid-hull).",
            fontsize=6.8, color=INK2)
    save(fig, "d0_fig1_frame_and_marks.png")


# 2. D0-1 to D0-3 at rest, and the yaw push ------------------------------------------------------
def fig_rest_and_push():
    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_axes([0.0, 0.0, 0.42, 1.0])
    basin(ax, "D0-1, D0-2, D0-3: at rest at S")
    hull(ax, *START, 0, label="S (5, 2)", dx=0.45)
    ax.text(0.3, 9.0, "Vessel floats free at S,\nbow up the basin,\nnobody touching it.\n\n"
            "Tethers, if used, hang\nslack during recording.", fontsize=7, color=INK2)

    # The push, top view of the hull.
    bx = fig.add_axes([0.47, 0.42, 0.5, 0.55])
    bx.set_title("D0-2: where to push (top view)", fontsize=9.5, fontweight="bold", loc="left")
    hull(bx, 0.0, 0.0, 0, lw=2.0)
    bx.text(0.0, 0.62, "bow", ha="center", fontsize=7, color=INK2)
    bx.text(0.0, -1.08, "stern (near-end wall side)", ha="center", fontsize=7, color=INK2)
    arrow(bx, (-0.95, -0.72), (-0.3, -0.72), colour=PUSH, lw=2.4)
    bx.text(-1.75, -0.6, "1. Push the STERN\nsideways, sharply\n(about 0.3 s)", fontsize=7, color=PUSH)
    arrow(bx, (0.45, 1.02), (-0.45, 1.02), colour=PUSH, lw=1.5, rad=0.35)
    bx.text(0.55, 0.72, "2. the bow swings\n    the other way:\n    the hull rotates\n    (yaw)", fontsize=7, color=PUSH)
    arrow(bx, (0.95, -0.05), (0.3, -0.05), colour=BAD, lw=1.4)
    bx.plot([0.55, 0.75], [-0.17, 0.07], color=BAD, lw=1.6)
    bx.plot([0.55, 0.75], [0.07, -0.17], color=BAD, lw=1.6)
    bx.text(1.0, -0.12, "not mid-side:\nit drifts sideways\nand hardly rotates", fontsize=6.8, color=BAD, va="center")
    bx.set_xlim(-2.0, 2.2)
    bx.set_ylim(-1.3, 1.45)
    bx.set_aspect("equal")
    bx.axis("off")

    # Timeline.
    tx = fig.add_axes([0.5, 0.08, 0.46, 0.22])
    tx.set_title("D0-2 timeline (2 min)", fontsize=9.5, fontweight="bold", loc="left")
    tx.add_patch(Rectangle((0, 0.35), 60, 0.3, color="#d0ebff"))
    tx.add_patch(Rectangle((60, 0.35), 60, 0.3, color="#d0ebff"))
    tx.plot([60, 60], [0.2, 0.8], color=PUSH, lw=2.5)
    tx.text(30, 0.5, "at rest, hands off", ha="center", va="center", fontsize=7)
    tx.text(90, 0.5, "at rest, hands off", ha="center", va="center", fontsize=7)
    tx.text(60, 0.88, "push at 1:00 (note the time)", ha="center", fontsize=7, color=PUSH)
    tx.set_xlim(-2, 122)
    tx.set_ylim(0, 1.1)
    tx.set_xticks([0, 30, 60, 90, 120])
    tx.set_xticklabels(["0:00", "0:30", "1:00", "1:30", "2:00"], fontsize=7)
    tx.set_yticks([])
    for s in ("left", "right", "top"):
        tx.spines[s].set_visible(False)
    save(fig, "d0_fig2_rest_and_push.png")


# 3. D0-4 manual driving --------------------------------------------------------------------------
def fig_manual_driving():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 7.8))
    basin(ax, "D0-4 (a): straight runs")
    hull(ax, *START, 0, label="S", dx=0.45)
    arrow(ax, (5.0, 3.0), (5.0, 12.0), colour=OWN, lw=2.0)
    ax.plot([5.0, 5.0], [12.0, 21.5], color=OWN, lw=1.6, ls="--")
    ax.plot([4.6, 5.4], [12.0, 12.0], color=OWN, lw=2.0)
    ax.text(4.3, 7.0, "1  SLOW\n    throttle A", fontsize=7, color=OWN, ha="right")
    ax.text(4.3, 12.0, "cut throttle\nat y = 12", fontsize=6.6, color=OWN, ha="right", va="center")
    ax.text(4.3, 17.5, "coast\n(about 11 m\nin the model)", fontsize=6.6, color=OWN, ha="right", va="center")
    ax.text(5.0, 21.9, "turn round slowly", fontsize=6.6, color=INK2, ha="center")
    arrow(ax, (6.5, 21.0), (6.5, 16.0), colour=ACT, lw=2.0)
    ax.plot([6.5, 6.5], [16.0, 3.5], color=ACT, lw=1.6, ls="--")
    ax.plot([6.1, 6.9], [16.0, 16.0], color=ACT, lw=2.0)
    ax.text(6.8, 18.8, "2  FASTER\n    throttle B", fontsize=7, color=ACT)
    ax.text(6.8, 16.0, "cut throttle\nat y = 16", fontsize=6.6, color=ACT, va="center")
    ax.text(6.8, 9.5, "coast\n(about 13 m\nin the model)", fontsize=6.6, color=ACT, va="center")

    basin(bx, "D0-4 (b): turns")
    r = 2.4                                    # February full-stick radius at 0.85 m/s
    th = [i / 72 * 2 * math.pi for i in range(73)]
    # 3a: run-up along x = 2.6 (the side opposite the turn), full stick RIGHT at y = 7.
    arrow(bx, (2.6, 2.2), (2.6, 6.9), colour="#2b8a3e", lw=1.6)
    bx.plot([5.0 + r * math.cos(math.pi - a) for a in th], [7.0 + r * math.sin(math.pi - a) for a in th],
            color="#2b8a3e", lw=1.6)
    arrow(bx, (5.0, 9.4), (5.6, 9.32), colour="#2b8a3e", lw=1.6)
    bx.text(0.2, 10.9, "3a  run-up\nat x = 2.6,\nfull stick\nRIGHT,\none turn", fontsize=6.6, color="#2b8a3e", va="top")
    # 3b: run-up along x = 7.4, full stick LEFT at y = 14.
    arrow(bx, (7.4, 9.6), (7.4, 13.9), colour="#5f3dc4", lw=1.6)
    bx.plot([5.0 + r * math.cos(a) for a in th], [14.0 + r * math.sin(a) for a in th], color="#5f3dc4", lw=1.6)
    arrow(bx, (5.0, 16.4), (4.4, 16.32), colour="#5f3dc4", lw=1.6)
    bx.text(7.7, 9.3, "3b  run-up\nat x = 7.4,\nfull stick\nLEFT,\none turn", fontsize=6.6, color="#5f3dc4", va="top")
    # 4: gentle S-bends at half stick.
    ys = [i * 0.1 for i in range(int(4.0 / 0.1) + 1)]
    bx.plot([5.0 + 1.3 * math.sin(2 * math.pi * y / 2.0) for y in ys], [17.0 + y for y in ys], color="#e67700", lw=1.6)
    bx.plot([4.4, 5.6], [21.0, 21.0], color="#e67700", lw=2.0)
    bx.text(0.25, 20.6, "4  gentle\nS-bends,\nhalf stick,\nslow; cut\nthrottle at\ny = 21", fontsize=6.6,
            color="#e67700", va="top")
    bx.text(5.0, 0.55, "circles drawn at the February radius, 2.4 m (4.8 m across)", fontsize=6.2, color=INK2,
            ha="center")
    fig.text(0.5, 0.005, "Order: 1, 2, 3a, 3b, 4.  The safety person drives; the recorder notes each start time.  "
             "Stay inside the dashed line (the fence of S1a).", ha="center", fontsize=7, color=INK2)
    save(fig, "d0_fig3_manual_driving.png")


# 4. D0-5 basin frame -----------------------------------------------------------------------------
def fig_basin_frame():
    fig, ax = plt.subplots(figsize=(4.8, 7.8))
    basin(ax, "D0-5: basin frame, four marks", fence=False)
    for name, x, y, hdg in (("M1", 1.0, 12.5, 0), ("M2", 9.0, 12.5, 0), ("M3", 5.0, 1.0, 90), ("M4", 5.0, 24.0, 90)):
        hull(ax, x, y, hdg)
    ax.text(0.3, 14.9, "M1  1 m from\nthe LEFT wall", fontsize=6.8, color=OWN, va="center", fontweight="bold")
    ax.text(9.7, 14.9, "M2  1 m from\nthe RIGHT wall", fontsize=6.8, color=OWN, va="center", ha="right", fontweight="bold")
    ax.text(6.1, 1.0, "M3  1 m from the NEAR end,\nbow to the right", fontsize=6.8, color=OWN, va="center", fontweight="bold")
    ax.text(6.1, 24.0, "M4  1 m from the FAR end,\nbow to the right", fontsize=6.8, color=OWN, va="center", fontweight="bold")
    for (x0, y0, x1, y1) in ((0.0, 11.4, 1.0, 11.4), (10.0, 11.4, 9.0, 11.4), (3.6, 0.0, 3.6, 1.0), (3.6, 25.0, 3.6, 24.0)):
        ax.annotate("", xy=(x0, y0), xytext=(x1, y1), arrowprops=dict(arrowstyle="<->", color=ACT, lw=1.0))
    ax.text(1.0, 10.9, "1 m", fontsize=6.3, color=ACT, ha="center", va="top")
    ax.text(9.0, 10.9, "1 m", fontsize=6.3, color=ACT, ha="center", va="top")
    ax.text(0.6, 20.8, "Tape from the REAL wall to\nmid-hull (the dot) = 1 m.\n30 s still at each mark;\n"
            "the recorder notes the time.\nOrder M1, M2, M3, M4;\nmove between them under\nradio control.",
            fontsize=6.8, color=INK2, va="top")
    ax.text(0.6, 7.6, "The real basin is wider than\ndrawn: M1 and M2 are 1 m from\nthe real side walls, so their\n"
            "distance apart is the real\nwidth less 2 m.", fontsize=6.6, color=INK2, va="top")
    save(fig, "d0_fig4_basin_frame.png")


if __name__ == "__main__":
    fig_overview()
    fig_rest_and_push()
    fig_manual_driving()
    fig_basin_frame()
