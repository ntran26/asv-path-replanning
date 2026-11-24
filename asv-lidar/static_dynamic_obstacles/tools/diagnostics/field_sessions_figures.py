"""Route figures for planning/FIELD_SESSIONS_PLAN.md: every calibration manoeuvre and trial case drawn
in the basin as planned.

Calibration routes are the current simulator plant (`src/ship.py`, v3 parameters) driven through
the same phases as `field_deployment/udp_manoeuvre.py` (run-up, manoeuvre, the basin fence), in the
10 m x 25 m basin of the simulator map (the width assumed for safety), the fence 1.5 m inside the
walls. Where the February 2026 radio-control logs measured the same manoeuvre, the measurement is
drawn beside the model. Trial routes are the deployed policy in simulation with both ships from
rest and the release offset of `field_suite_both_from_rest.py` (its trajectories.json).

Usage (from static_dynamic_obstacles/):
    python tools/diagnostics/field_sessions_figures.py [--only cal|trials]
Output: planning/field_sessions_figures/*.png
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bluefin")]
import ship                                          # noqa: E402

OUT = ROOT / "planning" / "field_sessions_figures"
TRAJ = ROOT / "results" / "field_suite" / "both_from_rest" / "trajectories.json"
W, H, MARGIN = 10.0, 25.0, 1.5
DT = 0.05
FULL_ASTERN = -24.0

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
OWN, TARGET, MEASURED = "#2a78d6", "#eb6834", "#1baf7a"
PANEL = "#c9c8c2"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "legend.frameon": False, "legend.fontsize": 8})


# ---------------------------------------------------------------------------
# Basin drawing
# ---------------------------------------------------------------------------
def basin(ax, title=None, fence=True, ylim=(0, H)):
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, lw=1.6, ec=INK2))
    if fence:
        ax.add_patch(Rectangle((MARGIN, MARGIN), W - 2 * MARGIN, H - 2 * MARGIN, fill=False, lw=0.9,
                               ec=MUTED, ls=(0, (4, 3))))
    ax.set_xlim(-0.6, W + 0.6)
    ax.set_ylim(ylim[0] - 0.6, ylim[1] + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 26, 5))
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left")


def hull(ax, x, y, hdg, color, alpha=1.0, filled=False):
    """The 1.73 x 0.50 m hull outline at (x, y), heading in degrees clockwise from +y."""
    L, B = 1.725, 0.50
    a = math.radians(hdg)
    fx, fy, sx, sy = math.sin(a), math.cos(a), math.cos(a), -math.sin(a)
    pts = [(L / 2, 0.0), (L / 2 - 0.3, B / 2), (-L / 2, B / 2), (-L / 2, -B / 2), (L / 2 - 0.3, -B / 2)]
    poly = [(x + p * fx + q * sx, y + p * fy + q * sy) for p, q in pts]
    ax.add_patch(Polygon(poly, closed=True, fc=color if filled else "none", ec=color, lw=1.0, alpha=alpha))


def mark(ax, x, y, text, color=INK2, dx=0.25, dy=0.25, ha="left"):
    ax.plot([x], [y], marker="o", ms=4, color=color)
    ax.annotate(text, (x, y), xytext=(x + dx, y + dy), fontsize=7.5, color=INK2, ha=ha)


# ---------------------------------------------------------------------------
# Plant runs, phased as udp_manoeuvre.py
# ---------------------------------------------------------------------------
def model_at(x, y, hdg, efficiency=0.5):
    m = ship.ShipModel(reverse_efficiency=efficiency)
    m._s[3, 0] = math.radians(hdg)
    m._s[5, 0], m._s[6, 0] = x, y
    return m


def inside(x, y):
    return MARGIN <= x <= W - MARGIN and MARGIN <= y <= H - MARGIN


def run(m, controller, t_max=200.0, fence=True, coast_after_fence=True):
    """controller(t, state, memo) -> (rudder, rpm, phase) or None to finish. Returns samples and events."""
    t, memo, out, events = 0.0, {"psi": 0.0, "prev": None}, [], []
    tripped = None
    while t < t_max:
        s = m.state_dict()
        if memo["prev"] is not None:
            memo["psi"] += (s["heading_deg"] - memo["prev"] + 540) % 360 - 180
        memo["prev"] = s["heading_deg"]
        if tripped is None and fence and not inside(s["x_m"], s["y_m"]):
            tripped = t
            events.append(("fence", t, s["x_m"], s["y_m"]))
        if tripped is not None:
            if not coast_after_fence or s["speed_mps"] < 0.1 or t - tripped > 40:
                break
            cmd = (0.0, 0.0, "after fence")
        else:
            cmd = controller(t, s, memo)
            if cmd is None:
                break
        out.append((t, s["x_m"], s["y_m"], s["heading_deg"], s["speed_mps"], cmd[0], cmd[1], cmd[2], memo["psi"]))
        m.update(cmd[1], cmd[0], DT)
        t += DT
    return out, events


def turn_controller(rpm, rudder, runup_s, heading_change):
    def c(t, s, memo):
        if t < runup_s:
            memo["psi0"] = memo["psi"]
            return (0.0, rpm, "run-up")
        if abs(memo["psi"] - memo["psi0"]) >= heading_change:
            return None
        return (rudder, rpm, "turn")
    return c


def zigzag_controller(rpm, delta_deg, psi_deg, runup_s, switches=4, after_s=6.0):
    helm = 100.0 * delta_deg / 40.0

    def c(t, s, memo):
        if t < runup_s:
            memo.update(psi0=memo["psi"], rud=helm, n=0, t_end=None)
            return (0.0, rpm, "run-up")
        psi = memo["psi"] - memo["psi0"]           # positive rudder turns heading up (starboard)
        if memo["n"] < switches:
            if (memo["rud"] > 0 and psi >= psi_deg) or (memo["rud"] < 0 and psi <= -psi_deg):
                memo["rud"] = -memo["rud"]
                memo["n"] += 1
                if memo["n"] == switches:
                    memo["t_end"] = t + after_s
        elif t >= memo["t_end"]:
            return None
        return (memo["rud"], rpm, "zig-zag")
    return c


def stop_controller(rpm, runup_s, turn_s=0.0, rudder=0.0, onset_s=0.0, rest_s=6.0):
    def c(t, s, memo):
        if t < runup_s:
            return (0.0, rpm, "run-up")
        if t < runup_s + turn_s:
            return (rudder, rpm, "turn")
        t_stop = runup_s + turn_s
        if "t_rest" not in memo and t > t_stop + 0.5 and s["u_body_mps"] <= 1e-3:
            memo["t_rest"] = t
        if "t_rest" in memo:
            return None if t > memo["t_rest"] + rest_s else (0.0, 0.0, "at rest")
        return (0.0, 0.0 if t < t_stop + onset_s else FULL_ASTERN, "astern")
    return c


def coast_controller(rpm, runup_s, coast_s):
    def c(t, s, memo):
        if t < runup_s:
            return (0.0, rpm, "run-up")
        if t < runup_s + coast_s:
            return (0.0, 0.0, "coast")
        return None
    return c


def draw_track(ax, samples, color, lw=1.6, ls="-", label=None, phases=None, alpha=1.0):
    xs = [p[1] for p in samples if phases is None or p[7] in phases]
    ys = [p[2] for p in samples if phases is None or p[7] in phases]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, label=label, alpha=alpha, solid_capstyle="round")


def draw_after_fence(ax, samples, events, color):
    after = [p for p in samples if p[7] == "after fence"]
    if after:
        ax.plot([p[1] for p in after], [p[2] for p in after], color=color, lw=1.2, ls=(0, (1, 1.5)))
    for kind, t, x, y in events:
        if kind == "fence":
            ax.plot([x], [y], marker="X", ms=7, color=INK, mec=SURFACE)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
# Calibration figures
# ---------------------------------------------------------------------------
def fig_rest_positions():
    fig, ax = plt.subplots(figsize=(3.6, 7.2))
    basin(ax, "D0 and S1-A: rest positions")
    for (x, y, label) in ((5.0, 12.5, "P1 mid-basin"), (2.0, 12.5, "P2 near one side wall"),
                          (8.0, 12.5, "P3 near the other side wall")):
        hull(ax, x, y, 0.0, OWN, filled=False)
    mark(ax, 5.0, 12.5, "P1 mid-basin", OWN, dx=0.4, dy=1.2)
    mark(ax, 2.0, 12.5, "P2", OWN, dx=-0.3, dy=1.2, ha="right")
    mark(ax, 8.0, 12.5, "P3", OWN, dx=0.3, dy=1.2)
    ax.text(0.2, 1.0, "5 min at rest at each,\nheading noted, then\none sharp yaw push", fontsize=7.5, color=INK2)
    ax.text(0.2, 22.6, "walls: light and matte\nblack sides identified\non the day (T-7)", fontsize=7.5, color=INK2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, "cal_01_rest_positions.png")


def fig_surge():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 7.0), gridspec_kw={"width_ratios": [1, 1.5]})
    basin(ax, "S1-C surge runs")
    runs = []
    for i, (rpm, coast) in enumerate(((6, False), (9, True), (12, True), (15, False), (18, False), (21, True), (24, False))):
        # run straight from y = 2; cut so that the coast stays inside the fence where it can
        m = model_at(2.0 + i * 1.0, 2.0, 0.0)
        u_ss = 0.0931 * rpm
        cut = {6: 20.0, 9: 9.0, 12: 7.0, 15: 5.0, 18: 4.5, 21: 4.0, 24: 3.5}[rpm]
        smp, ev = run(m, coast_controller(rpm, cut, 40.0), fence=True)
        runs.append((rpm, coast, smp, ev, u_ss))
        draw_track(ax, smp, OWN, phases=("run-up",), lw=1.4)
        draw_track(ax, smp, OWN, phases=("coast",), lw=1.4, ls=(0, (3, 2)))
        draw_after_fence(ax, smp, ev, OWN)
        ax.annotate(f"{rpm}", (2.0 + i * 1.0, 1.0), ha="center", fontsize=7, color=INK2)
    ax.text(0.2, 0.25, "RPM per lane (each run on the centre line; drawn apart)", fontsize=6.0, color=INK2)
    ax.plot([], [], color=OWN, lw=1.4, label="propulsion on")
    ax.plot([], [], color=OWN, lw=1.4, ls=(0, (3, 2)), label="propulsion off (coast)")
    ax.plot([], [], color=OWN, lw=1.2, ls=(0, (1, 1.5)), label="after the fence trips")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99))
    for rpm, coast, smp, ev, u_ss in runs:
        ts = [p[0] for p in smp]
        ax2.plot(ts, [p[4] for p in smp], color=OWN, lw=1.2, alpha=0.35 + 0.08 * runs.index((rpm, coast, smp, ev, u_ss)))
        ax2.annotate(f"RPM {rpm}", (ts[min(len(ts) - 1, int(3.0 / DT))], smp[min(len(smp) - 1, int(3.0 / DT))][4]),
                     fontsize=7, color=INK2, xytext=(3, 2), textcoords="offset points")
    ax2.set_xlabel("time from rest (s)")
    ax2.set_ylabel("speed (m/s)")
    ax2.set_title("speed against time (model)", loc="left")
    ax2.grid(color=GRID, lw=0.5)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.text(0.98, 0.02, "Each run is cut early enough to\nstop before the far wall; above\nRPM 12 the coast does not fit,\nso those runs end in a crash\nstop (S1-C2) or radio control.",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    save(fig, "cal_02_surge_coast.png")


def fig_circles():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 7.4))
    settings = ((6, 100, 2.5, 0.0, "RPM 6, full rudder to starboard"), (12, 100, 2.5, 0.0, "RPM 12, full rudder to starboard"),
                (12, -100, 7.5, 0.0, "RPM 12, full rudder to port"))
    for ax, (rpm, rud, lane, _, title) in zip(axes, settings):
        basin(ax, title)
        runup = 14.0 if rpm == 6 else 7.0
        m = model_at(lane, 2.0, 0.0)
        smp, ev = run(m, turn_controller(rpm, rud, runup, 720.0), fence=True)
        draw_track(ax, smp, OWN, phases=("run-up",), lw=1.4)
        draw_track(ax, smp, OWN, phases=("turn",), lw=1.8, label="model (v3): radius 5.3 m")
        draw_after_fence(ax, smp, ev, OWN)
        helm = next(p for p in smp if p[7] == "turn")
        side = 1.0 if rud > 0 else -1.0
        for r, ls, lab in ((2.4, "-", "February, 0.85 m/s: radius 2.4 m"), (3.9, (0, (4, 2)), "February, 1.34 m/s: radius 3.9 m")):
            ax.add_patch(Circle((helm[1] + side * r, helm[2]), r, fill=False, ec=MEASURED, lw=1.6, ls=ls))
            ax.plot([], [], color=MEASURED, lw=1.6, ls=ls, label=lab)
        mark(ax, lane, 2.0, "start", INK2, dx=0.3, dy=-0.9)
        mark(ax, helm[1], helm[2], "helm over", INK2, dx=-0.3 if side > 0 else 0.3, dy=0.3, ha="right" if side > 0 else "left")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("S1-D turning circles: entered from a run-up along the side opposite the turn; "
                 "X = fence trips (the model's circle would leave the 10 m width)", fontsize=9, color=INK2, x=0.02, ha="left")
    save(fig, "cal_03_turning_circles.png")


def fig_arcs():
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 7.2))
    for ax, (rpm, helm) in zip(axes, ((12, 25), (12, 50), (12, 75), (6, 50))):
        basin(ax, f"RPM {rpm}, helm {helm} %")
        m = model_at(2.5, 2.0, 0.0)
        smp, ev = run(m, turn_controller(rpm, helm, 14.0 if rpm == 6 else 7.0, 720.0), fence=True)
        draw_track(ax, smp, OWN, phases=("run-up",), lw=1.4)
        draw_track(ax, smp, OWN, phases=("turn",), lw=1.8)
        draw_after_fence(ax, smp, ev, OWN)
        turned = max(abs(p[8]) for p in smp if p[7] == "turn")
        ax.text(0.3, 0.6, f"{turned:.0f} deg turned before the fence", fontsize=7.5, color=INK2)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle("S1-D partial-helm settings: settled arcs, stopped by the fence (model; real turns are expected tighter)",
                 fontsize=9, color=INK2, x=0.02, ha="left")
    save(fig, "cal_04_partial_helm_arcs.png")


def fig_zigzags():
    fig = plt.figure(figsize=(10.5, 7.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.2], hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    basin(ax, "S1-E zig-zags (model)")
    specs = ((12, 20, 20, 3.0, "A: 20/20, RPM 12"), (12, 10, 10, 5.0, "B: 10/10, RPM 12"), (6, 20, 20, 7.0, "C: 20/20, RPM 6"))
    series = []
    for i, (rpm, d, p, lane, lab) in enumerate(specs):
        m = model_at(lane, 2.0, 0.0)
        smp, ev = run(m, zigzag_controller(rpm, d, p, 14.0 if rpm == 6 else 7.0, switches=2), fence=True)
        series.append((lab, smp))
        draw_track(ax, smp, OWN, phases=("run-up",), lw=1.2, alpha=0.6)
        draw_track(ax, smp, OWN, phases=("zig-zag",), lw=1.8)
        draw_after_fence(ax, smp, ev, OWN)
        ax.annotate(lab[0], (lane, 0.7), ha="center", fontsize=8, color=INK2, fontweight="bold")
    for i, (lab, smp) in enumerate(series):
        a = fig.add_subplot(gs[i, 1])
        zz = [q for q in smp if q[7] == "zig-zag"]
        t0 = zz[0][0]
        a.plot([q[0] - t0 for q in zz], [q[8] - zz[0][8] for q in zz], color=OWN, lw=1.6, label="heading change (deg)")
        a.step([q[0] - t0 for q in zz], [q[5] * 0.4 for q in zz], where="post", color=INK2, lw=1.0, ls=(0, (3, 2)),
               label="rudder (deg)")
        a.axhline(0, color=GRID, lw=0.8)
        a.set_title(lab, loc="left")
        a.grid(color=GRID, lw=0.5)
        for s in a.spines.values():
            s.set_visible(False)
        if i == 0:
            a.legend(loc="upper right", ncol=2)
        if i == 2:
            a.set_xlabel("time from the first helm order (s)")
    fig.suptitle("Started at one end, two switches (four do not fit in 25 m); the rudder switches on the measured "
                 "heading.\nOvershoots (model): A 4.8 and 5.2 deg; B 2.1 and 2.2; C 2.4 and 2.9", fontsize=9, color=INK2,
                 x=0.02, ha="left")
    save(fig, "cal_05_zigzags.png")


def fig_stops():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 7.4))
    ax = axes[0]
    basin(ax, "S1-C2 crash stops, straight")
    for lane, rpm in ((3.0, 9), (5.0, 12), (7.0, 15)):
        for eff, onset, ls, lab in ((0.5, 0.0, "-", "strong reverse (0.5, no delay)"), (0.25, 0.75, (0, (3, 2)), "weak reverse (0.25, 0.75 s)")):
            m = model_at(lane, 2.0, 0.0, efficiency=eff)
            runup = {9: 9.0, 12: 7.0, 15: 5.6}[rpm]
            smp, ev = run(m, stop_controller(rpm, runup, onset_s=onset), fence=True, coast_after_fence=False)
            st = next(p for p in smp if p[7] == "astern")
            rest = [p for p in smp if p[7] in ("astern", "at rest")]
            dx = -0.18 if eff == 0.5 else 0.18
            ax.plot([lane + dx, lane + dx], [st[2], rest[-1][2]], color=OWN if eff == 0.5 else TARGET, lw=3.0,
                    solid_capstyle="butt")
        draw_track(ax, smp, MUTED, phases=("run-up",), lw=1.0)
        ax.annotate(f"RPM {rpm}", (lane, 1.0), ha="center", fontsize=7, color=INK2)
    ax.plot([], [], color=OWN, lw=3, label="stopping distance, strong reverse")
    ax.plot([], [], color=TARGET, lw=3, label="stopping distance, weak reverse")
    ax.plot([], [], color=MUTED, lw=1, label="run-up")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.2, 21.0, "February full reverse:\n2.6-3.1 m from 1.2-1.35 m/s\n(the weak end)", fontsize=7.5, color=INK2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax = axes[1]
    basin(ax, "S1-C2 stop from a full-helm turn")
    m = model_at(2.5, 2.0, 0.0, efficiency=0.5)
    smp, ev = run(m, stop_controller(12, 7.0, turn_s=8.0, rudder=100.0, rest_s=40.0), fence=True, coast_after_fence=False)
    draw_track(ax, smp, MUTED, phases=("run-up",), lw=1.0)
    draw_track(ax, smp, OWN, phases=("turn",), lw=1.6, label="turn, helm 100 %, RPM 12")
    draw_track(ax, smp, INK, phases=("astern", "at rest"), lw=2.4, label="full astern to rest")
    st = next(p for p in smp if p[7] == "astern")
    end = smp[-1]
    hull(ax, st[1], st[2], st[3], OWN)
    hull(ax, end[1], end[2], end[3], INK)
    spin = abs(end[8] - st[8])
    ax.text(0.3, 2.6, f"model: heading turns a further\n{spin:.0f} deg after the stop (its yaw\ndamping vanishes at rest);\nFebruary: 0.6-2.9 deg/s left", fontsize=7.5, color=INK2)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("x (m)")
    save(fig, "cal_06_crash_stops.png")


def fig_steps():
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 4.6), sharex=False)
    t, rud = [0.0], [0.0]
    hold = 4.0
    for amp in (25, 50, 100):
        for target in (amp, 0, -amp, 0):
            t.append(t[-1] + hold)
            rud.append(target)
    axes[0].step(t, rud, where="pre", color=OWN, lw=1.6)
    axes[0].set_title("S1-B rudder steps, one repeat of each amplitude (5 repeats in the session); vessel held at the side", loc="left")
    axes[0].set_ylabel("rudder (%)")
    t2, r2 = [0.0, 3.0], [0.0, 0.0]
    for a, b in ((0, 100), (100, -100), (-100, 0)):
        t2.append(t2[-1] + abs(b - a) / 10.0)
        r2.append(b)
        t2.append(t2[-1] + 3.0)
        r2.append(b)
    axes[1].plot(t2, r2, color=OWN, lw=1.6)
    axes[1].set_title("S1-B ramp at 10 %/s", loc="left")
    axes[1].set_ylabel("rudder (%)")
    axes[1].set_xlabel("time (s)")
    for a in axes:
        a.grid(color=GRID, lw=0.5)
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "cal_07_rudder_steps.png")


def fig_target_check():
    fig, ax = plt.subplots(figsize=(3.6, 7.2))
    basin(ax, "S2-D target platform check")
    hull(ax, 7.5, 12.5, 0.0, OWN, filled=True, alpha=0.8)
    mark(ax, 7.5, 12.5, "own ship at rest", OWN, dx=-0.2, dy=1.3, ha="right")
    for x, lab in ((5.5, "2 m"), (3.5, "4 m"), (1.8, "5.7 m")):
        ax.add_patch(FancyArrowPatch((x, 22.5), (x, 2.5), arrowstyle="-|>", mutation_scale=9, color=TARGET, lw=1.4))
        ax.annotate(lab, (x, 23.0), ha="center", fontsize=7, color=INK2)
    ax.text(0.2, 0.4, "target passes at each offset at\n4-5 throttle settings, 2 runs each", fontsize=7, color=INK2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, "cal_09_target_check.png")


def fig_rc_backup():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 7.4))
    ax = axes[0]
    basin(ax, "RC-1 straight run and coast")
    m = model_at(5.0, 2.0, 0.0)
    smp, ev = run(m, coast_controller(9, 9.0, 30.0), fence=True)
    draw_track(ax, smp, OWN, phases=("run-up",), lw=1.6, label="steady throttle (stick held)")
    draw_track(ax, smp, OWN, phases=("coast",), lw=1.6, ls=(0, (3, 2)), label="throttle to zero")
    draw_after_fence(ax, smp, ev, OWN)
    mark(ax, 5.0, 2.0, "start", INK2, dx=0.3, dy=-0.9)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.2, 17.5, "3 runs at each of two\nthrottle settings; then the\nsame with full reverse\nstick (RC-4) once the\nESC reverse is confirmed", fontsize=7, color=INK2)
    for ax, side, lane in ((axes[1], 1.0, 2.5), (axes[2], -1.0, 7.5)):
        basin(ax, "RC-2 full-rudder circle, " + ("starboard" if side > 0 else "port"))
        ax.plot([lane, lane], [2.0, 9.0], color=OWN, lw=1.6, label="run-up at a moderate throttle")
        for r, ls, lab in ((2.4, "-", "expected (February): radius 2.4 m at 0.85 m/s"),):
            ax.add_patch(Circle((lane + side * r, 9.0), r, fill=False, ec=MEASURED, lw=1.8, ls=ls))
            ax.plot([], [], color=MEASURED, lw=1.8, label=lab)
        mark(ax, lane, 2.0, "start", INK2, dx=0.3, dy=-0.9)
        mark(ax, lane, 9.0, "stick to\nfull rudder", INK2, dx=-0.3 * side, dy=-1.6, ha="right" if side > 0 else "left")
        ax.legend(loc="upper left", fontsize=7)
        ax.text(0.2, 15.0, "two full turns, stick held;\nthen centre and stop;\n2 runs each side;\nthe throttle not raised\nabove February's 0.85 m/s\nsetting (the 1.34 m/s circle\nspanned 8.6 m)", fontsize=7, color=INK2)
    fig.suptitle("Back-up by radio control (S1a without udp_manoeuvre.py): RC-1 to RC-3; RC-3 is the rest positions "
                 "of cal_01. No zig-zag by hand.", fontsize=9, color=INK2, x=0.02, ha="left")
    save(fig, "rc_01_backup_routes.png")


# ---------------------------------------------------------------------------
# Trial figures and the dress rehearsal (simulated with the deployed policy)
# ---------------------------------------------------------------------------
def case_geometry(name):
    import field_test_cases as ftc
    import curriculum
    import train_formulation as tf
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    built, seed = ftc.TestCase().scenario(name)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0, vessel_randomisation=None)
    env.reset(seed=seed, options={"generated": copy.deepcopy(built), "initial_speed": 0.0})
    geo = {"obstacles": [list(map(tuple, p)) for p in env.obstacles], "path": [tuple(p) for p in env.path.points]}
    env.close()
    return geo


def draw_case(ax, name, geo, traj, sheet):
    basin(ax, None, fence=False)
    for poly in geo["obstacles"]:
        ax.add_patch(Polygon(poly, closed=True, fc=PANEL, ec=INK2, lw=0.8))
    (lx0, ly0), (lx1, ly1) = sheet["leg_start"], sheet["leg_goal"]
    ax.plot([lx0, lx1], [ly0, ly1], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.plot([lx1], [ly1], marker="*", ms=9, color=INK2)
    rest = traj.get("rest")
    cruise = traj.get("cruise")
    if cruise:
        c = cruise["trajectory"]
        ax.plot([p[1] for p in c], [p[2] for p in c], color=OWN, lw=0.9, alpha=0.35)
    if rest:
        tr = rest["trajectory"]
        ox, oy = [p[1] for p in tr], [p[2] for p in tr]
        tx, ty = [p[4] for p in tr if p[4] is not None], [p[5] for p in tr if p[5] is not None]
        ax.plot(ox, oy, color=OWN, lw=1.8)
        ax.plot(tx, ty, color=TARGET, lw=1.8)
        k = min(range(len(tr)), key=lambda i: math.hypot(tr[i][4] - tr[i][1], tr[i][5] - tr[i][2]) if tr[i][4] is not None else 1e9)
        hull(ax, tr[k][1], tr[k][2], tr[k][3], OWN, filled=True, alpha=0.7)
        d = math.hypot(tr[k][4] - tr[k][1], tr[k][5] - tr[k][2])
        ax.plot([tr[k][1], tr[k][4]], [tr[k][2], tr[k][5]], color=INK, lw=0.8, ls=(0, (1, 1.5)))
        ax.annotate(f"{d:.1f} m", ((tr[k][1] + tr[k][4]) / 2, (tr[k][2] + tr[k][5]) / 2), fontsize=7, color=INK,
                    xytext=(3, 3), textcoords="offset points")
    ox0, oy0 = sheet["own_at_t0"]
    tx0, ty0 = sheet["target_start"]
    hull(ax, ox0, oy0, sheet["own_heading_deg"], OWN)
    hull(ax, tx0, ty0, sheet["target_heading_deg"], TARGET)
    off = rest["offset_s"] if rest else 0.0
    rel = (f"target {off:.0f} s after own ship" if off > 0 else
           f"target {-off:.0f} s before own ship" if off < 0 else "released together")
    short = {"head-on": "head-on", "crossing, target to starboard (give way)": "crossing, give way",
             "crossing, target to port (stand on)": "crossing, stand on", "overtaking": "overtaking",
             "being overtaken (stand on)": "being overtaken"}.get(sheet["encounter"], sheet["encounter"])
    outcome = rest["outcome"] if rest else ""
    ax.set_title(f"{name}: {short}", loc="left", fontsize=9, pad=14)
    ax.text(0.0, 1.005, rel + f", {sheet['target_speed_mps']:.2f} m/s"
            + ("" if outcome == "goal" else f"; this seed: {outcome.replace('collision:', 'contact, ')}"),
            fontsize=7, color=INK2, transform=ax.transAxes, va="bottom")
    ax.set_xticks([0, 5, 10])


def fig_trials():
    import field_test_cases as ftc
    if not TRAJ.exists():
        print("trajectories.json missing: run field_suite_both_from_rest.py first")
        return
    trajs = json.loads(TRAJ.read_text())
    case = ftc.TestCase()
    for mode, fname, title in (("easy", "trial_T1_easy.png", "T1: target only, open basin"),
                               ("medium", "trial_T2_medium.png", "T2: target and layout L1"),
                               ("hard", "trial_T3_hard.png", "T3: target and layouts L3 (1-3) and L2 (4-5)")):
        names = case.names(mode)
        fig, axes = plt.subplots(1, 5, figsize=(13.5, 6.6))
        for ax, n in zip(axes, names):
            draw_case(ax, n, case_geometry(n), trajs.get(n, {}), case.field_sheet(n))
        axes[0].plot([], [], color=OWN, lw=1.8, label="own ship, both from rest (simulated)")
        axes[0].plot([], [], color=OWN, lw=0.9, alpha=0.35, label="own ship as verified (both moving at t = 0)")
        axes[0].plot([], [], color=TARGET, lw=1.8, label="target, from rest")
        axes[0].plot([], [], color=MUTED, lw=0.9, ls=(0, (4, 3)), label="leg; star = goal")
        fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(top=0.86, bottom=0.1, wspace=0.25)
        fig.suptitle(title + ": start marks (outlines), simulated routes with the deployed policy, closest approach",
                     fontsize=9.5, color=INK2, x=0.02, ha="left")
        save(fig, fname)


def fig_dress_rehearsal():
    import field_test_cases as ftc
    import curriculum
    import numpy as np
    import train_formulation as tf
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    algo = json.loads((ftc.DEFAULT_MODEL.parent / "config.json").read_text()).get("algo", "sac")
    model = tf.ALGORITHMS[algo].load(str(ftc.DEFAULT_MODEL), device="cpu")
    case = ftc.TestCase()
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 7.0))
    for ax, (n, lab) in zip(axes, (("easy_1", "open leg"), ("medium_1", "layout L1"), ("hard_4", "layout L2"),
                                   ("hard_1", "layout L3"))):
        built, seed = case.scenario(n)
        built = copy.deepcopy(built)
        env = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0, vessel_randomisation=None)
        obs, _ = env.reset(seed=seed, options={"generated": built, "initial_speed": 0.0})
        env.targets.clear()
        actor = tf.EpisodeActor(model) if hasattr(tf, "EpisodeActor") else None
        pts, outcome = [(env.asv_x, env.asv_y)], "timeout"
        for _ in range(400):
            a = actor(obs) if actor is not None else model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(np.asarray(a, dtype=np.float32).reshape(2))
            pts.append((env.asv_x, env.asv_y))
            if term or trunc:
                outcome = "goal" if info.get("reached_goal") else info.get("collision_kind", "end")
                break
        geo = {"obstacles": [list(map(tuple, p)) for p in env.obstacles]}
        env.close()
        basin(ax, f"{lab} ({outcome})", fence=False)
        for poly in geo["obstacles"]:
            ax.add_patch(Polygon(poly, closed=True, fc=PANEL, ec=INK2, lw=0.8))
        sheet = case.field_sheet(n)
        (lx0, ly0), (lx1, ly1) = sheet["leg_start"], sheet["leg_goal"]
        ax.plot([lx0, lx1], [ly0, ly1], color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.plot([lx1], [ly1], marker="*", ms=9, color=INK2)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=OWN, lw=1.8)
        hull(ax, lx0, ly0, sheet["own_heading_deg"], OWN)
        ax.set_xticks([0, 5, 10])
    fig.suptitle("S2-C dress rehearsal: the deployed policy from rest, no target (simulated); 2 runs per layout on the water",
                 fontsize=9.5, color=INK2, x=0.02, ha="left")
    save(fig, "cal_08_dress_rehearsal.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("cal", "trials"), default=None)
    args = ap.parse_args()
    if args.only in (None, "cal"):
        fig_rest_positions()
        fig_surge()
        fig_circles()
        fig_arcs()
        fig_zigzags()
        fig_stops()
        fig_steps()
        fig_target_check()
        fig_rc_backup()
    if args.only in (None, "trials"):
        fig_dress_rehearsal()
        fig_trials()


if __name__ == "__main__":
    main()
