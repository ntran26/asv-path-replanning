"""A picture of every development-set scenario (the 120 the PPO runs are scored on).

Writes `results/devset_gallery/`:

* `scenarios/NNN_<class>_<mode>.png` -- one figure per scenario: the basin
  envelope, the navigable polygon (basin `P_nav` or the channel), the head-on
  band where it applies, the static panels exactly as the environment realises
  them for that scenario's evaluation seed, the reference leg with start and
  goal, the own ship at spawn and where it would be at CPA on the leg at
  cruise, and the target at spawn, its constant-velocity track to CPA, and its
  hull at CPA;
* `sheet_<class>.png` -- the 20 scenarios of each class on one page;
* `index.csv` -- one row per scenario: geometry, encounter parameters, and the
  latest Tier 1 outcomes (run 8 final, run 8 best, run 7) and the reference
  controller's where they exist;
* `README.md` -- the composition, and how to read a figure.

    python tools/diagnostics/devset_gallery.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import scenario as scn  # noqa: E402
import targets as tgt  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT = ROOT / "results" / "devset_gallery"
CLASSES = ("head_on", "crossing", "overtaking", "being_overtaken", "null", "no_target")
COLOURS = {"nav": "#dfe9f5", "band": "#b9d3ee", "wall": "#555555", "panel": "#8c6d46",
           "path": "#2b6cb0", "own": "#1f7a3f", "target": "#c0392b"}
OUTCOMES = {"run8_final": "tier1_run8_supervisor_off", "run8_best": "tier1_run8best_supervisor_off",
            "run7": "tier1_run7_supervisor_off"}


def _md(df: pd.DataFrame) -> str:
    """A Markdown table without the optional `tabulate` dependency."""
    df = df.reset_index()
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return chr(10).join([head, rule] + body)


def _unit(deg):
    a = math.radians(deg)
    return np.array([math.sin(a), math.cos(a)])


def _outcomes() -> pd.DataFrame:
    frames = []
    for name, tag in OUTCOMES.items():
        path = ROOT / "results" / "tiers" / tag / "episodes.csv"
        if path.exists():
            d = pd.read_csv(path, keep_default_na=False)
            d = d[d["set"] == "development"][["scenario", "outcome"]].rename(columns={"outcome": name})
            frames.append(d.set_index("scenario"))
    ref = ROOT / "results" / "basin_devset_baselines" / "episodes.csv"
    if ref.exists():
        d = pd.read_csv(ref)
        d = d[d.policy == "reference"][["scenario", "outcome"]].rename(columns={"outcome": "reference"})
        frames.append(d.set_index("scenario"))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _draw(ax, i, built, env, outcomes, compact=False):
    w, h = cfg.MAP_WIDTH, cfg.MAP_HEIGHT
    ax.add_patch(MplPolygon([(0, 0), (w, 0), (w, h), (0, h)], closed=True, fill=False,
                            ec=COLOURS["wall"], lw=1.2))
    ax.add_patch(MplPolygon(env.boundary_polygon, closed=True, fc=COLOURS["nav"], ec="#7a9cc6", lw=0.8))
    if built.geometry_mode == "basin" and built.encounter_class == "head_on":
        band = built.channel.band().polygon()
        ax.add_patch(MplPolygon(band, closed=True, fc=COLOURS["band"], ec="none", alpha=0.7))
    for poly in env.obstacles:
        ax.add_patch(MplPolygon(poly, closed=True, fc=COLOURS["panel"], ec="k", lw=0.5))
    pts = np.asarray(env.path.points)
    ax.plot(pts[:, 0], pts[:, 1], color=COLOURS["path"], lw=1.2, ls="--")
    ax.plot(env.start_x, env.start_y, "o", color=COLOURS["own"], ms=4)
    ax.plot(env.goal_x, env.goal_y, "*", color="#d4a017", ms=10, mec="k", mew=0.4)
    ax.add_patch(MplPolygon(env.hull_polygon(), closed=True, fc=COLOURS["own"], ec="k", lw=0.4))

    tcpa = max(float(getattr(built, "tcpa_s", 0.0)), 0.0)
    if built.encounter_class != "no_target":
        t0 = np.array(built.target_spawn, dtype=float)
        v = float(built.target_speed) * _unit(float(built.target_heading))
        horizon = tcpa if built.encounter_class != "null" else 12.0
        t_cpa = t0 + v * horizon
        ax.add_patch(MplPolygon(tgt.hull_polygon(float(t0[0]), float(t0[1]), float(built.target_heading)),
                                closed=True, fc=COLOURS["target"], ec="k", lw=0.4))
        ax.annotate("", xy=t_cpa, xytext=t0, arrowprops=dict(arrowstyle="->", color=COLOURS["target"],
                                                             lw=1.0, ls=":"))
        ax.add_patch(MplPolygon(tgt.hull_polygon(float(t_cpa[0]), float(t_cpa[1]), float(built.target_heading)),
                                closed=True, fill=False, ec=COLOURS["target"], lw=0.8, ls="--"))
        if built.encounter_class != "null":
            s = min(cfg.U_NOM * tcpa, float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
            seg = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
            own_cpa = np.array([np.interp(s, seg, pts[:, 0]), np.interp(s, seg, pts[:, 1])])
            ax.plot(*own_cpa, "x", color=COLOURS["own"], ms=6, mew=1.5)

    ax.set_xlim(-0.5, w + 0.5)
    ax.set_ylim(-0.5, h + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([] if compact else range(0, 11, 2))
    ax.set_yticks([] if compact else range(0, 26, 5))
    geo = (f"basin {built.slant_realised_deg:+.1f} deg" if built.geometry_mode == "basin"
           else f"channel {built.nominal_width:.1f} m")
    enc = ""
    if built.encounter_class not in ("no_target",):
        side = ""
        if built.encounter_class == "crossing":
            side = " from " + ("port" if built.ct_deg < 180.0 else "stbd")
        enc = f"{side}\nDCPA {built.dcpa_m:.2f} m, TCPA {built.tcpa_s:.1f} s, k {built.speed_ratio:.2f}"
    row = outcomes.loc[i] if i in outcomes.index else None
    res = ""
    if row is not None and compact:
        # Sheets: one letter per policy -- G goal, T target, O obstacle, B boundary, t timeout.
        code = {"goal": "G", "collision:target": "T", "collision:obstacle": "O",
                "collision:boundary": "B", "timeout": "t"}
        names = {"run8_final": "r8", "run8_best": "r8b", "run7": "r7", "reference": "ref"}
        res = "\n" + " ".join(f"{names.get(k, k)}:{code.get(v, '?')}" for k, v in row.items()
                             if isinstance(v, str))
    elif row is not None:
        short = lambda o: "goal" if o == "goal" else o.replace("collision:", "hit ")
        res = "\n" + ", ".join(f"{k.replace('_final', '')}: {short(v)}" for k, v in row.items()
                               if isinstance(v, str))
    if compact:
        geo = (f"basin {built.slant_realised_deg:+.0f} deg" if built.geometry_mode == "basin"
               else f"ch {built.nominal_width:.1f} m")
        enc = ""
        if built.encounter_class != "no_target":
            side = (("P " if built.ct_deg < 180.0 else "S ")
                    if built.encounter_class == "crossing" else "")
            enc = f"\n{side}D {built.dcpa_m:.1f} T {built.tcpa_s:.0f}s"
        ax.set_title(f"#{i:03d} {geo} {len(env.obstacles)}p{enc}{res}", fontsize=7, loc="left")
        return
    ax.set_title(f"#{i:03d} {built.encounter_class} | {geo} | {len(env.obstacles)} panels{enc}{res}",
                 fontsize=9, loc="left")


def main():
    (OUT / "scenarios").mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    dev = tf.development_set(20)
    outcomes = _outcomes()
    rows = []
    by_class = {c: [] for c in CLASSES}
    for i, built in enumerate(dev):
        # The evaluation seed Tier 1 and the in-run evaluation use, so the panels
        # drawn are the panels the policies met.
        env.reset(seed=900_000 + i, options={"generated": built})
        fig, ax = plt.subplots(figsize=(4.2, 9.0))
        _draw(ax, i, built, env, outcomes)
        fig.tight_layout()
        name = f"{i:03d}_{built.encounter_class}_{built.geometry_mode}.png"
        fig.savefig(OUT / "scenarios" / name, dpi=110)
        plt.close(fig)
        by_class[built.encounter_class].append((i, built, [list(p) for p in env.obstacles],
                                                np.asarray(env.path.points).copy(),
                                                (env.start_x, env.start_y, env.goal_x, env.goal_y)))
        rec = {"scenario": i, "file": f"scenarios/{name}", "class": built.encounter_class,
               "mode": built.geometry_mode,
               "slant_deg": round(float(built.slant_realised_deg), 2) if built.geometry_mode == "basin" else "",
               "width_m": round(float(built.nominal_width), 2),
               "w_eff_at_cpa_m": round(float(built.w_eff_at_cpa), 2),
               "start": [round(env.start_x, 2), round(env.start_y, 2)],
               "goal": [round(env.goal_x, 2), round(env.goal_y, 2)],
               "panels": len(env.obstacles)}
        if built.encounter_class != "no_target":
            rec.update(crossing_side=("port" if built.ct_deg < 180.0 else "starboard")
                       if built.encounter_class == "crossing" else "",
                       dcpa_m=round(float(built.dcpa_m), 2), tcpa_s=round(float(built.tcpa_s), 2),
                       speed_ratio=round(float(built.speed_ratio), 2), ct_deg=round(float(built.ct_deg), 1),
                       spawn_range_m=round(float(built.spawn_range_m), 2),
                       dcpa_below_floor=getattr(built, "dcpa_below_floor", None),
                       crossing_escapable=getattr(built, "crossing_escapable", None))
        if i in outcomes.index:
            rec.update(outcomes.loc[i].to_dict())
        rows.append(rec)

    # One contact sheet per class, re-drawn from the same realised layouts.
    for cls, items in by_class.items():
        fig, axes = plt.subplots(2, 10, figsize=(22, 13))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (i, built, _obs, _pts, _sg) in zip(axes.ravel(), items):
            ax.axis("on")
            env.reset(seed=900_000 + i, options={"generated": built})
            _draw(ax, i, built, env, outcomes, compact=True)
        fig.suptitle(f"Development set -- {cls} ({len(items)}).   Title: slant or channel width, panels, crossing side (P/S), "
                     f"drawn DCPA (m) and TCPA.   Outcomes: G goal, T target hit, O obstacle, B boundary, t timeout; "
                     f"r8 run 8 final, r8b run 8 best, r7 run 7, ref reference controller", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"sheet_{cls}.png", dpi=90)
        plt.close(fig)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "index.csv", index=False)
    basin = d[d["mode"] == "basin"]
    slant = pd.to_numeric(basin["slant_deg"])
    comp = d.groupby(["class", "mode"]).size().unstack(fill_value=0)
    result_cols = [c for c in ("run8_final", "run8_best", "run7", "reference") if c in d]
    goals = (d[result_cols].eq("goal").groupby(d["class"]).mean().round(2)
             if result_cols else pd.DataFrame())
    readme = f"""# Development set gallery

The 120 scenarios every formulation run is scored on (`train_formulation.development_set(20)`:
20 per class, stage 5, the `development` seed namespace). Evaluation seeds are 900 000 + index,
so the static panels shown are exactly the ones the policies met.

## Composition

{_md(comp)}

- **Basin mode: {len(basin)} of {len(d)}.** Legs run from y = {cfg.BASIN_START_Y:g} m to y = {cfg.BASIN_GOAL_Y:g} m,
  both x drawn in [{cfg.BASIN_X_RANGE[0]:g}, {cfg.BASIN_X_RANGE[1]:g}] m (Paper 2's layout), inside the
  whole 10 x 25 m basin less the 0.40 m inset.
- **Slant:** mean |slant| {slant.abs().mean():.1f} deg, max {slant.abs().max():.1f} deg; {(slant.abs() > 5).mean():.0%} of basin legs exceed 5 deg,
  {(slant.abs() > 10).mean():.0%} exceed 10 deg. Independent x draws make large slants rare: two uniform
  draws 5 m apart at most, over a 20 m leg, cap the slant at 14.0 deg.
- **Channel mode: {len(d) - len(basin)} of {len(d)}**, only head-on, crossing and overtaking (the classes whose
  rule the width decides), widths {d[d['mode'] == 'channel'].width_m.min():.1f}-{d[d['mode'] == 'channel'].width_m.max():.1f} m.
- Panels per scenario: {d.panels.mean():.1f} on average ({(d.panels == 0).mean():.0%} have none).

## Goal rate by class (Tier 1, supervisor off)

{_md(goals) if not goals.empty else 'No Tier 1 results found.'}

## Reading a figure

- Grey outline: basin envelope (10 x 25 m). Pale blue: navigable polygon (basin `P_nav`, or the channel).
- Darker blue strip (basin head-on only): the path band head-on traffic keeps to (Rule 9(a) with Rule 14).
- Dashed blue: reference leg; green dot: start; gold star: goal; green hull: own ship at spawn;
  green x: where the own ship would be at CPA if it held the leg at cruise.
- Red hull: target at spawn; dotted red arrow: its constant-velocity track to CPA (12 s for null);
  dashed red outline: target hull at CPA.
- Brown squares: static panels (A*-feasible layouts, F74).
- Title: index, class, geometry (basin slant or channel width), panel count, crossing side,
  drawn DCPA / TCPA / speed ratio, and the latest outcomes (run 8 final, run 8 best checkpoint,
  run 7, reference controller).

Regenerate with `python tools/diagnostics/devset_gallery.py`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(readme)


if __name__ == "__main__":
    main()
