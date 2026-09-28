"""A picture of every frozen-suite scenario (Tier B, suite 3.1: 39 cells x 20).

The frozen-suite counterpart of `devset_gallery.py`.  Writes `results/frozen_gallery/`:

* `scenarios/<test ID>.png` -- one figure per test: the basin envelope, the
  navigable polygon, the head-on band where it applies, the static panels as the
  environment realises them for that test's episode seed, the reference leg,
  and the **target's trajectory** -- simulated against a nominal own ship that
  holds the leg at cruise, so a reactive or non-compliant target's manoeuvre
  shows up where its model makes it (time marks every 5 s on both tracks, both
  hulls at the nominal CPA);
* `sheets/<cell>_<test-ID prefix>.png` -- the tests of each cell (20 basin, 40 channel);
* `index.csv` -- one row per test: test ID, case id, index, episode seed, cell,
  geometry, encounter parameters, and the nominal CPA;
* `README.md` -- the composition, the ID scheme, and how to read a figure.

    python tools/diagnostics/frozen_gallery.py

Replay any test with a policy: `python tools/tiers/run_test.py --model ... <test ID>`.
It runs in one process at whatever priority it is started with; start it at
idle priority while a campaign trains.
"""
import copy
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
import suite  # noqa: E402
import targets as tgt  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402
from frozen_suite import TIER_B_SEED  # noqa: E402
from nominal import nominal_encounter  # noqa: E402

OUT = ROOT / "results" / "frozen_gallery"
COLOURS = {"nav": "#dfe9f5", "band": "#b9d3ee", "wall": "#555555", "panel": "#8c6d46",
           "path": "#2b6cb0", "own": "#1f7a3f", "target": "#c0392b",
           "re": "#e67e22", "nc": "#8e44ad", "vs": "#16a085"}
MARK_EVERY_S = 5.0


def _md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return chr(10).join([head, rule] + body)


def _draw(ax, tid, built, cell, env, nom, compact=False, variants=None):
    w, h = cfg.MAP_WIDTH, cfg.MAP_HEIGHT
    ax.add_patch(MplPolygon([(0, 0), (w, 0), (w, h), (0, h)], closed=True, fill=False,
                            ec=COLOURS["wall"], lw=1.2))
    ax.add_patch(MplPolygon(env.boundary_polygon, closed=True, fc=COLOURS["nav"], ec="#7a9cc6", lw=0.8))
    if built.geometry_mode == "basin" and built.encounter_class == "head_on":
        ax.add_patch(MplPolygon(built.channel.band().polygon(), closed=True,
                                fc=COLOURS["band"], ec="none", alpha=0.7))
    for poly in env.obstacles:
        ax.add_patch(MplPolygon(poly, closed=True, fc=COLOURS["panel"], ec="k", lw=0.5))
    pts = np.asarray(env.path.points)
    ax.plot(pts[:, 0], pts[:, 1], color=COLOURS["path"], lw=1.0, ls="--")
    ax.plot(env.start_x, env.start_y, "o", color=COLOURS["own"], ms=4)
    ax.plot(env.goal_x, env.goal_y, "*", color="#d4a017", ms=10, mec="k", mew=0.4)
    ax.add_patch(MplPolygon(env.hull_polygon(), closed=True, fc=COLOURS["own"], ec="k", lw=0.4))

    every = int(round(MARK_EVERY_S / nom["dt"]))
    own = nom["own"]
    ax.plot(own[::every, 0], own[::every, 1], "o", color=COLOURS["own"], ms=2.2 if compact else 3,
            alpha=0.8, zorder=5)
    trk = nom["target"]
    if len(trk):
        # Suite 3.4's robustness variants of this scenario: the same target
        # with the reactive (orange) and, in head-ons, the non-compliant
        # (purple) model, against the same stand-on own ship.
        for behaviour, vnom in (variants or {}).items():
            vt = vnom["target"]
            ax.plot(vt[:, 0], vt[:, 1], color=COLOURS[behaviour], lw=1.1, ls="--", zorder=4)
            ax.plot(vt[::every, 0], vt[::every, 1], "o", color=COLOURS[behaviour],
                    ms=1.8 if compact else 2.4, zorder=5)
        ax.plot(trk[:, 0], trk[:, 1], color=COLOURS["target"], lw=1.3, zorder=4)
        ax.plot(trk[::every, 0], trk[::every, 1], "o", color=COLOURS["target"],
                ms=2.2 if compact else 3, zorder=5)
        ax.add_patch(MplPolygon(tgt.hull_polygon(*trk[0]), closed=True, fc=COLOURS["target"],
                                ec="k", lw=0.4, zorder=6))
        k = nom["cpa_k"]
        ax.add_patch(MplPolygon(tgt.hull_polygon(*trk[k]), closed=True, fill=False,
                                ec=COLOURS["target"], lw=0.9, ls="--", zorder=6))
        ax.add_patch(MplPolygon(tgt.hull_polygon(*own[k]), closed=True, fill=False,
                                ec=COLOURS["own"], lw=0.9, ls="--", zorder=6))
        ax.plot([own[k, 0], trk[k, 0]], [own[k, 1], trk[k, 1]], color="k", lw=0.6, zorder=5)

    ax.set_xlim(-0.5, w + 0.5)
    ax.set_ylim(-0.5, h + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([] if compact else range(0, 11, 2))
    ax.set_yticks([] if compact else range(0, 26, 5))
    geo = ("basin" if built.geometry_mode == "basin" else f"ch {built.nominal_width:.2f} m")
    enc = ""
    if built.encounter_class != "no_target":
        side = (("P " if built.ct_deg < 180.0 else "S ") if built.encounter_class == "crossing" else "")
        enc = (f"\n{side}D {built.dcpa_m:.1f} T {built.tcpa_s:.0f}s k {built.speed_ratio:.2f}"
               f" | CPA {nom['cpa_range']:.1f} m")
    if compact:
        ax.set_title(f"{tid}\n{geo}, {len(env.obstacles)}p{enc}", fontsize=6.5, loc="left")
        return
    turned = "".join(f"\n{b.upper()}: turns {v['turned_deg']:.0f} deg, CPA {v['cpa_range']:.1f} m"
                     for b, v in (variants or {}).items())
    where = (f"{built.case_id}, #{suite.tier_b_index(*_cell_n(built.case_id))}"
             if built.case_id.startswith("B-") else built.case_id)
    ax.set_title(f"{tid} ({where})\n"
                 f"{cell['class']} | {cell['behaviour']} | {geo} | {len(env.obstacles)} panels{enc}{turned}",
                 fontsize=8, loc="left")


def _variant_tracks(env, built, index):
    """Nominal tracks of this scenario's robustness variants (suite 3.4), each on
    the headline's episode seed, so only the target model differs."""
    out = {}
    for behaviour, classes in suite.ROBUST_BEHAVIOURS.items():
        if built.encounter_class not in classes:
            continue
        variant = copy.deepcopy(built)
        variant.target_behaviour = suite.target_model(behaviour, built.encounter_class)
        env.reset(seed=TIER_B_SEED + index, options={"generated": variant})
        out[behaviour] = nominal_encounter(env)
    return out


def _cell_n(case_id):
    c, n = (int(v) for v in case_id.split("-")[1:3])
    return c, n


def main():
    (OUT / "scenarios").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    tier_b, short = suite.build_tier_b()
    cells = suite.tier_b_cells()
    rows, by_cell = [], {}
    for i, built in enumerate(tier_b):
        c_index, n = _cell_n(built.case_id)
        if suite.tier_b_index(c_index, n) != i:
            raise SystemExit(f"{built.case_id}: index {i} != tier_b_index -- a cell fell short")
        cell = cells[c_index]
        tid = suite.test_id(built.case_id)
        variants = _variant_tracks(env, built, i)
        env.reset(seed=TIER_B_SEED + i, options={"generated": built})
        nom = nominal_encounter(env)
        fig, ax = plt.subplots(figsize=(4.2, 9.0))
        _draw(ax, tid, built, cell, env, nom, variants=variants)
        fig.tight_layout()
        fig.savefig(OUT / "scenarios" / f"{tid}.png", dpi=110)
        plt.close(fig)
        by_cell.setdefault(c_index, []).append((tid, built, i))
        rows.append({"test_id": tid, "case_id": built.case_id, "index": i,
                     "episode_seed": TIER_B_SEED + i, "file": f"scenarios/{tid}.png",
                     "stratum": cell["stratum"], "class": cell["class"],
                     "behaviour": cell["behaviour"], "mode": built.geometry_mode,
                     "width_m": round(float(built.nominal_width), 2) if built.geometry_mode != "basin" else "",
                     "panels": len(env.obstacles),
                     "crossing_side": (("port" if built.ct_deg < 180.0 else "starboard")
                                       if built.encounter_class == "crossing" else ""),
                     "dcpa_m": round(float(built.dcpa_m), 2), "tcpa_s": round(float(built.tcpa_s), 1),
                     "speed_ratio": round(float(built.speed_ratio), 2),
                     "ct_deg": round(float(built.ct_deg), 1),
                     "target_behaviour_field": built.target_behaviour,
                     "nominal_cpa_m": round(nom["cpa_range"], 2),
                     "nominal_cpa_t_s": round(nom.get("cpa_t", float("nan")), 1),
                     "target_turned_deg": round(nom["turned_deg"], 1),
                     **{f"{b}_{k}": round(v[key], 2) for b, v in variants.items()
                        for k, key in (("cpa_m", "cpa_range"), ("turned_deg", "turned_deg"))}})
        if (i + 1) % 60 == 0:
            print(f"{i + 1}/{len(tier_b)}", flush=True)

    for c_index, items in by_cell.items():
        cell = cells[c_index]
        n_rows = -(-len(items) // 10)
        fig, axes = plt.subplots(n_rows, 10, figsize=(22, 6.5 * n_rows))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (tid, built, i) in zip(axes.ravel(), items):
            ax.axis("on")
            variants = _variant_tracks(env, built, i)
            env.reset(seed=TIER_B_SEED + i, options={"generated": built})
            _draw(ax, tid, built, cell, env, nominal_encounter(env), compact=True, variants=variants)
        prefix = items[0][0].rsplit("-", 1)[0]
        fig.suptitle(f"Frozen suite, Tier B cell {c_index:02d} ({prefix}): {cell['stratum']}, "
                     f"{cell['class']}, target behaviour {cell['behaviour']}.   Title: test ID; geometry, "
                     f"panels; crossing side (P/S), drawn DCPA (m), TCPA, speed ratio; nominal CPA.   "
                     f"Dots every {MARK_EVERY_S:g} s; own ship holding the leg at cruise.", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "sheets" / f"{c_index:02d}_{prefix}.png", dpi=90)
        plt.close(fig)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "index.csv", index=False)
    comp = d.groupby(["stratum", "class"]).size().unstack(fill_value=0)
    turned = {b: float((d[f"{b}_turned_deg"].dropna() > 1.0).mean())
              for b in suite.ROBUST_BEHAVIOURS if f"{b}_turned_deg" in d}
    readme = f"""# Frozen suite gallery (Tier B, suite {suite.SUITE_REVISION})

The {len(d)} scenarios of the default frozen suite, each under its test ID.  The static
panels are the ones the environment realises for that test's episode seed
({TIER_B_SEED:,} + index), so they are the panels a policy meets.  Shortfall: {short or 'none'}.

## Test IDs

`<stratum>-<class>-<behaviour>-<nn>`, nn counted from 01 within the cell.

| Part | Codes |
|---|---|
| stratum | BAS basin (01-20), CH channel 10 m (01-40) |
| class | HO head-on, CR crossing, OT overtaking, BO being overtaken, NU null |
| behaviour | CV constant velocity, RE reactive, NC non-compliant (null: CV only) |

Replay one or more with a policy (seconds each):

    python tools/tiers/run_test.py --model runs/sac_formulation_seed0_bl2/best_model.zip CH-CR-RE-07
    python tools/tiers/run_test.py --policy colregs_vo BAS-HO-NC-03 --supervisor both

The case id (`B-cc-nn`, cell and episode from 00) and the index (0-{len(d) - 1}) are accepted too.

## Composition

{_md(comp)}

The headline is constant-velocity only, like the development set; the robustness set
reruns the same scenarios with a reactive target (every encounter class) and, in head-ons,
a non-compliant one (`CH-CR-RE-007` is `CH-CR-CV-007` with a reactive target).

## Robustness variants as realised

Share of variants whose target heading changes by more than 1 deg against a stand-on own
ship: {', '.join(f'{k} {v:.0%}' for k, v in turned.items())} (a reactive target that is the
stand-on vessel, or already passing clear, holds its course).

## Reading a figure

- Grey outline: basin envelope (10 x 25 m). Pale blue: navigable polygon (basin `P_nav`, or the channel).
- Darker blue strip (basin head-on only): the band head-on traffic keeps to.
- Dashed blue: reference leg; green dot: start; gold star: goal; green hull: own ship at spawn.
- Green dots: the own ship holding the leg at cruise ({cfg.U_NOM:g} m/s), every {MARK_EVERY_S:g} s.
- Red hull and line: the target at spawn and its constant-velocity track against that own
  ship (the headline), dots every {MARK_EVERY_S:g} s (same instants as the green dots).
- Orange dashed: the same target with the reactive model (robustness set); purple dashed
  (head-on only): with the non-compliant model, which alters to port.
- Dashed outlines joined by a black line: both hulls at the nominal CPA.
- Brown squares: static panels.
- Title: test ID (case id, index); class | geometry | panels; crossing side, drawn
  DCPA / TCPA / speed ratio; nominal CPA range; the variants' turn and CPA.

Sheets (`sheets/`) show one cell each.  Regenerate with `python tools/diagnostics/frozen_gallery.py`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(readme)


if __name__ == "__main__":
    main()
