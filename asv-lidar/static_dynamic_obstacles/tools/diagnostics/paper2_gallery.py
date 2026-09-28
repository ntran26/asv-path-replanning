"""A picture of every scenario in the Paper 2 deployment-layout set.

Writes `results/paper2_gallery/`:

* `scenarios/<test ID>.png` -- one figure per target scenario (its FIX form: the
  constant-velocity target in red, with its VAR twin's track -- the same target
  varying its speed -- in green dashed, time marks every 5 s on all tracks) and
  one per layout without a target;
* `sheets/<layout>_<encounter>.png` -- the 20 scenarios of each cell on a page;
* `index.csv`, `field_sheet.csv` (each run's field set-up), and `README.md`
  (composition, ID scheme, field rules, how to read a figure).

    python tools/diagnostics/paper2_gallery.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import constant_temp as ct  # noqa: E402
import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import paper2_set as p2  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402
from frozen_gallery import MARK_EVERY_S, _draw  # noqa: E402
from nominal import nominal_encounter  # noqa: E402
from paper2_suite import index_frame  # noqa: E402

OUT = ROOT / "results" / "paper2_gallery"


def main():
    (OUT / "scenarios").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    records, short = p2.build(env)
    by_id = {r["test_id"]: r for r in records}
    shown = [r for r in records if r["speed"] == "FIX"] + \
            [r for r in records if r["encounter"] == "NT" and r["test_id"].endswith("-01")]
    by_cell = {}
    for r in shown:
        built = r["built"]
        variants = {}
        if r["speed"] == "FIX":
            twin = by_id[built.case_id.replace("-FIX-", "-VAR-")]
            env.reset(seed=r["episode_seed"], options={"generated": twin["built"]})
            variants["vs"] = nominal_encounter(env)
        env.reset(seed=r["episode_seed"], options={"generated": built})
        nom = nominal_encounter(env)
        cell = {"class": built.encounter_class, "behaviour": "cv"}
        tid = r["test_id"] if r["encounter"] != "NT" else r["test_id"].rsplit("-", 1)[0]
        fig, ax = plt.subplots(figsize=(4.2, 9.0))
        _draw(ax, tid, built, cell, env, nom, variants={"vs": variants["vs"]} if variants else None)
        ax.set_title(ax.get_title(loc="left").replace("\nVS:", "\nVAR:") + f"\n{p2.LAYOUTS[r['layout']]['name']}",
                     fontsize=8, loc="left")
        fig.tight_layout()
        fig.savefig(OUT / "scenarios" / f"{tid}.png", dpi=110)
        plt.close(fig)
        by_cell.setdefault((r["layout"], r["encounter"]), []).append((tid, r, variants))

    for (lay, enc), items in by_cell.items():
        if enc == "NT":
            continue
        fig, axes = plt.subplots(2, 10, figsize=(22, 13))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (tid, r, variants) in zip(axes.ravel(), items):
            ax.axis("on")
            env.reset(seed=r["episode_seed"], options={"generated": r["built"]})
            _draw(ax, tid, r["built"], {"class": r["built"].encounter_class, "behaviour": "cv"}, env,
                  nominal_encounter(env), compact=True, variants=variants)
        fig.suptitle(f"Paper 2 deployment-layout set: {p2.LAYOUTS[lay]['name']}, {enc}.   Red: "
                     f"constant-velocity target (FIX); green dashed: the same target varying its speed "
                     f"(VAR).   Dots every {MARK_EVERY_S:g} s; own ship holding the leg at cruise.", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "sheets" / f"{lay}_{enc}.png", dpi=90)
        plt.close(fig)

    d = index_frame(records)
    d.to_csv(OUT / "index.csv", index=False)
    pd.DataFrame(p2.field_sheet(records)).to_csv(OUT / "field_sheet.csv", index=False)
    comp = d.groupby(["layout", "encounter"]).size().unstack(fill_value=0)
    lays = "\n".join(f"| {k} | {v['name']} | {v['start']} -> {v['goal']} | "
                     + ", ".join(f"({x:g}, {y:g})" for x, y in v["panels"]) + " |"
                     for k, v in p2.LAYOUTS.items())
    readme = f"""# Paper 2 deployment-layout set (revision {p2.SET_REVISION})

A **separate** evaluation set, not part of the frozen suite: the three static-obstacle
layouts of the published Paper 2 field trials (Tran et al., *Drones* 10(9), 680, Fig. 8),
with and without a target ship. {len(d)} episodes per supervisor mode. Shortfall: {short or 'none'}.

## Layouts (1 m square panels, measured from the published Fig. 8)

| Layout | Scenario | Leg | Panel centres (m) |
|---|---|---|---|
{lays}

## Composition (episodes)

{comp.to_string()}

- **NT**: no target, each layout as flown, {p2.NO_TARGET_EPISODES} episode seeds (sensor-noise realisations).
- **HO / CRP / CRS / OT / BO**: head-on, crossing from port, crossing from starboard, overtaking,
  being overtaken; {p2.PER_CELL} scenarios per layout, each run twice: **FIX** (constant velocity, as in
  training) and **VAR** (the same target on the same seed, changing speed once on the approach --
  slower to 0.5-0.8x or faster to 1.25-1.5x, ramped at 0.05 m/s^2; `targets.T_VS`).
- **Field feasible** (revision 2.0): every target is one a second, Bluefin-class vessel can sail in
  the basin -- it starts at least {ct.FIELD_WALL_MARGIN_M:g} m inside the walls, holds one heading
  (a track the lane clamp would bend is redrawn), stops short of the wall as the boat would in the
  field (the encounter is over {ct.FIELD_POST_CPA_S:g} s before), keeps {ct.FIELD_TARGET_SPEED_RANGE[0]:g}-{ct.FIELD_TARGET_SPEED_RANGE[1]:g} m/s
  (the VAR final speed too), and clears every panel by {ct.TARGET_PANEL_CLEARANCE_M:g} m.
- `field_sheet.csv` gives each run's set-up: target start (x, y), heading to hold, speed, and for
  VAR when (seconds after the own ship leaves its start at cruise) to change speed and to what.

## Test IDs

`P2-<layout>-<encounter>-<FIX|VAR>-<nn>` and `P2-<layout>-NT-<nn>`. Replay any:

    python tools/tiers/run_test.py --model runs/sac_formulation_seed0_bl2/best_model.zip P2-L2-CRP-VAR-07

Run the whole set: `python tools/tiers/paper2_suite.py --model ... --tag ...`.

## Reading a figure

As the frozen-suite gallery: grey basin outline, pale blue navigable area, dashed blue leg,
green start, gold goal, brown panels; green dots the own ship holding the leg at cruise
({cfg.U_NOM:g} m/s); red the constant-velocity target and its track; **green dashed the VAR
twin's track** (same line, different timing); dashed hulls at the nominal CPA.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(readme)


if __name__ == "__main__":
    main()
