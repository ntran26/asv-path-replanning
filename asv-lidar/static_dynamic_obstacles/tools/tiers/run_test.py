"""Replay chosen frozen-suite (Tier B) tests, one episode each, with a figure.

A test is named by its test ID (`CH-CR-CV-007`; `CH-CR-RE-007` is the same
scenario with a reactive target, from the robustness set), its case id
(`B-06-006`, `B-06-006-RE`) or its headline index (`346`);
`results/frozen_gallery/index.csv` lists every scenario with a picture of each.
The episode is the one the frozen suite runs -- same scenario, same episode
seed -- so a result here matches that test's row in
`results/frozen_suite/<tag>/episodes.csv`.

    python tools/tiers/run_test.py --model runs/sac_formulation_seed0_bl2/best_model.zip CH-CR-CV-007
    python tools/tiers/run_test.py --model runs/tqc_formulation_seed0_bl2/best_model.zip BAS-HO-NC-003 B-06-006-RE --supervisor on
    python tools/tiers/run_test.py --policy colregs_vo CH-CR-CV-031

Writes `results/test_runs/<tag>/<test ID>_supervisor_<mode>.png` and appends
the outcome to `results/test_runs/<tag>/runs.csv`.  Only the named tests'
cells are generated, so a replay takes seconds, not the full build.
"""
import argparse
import copy
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import curriculum  # noqa: E402
import suite  # noqa: E402
import train_formulation as tf  # noqa: E402
from common import load_model, make_controller  # noqa: E402
from frozen_suite import TIER_B_SEED  # noqa: E402
from paper_paths import _draw  # noqa: E402

OUT = ROOT / "results" / "test_runs"


def replay(env, built, seed, actor=None, controller=None) -> dict:
    """One episode, recording what `paper_paths._draw` needs."""
    obs, _ = env.reset(seed=seed, options={"generated": built})
    if actor is not None:
        actor.reset()
    own, track, engaged, speeds = [], [], [], []
    min_range, cpa_step, estops = float("inf"), 0, 0
    while True:
        own.append((env.asv_x, env.asv_y, env.asv_h))
        speeds.append(float(env.model.u))
        engaged.append(any(c.engaged for c in env.encounter_contexts.values()))
        if env.targets:
            t = env.targets[0]
            track.append((t.x, t.y, float(t.heading)))
            rng = math.hypot(t.x - env.asv_x, t.y - env.asv_y)
            if rng < min_range:
                min_range, cpa_step = rng, len(own) - 1
        action = actor(obs) if actor is not None else controller.action(env, obs)
        obs, _, term, trunc, info = env.step(action)
        estops = int(info["estop/events"])
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return {"own": np.array(own), "target": np.array(track), "speeds": np.array(speeds),
            "engaged": np.array(engaged), "min_range": min_range, "cpa_step": cpa_step,
            "outcome": outcome, "steps": len(own), "built": built, "estops": estops,
            "obstacles": [np.asarray(p) for p in env.obstacles],
            "boundary": np.asarray(env.boundary_polygon),
            "path": np.asarray(env.path.points), "goal": (env.goal_x, env.goal_y)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tests", nargs="+", help="test IDs, case ids or Tier B indices")
    ap.add_argument("--model", type=Path, help="a trained policy (.zip)")
    ap.add_argument("--policy", choices=("los_dwa", "colregs_vo", "encounter_vo", "reference"),
                    help="a classical comparator instead of a model")
    ap.add_argument("--supervisor", choices=("off", "on", "both"), default="off")
    ap.add_argument("--tag", help="output folder name (default: from the model or policy)")
    args = ap.parse_args()
    if bool(args.model) == bool(args.policy):
        ap.error("give exactly one of --model or --policy")

    wanted = [(ref, *suite.resolve_test(ref)) for ref in args.tests]
    built_cells = {}
    for c_index in sorted({c for _, c, _, _ in wanted}):
        scenarios, short = suite.build_tier_b(cells=[c_index])
        if short:
            raise SystemExit(f"cell {c_index} is short of its count -- indices would not match the suite")
        built_cells[c_index] = scenarios

    if args.model:
        model_path = args.model if args.model.is_absolute() else ROOT / args.model
        tag = args.tag or f"{model_path.parent.name}_{model_path.stem}"
    else:
        tag = args.tag or args.policy
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    env = ASVLidarEnv(render_mode=None)
    actor = tf.EpisodeActor(load_model(str(model_path))) if args.model else None
    modes = ("off", "on") if args.supervisor == "both" else (args.supervisor,)

    rows = []
    for _ref, c_index, n, behaviour in wanted:
        built = built_cells[c_index][n]
        if behaviour != "cv":
            # A robustness variant: the headline scenario, another target model,
            # the same episode seed (`suite.robustness_variants`).
            built = copy.deepcopy(built)
            built.target_behaviour = suite.target_model(behaviour, built.encounter_class)
            built.case_id = f"{built.case_id}-{behaviour.upper()}"
        tid = suite.test_id(built.case_id)
        index = suite.tier_b_index(c_index, n)
        cell = dict(suite.tier_b_cells()[c_index], behaviour=behaviour)
        for mode in modes:
            env.estop_enabled = mode == "on"
            controller = make_controller(args.policy) if args.policy else None
            ep = replay(env, built, TIER_B_SEED + index, actor, controller)
            rows.append({"test_id": tid, "case_id": built.case_id, "index": index,
                         "stratum": cell["stratum"], "class": cell["class"],
                         "behaviour": cell["behaviour"], "supervisor": mode,
                         "outcome": ep["outcome"], "steps": ep["steps"],
                         "min_target_range": round(ep["min_range"], 3), "estops": ep["estops"]})
            fig, ax = plt.subplots(figsize=(3.6, 7.8))
            lc = _draw(ax, ep)
            geo = ("basin" if built.geometry_mode == "basin"
                   else f"channel {built.nominal_width:.2f} m")
            close = ("no target" if not np.isfinite(ep["min_range"])
                     else f"min range {ep['min_range']:.2f} m")
            ax.set_title(f"{tid} ({built.case_id}) - {tag}\n{cell['class']}, {cell['behaviour']}, {geo}"
                         f"\nsupervisor {mode}: {ep['outcome']}, {close}", fontsize=7.5, loc="left")
            cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label("speed (m/s)", fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"{tid}_supervisor_{mode}.png", dpi=150)
            plt.close(fig)

    d = pd.DataFrame(rows)
    log = out / "runs.csv"
    d.to_csv(log, mode="a", header=not log.exists(), index=False)
    pd.set_option("display.width", 200)
    print(d.to_string(index=False))
    print(f"\nfigures and runs.csv in {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
