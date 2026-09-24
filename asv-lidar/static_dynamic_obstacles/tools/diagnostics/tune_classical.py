"""Tune the classical comparators on the development set (C3a).

A comparator that loses on its library defaults proves nothing: the first
objection to any "learned beats classical" result is that the baseline was not
tuned. So each comparator is fitted on the **development set** — the same
scenarios the RL checkpoints are selected on, never the frozen suite — against
the **same score** used to pick an RL checkpoint:

    score = goal rate - 2 x collision rate

The search is **coordinate descent over a declared grid**: start from the
current values, take each parameter in turn, keep the value that scores best,
move on. It is reported in full (`search.csv`), so the tuning is auditable
rather than "we tried some numbers".

    python tools/diagnostics/tune_classical.py --controller colregs_vo
    python tools/diagnostics/tune_classical.py --controller los_dwa --per-class 10

Writes `results/classical_tuning/<controller>/`: `search.csv` (every
configuration tried, its score and per-class outcomes), `best.json` (the chosen
values, ready to paste into `src/constant_temp.py`) and `summary.txt`.

The grids live in `GRIDS` below: each entry names the module attribute the
controller reads, the `constant_temp` name it is stored under, and the values
to try. Both must be listed, because the modules bind their parameters at
import: the worker sets both so a swept value actually takes effect.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

OUT = ROOT / "results" / "classical_tuning"

# (module, attribute, constant_temp name, values to try)
GRIDS = {
    "colregs_vo": [
        ("classical.colregs_vo", "TAU_S", "CLASSICAL_KVO_TAU_S", [12.0, 20.0, 30.0]),
        ("classical.colregs_vo", "HARD_GAP_M", "CLASSICAL_KVO_HARD_GAP_M", [0.20, 0.35, 0.50]),
        ("classical.colregs_vo", "SIDE_FREE_DCPA_M", "CLASSICAL_KVO_SIDE_FREE_DCPA_M",
         [1.75, 2.50, 3.50]),
        ("classical.colregs_vo", "STAND_ON_RELEASE_S", "CLASSICAL_KVO_STAND_ON_RELEASE_S",
         [8.0, 12.0, 18.0]),
        ("classical.colregs_vo", "W_CHANGE", "CLASSICAL_KVO_W_CHANGE", [0.0, 0.3, 0.8]),
    ],
    "los_dwa": [
        ("classical.los_dwa", "SAFE_GAP_M", "CLASSICAL_DWA_SAFE_GAP_M", [0.15, 0.20, 0.35]),
        ("classical.los_dwa", "TARGET_HORIZON_S", "CLASSICAL_DWA_TARGET_HORIZON_S",
         [8.0, 10.0, 14.0]),
        ("classical.los_dwa", "DIST_CAP_M", "CLASSICAL_DWA_DIST_CAP_M", [1.0, 1.5, 2.5]),
        ("classical.los_dwa", "PATH_SCALE_M", "CLASSICAL_DWA_PATH_SCALE_M", [1.0, 2.0, 3.5]),
        ("classical.los_dwa", "COMMIT_S", "CLASSICAL_DWA_COMMIT_S",
         [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0)]),
    ],
}

_W = {}


def _init(controller: str, settings: dict) -> None:
    import importlib

    import torch
    torch.set_num_threads(1)
    import constant_temp as ct
    import curriculum
    import train_formulation as tf
    for (module_name, attr, const_name), value in settings.items():
        setattr(ct, const_name, value)
        module = importlib.import_module(module_name)
        setattr(module, attr, value)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None)
    _W["controller"] = controller


def _episode(job):
    from common import run_episode
    built, seed = job
    return run_episode(_W["env"], built, seed, _W["controller"])


def _score(rows: pd.DataFrame) -> float:
    """The RL checkpoint rule, so both sides are selected the same way."""
    return float((rows.outcome == "goal").mean() - 2.0 * rows.collided.mean())


def evaluate(controller: str, settings: dict, scenarios, processes: int) -> pd.DataFrame:
    jobs = [(b, 900_000 + i) for i, b in enumerate(scenarios)]
    with ProcessPoolExecutor(processes, initializer=_init,
                             initargs=(controller, settings)) as pool:
        return pd.DataFrame(list(pool.map(_episode, jobs, chunksize=1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=sorted(GRIDS), required=True)
    ap.add_argument("--per-class", type=int, default=10, help="development episodes per class")
    ap.add_argument("--processes", type=int, default=10)
    ap.add_argument("--passes", type=int, default=1, help="coordinate-descent passes")
    ap.add_argument("--validate", action="store_true",
                    help="score the defaults and the tuned values from best.json on the "
                         "full development set, instead of searching")
    args = ap.parse_args()

    import constant_temp as ct
    from common import development_set
    out = OUT / args.controller
    out.mkdir(parents=True, exist_ok=True)

    full = development_set(20)
    per = args.per_class
    scenarios = [b for i, b in enumerate(full) if i % 20 < per]
    grid = GRIDS[args.controller]
    settings = {(m, a, c): getattr(ct, c) for m, a, c, _ in grid}

    started, rows, history = time.time(), [], []

    def run(label, trial):
        d = evaluate(args.controller, trial, scenarios, args.processes)
        s = _score(d)
        by_class = d.groupby("class").apply(lambda x: (x.outcome == "goal").mean(),
                                            include_groups=False).round(2).to_dict()
        rows.append({"label": label, "score": round(s, 4),
                     "goal": round((d.outcome == "goal").mean(), 3),
                     "collision": round(d.collided.mean(), 3),
                     "target_coll": round(d.collided_target.mean(), 3),
                     **{f"goal_{k}": v for k, v in by_class.items()},
                     **{c: trial[(m, a, c)] for m, a, c, _ in grid}})
        pd.DataFrame(rows).to_csv(out / "search.csv", index=False)
        print(f"  {label:44s} score {s:+.3f}  goal {(d.outcome == 'goal').mean():.2f}", flush=True)
        return s

    print(f"{args.controller}: {len(scenarios)} development episodes per configuration, "
          f"{args.processes} workers", flush=True)
    if args.validate:
        # A 60-episode search can chase noise, so the chosen values are confirmed
        # on every development episode before they are saved.
        scenarios = full
        chosen = json.loads((out / "best.json").read_text())["values"]
        tuned = {(m, a, c): (tuple(chosen[c]) if isinstance(chosen[c], list) else chosen[c])
                 for m, a, c, _ in grid}
        base = run(f"defaults ({len(full)} episodes)", settings)
        keep = run(f"tuned ({len(full)} episodes)", tuned)
        verdict = "tuned" if keep > base else "defaults"
        text = (f"Validation on {len(full)} development episodes: defaults {base:+.3f}, "
                f"tuned {keep:+.3f} -> keep {verdict}" + "\n")
        (out / "validation.txt").write_text(text, encoding="utf-8")
        pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
        print(text)
        return 0
    best = run("defaults", settings)
    history.append(("defaults", dict(settings), best))

    for p in range(args.passes):
        for module_name, attr, const_name, values in grid:
            key = (module_name, attr, const_name)
            for value in values:
                if value == settings[key]:
                    continue                      # already scored as the incumbent
                trial = {**settings, key: value}
                s = run(f"pass{p + 1} {attr}={value}", trial)
                if s > best:
                    best, settings = s, trial
                    history.append((f"{attr}={value}", dict(settings), s))

    chosen = {c: settings[(m, a, c)] for m, a, c, _ in grid}
    with open(out / "best.json", "w") as fh:
        json.dump({"controller": args.controller, "score": best,
                   "episodes": len(scenarios), "per_class": per,
                   "values": {k: list(v) if isinstance(v, tuple) else v
                              for k, v in chosen.items()}}, fh, indent=1)
    lines = [f"Tuning {args.controller} on {len(scenarios)} development episodes "
             f"({per} per class), {time.time() - started:.0f} s",
             "score = goal rate - 2 x collision rate, the rule RL checkpoints are selected by",
             "", "== search (coordinate descent over the declared grid)",
             pd.DataFrame(rows).to_string(index=False), "",
             "== chosen"] + [f"  {k} = {v}" for k, v in chosen.items()] + [
             "", f"defaults scored {history[0][2]:+.3f}; chosen {best:+.3f}"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-len(chosen) - 4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
