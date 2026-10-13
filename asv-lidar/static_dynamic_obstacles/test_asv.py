"""Watch a trained policy (PPO, RecurrentPPO, SAC or TQC) and optionally save it as MP4.

    python test_asv.py --algo sac --seed 1                              # a random test set v4 episode
    python test_asv.py --algo sac --seed 1 --scenario BAS-HO-CV-001     # a chosen one
    python test_asv.py --algo ppo --seed 2 --set validation --encounter crossing --episodes 3
    python test_asv.py --algo tqc --set training --episodes 2            # stage-7 training scenarios
    python test_asv.py --model runs/sac_formulation_seed1_bl3/sac_2500000_steps.zip --scenario DV3-HO-CV-01
    python test_asv.py --algo sac --seed 1 --scenario BAS-CR-CV-003 --save-mp4 --headless

The policy is the run's best checkpoint (`runs/<algo>_formulation_seed<N>_bl3/best_model.zip`)
unless `--model` names a run directory or a `.zip`; the learner is read from the run's
`config.json` (`--algo` when there is none).  Actions are deterministic, and the safety layer
is off unless `--safety on`.

Scenario sets:

* **test** (default): test set v4, 1,000 episodes, run exactly as `tools/tiers/test_set.py`
  runs them (each with its own episode seed).  IDs are its `test_id`s
  (`results/test_set/v4/definition.csv`), e.g. BAS-HO-CV-001, FS-CRP-FIX-12.
* **validation**: the 270-episode validation set as the training evaluations run it
  (formulation overlay installed, episode i seeded 900,000 + i).  The decoupled part
  (120) is numbered here DEV-<HO|CR|OT|BO|NU|NT>-01..20; the coupled part (150) keeps
  its case IDs, e.g. DV3-CRP-CV-01.  Built on first use (several minutes) and cached in
  `results/validation_set/`.
* **training**: freshly generated stage-7 training scenarios (random seeds).

`--encounter` (head_on, crossing, overtaking, being_overtaken, null, no_target) filters
the test and validation sets; `--episodes N` picks N at random and prints their IDs, so
an episode can be replayed with `--scenario`.

The window shows the field and the telemetry panel.  Keys: SPACE pause, N next episode,
1-7 panel blocks, `,` / `.` panel history back / forward (L live), ESC or Q quit.  `--save-mp4` writes `results/videos/<run>_<scenario>.mp4`
(or the given path), H.264 at `--fps` (default 2, real time: one frame per 0.5 s
decision); `--headless` records without opening a window.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
LEARNERS = ("ppo", "recurrent_ppo", "sac", "tqc", "td3")
ENCOUNTERS = ("head_on", "crossing", "overtaking", "being_overtaken", "null", "no_target")
CLASS_CODES = {"head_on": "HO", "crossing": "CR", "overtaking": "OT", "being_overtaken": "BO",
               "null": "NU", "no_target": "NT"}
TRAINING_STAGE = 7
HOLD_S = 1.5                        # the last frame stays this long (window and video)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=LEARNERS[:4], help="learner of the run to load (with --seed)")
    ap.add_argument("--seed", type=int, default=0, help="training seed of the run (default 0)")
    ap.add_argument("--model", help="a run directory (its best_model.zip) or a model .zip instead")
    ap.add_argument("--set", choices=("test", "validation", "training"), default=None,
                    help="scenario set (default test; implied by --scenario IDs)")
    ap.add_argument("--scenario", nargs="+", metavar="ID", help="episode IDs to run, in order")
    ap.add_argument("--encounter", choices=ENCOUNTERS, help="only episodes of this encounter type")
    ap.add_argument("--episodes", type=int, default=1, help="episodes picked at random (default 1)")
    ap.add_argument("--safety", choices=("off", "on"), default="off", help="safety layer (default off)")
    ap.add_argument("--save-mp4", nargs="?", const="auto", default=None, metavar="PATH",
                    help="save each episode as MP4 (default results/videos/<run>_<scenario>.mp4)")
    ap.add_argument("--headless", action="store_true", help="no window (with --save-mp4: record only)")
    ap.add_argument("--fps", type=float, default=None, help="playback and video rate (default 2, real time)")
    ap.add_argument("--tag", default="bl3", help=argparse.SUPPRESS)
    return ap.parse_args()


def resolve_model(args) -> Path:
    if args.model:
        path = Path(args.model)
        if not path.is_absolute():
            path = path if path.exists() else ROOT / "runs" / path
    elif args.algo:
        path = ROOT / "runs" / f"{args.algo}_formulation_seed{args.seed}_{args.tag}"
    else:
        raise SystemExit("name the policy: --algo (with --seed) or --model")
    if path.is_dir():
        path = path / "best_model.zip"
    if not path.exists():
        raise SystemExit(f"model not found: {path}")
    return path.resolve()


def run_config(model_path: Path) -> Dict:
    cfg_path = model_path.parent / "config.json"
    return json.loads(cfg_path.read_text()) if cfg_path.exists() else {}


def learner_of(model_path: Path, config: Dict, algo: Optional[str]) -> str:
    if config.get("algo"):
        return config["algo"]
    if algo:
        return algo
    for name in sorted(LEARNERS, key=len, reverse=True):          # recurrent_ppo before ppo
        if model_path.stem.startswith(name):
            return name
    raise SystemExit(f"cannot tell the learner of {model_path.name}: give --algo")


def run_label(model_path: Path) -> str:
    """sacs1_bl3 for runs/sac_formulation_seed1_bl3/best_model.zip; the checkpoint name otherwise appended."""
    name = model_path.parent.name
    if "_formulation_seed" in name:
        algo, rest = name.split("_formulation_seed", 1)
        name = f"{algo}s{rest}"
    return name if model_path.stem == "best_model" else f"{name}_{model_path.stem}"


def baseline_of(config: Dict) -> Dict:
    """The frozen baseline the run trained on (baseline-v3 when the run does not say)."""
    path = (config.get("baseline_config") or {}).get("path") or "configs/baseline_v3.json"
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    return json.loads(path.read_text())


def install_overlay(baseline: Dict, cfg, tf) -> None:
    """The formulation overlay the run trained with (baseline-v3: formulation_v3)."""
    import importlib
    import baseline_config
    module = baseline_config.overlay_module(baseline)
    if module:
        importlib.import_module("formulation_v3" if module is True else str(module)).apply(cfg)
        tf.STAGE_SCHEDULE = cfg.CURRICULUM_STAGE_FRACTIONS


def test_set() -> List[Dict]:
    sys.path.insert(0, str(ROOT / "tools" / "tiers"))
    import test_set as ts
    kept, _ = ts.build_v4()
    return [{"id": it["test_id"], "built": it["built"], "seed": int(it["episode_seed"]), "class": it["class"],
             "about": f"test set v4, {it['source']} {it['cell']} {it['variant']}"} for it in kept]


def validation_set(tf, per_class: int, field_set, baseline: Dict) -> List[Dict]:
    """Built as the training evaluation builds it, once per formulation, then read from
    `results/validation_set/` (building takes several minutes)."""
    import pickle
    cache = (ROOT / "results" / "validation_set"
             / f"{baseline['id']}_{baseline['formulation_digest']}_{per_class}per_class.pkl")
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    print("building the validation set (several minutes, once; cached in results/validation_set/) ...",
          flush=True)
    dev = tf.development_set(per_class)
    out, counts = [], {}
    for built in dev:
        cls = built.encounter_class
        counts[cls] = counts.get(cls, 0) + 1
        out.append({"id": f"DEV-{CLASS_CODES[cls]}-{counts[cls]:02d}", "built": built, "class": cls,
                    "about": "validation set, decoupled part"})
    for built in field_set():
        out.append({"id": built.case_id, "built": built, "class": built.encounter_class,
                    "about": "validation set, coupled part"})
    for i, item in enumerate(out):
        item["seed"] = 900_000 + i                     # as FieldEvalCallback seeds them
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def pick(items: List[Dict], args) -> List[Dict]:
    import numpy as np
    if args.scenario:
        by_id = {it["id"]: it for it in items}
        missing = [s for s in args.scenario if s not in by_id]
        if missing:
            raise SystemExit(f"not in the {args.set} set: {', '.join(missing)} (e.g. {', '.join(list(by_id)[:3])})")
        return [by_id[s] for s in args.scenario]
    pool = [it for it in items if args.encounter in (None, it["class"])]
    if not pool:
        raise SystemExit(f"no {args.encounter} episodes in the {args.set} set")
    rng = np.random.default_rng()
    return [pool[i] for i in rng.choice(len(pool), size=min(args.episodes, len(pool)), replace=False)]


def outcome_of(info: Dict) -> str:
    if info.get("collided"):
        return f"collision:{info.get('collision_kind', '?')}"
    return "goal" if info.get("reached_goal") else "timeout"


class _Clock:
    """Stands in for the renderer's clock: the window at `fps`, or no wait when recording headless."""

    def __init__(self, fps: Optional[float]):
        import pygame
        self.fps, self.clock = fps, pygame.time.Clock()

    def tick(self, *_):
        return self.clock.tick(self.fps) if self.fps else 0


def main() -> int:
    args = parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    if args.scenario and args.set is None:
        ids = {("validation" if s.startswith(("DEV-", "DV3-")) else "test") for s in args.scenario}
        if len(ids) > 1:
            raise SystemExit("--scenario: IDs from one set at a time (test or validation)")
        args.set = ids.pop()
    args.set = args.set or "test"
    if args.set == "training" and (args.scenario or args.encounter):
        raise SystemExit("the training set is generated at random: no --scenario or --encounter")
    render = (not args.headless) or args.save_mp4 is not None

    model_path = resolve_model(args)
    config = run_config(model_path)
    algo = learner_of(model_path, config, args.algo)
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)
    import numpy as np
    import constants as cfg
    import curriculum
    import train_formulation as tf
    baseline = baseline_of(config)
    if args.set != "test":
        # The validation and training scenarios as the training process builds and runs
        # them; the test set runs without the overlay, as tools/tiers/test_set.py does.
        install_overlay(baseline, cfg, tf)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv

    if args.set == "test":
        chosen = pick(test_set(), args)
    elif args.set == "validation":
        import formulation_v3
        per_class = int((config.get("eval_per_class") or 20))
        chosen = pick(validation_set(tf, per_class, formulation_v3.field_development_set, baseline), args)
    else:
        seeds = np.random.default_rng().integers(1_000_000, 2_000_000, size=args.episodes)
        chosen = [{"id": f"TRAIN-S{TRAINING_STAGE}-{s}", "built": None, "seed": int(s), "class": "",
                   "about": f"training scenario, stage {TRAINING_STAGE}"} for s in seeds]

    model = tf.ALGORITHMS[algo].load(str(model_path), device="cpu")
    label = run_label(model_path)
    print(f"{label}: {algo}, {model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path}; "
          f"safety {args.safety}", flush=True)
    env = ASVLidarEnv(render_mode="human" if render else None, emergency_stop=args.safety == "on",
                      low_speed_start_frac=float(config.get("low_speed_start_frac") or 0.0)
                      if args.set == "training" else 0.0)
    if args.set == "training":
        env.set_scenario_stage(TRAINING_STAGE)
    fps = args.fps or 1.0 / cfg.UPDATE_RATE
    results, quit_all = [], False

    for item in chosen:
        if quit_all:
            break
        writer, video = None, None
        if args.save_mp4 is not None:
            import imageio.v2 as imageio
            if args.save_mp4 == "auto":
                video = ROOT / "results" / "videos" / f"{label}_{item['id']}.mp4"
            else:
                video = Path(args.save_mp4)
                if len(chosen) > 1:
                    video = video.with_name(f"{video.stem}_{item['id']}{video.suffix or '.mp4'}")
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video, fps=fps, codec="libx264", pixelformat="yuv420p",
                                        macro_block_size=8, ffmpeg_params=["-crf", "18"])
        options = {} if item["built"] is None else {"generated": item["built"]}
        obs, _ = env.reset(seed=item["seed"], options=options)
        if item["built"] is None and env.scenario is not None:
            item["class"] = env.scenario.encounter_class or ""

        def frame():
            import pygame
            surface = env.renderer.surface
            if writer is not None:
                writer.append_data(np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)))

        if render and env.renderer is not None:
            env.renderer.clock = _Clock(None if args.headless else fps)
            env.renderer.overlay = [f"{label}, safety {args.safety}", f"{item['id']}  seed {item['seed']}"]
            env.render()                                # redraw with the overlay
            frame()
        actor = tf.EpisodeActor(model)
        total, steps, info, paused = 0.0, 0, {}, False
        while True:
            if render and not args.headless:
                import pygame
                skip = False
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                                         and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                            quit_all = skip = True
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                            skip = True
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_7:
                            env.renderer.toggle_index(event.key - pygame.K_0)
                            env.render()
                        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_COMMA, pygame.K_PERIOD,
                                                                            pygame.K_l):
                            # the panel's history: back, forward, live
                            if event.key == pygame.K_l:
                                env.renderer.scrub = 0
                            else:
                                env.renderer.scrub_by(1 if event.key == pygame.K_COMMA else -1)
                            env.render()
                    if skip or not paused:
                        break
                    pygame.time.wait(50)
                if skip:
                    break
            obs, reward, term, trunc, info = env.step(actor(obs))   # step draws the frame
            total += float(reward)
            steps += 1
            if render:
                frame()
            if term or trunc:
                break
        result = outcome_of(info) if info else "skipped"
        if render and writer is not None:
            for _ in range(int(round(HOLD_S * fps))):
                frame()
        if render and not args.headless and not quit_all and result != "skipped":
            time.sleep(HOLD_S)
        if writer is not None:
            writer.close()
        results.append(result)
        print(f"{item['id']:<22} {item['class'] or '-':<16} {result:<20} {steps:>4} steps "
              f"({steps * cfg.UPDATE_RATE:.1f} s), return {total:8.1f}   {item['about']}"
              + (f"\n{'':22} -> {video.relative_to(ROOT) if video.is_relative_to(ROOT) else video}"
                 if video is not None and result != "skipped" else ""), flush=True)
    env.close()
    if len(results) > 1:
        done = [r for r in results if r != "skipped"]
        print(f"{sum(r == 'goal' for r in done)} of {len(done)} episodes reached the goal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
