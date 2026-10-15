"""Train one learner on the frozen formulation (baseline-v3): PPO, RecurrentPPO, SAC or TQC.

    python train_asv.py --algo sac --seed 1                            # new run to 3.0 M
    python train_asv.py --algo ppo --seed 2 --timesteps 1000000        # a shorter run
    python train_asv.py --algo sac --seed 1 --resume                   # continue, empty replay buffer
    python train_asv.py --algo sac --seed 1 --resume --replay-buffer   # continue on the saved buffer
    python train_asv.py --algo tqc --seed 0 --resume --timesteps 3500000   # continue past the budget

A front end to `src/train_formulation.py` with the settings of the paper runs
(`results/run_bl3_3m.sh`), so a run started here is the same experiment:

* **Formulation:** `configs/baseline_v3.json`; training refuses to start if the
  code no longer matches it.
* **Timesteps:** the curriculum stages stay on the frozen 2.5 M schedule (0.16,
  0.36, 0.64, 1.0, 1.5, 2.0 M).  Past 2.5 M the final stage holds (the paper
  runs train to 3.0 M, the default); a shorter run stops partway through the
  curriculum.
* **Evaluations:** the validation set every 200,000 steps with the safety layer
  off; the best checkpoint (goal - 2 x collision) is `best_model.zip`.  SAC and
  TQC runs past 2.5 M are also evaluated at 2.5 M, as seed 0 was.
* **CPU:** the measured faster setting (`results/bench_cpu_*.txt`).  On this
  laptop (12 logical CPUs, CPUs 0-3 the two performance cores) SAC and TQC run
  pinned with 4 PyTorch threads: learner on CPUs 0-3 at AboveNormal priority,
  workers on CPUs 4-11.  PPO and RecurrentPPO run unpinned.  Elsewhere nothing
  is pinned.

Continuing (`--resume`) starts from the run's latest checkpoint.  SAC and TQC
start with an empty replay buffer, refilled before updates resume, unless
`--replay-buffer` restores the buffer saved with that checkpoint (or a given
`.pkl`).  Buffers live outside the repository (`train_formulation.REPLAY_BUFFER_DIR`);
during a run only the latest is kept, and a finished run keeps its final one.
`--timesteps` above the run's budget extends it.

Output goes to `runs/<algo>_formulation_seed<N>_bl3/`, the console and
`runs/<algo>_formulation_seed<N>_bl3.log`; start and end are noted in
`results/train_seed.log`.  Training stops with the terminal (or Ctrl+C) and
continues from the last checkpoint (every 250,000 steps) with `--resume`.
Evaluation and test set v4 are separate steps, printed at the end; a policy is
watched with `test_asv.py`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
CONFIG = Path("configs") / "baseline_v3.json"
TAG = "bl3"
BUDGET = 2_500_000                 # the frozen run budget; the stage schedule's 100 %
DEFAULT_TIMESTEPS = 3_000_000      # the paper runs (decision, 2026-10-10)
LEARNERS = ("ppo", "recurrent_ppo", "sac", "tqc")
OFF_POLICY = ("sac", "tqc")
MILESTONES = ROOT / "results" / "train_seed.log"
# This laptop (i5-1245U): CPUs 0-3 are the two hyper-threaded performance cores,
# 4-11 the efficiency cores.  The pinned setting only applies on a 12-CPU Windows machine.
LEARNER_CPUS, WORKER_CPUS = [0, 1, 2, 3], list(range(4, 12))
PINNED_THREADS = 4


@dataclass
class Plan:
    run_dir: Path
    run_id: str
    argv: List[str]
    pinned: bool
    target: int
    resume_from: Optional[int] = None
    notes: List[str] = field(default_factory=list)


def _checkpoints(run_dir: Path, algo: str) -> List[int]:
    prefix = f"{algo}_"
    return sorted(int(p.name[len(prefix):-len("_steps.zip")]) for p in run_dir.glob(f"{prefix}*_steps.zip")
                  if p.name[len(prefix):-len("_steps.zip")].isdigit())


def _pin_supported() -> bool:
    return sys.platform == "win32" and os.cpu_count() == 12


def _saved_buffer(run_dir: Path, algo: str, steps: int) -> Optional[Path]:
    sys.path.insert(0, str(ROOT / "src"))
    from train_formulation import replay_buffer_path
    for path in (replay_buffer_path(run_dir, algo, steps),
                 run_dir / f"{algo}_replay_buffer_{steps}_steps.pkl"):
        if path.exists():
            return path
    return None


def plan(args, runs_dir: Path = ROOT / "runs", pin: Optional[bool] = None) -> Plan:
    """What to run, as train_formulation.py arguments; stops on an inconsistent request."""
    algo, tag = args.algo, args.tag
    run_dir = runs_dir / (f"{algo}_formulation_seed{args.seed}_{tag}" + ("_smoke" if args.smoke else ""))
    run_id = f"{algo}s{args.seed}_{tag}"
    pinned = (algo in OFF_POLICY) and (_pin_supported() if pin is None else pin)
    argv = ["--config", CONFIG.as_posix(), "--algo", algo, "--seed", str(args.seed), "--tag", tag,
            "--eval-safety-override", "off"]
    if pinned:
        argv += ["--torch-threads", str(PINNED_THREADS)]
    if args.smoke:
        argv.append("--smoke")
    notes: List[str] = []
    if args.replay_buffer is not None and algo not in OFF_POLICY:
        raise SystemExit(f"--replay-buffer: {algo} has no replay buffer (SAC and TQC only)")
    if args.replay_buffer is not None and not args.resume:
        raise SystemExit("--replay-buffer goes with --resume")

    if not args.resume:
        if (run_dir / "config.json").exists():
            raise SystemExit(f"{run_dir.name} already exists: continue it with --resume, or start a "
                             f"separate run with --tag")
        target = 4_096 if args.smoke else int(args.timesteps or DEFAULT_TIMESTEPS)
        if target <= 0:
            raise SystemExit("--timesteps must be positive")
        if not args.smoke:
            if target != BUDGET:
                argv += ["--train-to", str(target)]
            if algo in OFF_POLICY and target > BUDGET:
                argv += ["--extra-evals", str(BUDGET)]
            if target < BUDGET:
                notes.append(f"{target:,} steps stop partway through the curriculum "
                             f"(stages on the {BUDGET:,}-step schedule)")
        return Plan(run_dir, run_id, argv, pinned, target, notes=notes)

    if not (run_dir / "config.json").exists():
        raise SystemExit(f"nothing to continue: {run_dir} has no config.json")
    steps = _checkpoints(run_dir, algo)
    if not steps:
        raise SystemExit(f"nothing to continue: no {algo}_*_steps.zip checkpoint in {run_dir}")
    budget = int(json.loads((run_dir / "config.json").read_text())["timesteps"])
    target = int(args.timesteps or budget)
    if target < budget:
        raise SystemExit(f"--timesteps {target:,} is below the run's budget {budget:,}; a run "
                         f"cannot be shortened (latest checkpoint {steps[-1]:,})")
    if target == budget and (run_dir / "final_model.zip").exists():
        raise SystemExit(f"{run_dir.name} finished its {budget:,} steps; give --timesteps above "
                         f"{budget:,} to continue past it")
    if target <= steps[-1]:
        raise SystemExit(f"{run_dir.name} has reached {steps[-1]:,} of {budget:,} steps; give "
                         f"--timesteps above {steps[-1]:,} to continue past it")
    argv += ["--resume", (run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir).as_posix()]
    if target > budget:
        argv += ["--extend-to", str(target)]
    if algo in OFF_POLICY:
        if args.replay_buffer is None:
            argv += ["--replay-buffer", "none"]
            notes.append("replay buffer: empty, refilled before updates resume "
                         "(--replay-buffer restores the saved one)")
        else:
            buffer = (Path(args.replay_buffer) if args.replay_buffer != "saved"
                      else _saved_buffer(run_dir, algo, steps[-1]))
            if buffer is None or not buffer.exists():
                raise SystemExit(f"no saved replay buffer for {run_dir.name} at {steps[-1]:,} steps "
                                 f"(train_formulation.REPLAY_BUFFER_DIR); continue without --replay-buffer")
            argv += ["--replay-buffer", str(buffer)]
            notes.append(f"replay buffer: {buffer}")
    return Plan(run_dir, run_id, argv, pinned, target, resume_from=steps[-1], notes=notes)


def _run_name(tokens: List[str]) -> Optional[str]:
    """The run directory a training process works on, from its command line."""
    def value(flag, default=None):
        return tokens[tokens.index(flag) + 1] if flag in tokens[:-1] else default
    script = next((t for t in tokens if t.endswith(("train_formulation.py", "train_asv.py"))), None)
    if script is None:
        return None
    if script.endswith("train_formulation.py") and "--resume" in tokens:
        return Path(value("--resume")).name
    tag = value("--tag", TAG if script.endswith("train_asv.py") else "")
    name = f"{value('--algo', 'ppo')}_formulation_seed{value('--seed', '0')}" + (f"_{tag}" if tag else "")
    return name + ("_smoke" if "--smoke" in tokens else "")


def _other_trainers() -> List[str]:
    try:
        import psutil
    except ImportError:
        return []
    names = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            tokens = proc.info["cmdline"] or []
        except (psutil.Error, OSError):
            continue
        if proc.info["pid"] == os.getpid() or any("multiprocessing" in t for t in tokens):
            continue
        name = _run_name([str(t).replace("\\", "/") for t in tokens])
        if name:
            names.append(name)
    return names


def _install_pinning(tf) -> None:
    """Pin once the workers exist: workers to the efficiency cores, the learner (this
    process) to the performance cores at AboveNormal priority."""
    import psutil
    base = tf.SubprocVecEnv

    class PinnedSubprocVecEnv(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            for proc in self.processes:
                psutil.Process(proc.pid).cpu_affinity(WORKER_CPUS)
            me = psutil.Process()
            me.cpu_affinity(LEARNER_CPUS)
            me.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            print(f"[CPU] learner pinned to CPUs 0-3 (AboveNormal, {PINNED_THREADS} threads), "
                  f"{len(self.processes)} workers to CPUs 4-11", flush=True)

    tf.SubprocVecEnv = PinnedSubprocVecEnv


class _Tee:
    """Console output copied to the run log."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _note(text: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    try:
        with open(MILESTONES, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _millions(steps: int) -> str:
    return f"{steps / 1e6:g} M"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__.split("\n\n", 1)[1].split("\n\nA front")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", required=True, choices=LEARNERS, help="the learner")
    ap.add_argument("--seed", type=int, default=0, help="training seed (default 0)")
    ap.add_argument("--timesteps", type=int, default=None,
                    help=f"total steps; default {DEFAULT_TIMESTEPS:,} for a new run, the run's own budget "
                         "when continuing (a larger value extends it)")
    ap.add_argument("--resume", action="store_true", help="continue the run from its latest checkpoint")
    ap.add_argument("--replay-buffer", nargs="?", const="saved", default=None, metavar="PKL",
                    help="SAC/TQC with --resume: restore the replay buffer saved with the checkpoint "
                         "(or this file); without it the buffer starts empty")
    ap.add_argument("--tag", default=TAG, help=f"run-directory suffix (default {TAG}, the paper runs)")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, then stop")
    ap.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)   # 4,096-step launch check
    args = ap.parse_args()
    os.chdir(ROOT)

    p = plan(args)
    others = _other_trainers()
    if p.run_dir.name in others and not args.dry_run:
        raise SystemExit(f"{p.run_dir.name} is being trained by another process")
    span = (f"{p.resume_from:,} -> {p.target:,}" if p.resume_from is not None else f"0 -> {p.target:,}")
    cpu = (f"pinned, {PINNED_THREADS} threads" if p.pinned else "unpinned, default threads")
    print(f"[train_asv] {p.run_dir.name}: {span} steps; CPU {cpu}", flush=True)
    for note in p.notes:
        print(f"[train_asv] {note}", flush=True)
    if others:
        print(f"[train_asv] also training: {', '.join(sorted(set(others)))} -- both runs will be slower",
              flush=True)
    print("[train_asv] python src/train_formulation.py " + " ".join(p.argv), flush=True)
    if args.dry_run:
        return 0

    sys.path.insert(0, str(ROOT / "src"))
    import train_formulation as tf
    if p.pinned:
        _install_pinning(tf)
    log = open(ROOT / "runs" / f"{p.run_dir.name}.log", "w" if p.resume_from is None else "a",
               buffering=1, encoding="utf-8", errors="replace")
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
    verb = "continuing" if p.resume_from is not None else "training"
    _note(f"== {p.run_id}: {verb} {span} steps via train_asv.py (baseline-v3; CPU {cpu})")
    sys.argv = ["train_formulation.py", *p.argv]
    try:
        tf.main()
    except KeyboardInterrupt:
        _note(f"   {p.run_id}: stopped (Ctrl+C); continue with --resume")
        return 130
    except SystemExit as exc:
        if exc.code not in (None, 0):
            _note(f"   {p.run_id}: stopped ({exc.code})")
        raise
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.close()

    ckpt = p.run_dir / f"{args.algo}_{BUDGET}_steps.zip"
    if ckpt.exists() and not (p.run_dir / f"final_model_{BUDGET}.zip").exists():
        # As the paper runs: the 2.5 M checkpoint kept as the first phase's final model.
        shutil.copy2(ckpt, p.run_dir / f"final_model_{BUDGET}.zip")
        vecnorm = p.run_dir / f"{args.algo}_vecnormalize_{BUDGET}_steps.pkl"
        if vecnorm.exists():
            shutil.copy2(vecnorm, p.run_dir / f"final_vecnormalize_{BUDGET}.pkl")
    _note(f"   {p.run_id}: trained to {_millions(p.target)} -> {p.run_dir.relative_to(ROOT)}")
    rel = p.run_dir.relative_to(ROOT).as_posix()
    print("\nNext steps:")
    if args.tag == TAG and not args.smoke:
        print(f"  evaluations      bash results/train_seed.sh {args.algo} {args.seed} {CONFIG.as_posix()}")
    print(f"  test set v4      python tools/tiers/test_set.py --version 4 --model {rel}/best_model.zip "
          f"--tag {p.run_id} --safety off --processes 4")
    print(f"  final-model check python tools/tiers/test_set.py --version 4 --model {rel}/final_model.zip "
          f"--tag {p.run_id}_final --safety off --processes 4")
    print(f"  watch the policy python test_asv.py --model {rel}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
