"""Where the baseline runs stand: one line per planned learner x seed (A26).

    python tools/run_status.py

Runs are trained on demand, one at a time (`bash results/train_seed.sh <algo>
<seed>`); this lists which are done, which were started and stopped part-way
(with the last checkpoint), which are not started, each run's best
development-set score, and whether its frozen suite has been run.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CONFIG = ROOT / "configs" / "baseline_v2.json"


def _run_dir(algo: str, seed: int, tag: str) -> Path:
    return RUNS / f"{algo}_formulation_seed{seed}_{tag}"


def _steps(run: Path, algo: str) -> int:
    log = run.with_suffix(".log")
    if log.exists():
        hits = re.findall(r"total_timesteps\s*\|\s*(\d+)", log.read_text(errors="ignore")[-20_000:])
        if hits:
            return int(hits[-1])
    ckpts = [int(m.group(1)) for p in run.glob(f"{algo}_*_steps.zip")
             if (m := re.match(rf"{algo}_(\d+)_steps\.zip", p.name))]
    return max(ckpts, default=0)


def main() -> int:
    config = json.loads(CONFIG.read_text())
    total = int(config["run_args"]["timesteps"])
    rows, done = [], 0
    for algo in config["campaign"]["algos"]:
        for seed in config["campaign"]["seeds"]:
            run = _run_dir(algo, seed, config["campaign"].get("tag", "bl2"))
            best = ""
            summary = run / "eval_summary.json"
            if summary.exists():
                hist = [h for h in json.loads(summary.read_text()) if h.get("supervisor", "off") == "off"]
                if hist:
                    top = max(hist, key=lambda h: h["goal_rate"] - 2 * h["collision_rate"])
                    best = f"best dev goal {top['goal_rate']:.2f} @ {top['timesteps'] / 1e6:.1f} M"
            if (run / "final_model.zip").exists():
                state, done = "done", done + 1
            elif run.exists():
                # Started: either training now or stopped; the log's last
                # step count says how far it got either way.
                state = f"started {_steps(run, algo) / total:5.1%}"
            else:
                state = "not started"
            tag = config["campaign"].get("tag", "bl2")
            frozen = ROOT / "results" / "frozen_suite" / f"{algo}s{seed}_{tag}" / "summary.txt"
            evaluated = "frozen suite done" if frozen.exists() else ""
            rows.append(f"  {algo:<14} seed {seed}  {state:<16} {best:<28} {evaluated}")
    print(f"{config['id']}: {done} of {len(rows)} planned runs done")
    print("\n".join(rows))
    log = ROOT / "results" / "train_seed.log"
    if log.exists():
        print("\nlast log lines:\n  " + "\n  ".join(log.read_text().splitlines()[-4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
