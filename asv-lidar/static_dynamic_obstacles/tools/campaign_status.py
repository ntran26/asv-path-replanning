"""Where the baseline campaign stands (A26): one line per learner x seed.

    python tools/campaign_status.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CONFIG = ROOT / "configs" / "baseline_v1.json"


def _run_dir(algo: str, seed: int) -> Path:
    if algo == "ppo" and seed <= 1:
        return RUNS / f"ppo_formulation_seed{seed}_v11"      # run 11 is PPO seeds 0-1
    return RUNS / f"{algo}_formulation_seed{seed}_bl1"


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
            run = _run_dir(algo, seed)
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
                state = f"training {_steps(run, algo) / total:5.1%}"
            else:
                state = "pending"
            rows.append(f"  {algo:<14} seed {seed}  {state:<16} {best}")
    stop = " (stop file present: the campaign halts after the current run)" \
        if (RUNS / "CAMPAIGN_STOP").exists() else ""
    print(f"baseline-v1 campaign: {done} of {len(rows)} runs done{stop}")
    print("\n".join(rows))
    log = ROOT / "results" / "baseline_campaign.log"
    if log.exists():
        print("\nlast log lines:\n  " + "\n  ".join(log.read_text().splitlines()[-4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
