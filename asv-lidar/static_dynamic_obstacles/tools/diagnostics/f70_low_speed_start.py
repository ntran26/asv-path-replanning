"""Do low-speed starts break the solved encounter?  (F70)

Run 5 (F68: supervisor off in training, 15 % of episodes starting slow or at
rest) finished at goal 0.76 against run 4's 0.87.  In its last training quarter
being-overtaken episodes reached the goal 0.52 of the time against run 4's 0.80.

04a's backward solve places the target for an own ship already at `U_NOM` on the
path heading at t = 0 (`env.reset` says so).  An own ship starting at rest is
delayed by its acceleration, which moves every CPA: an overtaker solved to pass
clear of a cruising ship can run into a stationary one (A15's floor no longer
holds), and a crossing solved to be escapable (A22) may not be.

This replays the development set (20 per class, supervisor off) starting at
cruise and at rest, under

* `follower` -- hold path and speed; isolates what the start speed does to the
  geometry, with no learning involved;
* run 4's and run 5's final models.

    python tools/diagnostics/f70_low_speed_start.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

from common import development_set, run_pool  # noqa: E402

OUT = ROOT / "results" / "f70_low_speed_start"
MODELS = {"run4": ROOT / "runs" / "ppo_formulation_seed0_v4" / "final_model.zip",
          "run5": ROOT / "runs" / "ppo_formulation_seed0_v5" / "final_model.zip"}
STARTS = {"cruise": {"LOW_SPEED_START_FRAC": 0.0},
          "rest": {"LOW_SPEED_START_FRAC": 1.0, "LOW_SPEED_START_ZERO_SHARE": 1.0}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dev = development_set(20)
    rows = []
    for start, start_overrides in STARTS.items():
        overrides = dict(start_overrides, EMERGENCY_STOP_ENABLED=False)
        for name, policy, model in [("follower", "follower", None)] + [
                (k, "model", v) for k, v in MODELS.items()]:
            jobs = [(b, 900_000 + i, policy, None, {"scenario": i, "start": start, "who": name})
                    for i, b in enumerate(dev)]
            rows += run_pool(jobs, model_path=model, overrides=overrides)
            print(f"{start:6s} {name:8s} done ({time.time() - started:.0f} s)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "episodes.csv", index=False)

    pd.set_option("display.width", 250)
    table = d.groupby(["who", "class", "start"]).agg(
        goal=("outcome", lambda s: (s == "goal").mean()),
        target=("collided_target", "mean"),
        other=("outcome", lambda s: s.isin(["collision:obstacle", "collision:boundary"]).mean()),
    ).round(2).unstack("start")
    overall = d.groupby(["who", "start"]).agg(
        goal=("outcome", lambda s: (s == "goal").mean()),
        target=("collided_target", "mean")).round(3).unstack("start")
    text = (f"F70 -- development set, supervisor off, start at cruise vs rest "
            f"({len(d)} episodes, {time.time() - started:.0f} s)\n\n== overall\n{overall}\n\n"
            f"== per class\n{table}\n")
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
