"""Shared set-up for the C14 narrow head-on diagnostics (PROJECT_STATE F54, F55).

The same 100 head-on scenarios -- 20 per width at 5, 6, 7, 8 and 10 m, from the
development namespace -- and the run 2 final model, so every C14 script
replays identical encounters.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import curriculum  # noqa: E402
import scenario as scn  # noqa: E402
import train_formulation as tf  # noqa: E402

OUT = ROOT / "results" / "c14_narrow_head_on"
RUN2_MODEL = ROOT / "runs" / "ppo_formulation_seed0_v2" / "final_model.zip"
WIDTHS = (5.0, 6.0, 7.0, 8.0, 10.0)
PER_WIDTH = 20
WIDTH_BINS = ([0, 5.5, 6.5, 7.5, 9.0, 11.0], ["5", "6", "7", "8", "10"])


def head_on_scenarios():
    """The C14 scenario set, deterministic from the development seeds."""
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    gen = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    out = []
    for w in WIDTHS:
        j = found = 0
        while found < PER_WIDTH and j < 20 * PER_WIDTH:
            built = gen.sample(scn.seed_for("development", 300_000 + int(w * 10) * 1000 + j),
                               encounter_class="head_on", width=w)
            j += 1
            if built is not None:
                out.append(built)
                found += 1
    return out


def load_run2_model():
    from stable_baselines3 import PPO
    return PPO.load(RUN2_MODEL, device="cpu")
