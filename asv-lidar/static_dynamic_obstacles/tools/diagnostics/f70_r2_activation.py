"""How often does R-2's 8(e) slowdown carve-out fire, under each slowdown test?  (F70)

F68 moved R-2's "does slowing clear" question from the supervisor's full-astern
stop (`stop_clears`, A23) to the policy's own slowdown modelled as a coast at
RPM 0 (`slowdown_clears`).  A coast from cruise is still doing most of its
speed after the target has passed, so its DCPA is close to holding course: it
may "clear" only where the ship was already clear, and almost never where a
slowdown is the lawful answer.

Replays the development crossings and head-ons (supervisor off, run 5's final
model, deterministic) and, at every frame where a give-way context is engaged
with its compliant alteration inadmissible, records the current DCPA and both
tests.

    python tools/diagnostics/f70_r2_activation.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import train_formulation as tf  # noqa: E402
from common import development_set  # noqa: E402

OUT = ROOT / "results" / "f70_low_speed_start"
MODEL = ROOT / "runs" / "ppo_formulation_seed0_v5" / "final_model.zip"


def main() -> None:
    from stable_baselines3 import PPO
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None)
    env.estop_enabled = False
    model = PPO.load(str(MODEL), device="cpu")
    dev = development_set(20, classes=("crossing", "head_on"))
    rows = []
    for i, built in enumerate(dev):
        obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
        step = 0
        while True:
            for ctx in env.encounter_contexts.values():
                if ctx.engaged and ctx.gives_way and not ctx.turn_admissible and ctx.tcpa > 0.0:
                    rows.append({"episode": i, "class": built.encounter_class, "step": step,
                                 "u_own": ctx.u_own, "dcpa": ctx.dcpa, "tcpa": ctx.tcpa,
                                 "dcpa_if_slowed": ctx.dcpa_if_slowed,
                                 "dcpa_if_stopped": ctx.dcpa_if_stopped,
                                 "slowdown_clears": ctx.slowdown_clears,
                                 "stop_clears": ctx.stop_clears,
                                 "already_clear": ctx.dcpa >= cfg.ESTOP_CLEAR_DCPA_M})
            action = model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, _ = env.step(action)
            step += 1
            if term or trunc:
                break
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "r2_frames.csv", index=False)
    pd.set_option("display.width", 200)
    by = d.groupby("class").agg(
        frames=("step", "size"), episodes=("episode", "nunique"),
        already_clear=("already_clear", "mean"),
        slowdown_clears=("slowdown_clears", "mean"), stop_clears=("stop_clears", "mean")).round(2)
    unclear = d[~d.already_clear]
    by_unclear = unclear.groupby("class").agg(
        frames=("step", "size"),
        slowdown_clears=("slowdown_clears", "mean"), stop_clears=("stop_clears", "mean"),
        dcpa=("dcpa", "mean"), dcpa_if_slowed=("dcpa_if_slowed", "mean"),
        dcpa_if_stopped=("dcpa_if_stopped", "mean")).round(2)
    text = (f"R-2 candidate frames (give-way, engaged, alteration inadmissible), run 5 final, supervisor off\n\n"
            f"== all candidate frames\n{by}\n\n== frames not already clear (DCPA < {cfg.ESTOP_CLEAR_DCPA_M} m)\n"
            f"{by_unclear}\n")
    (OUT / "r2_activation.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
