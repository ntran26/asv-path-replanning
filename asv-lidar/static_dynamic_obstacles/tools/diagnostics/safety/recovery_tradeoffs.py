"""Record soft-recovery tradeoffs without changing the chosen commands.

From the project root, after an evaluation slot is available::

    python -B tools/diagnostics/safety/recovery_tradeoffs.py \
        --cases DV3-HO-CV-03,DV3-OT-CV-11,DV3-OT-CV-18,DV3-OT-VS-03,DV3-BO-CV-19 \
        --tag soft_regression_tradeoffs

Only the explicitly supplied development cases are loaded. The v5 settings are
ONE_DECISION_COMMIT=False, SOFT_RECOVERY=True, FEEDBACK_BACKUPS=False. Each call
to safety_recovery.choose is delegated unchanged to the original function.
Predictions are inspected, not recomputed, and no alternative is issued.

One process and one Torch thread are used. Outputs under results/safety_dev:
<tag>_metadata.json, <tag>_episodes.csv, <tag>_decisions.csv, and
<tag>_choices.jsonl. The latter preserves every complete-plan cost and command
for independently checking the reported zero-prefix alternatives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DECISION_FIELDS = (
    "mode", "case", "seed", "step", "time_s", "filter_mode_before", "recovery_steps_before",
    "sequence_count", "continuation_present", "selected_index", "selected_continuation",
    "selected_rudder", "selected_throttle", "selected_brake", "selected_first_s",
    "selected_clearance_m", "selected_violation", "selected_immediate_violation",
    "zero_prefix_count", "avoidable_prefix_violation", "best_zero_index",
    "best_zero_rudder", "best_zero_throttle", "best_zero_brake", "best_zero_violation",
    "best_zero_first_s", "best_zero_clearance_m", "best_zero_continuation",
    "minimum_immediate_violation", "minimum_total_violation", "outcome_after_step",
    "issued_rudder", "issued_rpm",
)
EPISODE_FIELDS = ("mode", "case", "seed", "outcome", "steps", "safety_v2_steps",
                  "safety_v2_brake_steps", "seconds", "soft_decisions", "avoidable_prefix_decisions")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", required=True, help="comma-separated development case IDs")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip")
    args = parser.parse_args()
    requested = list(filter(None, args.cases.split(",")))
    if not requested or len(set(requested)) != len(requested):
        parser.error("--cases must contain distinct, nonempty development case IDs")
    if Path(args.tag).name != args.tag or args.tag in (".", "..") or "\\" in args.tag:
        parser.error("--tag must be a plain filename stem")
    output = ROOT / "results" / "safety_dev"
    paths = {name: output / f"{args.tag}_{suffix}" for name, suffix in
             (("metadata", "metadata.json"), ("episodes", "episodes.csv"),
              ("decisions", "decisions.csv"), ("choices", "choices.jsonl"))}
    if any(path.exists() for path in paths.values()):
        parser.error("Refusing to overwrite diagnostic outputs; use a new --tag")

    import numpy as np
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from common import load_model, run_episode
    from env import ASVLidarEnv
    import safety_recovery
    import safety_v2
    import safety_v3
    import safety_v4
    import safety_v5
    from prediction_audit import development_cache_digest, load_development_cases

    curriculum.apply_stage(tf.PROPULSION_STAGE)
    # Match the completed soft-only ablation. Deliberately do not consume V*_SET
    # from the shell: this diagnostic must retain these three exact switches.
    safety_v5.ONE_DECISION_COMMIT = False
    safety_v5.SOFT_RECOVERY = True
    safety_v5.FEEDBACK_BACKUPS = False
    cfg.SAFETY_VERSION = 5
    sources = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "bluefin").glob("*.py"))
    sources += [Path(__file__), ROOT / "tools/tiers/common.py",
                ROOT / "tools/diagnostics/safety/prediction_audit.py"]
    metadata = {
        "checkpoint": str(args.model.resolve()), "checkpoint_sha256": sha(args.model),
        "config_sha256": sha(args.model.with_name("config.json")),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "source_hash_time": "before scenario generation and model loading",
        "scenario_cache_digest": development_cache_digest(),
        "requested_cases": requested, "mode": "v5", "episode_seed_base": 900120,
        "effective_safety_constants": {
            name: {k: repr(v) for k, v in vars(module).items() if k.isupper()}
            for name, module in (("v2", safety_v2), ("v3", safety_v3),
                                 ("v4", safety_v4), ("v5", safety_v5))},
        "effective_constants": {k: repr(v) for k, v in vars(cfg).items() if k.isupper()},
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "threads": 1, "processes": 1, "status": "building",
        "choice_behavior": "original safety_recovery.choose called unchanged",
        "prefix_interval_s": cfg.UPDATE_RATE,
        "prefix_definition": "zero integrated clearance deficit over the next issued control interval",
        "step_index": "zero-based decision index before env.step",
    }
    output.mkdir(parents=True, exist_ok=True)
    with paths["metadata"].open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    selected = load_development_cases(set(requested))
    metadata["cases"] = [{"case": built.case_id, "seed": 900120 + index,
                           "scenario_sha256": built.digest()} for index, built in selected]
    metadata["status"] = "running"
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    model = load_model(args.model)
    original_choose = safety_recovery.choose
    context, pending = {}, []

    def instrumented_choose(sequences, metrics, policy_reference, **kwargs):
        chosen = original_choose(sequences, metrics, policy_reference, **kwargs)
        env = context["env"]
        filter_ = env._safety_v2
        continuation = filter_.plan is not None and len(filter_.plan) > 1
        zero = np.flatnonzero(metrics.immediate_violation == 0.0)
        commands = np.asarray(sequences)[:, 0]
        throttle = np.where(np.isnan(commands[:, 1]), -1.5, commands[:, 1])
        reference = np.asarray(policy_reference)
        distance = ((commands[:, 0] - reference[0]) ** 2
                    + safety_v2.W_THROTTLE * (throttle - reference[1]) ** 2)
        best_zero = (int(zero[np.lexsort((zero, distance[zero], metrics.violation[zero]))[0]])
                     if len(zero) else None)

        def command_value(index, column):
            if index is None or np.isnan(commands[index, column]):
                return ""
            return float(commands[index, column])

        row = {"mode": "v5", "case": context["case"], "seed": context["seed"],
               "step": context["step"], "time_s": float(env.elapsed_time),
               "filter_mode_before": context["filter_mode_before"],
               "recovery_steps_before": context["recovery_steps_before"],
               "sequence_count": len(sequences), "continuation_present": continuation,
               "selected_index": int(chosen),
               "selected_continuation": bool(continuation and chosen == len(sequences) - 1),
               "selected_rudder": command_value(chosen, 0),
               "selected_throttle": command_value(chosen, 1),
               "selected_brake": bool(np.isnan(commands[chosen, 1])),
               "selected_first_s": float(metrics.first[chosen]),
               "selected_clearance_m": float(metrics.clear[chosen]),
               "selected_violation": float(metrics.violation[chosen]),
               "selected_immediate_violation": float(metrics.immediate_violation[chosen]),
               "zero_prefix_count": len(zero),
               "avoidable_prefix_violation": bool(metrics.immediate_violation[chosen] > 0.0 and len(zero)),
               "best_zero_index": "" if best_zero is None else best_zero,
               "best_zero_rudder": command_value(best_zero, 0),
               "best_zero_throttle": command_value(best_zero, 1),
               "best_zero_brake": "" if best_zero is None else bool(np.isnan(commands[best_zero, 1])),
               "best_zero_violation": "" if best_zero is None else float(metrics.violation[best_zero]),
               "best_zero_first_s": "" if best_zero is None else float(metrics.first[best_zero]),
               "best_zero_clearance_m": "" if best_zero is None else float(metrics.clear[best_zero]),
               "best_zero_continuation": bool(continuation and best_zero == len(sequences) - 1),
               "minimum_immediate_violation": float(metrics.immediate_violation.min()),
               "minimum_total_violation": float(metrics.violation.min())}
        full = {"decision": row, "policy_reference": reference.tolist(), "choose_kwargs": kwargs,
                "commands": commands.tolist(), "first": metrics.first.tolist(),
                "clear": metrics.clear.tolist(), "violation": metrics.violation.tolist(),
                "immediate_violation": metrics.immediate_violation.tolist(),
                "policy_distance": distance.tolist()}
        pending.append((row, full))
        return chosen

    episode_rows, decision_count, avoidable_count = [], 0, 0
    with paths["episodes"].open("x", newline="", encoding="utf-8") as episode_stream, \
            paths["decisions"].open("x", newline="", encoding="utf-8") as decision_stream, \
            paths["choices"].open("x", encoding="utf-8") as choice_stream:
        episode_writer = csv.DictWriter(episode_stream, fieldnames=EPISODE_FIELDS)
        decision_writer = csv.DictWriter(decision_stream, fieldnames=DECISION_FIELDS)
        episode_writer.writeheader()
        decision_writer.writeheader()

        class AuditEnv(ASVLidarEnv):
            def step(self, action):
                nonlocal decision_count, avoidable_count
                filter_ = self._safety_v2
                context["filter_mode_before"] = getattr(filter_, "mode", "uninitialized")
                context["recovery_steps_before"] = getattr(filter_, "recovery_steps", 0)
                pending.clear()
                result = super().step(action)
                info = result[-1]
                outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
                           "goal" if info["reached_goal"] else "timeout" if result[3] else "ongoing")
                for row, full in pending:
                    row.update(outcome_after_step=outcome, issued_rudder=float(self.rudder) / 100.0,
                               issued_rpm=float(self.rpm))
                    choice_stream.write(json.dumps(full, allow_nan=True) + "\n")
                    choice_stream.flush()
                    decision_writer.writerow(row)
                    decision_stream.flush()
                    decision_count += 1
                    avoidable_count += int(row["avoidable_prefix_violation"])
                context["step"] += 1
                return result

        env = AuditEnv(render_mode=None, emergency_stop=True)
        safety_recovery.choose = instrumented_choose
        try:
            for index, built in selected:
                context.update(env=env, case=built.case_id, seed=900120 + index, step=0)
                started = time.perf_counter()
                before_count, before_avoidable = decision_count, avoidable_count
                result = run_episode(env, built, 900120 + index, "model", model)
                row = {key: result[key] for key in
                       ("outcome", "steps", "safety_v2_steps", "safety_v2_brake_steps")}
                row.update(mode="v5", case=built.case_id, seed=900120 + index,
                           seconds=time.perf_counter() - started,
                           soft_decisions=decision_count - before_count,
                           avoidable_prefix_decisions=avoidable_count - before_avoidable)
                episode_writer.writerow(row)
                episode_stream.flush()
                episode_rows.append(row)
                print(json.dumps(row), flush=True)
            metadata["status"] = "complete"
        except BaseException as exc:
            metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            metadata["error"] = repr(exc)
            raise
        finally:
            safety_recovery.choose = original_choose
            env.close()
            metadata.update(completed_cases=len(episode_rows), soft_decisions=decision_count,
                            avoidable_prefix_decisions=avoidable_count)
            paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {decision_count} soft decisions; {avoidable_count} had a zero-prefix alternative.", flush=True)


if __name__ == "__main__":
    main()
