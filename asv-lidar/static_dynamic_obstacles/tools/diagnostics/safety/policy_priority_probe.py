"""Two original development counterfactuals, charged to the quick 100-run cap.

    python -B tools/diagnostics/safety/policy_priority_probe.py --tag dev_priority_v1

Uses the recorded development seeds, shared attempt tokens and a fresh output
directory. No automatic retries. This is a selected failure diagnostic, not an
overall success-rate estimate. A campaign STOP file cancels between episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers")]
CASES = {"DV3-CRP-VS-04", "DV3-CRS-CV-04"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in (".", "..") or "\\" in args.tag:
        parser.error("Use a plain tag")
    # Pin the exact helper bytes used even if a reporting-only helper is being
    # developed concurrently. Never import a partly different helper revision.
    helper_path = Path(__file__).with_name("quick_campaign.py")
    helper_bytes = helper_path.read_bytes()
    quick = types.ModuleType("quick_probe_budget_snapshot")
    quick.__file__ = str(helper_path)
    exec(compile(helper_bytes, str(helper_path), "exec"), quick.__dict__)
    if (quick.CAMPAIGN / "STOP").exists():
        raise RuntimeError("Campaign STOP marker is present; no episodes started")

    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from prediction_audit import load_development_cases, runtime_overrides
    from suite_eval import settings_snapshot
    from common import load_model, run_episode
    from env import ASVLidarEnv

    baseline = json.loads(quick.read_shared_bytes(quick.CAMPAIGN / "frozen_baseline.json"))
    quick.verify_reference_sources(baseline)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    model_path = Path(baseline["settings"]["checkpoint"])
    config = json.loads(model_path.with_name("config.json").read_text())
    for key, value in (config.get("constant_overrides") or {}).items():
        setattr(cfg, key, value)
    runtime_overrides()
    settings = settings_snapshot(model_path, ["dev_field"], ["v5"])
    for key in ("checkpoint_sha256", "config_sha256", "effective_constants", "runtime_versions"):
        if settings[key] != baseline["settings"][key]:
            raise ValueError(f"Reference {key} changed")
    for mode in ("v2", "v3", "v4"):
        if settings["effective_safety_constants"][mode] != baseline["settings"]["effective_safety_constants"][mode]:
            raise ValueError(f"Reference {mode} settings changed")
    selected = load_development_cases(CASES)
    inventory = json.loads(quick.read_shared_bytes(quick.INVENTORY))
    canonical = {c["case"]: c for c in inventory["cases"] if c["suite"] == "dev_field"}
    for index, built in selected:
        if (canonical[built.case_id]["seed"] != 900120 + index
                or canonical[built.case_id]["scenario_sha256"] != built.digest()):
            raise ValueError("Original development scene/seed mismatch")
    if {built.case_id for _, built in selected} != CASES:
        raise ValueError("Expected exactly the two declared development cases")
    metadata = {"settings": settings, "cases": [canonical[b.case_id] for _, b in selected],
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "budget_helper_sha256": hashlib.sha256(helper_bytes).hexdigest(),
                "scope": "Two original DV3 loss counterfactuals; selected-failure diagnostic"}
    directory = quick.CAMPAIGN / "runs" / args.tag
    directory.mkdir(parents=True, exist_ok=False)
    quick.write_json(directory / "metadata.json", metadata)
    lock = quick.CAMPAIGN / "active_run.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "tag": args.tag}, handle)
    try:
        model = load_model(model_path)
        cfg.SAFETY_VERSION = 5
        class AuditEnv(ASVLidarEnv):
            def reset(self, *args, **kwargs):
                result = super().reset(*args, **kwargs)
                self.priority_counts = {name: 0 for name in
                    ("verified_policy_preserved", "continuation_override_prevented",
                     "policy_feedback_evaluated", "policy_feedback_preserved")}
                return result
            def step(self, action):
                result = super().step(action)
                last = getattr(getattr(self, "_safety_v2", None), "last", {})
                for name in self.priority_counts:
                    self.priority_counts[name] += int(bool(last.get(name, False)))
                return result
        rows = []
        for index, built in selected:
            if (quick.CAMPAIGN / "STOP").exists():
                raise RuntimeError("Campaign stopped before next development episode")
            identity = {"tag": args.tag, "mode": "v5", "suite": "dev_field", "case": built.case_id,
                        "seed": 900120 + index, "scenario_sha256": built.digest(),
                        "run_metadata_sha256": quick.canonical_sha(metadata)}
            number = quick.reserve_attempt(quick.CAMPAIGN / "attempts", identity)
            env = None
            try:
                env = AuditEnv(render_mode=None, emergency_stop=True)
                started = time.perf_counter()
                result = run_episode(env, built, 900120 + index, "model", model)
                row = dict(result, **identity, attempt=number, seconds=time.perf_counter()-started,
                           **env.priority_counts)
                quick.write_json(directory / f"episode_{number:03d}.json", row)
                rows.append(row)
                print(f"Budget {number}/100; {built.case_id}: {row['outcome']}; "
                      f"prevented={row['continuation_override_prevented']}; "
                      f"policy_feedback={row['policy_feedback_preserved']}", flush=True)
            except BaseException as exc:
                quick.write_json(directory / f"error_{number:03d}.json", dict(identity, attempt=number, error=repr(exc)))
                raise
            finally:
                if env is not None:
                    env.close()
        quick.write_json(directory / "complete.json", {"completed": len(rows), "rows": rows})
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
