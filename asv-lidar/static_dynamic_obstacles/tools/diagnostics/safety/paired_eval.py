"""Single-process development evaluation; kept SAC, original per-case seeds.

python -B tools/diagnostics/safety/paired_eval.py off,v4,v5 tag
V2_SET / V3_SET / V4_SET / V5_SET accept comma-separated NAME=Python-literal overrides.
Historical results stay untouched. Use --cases for a development subset.
"""
from __future__ import annotations

import argparse
import ast
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modes")
    parser.add_argument("tag")
    parser.add_argument("--cases", default="")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if not set(modes) <= {"off", "v2", "v3", "v4", "v5", "oracle3", "oracle4", "oracle5"}:
        parser.error("modes must be off,v2,v3,v4,v5,oracle3,oracle4,oracle5")
    if Path(args.tag).name != args.tag:
        parser.error("tag must be a filename, without directories")
    output = ROOT / "results" / "safety_dev" / f"dev_{args.tag}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from common import load_model, run_episode
    from env import ASVLidarEnv
    import safety_v2
    import safety_v3
    modules = {"V2": safety_v2, "V3": safety_v3}
    if any(m.endswith(("4", "5")) for m in modes):
        import safety_v4
        modules["V4"] = safety_v4
    if any(m.endswith("5") for m in modes):
        import safety_v5
        modules["V5"] = safety_v5
    overrides = {}
    for prefix, module in modules.items():
        overrides[prefix] = {}
        for item in filter(None, os.environ.get(prefix + "_SET", "").split(",")):
            name, value = item.split("=", 1)
            if not hasattr(module, name) or not name.isupper():
                parser.error(f"unknown {prefix} setting: {name}")
            value = ast.literal_eval(value)
            setattr(module, name, value)
            overrides[prefix][name] = value
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from prediction_audit import load_development_cases, development_cache_digest
    checkpoint = ROOT / "runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip"
    source_paths = [ROOT / f for f in
                   ("src/constants.py", "src/env.py", "src/safety_v2.py", "src/safety_v3.py",
                    "src/safety_v4.py", "src/safety_v5.py", "src/safety_recovery.py", "src/safety_feedback.py",
                    "src/safety_prediction.py", "src/safety_perception.py", "src/safety_observer.py",
                    "src/classical/common.py", "src/formulation_v3.py", "src/ship.py",
                    "bluefin/dynamics.py", "bluefin/ship_model_v3.py",
                    "tools/diagnostics/safety/oracle.py", "tools/diagnostics/safety/paired_eval.py",
                    "tools/diagnostics/safety/prediction_audit.py")]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths if p.exists()}
    effective = {prefix: {k: repr(v) for k, v in vars(module).items() if k.isupper()}
                 for prefix, module in modules.items()}
    metadata = {"checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "model_config_sha256": hashlib.sha256(checkpoint.with_name("config.json").read_bytes()).hexdigest(),
                "episode_seed_base": 900120,
                "modes": modes, "overrides": overrides, "torch_threads": 1,
                "effective_module_constants": effective, "sources": source_hashes,
                "source_hash_time": "before scenario generation and model loading",
                "scenario_cache_digest": development_cache_digest(),
                "python": sys.version, "torch": torch.__version__}
    with output.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    selected = load_development_cases(set(filter(None, args.cases.split(","))) or None)
    metadata["cases"] = [{"case": b.case_id, "reset_seed": 900120 + i,
                           "scenario_sha256": b.digest()} for i, b in selected]
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    model = load_model(checkpoint)

    class EvalEnv(ASVLidarEnv):
        oracle = False

        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            if self.oracle:
                from oracle import install_oracle
                if cfg.SAFETY_VERSION == 5:
                    from safety_v5 import SafetyFilterV5
                    safety = SafetyFilterV5()
                elif cfg.SAFETY_VERSION == 4:
                    from safety_v4 import SafetyFilterV4
                    safety = SafetyFilterV4()
                else:
                    safety = safety_v3.SafetyFilterV3()
                self._safety_v2 = install_oracle(safety)
            return result

    rows = []
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", "case", "enc", "outcome", "v2", "estops"])
        writer.writeheader()
        for mode in modes:
            env = EvalEnv(render_mode=None, emergency_stop=mode != "off")
            env.oracle = mode.startswith("oracle")
            cfg.SAFETY_VERSION = int(mode[-1]) if mode != "off" else 1
            started = time.perf_counter()
            try:
                for j, (i, b) in enumerate(selected):
                    result = run_episode(env, b, 900120 + i, "model", model)
                    row = dict(mode=mode, case=b.case_id, enc=b.case_id.split("-")[1],
                               outcome=result["outcome"], v2=result["safety_v2_steps"],
                               estops=result.get("safety_v2_brake_steps", 0))
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    print(f"{mode} {j + 1}/{len(selected)} {b.case_id}: {row['outcome']}", flush=True)
            finally:
                env.close()
            print(f"{mode}: {(time.perf_counter() - started) / len(selected):.2f} s/episode", flush=True)
    import pandas as pd
    data = pd.DataFrame(rows)
    print(data.groupby("mode").outcome.value_counts().unstack(fill_value=0))
    print("Saved", output)


if __name__ == "__main__":
    main()
