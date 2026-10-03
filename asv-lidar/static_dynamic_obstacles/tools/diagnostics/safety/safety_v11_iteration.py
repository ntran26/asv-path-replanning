"""V11 development runner derived from the frozen V10 campaign runner.

Example: python -B tools/diagnostics/safety/safety_v11_iteration.py --tag persistence_three --modes v11
A separate runner keeps the active V10/V9 ablations byte-for-byte frozen.
Each tag is new, freezes its selection/code/settings, and refuses automatic retries.
No old campaign is modified. A STOP marker stops before the next episode.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers")]
from v9_paired_pilot import MODEL, EXPECTED_MODEL, jsonable, load_scenes, sha, write_json

CAMPAIGN = ROOT / "results/safety_dev/v10_iterations"
DEFAULT_SELECTION = ROOT / "results/safety_dev/v9_paired_pilot/selection.json"
VARIANTS = {
    "off": (1, None, {}),
    "v8": (8, "safety_v8", {}),
    "v9": (9, "safety_v9", {}),
    "v9_margin": (9, "safety_v9", {"require_current_plan": False, "prefer_policy_margin": True}),
    "v9_check": (9, "safety_v9", {"require_current_plan": True, "prefer_policy_margin": False}),
    "v10": (10, "safety_v10", {}),
    "v11": (11, "safety_v11", {}),
    "v11_prefix": (11, "safety_v11", {"enable_admission": False, "enable_persistence": False, "policy_prefix_search": True}),
    "v11_combined": (11, "safety_v11", {"policy_prefix_search": True}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--cases", default="", help="Comma-separated canonical prefixed IDs")
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--filter-kwargs", default="{}", help="Explicit constructor overrides, recorded for all enabled modes")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if Path(args.tag).name != args.tag or not args.tag or len(set(modes)) != len(modes) or not set(modes) <= set(VARIANTS):
        parser.error("Use a unique simple tag and registered modes")
    kwargs = json.loads(args.filter_kwargs)
    if not isinstance(kwargs, dict):
        parser.error("filter-kwargs must be an object")
    selection_raw = args.selection.read_bytes()
    selection = json.loads(selection_raw)
    requested = set(filter(None, args.cases.split(",")))
    cases = [r for r in selection["cases"] if not requested or r["case"] in requested]
    if not cases or len({r["case"] for r in cases}) != len(cases) or requested - {r["case"] for r in cases}:
        parser.error("Empty, duplicate or unknown scenario selection")
    out = CAMPAIGN / args.tag
    if out.exists():
        parser.error("Existing tag; no automatic retries or result reuse")

    import torch
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
    except ImportError:
        limits = None
        threadpool_info = lambda: []
    else:
        limits = threadpool_limits(limits=1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from common import load_model, run_episode
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    config = json.loads(MODEL.with_name("config.json").read_text())
    for name, value in (config.get("constant_overrides") or {}).items():
        setattr(cfg, name, value)
    if sha(MODEL.read_bytes()) != EXPECTED_MODEL:
        raise ValueError("Checkpoint identity changed")
    classes, options = {}, {}
    for mode in modes:
        version, module, defaults = VARIANTS[mode]
        classes[mode] = None if module is None else getattr(importlib.import_module(module), f"SafetyFilterV{version}")
        options[mode] = dict(defaults, **kwargs) if module else {}
        if module:
            classes[mode](**options[mode])  # Validate configuration without an episode.
    capture = importlib.import_module("trace_snapshot") if args.snapshots else None
    scenes, cache_hashes = load_scenes(cases)
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "bluefin").rglob("*.py"))
    # V11 inherits V10: archive its parent even though the mode name differs.
    # V8/V9-only modes do not import that parent.
    if not any(VARIANTS[mode][0] >= 10 for mode in modes):
        paths = [p for p in paths if p.name != "safety_v10.py"]
    paths += [Path(__file__), Path(__file__).with_name("v9_paired_pilot.py"), ROOT / "tools/tiers/common.py"]
    if capture:
        paths.append(Path(capture.__file__))
    sources = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    source_hashes = {name: sha(raw) for name, raw in sources.items()}
    out.mkdir(parents=True)
    with zipfile.ZipFile(out / "evaluated_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sources.items():
            archive.writestr(name, raw)
    manifest = dict(prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        authorization="Keep improving and testing until100% safe and no policy-success regressions",
        selection_source=str(args.selection), selection_sha256=sha(selection_raw), cases=cases,
        modes=modes, planned_runs=len(cases)*len(modes), process_id=os.getpid(), processes=1,
        filter_classes={m: None if classes[m] is None else classes[m].__module__ + "." + classes[m].__name__ for m in modes},
        constructor_options=options, filter_installation="Explicit instance immediately after native reset",
        snapshots=args.snapshots, checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=EXPECTED_MODEL,
        config_sha256=sha(MODEL.with_name("config.json").read_bytes()), source_sha256=source_hashes,
        cache_sha256=cache_hashes, threadpools=threadpool_info(), torch_threads=torch.get_num_threads(),
        native_thread_environment={k:os.environ[k] for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")},
        constants={k:repr(v) for k,v in vars(cfg).items() if k.isupper()}, low_speed_start_frac=0.0,
        scope="Development only; no claimed population estimate or formal safety guarantee")
    write_json(out / "manifest.json", manifest)
    (out / "attempts").mkdir()
    (out / "traces").mkdir()
    model = load_model(MODEL)

    class TraceEnv(ASVLidarEnv):
        def reset(self, **reset_kwargs):
            result = super().reset(**reset_kwargs)
            self._safety_v2 = classes[self.mode](**options[self.mode]) if self.estop_enabled else None
            self.last_observation = result[0]
            return result

        def step(self, action):
            before = [self.asv_x, self.asv_y, self.asv_h, self.u_body]
            diagnostic_before = capture.capture_before(self) if capture else None
            observation = self.last_observation
            start = time.perf_counter()
            result = super().step(action)
            seconds = time.perf_counter() - start
            filt = self._safety_v2
            if self.estop_enabled and type(filt) is not classes[self.mode]:
                raise AssertionError("Unexpected filter class")
            details = getattr(filt, "last", {})
            self.why_counts[details.get("why", "idle")] += 1
            self.steps_logged += 1
            record = dict(step=self.steps_logged, step_seconds=seconds, observation=observation,
                pre_state=before, post_state=[self.asv_x,self.asv_y,self.asv_h,self.u_body],
                policy_action=action, rudder_command=self.rudder/100., signed_rpm_command=self.rpm,
                changed=bool(getattr(self,"safety_v2_changed",False)), brake=bool(getattr(self,"_v2_brake",False)),
                filter=details, plan=getattr(filt,"plan",None))
            if capture:
                record["diagnostic_before"] = diagnostic_before
                record["diagnostic_decision"] = capture.capture_decision(self)
            self.trace.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
            self.last_observation = result[0]
            return result

    count, results = 0, []
    started = time.perf_counter()
    for index, case in enumerate(cases):
        for mode in modes[index % len(modes):] + modes[:index % len(modes)]:
            if (out / "STOP").exists() or (CAMPAIGN / "STOP").exists():
                print("STOP marker; completed records preserved", flush=True)
                return
            for name, expected in source_hashes.items():
                if sha((ROOT / name).read_bytes()) != expected:
                    raise RuntimeError(f"Frozen source changed: {name}")
            count += 1
            identity = dict(attempt=count,case=case["case"],mode=mode,seed=int(case["seed"]))
            write_json(out / "attempts" / f"{count:03d}.json", identity)
            cfg.SAFETY_VERSION = VARIANTS[mode][0]
            env = TraceEnv(render_mode=None, emergency_stop=mode!="off", low_speed_start_frac=0.0)
            env.mode, env.why_counts, env.steps_logged = mode, Counter(), 0
            episode_start = time.perf_counter()
            try:
                with (out / "traces" / f"{count:03d}_{mode}.jsonl").open("x",encoding="utf-8",newline="\n") as trace:
                    env.trace = trace
                    result = run_episode(env, copy.deepcopy(scenes[case["case"]]),int(case["seed"]),"model",model)
                    trace.flush()
                    os.fsync(trace.fileno())
                row = dict(identity,dataset=case["dataset"],stratum=case.get("selection_stratum","explicit"),
                    elapsed_s=time.perf_counter()-episode_start,why_counts=dict(env.why_counts),**result)
                write_json(out / "attempts" / f"{count:03d}_result.json",row)
                results.append(row)
                with (out / "episodes.csv").open("a",encoding="utf-8",newline="") as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(row))
                    if count==1:
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                print(f"{count}/{manifest['planned_runs']} {mode} {case['case']}: {row['outcome']} "
                      f"({row['elapsed_s']:.1f}s, interventions={row['safety_v2_steps']})",flush=True)
            finally:
                env.close()
    drift = [name for name,h in source_hashes.items() if sha((ROOT/name).read_bytes())!=h]
    write_json(out / "completion.json",dict(completed_runs=count,elapsed_s=time.perf_counter()-started,
        source_drift=drift,checkpoint_unchanged=sha(MODEL.read_bytes())==EXPECTED_MODEL,
        selection_unchanged=sha(args.selection.read_bytes())==sha(selection_raw)))
    print(json.dumps({m:dict(Counter(r["outcome"] for r in results if r["mode"]==m)) for m in modes}),flush=True)
    if limits is not None:
        limits.restore_original_limits()


if __name__ == "__main__":
    main()
