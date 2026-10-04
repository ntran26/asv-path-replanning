"""Replay saved FIX05 hull-heading observations; no environment or new episode.

The new adapter consumes only each recorded onboard raw tracker state and the
captured perception snapshot. Truth is consulted only after prediction, for the
reported heading error. A previously reconstructed exact checked plan is read
from its immutable audit and its original logged clearance is revalidated.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/safety_dev/v18_development/heading_memory_saved")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing audit: " + str(args.output_dir))
    run = ROOT / "results/safety_dev/v10_iterations/consistent_track_probe8"
    trace = run / "traces/003_v13.jsonl"
    manifest_path = run / "manifest.json"
    previous_path = ROOT / "results/safety_dev/v10_iterations/audits/fix05_v13_final_audit/audit.json"
    manifest = json.loads(manifest_path.read_text())
    previous = json.loads(previous_path.read_text())
    if sha(trace) != previous["trace_sha256"]:
        raise ValueError("Trace no longer matches the immutable plan audit")
    dependencies = ["src/classical/common.py", "src/safety_v2.py", "src/safety_v3.py",
                    "src/ship.py", "src/reference_controller.py", "bluefin/dynamics.py"]
    for name in dependencies:
        if sha(ROOT / name) != manifest["source_sha256"][name]:
            raise ValueError("Prediction source differs from evaluated bytes: " + name)
    import constants as cfg
    for name, text in manifest["constants"].items():
        try:
            setattr(cfg, name, ast.literal_eval(text))
        except (ValueError, SyntaxError):
            pass
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    from safety_heading_memory import HeadingMemoryPerception

    class RecordedBase:
        def snapshot(self, env):
            return self.value

    base = RecordedBase()
    adapter = HeadingMemoryPerception(base)
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line]
    if len(rows) != previous["result"]["steps"]:
        raise ValueError("Trace is not a complete committed saved episode")
    events, score = [], None
    for row in rows:
        decision = row["diagnostic_decision"]
        values = copy.deepcopy(decision["snapshot"])
        values.pop("units", None)
        for name in ("tangent", "right", "centre", "points", "edges_a", "edges_b"):
            values[name] = np.asarray(values[name], dtype=float)
        values["tracks"] = [cc.TrackView(t["id"], np.asarray(t["position"]),
            np.asarray(t["velocity"]), t["heading"]) for t in values["tracks"]]
        base.value = cc.Snapshot(**values)
        onboard = row["diagnostic_before"]["onboard"]
        raw = []
        for track in onboard["raw_tracks"]:
            values = copy.deepcopy(track)
            values["history"] = [(sample["scan_serial"], np.asarray(sample["points"]))
                                 for sample in track["history"]]
            raw.append(SimpleNamespace(**values))
        revised = adapter.snapshot(SimpleNamespace(pose_stale=onboard["pose_stale"],
                                                    tracker=SimpleNamespace(tracks=raw)))
        if adapter.last_heading_memory_stats["replacements"]:
            events.append(dict(step=row["step"],
                               replacements=copy.deepcopy(adapter.last_heading_memory_stats["replacements"])))
        if row["step"] == 34:
            saved_act = decision["actuators_before_decision"]
            act = cc.Actuators()
            act.servo, act.executed = saved_act["servo"], saved_act["executed"]
            act.buffer = copy.deepcopy(saved_act["buffer"])
            plan_record = next(p for p in previous["plan_checks"] if p["step"] == 34)
            # JSON null is the existing NaN astern marker, only in throttle.
            plan = np.asarray(plan_record["plan"], dtype=float)
            rollout = v3.rollout_seq(base.value, act, plan[None])
            original_first, original_clear = v2.SafetyFilterV2._evaluate(None, base.value, rollout)
            revised_first, revised_clear = v2.SafetyFilterV2._evaluate(None, revised, rollout)
            np.testing.assert_allclose(original_clear[0], plan_record["logged_clearance"],
                                       atol=1e-12, rtol=0.)
            score = dict(step=34, plan_origin=plan_record["plan_origin"],
                original_first_violation_s=None if np.isposinf(original_first[0]) else float(original_first[0]),
                original_clearance_m=float(original_clear[0]),
                revised_first_violation_s=None if np.isposinf(revised_first[0]) else float(revised_first[0]),
                revised_clearance_m=float(revised_clear[0]),
                original_headings_deg={str(t.id): math.degrees(t.heading) for t in base.value.tracks},
                revised_headings_deg={str(t.id): math.degrees(t.heading) for t in revised.tracks},
                truth_heading_deg_scoring_only=row["diagnostic_before"]["truth_scoring_only"]["targets"][0]["heading"],
                previous_full_truth_target_cv_clearance_m_scoring_only=
                    plan_record["weak"]["truth_initial_target_cv_scoring"]["clearance_m"])
    if score is None:
        raise ValueError("Missing required saved decision34")
    files = dependencies + ["src/safety_heading_memory.py", "src/safety_v18.py",
                            "src/safety_provisional_tracks.py", "src/constants.py",
                            "tests/test_safety_v18.py"]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope=__doc__, case=previous["case"], saved_mode="v13",
        new_episodes=0, environment_reset_calls=0, environment_step_calls=0,
        policy_inference_calls=0, saved_decisions=len(rows), overrides=events,
        same_plan_score=score,
        source_sha256={name: sha(ROOT / name) for name in files},
        trace=dict(path=str(trace.relative_to(ROOT)), sha256=sha(trace)),
        manifest=dict(path=str(manifest_path.relative_to(ROOT)), sha256=sha(manifest_path)),
        original_plan_audit=dict(path=str(previous_path.relative_to(ROOT)), sha256=sha(previous_path)),
        script_sha256=sha(Path(__file__)),
        inherited_age_budget_steps=int(cfg.TRACK_MAX_MISSES),
        citation="Granstrom, Baum & Reuter, Extended Object Tracking, https://arxiv.org/abs/1604.00970; shape/kinematic modelling inspiration only.",
        limitations=[
            "This applies the new adapter to captured V13 snapshots, not a reconstructed V16 or V18 episode.",
            "All later measurements/actions come from the original saved trajectory; alternative closed-loop outcomes are unknown.",
            "Position and velocity errors remain. The previous full-truth-target sensitivity passes this plan at +0.00705m; correcting only heading can increase conservatism when centre is biased.",
            "The held axis may be wrong during a genuine poorly observed turn; expiry is an engineering budget, not an uncertainty bound.",
            "Old unrelated controller hashes need not match: this audit only calls the checked shared rollout/evaluator and the new perception adapter."])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output_dir / "reproducer_evaluated_bytes.py").write_bytes(Path(__file__).read_bytes())
    markdown = f"""# Saved FIX05 hull-heading-memory applicability

Created {report['created_utc']}. No new episodes, environment reset/step or policy inference.

The new adapter was replayed causally over all {len(rows)} captured V13 decision snapshots. It changes only source21 at decisions34–36, carrying the114° fit from decision33/scan476 for1–3 decisions. Memory expires at decision37. The synthetic persistent view remains unchanged.

At decision34 the original velocity-course fallback is147.561654°, versus held114° and scoring-only true hull115.783768°. The exact previously reconstructed policy plan reproduces the logged clearance **+0.134074279m**. Changing only this axis yields **−0.290731058m**, first violation0.125s.

This verifies gated applicability and geometry sensitivity, not rescue. The target centre remains biased; the earlier full-truth-target sensitivity passes the same plan at+0.007051m. A more accurate axis alone therefore does not establish a more accurate complete collision forecast. Future raw observations remain those of the saved V13 episode.

The model separates hull extent/orientation from translational kinematics, inspired by [Granstrom, Baum & Reuter's extended-object tracking overview](https://arxiv.org/abs/1604.00970). The bounded hold here is an engineering adaptation, not that paper's probabilistic estimator or a safety guarantee.

`audit.json` records source, trace, manifest and original-plan-audit hashes. The evaluated script bytes are preserved beside it. Reproduce from the project directory with a NEW output path:

```powershell
python -B tools/diagnostics/safety/audit_heading_memory_saved.py --output-dir results/safety_dev/v18_development/heading_memory_saved_repeat
```

Existing output directories are refused. The working script lives under tools; the saved byte copy is an immutable provenance artifact.
"""
    (args.output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(dict(output=str(args.output_dir.relative_to(ROOT)) if args.output_dir.is_relative_to(ROOT)
                         else str(args.output_dir), overrides=events, score=score), indent=2))


if __name__ == "__main__":
    main()
