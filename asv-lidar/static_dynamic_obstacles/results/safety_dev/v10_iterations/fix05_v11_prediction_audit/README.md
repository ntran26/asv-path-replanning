# Working audit commands

Use the maintained diagnostic scripts below. The original `audit.py`, `projection_audit.py`, JSON results and `provenance.json` in this directory remain unchanged, preserving the hashes cited by the validated report.

Run from `asv-lidar/static_dynamic_obstacles`:

```powershell
python -B tools/diagnostics/safety/audit_v11_prediction.py
python -B tools/diagnostics/safety/audit_v11_projection.py
```

Without `--tag`, both commands read saved data and print findings without writing outputs. They locate the project from their script path, so absolute script paths also work from other directories. Neither constructs or advances an environment or calls a policy.

To save a new audit, choose **different, unused tags**:

```powershell
python -B tools/diagnostics/safety/audit_v11_prediction.py --tag MY_NEW_PREDICTION_TAG
python -B tools/diagnostics/safety/audit_v11_projection.py --tag MY_NEW_PROJECTION_TAG
```

Outputs go under `results/safety_dev/v10_iterations/audits/<tag>/`. Every existing tag is refused, including partial or empty directories. `--output-root` may select another directory within `results/safety_dev`; it never permits replacing an old output.

The default input is the completed `persistence_three/traces/001_v11.jsonl` FIX05 trace. Input selectors are explicit:

```powershell
python -B tools/diagnostics/safety/audit_v11_prediction.py --run-dir results/safety_dev/v10_iterations/persistence_three --trace 001_v11.jsonl
python -B tools/diagnostics/safety/audit_v11_projection.py --run-dir results/safety_dev/v10_iterations/persistence_three --trace 001_v11.jsonl --step 19 --source-id 4
```

The projection audit requires an existing anchor in the preceding frame and one truth target for scoring. It compares the original recorded state/plan/bank with an onboard-only geometric projection; it does not predict a new episode outcome. Both tools require a completed matching result and the evaluated prediction-source hashes. Unsupported calibrated/dual braking or target-envelope configurations are rejected rather than silently replayed with a different checker.

Validation performed without episodes: the working copies reproduced the original numerical findings, both refused existing tags, read-only execution worked from a different directory without changing outputs, and all four original code/data hashes still matched `provenance.json`. Validation outputs are the separate `audits/fix05_prediction_working_repro` and `audits/fix05_projection_working_repro` directories.
