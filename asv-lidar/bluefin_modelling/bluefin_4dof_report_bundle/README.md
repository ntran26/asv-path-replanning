# Bluefin 4DOF Report Bundle

## Purpose

This folder packages the current Bluefin 4DOF modelling work into one place for reporting.
It covers:

1. the original MATLAB 4DOF source model,
2. the current Python port,
3. the real-vessel logs and extracted benchmark metrics,
4. the parameter-tuning scripts,
5. the main result sets, plots, and evaluation summaries.

The goal of this bundle is to document the path from the MATLAB Bluefin 4DOF model to the current Python-based model and to show how the Python model was tuned and tested against real-vessel performance.

## Folder Layout

- `matlab/`
  - `Bluefin4DOFModel02.m`: primary MATLAB 4DOF source used for the Python port.
  - `Bluefin4DOFModel02_Solver.m`: MATLAB solver wrapper.
  - `Blue02.m`: earlier MATLAB model included for lineage/reference.

- `python/`
  - `ship_model_bluefin_4dof.py`: current Python 4DOF model.
  - `ship_model_bluefin_v2.py`: previous tuned baseline used for comparison.
  - `bluefin_test_utils.py`: shared metric extraction and plotting utilities.
  - `focused_4dof_sweep.py`: first 4DOF calibration sweep.
  - `fine_tune_4dof_sweep.py`: broader retune around the best 4DOF region.
  - `turn_timing_4dof_sweep.py`: timing-focused turn retune.

- `data/`
  - `test_3.log`, `test_4.log`: real-vessel logs.
  - `test_3_metrics.json`, `test_4_metrics.json`: extracted benchmark metrics from the logs.

- `results/focus/`
  - initial stable 4DOF tuning outputs
  - standard response/path plots
  - comparison JSON against the real-vessel metrics

- `results/fine/`
  - broader fine-tuning outputs
  - updated best configuration and summary
  - generated response/path plots for the best fine-tuned model

- `results/timing/`
  - timing-focused turn retuning outputs
  - generated response/path plots for the best timing-oriented model

- `notes/`
  - earlier modelling notes and PDF drafts used during the handoff and review process

## Modelling Workflow

### 1. MATLAB Source

The starting point for the current 4DOF work is `matlab/Bluefin4DOFModel02.m`.
This is the most complete MATLAB representation of the tested Bluefin vessel in the repo and includes:

- surge, sway, roll, and yaw dynamics,
- actuator-state dynamics,
- hydrodynamic hull terms,
- propeller and rudder coupling,
- bow-thruster terms.

`matlab/Blue02.m` is included because it was an earlier modelling branch used during the transition to the Python work and remains useful as a historical reference.

### 2. Python Port

The runnable Python port is `python/ship_model_bluefin_4dof.py`.
It follows the MATLAB structure but adds practical simulation guards such as:

- RK4 integration,
- low-speed numerical floors,
- command-to-shaft scaling to match the repo interface,
- a small set of calibration knobs to make fitting against real logs possible.

This Python model is the version now used for 4DOF tuning and test preview.

### 3. Real-Vessel Benchmark Data

The real-vessel benchmark comes from two log-driven manoeuvres:

- `data/test_3.log`: straight-line acceleration benchmark
- `data/test_4.log`: turning benchmark

The extracted target metrics are stored in:

- `data/test_3_metrics.json`
- `data/test_4_metrics.json`

These JSON files are the ground truth used by all sweep scripts.

### 4. Tuning Stages

The tuning was done in three stages.

#### Stage A: Initial 4DOF Sweep

Script:

- `python/focused_4dof_sweep.py`

Purpose:

- find a stable 4DOF region,
- align the straight-line response,
- get the first turn fit that beats the previous `v2` baseline.

Main output folder:

- `results/focus/`

#### Stage B: Fine Retune

Script:

- `python/fine_tune_4dof_sweep.py`

Purpose:

- widen the search around the best Stage A region,
- improve joint fit against both straight and turning metrics,
- update the default 4DOF calibration.

Main output folder:

- `results/fine/`

#### Stage C: Turn-Timing Retune

Script:

- `python/turn_timing_4dof_sweep.py`

Purpose:

- keep the fine-tuned straight-line fit,
- search for a turn configuration that better matches the real vessel’s heading-change timing,
- preserve this as an alternative timing-focused configuration rather than the new default unless it also wins on the standard objective.

Main output folder:

- `results/timing/`

## Current Default 4DOF Calibration

The current default Python 4DOF model in `python/ship_model_bluefin_4dof.py` was updated to the best standard-fit calibration found by the fine sweep.

Key model parameters:

```text
RPM_COMMAND_SCALE      = 90.0
PROPELLER_THRUST_SCALE = 2.2
PROPELLER_ADVANCE_SCALE = 1.0
RUDDER_FORCE_SCALE     = 0.12
RUDDER_YAW_SCALE       = 1.7
RUDDER_INFLOW_SCALE    = 1.0
RUDDER_X_DRAG_SCALE    = 0.01
LINEAR_SURGE_DAMP      = 1.5
LINEAR_YAW_DAMP        = 0.0
ROLL_DAMP_SCALE        = 4.0
ROLL_RESTORE_SCALE     = 1.2
```

Best standard-fit operating point from the sweep:

```text
straight_rpm = 14
turn_rpm     = 18
turn_rudder  = 30 deg
```

## Best Timing-Focused Alternative

The timing-focused sweep found a different configuration that better matches the real heading-change timing but is not the best on the standard combined objective.

Best timing-focused operating point:

```text
straight_rpm = 14
turn_rpm     = 17
turn_rudder  = 28 deg
RUDDER_FORCE_SCALE  = 0.10
RUDDER_YAW_SCALE    = 1.5
RUDDER_INFLOW_SCALE = 0.9
LINEAR_YAW_DAMP     = 0.1
```

This result is stored in:

- `results/timing/best_timing_4dof_config.json`
- `results/timing/best_timing_4dof_summary.json`

## Key Results

### Focused 4DOF Result

- best joint score: `5.5727`
- previous `v2` baseline score: `6.5462`
- conclusion: the 4DOF model beat the `v2` baseline on the benchmark

### Fine-Tuned 4DOF Result

- best joint score: `5.0313`
- previous focused 4DOF score: `5.5727`
- improvement vs focused 4DOF: `-0.5414`
- conclusion: this is the current best overall 4DOF fit and is the default model calibration

Selected fine-tuned turn metrics vs real:

| Metric | Real vessel | Fine 4DOF |
|---|---:|---:|
| Peak yaw rate [deg/s] | 32.8951 | 33.1244 |
| Time to 90 deg after turn [s] | 9.3869 | 6.3 |
| Time to 180 deg after turn [s] | 13.2874 | 11.0 |
| Radius first 90 deg [m] | 6.7641 | 6.9499 |
| Radius first 180 deg [m] | 4.1979 | 4.5370 |
| Speed at 10 s after turn [m/s] | 1.2812 | 1.2066 |

Interpretation:

- peak yaw rate is now almost exact,
- turn radius remains reasonably close,
- speed retention in the turn is much better than before,
- the remaining weakness is that the model still turns too early compared with the real vessel.

### Timing-Focused 4DOF Result

- baseline fine timing objective: `6.0986`
- best timing-focused objective: `5.7646`
- improvement on timing objective: `-0.3341`
- standard joint score of timing-focused model: `5.1633`

Interpretation:

- the timing-focused model improves the turn timing relative to the fine default,
- but the fine-tuned default still remains the better overall standard-fit model,
- therefore the timing-focused result is best treated as an alternative evaluation point, not the main default.

Selected timing-focused turn metrics vs real:

| Metric | Real vessel | Timing 4DOF |
|---|---:|---:|
| Peak yaw rate [deg/s] | 32.8951 | 30.6126 |
| Time to 90 deg after turn [s] | 9.3869 | 7.0 |
| Time to 180 deg after turn [s] | 13.2874 | 12.1 |
| Radius first 90 deg [m] | 6.7641 | 7.5934 |
| Radius first 180 deg [m] | 4.1979 | 4.7671 |
| Speed at 10 s after turn [m/s] | 1.2812 | 1.1403 |

## Which Result To Quote In The Report

For the main report narrative:

- use the fine-tuned 4DOF model as the current default calibrated Python model,
- use the timing-focused result as a secondary sensitivity study showing that turn timing can be improved further, but with some cost in overall fit.

Suggested interpretation:

1. the MATLAB 4DOF model was successfully ported to Python,
2. the Python port was calibrated against real-vessel straight and turning logs,
3. the fine-tuned 4DOF model outperformed the earlier `v2` benchmark,
4. the main remaining mismatch is turn timing rather than turn radius,
5. a timing-focused retune moved the timing closer to the real vessel but did not become the best overall fit.

## Plot Locations

Useful report figures are already included in the results folders:

- `results/focus/best_4dof_straight_response.png`
- `results/focus/best_4dof_straight_path.png`
- `results/focus/best_4dof_turn_response.png`
- `results/focus/best_4dof_turn_path.png`
- `results/fine/best_fine_4dof_straight_response.png`
- `results/fine/best_fine_4dof_straight_path.png`
- `results/fine/best_fine_4dof_turn_response.png`
- `results/fine/best_fine_4dof_turn_path.png`
- `results/timing/best_timing_4dof_turn_response.png`
- `results/timing/best_timing_4dof_turn_path.png`

## Reproduction

From the `python/` folder in this bundle, the main sweeps can be rerun with:

```bash
python -B focused_4dof_sweep.py
python -B fine_tune_4dof_sweep.py
python -B turn_timing_4dof_sweep.py
```

## Final Recommendation

Use the fine-tuned 4DOF model as the report’s current best Python model for the Bluefin vessel.

Use the timing-focused configuration only if the report needs to emphasize that turn timing can be pushed closer to the real vessel, while noting that it is not the best overall calibration on the standard combined benchmark.
