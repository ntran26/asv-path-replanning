# parameter_sweep.py explanation

## Purpose

This script is an automatic calibration tool for the simplified Python ship model.

It does this pipeline:

1. Load the real-vessel benchmark JSON files
2. Read the real control inputs `S1`, `S2`
3. Convert those real inputs into the simplified simulator inputs
   - rudder percent
   - rpm
4. Run the simulated ship model with many candidate parameter sets
5. Extract the same metrics from the simulated run
6. Compare simulated metrics vs real metrics
7. Rank the parameter sets by error

In short:

`real logs -> replay in sim -> measure -> score -> best parameters`

---

## Configuration section

### File paths
- `ROOT`: folder containing the script
- `REAL_TEST3_JSON`, `REAL_TEST4_JSON`: benchmark JSON files
- `OUTPUT_DIR`: where ranked sweep results are saved
- `SHIP_MODEL_MODULE`: Python module containing the ship model

### Fixed and swept parameters
- `FIXED_MASS = 64.55`: mass is kept fixed in this version
- `THRUST_GRID`, `DRAG_GRID`, `TURN_GRID`: candidate values to test

### Replay / mapping assumptions
- `RPM_MAX`: maximum rpm sent to the simplified model
- `RUDDER_CMD_MAX_PERCENT`: maximum rudder command percentage

### Optional overrides
These allow manual control mapping if the automatic inference is poor:
- `OVERRIDE_S1_NEUTRAL`
- `OVERRIDE_S2_NEUTRAL`
- `OVERRIDE_S1_SCALE`
- `OVERRIDE_S2_FULL_FWD`

### Metric weights
`WEIGHTS` defines how much each metric contributes to the total calibration score.

---

## Utility functions

### `wrap_180(deg)`
Normalizes an angle difference to `[-180, 180)`.

Used so heading differences are physically meaningful across wrap-around.

### `unwrap_heading_deg(yaw_deg)`
Turns wrapped headings like `179, -179` into a continuous sequence.

Needed to compute heading-change milestones such as time to 90 degrees and 180 degrees.

### `sample_at_time(t_rel, values, query_s)`
Interpolates a signal at a chosen time.

Used for metrics like speed at 10 s or yaw rate at 5 s.

### `first_crossing_time(t_rel, values, threshold)`
Returns the first time a signal crosses a threshold.

Used for rise-time metrics like time to 50 percent and 90 percent of peak speed.

### `first_abs_crossing_time(t_rel, values, threshold)`
Same idea, but for absolute values.

Used for turning metrics like first time heading change reaches 90 degrees or 180 degrees.

### `slope_over_window(t_rel, values, t1, t2)`
Fits a line over a time window and returns the slope.

Used to estimate acceleration from a short response window.

### `cumulative_distance(x, y)`
Computes accumulated path length from x-y points.

Used for distance-after-time metrics.

### `circle_fit_radius(x, y)`
Fits a circle to trajectory points and returns the radius.

Used for turning radius and diameter estimates.

### `first_sustained_index(values, threshold, count=3)`
Finds the first index where a signal stays above a threshold for several consecutive samples.

Used to detect motion onset robustly.

### `first_sustained_abs_index(values, threshold, count=3)`
Same idea, but using absolute value.

Used to detect turn onset from yaw rate.

### `safe_rel_error(sim, real, floor=1e-6)`
Computes relative error safely and avoids divide-by-zero.

Used in the scoring function.

---

## Data classes

### `ReplaySeries`
Stores the real replay signals:
- `t_sec`
- `s1`
- `s2`
- `yaw_rate_real`
- `u_body_real`

This makes it easy to pass the replay information around as one object.

### `ReplayMapping`
Stores how to convert real logged controls into simplified ship-model commands.

It includes:
- `s1_neutral`
- `s1_scale`
- `s2_neutral`
- `s2_full_fwd`

Methods:
- `s1_to_rudder_percent(s1_val)`: maps PWM-like rudder signal to rudder percent
- `s2_to_rpm(s2_val)`: maps throttle signal to rpm

This is the bridge between the real logs and the simulator.

---

## Loading helpers

### `load_json(path)`
Loads a JSON file into a Python dictionary.

### `build_replay_series(data)`
Extracts time series arrays from the benchmark JSON and returns a `ReplaySeries`.

### `infer_mapping(data, series)`
Infers the control mapping from the benchmark data.

It estimates:
- rudder neutral
- throttle neutral
- rudder scaling
- full-forward throttle

If manual overrides are not given, this function tries to guess them from the real logs.

---

## Ship-model loading and simulation

### `load_ship_model_module()`
Dynamically imports `ship_model.py`.

This lets the sweep patch the model parameters programmatically.

### `simulate_replay(series, mapping, thrust_coef, drag_coef, turn_coef, mass)`
This is the core simulator replay function.

What it does:
1. Imports the ship model
2. Patches the candidate parameters into the model
3. Recomputes yaw inertia from mass
4. Creates a fresh `ShipModel()`
5. Replays the real `S1/S2` history into the model
6. Records simulated outputs:
   - x
   - y
   - heading
   - yaw rate
   - forward speed

This gives simulated trajectories that correspond to the real test inputs.

---

## Simulated metric extraction

### `extract_sim_metrics(sim)`
Computes the same types of metrics from the simulated run that were extracted from the real logs.

It creates two groups:

### Straight / motion-aligned metrics
- detect motion start from `u_body`
- build motion-relative time
- compute:
  - peak forward speed
  - early acceleration
  - distance after 10 s
  - time to 50 percent peak
  - time to 90 percent peak
  - forward speed at 10 s

### Turn / turn-aligned metrics
- detect turn start from yaw rate
- unwrap heading
- compute:
  - peak absolute yaw rate
  - time to 90 degrees
  - time to 180 degrees
  - radius from first 90 degrees
  - radius from first 180 degrees
  - speed at 10 s into the turn

This makes the simulated metrics directly comparable with the real benchmarks.

---

## Scoring

### `score_against_real(sim_metrics, real3, real4)`
Compares simulated metrics against real metrics and turns all mismatches into a single scalar score.

It compares:
- speed peak
- speed acceleration windows
- distance at 10 s
- time to 50 percent and 90 percent peak speed
- time to 90 degrees and 180 degrees heading change
- peak yaw rate
- turning radii
- speed during the turn

Each error term is multiplied by a weight from `WEIGHTS`.

Lower total score means a better fit.

This is the objective function for calibration.

---

## Main sweep loop

### `main()`
Top-level controller for the whole calibration run.

What it does:
1. Creates the output folder
2. Loads the two real benchmark JSON files
3. Builds replay series
4. Infers control mappings
5. Builds all combinations of thrust, drag, and turn coefficient
6. For each combination:
   - simulate test 3 replay
   - simulate test 4 replay
   - extract simulated metrics
   - compare against real metrics
   - save score and metric details
7. Sorts all results by total score
8. Saves:
   - ranked CSV
   - top 10 JSON

---

## Why two replays are used

- `test_3` is used mainly for straight-line / surge behavior
- `test_4` is used mainly for turning behavior

So the script uses:
- straight metrics from simulated test 3
- turn metrics from simulated test 4

This is cleaner than trying to fit everything from one run.

---

## Core idea to remember

If you need to reproduce this workflow in the future, the main recipe is:

1. Collect real test logs
2. Extract control and response signals
3. Replay the real controls in simulation
4. Extract the same metrics from simulation
5. Compute mismatch
6. Sweep parameters
7. Choose the best set

That is the main logic behind the script.
