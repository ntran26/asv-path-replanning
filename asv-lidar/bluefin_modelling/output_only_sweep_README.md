# Output-only sweep scripts for the Bluefin MATLAB-style Python model

These two scripts avoid the unreliable `S1/S2 -> rudder/rpm` replay and instead
fit the simulator directly to the **measured motion metrics** from your real tests.

## Files

- `sweep_surge_output_only.py`
- `sweep_turn_output_only.py`

They use the existing helper module:
- `bluefin_test_utils.py`

## 1) Surge / straight-line sweep

This sweep uses the **real speed-test metrics** and compares them against a
constant-rpm, zero-rudder open-loop simulation.

### Example

```bash
python sweep_surge_output_only.py \
  --real-json test_3_comparison.json \
  --out-dir surge_sweep_results \
  --rpm-grid 12.7,14.0,15.0,16.0,18.0 \
  --thrust-grid 0.04,0.05,0.06,0.07,0.08,0.09 \
  --drag-grid 0.5,0.75,1.0,1.25,1.5
```

### Outputs

- `surge_sweep_ranked.csv`
- `surge_sweep_top20.json`
- `best_surge_config.json`
- `best_surge_metrics.json`
- `best_surge_comparison.json`
- `best_surge_response.png`
- `best_surge_path.png`

## 2) Turning sweep

This sweep uses the **real turning-test metrics** and compares them against a
constant-rpm, constant-rudder open-loop simulation.

It sweeps:
- test rpm
- test rudder angle
- `TURN_COEF`
- `RUDDER_FORCE_SCALE`
- `LINEAR_YAW_DAMP`

while keeping surge parameters fixed (you can set them with CLI args).

### Example

```bash
python sweep_turn_output_only.py \
  --real-json test_4_comparison.json \
  --out-dir turn_sweep_results \
  --thrust-coef 0.07 \
  --drag-coef 0.75 \
  --rpm-grid 12.7,14.0,16.0,18.0 \
  --rudder-grid 25,30,35,40 \
  --turn-coef-grid 1.0,1.5,2.0,3.0,4.0,5.0 \
  --rudder-force-grid 0.1,0.2,0.3,0.5,0.7,1.0 \
  --yaw-damp-grid 1.0,2.0,5.0,10.0
```

### Outputs

- `turn_sweep_ranked.csv`
- `turn_sweep_top20.json`
- `best_turn_config.json`
- `best_turn_metrics.json`
- `best_turn_comparison.json`
- `best_turn_response.png`
- `best_turn_path.png`

## Recommended workflow

1. Run the surge sweep first.
2. Pick the best surge candidate.
3. Use that candidate's `THRUST_COEF` and `DRAG_COEF` in the turning sweep.
4. Send back:
   - `best_surge_config.json`
   - `best_surge_comparison.json`
   - `best_turn_config.json`
   - `best_turn_comparison.json`
   - the ranked CSV or top20 JSON files if you want a deeper analysis.

## Important note

These are **output-only** sweeps. That means they do **not** try to reproduce
unknown controller commands. They instead ask:

- what open-loop straight test best matches the real straight response?
- what open-loop turning test best matches the real turning response?

This makes them much more reliable when the RC/ESC mapping is uncertain.
