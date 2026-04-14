# Blue02 Transfer Review Continuation

This note continues the earlier `Review Blue02 model transfer` task and records the current conclusion after the later 4DOF tuning and timing-focused follow-up sweep.

## Bottom line

- `ship_model_bluefin_v2.py` is **not** a faithful transfer of `Blue02.m`.
- It is a **Blue02-inspired surrogate** with extra empirical shaping added on purpose.
- The newer `ship_model_bluefin_4dof.py` is the stronger MATLAB-lineage model, because it is a guarded adapter of `Bluefin4DOFModel02.m`, which the repo handoff notes identify as the better source.
- The tuned 4DOF model is clearly better than the current `v2` benchmark fit, but it still turns too early relative to the real vessel, so it should be treated as a **better simulator**, not yet a fully trusted field-prediction model.

## Why `Blue02.m` is not the main truth source

The project handoff note explicitly says not to use `Blue02.m` as the main source because it is incomplete and internally inconsistent. It notes that:

- comments and state indexing do not agree,
- it claims a 9-state model but returns only 7 derivatives,
- it does not provide a clean full 3DOF/4DOF state evolution.

See [bluefin_4dof_subsection_and_handoff.md](bluefin_4dof_subsection_and_handoff.md).

## Why `ship_model_bluefin_v2.py` is not a faithful transfer

The file itself says it is a Bluefin-inspired nonlinear 3DOF model and documents two deliberate departures from the earlier form:

- an empirical speed-dependent thrust law,
- separate rudder scales for sway, yaw, and axial drag.

Those are modelling choices made for fit quality, not a strict transfer. See [ship_model_bluefin_v2.py](../python/ship_model_bluefin_v2.py) and the added empirical/tunable terms in that file.

So the right description of `v2` is:

- useful calibrated simulator,
- Blue02-lineage in spirit,
- not a line-by-line or structure-preserving transfer.

## Why the 4DOF model is the better transfer path

`ship_model_bluefin_4dof.py` states that it is a numerically guarded Python port of `Bluefin4DOFModel02.m`, with only practical runtime adjustments such as flow floors, RK4 integration, and command-scale bridging. See [ship_model_bluefin_4dof.py](../python/ship_model_bluefin_4dof.py).

It also preserves the richer state structure that matters for fidelity:

- surge and sway,
- roll and yaw,
- rudder actuator state,
- main propeller state,
- bow-thruster state.

See the reset/state fields in [ship_model_bluefin_4dof.py](../python/ship_model_bluefin_4dof.py).

## Current benchmark conclusion

The tuned 4DOF model now outperforms the `v2` baseline on the shared benchmark:

- tuned 4DOF joint score: `5.0313`
- previous 4DOF score: `5.5727`
- `v2` baseline score: `6.5462`

See [best_fine_4dof_summary.json](../results/fine/best_fine_4dof_summary.json).

By comparison, the `v2` benchmark still shows major mismatches in turn and transient shape, for example:

- peak yaw rate `19.33 deg/s` vs real `32.90 deg/s`,
- turn radius at first `90 deg`: `4.59 m` vs real `6.76 m`,
- speed at `10 s` after turn: `0.872 m/s` vs real `1.281 m/s`.

These quoted `v2` values were taken from the earlier archived `v2` benchmark comparison during the Blue02-lineage review pass.

## Remaining gap after the timing-focused continuation

The later timing-focused sweep improved the tuned 4DOF turn timing, but did not close the gap fully:

- baseline fine-tuned 4DOF: `90 deg = 6.3 s`, `180 deg = 11.0 s`
- best timing-focused candidate: `90 deg = 7.0 s`, `180 deg = 12.1 s`
- real vessel: `90 deg = 9.3869 s`, `180 deg = 13.2874 s`

See [best_timing_4dof_summary.json](../results/timing/best_timing_4dof_summary.json).

The best timing-focused candidate also came with some giveback in the other turn metrics:

- peak yaw rate `30.61 deg/s`,
- first `90 deg` radius `7.59 m`,
- first `180 deg` radius `4.77 m`,
- speed at `10 s` after turn `1.140 m/s`.

See [best_timing_4dof_config.json](../results/timing/best_timing_4dof_config.json).

This means the remaining error is no longer a simple “turn harder / turn less” gain problem. The model still builds heading change too quickly once the turn develops.

## Recommendation for field use

My recommendation is:

- `ship_model_bluefin_v2.py`: **no**, do not treat it as an accurate Blue02 transfer or as the best field-confidence model.
- tuned `ship_model_bluefin_4dof.py`: **better and worth using as the main Bluefin simulator**, but **not yet accurate enough to be the sole basis for open-loop field prediction**.

For field experiments, the tuned 4DOF model is reasonable if it is used with:

- conservative controller limits,
- staged on-water validation,
- domain randomization or robustness margins around turn timing,
- the expectation that the real vessel may respond later/slower in heading buildup than the model predicts.

## Current repo wiring

In this checkout, the main RL stack is not defaulting to the Bluefin model yet. `ship_model_selector.py` is currently set to:

- `SHIP_MODEL_VARIANT = "standard_3dof"`

with the Bluefin path still commented out. So any training or validation routed through the selector will keep using the older simplified model unless that switch is changed manually.

## Most likely next improvement

The timing-focused sweep suggests the remaining mismatch is structural, not just a missing scalar gain. The next worthwhile candidates are:

- extra steering actuator lag or deadband,
- slower yaw-moment buildup,
- effective yaw inertia changes,
- inflow/dynamic rudder response terms,
- controller robustness against turn-onset uncertainty rather than more static gain tuning.
