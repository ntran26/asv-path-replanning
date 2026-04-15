# I. Overview

There are currently four important ship-model stages in the project:
1. The original simplified Python model (ship_model.py)
    - One forward speed state
    - One heading state
    - One yaw-rate state
    - Thurst, drag and turning are represented by three adjustable coefficients
    - Fast and convenient for RL

2. The MATLAB Bluefin model (Blue02.m)
    - A nonlinear **surge-sway-yaw maneuvering model** with a rudder actuator and hydrodynamic derivatives
    - Closer to a classical marine maneuvering model
    - Harder to tune and more sensitive to implementation details

3. The updated Python model (ship_model_bluefin.py)
    - A **MATLAB-inspired 3-DOF model** that preserves the old Python interface:

    ```python
    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)
    ```

    - Adds surge, sway, yaw, rudder dynamics, added mass, nonlinear damping, and more realistic kinematics

4. The current Python 4DOF model (ship_model_bluefin_4dof.py)
    - A guarded Python port of the MATLAB `Bluefin4DOFModel02.m`
    - Adds roll, actuator-state dynamics, and a closer MATLAB-lineage structure
    - Is the current best calibrated Bluefin model in this repo

# II. The Original Python Model

The original `ship_model.py` defines just a few constants and a very compact state update: `MASS`, `THRUST_COEF`, `DRAG_COEF`, `TURN_COEF`, `RUDDEROFFSET`, and `MOMINERTIA = 0.5 * MASS * RUDDEROFFSET**2`. Its `_calc_forces()` computes forward thrust as `THRUST_COEF * rpm**2`, subtracts a quadratic drag `DRAG_COEF * v**2`, computes a rudder moment from the thrust and rudder angle, and damps yaw linearly with `TURN_COEF * w`. The `update()` function then integrates forward speed and yaw with a Verlet-like step and returns `(dx, dy, heading_deg, yaw_rate_degps)`. That is a very compact and useful RL-oriented model, but it has no sway velocity, no added mass, no rudder actuator, and no explicit separation between hull, propeller, and rudder forces.

The old model is excellent for:
    - Fast simulator
    - Small number of tunable parameters
    - Stable control-learning system

Representative code from the original simplified model:

```python
MASS = 64.55
THRUST_COEF = 0.04
DRAG_COEF = 10
TURN_COEF = 100

MAX_RUD_ANGLE = 30
RUDDEROFFSET = 3
MOMINERTIA = 0.5 * MASS * RUDDEROFFSET**2

def _calc_forces(self, rpm, rud):
    thrust = THRUST_COEF * rpm**2
    rud_angle = np.radians(MAX_RUD_ANGLE * rud / 100)
    fwd_thrust = thrust * np.cos(rud_angle) - DRAG_COEF * self._v**2

    rud_moment = thrust * np.sin(rud_angle) * RUDDEROFFSET
    moment = rud_moment - (TURN_COEF * self._w)
    return fwd_thrust, moment

def update(self, rpm, rud, dt):
    d = self._v * dt + self._a * dt * dt * 0.5
    self._h = self._h + self._w * dt + self._dw * dt * dt * 0.5

    dx = d * np.sin(self._h)
    dy = d * np.cos(self._h)

    thrust, moment = self._calc_forces(rpm, rud)
    a = thrust / MASS
    dw = moment / MOMINERTIA
```

This code shows why the model is so easy to run in RL:

- forward thrust is a single `rpm**2` law,
- drag is a single quadratic term,
- turning is represented by one rudder moment and one yaw-damping term,
- there is no explicit sway state or rudder actuator dynamics.

# III. The Blue02-Based Intermediate Model

The next step in the modelling process was to move beyond the simplified `ship_model.py` and borrow more structure from the MATLAB Bluefin work, especially `old_models/Blue02.m`.

That MATLAB file introduced a more classical manoeuvring-model structure:

- separate surge, sway, and yaw states,
- rudder-state dynamics,
- added-mass and hydrodynamic derivative terms,
- separated hull, propeller, and rudder contributions.

This led to the Blue02-inspired Python branch represented by `old_models/ship_model_bluefin.py` and later refined into the `v2` form used as the earlier calibrated benchmark reference in the project report bundle:

- [old_models/ship_model_bluefin.py](old_models/ship_model_bluefin.py)
- [old_models/ship_model_bluefin_v2.py](old_models/ship_model_bluefin_v2.py)
- [bluefin_4dof_report_bundle/python/ship_model_bluefin_v2.py](bluefin_4dof_report_bundle/python/ship_model_bluefin_v2.py)

This intermediate stage was important because it preserved the repo-friendly interface:

```python
model = ShipModel()
dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)
```

while introducing more realistic internal vessel dynamics than the original RL-oriented model.

However, this Blue02-based path also revealed an important limitation: `Blue02.m` was useful as lineage, but not ideal as the final truth source. It was harder to port cleanly, had internal inconsistencies, and the best-performing Python `v2` model ended up including empirical shaping terms for practical fit rather than being a strict MATLAB transfer.

Representative code from the Blue02-inspired `v2` stage:

```python
THRUST_LOW_SPEED_BOOST = 1.6
THRUST_BOOST_U0 = 0.7
THRUST_HIGH_SPEED_DECAY = 0.26

RUDDER_FORCE_SCALE = 0.32
RUDDER_YAW_SCALE = 2.60
RUDDER_X_DRAG_SCALE = 0.02

def _propeller_force(self, rpm: float, u_eff: float) -> float:
    n = max(rpm, 0.0)
    static_term = THRUST_COEF * n * abs(n)
    low_speed_boost = 1.0 + THRUST_LOW_SPEED_BOOST * math.exp(
        -u_eff / max(THRUST_BOOST_U0, 1e-6)
    )
    high_speed_decay = 1.0 / (1.0 + THRUST_HIGH_SPEED_DECAY * u_eff * u_eff)
    return (1.0 - TP) * static_term * low_speed_boost * high_speed_decay
```

and in the derivative block:

```python
x_prop = self._propeller_force(rpm, u_eff)

f_n = RUDDER_FORCE_SCALE * 0.5 * rho * AR * FALP * (
    u_r * u_r + v_r * v_r
) * math.sin(alpha_r)

x_rud = -RUDDER_X_DRAG_SCALE * abs(f_n) * abs(math.sin(delta))
y_rud = -(1.0 + AH) * f_n * math.cos(delta)
n_rud = -RUDDER_YAW_SCALE * rudder_arm * f_n * math.cos(delta)
```

This is the key step where the modelling became more practical and less literal:

- the thrust law was reshaped to better match the measured acceleration curve,
- rudder sway force, yaw authority, and axial drag were separated,
- the model became a calibrated Blue02-lineage simulator rather than a strict transfer.

# IV. Why The Current Model Moved To 4DOF

After the Blue02-lineage work, the modelling effort shifted to the MATLAB file `Bluefin4DOFModel02.m`, which is included in the report bundle here:

- [bluefin_4dof_report_bundle/matlab/Bluefin4DOFModel02.m](bluefin_4dof_report_bundle/matlab/Bluefin4DOFModel02.m)

This model was treated as the stronger source because it contains a fuller vessel representation:

- surge and sway,
- roll and yaw,
- rudder actuator state,
- main propeller state,
- bow-thruster state.

That richer structure made it a more promising basis for matching the real Bluefin vessel than the earlier Blue02 branch.

The current runnable Python model is:

- [ship_model_bluefin_4dof.py](ship_model_bluefin_4dof.py)

Its role is not to be a purely literal text-for-text translation, but a guarded runnable adapter of the MATLAB 4DOF equations. The main practical additions are:

- RK4 integration instead of a simple Euler step,
- low-speed numerical floors to avoid instability at startup,
- a command-scale bridge between the repo throttle command and the MATLAB propeller-rpm convention,
- a small set of calibration parameters that can be tuned against real-vessel logs.

Representative code from the current 4DOF model header:

```python
RPM_COMMAND_SCALE = 90.0
THRUSTER_COMMAND_SCALE = 60.0

RECOMMENDED_COMMAND_RPM_MAX = 18.0
RECOMMENDED_PROP_RPM_MAX = RPM_COMMAND_SCALE * RECOMMENDED_COMMAND_RPM_MAX
RECOMMENDED_PEAK_SPEED_MPS = 2.03

PROPELLER_THRUST_SCALE = 2.2
PROPELLER_ADVANCE_SCALE = 1.0
RUDDER_FORCE_SCALE = 0.10
RUDDER_YAW_SCALE = 1.7
RUDDER_INFLOW_SCALE = 1.0
RUDDER_X_DRAG_SCALE = 0.01
LINEAR_SURGE_DAMP = 1.5
LINEAR_YAW_DAMP = 0.0
ROLL_DAMP_SCALE = 4.0
ROLL_RESTORE_SCALE = 1.2
```

These constants show the new modelling philosophy: keep the core MATLAB structure, but expose a small number of physically interpretable scales that can be tuned against the real-vessel logs.

# V. Current 4DOF Modelling Process

The current modelling workflow is therefore:

1. start from the MATLAB source `Bluefin4DOFModel02.m`,
2. port the state equations into a Python model that still fits the repo interface,
3. run benchmark manoeuvres against real-vessel logs,
4. tune a small number of scaling parameters rather than rewriting the whole model structure,
5. compare the tuned 4DOF model against the previous Blue02-lineage `v2` benchmark.

The benchmark data used for this process are:

- [real_vessel_performance/test_3.log](real_vessel_performance/test_3.log): straight-line acceleration
- [real_vessel_performance/test_4.log](real_vessel_performance/test_4.log): turning response
- [test_3_metrics.json](test_3_metrics.json): extracted straight-line metrics
- [test_4_metrics.json](test_4_metrics.json): extracted turning metrics

The tuning and evaluation scripts were later packaged into the report bundle:

- [bluefin_4dof_report_bundle/python/focused_4dof_sweep.py](bluefin_4dof_report_bundle/python/focused_4dof_sweep.py)
- [bluefin_4dof_report_bundle/python/fine_tune_4dof_sweep.py](bluefin_4dof_report_bundle/python/fine_tune_4dof_sweep.py)
- [bluefin_4dof_report_bundle/python/turn_timing_4dof_sweep.py](bluefin_4dof_report_bundle/python/turn_timing_4dof_sweep.py)
- [bluefin_4dof_report_bundle/python/bluefin_test_utils.py](bluefin_4dof_report_bundle/python/bluefin_test_utils.py)

The actual 4DOF update interface is still designed to look like the old simulator:

```python
def update(self, rpm: float, rud: float, dt: float, *, thruster_rpm: float = 0.0):
    delta_cmd = float(np.clip(rud, -100.0, 100.0)) / 100.0 * math.radians(MAX_RUD_ANGLE)
    n1_cmd_rpm = max(float(rpm), 0.0) * RPM_COMMAND_SCALE
    n2_cmd_rpm = float(thruster_rpm) * THRUSTER_COMMAND_SCALE

    k1 = self._derivatives(s0, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
    k2 = self._derivatives(s0 + 0.5 * dt * k1, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
    k3 = self._derivatives(s0 + 0.5 * dt * k2, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
    k4 = self._derivatives(s0 + dt * k3, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
    s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

This is one of the most important implementation choices in the current model:

- the external interface still uses the repo-style `rpm` and `rud`,
- internally, the code converts them into the MATLAB-style actuator variables,
- RK4 was used to make the higher-order model stable enough to run from rest and inside the validation harness.

The tuning was carried out in three stages:

## Stage A: Focused 4DOF Sweep

Purpose:

- find a stable 4DOF operating region,
- align the straight-line response,
- obtain the first 4DOF fit that beat the old `v2` benchmark.

Main archived outputs:

- [bluefin_4dof_report_bundle/results/focus/best_4dof_vs_v2_summary.json](bluefin_4dof_report_bundle/results/focus/best_4dof_vs_v2_summary.json)
- [bluefin_4dof_report_bundle/results/focus/best_4dof_joint_config.json](bluefin_4dof_report_bundle/results/focus/best_4dof_joint_config.json)

## Stage B: Fine Retune Around The Best Region

Purpose:

- widen the search around the best Stage A region,
- improve the combined straight-plus-turn fit,
- update the default calibration of the 4DOF model.

Main archived outputs:

- [bluefin_4dof_report_bundle/results/fine/best_fine_4dof_summary.json](bluefin_4dof_report_bundle/results/fine/best_fine_4dof_summary.json)
- [bluefin_4dof_report_bundle/results/fine/best_fine_4dof_config.json](bluefin_4dof_report_bundle/results/fine/best_fine_4dof_config.json)

## Stage C: Timing-Focused Turn Retune

Purpose:

- improve heading-change timing during the turn,
- test whether the remaining mismatch was mainly a gain-tuning problem,
- keep the result as an alternative timing-oriented configuration rather than the default unless it also won on the standard objective.

Main archived outputs:

- [bluefin_4dof_report_bundle/results/timing/best_timing_4dof_summary.json](bluefin_4dof_report_bundle/results/timing/best_timing_4dof_summary.json)
- [bluefin_4dof_report_bundle/results/timing/best_timing_4dof_config.json](bluefin_4dof_report_bundle/results/timing/best_timing_4dof_config.json)

The 4DOF derivative block is where the MATLAB structure becomes visible. A representative excerpt is:

```python
j_adv = PROPELLER_ADVANCE_SCALE * onew * u / max(abs(n1_force) * d_prop, MIN_ADVANCE_RATIO)
kt = a0 + a1 * j_adv + a2 * j_adv * j_adv
xd_p = (
    PROPELLER_THRUST_SCALE
    * abs(n1_force) * n1_force * onet * kt * (d_prop**4)
    / (0.5 * l_ship * draft * u_mag * u_mag)
)

yd_r = RUDDER_FORCE_SCALE * (1.0 + a_h) * fd_n * math.cos(delta) * math.cos(phi)
nd_r = RUDDER_FORCE_SCALE * RUDDER_YAW_SCALE * (
    x_r + a_h * x_h
) * fd_n * math.cos(delta) * math.cos(phi)

roll_damping_moment = -ROLL_DAMP_SCALE * b44 * p
roll_restoring_moment = -ROLL_RESTORE_SCALE * c44 * phi
pdot = (
    kd * force_scale_k
    + roll_damping_moment
    + roll_restoring_moment
    + (z_h - z_g) * (my * vdot + mx * u * r)
) / m33
```

This excerpt matters because it shows exactly what changed from the 3DOF stage:

- propeller thrust now depends on the advance ratio `j_adv`,
- rudder forces depend on roll and inflow state,
- roll has its own dynamic equation with damping and restoring terms,
- the vessel is no longer just a surge-sway-yaw model with tuned extras.

# VI. Current Default 4DOF Calibration

The current default calibration in `ship_model_bluefin_4dof.py` corresponds to the best standard-fit result found in the fine sweep.

Key model parameters:

```text
RPM_COMMAND_SCALE       = 90.0
PROPELLER_THRUST_SCALE  = 2.2
PROPELLER_ADVANCE_SCALE = 1.0
RUDDER_FORCE_SCALE      = 0.12
RUDDER_YAW_SCALE        = 1.7
RUDDER_INFLOW_SCALE     = 1.0
RUDDER_X_DRAG_SCALE     = 0.01
LINEAR_SURGE_DAMP       = 1.5
LINEAR_YAW_DAMP         = 0.0
ROLL_DAMP_SCALE         = 4.0
ROLL_RESTORE_SCALE      = 1.2
```

Best operating point from the standard-fit sweep:

```text
straight_rpm = 14
turn_rpm     = 18
turn_rudder  = 30 deg
```

These values are archived in:

- [bluefin_4dof_report_bundle/results/fine/best_fine_4dof_config.json](bluefin_4dof_report_bundle/results/fine/best_fine_4dof_config.json)

# VII. Current Evaluation Summary

The most important conclusion from the current modelling stage is that the 4DOF model now performs better than the previous Blue02-lineage `v2` benchmark on the shared real-vessel tests.

Fine-tuned 4DOF summary:

- fine joint score: `5.0313`
- fine surge score: `3.9177`
- fine turn score: `1.1135`
- previous focused 4DOF score: `5.5727`
- previous `v2` baseline score: `6.5462`

This means:

- the 4DOF port successfully outperformed the earlier `v2` benchmark,
- the current default model is the best overall calibrated Bluefin model in this repo,
- but it is still not a perfect field-truth model.

Selected turn metrics for the current fine-tuned 4DOF model versus the real vessel:

- peak yaw rate: `33.12 deg/s` model vs `32.90 deg/s` real
- time to `90 deg` after turn: `6.3 s` model vs `9.39 s` real
- time to `180 deg` after turn: `11.0 s` model vs `13.29 s` real
- radius at first `90 deg`: `6.95 m` model vs `6.76 m` real
- radius at first `180 deg`: `4.54 m` model vs `4.20 m` real
- speed at `10 s` after turn: `1.207 m/s` model vs `1.281 m/s` real

Interpretation:

- turn radius is now reasonably close,
- peak yaw rate is almost exact,
- speed retention in the turn is much improved,
- the main remaining weakness is that the model still builds heading change too early compared with the real vessel.

The timing-focused continuation improved turn timing somewhat:

- baseline fine-tuned timing: `90 deg = 6.3 s`, `180 deg = 11.0 s`
- timing-focused candidate: `90 deg = 7.0 s`, `180 deg = 12.1 s`
- real vessel: `90 deg = 9.3869 s`, `180 deg = 13.2874 s`

So the remaining mismatch is likely structural, not just a single gain error.

# VIII. Practical Conclusion

The full modelling process now has two major transitions:

1. from the original simplified RL model to the Blue02-inspired 3DOF Python model,
2. from the Blue02-inspired model to the current 4DOF MATLAB-lineage Python model.

The practical conclusion at the current stage is:

- `ship_model.py` remains the simplest and fastest RL-oriented simulator,
- the Blue02-lineage `v2` model was an important intermediate step,
- `ship_model_bluefin_4dof.py` is now the main calibrated Bluefin model and the strongest representation of the tested vessel currently available in this repo.

At the same time, the 4DOF model should still be treated as:

- a better research and controller-development simulator,
- a stronger basis for realistic training and testing,
- but not yet a fully trusted open-loop field predictor without additional on-water validation.




