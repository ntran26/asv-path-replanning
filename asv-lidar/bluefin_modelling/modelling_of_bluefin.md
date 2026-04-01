# I. Overview

There are currently three different ship models in the project:
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

# II. The Original Python Model

The original `ship_model.py` defines just a few constants and a very compact state update: `MASS`, `THRUST_COEF`, `DRAG_COEF`, `TURN_COEF`, `RUDDEROFFSET`, and `MOMINERTIA = 0.5 * MASS * RUDDEROFFSET**2`. Its `_calc_forces()` computes forward thrust as `THRUST_COEF * rpm**2`, subtracts a quadratic drag `DRAG_COEF * v**2`, computes a rudder moment from the thrust and rudder angle, and damps yaw linearly with `TURN_COEF * w`. The `update()` function then integrates forward speed and yaw with a Verlet-like step and returns `(dx, dy, heading_deg, yaw_rate_degps)`. That is a very compact and useful RL-oriented model, but it has no sway velocity, no added mass, no rudder actuator, and no explicit separation between hull, propeller, and rudder forces.

The old model is excellent for:
    - Fast simulator
    - Small number of tunable parameters
    - Stable control-learning system


