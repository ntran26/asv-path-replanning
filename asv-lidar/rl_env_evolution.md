# `rl_env.py` Evolution Notes

This note summarizes the major design changes that led to the current `asv-lidar/rl_env.py` branch.

Related files that evolved with it:

- `asv-lidar/asv_lidar.py`
- `asv-lidar/train_test_asv.py`
- `asv-lidar/test_run.py`
- `asv-lidar/ship_model_selector.py`
- `asv-lidar/ship_model.py`
- `asv-lidar/ship_model_bluefin_4dof.py`

It focuses on the significant method changes rather than every coefficient tweak.

## 1. Project Context

The environment models an autonomous surface vessel (ASV) that must:

1. follow a nominal start-to-goal path,
2. avoid static obstacles and map borders,
3. learn from LiDAR-like sensing under realistic actuator and hull dynamics.

The core design tension has always been:

- keep the observation close to the physical sensor,
- keep the reward informative enough for RL,
- and avoid silently mixing inconsistent geometry, sensor, and action conventions.

## 2. Early Reward-Centric Phase

The earlier branch relied heavily on reward shaping to get obstacle avoidance working.

### 2.1 Path-Following Reward

The path term had the paper-style form:

```text
r_pf = -1 + (U_norm * cos(chi_tilde) + 1) * (exp(-gamma_e * |y_e|) + 1)
```

where:

- `U_norm` is normalized speed,
- `chi_tilde` is course error relative to the path,
- `y_e` is signed cross-track error.

### 2.2 Obstacle-Avoidance Reward

The obstacle term used beamwise inverse-distance penalties:

```text
w_i   = 1 / (1 + |gamma_theta * theta_i|)
pen_i = 1 / (gamma_x * max(d_i, epsilon_x)^2)
r_oa  = - sum(w_i * pen_i) / sum(w_i)
```

This was conceptually clean, but in a narrow channel it was noisy:

- border returns constantly contributed penalty,
- the agent often learned weak or ambiguous side choice,
- benchmark success frequently regressed after early improvements.

## 3. State-Consistency Fixes

Before larger redesigns, several state bugs and omissions were fixed.

### 3.1 Restoring Goal-Heading Updates

`target_heading` was restored so it updated every step from the current pose to the goal.

### 3.2 Signed Cross-Track Error

`tgt` moved from unsigned nearest distance to signed cross-track error:

```text
y_e = ((x_g - x_s)(y - y_s) - (y_g - y_s)(x - x_s)) / ||goal - start||
```

This gave the policy the side-of-path information it needs for recovery.

### 3.3 Hidden Dynamics In Observation

To make the observation closer to Markov, the following were added:

- `u_body`
- `v_body`
- `rudder_state`

These exposed actuator lag and sway dynamics that were previously hidden inside the ship model.

## 4. Sector and Guidance Experiments

Several intermediate branches explored:

- left / center / right LiDAR summaries,
- threat-gated directional rewards,
- recentering terms,
- temporary heading guides,
- simple two-layer guidance schemes.

These experiments helped diagnose the problem, but they also showed an important pattern:

- many runs achieved a useful behavior early,
- then regressed later under PPO,
- and reward complexity often made the behavior harder to interpret rather than more robust.

The main lesson from that phase was:

```text
reward shaping alone was not the real bottleneck
```

That pushed the design toward a cleaner paper-style baseline with better observation handling.

## 5. LiDAR Realism Rework

One of the biggest changes was aligning the simulated LiDAR with the physical sensor limitation:

```text
valid sensor range: 1 m <= d <= 16 m
```

### 5.1 Observed LiDAR

In the current `asv_lidar.py`, the observed beam value is:

```text
reported_range =
    16, if true_range < 1
    16, if true_range > 16
    true_range, otherwise
```

So the observation is intentionally ambiguous:

- “too close to measure” and
- “no obstacle within max range”

both appear as `16 m`.

### 5.2 Internal True LiDAR

To avoid destroying the reward gradient near obstacles, `asv_lidar.py` now stores:

- `ranges`: the hardware-faithful observed scan
- `true_ranges`: the unclipped simulated ranges

This separation was one of the most important architectural improvements in the project.

Why it matters:

- observation stays deployable,
- reward can still respond smoothly to near obstacles,
- collision truth no longer depends on sensor aliasing.

## 6. Geometry-Based Collision Restored

Earlier branches temporarily used LiDAR-threshold collision logic. That was later removed because it made the environment depend on a sensor ambiguity rather than true contact.

The current branch restores geometry-based collision:

```text
terminated = collided_by_geometry or reached_goal
```

The hull polygon is built from:

- `VESSEL_LENGTH`
- `VESSEL_WIDTH`
- `HULL_MARGIN`
- `HULL_FORWARD_SHIFT`

and then checked against:

- obstacle polygons
- map border

using polygon-intersection logic.

This is the current recommended split:

- observation: sensor-faithful
- reward: uses internal true distances
- collision: geometry truth

## 7. Current Paper-Style Reward

The current branch returns to a simpler paper-style reward with fixed lambda.

### 7.1 Path-Following Term

The current path term is:

```text
r_pf = -1 + (U_norm * cos(chi_tilde) + 1) * (exp(-GAMMA_E * |tgt|) + 1)
```

with:

```text
if U > 1e-6:
    course_deg = atan2(dx_pos, dy_pos)
else:
    course_deg = heading_deg

path_course_deg = atan2(goal_x - start_x, goal_y - start_y)
chi_tilde_deg = wrap(course_deg - path_course_deg)
cos(chi_tilde) = cos(deg2rad(chi_tilde_deg))
```

This is important because the heading convention of the repo is:

```text
heading 0 deg means motion along +y
dx = d * sin(h)
dy = d * cos(h)
```

### 7.2 Obstacle-Avoidance Term

The current OA term uses the internal true LiDAR distances:

```text
w_i   = 1 / (1 + |GAMMA_THETA * theta_i|)
x_i   = clip(true_range_i, EPSILON_X, LIDAR_RANGE)
pen_i = 1 / (GAMMA_X * x_i^2)
r_oa  = - sum(w_i * pen_i) / sum(w_i)
```

This keeps the paper-style form while avoiding blind-zone reward collapse.

### 7.3 Living Penalty

The living penalty is:

```text
r_exist = -lambda * (2 * ALPHA_R + 1)
```

### 7.4 Total Reward

The total reward is now:

```text
if collided:
    reward = (1 - lambda) * R_COLLISION
else:
    reward = lambda * r_pf + (1 - lambda) * r_oa + r_exist
```

The current branch does not add a separate goal bonus term.

This was a deliberate simplification:

- fewer overlapping incentives,
- easier benchmark interpretation,
- closer alignment with the reference paper baseline.

## 8. Current Observation Design

The current observation keeps both raw LiDAR and compact summaries.

### 8.1 Raw and Sector LiDAR

The observation includes:

- `lidar`: raw 90-beam observed LiDAR
- `lidar_sectors`: 18 pooled sector values

Sector pooling is computed from observed LiDAR only:

```text
sector_value = percentile_10(beam_ranges_in_sector)
```

This gives the policy:

- full beam-level structure,
- plus a cheaper coarse representation.

### 8.2 LiDAR Summary Features

The observation also includes compact front/side summaries:

- `front_min`
- `front_p10`
- `left_min`
- `right_min`
- `near_flag`

These are built from the observed LiDAR with valid-range filtering:

```text
valid = 1 <= lidar_d <= 16
```

Then:

```text
front_min = min(valid front beams) or 16
front_p10 = p10(valid front beams) or 16
left_min  = min(valid left beams) or 16
right_min = min(valid right beams) or 16
near_flag = 1 if front_min < 2 else 0
```

This keeps the observation lightweight while still exposing near-field structure.

### 8.3 Vehicle-State Channels

The observation also contains:

- `pos`
- `hdg`
- `u_body`
- `v_body`
- `rudder_state`
- `speed`
- `dhdg`
- `tgt`
- `target_heading`
- `distance_to_goal`

That combination is the current compromise between:

- sensor realism,
- low-dimensional navigation cues,
- and approximate Markov sufficiency.

## 9. Training Stages

The current branch now supports three explicit training stages inside `rl_env.py`.

### 9.1 Stage 1

Fixed start / goal and one deterministic obstacle:

```text
start = (5.0, 2.0)
goal  = (5.0, 20.0)
obstacle = [(3.5, 12.0), (4.5, 12.0), (4.5, 13.0), (3.5, 13.0)]
```

This is the simplest curriculum setup.

### 9.2 Stage 2

Fixed start / goal, but one random obstacle sampled in a bounded region.

### 9.3 Stage 3

The full random training setup:

- random start / goal
- random number of obstacles

The goal of this design is to stop asking PPO to solve the hardest distribution from the very first update.

## 10. Ship-Model Selection Layer

Another major improvement was the addition of:

- `ship_model_selector.py`

This file now provides a one-line switch between:

- `standard_3dof`
- `bluefin_4dof`

and exports:

- the active `ShipModel`
- model-specific `RPM_MAX`
- model-specific `U_MAX`
- body-state accessors
- a rudder-state accessor
- geometry constants

This avoids scattering model-specific assumptions throughout the repo.

## 11. Bluefin 4DOF Compatibility

The raw `ship_model_bluefin_4dof.py` is not a perfect direct replacement by itself because:

- its internal heading convention differs from the repo convention,
- and its raw startup behavior can be numerically aggressive at full rudder from rest.

The selector wrapper solves this by:

- converting between the raw and repo heading conventions,
- exposing the same public interface as the standard model,
- damping rudder authority at very low speed for stability.

So in the current branch:

```text
Bluefin 4DOF is intended to be used through ship_model_selector.py
```

not imported raw into `rl_env.py`.

## 12. Repo-Wide Rudder-Sign Cleanup

One subtle but important consistency fix was the rudder convention.

Previously, the main models behaved such that:

```text
negative command -> starboard
positive command -> port
```

which was opposite to the desired nautical convention.

The current public convention is now:

```text
negative rudder = port
positive rudder = starboard
```

This was implemented at the ship-model interface level, not only in the RL env.

That means:

- `ShipModel.update(..., rud, ...)` now interprets the public sign consistently
- `state_dict()["rudder_deg"]` also reports the public sign consistently
- `rudder_state` in the observation matches the same convention

This removed a hidden inconsistency that would otherwise keep leaking into downstream scripts.

## 13. Benchmark and Logging Evolution

`train_test_asv.py` evolved from a simple train/test script into a proper benchmark harness.

The current benchmark now logs:

- success rate
- collision rate
- border rate
- timeout rate
- mean reward
- mean speed
- mean distance to goal
- front clearance metrics
- reward components
- cross-track and heading errors

It also:

- saves benchmark history to JSON
- appends summary rows to CSV
- exports plots
- saves the best benchmark checkpoint

The earlier automatic early-stop logic was later removed so longer runs can proceed without forced termination.

## 14. Current Design Summary

The current branch can be summarized as:

### Environment

- geometry-based collision
- goal radius = `VESSEL_LENGTH / 2`
- stage-based training progression

### Sensor

- raw LiDAR observation is hardware-faithful
- blind-zone ambiguity is preserved in the observation
- true unclipped LiDAR is kept separately for reward shaping

### Reward

- paper-style `r_pf`
- paper-style `r_oa`
- living penalty
- fixed `lambda`
- no extra planner-specific shaping layers

### Observation

- raw 90-beam LiDAR
- sector-pooled LiDAR
- front/side LiDAR summaries
- vehicle pose, heading, speed, yaw rate
- hidden body-state channels

### Dynamics

- single-point model switching through `ship_model_selector.py`
- consistent public rudder sign across the main stack

## 15. Main Lessons From The Current Branch

The strongest lessons from the evolution so far are:

1. Sensor realism and reward smoothness should be separated.
   Observation should mimic the real sensor; reward should still have access to simulator truth when needed.

2. Collision should come from geometry, not sensor aliasing.
   A blind zone is an observation problem, not a contact-truth problem.

3. Reward simplification helped more than stacking more shaping terms.

4. Hidden-state exposure and sign consistency matter.
   Small state and convention mismatches can silently block learning.

5. Curriculum is necessary.
   Stage-based training is a cleaner way to build capability than repeatedly overfitting the reward.

## 16. Current Status

The current `rl_env.py` is best viewed as a clean, sensor-faithful paper-style baseline with:

- geometry-grounded termination,
- LiDAR observation plus compact summaries,
- internal true-distance reward shaping,
- stage-based training,
- and selector-based support for both 3DOF and Bluefin 4DOF models.

That makes it a much stronger foundation for the next round of experiments than the earlier reward-heavy branches.
