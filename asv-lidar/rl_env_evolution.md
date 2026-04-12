# `rl_env.py` Evolution Notes

This note summarizes the significant evolution of `asv-lidar/rl_env.py` and the related files that changed with it:

- `asv-lidar/train_test_asv.py`
- `asv-lidar/asv_lidar.py`

It focuses on the major design phases from the first reward-function analysis through the latest LiDAR-sector observation design. It intentionally skips small coefficient-only edits unless they changed the method in a meaningful way.

## 1. Project Context

The environment models an autonomous surface vessel (ASV) that must:

1. follow a nominal path from start to goal,
2. avoid static obstacles,
3. use LiDAR-like sensing and realistic ship dynamics.

The environment sits on top of:

- `ship_model.py`: nonlinear vessel dynamics and actuator response,
- `asv_lidar.py`: LiDAR beam simulation,
- `test_run.py`: fixed evaluation scenarios,
- `train_test_asv.py`: PPO/SAC training and benchmark logging.

The main research problem inside `rl_env.py` has been:  
how to shape the reward and observation so the policy both tracks the route and makes decisive obstacle-avoidance decisions.

## 2. Baseline Environment And Initial Reward

### 2.1 Initial Observation Structure

The earlier environment exposed a fairly compact observation:

- full LiDAR beam vector,
- ASV position and heading,
- yaw-rate,
- speed,
- path offset `tgt`,
- target-heading error.

This was enough to make learning possible, but not always enough to make behavior stable. In particular, the policy had to infer too much from raw beams alone.

### 2.2 Initial Path-Following Reward

The path-following reward had the general form

\[
r_{pf} = -1 + (U_n \cos\tilde{\chi} + 1)\left(e^{-\gamma_e |y_e|} + 1\right)
\]

where:

- \(U_n\) is normalized speed,
- \(\tilde{\chi}\) is path/course error,
- \(y_e\) is cross-track error,
- \(\gamma_e\) weights the cross-track penalty.

This structure was retained because it captures the intended guidance behavior well:

- move forward along the path,
- prefer low course error,
- prefer small lateral error.

### 2.3 Initial Obstacle-Avoidance Reward

The earliest OA reward used all LiDAR beams with an angle weight and inverse-distance penalty:

\[
w_i = \frac{1}{1 + |\gamma_\theta \theta_i|}
\]

\[
p_i = \frac{1}{\gamma_x \max(d_i,\epsilon_x)^2}
\]

\[
r_{oa}^{old} = -\frac{\sum_i w_i p_i}{\sum_i w_i}
\]

where:

- \(d_i\) is beam distance,
- \(\theta_i\) is beam angle,
- \(w_i\) downweights side beams,
- \(p_i\) penalizes short distances.

### 2.4 Why This Was Revisited

This worked in principle, but in a narrow channel it was too noisy:

- side walls constantly contributed penalty,
- tiny heading changes created reward jitter,
- the policy could not clearly distinguish “corridor walls” from “collision threat ahead”.

That led to the first major OA redesign.

## 3. Phase 1: Consistency Fixes In State And Observation

Before deeper reward redesign, several consistency issues had to be fixed.

### 3.1 Normalized `dhdg`

`dhdg` was changed from a raw yaw-rate style exposure to a normalized signal:

\[
dhdg = \mathrm{clip}\left(\frac{\dot{\psi}}{180}, -1, 1\right)
\]

Why:

- signed yaw rate matters,
- normalization helps PPO numerics,
- the previous observation range did not match the actual ship model behavior.

### 3.2 Restoring `angle_diff`

`target_heading` had to be recomputed every step using the current pose and goal:

\[
\Delta \psi_{goal} = \mathrm{wrap}_{[-180,180]}\left(\psi_{target} - \psi\right)
\]

Why:

- a stale heading-to-go observation weakens goal reacquisition,
- this directly contributed to some overshoot and border-drift behavior.

### 3.3 Signed Cross-Track Error

`tgt` moved from an unsigned nearest-path distance to a signed cross-track error:

\[
y_e = \frac{(x_g-x_s)(y-y_s) - (y_g-y_s)(x-x_s)}{\sqrt{(x_g-x_s)^2+(y_g-y_s)^2}}
\]

Why:

- the policy needs to know which side of the path it is on,
- absolute error alone cannot support efficient recovery.

These changes did not solve obstacle avoidance by themselves, but they improved state consistency and removed several avoidable learning handicaps.

## 4. Phase 2: OA Redesign From All-Beam Penalty To Clearance Barrier

To reduce the narrow-channel noise, OA shifted from “all beams always contribute” to “forward clearance triggers the penalty”.

### 4.1 Front-Sector Clearance

A forward sector was defined, originally with a half-angle of roughly \(45^\circ\):

\[
\mathcal{F} = \{i \mid |\theta_i| \le \theta_f\}
\]

Then a robust summary statistic was used instead of a beamwise sum:

\[
d_{front} = \mathrm{Percentile}_{10}\{d_i\}_{i \in \mathcal{F}}
\]

### 4.2 Barrier-Style OA Term

The next OA idea became:

\[
\delta_{oa} = \max\left(0,\frac{d_{safe}-d_{front}}{d_{safe}}\right)
\]

\[
r_{oa} = -\delta_{oa}^2
\]

Why this helped:

- no penalty when there is adequate forward clearance,
- much less wall-induced reward jitter,
- cleaner learning signal than the weighted inverse-distance sum.

### 4.3 Limitation

This formulation reduced noise, but it was too non-directional:

- it could say “front is blocked”,
- but not clearly “turn left” or “turn right”.

That led to a directional OA redesign.

## 5. Phase 3: Directional OA With Left / Center / Right Sectors

The next important step was to introduce explicit sector reasoning.

### 5.1 Sectorization

The front swath was split into:

- left sector,
- center sector,
- right sector.

The main sector clearances were defined from a 10th-percentile beam statistic:

\[
d_L,\; d_C,\; d_R
\]

with

\[
d_C = \mathrm{Percentile}_{10}\{d_i\}_{|\theta_i|\le\theta_c}
\]

and similar definitions for \(d_L\) and \(d_R\).

### 5.2 Center-Barrier OA

The center sector became the main collision barrier:

\[
z_c = \mathrm{clip}\left(\frac{d_{warn} - d_C}{d_{warn}-d_{crit}}, 0, 1\right)
\]

\[
r_{center} = -k_c z_c^2
\]

Near-collision strengthening was added with a second term:

\[
z_n = \mathrm{clip}\left(\frac{d_{near} - d_C}{d_{near}}, 0, 1\right)
\]

\[
r_{near} = -k_n z_n^2
\]

### 5.3 Directional Steering Reward

Directional asymmetry was then expressed through

\[
g = \tanh\left(\frac{d_R - d_L}{s_g}\right)
\]

and aligned with the rudder command:

\[
r_{dir} = k_d \, z_c \, g \, u
\]

where \(u\) is a bounded steering-alignment term.

### 5.4 Threat-Adaptive Blend

Instead of one fixed reward blend, the reward switched between “clear-water” and “threat” modes:

\[
threat = \max(z_c, z_n)
\]

\[
\lambda = \lambda_{clear} - (\lambda_{clear} - \lambda_{threat}) \cdot threat
\]

and the total reward became

\[
r = \lambda r_{pf} + (1-\lambda) r_{oa} + r_{exist}
\]

This was a major conceptual improvement because:

- path following dominates when safe,
- OA gains authority under real threat.

## 6. Phase 4: Geometry-Based OA Reward After Real LiDAR Limits Were Added

### 6.1 Sensor Model Change

`asv_lidar.py` was updated to match the real sensor behavior:

- valid range only in \([1\,m, 16\,m]\),
- any return below \(1\,m\) or above \(16\,m\) is reported as \(16\,m\).

That means the raw LiDAR cannot distinguish:

- “too close to measure” and
- “nothing detected”.

### 6.2 Consequence For OA Reward

This made raw LiDAR unreliable for reward shaping near collision.  
So the environment adopted a split:

- **reward shaping** uses geometry-based clearances,
- **observation** is allowed to move closer to the real sensor.

This was implemented through helper logic that computes true unsaturated geometry ranges for the relevant angles inside `rl_env.py`.

Why this was useful:

- the policy still gets a stable training signal,
- the reward is not fooled by the LiDAR blind zone.

## 7. Phase 5: Reward Simplification

After many coefficient-level reward tweaks, it became clear that too much shaping was making the system harder to reason about.

The reward was simplified to a cleaner core.

### 7.1 Current Path-Following Reward

The current path term remains:

\[
r_{pf} = -1 + (U_n \cos\tilde{\chi} + 1)\left(e^{-\gamma_e |y_e|} + 1\right)
\]

### 7.2 Current OA Reward

The current OA reward in the simplified design is:

\[
r_{oa} = r_{center} + r_{dir} + r_{near} + r_{speed}
\]

with

\[
r_{speed} = -k_v \, threat \, U_n^2
\]

The important point is that several extra shaping terms were removed, including:

- over-specific goal shaping terms,
- multiple border-specific reward terms,
- several commitment / wrong-way penalty variants,
- extra RPM-threat shaping terms.

### 7.3 Current Total Reward

The current total reward is now intentionally simple:

\[
r =
\begin{cases}
R_{collision}, & \text{if collision}\\[4pt]
R_{goal}, & \text{if goal reached}\\[4pt]
\lambda r_{pf} + (1-\lambda)r_{oa} - \alpha_R, & \text{otherwise}
\end{cases}
\]

Why this was an improvement:

- fewer conflicting incentives,
- easier benchmark interpretation,
- easier comparison between runs.

## 8. Phase 6: Observation Redesign

Once reward-only tuning started to plateau, the next major shift was to improve the observation.

### 8.1 Why Observation Needed To Change

The policy previously had to infer too much from raw beams and a few navigation variables.  
So several compact low-dimensional features were added:

- normalized distance to goal,
- compact sector summaries,
- actuator-state exposure.

### 8.2 Added Compact Features

The observation now includes:

- `goal_dist`
- `left_clear`, `center_clear`, `right_clear`
- `rudder_state`
- `rpm_state`

These are cheap to compute and much easier for PPO to exploit than forcing it to infer everything only from the beam vector.

### 8.3 First Version: Geometry-Derived Sector Observation

The first compact sector observation used geometry-based sector summaries.  
This helped learning, but it was not fully deployable because the real vessel will only have sensor returns, not simulator geometry.

## 9. Phase 7: LiDAR-Derived Sector Observation With Hysteresis

To make the observation more realistic for deployment, the compact sector features were switched to LiDAR-derived values.

### 9.1 Sector Computation

The front sector is split into left, center, right using LiDAR beam angles, and each sector uses a percentile-based clearance estimate.

However, because the real LiDAR reports any reading below \(1\,m\) as \(16\,m\), a pure instantaneous statistic is too brittle.

### 9.2 Hysteresis / Memory

To avoid an immediate jump from “near obstacle” to “fully clear”, a bounded recovery rule is used:

\[
d^{mem}_{k+1} = \min\left(d^{inst}_{k+1},\; d^{mem}_{k} + \Delta_{rec}\right)
\]

where:

- \(d^{inst}\) is the current LiDAR sector clearance estimate,
- \(d^{mem}\) is the memory-filtered value,
- \(\Delta_{rec}\) is a small recovery increment per step.

This means:

- sectors can collapse quickly when danger appears,
- but they recover gradually after the sensor loses close contact.

### 9.3 Why This Was Better Than Pure LiDAR-Sector Smoothing Alone

Hysteresis preserves deployability, but by itself it can flatten the observation too much.  
So the latest design adds:

- smoothed sector clears,
- instantaneous sector clears,
- explicit left-right asymmetry.

## 10. Latest Observation Design

The current observation uses both persistent and fresh LiDAR-sector information:

- `left_clear`, `center_clear`, `right_clear`: smoothed sector clears,
- `left_clear_instant`, `center_clear_instant`, `right_clear_instant`: instantaneous sector clears,
- `gap_asymmetry`: directional left-right clearance contrast.

The asymmetry term is

\[
gap\_asymmetry = \mathrm{clip}\left(\frac{d_R^{mem} - d_L^{mem}}{d_{warn}}, -1, 1\right)
\]

Why this is the current preferred approach:

- it stays sensor-faithful for deployment,
- it still carries persistent information through the blind zone,
- it gives the policy an explicit directional cue,
- it is computationally cheap.

## 11. Related File Evolution

### 11.1 `asv_lidar.py`

This file changed in one major way:

- it now enforces the real sensor range model:

\[
d_{reported} =
\begin{cases}
16, & d < 1 \\
16, & d > 16 \\
d, & 1 \le d \le 16
\end{cases}
\]

This was essential because many reward/observation choices depend on whether the LiDAR is idealized or realistic.

### 11.2 `train_test_asv.py`

The trainer evolved from a simple train/test harness into a benchmark-and-diagnostics tool.

Major additions included logging of:

- success / obstacle / border rates,
- path-following and OA contributions,
- OA activation fraction,
- sector-clearance summaries,
- actuator-state summaries,
- latest LiDAR-sector observation fields.

This file became important because the learning problem was not separable from the diagnosis problem.  
Without these metrics, many regressions would have been hard to interpret.

## 12. What Improved Over Time

The significant improvements across these phases were:

1. **Less reward noise**
   The OA term moved away from a global all-beam penalty toward a threat-focused barrier.

2. **Better directional reasoning**
   Left/center/right sectorization made it possible to encode “which side is more open”.

3. **Better state consistency**
   Normalized yaw rate, restored target-heading updates, and signed cross-track error all removed silent learning issues.

4. **Cleaner reward**
   Simplifying the total reward made it easier to interpret benchmark behavior and reduced incentive conflicts.

5. **More informative observation**
   Compact navigation and sector features reduced the burden on the policy network.

6. **Better deployment alignment**
   The latest observation is now based on LiDAR-derived sector summaries rather than simulator geometry.

## 13. Main Lesson From The Evolution

The biggest lesson was that reward shaping alone was not enough.

The environment improved most when the design moved in this order:

1. fix inconsistent state signals,
2. reduce OA reward noise,
3. simplify the reward,
4. add better low-dimensional observations,
5. make those observations closer to the real sensor.

In other words:

- first reward redesign helped,
- but the larger breakthroughs came when reward and observation were treated together.

## 14. Current Status

The present design can be summarized as:

- **reward**: geometry-stable, threat-adaptive, relatively simple,
- **observation**: LiDAR-faithful but augmented with compact sector summaries and directional cues,
- **benchmarking**: rich enough to diagnose whether failures come from path following, obstacle negotiation, or sensor compression effects.

That makes the current version a much stronger foundation for the next stage, including later transfer to the dynamic-obstacle environment.
