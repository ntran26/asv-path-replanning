# Paper 3 master artifact — verified technical facts

> **Kept current (decision, 2026-10-04).** This file describes only the best formulation: **baseline-v3**, the kept SAC 3 M policy (`configs/baseline_v3.json`, `src/formulation_v3.py`). Earlier versions are not described. Baseline-v4.3 was trained in full and not adopted (`planning_records/BASELINE_V4_PLAN.md`, section 5e), so baseline-v3 is the campaign's formulation. Terminology (2026-10-08): *decoupled* / *coupled* scenarios and the *validation set*.

**Written 2026-09-29 from the project's code**, as the source for the
master explainer artifact (`MASTER_ARTIFACT/MASTER_ARTIFACT_BRIEF.md`). Every number here is
what the simulator, controllers and tracker actually use. The observation,
encounter geometry and reward equations are in the companion file
`FORMULATION_EQUATIONS.md` and are not repeated here. Where a citation is
marked [VERIFY], check the reference before relying on it.

Two presentation rules from the author:

- **Vessel model:** describe the model and its identified parameters; do not
  mention the data files it was identified from.
- **Training/evaluation sets:** do not mention any "field deployment" test set.
  Stages 6–7 are described as training on more, and more evenly spread,
  three-obstacle layouts (Section 6 below).

---

## 1. Vessel and dynamics (Section 3 of the artifact)

**Platform.** Bluefin model-scale ASV: length overall L = 1.725 m, beam
B = 0.50 m, draft 0.193 m, mass m = 64.55 kg. Single propeller and a stern
rudder (±40°). A 360° 2D LiDAR sits at the bow (0.8625 m ahead of the
reference point). The target vessel is the same class of vessel and uses the
same hull outline (1.725 × 0.50 m).

**State** (7 values): surge u, sway v (body-frame velocities, m/s), yaw rate r
(rad/s), heading ψ (compass: 0 = north, clockwise positive), actual rudder angle
δ, position (x east, y north).

**Kinematics** (body velocities to world position):

  ẋ = u sin ψ + v cos ψ,  ẏ = u cos ψ − v sin ψ,  ψ̇ = r

**Kinetics** — three coupled equations (Newton's second law in a rotating body
frame, in the form of Fossen, 2011/2021, with MMG-style hydrodynamic terms,
Yasukawa & Yoshimura, 2015):

  m₁₁ u̇ = T(n, u) + X_damp + X_rud + m₂₂ v r
  m₂₂ v̇ = Y_damp + Y_rud − m₁₁ u r
  m₃₃ ṙ = N_damp + N_drift + N_Munk + N_rud

Masses include "added mass" (the water the hull must accelerate with it):

| Symbol | Value | Meaning |
|---|---|---|
| m₁₁ = m + X_u̇-added (3.662) | 68.21 kg | effective mass pushing forward |
| m₂₂ = m + Y_v̇-added (62.74) | 127.29 kg | effective mass pushing sideways (a hull resists sideways acceleration much more) |
| m₃₃ = I_z + J_z | 10.23 kg·m² | effective yaw inertia |

Forces and moments (identified parameter values in the table below):

- Propeller thrust: T = T₁₂ (n/12)² (1 + t_b e^{−u/u_b}), n in "rpm-units"
  (command scale 0–24, 12 = nominal field setting). The identified low-speed
  boost is t_b = 0 (its speed scale u_b = 0.5 m/s is then inert), so
  T = T₁₂ (n/12)².
- Surge damping: X_damp = −X_uu u|u| − X_vv v² − X_rr (L r)²
  (hull drag, plus speed lost to sideways and turning cross-flow).
- Rudder drag: X_rud = −X_δ |F_N| |sin δ|.
- Sway damping: Y_damp = −Y_v u v − Y_vv v|v|.
- Yaw damping: N_damp = −N_r u r − N_rr r|r|; drift moment N_drift = −N_uv u v.
- Munk moment: N_Munk = (m₁₁ − m₂₂) u v ≈ −59.1 u v (a real, destabilizing
  moment on any elongated body moving at an angle; it completes the Coriolis
  terms so the model conserves energy).
- Rudder normal force (a small wing):

    u_R = max(0.05, (1 − w_R) u + k_race √(n/12)),  v_R = v + l_R r
    α_R = δ − atan2(v_R, u_R)
    F_N = ½ ρ A_R f_α u_R² sin α_R

  with ρ = 1000 kg/m³, A_R = 0.0091 m², lift slope f_α = 2.693, wake
  fraction w_R = 0.22, l_R = −0.777 m. Rudder side force and yaw moment:
  Y_rud = −(1 + a_H) k_R F_N cos δ,  N_rud = −ℓ k_R F_N cos δ, with hull
  interaction a_H = 0.444 and lever arm ℓ = 1.207 m.

- **Rudder servo (actuator dynamics):** the commanded angle
  δ_c = −40° × a_δ passes through a transport delay t_d = 0.73 s,
  then a first-order lag with time constant τ = 0.90 s and a rate limit:

    δ_{k+1} = δ_k + clip((δ_c(t − t_d) − δ_k)(1 − e^{−Δt/τ}), ±δ̇_max Δt)

  (identified servo rate limit δ̇_max ≈ 2,985°/s, effectively non-binding; the
  field bridge's 20°/s command limiter is off in simulation; the delay and lag
  dominate). **Sign convention:** in these equations δ is positive to port (the
  identified model's and the transmitted command's convention), hence the minus
  sign: a positive policy command a_δ gives a negative δ and turns the vessel to
  starboard, the convention every steering law in Section 4 uses.

- **Integration:** fourth-order Runge–Kutta, sub-steps ≤ 0.05 s inside each
  0.1 s physics step; five physics steps per 0.5 s decision. Surge is clipped at
  ≥ 0 (the vessel does not go astern). The runtime safety layer's reverse thrust
  acts only as a braking force that removes forward momentum.

**Identified parameter values (nominal):**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| T₁₂ | 16.06 | N | thrust at 12 rpm-units |
| X_uu | 12.89 | N/(m/s)² | quadratic surge drag |
| X_vv | 50.64 | N/(m/s)² | surge drag from sway cross-flow |
| X_rr | 0.058 | N/(m/s)² | surge drag from yaw cross-flow |
| X_δ | 0.117 | — | rudder-induced drag |
| Y_v | 0.924 | N/(m/s)² | speed-dependent sway damping |
| Y_vv | 220.98 | N/(m/s)² | quadratic sway damping |
| k_R | 2.176 | — | rudder effectiveness |
| k_race | 0.0135 | m/s | propeller-race inflow at the rudder |
| N_r | 1.038 | N·m/((m/s)(rad/s)) | speed-dependent yaw damping |
| N_rr | 0.0173 | N·m/(rad/s)² | quadratic yaw damping |
| N_uv | −83.97 | N·m/(m/s)² | drift-induced yaw moment |
| τ | 0.895 | s | rudder servo lag |
| t_d | 0.730 | s | rudder transport delay |

**Resulting behavior:** steady speed 0.558 m/s at 6 rpm-units (the cruise used
in all training), 1.116 m/s at 12; steady yaw rate at full rudder about
10°/s at 12 rpm-units and about 5°/s at the 6 rpm-unit cruise (speed falls to
about 0.46 m/s in the turn); several seconds to build a turn (delay + lag +
inertia).

**Domain randomization:** each training episode draws a new parameter set.
Two of the 28 bootstrap solutions of the identification, θ_A and θ_B, are
picked uniformly at random (with replacement) and blended,
θ = λ θ_A + (1 − λ) θ_B with λ ~ U(0, 1); each parameter then gets Gaussian
jitter with standard deviation 5 % of its bootstrap standard deviation (scale
1.0), and keeps its sign. Blending whole parameter vectors keeps the
correlations between yaw parameters (k_R, N_r, N_uv), so every draw is a
physically consistent vessel (Tobin et al., 2017 for domain randomization).

**Sources:** Fossen, T. I. (2021). *Handbook of Marine Craft Hydrodynamics and
Motion Control*, 2nd ed., Wiley (3-DOF maneuvering models, added mass,
Coriolis and Munk terms, LOS guidance). Fossen (2011), 1st ed. Yasukawa &
Yoshimura (2015), MMG standard method, *J. Mar. Sci. Technol.* 20(1):37–52.
Paper 2 (Tran et al., 2026) for the platform.

## 2. Classical comparators (Section 4)

The reported comparators are LOS-PID + DWA (2.2) and the encounter-specific VO (2.4;
decision S18, 2026-10-20). COLREGs-VO (2.3) is not a reported comparator; its rule drives the
test set's reactive target. All comparators use the same perception (LiDAR scan, tracker, map boundary) as
the learned policies and the same actuator limits; they differ only in the
avoidance logic. Each is tuned on the decoupled part of the validation set (120 episodes) by
coordinate descent, scored by goal − 2 × collision, and pinned in a settings file whose digest
the code is checked against before a run.
- **Comparators v2 (2026-10-25), the reported settings:** `configs/comparators_v2.json` (digest
  `617782d1483005d7`).
  - The encounter-specific VO was searched on 60 of the 120 episodes (10 per class) over its
    horizon, hard gap, stand-on release, course-change weight and domain weight. Only the
    course-change weight moved, from the default 0.3 to 0.8 (score +0.350 → +0.475 on all 120).
  - Its passing side is judged on the engaged encounter (2.4 item 5).
  - The tuning used the baseline-v3 build of the validation episodes, the build that reproduces
    the learned policies' recorded validation runs.
  - LOS-PID + DWA is unchanged from v1: its v2 test-set run reproduces the v1 run in all 1,000
    episodes.
- **Comparators v1 (2026-10-19):** `configs/comparators_v1.json` (digest `0dd8eab26bbf954f`).
  COLREGs-VO raised its course-change weight from 0.3 to 0.8 (+0.525 → +0.575 on all 120); DWA
  kept its defaults (+0.650 either way).

Pseudocode: `PAPER/PAPER3_ALGORITHMS.md`, Algorithms A1 and A2.

**2.1 LOS guidance + heading PID (path following, shared).** Line-of-sight
guidance (Fossen, 2021): aim at a point Δ = 2.5 m ahead on the path,

  χ_d = χ_path + atan2(−e_y, Δ) − β̂,

where e_y is the cross-track error and β̂ the measured sideslip (capped ±20°).
Heading PID on the error e = χ_d − ψ (degrees):

  a_δ = clip(K_p e − K_d r[°/s] + I, −1, 1),  K_p = 1/35, K_d = 1/18,
  K_i = 1/600 (integral active only inside |e| < 10°, capped at 0.15).

Yaw-rate tracking (used by DWA): a_δ = r_d/10.4 + 0.08 (r_d − r) (deg/s); the
feed-forward 10.4 °/s per unit rudder is a scale on the candidate set, not a model
of the plant (the hull turns at about 5 °/s at full rudder at cruise). DWA's
rollouts run the same law on the same dynamics, so the trajectories it scores
are the ones the vessel sails; with this value the nine set-points spread over
about 9–36° of heading change in 6 s at cruise (check of 2026-10-10,
`results/classical/dwa_yaw_gain_check.md`).
Speed: rpm = u_d / (0.093 m/s per rpm) + 6 (u_d − u), clipped to 0–12 rpm.

**2.2 LOS-PID + Dynamic Window Approach (DWA)** (Fox, Burgard & Thrun, 1997).
LOS-PID drives the vessel until a track or a scan point comes within 9 m (a
point no more than 1 m abaft the own-ship centre), or the hull comes within
1.0 m of the boundary. Then, each decision:

1. **Candidates:** yaw-rate set-points r_d ∈ {−9, −6.75, …, +9} °/s × speeds
   {1.0, 0.5, 0} × U_nom × hold times {3, 6} s, plus plain LOS-PID at full
   and half U_nom.
2. **Dynamic window through the real dynamics:** each candidate is simulated
   closed-loop on the vessel model (delay, lag, inertia) for up to 16 s — not
   as an ideal arc, because this hull needs about 5 s to build a turn. After the
   hold time the prediction hands back to LOS-PID.
3. **Admissibility:** inadmissible if the predicted hull comes within 0.20 m of
   a scan point or within 0.05 m of the boundary over the first 5.5 m of travel
   (the map polygon is exact, the scan is not), or within 0.20 m of a tracked
   target's hull (constant velocity) within 10 s.
4. **Objective** (maximized over admissible candidates):

   G = 1.0·heading + 1.0·clearance + 0.5·velocity + 0.6·path − 0.25·smoothness

   heading = 1 − |LOS heading at the end point − predicted heading|/π;
   clearance = min(clearance, 1.5 m)/1.5 m, with clearance measured over the
   same windows as admissibility (fixed obstacles and the boundary over the
   first 5.5 m of travel, the tracked target over the first 10 s; the boundary
   clearance enters raised by 0.15 m, so its 0.05 m limit sits on the same
   0.20 m scale);
   velocity = mean predicted speed over the first 10 s/U_nom; path = 1 − mean
   |cross-track| over the first 5.5 m of travel/2 m;
   smoothness = |r_d − r_d,prev|/(18 °/s) when both the candidate and the last
   choice are yaw-rate set-points, 0 when both are plain LOS-PID, 1 when one of
   each, plus |speed fraction − last speed fraction|; 0 at the first decision
   of an avoidance.
5. **Fallback** if none is admissible: the candidate whose first violation
   comes latest, then most clearance, then the highest objective.

**DWA knows obstacles, not rules** — it is the "no COLREGs" reference.

**2.3 COLREGs-VO** (Kuwata et al., 2014; velocity obstacles after Fiorini &
Shiller, 1998) — not a reported comparator (S18); described because the test set's reactive
target (`T-RE`) runs this rule from its own side:

1. **Candidates:** course offsets −90° … +90° in 5° steps about the present
   heading, plus the LOS course; speeds {1, 0.75, 0.5, 0.25, 0} × U_nom.
2. **Own-ship motion model:** each candidate is a turn toward the new course at
   6°/s after a 1 s lag, with speed approaching the new value exponentially
   (time constants 60 s down, 20 s up); sampled every 0.25 s.
3. **Velocity obstacle (hard):** a candidate is rejected if, holding it, the
   own hull comes within 0.20 m of a target's hull within 20 s (target at
   constant velocity), or within 0.15 m of a scan point / 0.05 m of the
   boundary within 10 s.
4. **Classification (open-water roles):** α is the target's bearing from the
   own heading and β the own ship's bearing from the target's heading (both
   wrapped to ±180°). Being overtaken if |α| ≥ 112.5°; overtaking if
   |β| ≥ 112.5° (the own ship more than 22.5° abaft the target's beam) and the
   own ship is faster; head-on if |α| ≤ 22.5° and |β| ≤ 22.5° (each vessel
   within 22.5° of the other's bow); otherwise crossing — give-way if the
   target is to starboard (α > 0), stand-on if to port. The roles are
   recomputed every decision from the instantaneous heading, with no hold,
   hysteresis or latch.
5. **COLREGs constraint (give-way classes):** while the encounter is "live"
   (closing, predicted miss < 2.5 m within 20 s) the relative velocity must lie
   to starboard of the line of sight: (**p**_TS − **p**_OS) × (**v**_OS − **v**_TS)
   < 0 — "alter to starboard and pass astern". "Live" is judged per candidate,
   on that candidate's own velocity, so a candidate that by itself opens the
   pass beyond 2.5 m is exempt (a quick turn to port in a near-reciprocal
   head-on can qualify). Kuwata et al. judge it once on the encounter; decision
   A35 (2026-10-19) adopts their form for the reported comparator (settings v2, built after
   TQC seeds 1–2); the per-candidate form above is the frozen v1.
6. **Stand-on (Rule 17):** while a crossing target is to port or a target is
   overtaking it, the own ship keeps the preferred candidate (the LOS course at
   U_nom, i.e. it follows its path) as long as that candidate meets no hard
   limit within 12 s (targets) or 10 s (scan points and the boundary).
7. **Cost** over the allowed candidates:
   J = 1.0·|χ − χ_LOS|/π + 0.5·max(0, U_pref − U)/U_pref + 1.0·J_prox
   + 0.8·|χ − χ_prev|/π (course-change weight tuned from 0.3), with
   U_pref = U_nom = 0.558 m/s and J_prox = max(0, 0.60 m − c_min)/(0.60 m − 0.15 m),
   c_min being the candidate's smallest clearance to scan points and to the
   boundary (the boundary clearance raised by 0.10 m) over 10 s — a soft margin
   above the 0.15 m hard limit.
8. **Fallback:** if nothing satisfies both the velocity obstacle and the
   COLREGs constraint, drop the constraint rather than accept a collision
   (Kuwata's own rule); if nothing is safe, take the latest first contact.
9. **As the reactive target runs it** (`src/targets.py`, `T-RE`; checked 2026-10-24): every
   0.5 s the target classifies the encounter by the open-water roles with the own ship as its
   only traffic, and takes the course the rule picks, with these departures from items 1–8:
   - **course only:** its single candidate speed is its own speed; it never slows;
   - **preferred course:** its initial course, so it returns to that course once the pass is clear;
   - **side rule on the encounter:** the starboard constraint of item 5 is judged once for the
     encounter, not per candidate, and holds from the moment the encounter is live until the
     target stops closing (Kuwata's form, A35);
   - **hard clearance:** 0.75 m between hulls (the 1.25 m abeam semi-axis less 0.50 m), not
     0.20 m; with 0.20 m the target cut back 0.6 m ahead of the own ship;
   - **static panels:** ignored; at a fairway edge it stops maneuvering and keeps the edge heading;
   - **turn rate:** it turns toward the chosen course at 8°/s (it plans with item 2's 6°/s model).

**2.4 Encounter-specific VO** (an interpretation of Thyri & Breivik, 2022, a method for
confined waters; velocity obstacles after Fiorini & Shiller, 1998) — the reported
velocity-obstacle comparator (S18). Values are comparators v2 (2026-10-25, section 2 above):
the v1 defaults except the course-change weight, tuned from 0.3 to 0.8, and the passing side
judged on the engaged encounter (item 5).

1. **Encounter assessment:** the formulation's own encounter context (Algorithm 1): the
   latched class, required turn sense, engagement and in-extremis flag, the same assessment the
   learned policy observes.
2. **Candidates and motion model:** as COLREGs-VO: course offsets −90° … +90° in 5° steps
   about the present heading plus the LOS course, at {1, 0.75, 0.5, 0.25, 0} × U_nom; each swept
   along the turn model (1 s lag, 6°/s, speed time constants 60 s down, 20 s up), sampled every
   0.25 s.
3. **Hard velocity obstacles:** the own hull within 0.20 m of a target's hull within 20 s
   (constant velocity), or within 0.15 m of a scan point or 0.05 m of the boundary within 10 s.
4. **Encounter domain (soft):** the target's deepest penetration of the own-ship domain (the
   formulation's fore, aft and lateral semi-axes) scaled per class as (fore, aft, lateral):
   head-on (1.0, 1.0, 1.2), crossing (1.5, 1.0, 1.0), overtaking (1.0, 1.0, 1.2), being
   overtaken and none (1, 1, 1). Penalised, not forbidden, so in a narrow channel a small
   intrusion is allowed where no clear pass exists. The penetration is
   P_dom = max(0, 1 − ℓ), taken at its largest over the targets and the prediction, with
   ℓ = √((x/a_x)² + (y/a_y)²) the target centre's normalized elliptical distance in the own
   ship's frame at each point of the candidate's sweep: x ahead, y to starboard,
   a_x = fore × 3.14 m ahead of the own ship or aft × 1.57 m astern of it, a_y = lateral × 1.25 m.
   P_dom is 0 on the domain's edge and 1 at the own ship's reference point
   (`src/classical/encounter_vo.py`).
5. **Passing side (hard, give-way classes):** head-on: the target passes down the own ship's
   port side (Rule 14); crossing: pass astern of the target, from either side (the
   narrow-channel convention, S3); overtaking: pass on the side the required turn sense names.
   - **In v2**, while the formulation's encounter context reports the encounter engaged (Eq. 7:
     class ≠ none, 0 < TCPA ≤ 25 s, DCPA < 3.75 m, latched until clearing), the side binds every
     candidate that closes on the target. A candidate that does not close on it (slowing behind
     an overtaken vessel) has no pass to keep.
   - **v1** exempted any candidate whose own closest approach was beyond 2.5 m, so a quick turn
     to port could open a head-on pass. Example: a head-on target 0.3 m to starboard of the
     reciprocal track at 10 m; v1 turns to port (335°), v2 to starboard (35°).
   - The side is released in extremis, or when no clear candidate keeps the required side.
6. **Stand-on (being overtaken):** keep the LOS course at cruise while that velocity meets no
   hard limit within 12 s and keeps the passing side (Rule 17(a)); otherwise act as for any
   obstacle.
7. **Cost:** J = 1.0·|χ − χ_LOS|/π + 0.5·max(0, U_pref − U)/U_pref + 3.0·domain penetration
   + 0.8·|χ − χ_prev|/π (0.3 in v1) + 1.0·static penalty, the static penalty rising from 0 at 0.6 m
   clearance to scan points and the boundary to 1 at the 0.15 m hard limit.
8. **Fallback:** if nothing is clear, the candidate whose first violation comes latest, then
   the lowest cost. The choice is flown by the course autopilot (at 6°/s) and the speed loop
   of 2.1.

**Artificial potential field (APF) — background only.** Paper 2's comparator
(LOS + APF): the goal attracts, obstacles repel, and the vessel follows the
sum of the forces (Khatib, 1986). Fast and simple, but it gets stuck
in local minima between close obstacles and knows no rules. Explain it as
context; it is not a Paper 3 comparator.

## 3. Learners and training protocol (Section 5)

- Library: Stable-Baselines3 2.3.2 / sb3-contrib 2.3.0 (Raffin et al., 2021).
- **Common:** discount γ = 0.951 per 0.5 s step; reward normalization only
  (running scale, clip 10); 10 parallel simulators; 3.0 × 10⁶ environment
  steps (baseline-v3 curriculum, Section 6; stage 7 runs from 2.0 M to the end); validation-set evaluation every
  2 × 10⁵ steps; each seed represented by its best validation-set checkpoint
  (score = goal rate − 2 × collision rate); 3 seeds per learner; no setting
  tuned per learner or per seed. Settings are the Stable-Baselines3 /
  sb3-contrib defaults except: the common discount; PPO's 512 steps per
  simulator (library 2,048), minibatch 512 (library 64) and target KL 0.03
  (library none; added during development, when ten epochs over one rollout
  overfit it); SAC's and TQC's 10⁴ random warm-up steps (library 100) and one
  gradient step per collected transition, that is 10 per vectorized step of the
  10 simulators (`gradient_steps = 10`; the library's 1 per vectorized step would
  be one per 10 transitions). They are
  **not** the original papers' settings: PPO's paper used 2,048-step rollouts and
  minibatch 64 on MuJoCo, and the TQC paper 5 critics × 25 atoms with the
  number dropped set per environment; TQC here keeps the library's 2 critics
  (the comment in `src/train_formulation.py` calling them the paper's defaults
  is wrong; the frozen code is left unchanged).

| Learner | Type | Key settings |
|---|---|---|
| PPO (Schulman et al., 2017) | on-policy, policy gradient with clipped surrogate | lr 3e-4; 512 steps × 10 envs per update (5,120 transitions); minibatch 512; 10 epochs, so up to 100 gradient steps per update, fewer when the KL limit stops the epochs early; GAE λ = 0.95 (Schulman et al., 2016); clip ε = 0.2; target KL 0.03 |
| RecurrentPPO | on-policy, PPO + LSTM memory (Hochreiter & Schmidhuber, 1997) | PPO settings + one 256-unit LSTM layer for the actor and a separate one for the critic, placed after the features extractor and before the 256 × 256 heads |
| SAC (Haarnoja et al., 2018) | off-policy, maximum-entropy actor–critic | lr 3e-4; replay buffer 10⁶; batch 256; τ = 0.005; 10⁴ random warm-up steps; automatic entropy temperature with target entropy −2 (= −dim of the action; the library default, not overridden); 1 gradient step per environment step |
| TQC (Kuznetsov et al., 2020) | off-policy, distributional (quantile) SAC | SAC settings; 2 critics × 25 quantiles; drop the top 2 quantiles per critic |

- **Network (identical for all):** a scene encoder (lidar 27 + boundary 7 +
  ego 3 + path 3 + previous action 2 = 42 inputs → 128 → 128, two ReLU layers)
  and a target-slot encoder (target + context, 28 → 64
  → 64 → 32) whose input is gated by the presence bit; concatenated features
  (160) feed actor and critic heads of 256 × 256 ReLU.
- Development used PPO to iterate the formulation; the formulation itself
  references no algorithm.

## 4. LiDAR perception and tracking (Section 6)

**Sensor model:** 720 beams at 0.5° (full 360°), range 1.0–16 m (returns
closer than 1 m are lost in a dead zone). Pose noise 0.03 m / 0.2°, surge
noise 0.05 m/s, yaw-rate noise 1°/s. One scan per 0.5 s decision. Ranges are
exact ray casts: no per-beam range noise and no isolated beam dropout in
training (both left at zero until characterized from field data). No separate
perception-degradation study is part of the paper (decision, 2026-10-14).

**Pipeline, in order:**

1. **Gate:** discard returns outside the known navigable polygon (walls,
   people on the deck, anything beyond the water).
2. **Segment** (adaptive breakpoint detection, after Borges & Aldon, 2004):
   walk the scan in bearing order; consecutive returns j−1, j join
   the same cluster if the beams are adjacent and

     ‖**q**_j − **q**_{j−1}‖ ≤ ε + ρ_j Δθ,  ε = 0.35 m, Δθ = 0.5°

   (the ρΔθ term grows with range because neighbouring beams spread apart:
   1.7 cm at 2 m, 14 cm at 16 m). Keep clusters with ≥ 4 points. Merge across
   the 0°/360° seam.
3. **Ego-motion compensation:** lift each return into world coordinates using
   the **estimated** pose and the sensor's position at the bow:
   **q** = (x_s + ρ sin(ψ + θ), y_s + ρ cos(ψ + θ)).
4. **Associate** (greedy global nearest neighbour; Bar-Shalom et al., 2001):
   sort all track–cluster distances; accept the smallest unused
   pairs while the distance ≤ gate g = max(2.5 U_nom Δt, 0.30) = 0.70 m.
   Unmatched clusters start tentative tracks; confirmed after 2 hits; dropped
   when more than 3 consecutive updates are missed (a track coasts through up to
   3 misses, 1.5 s, and is removed at the 4th; `TRACK_MAX_MISSES` = 3).
5. **Estimate velocity** — constant-velocity Kalman filter (Kalman, 1960) per
   track, state **s** = [x, y, vₓ, v_y]ᵀ, Δt = 0.5 s:

     predict: **s**⁻ = F **s**, P⁻ = F P Fᵀ + Q
     update:  K = P⁻Hᵀ(H P⁻Hᵀ + R)⁻¹, **s** = **s**⁻ + K(**z** − H**s**⁻), P = (I − K H) P⁻

     F = [[1,0,Δt,0],[0,1,0,Δt],[0,0,1,0],[0,0,0,1]], H = [[1,0,0,0],[0,1,0,0]]
     Q = σ_a² [[Δt⁴/4,0,Δt³/2,0],[0,Δt⁴/4,0,Δt³/2],[Δt³/2,0,Δt²,0],[0,Δt³/2,0,Δt²]]
     (piecewise-constant white acceleration), σ_a = 0.10 m/s²;
     R = σ_z² I, σ_z = 0.05 m; initial velocity variance 0.5 (m/s)².

   **z** is the cluster centroid (`TRACK_MEASUREMENT = "centroid"` in
   baseline-v3, decision F64). A hull-fit measurement (the centroid plus a learned
   offset to the centre of a fitted LOA × B hull) is implemented but off: it cut
   close-range error but worsened the course estimate at 6–9 m, where encounters
   engage. Speed = |(vₓ, v_y)|, course = atan2(vₓ, v_y).
6. **Static or moving? — free-space motion evidence** (not a speed threshold).
   A speed threshold fails because a static panel's centroid slides as the
   viewpoint changes (the visible face changes), promoting panels to "vessels"
   in 28–33 % of frames during development. Instead, compare a track's
   points now with its points 2 s (4 scans) ago:

   - **Appear:** a current point lies where, 2 s ago, a beam passed straight
     through at least τ_p = 0.25 m beyond it, and no old return was within
     0.30 m of it (space that was empty is now occupied).
   - **Vacate:** an old point lies where a current beam now passes through,
     with no current return within 0.30 m (space that was occupied is now
     empty).

   A static solid can do neither from any viewpoint (a ray reaching a point on
   its surface must hit it; a newly revealed face was hidden, not empty). The
   neighbouring beams within τ_p laterally must also clear the point, so pose
   error cannot fake a pass-through. The update shows **motion** if
   appear + vacate ≥ 3 points and ≥ 10 % of the points compared.
   Hysteresis: promote to "moving" after 2 consecutive motion updates (1 s);
   demote after 4 quiet updates (2 s). Biased toward under-detection: calling a
   panel a vessel has COLREGs consequences. τ_p is sized from pose noise:
   τ_p = max(0.25, 5√2 σ_pose). (Related idea: free-space/ray-casting evidence
   in occupancy mapping and vehicle tracking, e.g. Petrovskaya & Thrun, 2009.)

   The moving track feeds the target branch and the encounter module. The
   static/moving verdict decides what becomes a target track; it does **not**
   remove the target's returns from the pooled sectors (next paragraph).

7. **Encounter classification and CPA** — equations in
   `FORMULATION_EQUATIONS.md` section 4 (CPA, ship domain, CRI after Waltz & Okhrin,
   2023, classification bands, engagement/latch).

**Sector pooling (feasibility pooling, Paper 2; Meyer et al., 2020a).** The
pooled `lidar` branch is computed right after the gate, from **all gated
returns** in the ±135° swath: static panels and the target ship's hull alike
(the order in the code is ray cast → gate → pool → segment → track). Checked
on 2026-10-17: at one instant of a head-on episode, removing the target hull
from the scan changes exactly the two sectors it covers (closeness 0.74 and
0.75 with it, 0 without). A target therefore reaches the policy twice: as an
obstacle in the pooled sectors and as a track in the target branch. 27
sectors over ±135° (15 × 6° ahead, 8 × 11.25° abeam, 4 × 22.5° on the
quarters). For each sector with beam ranges ρ₁…ρ_M in angular order:

- **Max pooling:** ρ̄ = max_j ρ_j — optimistic: one long beam through a gap
  narrower than the hull reports the sector as open.
- **Min pooling:** ρ̄ = min_j ρ_j — conservative: one short beam (a post, a
  corner) reports the whole sector as blocked even if a wide passage exists.
- **Feasibility pooling:** ρ̄ = the largest distance d such that the sector
  still contains a contiguous opening wider than the safe width w = B +
  2 × 0.15 = 0.80 m at range d. Algorithm: sort candidate levels d = ρ_(1) ≤
  ρ_(2) ≤ …; at each level sweep the beams in angular order, accumulating arc
  width Δθ·d for beams longer than d (a blocking beam adds half an arc and
  resets); the first level with no opening wider than w is ρ̄. It answers "how
  far could the vessel go in this direction", not "how far is the nearest or
  farthest return".
- Closeness c = 1 − ρ̄/16 ∈ [0, 1].

## 5. Runtime safety layer (Section 9)

**The reported layer** (decision 2026-10-25; `planning_records/SAFETY_V21_PLAN.md` section 15).
- **Scope:** this one layer is presented, and no other configuration or earlier version is
  described.
- **Position:** it sits between the deployed policy and the actuators.
- **When it is on:** off during training and in every validation-set evaluation. Learned behavior
  is always reported with it off, and its interventions are never credited to the policy.
- **Purpose:** to reduce own-vessel collisions. A higher success rate is welcome but is not its
  purpose.
- **Information:** onboard only:
  - the estimated pose and motion;
  - the LiDAR returns and the static points remembered from them;
  - the map boundary;
  - the target tracks.
  Simulator truth never enters a decision.

The layer has two parts.

1. **Predictive safety filter** (`src/safety_v16.py`, on the sampled filter of `src/safety_v2.py`).
   At every decision with a hazard in reach it forward-simulates, over 8 s with the identified
   vessel model (section 1) and the targets at constant velocity from their tracks:
   - **the plans:** the policy's command and a bank of alternative commands (a rudder-by-throttle
     grid, plus braking when a target track is within 7 m), each followed by recovery maneuvers
     (powered turns, low-propulsion continuation, braking);
   - **the pass test:** a plan passes when the hull keeps a 0.15 m margin from the scan points,
     the map boundary and the targets, plus small extra gaps (0.05 m to scan points, 0.02 m to
     the boundary, 0.10 m to targets), and a short straight run-out at the end of the horizon
     stays clear;
   - **the choice:** if the policy's command passes, it is issued unchanged; otherwise the passing
     command closest to the policy's is issued;
   - **no passing plan:** the policy's command goes through unchecked;
   - **partial hulls:** a target seen only in part is completed from its direction of motion and
     known dimensions.
2. **Permanent emergency stop** (`src/safety_estop.py`; `src/safety_v21.py`, class
   `SafetyFilterV21EC` with `estop_window_s=2.5, use_static_points=False, braking_fallback=False`).
   It runs only at decisions where the filter has no checked plan.
   - **First commands:** the policy's, the filter's issued command, the rudder-by-throttle grid
     and three full-astern rows.
   - **Continuations:** each first command is followed by every continuation in a fixed bank (hold
     the command; brake at once; turn then brake; turn; cruise then brake), rolled out for up to
     2.5 s under two reverse-thrust models (weak and strong).
   - **Trigger:** if every sequence is predicted to bring the collision hull (the hull inflated by
     0.15 m, the geometry a collision is scored with) outside the map boundary or into contact
     with a target within 2.5 s, the vessel commits to full astern with the rudder centred for
     the rest of the episode. Each sequence's contact time takes the later of the two models.
   - Remembered static points are not part of the trigger.
   - **Targets after the stop:** once the vessel commits, every target brakes in a straight line
     to rest at 0.471 m/s². This is an evaluation convention: the simulator's strongest own-ship
     braking.

**Outcomes and scoring.**
- **Stopped:** after a commitment, an episode ends as "stopped" once the own ship is still (at
  most 0.05 m/s and 1°/s) and every target has finished braking. A stop ranks like a timeout:
  worse than a goal, better than a collision.
- **Counted against the layer:** own-vessel collisions, that is panel, boundary, or a target
  struck by the own ship, including contacts while braking after a stop.
- **Not counted against it:** a target striking the stopped own ship. It is reported separately.
- **Changed decisions:** the share of decisions at which the issued command differs from the
  policy's, counting the held stop.
- **Results:** Section 10, from the results files.

## 6. Scenario generation, curriculum and data sets (Section 8)

**6.1 Workspace.** Basin 10 × 25 m, start y = 2 m, goal y = 22 m, each
x ~ U[2.5, 7.5] m (slant ≤ about 14°); navigable polygon = basin inset
0.40 m. Channels: 25 m long (the basin's length), width measured wall to wall
(the channel polygon is itself the navigable boundary, with no inset); nominal
width by stage (below); in every stage that uses channels the width varies
along the channel by up to 1.8 times (W_max/W_min ~ U[1.0, 1.8], piecewise linear through 2–4 control
points), with a straight reference path. The path offset is drawn once per
channel from a normal distribution with mean +0.12 and standard deviation 0.15,
clipped to [−0.30, 0.30], as a fraction of the half-width of the narrowest
section (positive = starboard, the Rule 9(a) station), and stays constant along
the path. Channels are used only for head-on, crossing and overtaking, with
probability 0.15 in stage 3 and 0.25 from stage 4 on (stages 6–7 included,
for the episodes that are not three-obstacle layouts).

**6.2 Encounter generation (class first, then solve backward).** Per episode:

1. Draw the class (training shares: head-on 0.20, crossing 0.22, overtaking
   0.16, being overtaken 0.14, null 0.11, no target 0.17; stage limits apply).
2. Draw the heading-crossing angle c from the class band (head-on 170–190°;
   crossing 67.5–175° or 185–292.5°, 60 % from port in training; overtaking,
   being overtaken and null −10–10°, near-parallel traffic, although the
   classifier accepts up to ±67.5° and ±20° — a crossing drawn in the 5° gaps
   either side of the head-on band fails the re-check in step 4 and is
   redrawn), the speed ratio
   k = U_TS/U_OS (head-on 0.70–1.30; crossing 0.60–1.40; overtaking
   0.40–0.55; being overtaken 1.50–2.20; null 0.85–1.15), the time to CPA T₀
   (head-on 12.3–17.0 s; crossing 7.9–14.8 s; overtaking 19.7–31.5 s; being
   overtaken 9.9–15.8 s) and the miss distance d₀ (U[0, 2.0] m; crossing
   U[0, 2.5] m; being overtaken U[f, max(2.0, f + 0.5)] m, where the floor
   f = the contact-free miss distance + 0.35 m). The contact-free miss distance
   is the smallest centre-to-centre miss distance at which the target's hull,
   moving along the straight relative track, never touches the own hull
   inflated by 0.15 m each side (found by bisection, cached per 0.5° of crossing
   angle and 0.05 of speed ratio); no being-overtaken draw lies below the floor.
   **The null class is placed, not solved:** range U[8, 15] m at a bearing of
   20–160° to either side, course within ±10° of the own ship's, speed ratio
   0.85–1.15, accepted only if its CPA is at least 4 m or does not occur within
   the 90 s episode; it has no T₀.
3. **Solve backward** for the spawn that produces exactly (d₀, T₀):

     ψ_TS = ψ_OS + c;  **v**_rel = U_OS **h**_OS − U_TS **h**_TS;
     **n̂** = ± perp(**v**_rel)/|**v**_rel|;
     **p**_TS(T₀) = **p**_OS(0) + U_OS T₀ **h**_OS + d₀ **n̂**;
     **p**_TS(0) = **p**_TS(T₀) − U_TS T₀ **h**_TS

   (**h** = unit heading vector). Sampling d₀ explicitly matters: without it the
   target is always dead on the track, and the policy never sees an encounter
   that needs no alteration.
4. **Validate:** re-classify the realized geometry (reject if the class came out
   different), check the spawn range R₀ (head-on 13.8–19 m, beyond the 12 m
   effective LiDAR range; crossing 5–10 m; overtaking 6 m to min(12, 16(1 − k)) m;
   being overtaken 4–6 m astern, with the own ship starting 6.9 m along the
   basin leg; null 8–15 m) and that the target track stays inside the water;
   reject and resample. The target is present from the first step and is
   acquired by the tracker as it comes into range.
   **Unescapable crossings:** each crossing draw first decides whether it is to
   be unescapable (probability 0.20), then its kinematics are redrawn until a
   scripted check agrees. In the check the own ship holds course at cruise for
   1.5 s, then either coasts to a stop or commits a 60° alteration in the
   required direction (starboard for a target from starboard, port for one
   from port); the crossing is escapable if either response keeps the hulls
   (0.15 m margin) apart and the own hull inside the water until 6 s past the
   drawn TCPA. The label is kept with the episode.
5. **Static obstacles:** 1.0 m panels along 25–70 % of the path with lateral
   offsets; in stages 1–5 panels are kept clear of ±0.4 T₀ of own-ship travel
   around the CPA so they do not decide the encounter (the **decoupled**
   scenarios: obstacle avoidance and collision avoidance happen one after the
   other).
6. **Static feasibility (A\*; Hart, Nilsson & Raphael, 1968):** grid 0.25 m,
   walls inflated 0.40 m and panels 0.45 m for the hull; a route must exist and
   be ≤ 2.25 × the leg length; otherwise redraw (up to 20 times, then thin
   the obstacles).

**6.3 Curriculum (baseline-v3, 3.0 × 10⁶ steps).** Stage boundaries are fixed
timesteps at 6.4, 14.4, 25.6, 40, 60 and 80 % of a 2.5 M base budget; training
then continues in stage 7 to 3.0 M.

| Stage | From | Episodes | Obstacles | Channel widths |
|---|---|---|---|---|
| 1 | 0 | no target | 0–1 | basin only |
| 2 | 0.16 M | no target | 0–3 | basin only |
| 3 | 0.36 M | head-on, crossing (both sides), null, no target; longer times to CPA | 0–1 | 7–10 m (15 %) |
| 4 | 0.64 M | all six types | 0–2 | 4.5–10 m (25 %) |
| 5 | 1.00 M | all six types | 0–3, weighted to 3 (0/1/2/3 at 0.10/0.20/0.30/0.40) | 3.5–10 m (25 %) |
| 6 | 1.50 M | all six types; 25 % three-obstacle spread layouts | 1–3, weighted to 3 (0.2/0.3/0.5) | 3.5–10 m |
| 7 | 2.00 M | all six types; 45 % three-obstacle spread layouts; 30 % of their targets change speed once | 1–3, weighted to 3 | 3.5–10 m |

- **Spread three-obstacle layouts (stages 6–7) — the coupled scenarios:** a leg with start and goal x
  in [2, 8] m (slant ≤ 14.1°) and three 1 m panels in one of three motifs —
  **gate + on-path** (two panels either side of the leg, 1.3–2.7 m from it,
  plus one near the path further on; 40 %), **slalom** (three panels
  alternating sides; 30 %), **side + on-path** (one on the path, one each side
  1.2–3.2 m off; 30 %) — positioned along 28–85 % of the leg, ≥ 0.9 m from the
  walls and ≥ 0.4 m apart. The encounter is generated on the leg with the
  panels fixed, **without** the CPA guard, so the encounter may happen beside a
  panel; targets start inside the basin, keep 0.20–1.10 m/s, stop short of the
  wall, and their straight track clears every panel by 0.30 m.
- **CPA guard:** halved in stage 6 (±0.2 T₀), removed in stage 7.
- **Varying-speed targets (stage 7):** the target changes speed once during
  the episode (a single speed step with bounded acceleration).
- **Space-time solvability (stages 6–7, every target episode):** forward
  reachability on the A\* free-space grid — the own ship as a point moving up
  to 1.1 m/s in any direction or waiting; every 0.5 s the reachable set grows
  by 0.55 m and every cell within 1.3 m of the target's position at that time
  is removed; solvable if a cell within 0.5 m of the goal is reached before the
  episode ends. Optimistic about the vessel (no turning radius), conservative
  about separation; it removes episodes no controller could solve. Unsolvable
  draws are redrawn.
- 15 % of episodes start slow (from rest or up to half cruise). Training
  targets always hold their course and never give way; in stage 7, 30 % of the
  three-obstacle layouts' targets change speed once.
- **Width thresholds** (the strata edges, derived rather than chosen): with the
  own ship at the Rule 9(a) station a quarter of the width from the starboard
  wall, a give-way alteration needs one abeam domain semi-axis of starboard
  room, so W_crossing ≥ 4 (a_abeam + w_wall) = 4 (1.25 + 0.65) = 7.60 m;
  head-on needs two vessels abreast, W_head-on ≥ 2 a_abeam + 2 w_wall = 3.80 m;
  overtaking adds a pose-drift margin over the time the vessels stay abreast,
  W_overtaking ≈ 3.80 + 2 ρ_pose t_exp = 3.80 + 2 × 0.0101 m/s × 22.8 s = 4.26 m,
  with t_exp = (a_ahead + a_astern + L_pp)/(U_OS − U_TS) at a speed ratio of 0.5
  (w_wall = 0.65 m wall clearance).

**6.4 Data sets** — disjoint random-seed ranges ("namespaces") so training,
selection and testing never share an episode (the coupled validation episodes
come from their own block at seeds 360,000+, in a gap no namespace uses, and the
coupled test episodes from dedicated blocks at 310,000–349,999, 450,000+ and
600,000+ that no training or validation draw uses):

| Set | Size | Use |
|---|---|---|
| Training set | generated on the fly from the training seeds | seeds 0–99,999 |
| Validation set, decoupled part | 120 episodes: 20 per episode type × 6 (stage-5 generator) | checkpoint selection for the learners, tuning of the classical comparators, and the learning curves |
| Validation set, coupled part | 150 episodes: 20 no-target + per encounter (head-on, crossing from port, crossing from starboard, overtaking, being overtaken) 20 constant-speed + 6 varying-speed, all space-time solvable | the same uses, together with the decoupled part (270 episodes in all) |
| Test set (test set v4; headline) | 1,000 held-out episodes: **664 decoupled** (basin and channel encounters of every type, with constant-velocity, reactive and noncompliant targets) and **336 coupled** (three-obstacle layouts with constant- and varying-speed targets); near-duplicates trimmed; every episode passes the fairness check (6.5) | touched once per policy |
| Frozen benchmark | the 800 basin and channel encounter episodes the test set's decoupled part is drawn from: 8 cells × 100 (basin head-on, crossing, overtaking, being overtaken and null; channel head-on, crossing and overtaking) | continuity with earlier tables |
| Robustness set | the same scenarios with a reactive target (700; it follows the COLREGs-VO rule from its own side) and, in head-ons, a noncompliant target that turns to port (200) | robustness to target behavior |

Selection score: goal rate − 2 × collision rate on the whole validation set
(both parts, 270 episodes), safety layer off, every 200,000 steps.

**6.5 Fairness check on the test set (oracle feasibility).** For each test episode, a library of
up to 574 manoeuvres is rolled out with perfect knowledge of the target's future motion, the
true vessel model and the policy's own action space: follow an A\* route around the panels,
then hold a course alteration (−70° to +70°) or a speed change (throttle −1 to +1) for 4, 8 or
14 s and until the target has passed, then return to the route. The check also finds when the
onboard perception first tracks the target (about 2 s in). An episode stays in the test set
only if some manoeuvre that starts at or after that moment reaches the goal with 0.2 m
clearance throughout; otherwise it is replaced by a fresh episode of the same kind that passes.
Of the 1,000 episodes, 67 were replaced (`src/oracle_feasibility.py`).

## 7. Field trials and the UDP bridge (Section 11)

(Paper 2's basin trials; Paper 3's physical encounter trials are planned. Take
the trial description — site, panels, procedure, results — from Paper 2.)

**Network:** the vessel's onboard computer streams telemetry over UDP; the
shore laptop runs the bridge (`udp_live_rl.py`).

1. The laptop binds a UDP socket (local port 5000) and sends `START\n` to the
   vessel (server port 5050) to register as a listener.
2. The vessel streams text lines; per control cycle one pose line and one LiDAR
   line (0.5 s cycle):
   - pose: `[HH:MM:SS.ffffff]Y,X,Yaw` (the vessel frame is rotated: the
     bridge uses (−Y, X, −Yaw) as (x, y, heading));
   - LiDAR: `[HH:MM:SS.ffffff][r0,r1,…,r719]` — 720 integer ranges, × 0.1 to
     metres, reversed to clockwise order, 0 = no return → max range, rotated
     to the simulator's beam-0 direction;
   - heading reference `HDG:` and actuator echo `S1:… S2:… RC` lines.
3. A streaming decoder latches the latest pose/actuator lines and emits a frame
   when a LiDAR line arrives; each LiDAR frame waits (≤ 0.3 s) for the pose
   line of its own cycle (stamps within 0.1 s), otherwise it uses the latest
   pose and counts the frame stale.
4. Velocity and yaw rate are differenced from successive poses; the reference
   path and the goal are set per test case in the basin frame.
5. **Observation** built exactly as in simulation (Paper 3 version below).
6. **Action to vessel:** rudder S1 = −100 · a_δ (the real rudder sign is
   opposite to the simulator's); propulsion S2 = (n/24) · 100 with n = clip(6 +
   6 a_n, 0, 12) rpm-units, so the policy commands S2 ∈ [0, 50] (cruise 25).
   Sent as `$CMD,<S1>,<S2>` (two decimals). A shadow mode computes and logs
   actions without sending them.
7. **Safety:** an operator can latch the stop (full astern S2 = −100 until
   stopped, then hold) and hand control back; the latch is the same code as
   the simulator's runtime layer.
8. Everything is logged (raw lines, decoded frames, observations, actions) and
   can be replayed through a fake vessel server for offline testing.

**Paper 3 observation in the bridge (planned update of Paper 2's adapter):**
the bridge runs the simulator's own perception modules on the real scan, so
the policy sees the same 70-value observation:

| Branch | From the real data |
|---|---|
| lidar (27) | gate the scan against the basin polygon → segment → track → keep static returns → feasibility pooling over ±135° → closeness |
| boundary (7) | ray-cast the basin's navigable polygon from the decoded pose |
| ego (3) | surge, sway, yaw rate from pose differencing (body frame) |
| path (3) | cross-track and course errors against the test case's reference leg |
| target (16) | the moving track (Kalman estimate) → CPA, domain distance, CRI, class |
| context (14) | the engagement state machine run on that track; previous executed action |

Paper 2's observation (225 beams over 270° pooled into 25 sectors, u, v, yaw rate, three path
errors, front clearance, side-clearance difference, local target cross-track)
is replaced; the three LiDAR-derived cues were dropped so the passing side is
learned rather than supplied.

## 8. References formerly marked [VERIFY] (checked 2026-10-10)

- Yasukawa, H. & Yoshimura, Y. (2015). Introduction of MMG standard method for ship maneuvering
  predictions. *Journal of Marine Science and Technology* 20(1):37–52.
- Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W. & Abbeel, P. (2017). Domain randomization
  for transferring deep neural networks from simulation to the real world. *IEEE/RSJ International
  Conference on Intelligent Robots and Systems (IROS)*, 23–30.
- Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots.
  *International Journal of Robotics Research* 5(1):90–98.
- Fiorini, P. & Shiller, Z. (1998). Motion planning in dynamic environments using velocity
  obstacles. *International Journal of Robotics Research* 17(7):760–772.
- Borges, G. A. & Aldon, M.-J. (2004). Line extraction in 2D range images for mobile robotics.
  *Journal of Intelligent and Robotic Systems* 40(3):267–297.
- Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and
  Navigation*. Wiley.
- Petrovskaya, A. & Thrun, S. (2009). Model based vehicle detection and tracking for autonomous
  urban driving. *Autonomous Robots* 26(2–3):123–139.
- Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009). Curriculum learning. *Proceedings of
  the 26th International Conference on Machine Learning (ICML)*, 41–48.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A.,
  Abbeel, P. & Levine, S. (2018b). Soft actor-critic algorithms and applications. arXiv:1812.05905
  (automatic entropy tuning with target entropy −dim(A)).
