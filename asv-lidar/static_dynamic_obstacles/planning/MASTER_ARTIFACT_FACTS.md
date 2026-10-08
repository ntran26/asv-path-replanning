# Paper 3 master artifact — verified technical facts

> **Kept current (decision, 2026-10-04).** This file describes only the best formulation: **baseline-v3**, the kept SAC 3 M policy (`configs/baseline_v3.json`, `src/formulation_v3.py`). Earlier versions are not described. If baseline-v4.3 passes its SAC pair (`planning/BASELINE_V4_PLAN.md`), this file is updated to v4.3.

**Written 2026-09-29 from the project's code**, as the source for the
master explainer artifact (`MASTER_ARTIFACT_BRIEF.md`). Every number here is
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

## 1. Vessel and dynamics (Section 2 of the artifact)

**Platform.** Bluefin model-scale ASV: length overall L = 1.725 m, beam
B = 0.50 m, draft 0.193 m, mass m = 64.55 kg. Single propeller and a stern
rudder (±40°). A 360° 2D LiDAR sits at the bow (0.8625 m ahead of the
reference point).

**State** (7 values): surge u, sway v (body-frame velocities, m/s), yaw rate r
(rad/s), heading ψ (compass: 0 = north, clockwise positive), actual rudder angle
δ, position (x east, y north).

**Kinematics** (body velocities to world position):

  ẋ = u sin ψ + v cos ψ,  ẏ = u cos ψ − v sin ψ,  ψ̇ = r

**Kinetics** — three coupled equations (Newton's second law in a rotating body
frame, in the form of Fossen, 2011/2021, with MMG-style hydrodynamic terms,
Yasukawa & Yoshimura, 2015 [VERIFY]):

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
  (command scale 0–24, 12 = nominal field setting). With t_b = 0: T ∝ n².
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
  δ_c = 40° × a_δ passes through a transport delay t_d = 0.73 s,
  then a first-order lag with time constant τ = 0.90 s and a rate limit:

    δ_{k+1} = δ_k + clip((δ_c(t − t_d) − δ_k)(1 − e^{−Δt/τ}), ±δ̇_max Δt)

  (the fitted rate limit is effectively non-binding; the delay and lag dominate).

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
in all training), 1.116 m/s at 12; yaw rate about 10.4°/s at full rudder;
several seconds to build a turn (delay + lag + inertia).

**Domain randomization:** each training episode draws a new parameter set as
a convex blend of two bootstrap solutions of the identification plus 5 %
jitter (scale 1.0). Blending whole parameter vectors keeps the correlations
between yaw parameters (k_R, N_r, N_uv), so every draw is a physically
consistent vessel (Tobin et al., 2017 for domain randomization [VERIFY]).

**Sources:** Fossen, T. I. (2021). *Handbook of Marine Craft Hydrodynamics and
Motion Control*, 2nd ed., Wiley (3-DOF maneuvering models, added mass,
Coriolis and Munk terms, LOS guidance). Fossen (2011), 1st ed. Yasukawa &
Yoshimura (2015), MMG standard method, *J. Mar. Sci. Technol.* 20:37–52
[VERIFY]. Paper 2 (Tran et al., 2026) for the platform.

## 2. Classical comparators (Section 3)

All comparators use the same perception (LiDAR scan, tracker, map boundary) as
the learned policies and the same actuator limits; they differ only in the
avoidance logic. Tuned on the validation set by coordinate descent, scored
by goal − 2 × collision.

**2.1 LOS guidance + heading PID (path following, shared).** Line-of-sight
guidance (Fossen, 2021): aim at a point Δ = 2.5 m ahead on the path,

  χ_d = χ_path + atan2(−e_y, Δ) − β̂,

where e_y is the cross-track error and β̂ the measured sideslip (capped ±20°).
Heading PID on the error e = χ_d − ψ (degrees):

  a_δ = clip(K_p e − K_d r[°/s] + I, −1, 1),  K_p = 1/35, K_d = 1/18,
  K_i = 1/600 (integral active only inside |e| < 10°, capped at 0.15).

Yaw-rate tracking (used by DWA): a_δ = r_d/10.4 + 0.08 (r_d − r) (deg/s).
Speed: rpm = u_d / (speed per rpm) + 6 (u_d − u).

**2.2 LOS-PID + Dynamic Window Approach (DWA)** (Fox, Burgard & Thrun, 1997).
LOS-PID drives the vessel until anything comes within 9 m (a track, a scan
point ahead, or within a hull length of the boundary). Then, each decision:

1. **Candidates:** yaw-rate set-points r_d ∈ {−9, −6.75, …, +9} °/s × speeds
   {1.0, 0.5, 0} × U_nom × hold times {3, 6} s, plus plain LOS-PID at two
   speeds.
2. **Dynamic window through the real dynamics:** each candidate is simulated
   closed-loop on the vessel model (delay, lag, inertia) for up to 16 s — not
   as an ideal arc, because this hull needs about 5 s to build a turn. After the
   hold time the prediction hands back to LOS-PID.
3. **Admissibility:** inadmissible if the predicted hull comes within 0.20 m of
   a scan point or of the boundary over the first 5.5 m of travel, or of a
   tracked target (constant velocity) within 10 s.
4. **Objective** (maximized over admissible candidates):

   G = 1.0·heading + 1.0·clearance + 0.5·velocity + 0.6·path − 0.25·smoothness

   heading = 1 − |LOS heading at the end point − predicted heading|/π;
   clearance = min(clearance, 1.5 m)/1.5 m; velocity = mean speed/U_nom;
   path = 1 − mean|cross-track|/2 m; smoothness = change from last choice.
5. **Fallback** if none is admissible: the candidate whose first violation
   comes latest, then most clearance.

**DWA knows obstacles, not rules** — it is the "no COLREGs" reference.

**2.3 COLREGs-VO** (Kuwata et al., 2014; velocity obstacles after Fiorini &
Shiller, 1998 [VERIFY]):

1. **Candidates:** course offsets −90° … +90° in 5° steps about the present
   heading, plus the LOS course; speeds {1, 0.75, 0.5, 0.25, 0} × U_nom.
2. **Own-ship motion model:** each candidate is a turn toward the new course at
   6°/s after a 1 s lag, with speed approaching the new value exponentially
   (time constants 60 s down, 20 s up); sampled every 0.25 s.
3. **Velocity obstacle (hard):** a candidate is rejected if, holding it, the
   own hull comes within 0.20 m of a target's hull within 20 s (target at
   constant velocity), or within 0.15 m of a scan point / 0.05 m of the
   boundary within 10 s.
4. **Classification (open-water roles):** being overtaken if the target
   bearing |α| ≥ 112.5°; overtaking if the own ship is ≥ 112.5° off the
   target's bow and faster; head-on if |α| ≤ 22.5° and the reciprocal angle
   ≤ 22.5°; otherwise crossing — give-way if the target is to starboard
   (α > 0), stand-on if to port.
5. **COLREGs constraint (give-way classes):** while the encounter is "live"
   (closing, predicted miss < 2.5 m within 20 s) the relative velocity must lie
   to starboard of the line of sight: (**p**_TS − **p**_OS) × (**v**_OS − **v**_TS)
   < 0 — "alter to starboard and pass astern".
6. **Stand-on (Rule 17):** hold course and speed while the preferred candidate
   stays clear for more than 12 s.
7. **Cost** over the allowed candidates:
   J = 1.0·|χ − χ_LOS|/π + 0.5·max(0, U_pref − U)/U_pref + 1.0·static-proximity
   + 0.3·|χ − χ_prev|/π.
8. **Fallback:** if nothing satisfies both the velocity obstacle and the
   COLREGs constraint, drop the constraint rather than accept a collision
   (Kuwata's own rule); if nothing is safe, take the latest first contact.

A second VO variant uses the paper's narrow-channel convention (give way to a
crossing target from either side); the contrast between the two isolates the
convention's effect.

**Artificial potential field (APF) — background only.** Paper 2's comparator
(LOS + APF): the goal attracts, obstacles repel, and the vessel follows the
sum of the forces (Khatib, 1986 [VERIFY]). Fast and simple, but it gets stuck
in local minima between close obstacles and knows no rules. Explain it as
context; it is not a Paper 3 comparator.

## 3. Learners and training protocol (Section 4)

- Library: Stable-Baselines3 2.3.2 / sb3-contrib 2.3.0 (Raffin et al., 2021).
- **Common:** discount γ = 0.951 per 0.5 s step; reward normalization only
  (running scale, clip 10); 10 parallel simulators; 3.0 × 10⁶ environment
  steps (baseline-v3 curriculum, Section 6; stage 7 runs from 2.0 M to the end); validation-set evaluation every
  2 × 10⁵ steps; each seed represented by its best validation-set checkpoint
  (score = goal rate − 2 × collision rate); 3 seeds per learner; learners use
  their original papers' default hyperparameters, no per-learner tuning.

| Learner | Type | Key settings |
|---|---|---|
| PPO (Schulman et al., 2017) | on-policy, policy gradient with clipped surrogate | lr 3e-4; 512 steps × 10 envs per update; minibatch 512; 10 epochs; GAE λ = 0.95 (Schulman et al., 2016); clip ε = 0.2; target KL 0.03 |
| RecurrentPPO | on-policy, PPO + LSTM memory (Hochreiter & Schmidhuber, 1997) | PPO settings + 256-unit LSTM, separate for actor and critic |
| SAC (Haarnoja et al., 2018) | off-policy, maximum-entropy actor–critic | lr 3e-4; replay buffer 10⁶; batch 256; τ = 0.005; 10⁴ random warm-up steps; automatic entropy temperature; 1 gradient step per environment step |
| TQC (Kuznetsov et al., 2020) | off-policy, distributional (quantile) SAC | SAC settings; 2 critics × 25 quantiles; drop the top 2 quantiles per critic |

- **Network (identical for all):** a scene encoder (lidar, boundary, ego, path,
  previous action → 128) and a target-slot encoder (target + context, 28 → 64
  → 64 → 32) whose input is gated by the presence bit; concatenated features
  (160) feed actor and critic heads of 256 × 256 ReLU.
- Development used PPO to iterate the formulation; the formulation itself
  references no algorithm.

## 4. LiDAR perception and tracking (Section 5)

**Sensor model:** 720 beams at 0.5° (full 360°), range 1.0–16 m (returns
closer than 1 m are lost in a dead zone). Pose noise 0.03 m / 0.2°, surge
noise 0.05 m/s, yaw-rate noise 1°/s. One scan per 0.5 s decision.

**Pipeline, in order:**

1. **Gate:** discard returns outside the known navigable polygon (walls,
   people on the deck, anything beyond the water).
2. **Segment** (adaptive breakpoint detection, after Borges & Aldon, 2004
   [VERIFY]): walk the scan in bearing order; consecutive returns j−1, j join
   the same cluster if the beams are adjacent and

     ‖**q**_j − **q**_{j−1}‖ ≤ ε + ρ_j Δθ,  ε = 0.35 m, Δθ = 0.5°

   (the ρΔθ term grows with range because neighbouring beams spread apart:
   1.7 cm at 2 m, 14 cm at 16 m). Keep clusters with ≥ 4 points. Merge across
   the 0°/360° seam.
3. **Ego-motion compensation:** lift each return into world coordinates using
   the **estimated** pose and the sensor's position at the bow:
   **q** = (x_s + ρ sin(ψ + θ), y_s + ρ cos(ψ + θ)).
4. **Associate** (greedy global nearest neighbour; Bar-Shalom et al., 2001
   [VERIFY]): sort all track–cluster distances; accept the smallest unused
   pairs while the distance ≤ gate g = max(2.5 U_nom Δt, 0.30) = 0.70 m.
   Unmatched clusters start tentative tracks; confirmed after 2 hits; dropped
   after 3 consecutive misses (1.5 s).
5. **Estimate velocity** — constant-velocity Kalman filter (Kalman, 1960) per
   track, state **s** = [x, y, vₓ, v_y]ᵀ, Δt = 0.5 s:

     predict: **s**⁻ = F **s**, P⁻ = F P Fᵀ + Q
     update:  K = P⁻Hᵀ(H P⁻Hᵀ + R)⁻¹, **s** = **s**⁻ + K(**z** − H**s**⁻), P = (I − K H) P⁻

     F = [[1,0,Δt,0],[0,1,0,Δt],[0,0,1,0],[0,0,0,1]], H = [[1,0,0,0],[0,1,0,0]]
     Q = σ_a² [[Δt⁴/4,0,Δt³/2,0],[0,Δt⁴/4,0,Δt³/2],[Δt³/2,0,Δt²,0],[0,Δt³/2,0,Δt²]]
     (piecewise-constant white acceleration), σ_a = 0.10 m/s²;
     R = σ_z² I, σ_z = 0.05 m; initial velocity variance 0.5 (m/s)².

   **z** is the cluster centroid plus a learned offset to the hull centre (a
   cluster only shows the faces the sensor sees). Speed = |(vₓ, v_y)|,
   course = atan2(vₓ, v_y).
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
   in occupancy mapping and vehicle tracking, e.g. Petrovskaya & Thrun, 2009
   [VERIFY].)

   Static clusters feed the lidar branch; the moving track feeds the target
   branch and the encounter module.

7. **Encounter classification and CPA** — equations in
   `FORMULATION_EQUATIONS.md` section 4 (CPA, ship domain, CRI after Waltz & Okhrin,
   2023, classification bands, engagement/latch).

**Sector pooling (feasibility pooling, Paper 2; Meyer et al., 2020a).** 27
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

## 5. Runtime safety layer

Off during training. When enabled: fires if an engaged give-way encounter is
in extremis (DCPA < d_req, 0 ≤ TCPA < 5 s), the compliant alteration is
inadmissible, and a full stop would let the target pass clear (≥ 1.76 m).
Then full astern (S2 = −100) until stopped (≤ 0.05 m/s; at most 8 s), hold at
zero thrust at least 2 s and at most 10 s, then return control. Interventions
are never counted as learned behavior.

## 6. Scenario generation, curriculum and data sets (Section 7)

**6.1 Workspace.** Basin 10 × 25 m, start y = 2 m, goal y = 22 m, each
x ~ U[2.5, 7.5] m (slant ≤ about 14°); navigable polygon = basin inset
0.40 m. Channels: parallel walls, width by stage (below), reference path offset
U[−0.30, 0.30] × half-width, mean +0.12 (starboard station). Channels are used
only for head-on, crossing and overtaking.

**6.2 Encounter generation (class first, then solve backward).** Per episode:

1. Draw the class (training shares: head-on 0.20, crossing 0.22, overtaking
   0.16, being overtaken 0.14, null 0.11, no target 0.17; stage limits apply).
2. Draw the heading-crossing angle c from the class band (head-on 170–190°;
   crossing 67.5–175° or 185–292.5°, 60 % from port in training; overtaking
   and being overtaken −67.5–67.5°; null −20–20°), the speed ratio
   k = U_TS/U_OS (head-on 0.70–1.30; crossing 0.60–1.40; overtaking
   0.40–0.55; being overtaken 1.50–2.20; null 0.85–1.15), the time to CPA T₀
   (head-on 12.3–17.0 s; crossing 7.9–14.8 s; overtaking 19.7–31.5 s; being
   overtaken 9.9–15.8 s) and the miss distance d₀ (U[0, 2.0] m; crossing
   U[0, 2.5] m; null ≥ 4 m; being overtaken above a contact-free floor + 0.35 m).
3. **Solve backward** for the spawn that produces exactly (d₀, T₀):

     ψ_TS = ψ_OS + c;  **v**_rel = U_OS **h**_OS − U_TS **h**_TS;
     **n̂** = ± perp(**v**_rel)/|**v**_rel|;
     **p**_TS(T₀) = **p**_OS(0) + U_OS T₀ **h**_OS + d₀ **n̂**;
     **p**_TS(0) = **p**_TS(T₀) − U_TS T₀ **h**_TS

   (**h** = unit heading vector). Sampling d₀ explicitly matters: without it the
   target is always dead on the track, and the policy never sees an encounter
   that needs no alteration.
4. **Validate:** re-classify the realized geometry (reject if the class came out
   different), check the spawn range and that the target track stays inside the
   water; reject and resample.
5. **Static obstacles:** 1.0 m panels along 25–70 % of the path with lateral
   offsets; in stages 1–5 panels are kept clear of ±0.4 T₀ of own-ship travel
   around the CPA so they do not decide the encounter.
6. **Static feasibility (A\*; Hart, Nilsson & Raphael, 1968):** grid 0.25 m,
   walls inflated 0.40 m and panels 0.45 m for the hull; a route must exist and
   be ≤ 2.25 × the leg length; otherwise redraw (up to 20 times, then thin
   the obstacles).

**6.3 Curriculum (baseline-v3, 3.0 × 10⁶ steps; stage 7 continues to 3.0 M).**

| Stage | From | Episodes | Obstacles | Channel widths |
|---|---|---|---|---|
| 1 | 0 | no target | 0–1 | basin only |
| 2 | 0.16 M | no target | 0–3 | basin only |
| 3 | 0.36 M | head-on, crossing (both sides), null, no target; longer times to CPA | 0–1 | 7–10 m (15 %) |
| 4 | 0.64 M | all six types | 0–2 | 4.5–10 m (25 %) |
| 5 | 1.00 M | all six types | 0–3, weighted to 3 (0/1/2/3 at 0.10/0.20/0.30/0.40) | 3.5–10 m (25 %) |
| 6 | 1.50 M | all six types; 25 % three-obstacle spread layouts | 1–3, weighted to 3 (0.2/0.3/0.5) | 3.5–10 m |
| 7 | 2.00 M | all six types; 45 % three-obstacle spread layouts; 30 % of their targets change speed once | 1–3, weighted to 3 | 3.5–10 m |

- **Spread three-obstacle layouts (stages 6–7):** a leg with start and goal x
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
  targets always hold their course and never give way.

**6.4 Data sets** — disjoint random-seed ranges ("namespaces") so training,
selection and testing never share an episode:

| Set | Size | Use |
|---|---|---|
| Training | generated on the fly | seeds 0–99,999 |
| Development | 120 episodes: 20 per episode type × 6 (stage-5 generator) | checkpoint selection |
| Three-obstacle validation set | 150 episodes: 20 no-target + per encounter (head-on, crossing from port, crossing from starboard, overtaking, being overtaken) 20 constant-speed + 6 varying-speed, all space-time solvable | checkpoint selection, with the validation set |
| Test set (headline) | 1,000 held-out episodes: basin and channel encounters of every type (constant-velocity, reactive and non-compliant targets) and three-obstacle layouts with constant- and varying-speed targets; near-duplicates trimmed; every episode passes the fairness check (6.5) | touched once per policy |
| Frozen benchmark | the 800 basin and channel encounter episodes the test set draws its encounter cells from (8 cells × 100) | continuity with earlier tables |
| Robustness set | the same scenarios with a reactive target (700; it follows the COLREGs-VO rule from its own side) and, in head-ons, a non-compliant target that turns to port (200) | robustness to target behavior |
| Width sweep | channels 10, 8, 7, 6, 5, 4.5, 4, 3.5 m | how the response changes with room |
| Perception degradation | pose drift, detection dropout, occlusion, velocity-estimate noise, separately and jointly | robustness to sensing |

Selection score: goal rate − 2 × collision rate on the validation sets,
safety layer off.

**6.5 Fairness check on the test set (oracle feasibility).** For each test episode, a library of
up to 574 manoeuvres is rolled out with perfect knowledge of the target's future motion, the
true vessel model and the policy's own action space: follow an A\* route around the panels,
then hold a course alteration (−70° to +70°) or a speed change (throttle −1 to +1) for 4, 8 or
14 s and until the target has passed, then return to the route. The check also finds when the
onboard perception first tracks the target (about 2 s in). An episode stays in the test set
only if some manoeuvre that starts at or after that moment reaches the goal with 0.2 m
clearance throughout; otherwise it is replaced by a fresh episode of the same kind that passes.
Of the 1,000 episodes, 67 were replaced (`src/oracle_feasibility.py`).

## 7. Field trials and the UDP bridge (Section 8)

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
