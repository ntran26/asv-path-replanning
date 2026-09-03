# Placeholder values for `Manuscript_Revised_HIGHLIGHTED_drones-4493946.docx`

Every `⟪…⟫` placeholder in the manuscript, resolved against our setup where the
value exists in the code or the evaluation data. Organised as:

* **§1 — Filled.** Value taken directly from the code, config, or evaluation data.
* **§2 — Cannot fill.** Physical facility, hardware datasheet, or field-measured
  quantities that exist nowhere in this repository. Listed so nothing is missed.
* **§3 — Flags.** Four places where the manuscript text conflicts with what the
  code and data actually show. These need a decision, not a number.

All results below are from the frozen 500-episode evaluation set. Source files
are in this folder; see `README.md` for the layout.

---

## §1 Filled values

### 2.5.1 Classical baseline tuning (manuscript line ~214)

| Placeholder | Value |
|---|---|
| random search over `⟪000⟫` configurations | **250** |

Matches the manuscript's description exactly: 250 configurations, 100 tuning
layouts with disjoint seeds, best configuration only evaluated on the 500-episode
set. Record: `tuning/apf_tuning_results.csv`.

*Note:* the tie-break in our implementation is RMS cross-track error among
configurations within **2** percentage points of the best success rate; the
manuscript says "one percentage point". Either change the text to 2, or I can
re-select from the existing record under a 1-point rule — the full search is
saved, so no re-running is needed.

### 3.1 Workspace and episode definition (line ~222)

| Placeholder | Value | Source |
|---|---|---|
| obstacle count distribution | **0.15, 0.15, 0.45, 0.15, 0.10** | `config.TRAIN_OBS_PROBS` |
| start lateral offset up to | **0.0 m** — the vessel spawns *on* the path | `env.reset` |
| start heading offset up to | **5.6 deg** (mean 0.8 deg) | measured over the 500 layouts |
| reference path length | **20.0 m** (20.00–20.10 across layouts) | `START_Y`=2.0 → goal at y=22 |
| episode horizon | **700** control steps | `config.MAX_EPISODE_STEPS` |
| equivalent to | **70 s** at the 10 Hz control period | 700 × 0.1 s |

The heading offset is not an independent randomisation: the vessel always starts
at heading 0 (aligned with +y), and 30 % of layouts have a slanted start/goal
pair, so the offset from the path tangent is whatever the slant produces — at
most 5.6 degrees. The sentence should be reworded to say this rather than
implying a sampled offset.

### 3.1 Sensing and manoeuvring geometry (line ~223)

| Placeholder | Value | Note |
|---|---|---|
| sector subtends at max range | **3.0 m** | 16 m × 10.8° — see flag §3.1 |
| turning radius at cruise | **3.0 m** | measured on `ShipModel` at 12 RPM, full rudder |
| stopping distance | — | see flag §3.2, the model gives 54.9 m |
| CTE at which `r_pf` halves | **3.5 m** (γ=0.20) or **13.9 m** (γ=0.05) | see flag §3.3 |

Supporting measurements on the vessel model at cruise (12 RPM): steady surge
speed **1.82 m/s**, steady turn rate at full rudder **9.6 deg/s**, turning radius
**3.02 m**, tactical diameter **6.04 m**.

### Table 4 — SAC across three seeds, 500 episodes

Rate metrics are **means**; continuous metrics are **IQM**. Intervals are 95 %
stratified bootstrap over obstacle count, pooled across seeds.

| Metric | Value | 95% CI | Seed spread (min–max) |
|---|---|---|---|
| Success rate (%) | **92.4** | [91.1, 93.7] | 90.0–94.0 |
| Obstacle collision rate (%) | **4.7** | [3.7, 5.8] | 1.6–8.6 |
| Border collision rate (%) | **2.9** | [2.1, 3.7] | 1.4–4.4 |
| Mean cross-track error (m) | **0.87** | [0.85, 0.88] | 0.75–0.90 |
| Min. obstacle clearance (m) | **0.44** | [0.42, 0.46] | 0.43–0.54 |
| Episode duration (s) | **20.3** | [20.2, 20.4] | 19.3–20.6 |

**Line ~412:** seed spread is **4.0** percentage points in success rate
(90.0 / 92.4 / 94.0 per seed). Suggested wording for the `⟪TBC⟫`:

> …indicating that the training procedure is **reproducible in the sense that
> all three seeds fall within four percentage points, though the spread is
> comparable to the differences between methods reported in Section 3.3.3 and
> should be weighed accordingly**.

**The IQM column should be deleted for the three rate rows.** IQM is degenerate
on a binary per-episode outcome — the middle 50 % of a mostly-successful set is
all ones, so every method reports exactly 100.0. Report the mean for rates and
IQM only for the continuous metrics.

### Table 5 — comparison against baselines, 500 episodes

| Metric | LOS-PID + APF | PPO (3 seeds) | SAC (3 seeds) |
|---|---|---|---|
| Success rate (%) | **96.8** | **90.5** [89.0, 91.9] | **92.4** [91.1, 93.7] |
| Obstacle collision rate (%) | **2.6** | **7.9** [6.6, 9.3] | **4.7** [3.7, 5.8] |
| Border collision rate (%) | **0.6** | **1.7** [1.1, 2.3] | **2.9** [2.1, 3.7] |
| Mean cross-track error (m) | **0.92** | **1.05** [1.03, 1.07] | **0.87** [0.85, 0.88] |
| Min. obstacle clearance (m) | **0.67** | **0.53** [0.50, 0.56] | **0.44** [0.42, 0.46] |
| Episode duration (s) | **25.4** | **22.0** [21.7, 22.4] | **20.3** [20.2, 20.4] |
| Control effort (rad²·s) | **3.15** | **2.71** [2.56, 2.87] | **4.29** [4.22, 4.35] |

**Line ~422:** `PPO (⟪0⟫ seeds)` → **3**.

LOS-PID + APF is the single best configuration from the 250-configuration search
(`tuning/los_apf_best.json`), as the manuscript's protocol describes. Two further
independent searches were run as a robustness check and give 97.2 % and 94.0 %;
they are available if a spread is wanted, but the manuscript text specifies one.

Control effort is converted to rad²·s from the logged squared rudder command:
`control_effort_deg2s × (π/180)²`.

### Table 6 — paired tests

The table has one row per comparison, but SAC and PPO each have three seeds, so
each comparison yields three paired tests. **Median across seeds** is given
first, with the per-seed range, so you can decide what to report.

| Comparison | CTE diff (m) | Wilcoxon p | Success diff (pp) | McNemar p |
|---|---|---|---|---|
| SAC vs. LOS-PID + APF | **−0.01** (+0.09 … −0.04) | **8.3e−04** (1.3e−13 … 0.184) | **−3.6** (−2.8 … −6.8) | **0.015** (3.4e−06 … 0.049) |
| SAC vs. PPO | **−0.13** (−0.09 … −0.73) | **2.7e−28** (3.1e−63 … 1.9e−04) | **+1.0** (+4.6 … +0.2) | **0.59** (0.015 … 1.0) |
| PPO vs. LOS-PID + APF | **+0.17** (+0.07 … +0.65) | **2.0e−17** (5.7e−56 … 6.9e−05) | **−7.0** (−4.6 … −7.4) | **7.7e−06** (4.3e−07 … 1.4e−03) |

Sign convention: positive CTE difference means the first method has the larger
error; positive success difference means the first method succeeds more often.

**Line ~474 interpretation paragraph — see flag §3.4 before drafting it.** The
ordering these numbers support is not the one the manuscript's surrounding text
anticipates.

### Appendix A — vessel model parameters (Table A1)

Values that map directly onto `src/ship.py`:

| Quantity | Symbol | Value | Unit | Source constant |
|---|---|---|---|---|
| Mass | m | **64.55** | kg | `MASS` |
| Added mass in surge | X_u̇ | **3.66** | kg | `MX` |
| Added mass in sway | Y_v̇ | **62.74** | kg | `MY` |
| Added yaw inertia | N_ṙ | **0.63** | kg·m² | `MOMINERTIA` = Iz 9.6038 + Jz 0.6309 |
| Linear surge damping | X_u | **2.0** | kg/s | `LINEAR_SURGE_DAMP` |
| Linear sway damping | Y_v | **18.0** | kg/s | `LINEAR_SWAY_DAMP` |
| Linear yaw damping | N_r | **1.5** | kg·m²/s | `LINEAR_YAW_DAMP` |
| Rudder moment arm | x_r | **1.05** | m | `X_RUDDER` = −1.05309 (magnitude) |
| Maximum rudder angle | δ_max | **40** | deg | `MAX_RUD_ANGLE` |
| Rudder rate limit | δ̇_max | **20** | deg/s | `MAX_RUD_RATE_DPS` |

Geometry used elsewhere in the paper and worth stating consistently: vessel
length **1.725 m**, beam **0.50 m**, draft **0.193 m**, wetted surface
**0.7614 m²**, hull inflation margin **0.15 m**, LiDAR mounted **0.8625 m**
forward of the vessel origin.

⚠ **`I_z` is currently filled as 10.45 in the manuscript; the model uses
10.2347** (Iz 9.6038 + Jz 0.6309). If 10.45 came from a physical measurement,
keep it and note the model value; otherwise correct it to 10.23.

---

## §2 Cannot fill — not present in this repository

These are physical-facility, hardware-datasheet, or field-measured quantities.
None of them can be derived from the simulation code or the evaluation data, and
I have not guessed at any of them.

**Test basin (line ~560)** — length, width, water depth; whether the facility has
wave and current generation.

**Field obstacles (line ~561)** — cylindrical float diameter, the three positions
relative to the basin datum, the three centre-to-centre separations, and the
basin-centreline path length.

**LiDAR datasheet (line ~563)** — RPLidar C1 native field of view, maximum range,
angular resolution, and scan rate. Our simulation uses a 270° swath, 16 m range
and 225 beams, but those are the *simulated cropped* values already stated in the
sentence; the native specification is a manufacturer figure.

**Odometry and timing (line ~564)** — scan and pose rates; whether velocities are
differentiated or estimated directly; filter type and time constant;
synchronisation method; end-to-end latency and its standard deviation; velocity
estimate standard deviations in surge, sway and yaw rate.

**System identification (line ~751 and Table A2)** — repetition counts for the
straight-line and turning-circle tests, the rudder angle held during the turning
circle, and every "Measured" column entry.

For Table A2's **"Simulated"** column I can supply values from `ShipModel` if
useful — steady surge speed at given propeller settings, time to 95 % of steady
speed, tactical diameter, advance, transfer, steady turn rate. Say the word and
I'll run them. The "Measured" and "Error (%)" columns still need the field data.

**Quantities in Table A1 with no clean counterpart in our model:** quadratic
damping X_uu, Y_vv, N_rr; rudder force coefficient C_δ; propeller thrust
coefficient C_n; actuator first-order lag τ_a. Our hull uses a different
parameterisation — friction via a Schoenherr-type coefficient, `DRAG_COEF`,
`THRUST_COEF`, and a rudder model built from area, lift slope and a force scale —
and the actuator is a **rate limiter with no first-order lag**, so τ_a has no
value in this implementation. Either drop these rows, or restate the table in the
parameterisation the code actually uses; I can produce that table if you prefer.

**Component analyses not yet run** — Table 7 (pooling operator comparison,
line ~480), Table 8 (reward-term contributions, lines ~511–527), the look-ahead
masking result (line ~503), the border-term ranking check (line ~529), and the
factor-of-difference in line ~528. All of these *are* computable from our setup
and would take roughly an hour of compute. They are not done yet — tell me if you
want them and I'll run them.

---

## §3 Flags — text that conflicts with the code or data

### 3.1 The sector-width sentence does not hold at 16 m

Line ~223 states the 10.8° sector "subtends approximately ⟪0.0⟫ m at the maximum
LiDAR range and is therefore comparable to the vessel beam of 0.50 m".

At 16 m a 10.8° sector subtends **3.02 m**, which is six vessel beams — not
comparable. One sector equals 0.50 m only at a range of **2.65 m**. The raw beam
spacing (1.2054°) subtends **0.34 m** at 16 m, which *is* comparable to the beam.

Three options: state 3.0 m and drop the "comparable to the vessel beam" clause;
recast the sentence around the *raw beam* resolution (0.34 m at 16 m); or keep
the argument but attach it to the decision range (one sector ≈ one vessel width
at 2.65 m). The third is closest to the intended meaning.

### 3.2 The stopping-distance claim cannot be supported by the model

Line ~223 says "D_max = 16 m exceeds the measured stopping distance of ⟪0.0⟫ m".

Coasting the model from cruise with propulsion cut gives a stopping distance of
**54.9 m** — over three times D_max and more than twice the basin length. The
hull has light linear surge damping (2.0 kg/s against ~68 kg of effective mass),
so it glides. The vessel also has no reverse thrust in this implementation, so
there is no active braking to measure.

If a *measured* field stopping distance exists, use it and note that the
simulation model is more weakly damped. Otherwise the clause should be dropped —
the turning-radius half of the sentence (3.0 m, comfortably inside 16 m) stands
on its own.

### 3.3 Which γ_e does the sentence mean?

Line ~223 gives the tracking sensitivity as 0.05. The code has **two** values:
`GAMMA_E_CLEAR = 0.20` when the way ahead is clear and `GAMMA_E_BLOCKED = 0.05`
when blocked, blended by the blockage measure. The half-value cross-track error
is **3.47 m** at 0.20 and **13.86 m** at 0.05. The latter exceeds the 10 m basin
width, so quoting 0.05 alone gives a figure that cannot occur in the workspace.
The sentence should state both values and that they are blended, and quote 3.5 m.

### 3.4 The three-seed framing changes the headline conclusion

This is the one to decide before drafting the interpretation paragraph at
line ~474.

The manuscript reports Table 4 and the Table 5 "SAC (proposed)" column as **three
seeds**, but the surrounding prose quotes the **deployed policy** — line ~413
says "achieves an overall success rate of 95 %", line ~416 says "the 94 % success
rate obtained in the present study", and the earlier table at lines ~369–374
carries 94 % / 0.80 m / 0.76 m / 20.78 s / 3 % / 3 %. These are two different
objects, and they give different answers:

| | Success | vs LOS-PID + APF |
|---|---|---|
| SAC, deployed policy | 95.0 % | −1.8 pp, **not significant** (p = 0.211) |
| SAC, three seeds | 92.4 % | −3.6 pp, **significant in 2 of 3 seeds** (p = 3.4e−06, 0.015, 0.049) |

Under the three-seed framing the classical baseline is **significantly better on
success rate**, and the paired tests in Table 6 say so. Under the deployed-policy
framing the two are statistically indistinguishable. The paper cannot present
Table 4 as three seeds and then interpret it with the deployed policy's number.

Whichever is chosen, what the data supports is: **SAC follows the path
significantly more accurately than PPO** (p ≤ 1.9e−04 in all three seeds) and
**completes episodes faster than both baselines**, while the classical stack
succeeds at least as often and keeps larger obstacle clearances (0.67 m against
0.44 m). "SAC outperforms the classical baseline on success" is not supportable
in either framing.

### 3.5 The propulsion curriculum described in §3.1 did not occur in the reported run

Line ~226 states that the first 700,000 timesteps are fixed-speed and "the
remaining 300,000 timesteps introduce speed control in three stages of 100,000
steps each". Lines ~531–532 then ask for results from the fixed-speed checkpoint
and from each stage transition.

The published run's own TensorBoard log
(`sac_log/asv_sac_2/events.out.tfevents.1781347673.*`) records
`eval/min_rpm = eval/mean_rpm = eval/max_rpm = 12.000` at **all nineteen
evaluations from 50k to 950k steps**. Propeller speed never varied. The staged
curriculum appears only in the *resumed* runs after 1M steps, which reached lower
success than the 1M checkpoint they continued from.

So §3.1's description does not match the run the results come from, and the
checkpoints §3.3.4.4 asks for do not exist for it. The three retrained SAC seeds
reported in Table 4 were likewise trained at fixed RPM throughout.

Either correct §3.1 to state fixed-speed training for the full 1M steps and drop
§3.3.4.4, or run the staged curriculum deliberately and report it as a separate
experiment. I would not fill numbers into §3.3.4.4 as it stands, because there is
no run they describe.

Related: with propeller speed pinned at 12 RPM the speed-regulation reward term
`r_spd` is **identically zero at every step** of training. Table 8's row for it
should read 0.000 / 0.0 %, not a measured value.

---

## Where these numbers come from

| Table | Source |
|---|---|
| Tables 4, 5 | `per_episode/sac_gs4_seed{0,1,2}_best/`, `ppo_fx_seed{0,1,2}_best/`, `los_apf_s1/` |
| Table 6 | same, paired on `episode_id` via `src/compare.py` |
| §3.1 workspace | `src/config.py`, `layouts/eval_layouts_v1.json` |
| §3.1 manoeuvring | `src/ship.py`, simulated directly |
| Appendix A | `src/ship.py` |
| Tuning budget | `tuning/apf_tuning_results.csv` |

Reproduce the tables with `python src/make_outputs.py --all`; full detail and the
reasoning behind each methodological choice is in `BASELINES_RESULTS.md` and
`MANUSCRIPT_BRIEF.md`.
