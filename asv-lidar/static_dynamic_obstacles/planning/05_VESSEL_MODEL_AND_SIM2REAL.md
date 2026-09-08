# 05 — Vessel Model Recalibration and Sim-to-Real

**Revision 2** — O3 resolved: sim-to-real is retained **in this paper as RQ4**, delivered
as a domain-randomisation ablation evaluated in the field (Study 3). This document is
otherwise unchanged by the two-vessel repositioning; the identification work is
independent of encounter scope.

**Handover target:** Claude chat (planning) + field work
**Depends on:** nothing — **start this in parallel**, it gates on basin booking lead time
**Feeds:** 03 (domain randomisation parameters), 01 (noise characterisation)

---

## 1. Purpose and framing

Paper 2 identified that the primary sim-to-field gap is not the decision-making module
but the **mismatch in dynamic response**. Field trajectories preserved the intended
behaviour but showed larger oscillations, wider turns, and roughly double the RMS
cross-track error.

**This is also a direct response to a Paper 2 concession.** Reviewer 1.4 requested the
complete numerical model, hydrodynamic coefficients, actuator dynamics, saturation
limits, delays and system-identification validation. The response conceded that the raw
identification data was not archived in a form supporting independent reporting, and the
model was described as *calibrated* rather than identified.

**So the deliverable is not merely a better model.** It is a publishable, archived
identification dataset with validation figures. Design the trial to produce that
artefact.

---

## 2. Diagnosis before design

The observed symptoms — wider turns, larger oscillation, doubled RMS CTE — point to
**actuator lag, underestimated yaw damping, and rudder effectiveness mismatch**, not to
missing degrees of freedom.

Add an explicit **actuator model** before adding hydrodynamic complexity:

- Servo rate limit
- First-order lag
- Transport delay (UDP round trip at 10 Hz control rate is itself a delay source)
- Thrust map (Paper 2 used thrust ∝ RPM²; verify)

Keep the 3-DOF Fossen structure with linear and quadratic damping. Existing values from
Paper 2 Appendix A: m = 64.55 kg, Iz = 10.45 kg·m², X_u̇ = 3.66 kg, Y_v̇ = 62.74 kg,
N_ṙ = 0.63 kg·m², X_u = 2.00 kg/s.

---

## 3. Manoeuvre set

| Manoeuvre | Identifies |
|---|---|
| Straight-line accel/decel at several throttle settings | Surge damping, thrust map |
| Turning circles at multiple rudder angles and speeds | Yaw damping, rudder force, drift angle |
| **Zig-zag 10/10 and 20/20** | Yaw inertia, damping, actuator lag |
| Rudder step response at fixed speed | Actuator lag, rise time |
| Pull-out or spiral | Course stability |
| Stop test | Deceleration, relevant to Rule 8(e) speed reduction |

**Zig-zag overshoot angles are the single most informative measurement for the observed
symptom.** Prioritise them if basin time is constrained.

---

## 4. Localisation for identification — no external instrumentation

**O6 reframed.** There is no IMU, no total station and no motion capture. The only sensors
are the LiDAR and a pose/heading estimate. This is a software problem, not a procurement
problem, which removes the longest lead time in the project.

### 4.1 Why rf2o is unfit and what replaces it

`rf2o_laser_odometry` does **scan-to-scan** matching: every estimate is relative to the
previous one, so error integrates and drift is unbounded. Adequate for closed-loop control,
not for identifying dynamics.

**Scan-to-map registration** replaces it. Register each scan against the *surveyed facility
geometry* rather than against the previous scan. Error becomes bounded rather than
accumulating. For identification runs the basin is empty, which is the easiest possible case.

Note the LiDAR does not see the pool edge at all — it sees the facility walls 1–2 m beyond
it. Those walls are rigid, static and surveyable, which makes them a better reference than
the boundary itself.

### 4.2 What is observable

Two long parallel walls within range; end walls typically beyond the ~8–12 m sensor range.

| Quantity | Observability | Source |
|---|---|---|
| Heading | **Excellent** | Wall orientation — strongest constraint available |
| Lateral position | **Excellent** | Wall-to-wall distance |
| Along-track position | **Sparse** | Only where recessed doorways or protruding benches are in range |

Two parallel walls alone are degenerate along their axis. The facility's fixed features —
doorways, benches, wall-mounted equipment — break that degeneracy. **Survey them; do not
build artificial features, and do not erect a barrier that would occlude them** (this is why
O5 resolved to software gating).

### 4.3 Consequences for the manoeuvre set

| Manoeuvre | Viable? | Note |
|---|---|---|
| Zig-zag 10/10, 20/20 | **Yes** | Overshoot angles are read directly off the heading trace — the best-constrained quantity. Prioritise these |
| Turning circles | **Partial** | Needs 2D trajectory; along-track constrained only near features |
| Rudder step response | **Yes** | Heading-based |
| Straight-line accel/decel | **Weak** | Fully along-track dependent. Two workarounds below |
| Stop test | **Weak** | Same |

**Surge workarounds.** (a) Time between surveyed features gives absolute mean speed over a
segment — sparse but drift-free. (b) Position straight-line runs so an end wall stays within
LiDAR range, giving absolute along-track for the acceleration phase, which is what surge
damping needs.

### 4.4 Fitting without a gyro

Differentiating heading at 10 Hz gives yaw rate with noise comparable to the rates being
identified; filtering fixes the noise but adds lag, corrupting the lag parameter itself.

**Fit to heading directly.** Let the model integrate r internally and compare predicted
against measured heading in the prediction-error objective. No differentiation, and it suits
the available instrumentation exactly.

### 4.5 Validation without external truth

- **Static tests** — vessel held at surveyed positions; gives absolute accuracy
- **Closed-loop runs** — return to the same physical point; gives drift
- Both cheap, both reportable, and together they discharge the archiving obligation

### 4.5a Reflectivity risk — check this first

**One full-length wall of the facility is matte black** (the climbing-wall side). The C1
operates near 905 nm, where carbon-based black finishes can fall to single-digit
reflectivity. If that wall returns poorly, the wall-to-wall lateral constraint disappears
and scan-to-map degrades to relative odometry.

**Measure before designing around it.** First action on the retained logs: plot return
density and range against bearing for runs of known heading. A return-rate hole on the black
side settles it either way.

**Likely mitigation — register against surveyed landmarks, not a continuous polygon.**
Mounted on that wall are objects that will return strongly regardless of the wall itself: a
white sign, the climbing net and its holds, aluminium ladders, and the structural rail along
the top. Registering against a sparse set of surveyed landmarks is more robust than
continuous wall matching, and it solves the along-track constraint at the same time, since
those features are distinctive and irregularly spaced.

### 4.6 Risks to manage

- **Movable deck items.** The far end carries chairs, tables, a whiteboard and a mobile
  equipment rack, all near scan height and all liable to shift between sessions. Survey only
  genuinely fixed features — doorways, structural columns, climbing-wall hardware,
  wall-mounted fixtures. Photograph the reference set each session
- **Water-surface returns** at grazing angles may produce spurious near-field points.
  Check the existing logs before trusting registration
- **Operators on the deck** sit at scan height and move. Handled by the geometric gate
  (01 §3.4), but they must not enter the localisation reference either

### 4.7 IMU — CONFIRMED, specification

An IMU will be added. This removes the yaw-rate observability constraint and, via the
accelerometer, largely rescues the surge measurements that two parallel walls could not
constrain. The manoeuvre set in §3 is now fully covered.

**Log raw gyro and accelerometer, not a fused orientation output.** Fusion firmware applies
unknown filtering that corrupts exactly the lag and damping parameters being identified.
BNO055 in raw mode, ICM-20948, or a Pixhawk log all work. 100 Hz or better.

**Time synchronisation is the detail that will bite.** A constant offset between IMU and LiDAR
timestamps appears in the fit as actuator lag, so a clock error would be absorbed into the
model as a physical parameter. Log both on one clock if possible; otherwise begin each run with
a sharp yaw impulse visible in both streams and align on it.

**Mounting and calibration.** Rigid mount near the centre of gravity, axes aligned with body
axes, mounting offset recorded. Stationary period at the start of every run for gyro bias
estimation.

**Sensor fusion.** Scan-to-map supplies absolute pose at 10 Hz and is drift-free; the IMU
supplies yaw rate and acceleration at high rate. The two are complementary — the registration
anchors, the IMU fills in between and provides the derivatives directly. This also improves the
`ego` observation branch in the field, reducing the u/v/r sim-to-real gap identified in §6.

**Fitting.** With a gyro available, fit to yaw rate *and* heading and cross-check the two.
Disagreement indicates a synchronisation or mounting error rather than a model deficiency.


## 5. Fit and validation

**Fit:** prediction-error minimisation over the manoeuvre set. Hold out one zig-zag and
one turning circle. Report the fit metric and parameter confidence intervals — the CIs
are needed for §7.

**Validation (free, do this first):** replay the exact command sequences from the
existing Paper 2 field logs through the calibrated model and overlay the trajectories
against the recorded ones. The logs are already retained. This costs no basin time and
produces a strong figure.

**Then check:** retrain the policy on the calibrated model and see whether the CTE gap
narrows.

**If it does not, that is itself a reportable finding** — it localises the gap to
localisation and disturbance rather than dynamics, which is a legitimate and useful
result rather than a failure.

---

## 6. Noise characterisation from existing logs

Three items, all extractable from the retained Paper 2 logs without basin time. Group
them as one subsection of the protocol rather than three scattered decisions — together
they are the direct answer to Reviewer 1.4.

| Quantity | Feeds |
|---|---|
| rf2o pose drift magnitude and character | boundary raycast noise (01 §3.3), tracker ego-motion compensation |
| Ego velocity error — u, v, r differentiated from noisy pose | `ego` observation branch (a sim-to-real gap in the *observation*, not just the dynamics) |
| LiDAR returns per revolution and range noise | raw scan simulation (01 §2.3) |
| Aft self-occlusion sector from mount geometry | masked bearing range in raw scan |
| Scan motion distortion at realistic yaw rates | tracker velocity estimate noise |

---

## 7. Domain randomisation

**A better nominal model reduces bias but does not create robustness.**

Pair identification with randomisation over the identified parameters ± their confidence
intervals, plus the noise sources in §6 and disturbance injection (wind, current — Paper
2's field tests were calm water with no generated waves or current, and this was
acknowledged as a limitation).

*"We identified the model and randomised within identification uncertainty"* is a
materially stronger claim than either half alone.

---

## 8. Field trial design for the dynamic case

Separate from the identification trial, and needing its own planning.

**Open questions:**

- **What is the target vessel physically?** A second Bluefin, a towed float, an RC boat?
  Each has different fidelity and different control precision.
- **Repeatability.** How is a consistent encounter geometry achieved across runs? Paper
  2 ran each of three scenarios once and had to qualify the claims to "feasibility rather
  than robustness". Repeated trials would materially strengthen Paper 3.
- **Dual localisation.** Both vessels need pose in a common frame. Does the target need
  its own localisation, or is a scripted trajectory with external tracking sufficient?
- **Abort procedure and safety.** Two vessels converging in a confined basin.
- **Risk assessment and ethics/basin booking lead time.**

Field deployment and simulation now both use **one** dynamic target (S1), so the field
trials exercise the same configuration the policy was trained on rather than a reduced
case. This is the main practical benefit of the two-vessel repositioning: every reported
behaviour is physically reproducible.

---

## 9. Study 3 — domain randomisation ablation (RQ4)

**O3 resolved: sim-to-real is retained in this paper as RQ4.** It is delivered as an
ablation, not as a demonstration, which is what makes it a research question rather than a
validation section.

**RQ4 — does domain randomisation over identified model uncertainty close more of the
sim-to-field gap than improved nominal identification alone?**

Three policies, each trained with five seeds and evaluated in the field on the same
scenario set:

| Arm | Training model |
|---|---|
| A | Paper 2 nominal model (uncalibrated baseline) |
| B | Identified nominal model, no randomisation |
| C | Identified model + randomisation over parameters ± identification CI, plus the §6 perception noise |

Report the sim-to-field delta per arm on RMS cross-track error, success rate and COLREGs
compliance. The interesting comparison is B vs C: if B alone closes the gap, better
modelling was sufficient; if it does not, robustness was the missing ingredient. **Either
outcome is reportable**, which is what makes this a safe study to pre-commit to.

Arm A may be dropped if basin time is tight — B vs C carries the research question.

This is also the third novelty axis (N3) and the reason the paper retains a
deployment-oriented framing after the encounter scope was narrowed.

---

## 10. Open items

- ~~IMU feasibility~~ — **confirmed, will be added** (§4.7). Specify part, logging rate and
  time-sync method
- ~~O6 external instrumentation~~ — resolved: scan-to-map pipeline, no purchase
- ~~O5 physical barrier~~ — resolved: software gating, walls retained for localisation
- **Measure black-wall return rate from existing logs (§4.5a) — do this first**
- Survey the facility geometry and genuinely fixed features only
- Check whether suspension lines cross the scan plane near the pool edges
- Target vessel platform and repeatability protocol (§8)
- Whether Arm A is retained in Study 3, subject to basin time
- Book basin time for the identification trial
- Run the log-replay validation (§5) — no booking required, do this first
- Extract all four noise characterisations (§6) — no booking required
