# 05 — Vessel Model Recalibration and Sim-to-Real

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

## 4. Instrumentation — decide before booking (O6)

**`rf2o_laser_odometry` is not adequate ground truth for identifying yaw dynamics.** It
is scan-matching odometry: it drifts, and it degrades further once moving objects are in
the scan — which is precisely the Paper 3 condition.

Required:

- **IMU logging yaw rate at high rate.** Cheap, high value, directly measures the
  quantity the zig-zag tests are about.
- **External ground truth**, ideally: total station, overhead camera with fiducial
  markers, or motion capture if the basin supports it.

This is an equipment and booking decision with lead time, which is why 05 starts in
parallel with everything else.

---

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

Note that field deployment has **one** target, versus up to three in simulation. The
masked-slot configuration must be well represented in training (01 §6.2).

---

## 9. Scope decision (O3)

Two open questions, related:

1. Does sim-to-real transfer become a standalone research question (RQ4) within Paper 3?
2. Should this work split off as its own short contribution — an identification and
   sim-to-real study using the existing logs plus one dedicated trial?

Arguments for splitting: it de-risks both papers, produces a publishable artefact while
the COLREGs training campaign runs, and directly discharges the Reviewer 1.4 concession
in a venue where it is the main contribution rather than an appendix.

Argument against: it removes the field validation from Paper 3, weakening the
deployment-oriented framing that distinguished Paper 2.

**Revisit once §3 has a concrete manoeuvre plan and a basin booking date.**

---

## 10. Open items

- **O6** — external ground truth instrumentation; gates the booking
- **O3** — standalone RQ / separate paper
- **O5** — physical barrier at basin edge (shared with 01; resolve in the same site visit)
- Book basin time for the identification trial
- Run the log-replay validation (§5) — no booking required, do this first
- Extract all four noise characterisations (§6) — no booking required
