# Vessel-model calibration: two basin sessions

Date: 2026-11-04. Status: plan, before booking. It carries forward the part-2 basin plan
(`planning_records/PART2_BASIN_PLAN.md`, 2026-09) with the changes of section 1, and is the plan
of record for the calibration sessions. The field trials that follow are planned in
`FIELD_TRIALS_PLAN.md`.

## 1. Purpose in Paper 3, and what changed

The deployed policy is frozen (SAC seed 0, baseline-v3, best validation checkpoint). The
calibration does **not** retrain it, and the domain-randomisation retraining ablation (the old
RQ4, Study 3) stays out of the paper. The calibration serves four uses:

1. **Model provenance.** An archived identification dataset and the standard manoeuvring
   quantities (turning circle, zig-zag overshoots, thrust curve, actuator response, stopping),
   which the simulator description (Section 2.3.5) cites. This answers the Paper 2 reviewer
   concession that the identification data were not archived.
2. **Coverage of the training distribution.** Training randomised the plant by blending whole
   bootstrap parameter vectors of the identified model (`bluefin/ship_model_v3.sample_params`).
   The newly identified parameters, and their day-to-day variation, are checked against that
   randomisation. Inside it: the real vessel lies within the distribution the policy trained on,
   a statement for Section 5. Outside it: the field results are read as out-of-distribution
   transfer, and the paper says so.
3. **A calibrated simulator for the sim-to-field comparison** (`FIELD_TRIALS_PLAN.md`
   section 7): each field scenario re-simulated with the calibrated model beside the training
   model. It is an analysis model only; no Section 4 result changes.
4. **The safety layer's physical assumptions.** The reported layer (E1) ends in a full-astern
   stop with the rudder centred, a command never issued on the water. Its detector brackets the
   reverse response between a weak model (efficiency 0.25, onset 0.75 s) and a strong one (0.5,
   no delay), and the main residual collision mode on test set v4 is drift into the boundary
   after a committed stop (16 of 70 own-vessel collisions). The crash stops measure all three.

**Changes from the part-2 plan:**
- Crash stops (S1-C2) extended with stops from a steady turn, the residual yaw and sway that
  carried the stopped hull into walls in simulation.
- Session 2's closed-loop block (S2-C) no longer compares two retrained policies. It runs the
  deployed policy on the static layouts as the field-trial dress rehearsal and the closed-loop
  check of the calibrated simulator.
- New bench items: the Paper 3 bridge (P-9) and the safety layer's run time on the shore laptop
  (P-10). New thresholds T-12 to T-14 for the stop.

## 2. What the current model leaves unmeasured

The current model (`bluefin/REPORT.md`, identified from 11 minutes of July 2026 field logs, 6/6
holdout runs better than the previous model) rests on four assumptions:

| Gap | Current state | Fixed by |
| --- | --- | --- |
| Sustained turns | **priors, not measurements**: turn rate 2.2/5.0/7.7/10.0 deg/s at 25-100 % helm, radius 3.05 L, speed ratio 0.82 imposed as constraints; the July runs hold no sustained turn | S1-D turning circles |
| Thrust vs RPM | `T12` anchored at one operating point; square law assumed | S1-C RPM sweep and coast-downs |
| Actuator decomposition | `rud_rate`, `rud_tau`, `rud_delay` trade off freely; 0.73 s is an effective lump of latency, servo and estimator lag | S1-B servo steps with an independent rudder measurement |
| Reverse thrust and its delay | **absent**: no July run commands astern; the simulator assumes efficiency 0.5 with no delay | S1-C2 crash stops |
| Low-speed sway and yaw decay | no data near rest | S1-C2 stops from a turn, S1-C coast-downs |

The bootstrap intervals resample runs from one session, so they are a lower bound on parameter
uncertainty; S2-A measures day-to-day variation.

## 3. Before booking (bench, no basin time)

| Item | What | Why it cannot wait for the basin |
| --- | --- | --- |
| P-1 | Telemetry and log rate raised to at least 10 Hz for pose and command in session 1 | servo dynamics are faster than the 1 Hz Nyquist limit of 2 Hz logs |
| P-2 | Every script derives dt from timestamps and checks it against the logged configuration | session 1 runs at a different rate from the 2 Hz control rate |
| P-3 | Independent rudder angle measurement: rudder feedback if it exists, else a 60 fps camera on a protractor at the rudder post with a synchronising event | without it S1-B yields another lumped delay |
| P-4 | Rate limiter state logged (off by default, as in training: `RUDDER_COMMAND_LIMIT = False`) | the simulator and the bridge must agree |
| P-5 | One clock for IMU (100 Hz or more, raw gyro and accelerometer), LiDAR and commands, or a measured offset; a stationary period and a sharp yaw impulse at the start and end of each session | a clock offset appears in the fit as actuator lag |
| P-6 | Analysis pre-registered: empty tables, metric definitions, thresholds (section 7) and the committed fit script | session 1's mid-session check runs that script unchanged |
| P-7 | Landmark survey of fixed facility features for scan-to-map registration; furniture at the far end excluded | registration anchors the pose without external instrumentation |
| P-8 | Reverse on the bench: the ESC accepts negative S2, dead band, ramp, current on a direct +62 to -100 reversal | **if the ESC cannot reverse, S1-C2 is void and the safety layer's stop does not exist on the water** |
| P-9 | Paper 3 bridge (`FIELD_TRIALS_PLAN.md` section 3), at least for S2-C | the existing bridge builds the Paper 2 observation |
| P-10 | Run time of the policy plus the E1 layer per decision on the shore laptop, p99 below the 0.5 s decision interval with communication | otherwise the layer cannot run in the loop at 2 Hz |

## 4. Session 1: identification (about 4 h)

Ordered by irreversibility; the priority block S1-D comes before the discretionary ones.

| Block | Time | Content | Yields |
| --- | --- | --- | --- |
| S1-A Sensing baseline | 25 min | Stationary at a known pose, 5 min, then a sharp yaw impulse; at three positions (mid-basin, near the light wall, near the matte black wall) | gyro bias, localisation drift at rest, LiDAR returns and range noise, return rate against the black wall, aft self-occlusion |
| S1-B Actuators (limiter off) | 35 min | Rudder steps 0 to +/-25, 50, 100 % and +/-100 to -/+100 %, 5 repeats each; one 10 %/s ramp | servo rate, first-order lag and transport delay, separately |
| S1-C Surge | 45 min | Acceleration from rest at RPM 6, 9, 12, 15, 18, 21, 24 (2 runs each); coast-downs from RPM 9, 15, 21 (3 runs each) | thrust curve and drag, separately |
| S1-C2 Crash stops | 30 min | From straight settled speed at RPM 9, 12, 15 (3 runs each); **from a steady turn at RPM 12, helm 50 % and 100 %, both signs (2 runs each)**; full astern, rudder centred, as the reported layer's stop; logged to rest; side camera on a marked line | reverse efficiency, thrust delay, astern overshoot, head reach; heading change, yaw-rate decay and lateral drift from the stop command to rest |
| S1-D Turning circles (priority 1) | 70 min | Helm 25, 50, 75, 100 %, both signs, RPM 12, two full turns each from settled speed; helm 50 and 100 % at RPM 18 | steady turn rate, radius, speed loss, advance, transfer, tactical diameter; drift-moment asymmetry |
| S1-E Zig-zags | 30 min | 20/20 (3 repeats) and 10/10 (2 repeats), both initial directions, RPM 12 | overshoot angles, yaw inertia and damping |
| S1-F Mid-session check | 25 min | The committed fit script on the day's data; thresholds of section 7; anything failing is reshot | keeps session 2 free for validation |

If a full turning circle does not fit the basin at RPM 12, the largest settled arc is logged (a
settled 180 deg arc still yields steady rate and speed) and the constraint is recorded.

## 5. Session 2: validation and dress rehearsal (about 4 h)

Session 2 must not become a second identification session: the model's accuracy claim needs data
it was not fitted on.

| Block | Time | Content | Yields |
| --- | --- | --- | --- |
| S2-A Repeatability | 45 min | Without refitting: turning circles at helm 50 and 100 %, one RPM sweep point, one 20/20 zig-zag | day-to-day parameter variation (the honest randomisation width) |
| S2-B Open-loop replay | 45 min | Fixed command sequences from session 1 and from the policy, no feedback, against the model's free-run prediction; three crash stops at RPM 12 as holdout | heading and position error against horizon (as `bluefin/REPORT.md` V1); holdout for the stop parameters |
| S2-C Closed-loop dress rehearsal | 80 min | The deployed policy through the Paper 3 bridge, no target, on layouts L1, L2 and L3 and an open leg, 2 runs each; field-trial logging and the safety layer in the field-trial mode | closed-loop check of the calibrated simulator (each run re-simulated from its logged start); rehearsal of the trial procedure, positioning, reset and abort |
| S2-D Contingency | 40 min | Unallocated; recovers a lost session-1 block at the cost of S2-C | - |

## 6. Fit and outputs

- Fit on session 1 only, with the committed scripts (`bluefin/fit_final.py`, `validate.py`,
  `acceptance.py`); validate on session 2 only. If a session-1 block has to be refitted with
  session-2 data, the paper says so rather than reporting the holdout as independent.
- **Outputs:**
  1. raw logs with the configuration of every run, and a manoeuvre index (run, manoeuvre,
     parameters, validity);
  2. the landmark survey;
  3. the manoeuvring quantities of section 4 in reportable form;
  4. an updated parameter set with day-to-day intervals and the acceptance suite re-run;
  5. a coverage report: the identified parameters and manoeuvring quantities against the
     training randomisation;
  6. a safety-layer report: the measured reverse response against the detector's weak and strong
     models, stopping distance, and drift after a stop from a turn;
  7. the calibrated analysis model for `FIELD_TRIALS_PLAN.md` section 7.
- The training simulator (baseline-v3) stays frozen.

## 7. Pre-committed acceptance thresholds (filled before session 1, checked at S1-F)

| Id | Quantity | Block | Threshold |
| --- | --- | --- | --- |
| T-1 | servo rate limit | S1-B | resolved to +/-20 %, separated from delay |
| T-2 | transport delay | S1-B | resolved to +/-0.1 s |
| T-3 | thrust curve exponent | S1-C | fitted, not assumed; at least 5 RPM points usable |
| T-4 | steady turn rate at full helm | S1-D | measured at 3 or more helm settings, both signs |
| T-5 | turning radius | S1-D | measured; replaces the 1.5-4 L prior band |
| T-6 | zig-zag overshoot | S1-E | both manoeuvres, 2 or more valid repeats |
| T-7 | black-wall return rate | S1-A | resolved against bearing at 3 or more headings |
| T-8 | localisation drift | S1-A | stationary drift rate quantified |
| T-9 | reverse thrust efficiency | S1-C2 | resolved to +/-0.1, at 2 or more RPM settings |
| T-10 | thrust transport delay | S1-C2 | resolved to +/-0.1 s |
| T-11 | crash-stop head reach from RPM 12 | S1-C2 | measured; at most 1.5 Lpp (2.36 m), or recorded as failing |
| T-12 | reverse response against the detector | S1-C2 | efficiency and onset inside the weak-strong bracket (0.25-0.5, 0-0.75 s), or recorded as outside with its consequence for the layer |
| T-13 | drift after a stop from a turn | S1-C2 | heading change and lateral drift to rest measured at helm 50 and 100 %, both signs |
| T-14 | layer run time on the shore laptop | P-10 | p99 below 0.5 s per decision with communication |

## 8. Contingency ordering

If session 1 runs short, drop in this order (last dropped is most valuable):
1. S1-C RPM sweep at the extremes (keep 9, 12, 18);
2. S1-E 10/10 zig-zag (keep 20/20);
3. S1-C2 straight stops at RPM 9 and 15 (keep RPM 12) and the helm-50 % stops from a turn (keep
   helm 100 %);
4. S1-B ramp (keep the steps);
5. S1-D RPM 18 circles (keep RPM 12, every helm setting, both signs).

**Never dropped:** S1-A, S1-D at RPM 12, S1-B steps, S1-C2 at RPM 12 (straight and from a turn),
S1-F. A block lost entirely is recovered in S2-D at the cost of S2-C, never by refitting on
session-2 validation data.

## 9. Open items

| Id | Question | Consequence |
| --- | --- | --- |
| O-1 | Can the vessel telemetry exceed 2 Hz, and by how much? | decides whether S1-B separates the actuator terms |
| O-2 | Is the IMU installed, raw at 100 Hz or more, on the bridge clock? | surge and yaw-rate identification; field ego estimate |
| O-3 | Does a full turning circle fit the basin at RPM 12? | tactical-diameter quantities available or not |
| O-4 | Rudder position feedback, or the P-3 camera method? | S1-B decisive or lumped |
| O-5 | Is the black wall still present and matte? | registration design |
| O-6 | Does the ESC reverse, with what dead band and ramp (P-8)? | **the layer's stop on the water** |
| O-7 | Is a direct full-ahead-to-full-astern reversal safe for motor, gearbox and battery? | if a ramp is needed, the layer's stop and the simulator's stop distances change |
| O-8 | Basin booking dates and lead time; risk assessment | schedule |
