# Vessel-model calibration: two basin sessions

Date: 2026-11-04; bench results of the Paper 3 bridge added 2026-11-05 (section 3a); turns at
the cruise setting (RPM 6) added 2026-11-14 (S1-D, S1-E). Status:
plan, before booking. It carries forward the part-2 basin plan
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
- Turning circles (S1-D) and a 20/20 zig-zag (S1-E) at the cruise setting, RPM 6, beside those
  at RPM 12. The deployed policy cruises at RPM 6 and commands RPM 0-12, while the July model
  was identified at RPM 12 (18 of 20 runs) and the turning priors are set at RPM 12; without
  these runs the vessel's turning would not be measured at the speed it runs at in the trials.
- Session 2's closed-loop block (S2-C) no longer compares two retrained policies. It runs the
  deployed policy on the static layouts as the field-trial dress rehearsal and the closed-loop
  check of the calibrated simulator.
- New bench items: the Paper 3 bridge (P-9), run time on the shore laptop (P-10), the shore
  laptop's network (P-11) and an operator stop independent of it (P-12). New thresholds T-12 to
  T-14 for the stop and run time, and T-15 for phantom tracks (from the bridge's bench results,
  section 3a).

## 2. What the current model leaves unmeasured

The current model (`bluefin/REPORT.md`, identified from 11 minutes of July 2026 field logs, 6/6
holdout runs better than the previous model) rests on four assumptions:

| Gap | Current state | Fixed by |
| --- | --- | --- |
| Sustained turns | **priors, not measurements**: turn rate 2.2/5.0/7.7/10.0 deg/s at 25-100 % helm (RPM 12), radius 3.05 L, speed ratio 0.82 imposed as constraints; the July runs hold no sustained turn, and none at the cruise setting (RPM 6) | S1-D turning circles at RPM 12 and 6 |
| Thrust vs RPM | `T12` anchored at one operating point; square law assumed | S1-C RPM sweep and coast-downs |
| Actuator decomposition | `rud_rate`, `rud_tau`, `rud_delay` trade off freely; 0.73 s is an effective lump of latency, servo and estimator lag | S1-B servo steps with an independent rudder measurement |
| Reverse thrust and its delay | **absent**: no July run commands astern; the simulator assumes efficiency 0.5 with no delay | S1-C2 crash stops |
| Low-speed sway and yaw decay | no data near rest | S1-C2 stops from a turn, S1-C coast-downs |

The bootstrap intervals resample runs from one session, so they are a lower bound on parameter
uncertainty; S2-A measures day-to-day variation.

## 3. Before booking (bench, no basin time)

| Item | What | Why it cannot wait for the basin |
| --- | --- | --- |
| P-1 | Telemetry and log rate raised to at least 10 Hz for pose and command in session 1, and pose resolution finer than the July 0.1 m and 0.1 deg | servo dynamics are faster than the 1 Hz Nyquist limit of 2 Hz logs; at the July resolution the bridge's ego inputs differ from the simulator's by 0.11-0.16 on average (normalised units, section 3a). At 10 Hz the decoder's velocity from pose differences spans 0.1 s and gets noisier, so its estimator changes with the rate (or the IMU supplies the ego motion, O-2) |
| P-2 | Every script derives dt from timestamps and checks it against the logged configuration | session 1 runs at a different rate from the 2 Hz control rate |
| P-3 | Independent rudder angle measurement: rudder feedback if it exists, else a 60 fps camera on a protractor at the rudder post with a synchronising event | without it S1-B yields another lumped delay |
| P-4 | Rate limiter state logged (off by default, as in training: `RUDDER_COMMAND_LIMIT = False`) | the simulator and the bridge must agree |
| P-5 | One clock for IMU (100 Hz or more, raw gyro and accelerometer), LiDAR and commands, or a measured offset; a stationary period and a sharp yaw impulse at the start and end of each session | a clock offset appears in the fit as actuator lag |
| P-6 | Analysis pre-registered: empty tables, metric definitions, thresholds (section 7) and the committed fit script | session 1's mid-session check runs that script unchanged |
| P-7 | Landmark survey of fixed facility features for scan-to-map registration; furniture at the far end excluded; the bridge then runs with the registered map frame (`--origin map`) | registration anchors the pose without external instrumentation; it is also the expected fix for the phantom moving tracks at the basin wall (section 3a) |
| P-8 | Reverse on the bench: the ESC accepts negative S2, dead band, ramp, current on a direct +62 to -100 reversal | **if the ESC cannot reverse, S1-C2 is void and the safety layer's stop does not exist on the water** |
| P-9 | Paper 3 bridge (`FIELD_TRIALS_PLAN.md` section 3), at least for S2-C. **Done 2026-11-05** (section 3a). Repeated 2026-11-14 on the personal laptop (shore-laptop candidate, O-10; `field_deployment/UDP_BENCH_HANDOVER.md`, section 4): parity 32/32 runs and 1,497 decisions equal; the 19 July logs in process and three over UDP loopback equal to the in-process run and reproduced offline, so the UDP replay is no longer provisional. No separate two-device test (handover step 6 not run); UDP across devices is first exercised by the vessel link (handover step 7) at the next basin booking | the existing bridge builds the Paper 2 observation |
| P-10 | Run time per decision on the shore laptop, with communication: the command path (observation, policy, command) and, for the optional active block only, the E1 layer in the loop; p99 below the 0.5 s decision interval | otherwise the policy, or the layer, cannot run at 2 Hz. Development laptop (section 3a): command path met, E1 in the loop not met. Personal laptop (2026-11-14, 2 cores, other applications holding about half the CPU): command path p99 0.03 s on test-set streams and 0.09 s from the LiDAR line to the sent command on the July logs over UDP loopback, met; E1 in the loop p99 0.89 s with 5 % of decisions above 0.5 s, not met. Active mode therefore runs at 2 Hz on neither laptop; shadow mode is unaffected. The vessel-link latency is not in these figures |
| P-11 | Shore laptop and network: free UDP ports for the bridge and the vessel link, UDP allowed on the session network | on the development laptop UDP 5000, 5001 and 5050 are held by system services (the July bridge listened on 5000), and the university network blocks UDP. Personal laptop (2026-11-14), the shore laptop, linked to the laptop that controls the vessel with the July ports and command line (bridge on 5000, the vessel-control laptop on 5050; `udp_live_p3.py` keeps the July option names and accepts `--shadow` for `--no-send`): UDP 5000 (bridge), 5001, 15000 and 15050 free; 5050 held by the Connected Devices Platform service, which matters only for a fake vessel on the same laptop; UDP not restricted on its networks. No two-device bench test (handover step 6 not run); the vessel link (handover step 7: START reaches the vessel, frames and pose pair at 2 Hz, no `$CMD` sent) is scheduled for the start of the next basin booking, before any session |
| P-12 | An operator stop independent of the shore laptop (radio-control override or vessel-side kill), tested on the bench | the bridge's stop key acts only while the bridge runs and its window has focus |

## 3a. Bench results of the Paper 3 bridge (2026-11-05)

Record: `field_deployment/README_P3_BRIDGE.md`, outputs under `field_deployment/p3_bench/`
(development laptop, not the shore laptop).

- **Bridge.** `field_deployment/udp_live_p3.py` (live, UDP) on `p3_bridge.py` (core). The
  observation is built by the simulator's own environment class with its sensor models replaced
  by the live scan, pose and ego motion; `src/`, `configs/`, `env.py` and `udp_live_rl.py` are
  unchanged. E1 in shadow mode runs in a separate worker process, so it never delays a command.
- **Parity (P-9):** 16 test set v4 episodes in shadow and active mode, 32/32 runs and 1,497
  decisions exactly equal to the simulator (observation, action, issued command, every
  safety-layer decision); the simulator side equals the recorded E1 test-set traces in 16/16
  episodes; with the staleness model on, 6/6.
- **Shadow replay:** all 19 July logs in process (1,085 decisions) without error; three over UDP
  loopback with `fake_vessel_replay.py`, equal to the in-process run; repeated 2026-11-14 on the
  personal laptop with the same result (P-9), so no longer provisional. UDP from a second device
  is first exercised by the vessel-link check at the next basin booking.
- **Run time (P-10, T-14):** command path p99 0.04 s on test-set streams and 0.09 s from the
  LiDAR line to the sent command on the July logs. E1 in the loop p99 0.89 s (test set) and
  2.06 s (July logs): **active mode does not run at 2 Hz on this laptop**; shadow mode is
  unaffected.

**Findings that change the sessions:**

| Finding | Where it is handled |
| --- | --- |
| Phantom moving tracks: in the target-free July logs, 42 % of decisions carry a moving track, 61 % of them within 1 m of the map boundary or beyond it. They reach the policy's target branch, and the shadow layer would commit a stop in 15 of 19 replays. The July runs were open loop at about twice the deployed cruise speed, so this tests the pipeline, not the trials | P-7 (registered map frame), S2-C and T-15 |
| Telemetry resolution: at 0.1 m and 0.1 deg at 2 Hz the ego, target and context inputs drift well away from the simulator's | P-1, O-1, O-2 |
| Pose reference point: whether the localisation reports the hull reference point or the LiDAR (0.86 m ahead) is not established; the bridge supports both (`--pose-point`) | S1-A, O-9 |
| UDP ports and network on the development laptop | P-11 |
| Operator stop depends on the shore laptop's bridge window | P-12 |

## 4. Session 1: identification (about 4 h 50 min)

Ordered by irreversibility; the priority block S1-D comes before the discretionary ones.

| Block | Time | Content | Yields |
| --- | --- | --- | --- |
| S1-A Sensing baseline | 25 min | Stationary at a known pose, 5 min, then a sharp yaw impulse; at three positions (mid-basin, near the light wall, near the matte black wall), at a surveyed pose and heading | gyro bias, localisation drift at rest, LiDAR returns and range noise, return rate against the black wall, aft self-occlusion; **the pose reference point** (the telemetry pose against the surveyed hull and LiDAR positions, O-9); the bridge's conversion from real telemetry, which the bench cannot check |
| S1-B Actuators (limiter off) | 35 min | Rudder steps 0 to +/-25, 50, 100 % and +/-100 to -/+100 %, 5 repeats each; one 10 %/s ramp | servo rate, first-order lag and transport delay, separately |
| S1-C Surge | 45 min | Acceleration from rest at RPM 6, 9, 12, 15, 18, 21, 24 (2 runs each); coast-downs from RPM 9, 15, 21 (3 runs each) | thrust curve and drag, separately |
| S1-C2 Crash stops | 30 min | From straight settled speed at RPM 9, 12, 15 (3 runs each); **from a steady turn at RPM 12, helm 50 % and 100 %, both signs (2 runs each)**; full astern, rudder centred, as the reported layer's stop; logged to rest; side camera on a marked line | reverse efficiency, thrust delay, astern overshoot, head reach; heading change, yaw-rate decay and lateral drift from the stop command to rest |
| S1-D Turning circles (priority 1) | 90 min | Helm 25, 50, 75, 100 %, both signs, RPM 12, two full turns each from settled speed; **helm 50 and 100 %, both signs, at the cruise setting RPM 6**, two full turns each; helm 50 and 100 % at RPM 18 | steady turn rate, radius, speed loss, advance, transfer, tactical diameter; drift-moment asymmetry; the turn at the speed the trials run at |
| S1-E Zig-zags | 40 min | 20/20 (3 repeats) and 10/10 (2 repeats), both initial directions, RPM 12; **20/20 at RPM 6 (2 repeats), both initial directions** | overshoot angles, yaw inertia and damping, at RPM 12 and at the cruise setting |
| S1-F Mid-session check | 25 min | The committed fit script on the day's data; thresholds of section 7; anything failing is reshot | keeps session 2 free for validation |

If a full turning circle does not fit the basin at RPM 12, the largest settled arc is logged (a
settled 180 deg arc still yields steady rate and speed) and the constraint is recorded. The
current model puts the radius at about the same value at RPM 6 as at RPM 12 (5.2 m at full helm,
12 m at helm 50 %), with the turn rate halved: one full turn takes about 71 s at full helm and
143 s at helm 50 %, which sets the 20 min the RPM 6 circles add to S1-D.

## 5. Session 2: validation and dress rehearsal (about 4 h)

Session 2 must not become a second identification session: the model's accuracy claim needs data
it was not fitted on.

| Block | Time | Content | Yields |
| --- | --- | --- | --- |
| S2-A Repeatability | 45 min | Without refitting: turning circles at helm 50 and 100 %, one RPM sweep point, one 20/20 zig-zag | day-to-day parameter variation (the honest randomisation width) |
| S2-B Open-loop replay | 45 min | Fixed command sequences from session 1 and from the policy, no feedback, against the model's free-run prediction; three crash stops at RPM 12 as holdout | heading and position error against horizon (as `bluefin/REPORT.md` V1); holdout for the stop parameters |
| S2-C Closed-loop dress rehearsal | 80 min | The deployed policy through the Paper 3 bridge with the registered map frame, no target, on layouts L1, L2 and L3 and an open leg, 2 runs each; field-trial logging and the safety layer in the field-trial mode (shadow) | closed-loop check of the calibrated simulator (each run re-simulated from its logged start); **the phantom-track check (T-15)**: with no target on the water, any moving track is a phantom; rehearsal of the trial procedure, positioning, reset and abort |
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
| T-4 | steady turn rate at full helm | S1-D | measured at 3 or more helm settings, both signs, at RPM 12; and at helm 100 %, both signs, at RPM 6 |
| T-5 | turning radius | S1-D | measured; replaces the 1.5-4 L prior band |
| T-6 | zig-zag overshoot | S1-E | both manoeuvres, 2 or more valid repeats |
| T-7 | black-wall return rate | S1-A | resolved against bearing at 3 or more headings |
| T-8 | localisation drift | S1-A | stationary drift rate quantified |
| T-9 | reverse thrust efficiency | S1-C2 | resolved to +/-0.1, at 2 or more RPM settings |
| T-10 | thrust transport delay | S1-C2 | resolved to +/-0.1 s |
| T-11 | crash-stop head reach from RPM 12 | S1-C2 | measured; at most 1.5 Lpp (2.36 m), or recorded as failing |
| T-12 | reverse response against the detector | S1-C2 | efficiency and onset inside the weak-strong bracket (0.25-0.5, 0-0.75 s), or recorded as outside with its consequence for the layer |
| T-13 | drift after a stop from a turn | S1-C2 | heading change and lateral drift to rest measured at helm 50 and 100 %, both signs |
| T-14 | run time on the shore laptop | P-10 | command path p99 below 0.5 s per decision with communication (required); E1 in the loop p99 below 0.5 s (required only for the optional active block; not met on the development laptop, nor on the personal laptop, p99 0.89 s with 5 % of decisions above 0.5 s, 2026-11-14) |
| T-15 | phantom moving tracks | S2-C | on the target-free runs, the share of decisions with a moving track and the shadow layer's stop commitments reported; the trials start only if the registered map frame removes the wall phantoms, or the remaining rate is recorded with its effect on the target branch |

## 8. Contingency ordering

If session 1 runs short, drop in this order (last dropped is most valuable):
1. S1-C RPM sweep at the extremes (keep 9, 12, 18);
2. S1-E 10/10 zig-zag (keep 20/20);
3. S1-C2 straight stops at RPM 9 and 15 (keep RPM 12) and the helm-50 % stops from a turn (keep
   helm 100 %);
4. S1-B ramp (keep the steps);
5. S1-D RPM 18 circles (keep RPM 12, every helm setting, both signs), which lie above the
   policy's propulsion range;
6. S1-E 20/20 at RPM 6, then the S1-D helm-50 % circles at RPM 6 (keep helm 100 % at RPM 6,
   both signs).

**Never dropped:** S1-A, S1-D at RPM 12, S1-D at RPM 6 helm 100 % (both signs), S1-B steps,
S1-C2 at RPM 12 (straight and from a turn), S1-F. A block lost entirely is recovered in S2-D at the cost of S2-C, never by refitting on
session-2 validation data.

## 9. Open items

| Id | Question | Consequence |
| --- | --- | --- |
| O-1 | Can the vessel telemetry exceed 2 Hz, and by how much, and at what pose resolution? | decides whether S1-B separates the actuator terms, and how close the bridge's ego inputs come to the simulator's |
| O-2 | Is the IMU installed, raw at 100 Hz or more, on the bridge clock? Installed on the Bluefin (2026-11-23); its line format and rate are first seen in a data-collection session on 2026-11-26, recorded with `field_deployment/udp_raw_logger.py` (no decoder, so an unknown line type cannot stop the recording) | surge and yaw-rate identification; field ego estimate |
| O-3 | Does a full turning circle fit the basin at RPM 12? | tactical-diameter quantities available or not |
| O-4 | Rudder position feedback, or the P-3 camera method? | S1-B decisive or lumped |
| O-5 | Is the black wall still present and matte? | registration design |
| O-6 | Does the ESC reverse, with what dead band and ramp (P-8)? | **the layer's stop on the water** |
| O-7 | Is a direct full-ahead-to-full-astern reversal safe for motor, gearbox and battery? | if a ramp is needed, the layer's stop and the simulator's stop distances change |
| O-8 | Basin booking dates and lead time; risk assessment | schedule |
| O-9 | Does the localisation report the hull reference point or the LiDAR's position (0.86 m ahead)? | the bridge's `--pose-point` setting; a wrong choice shifts every range and boundary ray by up to 0.86 m |
| O-10 | Which laptop is the shore laptop, and does it have free UDP ports and a network that passes UDP (P-11)? | the live bridge cannot connect otherwise; P-10 is measured on it. Decided 2026-11-14: the personal laptop (Intel i7-6600U, 2 cores, 4 threads, 7.9 GB, Windows 10), linked to the laptop that controls the vessel as in July; parity exact, bridge port free, UDP not restricted, command path met, E1 in the loop not met (as on the development laptop, which the university network and its held ports exclude). The link is first tested by handover step 7 at the next basin booking; on the day, other applications closed |
