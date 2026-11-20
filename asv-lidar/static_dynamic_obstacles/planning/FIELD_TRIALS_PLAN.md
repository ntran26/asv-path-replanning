# Field trials (Paper 3, Section 5): design

Date: 2026-11-04 (decisions and the bridge's bench results 2026-11-05). Status: plan; the target
platform (D-1) is still open; the Paper 3 bridge is built (section 3). Decision S13(a) (2026-10-14) puts
the field trials in this paper; this file is the design the `[PENDING — field-trial design]`
markers of the draft wait for. The vessel-model calibration that precedes it is
`FIELD_CALIBRATION_PLAN.md`.

## 1. Purpose and what Section 5 can claim

- **Contribution C5:** the deployed policy runs on the model-scale vessel in the basin, its
  observation built from the real LiDAR scan and pose by the same perception modules as in
  simulation, in encounters with a moving target. The transfer evidence is reported apart from the
  numerical results.
- **The independent evaluation of the deployed policy.** The deployed learner was chosen on test
  set v4, so its test-set figures include that choice; the field trials are the evaluation it was
  not chosen on (the draft's limitations list says so).
- **What can be claimed:** behaviour per encounter type and the size of the sim-to-field gap,
  under the stated conditions (one basin, calm water, constant-velocity targets). With tens of runs
  the field success counts are reported as n/N per scenario, not as a rate comparable with the
  1,000-episode test set.

## 2. Fixed configuration

| Item | Setting |
| --- | --- |
| Policy | `runs/sac_formulation_seed0_bl3/best_model.zip` (SHA-256 `993db156...c2bdc8`), unchanged |
| Observation | the baseline-v3 70-value observation: 27 pooled LiDAR sectors, 7 map-boundary rays, ego motion, path terms, the tracked-target branch and the encounter-context branch, built by the simulator's own modules |
| Decision rate | 2 Hz; rudder command limiter as in training (off, `RUDDER_COMMAND_LIMIT = False`) |
| Speed | cruise 0.558 m/s at RPM 6; propulsion range 0-12 (no astern from the policy) |
| Safety layer | E1 (V16 with the permanent stop, boundary-and-targets trigger, collision-hull detector, W = 2.5 s), in shadow mode (D-4), in a separate worker process off the command path |
| Operator stop | always armed on the bridge, independent of the layer; plus a stop independent of the shore laptop (`FIELD_CALIBRATION_PLAN.md` P-12) |
| Map | the basin's navigable polygon as in simulation, in the map frame registered on the landmark survey (`--origin map`, P-7), at the estimated pose |

## 3. Software before any trial: the Paper 3 bridge

The existing bridge (`field_deployment/udp_live_rl.py`, last changed 2026-09-14) builds the Paper 2
observation and carries only the original stop latch. The deployed policy cannot run on the
vessel until a Paper 3 bridge exists:

1. **Observation adapter.** The live LiDAR scan and pose pass through the simulator's perception:
   clustering, the target tracker, encounter classification and latch, the map-boundary rays and
   the pooled sectors. One implementation, imported, not copied.
2. **Safety-layer adapter.** E1 reads an environment interface (estimated pose, measured ego
   motion, tracks and contexts, the boundary polygon, the issued-command callback); the adapter
   provides it from live data and routes the stop to full astern with the rudder centred.
3. **Parity tests, on the bench:**
   - *Observation parity:* a simulated episode's sensor stream replayed through the bridge gives the
     environment's observation exactly, decision by decision.
   - *Action parity:* the same replay gives the same policy action and the same safety-layer
     decision as the simulator.
   - *Shadow run:* the bridge against `fake_vessel_replay.py` on the July logs.
4. **Run time** on the shore laptop (`FIELD_CALIBRATION_PLAN.md` P-10).
5. **Logging** (section 5): everything the sim-to-field comparison and the future-work field-data
   study need, on one clock.

**Built 2026-11-05.** `field_deployment/udp_live_p3.py` (live) on `field_deployment/p3_bridge.py`
(core); record and bench results `field_deployment/README_P3_BRIDGE.md`, summarised in
`FIELD_CALIBRATION_PLAN.md` section 3a. The bridge subclasses the simulator's environment and
replaces only its sensor models with the live data, so the observation is the simulator's own
code (item 1). Items 2 and 3 pass: observation, action, issued command and every E1 decision
exactly equal to the simulator on 32 runs (1,497 decisions); all 19 July logs replay in shadow
mode. Item 5 is in place (per run: `decisions.jsonl`, `shadow.jsonl`, `telemetry.log`,
`summary.json`).

**Before the trials** (from the bench results):
- **Phantom moving tracks.** In the target-free July logs a moving track is present on 42 % of
  decisions, mostly at the basin wall, and it reaches the policy's target branch. In the trials
  a phantom would compete with the real target in the tracker and the encounter classifier.
  The registered map frame (P-7) is the expected fix; the dress rehearsal checks it (T-15), and
  the family A runs report the remaining rate (section 6, Perception).
- **Active mode.** E1 in the loop takes p99 0.89-2.06 s per decision on the development laptop,
  above the 0.5 s interval. Shadow mode (D-4) is unaffected; the optional active block needs the
  layer to meet T-14 on the shore laptop, besides the ESC reversal (P-8).
- **Pose reference point** (hull or LiDAR) settled at S1-A; **telemetry resolution** raised
  (P-1); **UDP ports and network** on the shore laptop (P-11); the UDP test repeated on the
  field device.

## 4. Scenario set

**Target behaviour: constant velocity only.** Reactive, noncompliant and speed-changing targets are
not reproducible on the water with the control available; this is stated as a limitation. In
simulation the deployed policy reaches 80 % on the basin layouts with fixed-speed targets (44 of
55) and 52 % with speed-changing ones.

**Families:**

| Family | Content | Role |
| --- | --- | --- |
| A. No target | path following on layouts L1, L2, L3 and an open leg | link to Paper 2; perception and path-keeping baseline; also run in calibration S2-C |
| B. Open-basin encounters | head-on, crossing from starboard, crossing from port, overtaking, being overtaken, with no panel near the encounter | the cleanest evidence of encounter-consistent behaviour (the snapshot figures of the field) |
| C. Coupled encounters | layouts L1-L3 with a target, the test set's L-cells | the hardest family, with simulated twins in test set v4 |

The layouts are Paper 2's basin layouts (`src/paper2_set.py`; 1 m panels): L1 (5.0, 8.0),
(7.2, 15.7), (2.0, 16.5); L2 (2.0, 8.5), (7.2, 9.3), (6.0, 17.0); L3 (1.5, 8.5), (6.5, 8.5),
(5.0, 17.0).

**Deployed policy in simulation on the same layouts** (test set v4 L-cells, successes / episodes,
both target variants):

| Encounter | L1 | L2 | L3 |
| --- | --- | --- | --- |
| Head-on | 10/10 | 0/5 | 4/5 |
| Crossing from starboard | 7/12 | 2/6 | 4/5 |
| Crossing from port | 2/11 | 3/5 | 4/6 |
| Overtaking | 6/10 | 5/5 | 3/5 |
| Being overtaken | 10/10 | 5/5 | 5/5 |

**Selection rule: by coverage, never by simulated success.** Every encounter type appears; the
simulated result of the same cell is reported beside each field result, so a cell that fails in
simulation (L2 head-on, 0/5) is expected to fail on the water too, and says something either way.

**Repeats:** 3 runs per scenario (2 at minimum), each from the same marked start, the variation
reported as a range.

**Core set** (about 40 runs; priority tiers for the contingency of section 9):

| Tier | Runs |
| --- | --- |
| 1 | A: L1, L2, L3, 1 run each (3); B: head-on, crossing from starboard, crossing from port, overtaking, 3 each (12); C on L1: head-on, crossing from starboard, crossing from port, 3 each (9) |
| 2 | B: being overtaken, 3 (3); C on L2 and L3: head-on and crossing from starboard, 2 each (8) |
| 3 | C: overtaking on L1 and L2, 2 each (4); A repeats (3) |

Being overtaken needs a target faster than the own ship's 0.56 m/s, and overtaking a target slower
than it (decision D-1).

**Field test suite (2026-11-20).** Fifteen named scenarios, `easy_1`-`easy_5` (open basin,
family B), `medium_1`-`medium_5` (L1) and `hard_1`-`hard_5` (L2 or L3), numbered by encounter
(1 head-on, 2 crossing from starboard, 3 crossing from port, 4 overtaking, 5 being overtaken):
`src/field_test_cases.py`, built by `tools/tiers/field_suite.py`, record
`results/field_suite/README.md`. Each reaches the goal in simulation on 10 of 10 noise seeds on
the identified plant and in 8-10 of 10 runs with a randomised plant and a perturbed start and
target launch; 12 of 15 meet the Section 3.4.3 rule conditions on the nominal run. At 3 repeats
the suite is 45 runs. The families, repeats and tiers above apply unchanged. The bridge loads a
case by name (`udp_live_p3.py --scenario easy_1`).

## 5. Measurement

- **Own ship:** scan-to-map pose anchored on surveyed landmarks, with the IMU
  (`FIELD_CALIBRATION_PLAN.md` P-5, P-7).
- **Target ground truth** (decision D-2): target metrics (range, DCPA, passing side) need the
  target's true track, independent of the own ship's tracker, which is itself under test.
- **Logged per run, on one clock:** raw LiDAR ranges, pose and IMU, the observation vector,
  the policy action, the issued command after the layer and the bridge, the tracker's tracks and
  the encounter context, every safety-layer decision (branch, predicted clearances, stop
  detector), the target ground truth, start and goal, layout, run identifier, operator
  interventions, and video.

## 6. Metrics

The simulation definitions (`src/metrics.py`, the paper's per-axis tables), so field and
simulation are compared like for like; no weighted scalar.

| Axis | Metrics |
| --- | --- |
| Outcome | goal; contact by type (panel, boundary, target); safety-layer stop; operator stop; timeout |
| Clearance | minimum distance to panels and boundary; minimum target range; DCPA at the closest approach |
| Path keeping | RMS cross-track error; path-length ratio; completion time |
| Motion | mean and minimum speed; rudder rate and reversals |
| Encounter behaviour | encounter class and engagement time as perceived; first alteration (direction, size, time after engagement); passing side; speed reduction; shown as trajectory snapshots, with no per-rule compliance table (as in Section 4) |
| Safety layer | decisions where it changed (or, in shadow mode, would have changed) the command; stops; reported apart from the policy |
| Perception | track confirmation delay; target position and heading error of the track against ground truth; encounter-classification agreement with truth; phantom moving tracks (share of decisions with a moving track that is not the target, from family A and from the decisions before the target enters the scan) |

## 7. Sim-to-field comparison

Three levels, the first two per run:

1. **Model level (open loop).** The field command sequences replayed through the training model
   and through the calibrated model; heading and position error against horizon, as in the model
   report.
2. **Closed-loop twin.** Each field run re-simulated: the same layout and start pose (from the
   log), the target scripted along its measured track, the same frozen policy, in the training
   simulator and in the calibrated model, several times with sensor-noise seeds. Reported:
   - outcome agreement (field against twin, a 2 x 2 table);
   - the field value of each metric against the twin's distribution (difference and percentile);
   - trajectory overlays.
3. **Population.** Field successes per family and cell against the test set's same cells, with the
   small-sample caveat.

**Attribution.** If the calibrated twin reproduces the field better than the training simulator, the
gap is in the dynamics; if not, it is in perception, localisation or disturbance. Either outcome is
reportable.

## 8. Procedure and safety

- Pre-registration before the first trial session: the scenario list and order, the metric
  definitions, the analysis scripts committed, the empty result tables.
- Start: the own ship held at the marked start, heading set, policy engaged at the start signal;
  the target launched from its own mark so that the encounter geometry (time and place of the
  closest approach) matches the scenario.
- Abort: the operator stop on the bridge, and a person at the target's control. Abort criteria
  are written down before the session.
- Run sheet per run: scenario, repeat, start time, any deviation (target late, start misplaced),
  validity flag. Invalid runs are kept and reported as invalid, not dropped.
- Risk assessment and basin booking as for the calibration sessions.

## 9. Sessions

Two trial sessions of about 4 h after the calibration sessions (decision D-5), at roughly 6-8
min per run including reset:
- **Session T1:** tier 1 (24 runs).
- **Session T2:** tier 2, then tier 3 as time allows, and any safety-layer-active block (D-4).
- **Contingency:** tier 3 is dropped first, then tier 2's L3 runs. Family B's four core encounters
  and family A are never dropped.

## 10. Decisions

Decided 2026-11-05 (D-1, D-2, D-4, D-5); D-3 and D-6 follow the recommendation unless changed.

| Id | Decision | Status |
| --- | --- | --- |
| D-1 | **Target vessel** | **Open: the platform is not fixed**, and it may be smaller than the Bluefin. If its hull sits too low for the LiDAR's scan plane, panels are attached around it to make it detectable. Consequence to check on the bench: the tracker's admission gates and the safety layer's partial-hull completion assume a target of the own ship's size (1.73 x 0.50 m), so the detected footprint of the final target (with any panels) is measured and compared with those assumptions before the trials |
| D-2 | **Target ground truth** | **Provided, after the two calibration sessions**; the method is fixed then. Target metrics (range, DCPA, passing side, tracker error) wait on it |
| D-3 | Speed-changing targets | none; stated as a limitation |
| D-4 | **Safety layer mode** | **Shadow mode**: the layer computes and logs every decision without acting; the operator stop is the physical safety. Learned behaviour is then reported as the policy's own, as in Section 4, and the layer's would-be interventions are measured on the water. An active block on the hardest scenarios is optional, and possible only if the layer meets T-14 on the shore laptop (p99 0.89-2.06 s on the development laptop, 2026-11-05) and the ESC reverses (P-8) |
| D-5 | **Trial sessions** | **Two after calibration**, more if needed; the plan of section 9 is sized for two |
| D-6 | Static-obstacle material and contact tolerance | as Paper 2 |
