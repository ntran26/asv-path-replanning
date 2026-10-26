# Safety layer V21: refining V16 and V16 with the boundary-and-targets stop on the deployed policy

Date: 2026-10-25. Status: development in progress.

Starting point: [complete safety record](SAFETY_LAYER_COMPLETE_SUMMARY.md) (V1-V19 and sections 16-20), [V20 plan](SAFETY_V20_PLAN.md) (section 16: V16 remains the reference), [emergency-stop protocol](SAFETY_ESTOP_PROTOCOL.md).

## 0. Decision in brief

Decision (2026-10-25): refine the best safety-layer design on the deployed policy, and refine and compare both candidate designs:

- **Design A: V16**, the predictive filter alone. On test set v4 (2026-10-10): 904 goals, 96 contacts, 11 lost SAC successes.
- **Design B: V16 with the permanent emergency stop**, triggered by the map boundary and target ships only, W = 2.5 s (`v16_estop_bt`, protocol section 8). On test set v4: 893 goals, 77 own-ship contacts, 8 target strikes on the stopped ship, 21 stops.

Development uses the validation set, with the 72 earlier development cases as regression checks. Test set v4 runs once at the end, on request only, with no tuning on its outcomes.

## 1. Fixed objects

| Object | Identity |
| --- | --- |
| Policy | SAC seed 0, best validation checkpoint at 3.0 M: `runs/sac_formulation_seed0_bl3/best_model.zip`, SHA-256 `993db156...c2bdc8`, identical to `kept_best_3M/best_model.zip` used throughout the safety record |
| Formulation | `configs/baseline_v3.json`, digest `03a9c0c22f53ee0b` |
| Development set | the 270 baseline-v3 validation episodes the policy was selected on: 120 decoupled (namespace DVD) and 150 coupled (DV3); `results/safety_dev/v21_development/selection_validation.json` from `tools/diagnostics/safety/prepare_validation_set.py` |
| Runner | `tools/diagnostics/safety/v21_iteration.py`: the V20 runner with the DVD namespace and the `formulation_v3` overlay installed for tags holding DVD scenes |
| Report | `tools/diagnostics/safety/v21_report.py` |

**Set reproduction.** The decoupled scenes are built with the `formulation_v3` overlay, which sets the stage-5 panel draws; without it the encounters are the same but 29 rather than 53 of the 120 scenes carry three panels. Of the 150 cached DV3 scenes, 111 differ from a fresh build only in `n_obstacles` on a fixed layout, a field the environment ignores when `fixed_obstacles` is set. With these settings the policy alone (OFF) reproduces the recorded selection evaluation at 3.0 M in all 270 episodes: outcome, steps and return identical.

## 2. Baselines on the validation set

Runs `val_off_check`, `val_v16_bt_c1`-`c6`; report `results/safety_dev/v21_development/reports/validation_base/`.

| Controller | Goals | Obstacle | Boundary | Target | Target strikes on the stopped ship | Stopped | Contacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 217 | 32 | 4 | 17 | - | - | 53 |
| V16 | 240 | 20 | 7 | 3 | - | - | 30 |
| V16 + stop (boundary and targets) | 236 | 20 | 6 | 1 | 1 | 6 | 28 |

- V16 against OFF: 28 rescued SAC failures, 5 lost SAC successes.
- V16 + stop against V16: no goal gained, 4 lost (2 false-alarm stops, 2 boundary contacts during the stop).
- Obstacle contacts are 20 of V16's 30; 16 of them are in coupled layouts. The stop's trigger leaves them alone by design (no static points).

## 3. Failure inventory: why V16 issues unchecked commands

**Contacts follow unchecked decisions.** 27 of V16's 30 contacts end after a run of unchecked decisions (`no escape`, where nothing passes and SAC's command is issued unchecked, or `last certificate`). The runs last up to 23 decisions (11.5 s). The three exceptions are two contacts within the first two decisions (DV3-BO-CV-10, DV3-BO-VS-05) and one contact after a checked decision with an optimistic forecast (DV3-CRP-CV-13). This repeats, on fresh data, the V20 audit of the 72 development cases (21 of 22).

**Probe.** `tools/diagnostics/safety/v21_unchecked_probe.py` replays V16 on every validation episode (outcomes identical to the runs in 270 of 270) and, at each unchecked decision, scores a fixed bank from V16's own snapshot and actuator history at three levels: V16's checker; the collision hull without the extra gaps and run-out; the uninflated hull. Output: `results/safety_dev/v21_development/probe_unchecked/`. Over 655 unchecked decisions in 102 episodes:

1. **Braking excluded without traffic.** In 116 unchecked decisions a plan passing V16's full checker exists once braking candidates and recoveries are admitted. V16 admits braking only with a target track within 7 m (V2 iteration 6). In 8 of the 28 failed episodes the final unchecked run starts at such a decision (DVD-NU-11, DV3-CRS-CV-19, DV3-NT-CV-06, DV3-NT-CV-04, DVD-HO-09, DV3-CRS-VS-01, DV3-HO-CV-13, DV3-HO-CV-10), including the lost SAC success DV3-CRS-VS-01 and both no-target failures. In every one of them all passing sequences brake.
2. **Buffers alone block every plan.** In 299 unchecked decisions some plan passes once the extra gaps (0.05 m static, 0.02 m boundary, 0.10 m target) and the 2 m terminal run-out are removed, against 116 with them; with the uninflated hull, 505.
3. **The basin's end walls.** The basin's navigable polygon is inset 0.4 m on all four sides (x 0.4-9.6 m, y 0.4-24.6 m). The environment scores a boundary contact against the polygon's sides but against the basin's ends at y = 0 and 25 m (`env._border_clearance`). Near either end V16 therefore sees its hull outside the boundary 0.4 m before a contact is scored, every plan fails at its first sample, and V16 issues SAC's command unchecked. In DVD-BO-16 this lasts 22 decisions, after which SAC reaches the east side. 35 first-sample boundary failures carry this mismatch. 254 of the 270 validation scenes are basin scenes; channel polygons span the full length.
4. **Static points with no obstacle near.** 81 first-sample static failures are against remembered points with no true obstacle within 0.3 m and no target within 1 m (23 in scenes without obstacles). These are most likely returns of the target remembered as static points along its earlier track (the V19 mechanism, wider than exact current returns).

**Emergency-stop detector geometry.** The detector (protocol section 2) counts contact with the uninflated own hull. The environment scores a collision with the hull inflated by `HULL_MARGIN` (0.15 m) against obstacles, the boundary limits and the targets' true hulls. The detector therefore predicts contact 0.15 m later than it is scored at the sides and against obstacles and targets, and 0.25 m earlier at the basin ends (the 0.4 m polygon inset less 0.15 m).

## 4. Candidate refinements

Each targets one mechanism of section 3, relaxes no check unless stated, and is a constructor option of `src/safety_v21.py` so that disabling it reproduces its base.

| Id | Change | Mechanism | Status |
| --- | --- | --- | --- |
| R1 | **Braking fallback.** At a decision V16 leaves unchecked (`no escape`; variant: also `last certificate`), score the primitive bank with braking candidates and recoveries admitted, from V16's own snapshot, actuator history, rollout and evaluation; commit the passing plan closest to SAC by V6's rule, as V6's `brake` branch does. | 1 | implemented; validation run 2026-10-25 |
| R2 | **Graded buffers.** If nothing passes even with braking, select at the largest buffer level at which a plan passes (run-out off, then gaps halved, then no extra gaps), instead of SAC unchecked. This relaxes the checker only where V16 currently checks nothing. | 2 | planned |
| R3 | **Scored boundary limits.** Check the hull against the limits the environment scores (polygon sides, basin ends at y = 0 and 25 m) rather than the 0.4 m-inset end edges. | 3 | planned |
| R4 | **Stale target returns.** Drop remembered static points that lie on a confirmed track's recent path. | 4 | planned; V19's exact-return transfer was too narrow |
| E1 | **Stop detector on the scored geometry.** The inflated hull and the scored boundary limits in the emergency-stop contact check. | detector geometry | planned (design B) |

## 5. Promotion rule (fixed before any V21 result)

A refinement of a base (V16 for design A, V16 + stop for design B) is promoted when, on the 270 validation episodes, paired by case, seed and digest:

1. goals do not fall and contacts (own-ship contacts and target strikes on the stopped ship) do not rise, and at least one of the two improves; and
2. on the 72 earlier development cases, goals do not fall and contacts do not rise.

Every run is reported, promoted or not. Refinements are combined only after each has passed alone. Test set v4 runs once, on request, for the final design A and design B.

## 6. R1 result, 2026-10-25

Runs `val_v21_r1_c1`-`c6`; report `results/safety_dev/v21_development/reports/validation_r1/`.

| Controller | Goals | Obstacle | Boundary | Target | Strikes | Stopped | Contacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V16 | 240 | 20 | 7 | 3 | - | - | 30 |
| V21 (R1 at `no escape`) | 240 | 16 | 10 | 4 | - | - | 30 |
| V21 (R1 also at `last certificate`) | 239 | 16 | 11 | 4 | - | - | 31 |
| V16 + stop | 236 | 20 | 6 | 1 | 1 | 6 | 28 |
| V21 + stop | 236 | 17 | 11 | 1 | 0 | 5 | 29 |

**Not promoted.** R1 changes no goal against V16 (0 gained, 0 lost) and moves contacts between types. In the eight episodes it targets, the braking plan brings the vessel to near rest beside the hazard (surge 0.01-0.15 m/s; DVD-NU-11 runs 126 decisions against 66). There V16 again finds no passing plan, now because the hull is already inside the checker's buffers, the fallback finds none either, and SAC's command from rest ends in contact. The braking fallback delays the contact rather than avoiding it: its terminal state lies in mechanism 2.

**Amendment (2026-10-25, before any R2 result).** Because R1's failures end in mechanism 2, R2 is tested both alone and combined with R1. This departs from the rule of section 5 that refinements are combined only after each passes alone; the combination is reported against the same promotion rule.

## 7. R2 result, 2026-10-25

Runs `val_v21_r2_c1`-`c6`; report `results/safety_dev/v21_development/reports/validation_r2/`.

| Controller | Goals | Obstacle | Boundary | Target | Strikes | Stopped | Contacts | Against its base (gained / lost) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V16 | 240 | 20 | 7 | 3 | - | - | 30 | - |
| R2 (graded buffers) | 238 | 18 | 10 | 4 | - | - | 32 | 1 / 3 |
| R1 + R2 | 240 | 16 | 10 | 4 | - | - | 30 | 1 / 1 |
| V16 + stop | 236 | 20 | 6 | 1 | 1 | 6 | 28 | - |
| R2 + stop | 233 | 20 | 9 | 1 | 1 | 6 | 31 | 1 / 4 |
| R1 + R2 + stop | 235 | 18 | 11 | 1 | 0 | 5 | 30 | 1 / 2 |

**Not promoted.** Neither the braking fallback nor the graded buffers, alone or together, improves on its base. Both act only where V16 issues unchecked commands, and in closed loop they redistribute contacts between types rather than prevent them. As in V20, the offline availability of a passing plan at those decisions does not predict a better episode.

## 8. R4 offline gate, 2026-10-25: failed

`tools/diagnostics/safety/v21_static_audit.py --trail` applies R4's rule offline to V16's own snapshots on all 270 episodes (`results/safety_dev/v21_development/static_audit_trail/`). Over V16's active decisions, the rule removes 87.6 % of the near-hull points lying on the target's true earlier track (2,029 of 2,315), and also 3.6 % of the near-hull points of real obstacles (730 of 20,093), in 5 episodes. In DV3-CRS-CV-17 (603 of them) the tracker carries a false moving track on a static panel: track 2 at the panel's position, estimated at 0.38 m/s, while the true target is 3 m away. R4 then deletes the panel. **R4 is not run in closed loop.** A version would need evidence that a track is a vessel before its footprint can clear memory.

**New mechanism 5: false moving tracks on static panels.** The snapshot leaves out static points within 2 m of any track (`classical.common.Perception`), so a panel tracked as moving is represented to the filter only as a target extrapolated at constant velocity, here moving away from where the panel stays. An audit of how often this happens and whether it precedes V16's obstacle contacts is running (`tools/diagnostics/safety/v21_track_audit.py`).

**Mechanism 5 audit result** (`results/safety_dev/v21_development/track_audit/`, all 270 episodes, replay outcomes identical to the runs). A moving track with no true target within 1.5 m and a true obstacle within 0.8 m appears in 6 episodes (5 coupled, 1 decoupled): 4 where both OFF and V16 reach the goal, 1 rescue, and 1 failure, DV3-CRP-CV-13. That failure is the one V16 contact preceded by checked decisions (section 3): its last 6 decisions carry a false moving track on the panel it then contacts, so the checker's forecast was optimistic because the panel was extrapolated as a moving target. The mechanism is real but rare on this set.

## 9. R3 result and parity, 2026-10-25

Runs `val_v21_r3_c1`-`c6`; report `results/safety_dev/v21_development/reports/validation_r3/`.

- **Parity.** V21 with every option off (`v21_off`) reproduces V16 in all 270 episodes: outcome, steps, return and intervention count identical.
- **R3 (scored boundary limits): neutral.** It changes V16's decisions in 43 episodes (steps or intervention count differ); in DVD-BO-16 the final unchecked run shrinks from 22 to 8 decisions. No outcome changes: 240 goals, 30 contacts, 0 gained, 0 lost. **Not promoted** under section 5 (no improvement), although it removes a modelling error at no measured cost.

**Reading so far (design A).** Three refinements that each remove a verified error in V16's decision logic or hazard model (braking excluded without traffic, buffers that block every plan, the basin's end edges) leave the validation outcomes unchanged, and the fourth (stale target returns) cannot be made safe without vessel evidence. V16's remaining validation failures are mostly real-obstacle contacts in coupled layouts after V16's search finds no passing plan; these local corrections change when and how V16 fails there, not whether.

## 10. E1 result (design B), 2026-10-25

Runs `val_v21_e1_c1`-`c6`; report `results/safety_dev/v21_development/reports/validation_e1/`.

| Controller | Goals | Obstacle | Boundary | Target | Strikes | Stopped | Contacts | Against V16 + stop (gained / lost) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V16 + stop | 236 | 20 | 6 | 1 | 1 | 6 | 28 | - |
| V21 + stop, all options off (parity) | 236 | 20 | 6 | 1 | 1 | 6 | 28 | 0 / 0, identical |
| E1 (detector on the collision hull) | 222 | 18 | 6 | 0 | 1 | 23 | 25 | 0 / 14 |
| R3 + E1 | 222 | 18 | 7 | 0 | 1 | 22 | 26 | 0 / 14 |

**Not promoted.** E1 turns 6 contacts into stops and 12 goals into false-alarm stops (2 more goals into contacts). With the scored hull, the nominal-model forecast predicts certain contact in many states the policy then leaves safely; the uninflated hull of the protocol's detector had been acting as an allowance for that forecast's pessimism. W = 2.5 s was chosen for the uninflated hull by the protocol's rule; a version of E1 would need W re-chosen by the same rule.

## 11. Status, 2026-10-25

| Design | Refinement | Validation result | Decision |
| --- | --- | --- | --- |
| A (V16) | R1 braking fallback | 240 goals, 30 contacts (= V16) | not promoted |
| A | R2 graded buffers, R1 + R2 | 238 / 32; 240 / 30 | not promoted |
| A | R3 scored boundary limits | 240 / 30, identical outcomes | not promoted (neutral) |
| A | R4 track-footprint exclusion | offline gate failed (removes real obstacle points) | not run |
| B (V16 + stop) | R1, R2, R1 + R2 with the stop | 236 / 29; 233 / 31; 235 / 30 (base 236 / 28) | not promoted |
| B | E1, R3 + E1 | 222 / 25; 222 / 26 | not promoted |

V16 and V16 with the boundary-and-targets stop remain the two reference designs. Their test set v4 results (2026-10-10; 904 and 893 goals) were run in the earlier runtime environment; the validation runs above are the first in this one.

## 12. Objective restated and the test set v4 run, 2026-10-25

**Decision (2026-10-25): the safety layer's purpose is to reduce collisions made by the own vessel.** It is not required to raise the success rate, nor to prevent a target ship from striking the own vessel. Own-vessel collisions are the obstacle, boundary and target contacts of the moving own ship; a target striking the stopped ship (`collision:by_target`), a stop and a timeout are not. `v21_report.py` now reports this count (`own_vessel`).

**Finalization (decision, 2026-10-25).** V16 and V16 with the boundary-and-targets stop run on all 1,000 test-set-v4 scenarios with a fresh OFF, in this runtime environment (tags `ts4_final_c01`-`c10`, modes `off`, `v16`, `v16_estop_bt`).

**E1 added under the restated objective, chosen on the validation set before any test-set outcome was read.** Own-vessel collisions on the 270 validation episodes (sections 2 and 10):

| Controller | Own-vessel collisions | Goals | Stopped | Strikes on the stopped ship |
| --- | ---: | ---: | ---: | ---: |
| OFF | 53 | 217 | - | - |
| V16 | 30 | 240 | - | - |
| V16 + stop | 27 | 236 | 6 | 1 |
| V16 + stop, E1 detector (`v21_e1_estop_bt`) | 24 | 222 | 23 | 1 |
| V16 + stop, R3 + E1 | 25 | 222 | 22 | 1 |

E1 has the fewest own-vessel collisions; its cost under section 5 (goals turned into stops) is not a cost under the restated objective. It runs on test set v4 as `ts4_final_e1_c01`-`c10` after the three modes above, with no setting changed after any test-set outcome.

## 13. Test set v4 result, 2026-10-25

Runs `testset_v4_main/runs/ts4_final_c01`-`c10` (OFF, V16, V16 + stop) and `ts4_final_e1_c01`-`c10` (E1); report `results/safety_dev/testset_v4_main/reports/ts4_final/`. All 1,000 scenarios, paired by test ID, seed and digest. OFF reproduces the deployed policy's recorded test-set run (`results/test_set/v4/sacs0_bl3_metrics`) in all 1,000 episodes: outcome and steps identical, returns equal to rounding.

| Controller | Goals | Own-vessel collisions (obstacle / boundary / target) | Strikes on the stopped ship | Stopped | Timeout | Changed decisions | Episodes with a change |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| OFF | 865 | 135 (70 / 12 / 53) | - | - | 0 | - | - |
| V16 | 903 | 97 (47 / 24 / 26) | - | - | 0 | 3.8 % | 34 % |
| V16 + stop | 892 | 78 (47 / 19 / 12) | 8 | 21 | 1 | 6.1 % | 36 % |
| V16 + stop, E1 detector | 868 | 70 (45 / 17 / 8) | 9 | 52 | 1 | 8.9 % | 36 % |

Own-vessel collisions, scenario by scenario: V16 removes 49 of OFF's and adds 11 (net -38); V16 + stop removes 64 and adds 7 (net -57); E1 removes 74 and adds 9 (net -65). Against V16 + stop, E1 removes 11 and adds 3. Decoupled / coupled own-vessel collisions: OFF 46 / 89, V16 32 / 65, V16 + stop 19 / 59, E1 18 / 52.

The test set reproduces the validation ordering under the restated objective (E1 < V16 + stop < V16 < OFF). V16 and V16 + stop agree with the earlier runtime's 2026-10-10 runs to within one episode (904 / 96 and 893 / 77). The changed-decision rates count every held decision after a stop.

## 14. E1 with the remembered static points in the trigger (pre-registered 2026-10-25)

Under the restated objective a false-alarm stop is not a cost, so the protocol's rule for W (most caught contacts with zero false alarms, section 4 of the protocol) no longer fits, and the remembered static points that were left out of the trigger to avoid false alarms (protocol section 8) can return. 44 of E1's 70 test-set own-vessel collisions are obstacle contacts with no stop, and 15 happen after a committed stop (drift into the boundary).

**Configurations** (V16 + stop, E1 detector, static points included): W = 1.5, 2.5 and 3.5 s (`v21_e1s15_estop`, `v21_e1s25_estop`, `v21_e1s35_estop`), on the 270 validation episodes.

**Selection rule, fixed before any result:** the configuration with the fewest own-vessel collisions, among those that reach at least the policy's own validation goals (217; a layer that completed fewer missions than the policy alone could win by stopping at once); ties go to fewer stops. It runs on test set v4 only if it has fewer own-vessel collisions than E1 on validation (24).

**Result (validation, runs `val_v21_e1s_c1`-`c6`; report `results/safety_dev/v21_development/reports/validation_e1s/`).**

| Configuration | Own-vessel collisions (obstacle / boundary / target) | Goals | Stopped | Strikes | Against E1: own-vessel removed / added |
| --- | --- | ---: | ---: | ---: | --- |
| OFF | 53 (32 / 4 / 17) | 217 | - | - | - |
| E1 (boundary and targets) | 24 (18 / 6 / 0) | 222 | 23 | 1 | - |
| E1 + static points, W = 1.5 s | 23 (17 / 6 / 0) | 213 | 30 | 4 | 2 / 1 |
| E1 + static points, W = 2.5 s | 20 (14 / 6 / 0) | 213 | 36 | 1 | 4 / 0 |
| E1 + static points, W = 3.5 s | 19 (13 / 6 / 0) | 212 | 38 | 1 | 5 / 0 |

The static points in the trigger lower own-vessel collisions below E1's, mainly obstacle contacts (18 to 13-14), with diminishing returns as the window grows; boundary contacts (6, mostly drift after a committed stop) do not change. **No configuration meets the pre-registered guard**: each completes fewer missions than the policy alone (212-213 against 217). Under the rule, none runs on test set v4; the guard itself is open to decision.

**Decision (2026-10-25): the guard is relaxed.** A layer may complete fewer missions than the policy alone if it makes fewer own-vessel collisions. The configuration with the fewest own-vessel collisions on validation, **E1 + static points at W = 3.5 s** (`v21_e1s35_estop`), runs on test set v4 as `ts4_final_e1s35_c01`-`c10`, chosen before any of its test-set outcomes were read.

**Test set v4 result** (runs `ts4_final_e1s35_c01`-`c10`; report `testset_v4_main/reports/ts4_final/`, all five modes paired):

| Controller | Goals | Own-vessel collisions (obstacle / boundary / target) | Strikes | Stopped | Timeout | Changed decisions |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| E1 | 868 | 70 (45 / 17 / 8) | 9 | 52 | 1 | 8.9 % |
| E1 + static points, W = 3.5 s | 820 | 67 (43 / 17 / 7) | 12 | 100 | 1 | 12.8 % |

- Against E1, scenario by scenario: 11 own-vessel collisions removed and 8 added (net -3). The added ones are goals turned into contacts (4 obstacle, 2 boundary) and stops followed by contact (2). Against OFF: 80 removed, 12 added (net -68).
- 46 of its 67 remaining own-vessel collisions happen after the stop was committed (30 obstacle, 16 boundary): the stop now fires near obstacles, but the vessel still reaches them while braking or drifting.
- The test set keeps the validation ordering (fewest own-vessel collisions), but the margin over E1 falls from 5 (validation) to 3, a paired difference of 11 against 8 that a sign test does not separate from zero, while the cost rises to 48 fewer goals than E1 and 100 stops (10 % of episodes).

## 15. Decision: the paper's safety layer, 2026-10-25

**E1 is the runtime safety layer the paper reports**: V16 with the permanent emergency stop triggered by the map boundary and target ships, its detector predicting the collision as the environment scores it (hull inflated by 0.15 m), W = 2.5 s (`v21_e1_estop_bt`: `safety_v21.SafetyFilterV21EC(estop_window_s=2.5, use_static_points=False, braking_fallback=False)`). The paper reports this layer alone: neither the W = 3.5 s configuration nor any earlier version is described. The layer is there to mitigate the collision rate; a higher success rate is welcome but not its purpose.

Figures for the paper (test set v4, 1,000 episodes, deployed policy; report `testset_v4_main/reports/ts4_final/`):

| | Layer off | Layer on (E1) |
| --- | ---: | ---: |
| Own-vessel collisions | 135 (13.5 %) | 70 (7.0 %) |
| obstacle / boundary / target | 70 / 12 / 53 | 45 / 17 / 8 |
| decoupled / coupled | 46 / 89 | 18 / 52 |
| Goals | 865 | 868 |
| Stopped | - | 52 |
| Target strikes on the stopped ship (targets brake after the stop, an evaluation convention) | - | 9 |
| Timeouts | 0 | 1 |
| Decisions where the layer changed the command | - | 8.9 % (all decisions, including the held stop) |
| Episodes with a change | - | 36 % |

Validation set (270): own-vessel collisions 53 -> 24, goals 217 -> 222, stops 23; changed decisions 13.5 % in all, 4.6 % outside the episodes that end in a stop.
