# V13 FIX05 saved prediction audit

The completed V13 episode still contacts the target after **39 decisions / 19.5 s**, with 11 interventions and five issued astern commands. The proposed explanation that fast braking turns an accepted plan unsafe is **not supported in this trace**. The strongest remaining measured defects concern target geometry and its transition to a stopped target.

Case: `TS2:P2-L1-CRP-FIX-05`, reset seed `470024`; run `consistent_track_probe8`, trace `003_v13.jsonl`. These results use only this completed saved episode. No environment was created, reset or stepped; zero new episodes were consumed. Truth is used only for scoring and explicitly labelled fixed-plan sensitivity checks.

## Brake response

All 33 retained or reconstructed selected plans reproduce their logged weak-model clearance within 2e-6 m. There are 28 passing plans. **None becomes failing under the existing nominal immediate brake model.** Stronger braking generally increases clearance here.

| Error scored against issued commands | Weak / delayed | Nominal immediate |
| --- | ---: | ---: |
| Five astern decisions: end-surge MAE | 0.2004 m/s | 0.0159 m/s |
| Five astern decisions: displacement MAE | 0.0579 m | 0.0192 m |
| From step 14, recorded commands, 2 s position error | 0.6877 m | 0.0844 m |
| From step 14, recorded commands, 4 s position error | 1.5323 m | 0.1305 m |

The nominal model is a much better measured response, but that is not evidence that enabling it rescues this episode. The weak model resets brake onset at each forecast; the two response scenarios are empirical models, not a complete uncertainty bound. Commands in the multi-step comparison are the actually recorded future commands, not simulated future SAC decisions.

## What changes before failure

1. **Step 29: source 21 appears alongside persisted source 4.** Source 21's centre error is 0.4485 m and velocity error 0.2149 m/s; source 4's corresponding errors are 0.0974 m and 0.0171 m/s. The same retained plan has source-21 clearance **-0.5516 m**, source-4 clearance **+0.0617 m**, and true-current-target CV clearance **+0.1671 m**. Source-ID continuity therefore matters: V13's exact-ID ownership cannot associate this newly numbered view with the already persisted target.
2. **Step 32 (t=15.5 s): the real target stops.** It remains at `[8.7041708, 8.1340296]` thereafter. The old source-4 CV hypothesis keeps moving, reaches 0.7207 m centre error by step 35, and expires at step 36. Thus treating the old moving hypothesis as universally correct would also be wrong. Source 21's later small measured velocity partly reflects a real motion change.
3. **Step 34: a missing hull orientation creates an isolated passing check.** Source 21 had fitted heading 114 degrees through step 33; its `last_fit_heading_deg` becomes `None` at step 34. `classical/common.py` substitutes the direction of measured velocity, **147.562 degrees**, while the real hull remains 115.784 degrees. This is a course/shape substitution, not a measured target turn. The next view heading is 118.968 degrees. Centre error at step 34 is still 0.3018 m.
4. **Steps 35?39: every selected action is `no escape` and SAC is returned unchanged.** The last passing plan from step 34 is not retained by the selected preset. However, retention alone is not supported as a rescue: its shifted tail already fails at step 35 under both models, with first predicted violation at 0.125 s.

| Step-34 fixed-plan sensitivity | Weak clearance | Fast clearance |
| --- | ---: | ---: |
| Recorded onboard target views | +0.1341 m | +0.1542 m |
| Only source-21 heading replaced by prior onboard heading 114 degrees | -0.2907 m | -0.2499 m |
| Only source-21 heading replaced by true heading | -0.2863 m | -0.2304 m |
| Complete true stationary target, for scoring only | +0.0071 m | +0.0871 m |
| Same plan shifted to step 35, current onboard views | -0.2192 m | -0.1405 m |

Heading and centre must be considered together. Fixing the heading while retaining the biased centre increases conservatism; it does not establish that a collision-free maneuver will be found. From steps 29?39 the one-step own-ship displacement errors are mostly 0.002?0.024 m (0.041 m in the final decision), considerably smaller than the target centre errors of roughly 0.24?0.45 m.

## Next hypothesis, without changing the controller

Add a narrowly scoped orientation observation rule: when a fresh shape fit loses its heading, retain the last finite fitted hull axis for the existing bounded coasting lifetime instead of instantly rotating the collision rectangle to a noisy velocity vector. Keep translation velocity separate from hull orientation. Pair this with the existing geometry-consistent centre projection, and explicitly handle motion changes/reacquisition; do not silently merge different raw IDs without association evidence.

First validate synthetically and on saved frames: valid-fit -> missing-fit -> valid-fit transitions, stationary targets with noisy velocity, genuine turns with renewed shape measurements, expiry, and exact-ID reuse. In this trace, the regression assertion is that disappearance of the step-34 shape measurement cannot alone produce the 33.6-degree hull rotation and +0.1341 m certificate. Re-score the complete plan and record any new infeasibility; do not claim this rule rescues FIX05. Missing orientation could alternatively be represented as a bounded set of orientations, but a union may remove every feasible plan and must be assessed separately.

This is an engineering hypothesis motivated by separating object extent from spatial measurements and kinematic state, as discussed in [Granstrom, Baum and Reuter, Extended Object Tracking](https://arxiv.org/abs/1604.00970). It is not their full estimator. Likewise, repeatedly finding a finite-horizon passing plan does not implement the uncertainty treatment and continuing safety policy of [Wabersich and Zeilinger's predictive safety filter](https://arxiv.org/abs/1812.05506); no formal or episode-level safety guarantee follows here.

## Reproduce

Run from the project root, with a **new** tag:

```powershell
python -B tools/diagnostics/safety/audit_v13_braking.py --run-dir results/safety_dev/v10_iterations/consistent_track_probe8 --trace 003_v13.jsonl --tag NEW_FIX05_AUDIT
```

Omit `--tag` for read-only output. Existing tags are rejected. `audit.json` contains one-step errors, every selected plan and both complete predictions, source-wise target errors, recorded-future target sensitivity, and shifted-tail checks. `geometry_transition.json` records the raw fitted-heading transition. `provenance.json` records hashes and validation. `reproducer_evaluated_bytes.py` preserves the evaluated script bytes; use the working tool path for execution.

The fixed-plan future-target sensitivity linearly interpolates the saved 0.5 s target positions and ends at the last available pre-state. It is not a new closed-loop rollout. SAT gaps use inflated rectangular hulls and existing filter margins; they are not the simulator's polygon contact test. Nothing here proves physical unavoidability or a rescue from any proposed change.
