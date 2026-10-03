# FIX05 V11 saved-trace prediction audit

At decision 19, a dead-zone-triggered hull-centre shift makes the persisted target hypothesis reject an otherwise feasible recorded backup. A second, biased view of the same source also restricts the planner. An onboard-only geometry-consistent centre projection plus exact-source ownership restores passing plans in the saved state without changing any margin. This is a prediction sensitivity result, **not a demonstrated closed-loop rescue**.

Captured 2026-10-03 08:46:52 UTC. Case **TS2:P2-L1-CRP-FIX-05**, reset seed **470024**, scenario SHA-256 `b1aae753c8a63ff87e5a751905a9f5187c701301f6d9849b7ffc537a8cd58078`. The source is `persistence_three/traces/001_v11.jsonl`: 29 decisions, eight interventions, target contact after 14.5 simulated seconds. The reported 24.42 seconds in the episode result is wall-clock runtime.

No new episodes, environment construction/reset/step, policy calls, or evaluated-source edits occurred. Current relevant source bytes match the evaluated manifest; hashes, flags and artifact hashes are recorded in `provenance.json`. V11 SHA-256: `c15435494f7df9528bf370535ba2143f6a579d45f03532614d3640deb8d624ae`. Track admission/persistence were enabled with an 8-second coast; extra policy-prefix search, calibrated braking, dual braking and target-turn hypotheses were disabled. V10 feasible-policy and margin preferences were enabled; its any-feasible preference was disabled.

## Recorded sequence and lost backup

The first intervention is decision 13, starting at t=6.0 s. Decisions 14–16 issue full astern. Decision 18, t=8.5 s, is the last currently passing selected plan: minimum clearance **+0.012264 m**, limited by base track 4. Its persistent counterpart alone permits +0.201072 m; scoring the same own-ship forecast against the recorded true target gives +0.203855 m.

Decisions 19–22 execute the four bounded `last certificate` continuations despite failed current checks. Decisions 23–29 report `no escape`, discard the backup and issue SAC's actions. At decision 23, the boat has nearly stopped; its subsequent commanded propulsion reaches 11.45 and 11.27 RPM. The eventual contact therefore follows a loss of the backup, rather than a last-step claim of positive clearance.

The target's recorded velocity and heading are constant throughout. By decision 29, the filter has three hypotheses for this one ship: base ID 4, base ID 7 and persistent source 4. Their position errors are respectively **1.257, 0.496 and 0.335 m**. Treating every hypothesis as an independent simultaneous vessel restricts feasibility, but simply deleting duplicates is insufficient at decision 19.

## Identified centre-refresh defect

Persistent source 4 was independently admitted at decision 16. At decision 18 its centre error is 0.0198 m and its 8-second constant-velocity position error is 0.057 m. At decision 19 an accepted refresh changes its centre to `[6.171763, 9.075636]`, while truth is `[6.287085, 9.301650]`: **0.253735 m error**. Its velocity remains accurate to 0.0171 m/s.

The fitted cluster spans 1.720733 m longitudinally but only 0.242592 m laterally; the nominal hull dimensions are 1.725 by 0.5 m. `tracking.hull_fit_centre` completes an incompletely observed dimension. At decision 18, the nearest return is 1.17765 m and neither side is flagged as clipped. The lateral completion lies −0.13487 m from the observed midpoint, away from the sensor. At decision 19, the nearest return is 1.04623 m, inside the existing 1.1 m dead-zone detection threshold. The `hi_clipped` branch instead completes from the opposite endpoint, placing the centre +0.12870 m from the midpoint. This causes a side-to-side jump. The persistence admission rule checks longitudinal extent and coherent motion, but does not reject this inconsistent lateral centre refresh.

## Proposed update scored without new thresholds

For observed projected coordinates `[lo, hi]` and known hull size `s`, every enclosing centre lies in `[hi − s/2, lo + s/2]`. On a partially observed axis, project the prior anchor's CV-predicted centre onto this interval. On an axis already full under the existing extent tolerance, retain the observed midpoint. Keep the current heading, measured velocity and admission gates.

At decision 19 this yields `[6.281293, 9.305073]`: centre error falls from **0.253735 to 0.006728 m**; innovation from the prior falls from **0.255047 to 0.022523 m**. No truth enters this update. Truth is used afterward to score it.

The table uses the same 120 original primitive plans and own-ship forecasts at decision 19, with all hard checks and the existing 0.15 m acceptance margin unchanged. Stored continuations are padded to the complete horizon using V7's existing convention.

| Target representation | Hard-passing bank plans | Plans passing 0.15 m | Stored backup clearance |
|---|---:|---:|---:|
| Recorded V11 union | 0 | 0 | −0.125067 m |
| Remove duplicate base ID 4 only | 0 | 0 | −0.125067 m |
| Project anchor centre, retain base ID 4 | 9 | 0 | −0.003673 m |
| Project anchor and replace its duplicate base ID 4 | 27 | 9 | +0.126649 m |
| True target, scoring only | 45 | 9 | +0.207412 m |

At decision 23, the combined proposed representation gives **59 hard-passing / 50 margin-passing** bank plans, against recorded V11's 0/0; true-target scoring gives 59/53. At decision 29, every scored representation has zero hard-passing plans in this bank. This supports correcting the earlier estimate; it does not establish physical unavoidability at the final state or justify a larger search budget.

The ownership comparison is specific to source 4, which was first admitted independently (`base_admitted=False`). It does not assume every future persistent view should replace a reliable live track. Cross-ID association with source 7 and eventual coast uncertainty remain separate limitations.

## Own-ship forecast error

The actual issued rudder and **signed RPM**, the recorded onboard snapshot and the recorded pre-command actuator history feed the identified safety model. Truth scores the endpoint and displacement afterward. Prediction timestep is 0.125 s; the recorded plant timestep is 0.1 s. Thus these residuals include state-estimation, identified-model and integration effects; they do not isolate plant parameters.

| One-decision residual | All 29 decisions | 25 non-astern decisions | Four actual astern decisions |
|---|---:|---:|---:|
| Position error, mean | 0.0417 m | 0.0367 m | 0.0732 m |
| Displacement error, mean | 0.0223 m | 0.0163 m | 0.0600 m |
| Heading error, mean absolute | 0.755° | 0.745° | 0.817° |
| Surge endpoint error, mean absolute | 0.0487 m/s | 0.0214 m/s | 0.2192 m/s |

There are six brake requests but only four −24 RPM dispatches; decisions 21–22 dispatch **0 RPM** after stopping. Replacing only the weak/delayed braking response with the existing observer's immediate nominal reverse response reduces astern surge error to 0.0199 m/s. This is diagnostic evidence, not a new brake calibration or permission to assume the real reverse law.

The delayed safety rollout restarts its brake timer and predicts no reverse deceleration within the first 0.5 s. Longer forecasts conditioned on the actual saved commands accumulate substantial error: from decision 14, 2-second position error is 0.688 m and 4-second error 1.588 m. Braking-model mismatch is material, but the stored-plan target-only sensitivity shows that target representation already explains the loss of feasibility in this case. A parameter-only change would not repair the observed centre jump and duplicate constraints.

The reported `truth_endpoint_target_gap_m` is the **same rectangular SAT surrogate evaluated at true poses**, not exact polygon collision distance. Its negative value one step before the simulator contact must not be interpreted as an earlier physical collision. The final target endpoint is a labelled 0.5-second CV extrapolation; all preceding target endpoints use saved truth states.

## Recommendation and limits

Test geometry-consistent centre refresh together with exact-source ownership for independently admitted anchors. The saved residual identifies a specific measurement-model inconsistency; widening clearance tolerances would leave it intact. Then test uncertainty-aware braking separately, using broader development residuals and calibration validation rather than this single outcome.

The geometric projection is derived here. Its methodological basis is modelling observed surface returns separately from the full object extent, with temporal prediction/update; see Granström, Baum and Reuter, *Extended Object Tracking: Introduction, Overview and Applications*, [Sec. II-C and VI-C](https://arxiv.org/html/1604.00970). This adaptation is not their complete Bayesian estimator or a safety proof. Passing finite model checks also lacks the terminal safe-set and uncertainty conditions required by [Wabersich and Zeilinger, Sec. 4.1–4.2](https://arxiv.org/html/1812.05506v4#S4.SS1).

`audit.py` reproduces onboard one-step forecasts, per-track errors and recorded-command-conditioned drift. `projection_audit.py` reproduces the projection and representation comparisons. Both run from the project root with `python -B`; they refuse to overwrite existing JSON outputs. `audit.json` includes unextended stored-tail checks as labelled; `projection.json` contains the authoritative full-horizon padded checks, which reproduce the logged decision 18–22 clearances to numerical precision. No projected estimate has been run in a new episode by this audit.

Independent V13 implementation review is recorded in `v13_peer_review.json`. The pure centre helper matches the independently derived decision-19 projection within 1e-12. Replaying the 29 saved onboard frames reproduces every original V11 target-view set, and V13 with both new switches disabled gives identical track values and ordering on every frame. With both switches enabled, earlier geometry updates change the decision-19 prior slightly; its resulting centre `[6.274764, 9.291685]` remains about 0.016 m from truth. This replay reads saved sensor frames and does not advance an environment or establish an episode outcome.
