# BO04: V15 fixes the first rejection, then encounters target-estimate error

Completed case `DV3:DV3-BO-CV-04`, seed 900247. This is saved-data analysis only; no environment, policy, controller decision, search, or episode was run. The surrounding V15 campaign may still be running.

V15 correctly preserves SAC at decision 11. Its parent proposes a currently failing last certificate (-0.421260 m); the conditional search requests minimum clearance zero and finds +0.139726 m. Earlier V14 plus prefix search found that same maximum but rejected it under the +0.15 m trigger requirement. V15 therefore matches the successful SAC-only commands and recorded trajectory through decision 11.

The new first command divergence is **decision 12**, again at the exact same recorded pre-state and SAC action as OFF. The shifted V15 certificate now fails (-0.113624 m); all 192 extra prefix candidates fail, best -0.057712 m. V15 issues the last certificate and ultimately contacts the target at decision 38. V14 default first diverged at 11 and contacted the boundary at 120; V14 plus prefix first diverged at 11 and contacted the target at 46. SAC-only reaches the goal in 31 decisions.

## The known successful continuation is rejected by the target representation

The actually issued OFF commands for decisions 12-27 were scored using V15's captured decision-12 snapshot and pre-command actuator history. These future commands are noncausal diagnostic inputs, unavailable online.

| Same 16-command sequence, current model | Minimum clearance |
| --- | ---: |
| Recorded onboard inputs | -0.287004 m, target-limited |
| Raw yaw substituted only | -0.287531 m |
| True current own pose/u/v/yaw, scoring only | -0.323873 m |
| True current target CV state, scoring only | +0.188670 m |
| Existing onboard fitted target centre only | +0.143553 m |
| Frozen V16 adapter after sequential saved-frame replay | +0.093623 m |

With the recorded inputs, target margin first fails at 2.125 s. Static-memory margin is +0.188670 m and boundary margin +1.429243 m. The target estimate also rejects the **actual saved OFF own-ship positions** at decision endpoints, with -0.337460 m minimum target margin. Thus own rollout accuracy alone cannot remove this erroneous rejection. Own-model endpoint error at eight seconds is 0.075411 m and 1.028401 degrees.

Source 16 is a fresh, confirmed dynamic track, but its centroid-based centre is 0.651121 m from the current true centre. Its fitted hull heading is 18 degrees versus true 350.953624 degrees, a 27.046376-degree error. Velocity error is 0.063164 m/s. No persistent anchor exists: the full-length admission rule rejects its partial observed extent. Correcting heading alone does not suffice; the biased centre remains decisive.

The target recorded in the V15 trace follows the decision-12 current CV state to within 3e-14 m for the next eight seconds. The OFF trace does not capture future target states; true-target replacements are still labelled sensitivity checks rather than unrecorded counterfactual measurements. Endpoint scoring samples every 0.5 s and does not prove continuous hull clearance.

## Separate frozen V16 qualification check

`MotionAxisPerception` was called once on each saved onboard frame 1-12, wrapping recorded V15 base snapshots. It received no truth and did not backfill earlier track history. Its source SHA256 is `e2a437e726d1f49a787cceb9e1686b11ebfbde9499f6cd6ca5141011cb5569b5`.

Source 16 first exists in the base snapshot at decision 10. The adapter collects frames 10, 11 and 12 and qualifies it at 12: current cluster has 18 points, longitudinal extent 0.948932 m, beam extent 0.416754 m, and 18 motion-evidence violations. Measured course 352.596692 degrees is 1.643068 degrees from the true hull axis. The accepted historical fit residual is 0.016773 m; residual against measured velocity is 0.143687 m.

The corrected centre `[4.969472, 9.589679]` has about 0.0492 m current error, and the known successful continuation becomes hard-passing at +0.093623 m under unchanged checks. This establishes qualification and fixed-sequence feasibility on saved inputs. It does **not** show that online search finds that sequence, that later estimates remain valid, or that V16 rescues the episode. No gate or controller source was changed for this audit.

`audit.json` records paired provenance, source hashes, exact command comparisons, separate geometry sensitivities, observed cluster extents, V16 adapter diagnostics, and prediction errors. `reproduce.py` performs only this saved-data calculation.
