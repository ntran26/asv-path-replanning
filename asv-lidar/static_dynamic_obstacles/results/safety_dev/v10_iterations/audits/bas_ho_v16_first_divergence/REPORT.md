# BAS-HO-NC-059: first-intervention audit

Saved data only. No simulator episodes, policy calls, controller edits or new search were performed. Reproduce from the project directory with `python -B results/safety_dev/v10_iterations/audits/bas_ho_v16_first_divergence/reproduce.py`. The script checks checkpoint/configuration, case seed/scene digest and the model-source hashes against the recorded manifests, then reconstructs the recorded snapshot and actuator state. Detailed evidence and input SHA-256 values are in [audit.json](audit.json).

The first intervention is **decision 20**. OFF reaches the goal in 54 decisions; V16 collides with the target in 28 decisions after seven interventions. V15 and V16 have identical recorded commands and states throughout these 28 decisions, so the motion-axis adapter does not change this failure. At decision 20, OFF and V16 still have the same recorded pre-state `[x, y, heading, surge]` and SAC action. The older OFF trace does not record sway, yaw or future target states.

## A passing SAC backup already exists

V16 executes full astern with rudder -0.544810. Its checked stored continuation has minimum margin **+0.500991 m**. Replacing only its first command with SAC, while keeping exactly the same remaining tail, also passes all current checks: **+0.099951 m**. Both values reproduce from the saved snapshot to numerical precision.

The intervention therefore does not mean that every SAC-first continuation was rejected:

- V7 requires +0.15 m and rejects this +0.099951 m tail.
- V10 sees both plans passing and prefers the larger-margin braking continuation; `prefer_any_feasible_policy` is false.
- V15 sees the parent above +0.15 m and skips the optional prefix search with `adequate_parent_clearance`.

The existing broader-prefix episode (`persistent_prefix32`, attempt 001) reaches the identical decision-20 snapshot and makes the same first intervention. Its extra 192-plan search finds **122 hard-passing SAC-first plans**, with best margin **+0.137479 m**, but accepts none at +0.15 m. Thus merely enabling the existing broader search does not preserve this action. At this saved state, the existing `prefer_any_feasible_policy=True` option would preserve SAC using the already checked +0.099951 m tail, without another search. This is a local code-path consequence, not an episode-rescue result; previous broad feasibility-preference tradeoffs remain relevant.

The one-second committed-SAC search is a different bank: it finds no passing plan, with best -0.235386 m. The distinction between its fixed prefix and the half-second prefix matters.

## The known successful continuation is rejected by the target forecast

The actual next 16 OFF commands, decisions 20–35, provide a noncausal diagnostic continuation. Rolling them out from the V16 snapshot gives:

| Scoring variant | Minimum margin (m) |
| --- | ---: |
| Recorded onboard model | -0.793390 |
| Fresh onboard fitted target centre only | -0.749010 |
| True current target centre, velocity and heading, still constant velocity | -0.675899 |
| True current own state only | -0.894763 |
| Raw measured yaw only | -0.795646 |

The original failure is exclusively against the target: static margin +0.147592 m, boundary margin +1.866083 m. Replacing the own rollout with the actual saved OFF own-position/heading endpoints still yields a target margin of -0.830506 m against the onboard constant-velocity forecast. Better own-yaw prediction or centre correction alone therefore does not make this known successful continuation pass. Filtered yaw at decision 20 is already close to truth: 15.6405 versus 15.7075 degrees/s; raw yaw is 14.8199 degrees/s.

## Target representation and maneuver evidence

At decision 20 there is one target view, persistent synthetic ID -1000000 for raw source 1. It is freshly refreshed, age zero, and replaces the same-source base view. Its centre error is about 0.198 m, compared with about 0.033 m for the fresh raw fitted centre. The geometry prior retains a partial-beam-axis offset: observed length 1.69445 m, width 0.24412 m. V16 handles fresh positive-ID base views and does not revise this synthetic persistent view.

There is also clear causal evidence that constant velocity is a poor target model here: the onboard fitted headings at decisions 13–20 are **132, 128, 124, 120, 116, 112, 108, 104 degrees**, corresponding to -8 degrees/s. Recorded target truth follows that turn before decision 20. At the next recorded decision it abruptly changes from 104.0837 to 185.9683 degrees, a change of +81.8846 degrees over 0.5 seconds. Its position deviates from the decision-20 true-current-state constant-velocity forecast by 0.242 m after 0.5 seconds and 1.835 m after 2.5 seconds.

The persistent representation then retains heading 104 degrees at decisions 21 and 22, while its source has current measurements. Refresh is rejected as `oversized_cluster`; the raw fitted headings are themselves wrong (124 and 96 degrees). At decision 23, enough new observations produce a heading near 186 degrees, but the selected recovery plan no longer passes. Both forecast mismatch and delayed adaptation after a maneuver are visible. A direct switch to the raw fitted heading at the first rejected refresh would not reliably fix this example.

## What this supports, and what it does not

There are two separate development questions: whether a hard-passing SAC-first backup should sometimes be preferred to a larger-margin braking backup, and whether target maneuver evidence should change the prediction/adaptation model. The former has an existing option with measurable global tradeoffs. The latter is more specific than adding search samples: a constant-turn or maneuvering-target estimator can use the observed heading history, but cannot infer the later abrupt turn from scenario labels or hidden target behavior. The existing finite CV-plus-turn union would remain rejecting if it retains the failing CV hypothesis; merely adding hypotheses cannot remove this false rejection.

The OFF trace lacks future target truth, so this audit does **not** substitute V16's post-intervention target trajectory for OFF's counterfactual target trajectory. The recorded maneuver establishes a failure of the V16 constant-velocity forecast, not what an alternative control would have caused. Actual-own endpoint checks use 0.5-second samples. No local passing backup or offline sensitivity proves a closed-loop rescue, and this case alone does not justify a new outcome-separating threshold.
