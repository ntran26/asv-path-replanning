# CH-HO-CV-073 saved-command audit

SAC reached the goal in 54 decisions; V14 collided with the target in 28.

The first command difference is decision 13. Earlier commands and recorded states match exactly, as do the current recorded state and SAC action. SAC issued rudder 0.786289, RPM 10.934205; V14 issued rudder -1.000000, RPM -24.000000.

| Saved sequence | Hard passing | Minimum clearance (m) | First violation (s) |
|---|---:|---:|---:|
| stored_v14_proposal | True | 0.034454 | none |
| policy_first_same_v14_tail | False | -0.801864 | 5.125 |
| saved_successful_sac_sequence | False | -0.360848 | 3.875 |

| Sequence | Component | Horizon minimum (m) | At (s) | Terminal minimum (m) |
|---|---|---:|---:|---:|
| stored_v14_proposal | static | inf | outside point broadphase | not checked |
| stored_v14_proposal | boundary | 2.307859 | 1.625 | not checked |
| stored_v14_proposal | target_67 | 0.034454 | 8.0 | not checked |
| policy_first_same_v14_tail | static | inf | outside point broadphase | not checked |
| policy_first_same_v14_tail | boundary | 2.103978 | 2.375 | not checked |
| policy_first_same_v14_tail | target_67 | -0.801864 | 8.0 | not checked |
| saved_successful_sac_sequence | static | inf | outside point broadphase | 0.28346603190870556 |
| saved_successful_sac_sequence | boundary | 1.354555 | 7.875 | not checked |
| saved_successful_sac_sequence | target_67 | -0.360848 | 4.625 | not checked |

The current onboard target centre error is 0.723m, velocity error 0.041m/s, and hull-axis error -29.14°. No persistent hypothesis was published at this decision.

Both logged proposal and same-tail policy clearances reproduce to 1e-12. Every reported prediction component is checked against the unchanged evaluator.

The known successful future SAC commands fail the captured target check, even when actual SAC endpoints replace predicted own motion. A larger maneuver search alone cannot make this same trajectory pass this target representation. This localizes the discrepancy to target representation/forecast or clearance conservatism; the old OFF trace does not contain future target truth to separate those fully.

| Onboard geometry sensitivity | Current centre error (m) | Axis error (deg) | Saved SAC sequence margin (m) | Hard passing |
|---|---:|---:|---:|---:|
| captured_view | 0.722925 | -29.137 | -0.360848 | False |
| course_axis_only | 0.722925 | -4.138 | -0.036022 | False |
| raw_fit_centre_only | 0.275406 | -29.137 | -0.009087 | False |
| raw_fit_centre_and_course_axis | 0.275406 | -4.138 | 0.283466 | True |
| motion_axis_face_centre_only | 0.012305 | -29.137 | -0.270099 | False |
| motion_axis_face_centre_and_axis | 0.012305 | -4.138 | 0.066853 | True |

Geometry sensitivities retain the measured velocity and original collision checker. Motion supplies the orientation, and the known hull dimensions complete unseen faces using the existing `src/tracking.py:hull_fit_centre` construction. Current truth scores errors after construction and never enters a forecast. This is diagnostic evidence, not an implemented replacement or a claim that course always reveals hull heading.

[audit.json](audit.json) contains component minima, actual-endpoint scoring, model errors, full first-decision diagnostics, trace/source/archive hashes and exact provenance. [prediction_components.csv](prediction_components.csv) contains every predicted sample.

Reproduce with `python -B tools/diagnostics/safety/audit_ch_ho_policy.py --tag NEW_TAG`.

Limitations:

- The successful future SAC commands are retrospective evidence and unavailable to the online filter.
- Recorded state parity covers x,y,heading,surge and policy action; the old OFF trace lacks full sway/yaw/actuator observations.
- Current target truth only scores estimation error; no true target future is supplied or assumed.
- Scoring actual SAC endpoints against the captured onboard constant-velocity target separates own-motion error, but is not actual future target separation.
- Actual-trajectory checks use .5s endpoints and omit terminal extension; prediction checks use .125s samples and the original terminal rules.
- Motion-axis geometry sensitivities are hypotheses, not validated detections: course can differ from hull heading, visible returns may omit both faces, and a single case cannot establish a safe replacement rule.
- This is one completed case in an ongoing campaign, not a population performance claim.
