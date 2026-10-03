# BAS070 recorded successful-policy future

Saved-only audit of `persistent_prefix32`, attempt 014, `TS2:BAS-NU-CV-070`, seed 400469. No new episode, environment/reset/step, policy inference, or hidden-state initialization.

The first command difference is decision 10. All earlier commands and recorded pre-states match the original successful SAC trace; the current pre-state matches to 1e-9. Sixteen actual subsequent SAC commands are replayed from V14's captured onboard snapshot and pre-command actuator history. They are retrospective evidence, not commands available to an online filter.

| Fixed successful-command forecast | Predicted minimum | Actual recorded poses against same frozen scene |
| --- | ---: | ---: |
| Static memory | **-0.208106 m**, first violation 2.25 s | **+0.085633 m** |
| Boundary | +1.483513 m | +1.247167 m |
| Track 86 | +4.881946 m | +5.791688 m |

The actual-pose column uses only 0.5 s endpoints and omits the terminal extension because true sway speed is unavailable in the old SAC trace. It is not a continuous collision proof or true-future target reconstruction. Predicted checks retain the complete original 0.125 s sampling and terminal rules.

Prediction error at 2.5 s is 0.172647 m and 15.0211 degrees; at 8 s it is 0.648792 m and 10.6954 degrees. The first override therefore does not establish impending boundary contact: the model mispredicts the successful obstacle maneuver. Its later boundary contact follows a changed trajectory.

The exact ordinary-bank replay gives best nonbraking margin 0.1818278657 m and currently passing continuation 0.2647659117 m. The existing ordinary floor is therefore **0.0647659117 m**, below the separately searched policy-first backup's recorded 0.1033911241 m. The chosen intervention's clearance is 0.1181799009 m. The backup search's stricter 0.15 m acceptance rule is a separate, measured selector inconsistency.

`audit.json` contains full fixed commands, component minima, every half-second error, exact bank reconstruction and source/data hashes. Reproduce using:

```powershell
python -B tools/diagnostics/safety/audit_saved_policy_future.py --run-dir results/safety_dev/v10_iterations/persistent_prefix32 --trace 014_v14.jsonl --tag NEW_TAG
```

Omit `--tag` for read-only output. See [initial-state decomposition](../bas070_initial_state_decomposition/REPORT.md) before attributing the error to observer initialization.
