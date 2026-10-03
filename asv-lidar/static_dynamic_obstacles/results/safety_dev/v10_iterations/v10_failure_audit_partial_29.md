# V10 saved-trace audit (partial snapshot)

Committed results included: 29/32; failures: 14; failures on previously successful SAC cases: 9.

Saved committed results only. Labels describe recorded filter state and commands, not proof of a particular action causing contact. any_safe refers to original bank; search may separately find a plan. No episodes or controller calls.

Commands are compared at the same decision index until the first difference; later policy observations can differ after the trajectories diverge. Every included scene digest/seed and checkpoint/config matches the fresh prior pilot.

| Case | Outcome | SAC / V8 | First change | Changed / decisions | Terminal state | Final no-escape run | Passing SAC tail retained intervention |
|---|---|---|---:|---:|---|---:|---:|
| TS2:BAS-HO-NC-059 | collision:target | goal / collision:target | 21 | 4/27 | no_escape_policy_pass_through | 3 | 0 |
| DV3:DV3-BO-CV-04 | collision:boundary | goal / collision:boundary | 11 | 12/53 | no_escape_policy_pass_through | 7 | 0 |
| TS2:CH-CR-RE-038 | collision:target | goal / collision:target | 17 | 4/24 | unchanged_command_with_passing_model_plan | 0 | 0 |
| TS2:CH-HO-CV-073 | collision:target | goal / collision:target | 13 | 6/28 | no_escape_policy_pass_through | 10 | 0 |
| TS2:BAS-BO-CV-063 | collision:target | goal / collision:target | 5 | 7/18 | no_escape_policy_pass_through | 7 | 1 |
| TS2:P2-L2-CRP-VAR-16 | collision:obstacle | goal / collision:obstacle | 7 | 30/60 | active_changed_command | 0 | 4 |
| TS2:CH-CR-CV-007 | collision:obstacle | goal / collision:obstacle | 11 | 26/66 | no_escape_policy_pass_through | 4 | 9 |
| TS2:CH-CR-CV-031 | collision:boundary | goal / collision:boundary | 15 | 21/113 | no_escape_policy_pass_through | 22 | 4 |
| TS2:BAS-NU-CV-070 | collision:boundary | goal / collision:boundary | 7 | 4/79 | no_escape_policy_pass_through | 14 | 3 |
| TS2:P2-L1-CRP-FIX-05 | collision:target | collision:target / collision:target | None | 0/18 | unchanged_command_with_passing_model_plan | 0 | 0 |
| TS2:P2-L1-CRS-VAR-08 | collision:target | collision:target / collision:target | None | 0/21 | unchanged_command_with_passing_model_plan | 0 | 0 |
| TS2:P2-L1-CRP-VAR-14 | collision:target | collision:target / collision:target | None | 0/17 | unchanged_command_with_passing_model_plan | 0 | 0 |
| TS2:CH-CR-CV-059 | collision:target | collision:target / collision:target | 7 | 4/14 | no_escape_policy_pass_through | 4 | 0 |
| TS2:P2-L2-HO-FIX-11 | collision:target | collision:obstacle / collision:target | 14 | 4/28 | no_escape_policy_pass_through | 11 | 0 |

Full first-divergence records, final10 decisions, and SHA256 input provenance: [v10_failure_audit_partial_29.json](v10_failure_audit_partial_29.json).

No-escape pass-through at contact does not imply the safety layer had no earlier effect. Conversely, a positive model margin at contact shows disagreement with the actual episode, but does not by itself isolate geometry, perception or dynamics.
