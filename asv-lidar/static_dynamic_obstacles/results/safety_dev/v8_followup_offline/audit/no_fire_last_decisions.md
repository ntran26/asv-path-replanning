# V8 never-fired collisions: last-decision supplement

Of the 71 unrescued test-set-v2 failures, 17 never trigger V8. Their recorded policy and V8 trajectories therefore coincide. At the final decision before collision, **14 say `no escape` and 3 say `nominal`**. The 14 no-escape cases have policy margin `-inf`; this indicates failure to find a passing primitive, not a proof that physical escape is impossible.

| Never-fired target collision | Final step (zero-based) | Checked clearance | Urgent |
|---|---:|---:|---|
| TS2:P2-L1-CRP-FIX-05 | 17 | 0.52241985 m | false |
| TS2:P2-L1-CRP-VAR-14 | 16 | 0.37035265 m | false |
| TS2:P2-L1-CRS-VAR-08 | 20 | 0.16501728 m | false |

These three collisions occur after the filter passes the policy with positive predicted clearance. They motivate separate investigation of target perception and target/dynamics prediction, rather than treating every failure as a trigger threshold problem. The CSV contains neither target tracks nor full prediction snapshots, so it cannot distinguish those mechanisms or support an exact offline recheck.

This inference about the final decision applies only to the 17 episodes that never fire. Shadow rows from an intervened episode are on SAC's trajectory and cannot describe the filtered branch's last decision.

All 17 rows, seeds, final outcomes, reasons, margins and monitor flags are in [no_fire_last_decisions.json](no_fire_last_decisions.json). Inputs are the existing [base steps](../../trigger_counterfactual/steps.csv) and [nohold steps](../../trigger_counterfactual/v7_nohold/steps.csv), joined to their episode/branch tables. The script reuses the original audit's strict reconstruction and verifies all six input hashes against [its provenance](provenance.json). The supplement records its own script and dependency hashes; the original audit and raw data are unchanged. No episodes or prediction rollouts were run.
