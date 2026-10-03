# Saved boundary-regression comparison

No new episodes, policy inference or environment calls. The three cases share exact scenario digests, reset seeds, checkpoint and model configuration across the compared manifests. Their pre-intervention recorded states match SAC to 1e-9. This is a targeted mechanism audit, not a performance estimate.

| Case | SAC | V8 | V9 | V10 default | V10 any-feasible | Prefix-only V11 |
| --- | --- | --- | --- | --- | --- | --- |
| DV3-BO-CV-04 | goal | boundary | goal | boundary | boundary | boundary |
| CH-CR-CV-031 | goal | boundary | goal | boundary | target | boundary |
| BAS-NU-CV-070 | goal | boundary | boundary | boundary | boundary | boundary |

All three first interventions occur while the one-second shadow monitor reports no threatening hazard. That short-horizon observation does not certify longer-term safety.

- **BO04:** V8/V10/prefix first differ at decision 11. SAC requests rudder -0.767577, 11.679958 RPM; the filter issues -0.638258, 11.790726 RPM from an expired continuation. The proposed full plan and same-tail policy plan both predict violation at 3 s, with clearances -0.421260 and -0.427251 m. V9 suppresses that override and exactly reproduces the entire 31-step successful SAC command trajectory. Prefix search finds 57 hard-passing plans but rejects all because its best +0.139726 m is below 0.15 m.
- **CH031:** first difference at decision 15 is a passing intervention, rudder -0.5 / 0 RPM instead of SAC +0.886980 / 0.522504 RPM. Chosen clearance is +0.248309 m; same-tail SAC also passes, +0.062369 m. Prefix search finds 91 passing plans, best +0.141487 m, below the current ordinary pool floor of 0.15 m. V9 later avoids an expired-plan override and succeeds; simply accepting every feasible policy tail instead changes the failure to target contact. There is no evidence for treating every low-margin passing policy plan as safe in closed loop.
- **BAS070:** V8/default V10 first override at decision 7. Prefix search preserves those early commands and leaves **one actual intervention, decision 10**, which still changes the goal into a boundary contact. At that state SAC requests +0.897853 rudder / 11.289418 RPM; the filter issues -0.5 / 12 RPM. The selected plan passes with +0.118180 m. The search has 121 hard-passing policy-first plans, best +0.103391 m, but rejects them at 0.15 m.

## Exact selector-floor discrepancy

The existing ordinary branch computes `best = max(best currently passing nonbraking candidate, currently passing continuation)` then `floor = min(0.15, best - 0.20)`. This is an eligibility floor within an already hard-feasible pool; it does not alter obstacle, boundary, target or terminal checks.

| Case / first divergence | Current ordinary floor | Best hard-passing policy prefix | Consequence |
| --- | ---: | ---: | --- |
| BO04 / 11 | absent: no original plan passes | +0.139726 | A new feasible plan exists while the selected fallback is currently failing. |
| CH031 / 15 | +0.150000 | +0.141487 | The prefix remains ineligible under the same ordinary rule. |
| BAS070 / 10 | **+0.06476591174532181** | **+0.10339112412476607** | Prefix satisfies the ordinary selector floor, but is rejected by the separate search trigger. |

BAS's exact floor is recoverable even from old scalar logs: the currently checked continuation is +0.2647659117453218 m, greater than the full rounding interval of the reported best nonbraking margin 0.18 m. The newer full-snapshot replay confirms the exact nonbraking best **+0.18182786571257586 m** and the same continuation/floor. CH's rounded best 0.43 m is enough to prove the floor saturates at 0.15 m. BO has no ordinary feasible pool and must not be assigned a fictitious floor derived from its stale certificate.

A bounded policy-prefix search minimizing modification of the current input is motivated by [Wabersich and Zeilinger, Sec. 4.1, Eq. 5](https://arxiv.org/html/1812.05506v4#S4.SS1). The present finite search, deterministic model and absent invariant terminal set do not implement that paper's guarantee.

## Proposed V15 semantics, after the active freeze

1. Record this decision's exact `selection_best` and `selection_floor` in V6 only when its ordinary passing pool is actually evaluated. Initialize both to null each call; idle/search/brake-only/no-escape branches must not inherit old values. A currently passing braking continuation still belongs to the ordinary continuation comparison.
2. Add a protected V11 prefix-minimum hook with unchanged default 0.15. A V15 subclass can use `max(0, selection_floor)` only when the exact current branch and finite best/floor metadata are present and internally consistent with the existing formula.
3. Separately permit minimum zero only when the current V10 pair check explicitly evaluates the actually proposed encoded command/full tail as failing, uses the same pre-command actuator history and checker, has finite valid scores, and has not already preserved/replaced the policy. Missing checks, NaN scores, old negative margins, or a label such as last-certificate alone are insufficient. This replaces a currently failing fallback only with a newly hard-passing complete policy-first backup.
4. Search-escape and separate brake-only branches retain 0.15. Command-limiter skips, first-action equality, NaN astern transport, complete horizon/padding, dynamic/static/boundary/terminal checks, and dual-response callbacks remain unchanged.

Tests should prove disabled/default parity, exact unrounded same-call floor, stale/undefined score rejection, nonnegative clamp, explicit failed-pair gating, both branches' scope, and unchanged parent state/action on failed search. A finite-horizon passing plan can still fail physically due to model/perception error, and reducing the extra trigger buffer has that documented limitation. This is not a predicted rescue count.

## Boundary-specific late-stage limitation

BAS eventually approaches the end wall after missing the goal. At decision 64 the closest boundary clearance is 1.422 m, all sampled plans fail, and no target is within the traffic gate. Only three nonbraking recovery templates are available; astern candidates and search brakes are excluded because `traffic=False`. The next ten decisions are no-escape until boundary contact. CH's late side-wall phase has the same absence of astern, but BO still has a nearby target track and therefore does not share this restriction.

A separate saved-state test can add emergency astern candidates for a boundary threat and require the same complete hard checks. That tests missing actuation authority, not a longer horizon. It may prevent contact without restoring goal success and is not yet demonstrated. Never turn this into an unconditional brake beside every wall.

## New full-snapshot evidence

Completed `persistent_prefix32` results for these cases are BO04 target contact (46 steps), CH031 goal (52 steps, zero interventions), and BAS070 boundary (73 steps, one intervention). Thus the CH perception change already resolves that earlier regression; do not treat its older outcome as current.

BAS's full snapshot permits replay of the recorded successful SAC future from the common decision-10 state. The model predicts **static-memory** violation at 2.25 s, minimum -0.208106 m; boundary clearance stays +1.483513 m. Actual successful SAC poses scored against the same frozen points stay +0.085633 m at the available 0.5 s samples. See [BAS replay](../bas070_recorded_policy_future/REPORT.md) and [initial-state decomposition](../bas070_initial_state_decomposition/REPORT.md). The initial intervention is caused by the predicted obstacle maneuver, while the eventual contact is with a boundary.

The original off/V8/V9, V10 and prefix-only traces do not contain full onboard snapshots or actuator history. Their policy observations and true x/y/heading/surge cannot faithfully reconstruct those missing prediction inputs. No replay of those missing states is claimed.

## Artifacts

`audit.json` records every relevant saved decision, first command divergences, source/trace/manifest hashes and exact floor derivations. Run `python -B tools/diagnostics/safety/audit_boundary_regressions.py --tag NEW_TAG`; omit the tag for read-only output. Existing tags are refused. `reproducer_evaluated_bytes.py` preserves the evaluated working script bytes.
