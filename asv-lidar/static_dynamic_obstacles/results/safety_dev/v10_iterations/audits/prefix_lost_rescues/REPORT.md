# Why the three prefix-only rescues were lost

This uses completed saved `prefix_success16` and `prefix_remaining16` records versus `conditional_policy_32`. Admission and persistence were disabled in the prefix-only runs. No episodes, policy calls, environment calls, source edits, classifier fitting, or new threshold selection were performed. The full 32-case sum has two gained goals and three lost goals versus V10.

All five outcome trades first change an issued command at an exactly matching recorded pre-state and SAC action. Each prefix plan has greater predicted clearance than the parent's selected passing plan at that decision:

| Case | Trade | First command difference | Parent reason | Parent clearance | Prefix clearance |
| --- | --- | ---: | --- | ---: | ---: |
| DV3-HO-VS-01 | goal to boundary contact | 27 | turn | +0.244041 | +0.335032 |
| P2-L1-CRP-VAR-12 | goal to target contact | 3 | searched escape | +0.265700 | +0.286242 |
| P2-L2-HO-FIX-19 | goal to obstacle contact | 16 | turn | +0.289740 | +0.320268 |
| BAS-BO-CV-063 | target contact to goal | 5 | turn | +0.049613 | +0.780840 |
| CH-CR-CV-007 | obstacle contact to goal | 11 | turn | +0.291030 | +0.342730 |

The original V10 same-tail policy alternatives also passed hard checks in all three losses, but had smaller clearances of +0.054801, +0.019490 and +0.061001 m respectively. V10 therefore retained the recovery. The new search found different tails exceeding its existing +0.15 m acceptance margin and issued SAC instead. None of these first changes was an unchecked `no escape` policy pass.

## Three different downstream patterns

**HO-VS-01:** There is only one accepted extra prefix, at decision 27. Its shifted continuation still passes at decisions 28 and 29 with +0.362623 and +0.364305 m. The controller issues different commands from the original searched tail immediately at 28, finds another passing policy plan, then accepts nominal actions. Boundary contact occurs at decision 120, about 47 s after the first changed decision's pre-state, far beyond the original eight-second horizon. The final steps mix small positive margins and currently failing last certificates. This is a changed long-term trajectory; the saved evidence does not show that the initial prefix itself predicted an imminent contact incorrectly.

**CRP-VAR-12:** At decision 3, the original primitive bank has `any_safe=False`, but the parent's escape search has found a passing +0.265700 m selected recovery (best searched escape +0.471613 m). The prefix replaces it with +0.286242 m. At decision 4, that continuation has collapsed to -0.965157 m; the controller brakes instead of executing the stored ordinary-stop/right-rudder command. Later prefixes are accepted at 5 and 9 with +0.153396 and +0.159054 m. The last six decisions are `no escape`, ending in target contact at 34. The immediate collapse is evidence of lost model feasibility, but these traces lack full onboard snapshots, so perception changes, target estimates, own-model mismatch, and terminal extension cannot be separated from the saved fields alone.

**HO-FIX-19:** The decision-16 prefix plans hard-right full astern starting at the next decision. At 17, this shifted continuation still passes with **+0.238881 m**, but the controller selects a different `turn` plan with only **+0.044402 m**. The original nonbraking bank's rounded best margin is +0.11 m. Under the existing selection formula,

`floor = min(0.15, max(best_nonbraking_bank, passing_continuation) - 0.20)`,

the exact floor is +0.038881 m because the continuation dominates the bank. Thus the much tighter turn legitimately passes the implemented selection floor and wins on action distance. At 18 another turn has +0.089251 m; from 19 onward the selected plans fail current checks, and contact occurs at 21. The episode never executes the initially certified eight-second backup. This identifies a concrete selection-margin tradeoff worth an independent ablation, not proof that retaining the backup would rescue the episode.

## What the logs do and do not support as a gate

- Requiring the parent to be currently failing would block the first changed decision in all three losses **and both gains**. It cannot isolate the losses from these benefits using the observed initial state.
- Requiring prefix clearance to exceed the selected parent's clearance would allow all five first changes. It therefore does not prevent these three initial divergences.
- Comparing against the best bank rather than selected parent also lacks a clean separation: HO-FIX19's +0.320 prefix is below the rounded +0.51 bank best, but gained CH-CR007's +0.343 prefix is likewise below its +0.48 bank best. CRP-VAR12's prefix is below its best searched escape; HO-VS01's is above its bank best.
- A later CH-CR007 prefix at decision 30 has +0.155544 m versus a +0.157135 m passing parent continuation. A relative-margin rule would reject this later change, but the resulting closed-loop outcome is unknown.

No scenario-free condition in these existing status/margin comparisons cleanly separates the three losses from both gains. These are local recorded-decision counterfactuals, not predicted outcomes under a new gate. The evidence supports examining how a passing continuation is replaced and how quickly its model validity changes; it does not justify fitting a new threshold to five outcome trades.

`audit.json` contains input trace hashes, paired seed/scene/checkpoint/config checks, complete first-divergence diagnostics, all accepted prefixes, every subsequent decision's recorded margins, and rounded-log floor intervals. The finite continuation-clearance ranges after divergence are [-0.233703,+0.364305] m for HO-VS01, [-0.965157,+0.178643] m for CRP-VAR12 and [-0.047504,+0.238881] m for HO-FIX19. These ranges summarize changing snapshots and plans, not uncertainty bounds on a single prediction.

The policy-prefix module already states the relevant limitation: finite sampled model feasibility supplies neither a robust terminal safe set nor recursive feasibility. Current-policy preservation and eventual goal preservation are distinct outcomes. No failed finite bank proves physical unavoidability.
