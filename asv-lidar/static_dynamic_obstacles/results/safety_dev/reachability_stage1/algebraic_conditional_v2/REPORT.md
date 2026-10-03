# Conditional saved-trace containment diagnostic

Case: TS2:BAS-NU-CV-070; decision 11; requested horizon 8 s.
Status: **unknown**. Stop reason: union box width resource limit; no narrowing applied.
Delay branches: [12, 13, 14, 15, 16]; elapsed diagnostic time: 11.796 s.
Scored endpoint misses: []. No new episodes were run.

| Horizon (s) | Status | All six true endpoint states contained |
|---:|---|---|
| 0 | enclosed | True |
| 0.5 | enclosed | True |
| 1 | unknown_width_limit | None |

The BAS-HO target-turn check uses its own recorded filtered branch only:

- Target 0: 81.884644 degrees over 0.5 s; declared 8 degrees/s bound violated: True.

- Gaussian sensor noise and parameter jitter have unbounded support; declared finite boxes are conditional hypotheses.
- Recorded future controls are NONCAUSAL same-branch diagnostic inputs, not an executable feedback policy.
- Truth is used only after propagation for PREdecision endpoint scoring, never to construct/tune bounds.
- Natural interval arithmetic loses parameter/state dependence; wide/failed boxes mean unknown, never safe.
- Experimental mpmath.iv real-arithmetic numerical-map enclosure; binary64 roundoff, continuous ODE and swept geometry are unvalidated.
- No obstacle/target occupancy certificate, recursive feasibility, terminal backup or mission safety guarantee.
- Actual plant parameters and servo are unlogged, so a containment miss cannot distinguish an assumption violation from an implementation error.

Reachable-set architecture reference: [Kochdumper et al.](https://arxiv.org/abs/2210.10691). This prototype does not inherit its guarantees.
