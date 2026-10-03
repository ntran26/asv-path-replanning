# Fixed-candidate full-DV3 upper bound

The completed `broad_v6_50` comparison records **33/50 v6 goals**, versus
23/50 for fresh v4: 13 gained goals and three lost goals. V6 still has 17
contacts in those original DV3 cases.

Even if **every one of the remaining 100 original cases succeeds**, this frozen
candidate reaches at most **133/150 = 88.67%**. At least 95% requires 143 goals,
so at least 43 selected goals were necessary. The measured 33 rules that out.
This is a deterministic finite-suite upper bound, not an estimated population
success rate or confidence interval. Failure-enriched selection does not affect
this counting argument. The untested 100 cases may lower the actual full-suite
result. The bound does not constrain a future changed candidate.

All 150 original cases stay in the denominator, including initially unobservable
BO panel starts and cases that may be dynamically unrecoverable at reset.
No failed case is excluded to raise the percentage.

Budget at this note: **149/150 consumed**, including five completed legacy
verification simulations. The prepared final calibrated-boundary probe was
blocked before evaluation/token 150 by an unexpected external source change;
there is no probe outcome. The source change and native integration were reviewed in [the integration audit](integration_audit.json).

[Source-hashed calculation](finite_suite_bound.json) |
[Campaign report](report.md) |
[Completed paired results](runs/broad_v6_50/complete.json)
