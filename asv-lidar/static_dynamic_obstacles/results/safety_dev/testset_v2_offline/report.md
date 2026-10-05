# Test-set-v2 safety development: saved-data findings and experimental V7

**No new episodes were run.** This continuation is limited to
saved-data analysis and code tests. V7 is implemented but has no
measured success rate. The old V6 report remains unchanged.

## What the saved evidence shows

| Saved comparison | Cases | Policy goals | Filter goals | Rescued failures | Lost policy goals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete test-set-v2 SAC baseline | 1000 | 872 | Not evaluated | Not evaluated | Not evaluated |
| Exact archived OFF/V4 pairs within v2 | 735 | 684 | 680 | 19 | 23 |

The 128 SAC failures comprise 61 target, 55 obstacle and 12 boundary contacts.
Crossings contribute 76 failures, including 46 target contacts. All 15 L2
head-on cases fail: 12 obstacle and three boundary contacts. Deployment
layouts contribute 76 failures among 239 cases; the frozen-source subset
contributes 52 among 761.

The 735 historical pairs all belong to the frozen-source portion. The saved
field-layout cases have no archived paired V4 results. The 23 lost policy goals
end in 16 target, four obstacle and three boundary contacts. Another 262
preserved-goal episodes contain V4 interventions; those interventions cannot
all be labelled false alarms because the filter changes subsequent states.

See [all 1,000 saved results analyzed](analysis/report.md),
[failure inventory](analysis/failures.csv), and
[historical matched comparisons](saved_v4_20261002T183354Z_5a395bcf/REPORT.md).
The latter includes every paired outcome, all 1,000 coverage identities and
hash/scene/seed checks; missing results were not reconstructed.

## What changed

Experimental **V7** tests a precise opportunity to avoid an override: can the
proposed recovery still pass all existing checks if SAC acts for the next
decision? It substitutes only that first command, preserves the backup tail,
checks the full eight-second horizon with the unchanged margin, and allows
SAC through only when this complete check passes. Failed substitutions retain
V6's proposal. The policy is never admitted merely because a classifier says
that the scenario resembles a successful case.

This adaptation is inspired by first-action minimal intervention with a
feasible backup in [Wabersich and Zeilinger (2021)](https://arxiv.org/abs/1812.05506v4).
It is not their formal safety guarantee or a simulation of SAC's future feedback.

A separate **shadow risk monitor** records per-hazard clearance, near-term
constraint violation, clearance trend and persistent evidence using only
onboard estimates and the current policy action. Its output never changes a
command. Separating monitoring from intervention follows
[Hsu, Hu and Fisac (2024)](https://arxiv.org/abs/2309.05837); the one-second
diagnostic horizon and persistence labels remain unvalidated engineering choices.

Code: `src/safety_v7.py`, `src/safety_risk_monitor.py`; experimental runtime
selection is `SAFETY_VERSION = 7`. Existing defaults are unchanged. Detailed
method citations, behavior, limitations and integration are in the
[V7 plan](../../../planning/archive/safety/SAFETY_LAYER_V7_PLAN.md).

## Why there is no learned success/failure trigger yet

Saved v2 results contain whole-episode summaries, not predecision LiDAR,
tracking, action or pose sequences. Mean speed, minimum range and terminal
outcome would leak future information if treated as online features. The
analysis identifies useful failure groups, but cannot establish when an
intervention is necessary or when SAC will recover by itself.

For future comparisons, [151 successful controls](analysis/successful_controls.csv)
were selected using recorded static geometry in the same cell and variant:
113 failures receive two controls each; all 15 L2-HO failures have no successful
same-cell control. The matching method is adapted from the repository's
test-set geometric selection and documented in the analysis report. These
controls are a regression inventory, not a validated trigger or full-suite test.

All test-v2 cases used here are now development evidence for this safety layer.
The candidate may still lose goals through prediction error or repeated backup
deferral. No improvement over 87.2%, and no 95-100% result, is claimed.

## Verification

**96 focused tests passed.** These checks cover policy substitution, full-horizon acceptance,
braking and actuator history, monitor isolation/persistence, version dispatch
stopped before physics, and saved-data integrity. Exact commands and passing
counts are in [verification.json](verification.json). Source hashes and the
unchanged constants/checkpoint identity are in [integration_audit.json](integration_audit.json).
No environment episode was started, and no evaluation ledger was advanced.

During final verification another task added a training start-clearance redraw
and helper to `env.py`. Those changes were preserved and isolated in
[the external-change audit](external_env_change.diff). All 11 native dispatch
checks were repeated successfully against the resulting file; they stop before
physics and do not validate the separate training change.
