# V8 follow-up: preserved baseline and experimental V9

**No new episodes were run.** The completed counterfactual data confirm V8 at
**912/1000 goals** on test set v2, with 57 rescued failures and 17 lost SAC
successes. V9 is implemented and unit-tested, but its success rate is unmeasured.

| Existing saved result | SAC | V7 | V8 |
| --- | ---: | ---: | ---: |
| Test set v2 (1000) | 872 | 904 | 912 |
| DV3 (150) | 111 | 128 | 127 |

The [saved-data audit](audit/report.md) reconstructs all 1,150 cases, checks
first-fire branch coverage and exports the
[71 unrescued failures](audit/remaining_71_ts2_failures.csv) and
[17 lost policy successes](audit/broken_17_ts2_successes.csv).
These are development results, not untouched validation of this follow-up.

## Findings that guide the changes

V8 still permits `last certificate`: following a stored recovery after its
current check fails. Its first-fire outcomes are zero rescues/three losses on
test set v2, but four rescues/one loss on DV3. Requiring a currently passing
plan is a concrete hypothesis, not an established performance improvement.

A blanket larger margin is also unsupported: 22 of V8's 57 test-set rescues
began below the 0.15 m nominal trigger margin. The new comparison therefore
keeps the existing hard geometry checks and evaluates SAC and the proposed
override with the same backup tail.

The [last-decision supplement](audit/no_fire_last_decisions.md) finds three
never-intervened target collisions with positive predicted margins immediately
before contact. This suggests prediction/perception failures in addition to
limited rescue capability; the CSVs cannot distinguish the two explanations.

## Implemented

- **V8 isolation fix:** an instance method disables hold-back. It no longer
  temporarily changes a shared constant that another filter could observe.
  Its serial behavior and recorded result remain the baseline.
- **Experimental V9:** recheck the actual proposed command and full backup.
  Suppress overrides whose current check fails. Where SAC with the same tail
  also passes and has at least as much clearance, preserve SAC instead. A
  suppressed override does not imply that SAC itself is certified safe.
- **Optional turning-target predictions:** CV plus sampled port/starboard
  turns, including changing target hull orientation. All active trajectory
  checks share the same copied estimates. This option remains disabled by
  default because its turn-rate assumption has not been calibrated.
- **Version selection:** native environment and suite CLIs accept V9. The
  counterfactual runner now instantiates the requested V8/V9 class and records
  it accurately instead of silently choosing V7.

The feasible-backup/minimal-intervention design follows
[Wabersich and Zeilinger](https://arxiv.org/abs/1812.05506v4). Multiple target
scenarios follow [Johansen et al., Sections 3.3–3.4](https://torarnj.folk.ntnu.no/colregs_cams.pdf),
with constant-turn model inspiration from
[Li and Jilkov](https://doi.org/10.1109/TAES.2003.1261132). These are engineering
adaptations, with no imported formal safety guarantee.

Implementation: `src/safety_v9.py`, `src/safety_target_prediction.py`.
Experimental selection: runtime `SAFETY_VERSION = 9`; target-turn rate defaults
to zero. Details and limitations are in the [V9 plan](../../../planning/archive/safety/SAFETY_LAYER_V9_PLAN.md).

## Verification and limits

**134 focused pytest tests passed**, plus three supplemental synthetic data checks.
Tests exercise synthetic states, predictors, mocked filter selection
and version dispatch stopped before physics. The old V8 test that actually
steps an environment was not run. Commands, counts and source hashes are in
[verification.json](verification.json) and [integration_audit.json](integration_audit.json).

Saved shadow CSVs do not contain target states, full plans or serialized branch
states, so they cannot validate the new target model or reconstruct V9 outcomes.
The three changes need separate paired episode evaluation if later authorized.
Independent switches are available for the two V9 guards and the target ensemble.
No increase above 91.2%, or achievement of 95–100%, is claimed here.
