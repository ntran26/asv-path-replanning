# V19: transfer current returns owned by an admitted moving target

Status: experimental, developed from saved TS2/DV3 evidence. V19 is a separate change atop V16, not combined with V18. Defaults, constants and the policy checkpoint are unchanged.

## Mechanism and citation

Ordinary perception removes returns near published dynamic tracks before the safety adapters admit additional motion-confirmed targets. Those additional targets can therefore appear twice in the safety forecast: as a moving hull and as a stationary cloud. This can reject a successful manoeuvre because the predicted stationary cloud never moves away.

V19 transfers only exactly matching current returns from static-point constraints to an already published, freshly updated safety target. It requires the existing full-extent, endpoint-translation and residual motion gates, a fresh unique confirmed raw source and current scan, a measured admission/refresh, and a corresponding final age-zero target hypothesis. It uses neither a distance-radius mask nor grid-cell removal. Points in older scan-memory batches, ambiguous point ownership, unrelated points and underlying sensor/tracker/memory data remain unchanged.

The modelling inspiration is Nuss et al. (2016), *A Random Finite Set Approach for Dynamic Occupancy Grid Maps with Real-Time Application*: stationary occupancy assumptions differ from dynamic prediction and measurement updates. [Primary paper](https://arxiv.org/abs/1605.02406). V19 is a deterministic engineering association rule, not that paper's PHD/MIB estimator or a calibrated classification bound.

A merged wall/target cluster or incorrect association can still give false ownership. Existing size/motion gates and exact membership reduce the scope but do not prove that every removed point is dynamic. This method needs collision and preserved-policy-success tests, not just a more favourable predicted margin.

## Saved-data evidence

In DV3-CRS-CV-04 at decision13, raw source77 is not in the environment's published track list but has a fresh independently admitted persistent target view. Nine of58 static snapshot points are exact members of its current cluster. The worst static-contact point is about2.645m from every true static obstacle, measured offline only. Removing those nine changes the recorded successful SAC-tail static clearance from -0.430486 to +0.118128m, leaving49 other points and boundary clearance unchanged. Target-CV clearance still fails at -0.235371m. This diagnoses double representation; it does not establish an episode rescue.

Per-point free-space appearance evidence alone would remove none of the nine points. The experimental rule instead relies on the existing coherent cluster-motion admission. It must retain unrelated/old/ambiguous returns in synthetic tests.

## Evaluation

Compare V16, V18 and V19 on the same frozen12 development cases in `results/safety_dev/v18_development/selection12.json`. Run at most two single-threaded evaluation processes; each tag archives its exact sources, scenario identities and checkpoint. Keep all outcomes, including regressions. The separately frozen18-case primary subset is evaluation only. Do not tune from its outcomes or present it as the full1000-case test set v3.

Storage audit and the rejected physical-model-bank experiment are documented in `SAFETY_LAYER_V18_PLAN.md`.

## Completed development probe

All36 matched V16/V18/V19 runs on12 scenarios completed with unchanged source/checkpoint/selection hashes. Each controller reaches5 goals. V19 changes DV3-CRS-CV-04 from a boundary contact to an obstacle contact; this is not a rescue. There are zero gained or lost V16 goals. V19 remains experimental and is not promoted. The recorded advancement rule permits the separately frozen18-case test-set-v3 comparison (OFF/V16/V19,54 runs). Primary outcomes will be reported separately, without tuning.

## Completed primary quick check

All54 OFF/V16/V19 runs on18 frozen primary-v3 cases are complete and validated. SAC reaches14/18 goals; V16 and V19 each reach15/18. Both rescue FS-CRP-VAR-023 and preserve all14 SAC successes. V19 gains0 and loses0 goals against V16. Three collisions remain. The candidate is not promoted, and this is not the full1000-case benchmark. No primary outcomes were used for tuning. See `results/safety_dev/testset_v3_main/quick_v19_report/report.md` and `results/safety_dev/v18_development/report.md`. Total new runs in this continuation:90; none queued.
