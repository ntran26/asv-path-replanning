# V18: recent hull orientation when the current fit is missing

Status: tested on12 targeted development cases: V18 and fresh V16 both reach5 goals, with no gained or lost V16 goals. V18 is not promoted. V19 completed the small frozen primary check under the recorded tie rule:15/18 goals, matching V16, versus SAC14/18. No default controller or constants are changed.

## Method and attribution

The tracker can lose its fitted hull heading while retaining a moving/stopping target. Its fallback uses velocity course, which can rotate the predicted hull because of velocity noise rather than measured shape. V18 holds the most recent validated hull axis for the same observed source, bounded by the existing `TRACK_MAX_MISSES` age budget. Only fresh, matched, confirmed, new-serial observations can update or use it. Missing identities, expiry and detectable identity resets invalidate memory. Synthetic persistent views and a current V16 motion-axis correction take precedence.

The modelling inspiration is Granstrom, Baum and Reuter (2016), *Extended Object Tracking: Introduction, Overview and Applications*, Section II: separate object kinematics and extent from individual spatial measurements. [Primary paper](https://arxiv.org/abs/1604.00970). This deterministic bounded hold is an engineering hypothesis, not their probabilistic estimator or a safety guarantee. A turning hull with poor observations can invalidate a held orientation.

Implementation: `src/safety_heading_memory.py`, `src/safety_v18.py`; tests: `tests/test_safety_v18.py`. `hull_heading_memory=False` preserves V16's perception object. V18 inherits V16 rather than V17. It changes neither margins nor policy preference, and uses no scenario labels, future policy calls or simulator truth.

## Development evidence and evaluation design

Saved FIX05 data showed a missing fit replacing a recent roughly114-degree hull axis with noisy roughly148-degree course. Rechecking the same plan with the retained axis changes predicted clearance from positive to negative. This motivates a model correction; it does not demonstrate that a changed action can rescue the episode.

Before any V18 episode, `results/safety_dev/v18_development/selection12.json` freezes12 development cases: the motivating case, all six known V16 losses of SAC successes across the32/40 cohorts, two geometry rescues, an active rescue and two successful no-intervention controls. The paired development probe compares fresh V16, V18 and the separate V19 candidate with identical scenes, seeds and checkpoint;36 runs are complete. All three reach5/12 goals. The cohort deliberately contains all six previously broken SAC successes, so this is not a full-benchmark success rate. See `results/safety_dev/v18_development/validated_development_report/report.md`.

`results/safety_dev/testset_v3_main/quick_v18_case_list.json` separately freezes18 primary cases by source and encounter, with a second target variant where available. Ranking uses a fixed hash of case identity and does not consult outcomes. The full1000-case selection remains unchanged. This subset covers every source/encounter group; it is a quick check, not a full-set success estimate. Freeze controller code before primary evaluation and do not tune from these outcomes. Compare fresh SAC OFF, V16 and V18 if the development probe supports taking the candidate forward.

## Rejected model adaptation

A finite physical model bank was tested offline on37 saved episode versions, selecting from identified, bootstrap-mean and28 existing bootstrap parameter vectors using only the first ten past measured transitions. Selection used measurement-noise-scaled prediction error and then froze. Inspiration: Ljung's prediction-error system identification; [primary technical report](https://liu.diva-portal.org/smash/get/diva2:316694/FULLTEXT01.pdf). The candidate histories used their own actuator delay/servo dynamics.

It improved one-step yaw prediction but worsened many continuous8-second forecasts, including12/20 full-length V15 forecasts and2/4 V16 forecasts. No fitted bank enters the controller. See `results/safety_dev/v10_iterations/audits/bootstrap_model_bank_cohort/REPORT.md` for denominators and counterexamples.

## Storage maintenance

Completed495 trace files in20 immutable runs were transparently NTFS-compressed. All content hashes and paths remain valid;945,003,157 allocated bytes (0.880GiB) were recovered. Prior report hash references were checked. Source archives, results and scene caches were retained. A bytecode deletion preflight encountered a cloud reparse attribute and stopped before deleting anything. Maintenance audit: `results/safety_dev/maintenance/ntfs_compact_20261003T111517Z/README.md`.

Constants, the SAC baseline3 kept-best3M checkpoint and the separate Paper2 project remain unchanged. At most two single-threaded evaluation processes may run; active training is protected.

A second saved-only bank experiment selected the same physical vectors with a continuous five-second training forecast rather than resetting to measurements at each step. The first ten past commands, measurement-noise weights and held validation endpoints stayed fixed. It still worsened V15 full-eight-second heading RMSE (4.0224 to4.8152 degrees) and position RMSE (0.18559 to0.21165m), despite improvement on the smaller V16 probe. Stop this bank family without tuning weights or window length further. The same Ljung prediction-error citation applies. See `results/safety_dev/v10_iterations/audits/bootstrap_continuous_bank_cohort/REPORT.md`; no episodes or production changes came from either bank audit.

Final cleanup including the new completed traces compressed585 files and reclaimed974,323,274 allocated bytes (0.907GiB), all hashes unchanged. The final report is `results/safety_dev/v18_development/report.md`. All90 new episode runs are complete.
