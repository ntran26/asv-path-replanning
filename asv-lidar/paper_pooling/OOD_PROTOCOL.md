# Out-of-distribution generalisation test — pre-registered protocol

**Written and committed before any OOD layout was generated or any episode run.**
Its purpose is to fix the design, the methods, the metrics and the reporting rule
in advance, so that the result cannot be selected after the fact. Whatever the
numbers show, they go into `BASELINES_RESULTS.md`.

---

## 1. Question

Does the learned policy generalise better or worse than a hand-tuned classical
controller to layout conditions **neither** was trained or tuned on?

This is a fair question to ask of both families, because the shift is equally
outside both of their design distributions:

* SAC and PPO trained on obstacle counts `TRAIN_OBS_COUNTS = [0, 1, 2, 3, 4]`
  with `PATH_MODE = "straight"` (`src/config.py`).
* LOS+APF's parameters were selected by random search on
  `eval_layouts/tune_layouts_v1.json`, which is 20 layouts each at obstacle
  counts 0–4, straight paths only.

Neither has seen 5 obstacles. Neither has seen a curved reference path.

## 2. Conditions (fixed here, before generation)

Three sets, 100 layouts each, generated with the same
`generate_suite.build_scenario` machinery and the same inflated-grid A*
feasibility filter as the evaluation suite, from a **disjoint seed base**
(7,000,000+, against 675,974–1,076,073 for evaluation and 5,000,000+ for tuning).

| Set | Obstacle counts | Path mode | What is out of distribution |
|---|---|---|---|
| `ood_obs5` | **5** | straight | obstacle density above anything trained or tuned on |
| `ood_curve` | 0–4 | **curve** | reference-path geometry never seen |
| `ood_curve_obs5` | **5** | **curve** | both at once |

Everything else is held at the values used throughout the study: 10 × 25 m
basin, `RPM_STAGE = 1`, `MAX_EPISODE_STEPS = 700`, same observation, same action
space, same termination rule, same success definition.

## 3. Methods

Evaluated on all three sets, through the same harness, with no per-condition
adjustment of any kind:

* SAC (published) — `models/sac_model_1M.zip`
* SAC (retrained) — best checkpoint, seeds 0/1/2
* PPO — best checkpoint, seeds 0/1/2
* LOS+APF — the three independently tuned configurations s1/s2/s3

**Best checkpoints are used for the learned methods**, not the 1M-step finals.
Reason, fixed in advance: PPO's final checkpoints collapsed in 2 of 3 seeds and
fail everywhere, so including them would tell us about the collapse rather than
about generalisation. The question here is whether the learned *representation*
transfers, so each learned method is given its strongest form. This is stated
wherever the results are reported.

**No parameter of any method is re-tuned for these conditions.** The LOS+APF
configurations are exactly those selected on the straight-path tuning set. That
is the point of the test.

## 4. Metrics and analysis

The existing per-episode schema, unchanged. Primary endpoint: **success rate**.
Secondary, reported regardless: RMS cross-track error, minimum obstacle
clearance, minimum lateral border clearance, control effort, mean absolute
rudder rate, completion time.

Paired tests on `episode_id` within each set: exact McNemar on success, Wilcoxon
signed-rank on the continuous metrics — the same procedure as the in-distribution
comparison.

**Degradation** is reported as the change from each method's own in-distribution
result on the frozen 500, so the comparison is between *how much each method
loses*, not merely where it ends up.

## 5. Directions this could go

Recorded in advance so that neither outcome can be presented as the expected one:

* **Favouring the learned policies.** A network trained across a distribution of
  layouts may interpolate to denser or curved geometry, whereas the APF's
  behaviour is governed by fixed thresholds (`c_threshold`, `k_rep`,
  `side_tie`) selected for a specific obstacle density.
* **Favouring the classical controller.** LOS guidance is derived from path
  geometry and is in principle indifferent to whether the path is straight or
  curved, while the learned policies have only ever seen straight-path
  observations — `course_error` and `lookahead_course_error` will take on
  combinations they never encountered.
* **Favouring neither.** All methods may degrade together, in which case the test
  is uninformative and will be reported as such.

The curved-path condition in particular has a plausible mechanism pointing each
way. No prediction is being made here.

## 6. Reporting rule

Every number produced under this protocol is reported in
`BASELINES_RESULTS.md`, including conditions where SAC performs worst. No set is
dropped, no method is dropped, and no condition is added afterwards to replace
one that came out unfavourably. If a further condition is ever run, it is
appended with its own justification and clearly marked as post-hoc.

## 7. On retraining

Retraining SAC or PPO on 5-obstacle or curved layouts would remove exactly the
property under test — those conditions would then be in-distribution and the
experiment would answer a different question. Training is therefore **not**
extended for this test.

The single exception, fixed in advance: if all methods fail near-completely on a
set (making it uninformative rather than discriminating), that set may be
re-run with an extended training distribution applied **equally** to SAC and
PPO, and it will then be labelled an in-distribution comparison on a harder
task, not a generalisation result.
