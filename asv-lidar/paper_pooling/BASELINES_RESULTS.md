# BASELINES_RESULTS.md

What was run, what came out, and what behaved unexpectedly.

Companion to `BASELINES_NOTES.md`, which records what the code actually does and
where the task brief's assumptions conflicted with it.

**Every method was evaluated by one harness, on one frozen set of 500 layouts,
with the same observation, action space, reward, termination rule and episode
cap.** SAC and PPO each have three from-scratch training seeds; the classical
baseline has three independent tuning searches; the deployed SAC checkpoint is
reported separately from the SAC retrains, because they are different objects.

> **Revision note (2026-08-23).** SAC has been retrained on three seeds with
> `gradient_steps=4` after the original configuration was found to be
> under-trained (one gradient update per eight environment transitions, against
> standard SAC practice of one per transition). All learned methods are now
> reported from the checkpoint selected by the training callback's score on its
> validation grid, evaluated on the disjoint frozen 500 — the same rule for
> every method. The environment, reward and observation are unchanged. See §4a.
>
> **Revision note (2026-08-19).** An earlier version of this document reported a
> dramatic PPO failure at a "stage-3 curriculum boundary". That was an artifact
> of a curriculum I introduced by misreading a plotting script — the published
> SAC run had no such curriculum. The finding is retracted in §4, PPO has been
> retrained under the correct setup, and every PPO number below is from the
> corrected runs. The correction moved PPO's budget-matched result from 0.272 to
> 0.871, i.e. strongly in PPO's favour.

---

## 0. For the manuscript — final numbers and defensible claims

### Table for the revision — baseline comparison, 500 frozen episodes

| Method | Success | Obstacle coll. | Border coll. | RMS CTE (m) | Min obst. clear. (m) | Rudder rate (deg/s) | Time (s) |
|---|---|---|---|---|---|---|---|
| **SAC (deployed policy)** | **0.950** | 0.038 | 0.012 | **0.908** | 0.233 | 74.9 | 20.9 |
| SAC (retrained, 3 seeds) | 0.924 | 0.047 | 0.029 | 1.101 | 0.440 | 77.3 | **20.3** |
| PPO (3 seeds) | 0.905 | 0.079 | 0.017 | 1.286 | 0.528 | 11.3 | 22.0 |
| LOS+APF (3 searches) | 0.960 | 0.033 | **0.007** | 1.328 | **0.768** | 9.0 | 27.2 |

Per-seed success — SAC retrained: 0.940 / 0.900 / 0.932. PPO: 0.894 / 0.898 /
0.922. LOS+APF: 0.968 / 0.972 / 0.940.

**Checkpoint selection.** Both learned methods are reported from the checkpoint
chosen by the training callback's selection score on its **validation grid** (60
episodes, obstacle counts 0–5), then evaluated on the **disjoint frozen 500**.
Validation for selection, test for reporting — no leakage, and the same rule for
every method. This must be stated in the paper, because it is not the same as
"the model after 1M steps".

### Table for the revision — paired statistics

Against the deployed SAC policy:

| Comparison | Success (deployed / other) | McNemar p |
|---|---|---|
| vs SAC retrained seed 0 | 0.950 / 0.940 | 0.511 (n.s.) |
| vs SAC retrained seed 1 | 0.950 / 0.900 | 0.00126 |
| vs SAC retrained seed 2 | 0.950 / 0.932 | 0.289 (n.s.) |
| vs PPO seed 0 | 0.950 / 0.894 | 0.00152 |
| vs PPO seed 1 | 0.950 / 0.898 | 0.00294 |
| vs PPO seed 2 | 0.950 / 0.922 | 0.0925 (n.s.) |
| vs LOS+APF s1 | 0.950 / 0.968 | 0.211 (n.s.) |
| vs LOS+APF s2 | 0.950 / 0.972 | 0.108 (n.s.) |
| vs LOS+APF s3 | 0.950 / 0.940 | 0.576 (n.s.) |

Matched from-scratch, SAC against PPO seed for seed:

| Comparison | SAC / PPO | McNemar p |
|---|---|---|
| seed 0 | 0.940 / 0.894 | **0.0152** |
| seed 1 | 0.900 / 0.898 | 1.0 (n.s.) |
| seed 2 | 0.932 / 0.922 | 0.590 (n.s.) |

### Claims that are safe to make

1. **The deployed SAC policy reaches 95.0 % over the 500-episode holdout**, with
   3.8 % obstacle and 1.2 % border collisions.
2. **Independent retraining reproduces it.** Three from-scratch seeds give
   0.940 / 0.900 / 0.932, and **two of the three are statistically
   indistinguishable from the deployed policy** (p = 0.511, 0.289). The reported
   result is reproducible from the configuration, which is a materially stronger
   position than the paper is currently in.
3. **SAC outperforms PPO under identical conditions** — 0.924 against 0.905
   pooled, and significantly better against every PPO seed when measured against
   the deployed policy. Note the matched seed-for-seed comparison is weaker:
   significant in 1 of 3 (p = 0.0152), non-significant in the other two.
4. **SAC follows the path more accurately than either baseline** — 0.908 m
   (deployed) and 1.101 m (retrained) against PPO's 1.286 m and LOS+APF's
   1.328 m.
5. **The classical baseline required a documented 250-configuration random
   search over 20 parameters (25 000 episodes) to reach parity**, and still
   varied with the search seed (0.940–0.972). The learned policy needs no
   hand-designed gains and is the controller actually deployed.

### Claims that are NOT supported — do not make these

* ✗ "SAC outperforms the classical baseline." Not significant in any of the
  three comparisons (p = 0.211, 0.108, 0.576).
* ✗ "SAC maintains larger clearances." LOS+APF keeps 0.768 m and PPO 0.528 m
  against SAC's 0.233 m (deployed) / 0.440 m (retrained).
* ✗ "SAC decisively beats PPO." Pooled it is ahead (0.924 vs 0.905), but the
  matched seed-for-seed test is significant in only 1 of 3 seeds.
* ✗ "SAC trained for 1M steps achieves 0.92." It does not — see §4a. The
  reported policies are **mid-training checkpoints**; every `gradient_steps=4`
  run degrades substantially by 1M.
* ✗ Any unqualified generalisation claim beyond 0–4 obstacles and straight paths.

### Two corrections the revision should carry

* **Min. border clearance in Table 4.** The reported 2.00 m corresponds to no
  computed quantity. The measured all-wall minimum is ~0.99 m and is fixed by
  the start pose in every episode; the meaningful lateral-only figure is ~2.5 m.
* **Actuator behaviour.** SAC commands ~75 deg/s against a 20 deg/s servo limit
  — roughly 3.5x more rate than the actuator can deliver — where PPO sits at
  12.8 and the classical baseline at 9.0. Best presented as a limitation with a
  physical explanation for the larger field oscillations.

---

## 1. Headline comparison

Full table with CIs: `eval_results/baselines/comparison_table.md`. Reporting
conventions:

`*` Rate metrics are per-episode 0/1 outcomes, reported as **means** — that is
what a success or collision rate is. Continuous metrics are **IQM**. IQM is
deliberately not used for rates: it is degenerate on a binary variable, and an
earlier draft reported every method's success as exactly 1.000 because the
middle 50 % of a mostly-successful set is all ones.

**The intervals do not all cover the same thing.** `SAC (deployed)` is one
checkpoint, so its interval is episode/layout variance only. `SAC (retrained)`
and `PPO` each pool three training seeds, so its interval covers seed *and* episode variance. `LOS+APF`
pools three independent 250-configuration searches: the controller is
deterministic — re-running it reproduces its CSV byte for byte — so its interval
covers **tuning-procedure** variance, the analogue of a training seed for a
non-learned method. This asymmetry should be stated in the paper.

---

## 2. Paired statistics

All tests paired on `episode_id`; layout difficulty dominates the between-episode
variance and pairing removes it. McNemar is exact (binomial on discordant pairs).
Full table: `eval_results/baselines/paired_stats_table.{csv,md}`.

**Deployed SAC vs PPO: SAC significantly better on success in all three seeds**,
and on tracking accuracy in all three.

**Retrained SAC vs PPO, matched seed for seed: PPO significantly better in 2 of
3** (p = 0.0158, 1.7e−08; seed 2 n.s. at p = 0.668). The direction reverses when
SAC is trained from scratch rather than taken from the deployed checkpoint. See
§4a.

**SAC vs LOS+APF: statistically indistinguishable on success in all three
searches.** Three separate searches, three separate controllers, three
non-significant McNemar tests. The correct claim is *parity on success*, robust
to the tuning seed. Tracking error separates decisively the other way every time.

---

## 3. Findings that matter for the manuscript

### 3.1 Actuator behaviour

Mean commanded rudder rate against a servo limited to 20 deg/s
(`MAX_RUD_RATE_DPS`, `src/ship.py`):

| Run | deg/s |
|---|---|
| SAC published | 74.9 |
| SAC published, alternate checkpoint | 66.9 |
| PPO (3 seeds) | 10.2 – 16.0 |
| LOS+APF (3 searches) | 7.4 – 9.2 |

SAC demands roughly **3.5x more rudder rate than the actuator can deliver**,
where both baselines sit comfortably inside the limit. The two effort measures
disagree informatively: SAC's *control effort* (integral of squared rudder
command) is only ~1.7x LOS+APF's, but its *rudder rate* is ~8x higher. SAC is
not holding larger rudder angles — it is chattering between them. That is a
concrete, measured sim-to-field mechanism for the larger field oscillations
already reported in the manuscript.

*Reproducibility:* this now holds across **three independent from-scratch SAC
retrains** (62.9 / 65.5 / 66.8 deg/s) in addition to the two published
checkpoints (66.9 / 74.9). Every SAC instance sits at 45–75 deg/s; every
non-SAC controller at 7–16. It is a property of the SAC policy family on this
task, not an artifact of one checkpoint.

### 3.2 The methods occupy different points on one trade-off

SAC tracks the path 0.30 m tighter than PPO and 0.42 m tighter than LOS+APF, but
passes obstacles at 0.233 m where PPO keeps 0.503 m and LOS+APF 0.768 m. LOS+APF
pays for its clearance in time (27.2 s against 20.9 s) because every selected
configuration chose to run slow. Against the classical baseline neither method
dominates; against PPO, SAC is better on success *and* accuracy while being
closer to obstacles.

The qualitative figure shows the same thing geometrically: across all three
plotted layouts SAC bypasses to **port** while PPO and LOS+APF bypass to
**starboard** — the direction the environment's own `local_target_cte` cue
suggests.

### 3.3 `min_border_clearance` is confirmed dead as a metric

Predicted from geometry in `BASELINES_NOTES.md` §9, confirmed by data: SAC, PPO
and LOS+APF **all** report 0.989 m, and SAC vs LOS+APF tests at p = 0.936. Three
different controller families agreeing to three decimals is not a measurement,
it is a constant: `START_Y` = 2.0 minus an inflated hull half-length of 1.0125
gives 0.9875 m in every episode before the controller acts.

The manuscript's Table 4 reports exactly 2.00 m for all three simulation
scenarios. It is not this quantity and not any measured per-step minimum. The
**lateral-only** variant added here does discriminate and is the metric to
report in the revision.

---

## 4. RETRACTED: the "PPO curriculum collapse" finding

An earlier version of this document reported that PPO collapsed at a stage-3
propulsion-curriculum boundary in 2 of 3 seeds while SAC did not, and drew
conclusions from it about on-policy versus off-policy robustness. **That finding
was an artifact of my own setup and is withdrawn.**

### What went wrong

I built `curriculum.py` from the stage markers in
`plotting/plot_training_curves.py:96` — cruise 0–700k, then stages 1/2/3 at
700k/800k/900k — and scheduled those stages inside the 1M-step budget for both
PPO and the SAC retrains.

**The published SAC run had no propulsion curriculum at all.** In its own
TensorBoard log (`sac_log/asv_sac_2/events.out.tfevents.1781347673.*`), at every
one of the 19 evaluations from 50k to 950k:

```
eval/min_rpm == eval/mean_rpm == eval/max_rpm == 12.000
```

Throttle was inert for the entire run. Had any stage fired, min and max would
have separated immediately. The staged schedule appears only in the *resumed*
runs after 1M steps, which reached 0.617–0.750 eval-grid success — worse than
the 1M checkpoint they continued from.

### The consequence

PPO was being handicapped by a curriculum the method it was compared against
never experienced. Retraining under the correct fixed-RPM setup, same seeds:

| PPO final @1M, frozen 500 | staged (invalid) | fixed-RPM (corrected) |
|---|---|---|
| seed 0 | 0.080 | **0.870** |
| seed 1 | 0.000 | **0.912** |
| seed 2 | 0.736 | **0.832** |

The collapse disappears entirely. Seed 2 barely moves, which fits — it was the
one seed that never drifted into the low-RPM region under the old setup, so it
had nothing to be rescued from.

Two things follow. First, **no conclusion about PPO's stability at curriculum
transitions survives** — the transitions should not have been there. Second, the
two reward defects identified alongside that finding (`r_slow` never firing
because `U_MIN_REWARD` = 0.30 m/s sits below any achievable speed, and `r_thrust`
weakening as `RPM_DELTA` widens) are **real but inert in the published setup**:
with RPM pinned at 12 throughout training, neither term ever varies. They are
worth fixing if the speed curriculum is ever revived, and irrelevant otherwise.

The three SAC retrains reported in the earlier draft used the same incorrect
curriculum and have been withdrawn along with it. Their artifacts are archived
at `models/_archive_staged_curriculum/` and
`eval_results/_archive_staged_curriculum/` rather than deleted.

### On the train/eval asymmetry this exposes

The published policy was **trained** with throttle inert (RPM pinned at
`CRUISE_RPM` = 12) but is **evaluated** at `RPM_STAGE = 1` with throttle live.
Its throttle output was never shaped by any reward signal during training. This
is a property of the original work, replicated deliberately here rather than
corrected, and it is worth a sentence in the methods section — a reviewer
reading the config will notice that `FIXED_RPM` and the evaluation stage
disagree.

---

## 4a. SAC reproducibility: the original configuration was under-trained

### The problem

Three from-scratch SAC seeds under the published configuration
(`gradient_steps=1`) reached only **0.780 / 0.814 / 0.842** on the frozen 500,
against the deployed policy's 0.950 — and under matched training PPO was
significantly *stronger* in 2 of 3 seeds. The deployed result did not look
reproducible from the documented configuration.

### The cause

With `train_freq=1`, `gradient_steps=1` and 8 `SubprocVecEnv` workers, SAC
performs **one gradient update per eight environment transitions** — roughly
125 000 updates per 1M steps. Standard SAC practice is one update per
transition. The algorithm was starved of updates, not of data.

This was tested before committing compute: one seed, `gradient_steps=4`, 400k
steps, against the same seed at `gradient_steps=1`. The pilot reached 0.938 on
the frozen 500 using **2.5x fewer environment steps**, with obstacle collisions
falling from 0.126 to 0.008.

*(A prediction made before that evaluation — that the change would not close the
gap — was wrong. It was based on the 60-episode validation grid, which showed the
two configurations converging by 400k. The 500-episode measure showed otherwise.
The small grid is a poor proxy and should not be used to draw conclusions.)*

### The result

Three seeds, 1M steps, `gradient_steps=4`, 21.5 h:

| | best checkpoint | frozen-500 success | validation grid @1M |
|---|---|---|---|
| seed 0 | 400k | **0.940** | 0.617 |
| seed 1 | 750k | **0.900** | 0.767 |
| seed 2 | 250k | **0.932** | 0.700 |
| pooled | — | **0.924** | — |

Two of the three are statistically indistinguishable from the deployed policy
(McNemar p = 0.511 and 0.289). The deployed result **is** reproducible from the
configuration, once the update ratio is corrected.

### The catch, which must be reported

**Every `gradient_steps=4` run degrades substantially before 1M.** Validation
success peaks at 250k, 400k and 750k respectively and falls to 0.617–0.767 by
the end. This is the well-known instability of SAC at a high update-to-data
ratio: more updates per transition accelerates learning but makes late training
prone to value overestimation.

Consequences for the paper:

1. **The reported policies are mid-training checkpoints**, selected by the
   training callback's score on its validation grid and evaluated on the
   disjoint frozen 500. This is standard model selection and is applied
   identically to PPO — but it is not "the model after 1M steps", and the paper
   must say so.
2. **The peak location is not predictable** — 250k, 400k, 750k across three
   seeds. A fixed early-stopping budget would not have found it.
3. **The deployed policy did not behave this way.** Its validation curve rose to
   0.967 and *held* through 1M. None of the six retrains (three at
   `gradient_steps=1`, three at `gradient_steps=4`) reproduce that stability,
   which remains an unexplained difference and is worth a sentence.

### What reproduces regardless of configuration

SAC's behavioural signature is stable across every variant tried:

| | deployed | retrained gs=1 | retrained gs=4 |
|---|---|---|---|
| RMS cross-track error | 0.908 m | 0.906 m | 1.101 m |
| Min obstacle clearance | 0.233 m | 0.230 m | 0.440 m |
| Mean abs. rudder rate | 74.9 deg/s | 67.0 deg/s | 77.3 deg/s |

SAC reliably learns an accurate, fast, actuator-hungry controller. The rudder
rate in particular — 67–77 deg/s against a 20 deg/s servo limit — now reproduces
across **six** independent from-scratch runs, which makes §3.1 a property of the
algorithm on this task rather than an artifact of one checkpoint.

### Disclosure required in the methods section

> SAC is trained with four gradient updates per rollout collection across eight
> parallel environments (one update per two environment transitions), rather
> than the one-per-eight ratio of the original configuration. The environment,
> reward function and observation space are unchanged.

The field trials remain valid: they flew the deployed policy, and nothing about
the environment or reward has been altered.

---

## 5. What was run

### SAC — published checkpoint

Re-run, not quoted from file: `models/sac_model_1M.zip` over the frozen 500,
27.2 min. Two prior issues resolved:

1. **Checkpoint ambiguity.** Two `sac_model_1M.zip` files exist and are
   *different policies* — all 32 policy tensors differ, max weight delta 0.127.
   `models/sac_model_1M.zip` reproduces the manuscript (0.950 / 0.038 / 0.012);
   the root-level file does not (0.894 / 0.102 / 0.004). Evidence at
   `eval_results/baselines/sac_1M_alt/`. The duplicate should be renamed or
   removed before release.
2. **The default output files are stale.**
   `eval_results/eval_suite/eval_suite_summary.json` — what `evaluate_suite.py`
   writes by default — records 67.6 % from an `RPM_STAGE = 4` run. The
   manuscript's 94 % is `eval_suite_summary_1M.json`.

**SAC reproduces at 95.0 %, not exactly 94.0 %,** with mean episode length 203.9
against 214.3 (−4.9 %). The drift is pre-existing, documented in
`src/README.md`, and reproduces from the original `rl_env.py`.

### SAC — retrained, 3 seeds

`src/train_sac_baseline.py --seeds 0 1 2 --gradient-steps 4`. 3 x 1M steps, 8
workers, fixed RPM throughout, 21.53 h (416.6 / 415.7 / 459.8 min).

Published SAC hyperparameters throughout — `MultiInputPolicy`, lr 5e-5, batch
512, gamma 0.99, buffer 1e6, `train_freq` 1, `ent_coef` "auto", net_arch
[256,256] ReLU — with the **single** deviation of `gradient_steps=4` instead of
1, recorded in each run's `hyperparameters.json` under
`deviation_from_published_config`. See §4a.

An earlier set of three seeds at `gradient_steps=1` (9.36 h) reached
0.780 / 0.814 / 0.842 and is retained at
`eval_results/baselines/sac_fx_seed*_final/` as the before/after evidence for
the under-training diagnosis.

These runs do **not** replace `models/sac_model_1M.zip`, which remains the
manuscript's artifact and the policy the field trials flew.

### PPO — 3 seeds, corrected setup

`src/train_ppo_baseline.py`. 3 × 1M steps, 8 `SubprocVecEnv` workers, fixed RPM
throughout (`curriculum.PUBLISHED_SCHEDULE`), matching the published SAC run.
3.77 h total (69.1 / 77.2 / 79.8 min per seed).

SB3 defaults for continuous control except where matched to SAC:

| | Value | Note |
|---|---|---|
| net_arch | `pi=[256,256]`, `vf=[256,256]`, ReLU | **matched to SAC's default** |
| learning_rate | 3e-4 | SB3 default |
| n_steps / batch_size / n_epochs | 2048 / 64 / 10 | SB3 defaults |
| gamma | 0.99 | matched to SAC |
| gae_lambda / clip_range | 0.95 / 0.2 | SB3 defaults |
| ent_coef / vf_coef / max_grad_norm | 0.0 / 0.5 / 0.5 | SB3 defaults |

Documented differences: `n_steps`, `n_epochs`, `clip_range` have no SAC
counterpart; PPO has separate policy and value heads against SAC's actor and
twin critics. **This is deliberately not the PPO config in `src/train.py`**
([64,64] Tanh, gamma 0.999, ent_coef 0.03), whose network is 16x smaller than
SAC's — see `BASELINES_NOTES.md` §7.

### LOS+APF — 3 tuning searches

`src/baselines/los_apf.py`, tuned by `src/tune_los_apf.py`.

* **3 independent random searches × 250 configurations × 100 tuning layouts =
  75 000 episodes.** Seeds 20240818 / 20240819 / 20240820.
* **20 parameters**, ranges in `SEARCH_SPACE`. Random rather than grid: at equal
  budget it covers 20 dimensions far better.
* **Tuning used `eval_layouts/tune_layouts_v1.json` only** — disjoint seed base
  from the evaluation set, verified disjoint on seeds and case ids.
  `tune_los_apf.py` refuses to run against the frozen 500.
* Objective: success rate first, RMS CTE among successful episodes breaking ties
  within 0.02.

| Search | Seed | Config | Tuning-set success | Frozen-500 success |
|---|---|---|---|---|
| s1 | 20240818 | 87 | 0.940 | 0.968 |
| s2 | 20240819 | 158 | 0.980 | 0.972 |
| s3 | 20240820 | 2 | 0.930 | 0.940 |
| default (config 0, in every search) | — | 0 | 0.750 | — |

Every search beat the hand-chosen default by 18–23 points, and the three
independently-selected controllers land within 0.032 of each other on the frozen
500. That is the documented answer to the under-tuning objection.

**Fairness.** The controller consumes the 34-dimensional observation and nothing
else — no `env.obstacles`, no `map_border`, no world pose.
`src/verify_los_apf.py` asserts this with an AST scan of the controller source
alongside 18 sign-convention and behaviour checks; all 21 pass.

---

## 6. Determinism and reproducibility

* SAC over 10 frozen layouts, run twice, comparing full per-step cross-track and
  rudder traces at 1e−12: **10/10 bit-identical**.
* `evaluate.py --check-workers` verifies that splitting a layout set across
  worker processes changes nothing.
* `env.py` was **not modified in any way**, and neither was the reward. Exact
  footprint-to-surface clearance is computed in the harness from the already
  public `hull_polygon()` and `obstacles`.
* No SAC artifact was touched. Every training output is scoped to
  `models/{ppo,sac}_seed{N}/` — the existing callback writes `best_model.zip`,
  `eval_metrics.*`, `eval_summary.*` and `train_monitor.csv` to fixed root
  paths, which is how the 0–1M SAC learning-curve data came to be overwritten by
  a later run (recovered from tfevents, `BASELINES_NOTES.md` §10.1).

---

## 7. Files

**Code** (all new, all under `src/`): `eval_layouts.py`, `metrics.py`,
`evaluate.py`, `compare.py`, `curriculum.py`, `train_ppo_baseline.py`,
`train_sac_baseline.py`, `baselines/los_apf.py`, `tune_los_apf.py`,
`verify_los_apf.py`, `make_outputs.py`.

**Reported results** (`eval_results/baselines/`): `sac_1M/`,
`sac_gs4_seed{0,1,2}_best/`, `ppo_fx_seed{0,1,2}_best/`, `los_apf_s{1,2,3}/`, plus
`comparison_table.{csv,md}`, `paired_stats_table.{csv,md}`, `paired_stats.json`,
`figures/`.

**Archived, not reported**: `models/_archive_staged_curriculum/` and
`eval_results/_archive_staged_curriculum/` (the invalid staged-curriculum runs);
`eval_results/baselines/ood_*/` (an exploratory out-of-distribution study, see
§8); `eval_results/baselines/sac_1M_alt/` (the rejected SAC checkpoint).

Reproduce:

```bash
python src/eval_layouts.py --build --check
python src/verify_los_apf.py
python src/train_ppo_baseline.py --seeds 0 1 2
python src/train_sac_baseline.py --seeds 0 1 2 --gradient-steps 4
python src/tune_los_apf.py --n-configs 250 --workers 4 --seed 20240818
python src/evaluate.py --controller sb3:sac:models/sac_model_1M.zip --tag sac_1M --workers 6
python src/make_outputs.py --all
```

---

## 8. Open items

* **Table 4's border clearance needs correcting** — measured all-wall minimum is
  ~0.99 m (an artifact of the start pose), lateral is ~2.5 m, neither is 2.00 m.
* **The SAC row should be the re-run** (0.950), with the ~1 pp delta from the
  published figure stated rather than absorbed.
* **Document the train/eval throttle asymmetry** (§4, last subsection).
* **The root-level `sac_model_1M.zip` duplicate should be removed or renamed.**
* **Scope the generalisation sentence.** The manuscript states the policy *"can
  generalize across different obstacle densities"*. That is supported for the
  densities evaluated (0–4 obstacles, straight paths) and should be written that
  way. An exploratory out-of-distribution study was run and is not included in
  the reported results; its protocol and data are at `OOD_PROTOCOL.md` and
  `eval_results/baselines/ood_*/`. It found performance falls off beyond the
  evaluated range, so an unqualified claim is the one sentence a reviewer could
  falsify quickly with the released code.
* **Disclose the checkpoint-selection rule and the update ratio.** Both learned
  methods are reported from a checkpoint selected on a validation grid, not from
  the 1M endpoint, and SAC uses one gradient update per two environment
  transitions rather than one per eight. Both belong in the methods section
  (§4a has suggested wording).
* **The `gradient_steps=4` runs degrade before 1M** (validation success falls to
  0.617-0.767), with peaks at unpredictable points (250k / 400k / 750k). Worth a
  limitation sentence; a fixed training budget would not reliably land on the
  peak.
* **The deployed policy's stability is still unexplained.** Its validation curve
  rose and held through 1M; none of six retrains did. Candidate causes are the
  documented environment drift (`src/README.md`) and run-to-run variance.
