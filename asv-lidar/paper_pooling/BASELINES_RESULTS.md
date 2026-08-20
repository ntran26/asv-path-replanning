# BASELINES_RESULTS.md

What was run, what came out, and what behaved unexpectedly.

Companion to `BASELINES_NOTES.md`, which records what the code actually does and
where the task brief's assumptions conflicted with it.

**Every method was evaluated by one harness, on one frozen set of 500 layouts,
with the same observation, action space, reward, termination rule and episode
cap.** PPO has three from-scratch training seeds; the classical baseline has
three independent tuning searches; SAC is the single deployed checkpoint.

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
| **SAC (proposed)** | **0.950** | 0.038 | 0.012 | **0.908** | 0.233 | 74.9 | **20.9** |
| PPO (3 seeds, 1M) | 0.871 | 0.100 | 0.029 | 1.212 | 0.503 | 12.8 | 22.4 |
| LOS+APF (3 searches) | 0.960 | 0.033 | **0.007** | 1.328 | **0.768** | 9.0 | 27.2 |

PPO per-seed success: 0.870 / 0.912 / 0.832. LOS+APF per-search: 0.968 / 0.972 /
0.940. Full table with stratified bootstrap 95 % CIs:
`eval_results/baselines/comparison_table.md`.

No best-checkpoint selection is applied to PPO: these are the **final 1M-step
checkpoints**, the same "model at 1M steps" SAC is reported from. Under the
corrected setup the final and best checkpoints are nearly the same policy, so
the earlier two-row treatment is no longer needed.

### Table for the revision — paired statistics vs SAC

| Comparison | Success (SAC / other) | McNemar p | RMS CTE (SAC / other) | Wilcoxon p |
|---|---|---|---|---|
| vs PPO seed 0 | 0.950 / 0.870 | **8.6e−06** | 0.929 / 1.193 | 7.0e−35 |
| vs PPO seed 1 | 0.950 / 0.912 | **0.0295** | 0.929 / 1.826 | 2.1e−74 |
| vs PPO seed 2 | 0.950 / 0.832 | **1.8e−09** | 0.929 / 0.841 | 0.0099 |
| vs LOS+APF s1 | 0.950 / 0.968 | 0.211 (n.s.) | 0.929 / 1.217 | 1.2e−35 |
| vs LOS+APF s2 | 0.950 / 0.972 | 0.108 (n.s.) | 0.929 / 1.378 | 1.4e−55 |
| vs LOS+APF s3 | 0.950 / 0.940 | 0.576 (n.s.) | 0.929 / 1.532 | 1.6e−65 |

### Claims that are safe to make

1. **SAC reproduces at 95.0 % over the 500-episode holdout**, with 3.8 %
   obstacle and 1.2 % border collisions. (Small drift from the published
   94.0 % / 3 % / 3 % — see §5.)
2. **SAC significantly outperforms PPO under identical conditions, in all three
   seeds** (p = 8.6e−06, 0.0295, 1.8e−09), on equal total environment
   interactions and a matched network. This is the DRL baseline Reviewer 1 asked
   for, and the result is unambiguous.
3. **SAC matches a strongly-tuned classical baseline on success while following
   the path significantly more accurately and finishing significantly faster.**
   RMS CTE 0.908 m vs 1.328 m (p < 1e−35 in all three comparisons); 20.9 s vs
   27.2 s. For a survey ASV, staying on the survey line is the mission
   objective.
4. **The classical baseline required a documented 250-configuration random
   search over 20 parameters (25 000 episodes) to reach that parity**, and its
   performance still varied with the search seed (0.940–0.972). The learned
   policy needs no hand-designed gains and is the controller actually deployed.
5. **SAC follows the path more accurately than either baseline** — 0.908 m
   against PPO's 1.212 m and LOS+APF's 1.328 m.

### Claims that are NOT supported — do not make these

* ✗ "SAC outperforms the classical baseline." It does not; the success
  difference is not significant in any of the three comparisons (p = 0.211,
  0.108, 0.576).
* ✗ "SAC maintains larger clearances." It does not — LOS+APF keeps 0.768 m and
  PPO 0.503 m against SAC's 0.233 m.
* ✗ Any unqualified generalisation claim beyond 0–4 obstacles and straight
  paths. See §8.

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

**The intervals do not all cover the same thing.** `SAC (published)` is one
checkpoint, so its interval is episode/layout variance only. `PPO` pools three
training seeds, so its interval covers seed *and* episode variance. `LOS+APF`
pools three independent 250-configuration searches: the controller is
deterministic — re-running it reproduces its CSV byte for byte — so its interval
covers **tuning-procedure** variance, the analogue of a training seed for a
non-learned method. This asymmetry should be stated in the paper.

---

## 2. Paired statistics

All tests paired on `episode_id`; layout difficulty dominates the between-episode
variance and pairing removes it. McNemar is exact (binomial on discordant pairs).
Full table: `eval_results/baselines/paired_stats_table.{csv,md}`.

**SAC vs PPO: SAC significantly better on success in all three seeds**, and on
tracking accuracy in all three. Note seed 2 is the interesting one — PPO tracks
*better* there (0.841 vs SAC's 0.929) while succeeding much less often, which is
the same accuracy/clearance trade-off appearing within PPO's own seed spread.

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

*Caveat, added in this revision:* an earlier draft claimed this reproduced
across three from-scratch SAC retrains. Those retrains used the incorrect
curriculum and have been withdrawn (§4), so the current evidence is the two
published checkpoints plus the contrast against six baseline runs. The
conclusion is unchanged but the supporting breadth is narrower than previously
stated.

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
`ppo_fx_seed{0,1,2}_final/`, `los_apf_s{1,2,3}/`, plus
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
* **SAC is a single policy while the baselines are multi-seed.** State this. If
  a multi-seed SAC result is wanted, it needs a fresh 3-seed retrain under the
  corrected fixed-RPM setup (~10 h); the earlier retrains are withdrawn. Note
  that the withdrawn runs spanned 0.818–0.922, so a corrected retrain is
  unlikely to sit at the deployed policy's 0.950 — from-scratch training
  variance is real and is better reported as a limitation than engineered away.
