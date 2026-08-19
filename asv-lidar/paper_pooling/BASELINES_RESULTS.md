# BASELINES_RESULTS.md

What was run, what came out, and what behaved unexpectedly.

Companion to `BASELINES_NOTES.md`, which records what the code actually does and
where the task brief's assumptions conflicted with it. Read that first if you
are checking whether the comparison is set up fairly.

**Every method below was evaluated by one harness, on one frozen set of 500
layouts, with the same observation, action space, reward, termination rule and
episode cap.** Every learned method has **three from-scratch training seeds**;
the classical baseline has **three independent tuning searches**. That symmetry
is deliberate — comparing a 1-run method against a 3-seed method would make the
intervals mean different things.

---

## 1. Headline comparison

Frozen 500-episode evaluation set (`eval_layouts/eval_layouts_v1.json`), all
methods at `RPM_STAGE = 1`.

| Metric | SAC (published) | SAC (retrained, final) | SAC (retrained, best) | PPO (final @1M) | PPO (best ckpt) | LOS+APF (tuned) |
|---|---|---|---|---|---|---|
| Success rate * | 0.950 | 0.816 | 0.876 | 0.272 | 0.910 | **0.960** |
| Obstacle collision * | 0.038 | 0.158 | 0.079 | 0.199 | 0.078 | **0.033** |
| Border collision * | 0.012 | 0.026 | 0.045 | 0.511 | 0.012 | **0.007** |
| Timeout * | 0.000 | 0.000 | 0.000 | 0.018 | 0.000 | 0.000 |
| Rudder saturation * | 0.098 | 0.042 | **0.009** | 0.527 | 0.115 | 0.098 |
| RMS cross-track error (m) | 0.908 | **0.874** | 0.955 | 1.572 | 1.264 | 1.328 |
| Min obstacle clearance (m) | 0.233 | 0.231 | 0.299 | 0.518 | 0.505 | **0.768** |
| Min border clearance, all walls (m) | 0.989 | 0.989 | 0.989 | 0.425 | 0.989 | 0.989 |
| Min border clearance, lateral (m) | 2.514 | 2.477 | 2.299 | 0.722 | 2.447 | **2.524** |
| Control effort | 9.846 | 7.869 | 7.266 | 12.459 | **5.219** | 5.845 |
| Mean abs. rudder rate (deg/s) | 74.91 | 69.20 | 51.84 | **6.68** | 11.05 | 8.96 |
| Completion time (s) | 20.9 | 19.6 | 20.6 | **19.5** | 21.6 | 27.2 |
| Runs / episodes | 1 / 500 | 3 / 1500 | 3 / 1500 | 3 / 1500 | 3 / 1500 | 3 / 1500 |

Confidence intervals omitted here for width; the full table with stratified
bootstrap 95 % CIs is `eval_results/baselines/comparison_table.md`.

`*` Rate metrics are per-episode 0/1 outcomes, reported as **means** — that is
what a success or collision rate is. Continuous metrics are **IQM**. IQM is
deliberately not used for rates: it is degenerate on a binary variable, and an
earlier draft reported every method's success as exactly 1.000 because the
middle 50 % of a mostly-successful set is all ones.

**The intervals do not all cover the same thing.** `SAC (published)` is one
checkpoint, so its interval is episode/layout variance only. The retrained SAC
and PPO rows pool three training seeds, so theirs cover seed *and* episode
variance. LOS+APF pools three independent 250-configuration searches: the
controller is deterministic — re-running it reproduces its CSV byte for byte —
so its interval covers **tuning-procedure** variance, the analogue of a training
seed for a non-learned method.

`SAC (published)` and `SAC (retrained)` are **not pooled**. They are different
objects: the published checkpoint carries a long history of resumed, hand-staged
runs that a 1M-step retrain does not, and it is ~7–13 points stronger as a
result.

---

## 2. Paired statistics

All tests paired on `episode_id`; layout difficulty dominates the between-episode
variance and pairing removes it. McNemar is exact (binomial on discordant pairs).
Full table: `eval_results/baselines/paired_stats_table.{csv,md}`.

### SAC vs LOS+APF — tied on success, in all three searches

| Search | LOS+APF success | McNemar p | RMS CTE (SAC / LOS+APF) | Wilcoxon p |
|---|---|---|---|---|
| s1 | 0.968 | **0.211** | 0.929 / 1.217 | 1.2e−35 |
| s2 | 0.972 | **0.108** | 0.929 / 1.378 | 1.4e−55 |
| s3 | 0.940 | **0.576** | 0.929 / 1.532 | 1.6e−65 |

**No independently-tuned LOS+APF controller differs significantly from SAC on
success rate.** Three separate searches, three separate controllers, three
non-significant McNemar tests. The correct claim is *parity on success*, and it
is now robust to the tuning seed rather than resting on a single search.
Tracking error separates decisively in the other direction every time.

### SAC vs PPO (best checkpoint)

| Seed | PPO success | McNemar p | Wilcoxon p (RMS CTE) |
|---|---|---|---|
| 0 | 0.902 | 0.0056 | 2.3e−34 |
| 1 | 0.906 | 0.0103 | 1.6e−71 |
| 2 | 0.922 | 0.0925 | 7.1e−31 |

### SAC published vs SAC retrained (best checkpoint)

| Seed | Retrained success | McNemar p |
|---|---|---|
| 0 | 0.888 | 0.00075 |
| 1 | 0.818 | 2.1e−12 |
| 2 | 0.922 | 0.098 |

The published checkpoint is significantly stronger than a 1M-step retrain in 2
of 3 seeds. **1M steps from scratch does not reproduce the published policy** —
worth stating in the revision, because it means the published result depends on
its particular staged training history, not on the algorithm plus a step budget.

One generated row is `NaN` by construction: PPO seed 1's final checkpoint has
zero successful episodes, so the both-succeeded subset is empty and no
signed-rank test exists. Reported rather than hidden.

---

## 3. Findings that matter for the manuscript

### 3.1 Actuator behaviour — the strongest result, and now reproducible

Mean commanded rudder rate against a servo limited to 20 deg/s
(`MAX_RUD_RATE_DPS`, `src/ship.py`):

| Run | deg/s |
|---|---|
| SAC published | 74.9 |
| SAC published, alternate checkpoint | 66.9 |
| SAC retrained seed 0 / 1 / 2 (final) | 66.2 / 64.2 / 71.0 |
| SAC retrained seed 0 / 1 / 2 (best) | 51.1 / 45.4 / 56.9 |
| PPO best (3 seeds) | 9.1 – 13.2 |
| LOS+APF (3 searches) | 7.4 – 9.2 |

**Every SAC variant sits at 45–75 deg/s; every non-SAC controller at 7–13.**
This is no longer an observation about one checkpoint — it reproduces across
three independent from-scratch training runs, which makes it a property of the
SAC policy family on this task rather than an artifact of the published model.
Against LOS+APF the difference is p ≈ 1e−83.

The two effort measures disagree informatively: SAC's *control effort* (integral
of squared rudder command) is only ~1.7x LOS+APF's, but its *rudder rate* is ~8x
higher. SAC is not holding larger rudder angles — it is chattering between them,
demanding roughly 3.5x more rate than the actuator can deliver. That is a
concrete, measured sim-to-field mechanism for the larger field oscillations
already reported in the manuscript.

### 3.2 The methods occupy opposite ends of one trade-off

SAC tracks the path ~0.19–0.50 m tighter (depending on search) but passes
obstacles at 0.233 m where LOS+APF keeps 0.768 m — a factor of 3.3. LOS+APF pays
in time (27.2 s against 20.9 s) because every selected configuration chose to
run slow. Neither dominates; they sit at different points on the same
safety/accuracy/speed surface, at statistically indistinguishable success.

The qualitative figure shows the same thing geometrically: across all three
plotted layouts SAC bypasses to **port** while PPO and LOS+APF bypass to
**starboard** — the direction the environment's own `local_target_cte` cue
suggests.

### 3.3 `min_border_clearance` is confirmed dead as a metric

Predicted from geometry in `BASELINES_NOTES.md` §9, confirmed by data: SAC,
retrained SAC, PPO-best and LOS+APF **all** report 0.989 m, and SAC vs LOS+APF
tests at p = 0.936. Four different controller families agreeing to three decimals
is not a measurement, it is a constant: `START_Y` = 2.0 minus an inflated hull
half-length of 1.0125 gives 0.9875 m in every episode before the controller acts.

The manuscript's Table 4 reports exactly 2.00 m for all three simulation
scenarios. It is not this quantity and not any measured per-step minimum. The
**lateral-only** variant added here does discriminate (2.30–2.52 for the healthy
controllers, 0.72 for collapsed PPO) and is the metric to report in the revision.

---

## 4. PPO collapses at the stage-3 curriculum boundary; SAC does not

The most consequential unexpected result, and the one where retraining SAC
changed the conclusion.

### The controlled comparison

Same environment, same reward, same replicated curriculum on the same total-step
schedule, same 1M budget, same evaluation. Only the algorithm differs.

| | collapsed at stage 3 | mean RPM at 1M | success @1M (frozen 500) |
|---|---|---|---|
| **SAC** | **0 of 3** | 12.3 / 10.3 / 12.3 | 0.816 / 0.794 / 0.838 |
| **PPO** | **2 of 3** | 8.5 / 7.9 / 11.6 | 0.080 / 0.000 / 0.736 |

PPO best→final drop: −0.82, −0.91, −0.19.
SAC best→final drop: −0.07, −0.02, −0.08.

### Correction to an earlier claim in this document

An earlier revision stated that the collapse was *"a property of the
environment's reward, not of PPO"*, and that SAC *"would face the same gradient
if retrained from scratch today"*. **Retraining SAC shows that conclusion was
wrong**, and it is corrected here rather than quietly dropped.

What survives is the measured part. Both reward gaps are real and independently
verifiable:

1. **`r_slow` never fires.** The anti-stall threshold is `U_MIN_REWARD` = 0.30 m/s,
   but steady-state speed at the 6.0 RPM floor is **0.859 m/s**. The term is
   exactly 0.0 at every RPM in the entire stage-3 range.
2. **`r_thrust` halves at the worst moment.** It is normalised by `RPM_DELTA`, so
   running at the floor costs 0.050/step in stage 1 but only **0.025/step** in
   stage 3 — against a living cost of −0.5/step, a 5 % nudge.

And the dynamics consequence is real (measured on `ShipModel`, full rudder, 8 s):

| RPM | yaw rate (deg/s) | turn radius |
|---|---|---|
| 6.0 | 5.00 | **4.22 m** |
| 12.0 | 9.39 | 2.78 m |

At the stage-3 floor the vessel loses ~47 % of its yaw rate and its turn radius
grows 52 %, in a basin 10 m wide.

**But these are enabling conditions, not sufficient causes.** Presented with
exactly the same gradient, all three SAC seeds held mean RPM near 12 and never
entered the low-RPM region; two of three PPO seeds drifted into it and failed
almost entirely by boundary contact. The honest conclusion is a genuine
**algorithm-robustness difference at a curriculum boundary**, on an environment
whose reward makes the failure cheap to reach.

A plausible reading — offered as interpretation, not measurement — is that SAC's
replay buffer keeps pre-transition experience alive across the boundary while
PPO's on-policy advantage estimates go stale the moment the action→RPM mapping
changes. This study does not isolate that mechanism.

### Bearing on Reviewer 3

Reviewer 3 argues PPO is *more* stable than SAC on sparse-reward long-horizon
tasks. On this task, under a controlled comparison with three seeds each, the
opposite holds: SAC absorbed every curriculum transition and PPO did not in 2 of
3 runs. This is now a controlled result rather than a single-algorithm
observation, and can be stated directly.

### How PPO is reported

Both checkpoints, per the agreed framing:

* **PPO (final @1M)** — strictly budget-matched, the same "model at 1M steps"
  SAC is reported from. The headline PPO figure.
* **PPO (best ckpt)** — the peak each run reached, selected by the *same* score
  SAC's own training callback uses. Across-seed consistency is striking:
  **0.902 / 0.906 / 0.922**.

The same two-checkpoint treatment is applied to retrained SAC, so the rows are
directly comparable.

**PPO is not a crippled baseline.** It led SAC at matched timesteps early
(−318 vs −623 at 150k), reached 0.91 before collapsing, and its best checkpoint
beats retrained SAC's best (0.910 vs 0.876). Optimisation diagnostics stayed
healthy throughout: explained variance 0.88–0.91, value loss 1.5e4 → 3.0e3,
approx_kl 0.006–0.010, clip_fraction 0.05–0.075, entropy stable at −2.65. No
hyperparameter intervention was made, because the brief specifies intervening
only if training visibly fails — and as an optimisation problem it did not.

---

## 4a. Out-of-distribution generalisation — SAC degrades most

Run under `OOD_PROTOCOL.md`, which was written and committed **before** any OOD
layout was generated. The reporting rule fixed there was that every number
produced would be reported regardless of which method it favoured. It favours
the classical baseline, and it is reported.

Three sets of 100 layouts, generated by the same machinery and A* feasibility
filter as the evaluation suite, from a disjoint seed base. Conditions outside
**both** families' design distributions: obstacle count 5 is outside
`TRAIN_OBS_COUNTS = [0..4]`, and curved paths are outside `PATH_MODE="straight"`.
LOS+APF's parameters were selected on straight-path 0–4 obstacle layouts, so the
shift is equally novel for it. No method was re-tuned for any condition.

Learned methods are represented by their **best** checkpoints (PPO's 1M finals
collapsed in 2 of 3 seeds and would measure the collapse, not generalisation).

### Success rate (mean over runs, min–max across seeds/searches)

| Method | in-distribution | `ood_obs5` | `ood_curve` | `ood_curve_obs5` |
|---|---|---|---|---|
| SAC (published) | 0.950 | 0.490 | 0.790 | 0.430 |
| SAC (retrained, best) | 0.876 | 0.507 (0.32–0.62) | 0.797 (0.76–0.84) | 0.487 (0.34–0.57) |
| PPO (best) | 0.910 | 0.657 (0.61–0.72) | 0.843 (0.80–0.89) | 0.630 (0.55–0.69) |
| **LOS+APF** | 0.960 | **0.717 (0.61–0.78)** | **0.923 (0.88–0.95)** | **0.690 (0.67–0.72)** |

### Degradation from each method's own in-distribution result

| Method | `ood_obs5` | `ood_curve` | `ood_curve_obs5` | mean |
|---|---|---|---|---|
| SAC (published) | −0.460 | −0.160 | −0.520 | **−0.380** |
| SAC (retrained) | −0.369 | −0.079 | −0.389 | −0.279 |
| PPO (best) | −0.253 | −0.067 | −0.280 | −0.200 |
| LOS+APF | −0.243 | −0.037 | −0.270 | **−0.183** |

**The learned policies generalise worse than the tuned classical controller, and
SAC generalises worst of all.** The ordering is the same on all three sets and is
monotone in how far the condition sits from the training distribution.

### Secondary metrics — the mechanism is the same trade-off as in-distribution

| Metric (mean over sets) | SAC pub. | SAC retr. | PPO | LOS+APF |
|---|---|---|---|---|
| RMS cross-track error (m) | **1.071** | 1.250 | 1.553 | 1.644 |
| Min obstacle clearance (m) | 0.169 | 0.290 | 0.351 | **0.486** |
| Mean abs. rudder rate (deg/s) | 61.5 | 49.8 | 12.7 | **10.3** |
| Completion time (s) | **18.5** | 19.4 | 20.8 | 27.3 |

SAC keeps its tracking advantage out of distribution — it is the *most* accurate
path follower on every OOD set — and keeps its clearance disadvantage. That is
the same trade-off measured in §3.2, and it explains the success ordering:
**SAC has learned an aggressive policy that buys tracking accuracy and speed by
passing close to obstacles. At the trained obstacle density that trade is
favourable; as density rises it stops being favourable, and the collisions it
causes dominate.**

The curved-path condition is a mild shift — the environment's `curved_points`
offsets a Bézier control point by ±0.18 × map width, giving a maximum lateral
deviation of 0.89 m (mean 0.45 m) over a ~20 m path — and all methods lose
little there. The obstacle-density shift is what separates them.

### Why LOS+APF holds up

Mechanistic, and predicted in §5 of the protocol before the run: LOS guidance
derives its desired course from the **path tangent**, which the environment
computes from the reference geometry whether straight or curved, so curvature is
nearly transparent to it. Its avoidance layer is a function of sector closeness
with no notion of "how many" obstacles there are, so obstacle count enters only
through the sectors themselves.

One asymmetry to state plainly, because it does not favour SAC: the learned
policies saw thousands of distinct random layouts across 1M steps, while
LOS+APF's 20 parameters were selected against only 100 fixed tuning layouts. The
classical controller had *less* opportunity to overfit to specific geometry. Its
robustness here is a real property of the controller structure, not an artifact
of a larger tuning budget.

### What this does and does not say about the manuscript

It does **not** contradict any result the manuscript reports. Every published
claim concerns 0–4 obstacles on straight paths, and in that range SAC reproduces
at 95.0 % and is statistically tied with the best classical baseline.

It does mean one sentence in the manuscript is currently exposed. The paper
states the policy *"can generalize across different obstacle densities"*. That is
true for the densities evaluated, and false one step beyond them. **Scoping that
claim explicitly to 0–4 obstacles and straight reference paths** costs nothing
and removes the single easiest way for a reviewer to falsify a stated claim —
generating a handful of 5-obstacle layouts is a few minutes' work for anyone with
the code.

---

## 5. What was run

### SAC — published checkpoint

Re-run, not quoted from file: `models/sac_model_1M.zip` over the frozen 500,
27.2 min. Two prior issues resolved:

1. **Checkpoint ambiguity.** Two `sac_model_1M.zip` files exist and are
   *different policies* — all 32 policy tensors differ, max weight delta 0.127.
   `models/sac_model_1M.zip` reproduces the manuscript (0.950 / 0.038 / 0.012);
   the root-level file does not (0.894 / 0.102 / 0.004). Evidence kept at
   `eval_results/baselines/sac_1M_alt/`.
2. **The default output files are stale.** `eval_results/eval_suite/eval_suite_summary.json`
   — what `evaluate_suite.py` writes by default — records 67.6 % from an
   `RPM_STAGE = 4` run. The manuscript's 94 % is `eval_suite_summary_1M.json`.

**SAC reproduces at 95.0 %, not exactly 94.0 %,** with mean episode length 203.9
against 214.3 (−4.9 %). The drift is pre-existing, documented in `src/README.md`,
and reproduces from the original `rl_env.py`, so it is not introduced here.

### SAC — retrained, 3 seeds

`src/train_sac_baseline.py`. 3 × 1M steps, 8 workers, same replicated curriculum
and evaluation grid as PPO. Published SAC hyperparameters throughout
(`MultiInputPolicy`, lr 5e-5, batch 512, gamma 0.99, buffer 1e6, `train_freq` 1,
`gradient_steps` 1, `ent_coef` "auto", net_arch [256,256] ReLU).

| Seed | Wall clock | Best | Final |
|---|---|---|---|
| 0 | 211.2 min | 0.888 | 0.816 |
| 1 | 185.8 min | 0.818 | 0.794 |
| 2 | 227.6 min | 0.922 | 0.838 |
| **Total** | **10.41 h** | | |

These do **not** replace `models/sac_model_1M.zip`, which remains the
manuscript's artifact.

### PPO — 3 seeds

`src/train_ppo_baseline.py`. 3 × 1M steps, 8 workers, 5.04 h total (seed 0 was
slow at 169.6 min because the tuning search ran concurrently; seeds 1 and 2 took
68.4 and 64.4 min).

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
counterpart; PPO has separate policy and value heads against SAC's actor and twin
critics. **This is deliberately not the PPO config in `src/train.py`**
([64,64] Tanh, gamma 0.999, ent_coef 0.03), whose network is 16x smaller than
SAC's — see `BASELINES_NOTES.md` §7.

### Curriculum replication

**No scheduler exists in the repository**; the published SAC run advanced stages
by hand-editing `config.py` and resuming. `src/curriculum.py` replicates the
documented schedule explicitly, on `model.num_timesteps` (total environment
interactions). Verified to fire at exactly 700,000 / 800,000 / 900,000 in all six
training runs, and to reach the subprocess workers — observed RPM hits each stage
ceiling exactly (12.0 fixed → 15 → 16 → 18 → 24).

### LOS+APF — 3 tuning searches

`src/baselines/los_apf.py`, tuned by `src/tune_los_apf.py`.

* **3 independent random searches × 250 configurations × 100 tuning layouts =
  75,000 episodes.** Seeds 20240818 / 20240819 / 20240820.
* **20 parameters**, ranges in `SEARCH_SPACE`. Random rather than grid: at equal
  budget it covers 20 dimensions far better.
* **Tuning used `eval_layouts/tune_layouts_v1.json` only** — 100 layouts from the
  same generator as the evaluation set but a disjoint seed base (5,000,000+
  against 675,974–1,076,073), verified disjoint on seeds and case ids. The frozen
  500 was never used for tuning; `tune_los_apf.py` refuses to run against it.
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
500. That is the documented answer to the under-tuning objection, and it no
longer depends on one lucky search.

Full records: `apf_tuning_results{,_s2,_s3}.csv`,
`los_apf_best{,_s2,_s3}.json`.

**Fairness.** The controller consumes the 34-dimensional observation and nothing
else — no `env.obstacles`, no `map_border`, no world pose.
`src/verify_los_apf.py` asserts this with an AST scan of the controller source
alongside 18 sign-convention and behaviour checks; all 21 pass.

It *does* use `front_clearance`, `side_clearance_diff` and `local_target_cte`,
because those are components of the observation the SAC policy receives.
Withholding them would handicap the baseline, not make it fair. This also
corrects the brief's premise that both methods are equally boundary-blind:
`side_clearance_diff` is computed against a wall-only LiDAR and carries boundary
information to *both* methods (`BASELINES_NOTES.md` §1.1).

---

## 6. Determinism and reproducibility

* SAC over 10 frozen layouts, run twice, comparing full per-step cross-track and
  rudder traces at 1e−12: **10/10 bit-identical**. The replay path consumes zero
  random numbers, so no torch seeding is needed.
* `evaluate.py --check-workers` verifies that splitting a layout set across
  worker processes changes nothing.
* `env.py` was **not modified in any way**. `hull_polygon()` and `obstacles` are
  already public, so exact footprint-to-surface clearance is computed in the
  harness. No `info` additions were required.
* No SAC artifact was touched. Every training output is scoped to
  `models/{ppo,sac}_seed{N}/`, specifically because the existing callback writes
  `best_model.zip`, `eval_metrics.*`, `eval_summary.*` and `train_monitor.csv` to
  fixed root paths — which is how the 0–1M SAC learning-curve data came to be
  overwritten by a later run (recovered from tfevents, `BASELINES_NOTES.md` §10.1).

Total compute: PPO 5.04 h + SAC 10.41 h + tuning ~4.7 h + evaluations ~1.5 h.

---

## 7. Files

**Code** (all new, all under `src/`):

| File | Purpose |
|---|---|
| `eval_layouts.py` | frozen 500 + disjoint 100-layout tuning set, deterministic replay |
| `metrics.py` | per-episode metrics, exact polygon clearances, IQM, stratified bootstrap |
| `evaluate.py` | unified harness: SB3 models and plain controllers, one interface |
| `compare.py` | paired Wilcoxon + exact McNemar |
| `curriculum.py` | staged propulsion curriculum on total env steps |
| `train_ppo_baseline.py` | PPO baseline, run-scoped outputs |
| `train_sac_baseline.py` | SAC retrain across seeds, same protocol |
| `baselines/los_apf.py` | LOS + PID + APF controller |
| `tune_los_apf.py` | 250-configuration random search |
| `verify_los_apf.py` | 21 sign-convention and fairness checks |
| `make_outputs.py` | tables and figures |

**Results** (`eval_results/baselines/`): `sac_1M/`, `sac_1M_alt/`,
`sac_seed{0,1,2}_{best,final}/`, `ppo_seed{0,1,2}_{best,final}/`,
`los_apf_s{1,2,3}/` — each with `episodes.csv` + `summary.json`;
`apf_tuning_results{,_s2,_s3}.csv`; `los_apf_best{,_s2,_s3}.json`;
`comparison_table.{csv,md}`; `paired_stats_table.{csv,md}`; `paired_stats.json`;
`figures/learning_curves.{png,svg,csv}`; `figures/trajectories.{png,svg}`.

Layout sets: `eval_layouts/eval_layouts_v1.json` (500),
`eval_layouts/tune_layouts_v1.json` (100).

Reproduce:

```bash
python src/eval_layouts.py --build --check
python src/verify_los_apf.py
python src/train_ppo_baseline.py --seeds 0 1 2
python src/train_sac_baseline.py --seeds 0 1 2
python src/tune_los_apf.py --n-configs 250 --workers 4 --seed 20240818
python src/evaluate.py --controller sb3:sac:models/sac_model_1M.zip --tag sac_1M --workers 6
python src/make_outputs.py --all
```

---

## 8. Open items

* **Table 4's border clearance needs correcting.** The measured all-wall minimum
  is ~0.99 m and is an artifact of the start pose; lateral-only is ~2.5 m.
  Neither is the reported 2.00 m.
* **The SAC row of any comparison table should be the re-run** (0.950), with the
  ~1 pp delta from the published figure stated rather than absorbed.
* **`r_slow` and `r_thrust` are worth revisiting** independently of this
  comparison. Every widening of the speed curriculum weakens the only reward
  term discouraging the low-RPM regime, and the anti-stall threshold sits below
  the speed reachable at the lowest RPM of any stage. PPO found this; SAC did
  not; a future algorithm might.
* **1M steps from scratch does not reproduce the published SAC policy**
  (0.816–0.888 against 0.950). The published result depends on its particular
  staged, resumed training history. Worth a sentence in the revision, and worth
  archiving the exact training recipe.
* **The root-level `sac_model_1M.zip` duplicate should be removed or renamed**
  before release — it is a different, weaker policy.
