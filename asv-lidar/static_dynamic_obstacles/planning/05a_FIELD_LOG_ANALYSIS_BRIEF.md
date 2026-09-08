# 05a — Field Log Analysis Brief

**For:** Claude Code, working on `field_deployment/` (30 logs, 5597 pooled scans, 130 KB–2 MB each).
**Delivers:** T1 in `02b_DECISIONS_AND_TASK_ORDER.md §4`, plus five extractions beyond it.
**No basin time required.** Everything here comes from logs already on disk.

---

## 0. What the output is

Not a list of numbers. Two artefacts:

1. **An archived identification dataset** — versioned, hashed, with a schema and the script
   that regenerates every figure. This is the direct answer to the Paper 2 Reviewer 1.4
   concession, where the response had to admit the identification data was not archived in a
   form supporting independent reporting. The dataset is the deliverable; the constants are a
   by-product.
2. **A distribution for every constant, not a point estimate.** Each extracted value needs a
   confidence interval, because the CI *is* the domain-randomisation range in `03 §7`. A
   constant reported without one cannot be randomised over, and system identification without
   randomisation reduces bias but creates no robustness.

Write results to `field_analysis/` with one notebook or script per extraction, and a single
`IDENTIFICATION_DATASET.md` collecting the constants, their CIs, and the figure that
justifies each.

---

## 1. Step 0 — inventory the channels before anything else

Several extractions below assume channels that may not exist. Establish first, and report the
answer before proceeding:

| Channel | Needed for | If absent |
|---|---|---|
| Commanded RPM | E1, E8 | E1 falls back to §2.3 methods only; say so rather than substituting a guess |
| Commanded rudder | E8 | Drop E8, report it |
| rf2o pose (x, y, ψ) | E1, E2, E7 | Blocking — flag immediately |
| Per-scan timestamp | everything | Blocking |
| Per-beam timestamp / `time_increment` | E6 | Already known absent; confirm |
| Run metadata (start/goal, path definition) | E1 fallback (c) | Reduces E1 to relative methods |

Also record, per log: duration, scan count, whether the vessel was under closed-loop control
or manual, and whether the run terminated normally. Runs that ended in a collision or a
manual takeover should be tagged, not silently pooled.

**Report the quantisation.** Ranges are logged as integer decimetres (values 10–153). Uniform
quantisation at 0.1 m has standard deviation `q/√12 = 0.029 m`. That is a **hard floor on
every range-noise figure in this analysis** — `KF_MEAS_NOISE_POS = 0.05 m` is close enough to
it that the logs cannot separate sensor noise from quantisation. Report measured range noise
explicitly as an *upper bound* of sensor + quantisation, and add a note for the next session:
log raw millimetre ranges if the driver exposes them, because this analysis threw away most of
the C1's native precision.

---

## 2. E1 — Speed calibration *(highest value; settles `02b` C2)*

The simulator produces 1.77 m/s at `CRUISE_RPM = 12`, which is Froude 0.45 for a 1.57 m LBP
hull — semi-planing, and implausible for a 64.55 kg displacement model. The field figure of
record is 0.55 m/s (Fr 0.14). **Expect the answer to land in 0.4–1.0 m/s** (Fr 0.10–0.25). A
result outside that band means the method is wrong, not the vessel.

### 2.1 The trap — read this before writing any code

**Do not take surge speed from rf2o's x-velocity.** From mid-basin only the two long parallel
walls are in range; the end walls sit beyond the sensor horizon. Two parallel walls constrain
lateral position and heading and leave **along-track motion completely unobservable**. Scan
matching in that geometry will happily report a confident, smooth, and wrong surge velocity —
it is the degenerate direction, and it is exactly the direction E1 needs.

Lateral and heading estimates from rf2o are well conditioned and can be trusted. Surge cannot.
Use the three methods below instead, and cross-check them against each other.

### 2.2 Method (a) — feature transit timing *(primary)*

Find scans containing a distinctive fixed feature: a recessed doorway, a protruding bench, a
ladder, a corner. Track its bearing and range across consecutive scans. For a feature at
lateral offset `d` passing from bearing `+θ` to `−θ`, the along-track distance covered is
`2·d·tan θ`, and speed is that over the elapsed time.

Prefer features at short range and large lateral offset — the bearing sweeps fastest and the
geometry is best conditioned. Report per-transit speed estimates, not a pooled mean, so the
spread is visible.

### 2.3 Methods (b) and (c) — cross-checks

**(b) End-wall closing rate.** Find segments where a return appears near bearing 0° with a
monotonically decreasing range. `du/dt` of that range *is* surge speed directly, with no
scan-matching involved. These segments will be short and rare but they are the cleanest
measurement in the whole dataset — the estimator is a first difference of a directly sensed
quantity.

**(c) Gross transit.** Known start-to-goal distance divided by run duration. Coarse, biased
low by acceleration and turning, but it cannot be wrong by a factor of three, so it is the
sanity check that catches a broken pipeline.

### 2.4 Output

- `U_REF` with CI, at the commanded RPM in force during the runs
- `THRUST_CAL`: the scalar on the thrust map making simulator steady speed at `CRUISE_RPM`
  equal `U_REF`
- The three methods' estimates side by side. **If they disagree by more than ~20%, report the
  disagreement rather than averaging it** — a discrepancy between feature transit and end-wall
  closing would point at a timestamp problem, which matters more than the speed does
- Whether RPM was fixed across all runs, or varied

---

## 3. E2 — rf2o pose drift *(the `CONSTANTS §6` zeros)*

Currently `BOUNDARY_POSE_NOISE_XY`, `_HEADING_DEG` and `_WALK` are all 0.0, which leaves the
`01 §3.3` sim-to-real gap wide open. No headline training run should start until these have
numbers.

### 3.1 Method — wall-fit residual in the world frame

The two long walls are straight, parallel and fixed. That is ground truth, free, in every scan.

1. Segment the wall returns in each scan (RANSAC line fit in body frame, two dominant lines).
2. Transform each fitted line into the rf2o world frame using the estimated pose at that scan.
3. A fixed wall that appears to **translate** in the world frame is lateral pose drift. One
   that appears to **rotate** is heading drift. Plot both against time.

This needs no external ground truth and it separates the two error components cleanly, which
a return-to-start check cannot.

**Along-track drift is invisible to this method** — parallel walls again. Do not report a
number for it. Report instead that it cannot be bounded from these logs, which is precisely the
justification for the IMU (`05 §4.7`) and for registering against distinctive sparse features
rather than the wall polygon. A stated unmeasurable is worth more here than an estimate that
would be silently wrong.

### 3.2 Separating white noise from random walk

Do not fit a single number. Run an **Allan-variance decomposition** on the pose increments:
plot increment variance against averaging window `τ` on log axes. White noise falls as `1/τ`,
random walk rises as `τ`, and the minimum separates them. That gives `BOUNDARY_POSE_NOISE_XY`
and `_HEADING_DEG` (white) and `_WALK` (random walk) as distinct fitted quantities rather than
one number split by guesswork.

Do this per log and pool, so between-run variability is visible — that spread is the
randomisation range.

### 3.3 Condition on yaw rate

Scan matching degrades during turns. Report drift separately for straight segments and for
`|r|` above a threshold. If turning drift is materially worse, `03 §7`'s randomisation should
be yaw-rate-dependent rather than constant, which is a more faithful model and a reportable
finding in its own right.

---

## 4. E3 — Black-wall return rate *(the `05` first action)*

Concern: the climbing-wall side is matte black, and the RPLidar C1 near 905 nm may get
single-digit reflectivity from carbon-based black finishes. If it does, the lateral wall-to-wall
constraint that scan-to-map localisation depends on is destroyed on one side.

### 4.1 Method — matched-bin comparison, not a bearing histogram

**The confounder that will ruin this if ignored:** return rate depends strongly on range and on
incidence angle, independent of reflectivity. A raw return-rate-versus-bearing plot will show
structure caused by geometry and tell you nothing about the paint.

So:

1. Assign each return to a wall using the estimated pose (which wall is the climbing-wall side
   is fixed by basin geometry; establish it once from run metadata or a site photo).
2. Compute, for each wall, the return rate as a function of **(range bin, incidence-angle bin)**.
3. Compare the two walls **only in bins both populate**. Report the ratio per bin and pooled.
4. Report **range noise** per wall as well — residual to the fitted line, remembering the
   0.029 m quantisation floor from §1. Low-reflectivity surfaces give noisier ranges as well as
   more dropouts, and noise may be the more sensitive indicator here.

### 4.2 What the answer decides

- **Materially lower return rate on the black wall** → scan-to-map must register against a
  sparse landmark set (white signage, climbing net and holds, aluminium ladders, the top
  structural rail) rather than a continuous wall polygon. Survey those features specifically.
- **No material difference** → continuous-wall registration stands, and the risk closes.

Either result is publishable as sensor characterisation. Report the negative as confidently as
the positive.

---

## 5. E4 — Model the no-return process properly

A mean of 506 of 720 bins carry a return, with 96.5% of empty bins in contiguous runs longer
than 3 bins. So they are no-return arcs, not angular under-sampling — that is settled.

What is not settled is the model. **`LIDAR_DROPOUT_P` as a scalar is probably the wrong
parameterisation.** Fit instead:

```
p(no return | range, incidence angle, surface)
```

and report the fitted surface. Also fit the **run-length distribution** of contiguous
no-return arcs, because the simulator needs to reproduce arcs, not independent per-beam
dropouts — independent dropout at the same marginal rate produces a completely different
pooled `c_t` and would make Study 2's dropout axis unrepresentative.

Output: `LIDAR_DROPOUT_P` (marginal, for reference), the conditional model, and the arc
run-length distribution. State which one the simulator should use.

---

## 6. E5 and E6 — confirm two negatives, then stop

**E5, aft self-occlusion.** Already assessed: no bin is zero in more than 98% of scans in any
log, and the peak zero-rate bearing wanders between 108° and 359°, so it tracks the scene
rather than the mount. Confirm with one additional test and then stop: look for a **body-frame**
bearing sector that is consistently return-free across *all thirty* logs. Self-occlusion by a
mast or superstructure would also show as returns below `LIDAR_MIN_RANGE`, reported as
no-return, at a fixed body bearing. If that sector does not exist, the negative is solid and the
static-spin recording stays scheduled for the next session — ten minutes, at the top of the
list, because it gates the being-overtaken class and hence the whole Rule 17 contribution.

**E6, motion distortion.** Not assessable: one wall-clock stamp per revolution, no per-beam
times. Confirm and do not work around it.

But do produce the **analytic bound**, which is free: from the rf2o yaw-rate series, the smear
at a 10 Hz scan rate is `r × 0.1` radians. At 0.2 rad/s that is 1.15°, about 2.3 beams. Report
the distribution of implied smear across all logs. That bounds the effect without per-beam
times and tells you whether the missing measurement matters at all.

---

## 7. E7 — Constants that fall out of E2 for free

Once pose drift is characterised, three more constants follow without new analysis:

| Constant | Derivation |
|---|---|
| `DYNAMIC_SPEED_ON` / `_OFF` | The dominant source of apparent motion in a static object **is** ego-pose error, which affects every object in the scan identically. So the apparent velocity a fixed object acquires equals the pose drift rate. Set `_ON` above the **99th percentile** of measured drift rate, not the 95th — bias toward under-detection, because promoting a static panel to a target ship is a false positive with COLREGs consequences |
| `EGO_YAW_RATE_NOISE_DPS`, and `r_dead` in `02a §6.2` | Paper 2 had no IMU, so `r` came from differentiating rf2o heading. The std of `r` over known-straight segments is therefore a **conservative upper bound** on the IMU's noise floor — usable as a placeholder until the gyro lands, and honest about being an upper bound |
| `KF_MEAS_NOISE_POS` | From E3's range-noise figure, floored at the 0.029 m quantisation limit |

Also worth computing: the drift rate over a typical encounter duration (~100 steps, 10 s). If
accumulated drift over that window approaches `d_req = 2.36 m`, the DCPA estimate driving every
COLREGs term is unreliable at exactly the timescale that matters, and Study 2's pose-drift axis
becomes the headline rather than a robustness curve.

---

## 8. E8 — Log-replay validation *(free figure, if commands were logged)*

Replay the commanded rudder and RPM series through the recalibrated model and overlay the
predicted trajectory on the rf2o trajectory. Costs no basin time and produces a validation
figure directly.

Two cautions. Compare **heading and lateral position**, which rf2o constrains, and treat
along-track agreement as uninformative for the reason in §2.1. And fit to heading directly
rather than to differentiated yaw rate — differentiation amplifies the quantisation noise that
§1 flags. Cross-check by fitting to yaw rate as well: **disagreement between the two fits
indicates a timestamp or mounting error**, not a hydrodynamic one, and that is worth catching
before the basin session rather than after.

---

## 9. Reporting rules

- **Report negatives as findings.** "Along-track drift cannot be bounded from these logs" and
  "no aft occlusion arc is detectable" are results. They justify the IMU and the static spin
  respectively, and a stated unmeasurable is worth more than a confident wrong number.
- **Never pool across logs without showing the spread.** Between-run variability is the
  randomisation range.
- **Flag any result outside its plausibility band** rather than adopting it. E1 outside
  0.4–1.0 m/s means the method is broken.
- **Do not substitute a guess for a missing channel.** If commanded RPM is absent, say E1 ran
  on §2.3 methods only.
- Every constant lands in `src/constants.py` with its CI in the adjacent comment and its
  figure referenced by name.

## 10. Do not

- Take surge speed from rf2o (§2.1). This is the one that will silently produce a plausible,
  confident, wrong `U_REF` and propagate it through every speed gate in the reward.
- Fit a single scalar to pose drift. White noise and random walk are different constants with
  different consequences and the Allan decomposition separates them cheaply.
- Plot return rate against bearing and conclude anything about reflectivity without matching on
  range and incidence.
- Model dropout as independent per-beam. The arcs are the phenomenon.
- Report range noise below 0.029 m. That is the quantisation floor, not the sensor.
