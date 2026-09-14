# Review of `REPORT.md` — 05 part 1, vessel model identification

An independent check of the identification report against its own code, its
own `validation.json`, and the raw July logs. Written by Claude Code on
2026-09-13. `REPORT.md` is left unedited; corrections are listed in §5 for
whoever owns it.

**Verdict.** The model is sound and better than v2 on every horizon, and the
simulator-readiness work is real: the acceptance suite reproduces exactly. But
**five statements in the report are contradicted by its own numbers or by the
logs**, one training-data problem went undetected, and the headline holdout
result is narrower than the prose says. None of it blocks using the model;
several items should be fixed before any of it is quoted in the paper.

---

## 1. Method

| What | How |
|---|---|
| Acceptance suite | re-ran `acceptance.py` unmodified (10.5 min) |
| Holdout tables | compared every cell of §6 against `validation.json` |
| Deployment parity | parsed all 19 July logs directly: `#CONFIG`, `#ACTION`, raw pose and LiDAR lines |
| Pose staleness | matched each `#ACTION` to its own frame's pose line by timestamp, and recorded which arrived first in the file |
| Drift anomaly | recomputed course − heading per segment with the report's own 0.5 m/s gate, then inspected step lengths, run lengths and snap-back |
| Parameters | position of every fitted value inside its optimiser bound |
| Surge physics | steady speed, coast-down, stopping distance from the fitted `T12`, `X_uu`, `M11` |

---

## 2. Confirmed

| Claim | Evidence |
|---|---|
| **7/7 acceptance tests pass** | reproduced exactly, including A2's 2.2/5.0/7.7/10.0 °/s and A3's 0.235 |
| Control and telemetry at 2 Hz | `t_frame` 0, 0.497, 0.997 s; `dt_cmd` 0.4999 |
| `$CMD` carries `limited_rudder` | log 07-03/1, frame 1: `raw_rudder` −64.99, `limited_rudder` −5.00 at `dt_cmd` 0.1 — the 50 %/s limiter visible on the first step |
| Sign chain `raw = −100·a0` | `a0` +0.6499 → `raw_rudder` −64.988 |
| RPM 12 ↔ S2 50 | `rpm=12.000,S2=50.000`; bridge maps `S2 = rpm/24·100` |
| §6 V1 holdout table | every cell matches `validation.json` to the printed precision |
| v3 beats v2 at every horizon | heading and position, all six horizons |
| V3: v3 better on 6/6 holdout runs | confirmed — see §4 for how much |
| Turning is prior-driven, not identified (§7) | the report is explicit and correct about this |

**One result the report does not claim, and should.** The fitted surge balance
gives a steady speed at 12 rpm-units of **`sqrt(T12/X_uu)` = 1.116 m/s**. The
02b T1 log-mining task, which never saw the fit, measured a speed-over-ground
median of **1.14 m/s** across 18 runs. Two independent methods agree to 2 %.
That settles what the vessel does at 12 RPM, and it bears directly on F24: the
"0.55 m/s field measurement" cited in 02b C2 and 03a §1.1 is not a measurement,
and 0.55 m/s corresponds to 6 rpm-units on this plant (0.558 m/s).

---

## 3. Contradicted

### 3.1 "Only at 1 s does a naive predictor beat v3" — also at 2 s

From `validation.json`, holdout V1, heading RMSE:

| horizon | freeze | const. rate | v2 | v3 | best |
|---|---|---|---|---|---|
| 1 s | 4.10 | **1.22** | 2.43 | 1.31 | const. rate |
| 2 s | 9.01 | **4.98** | 5.90 | 5.19 | **const. rate** |
| 3 s | 12.87 | 10.04 | 9.40 | **7.90** | v3 |
| 5 s | 20.69 | 22.93 | 12.29 | **10.17** | v3 |
| 10 s | 20.13 | 31.43 | 19.80 | **13.82** | v3 |
| free run | **22.94** | 30.32 | 33.17 | 23.53 | **freeze** |

### 3.2 v3's free-run heading is worse than assuming the vessel never turns

§6 observes that v2's free-run error (33.2°) "is worse than assuming the vessel
never turns (22.9°)". **So is v3's**: 23.53° against 22.94°. The report states
the failure only for v2. v3 beats every naive baseline at 3, 5 and 10 s, and at
no other horizon.

### 3.3 "The V6 regression reported in the previous iteration is gone" — it flipped

Drift-angle SD on the holdout: measured **5.68°**, v2 5.39°, v3 **4.78°**. v3 is
0.89° from the measurement, v2 is 0.28°. v3 is bolded as the better model in the
§6 table; by this metric it is the worse one. The previous iteration
over-drifted (IQR 14° against 8.5°); this one under-drifts. The regression
changed sign rather than disappearing.

### 3.4 "Controller pose is one frame old" — it is on 41 % of frames

§9's parity table, and `VesselSim(obs_delay_steps=1)`, model the controller's
pose as always one frame stale. Matching each `#ACTION` to its own frame's pose
line across all 1,085 July frames:

| | frames | share |
|---|---|---|
| pose line arrived before the action (current) | 642 | 59.2 % |
| pose line arrived after the action (stale) | 443 | **40.8 %** |

Consistent across runs (28.6–56.8 %). The pose line races the LiDAR line at the
bridge, and `x_real/y_real/yaw_real` latch whichever arrived last. Always-stale
overstates the delay about 2.4×. Faithful parity is staleness with
p ≈ 0.41 per frame — and, better, a bridge that waits for the frame's pose line
before computing the action, which removes the staleness at no cost.

### 3.5 The training-day drift SD is not a low-speed artefact

`simulate.drift_iqr` attributes the 36° training-day SD (against 6° on the
holdout) to "low-speed wrap-around". The outliers survive the same 0.5 m/s gate:

| training segment | moving samples | drift SD | IQR | \|drift\| > 90° |
|---|---|---|---|---|
| `trial` | 36 | 86.8° | 171.3° | **30.6 %** |
| `trial_2` | 31 | 61.6° | 8.4° | **16.1 %** |
| `calibration#1` | 28 | 61.9° | 11.1° | **14.3 %** |
| the other 11 | 25–40 | 4.5–7.3° | 7–9° | 0 % |

**They are not localisation jumps.** The bad samples come in sustained runs (up
to 10 consecutive steps in `trial`), never snap back (0–9 %), and are *shorter*
than normal steps (0.30–0.36 m per 0.5 s against 0.50–0.61 m). In `trial` from
t = 18.5 to 22.5 s the pose moves steadily on a course of 124–135° while heading
holds at −59°: a clean ~180° disagreement at ~0.65 m/s for four seconds. All
three windows sit at t ≈ 18–24 s, the end of each run.

That is **sustained motion against the heading** — most consistent with an
un-trimmed end-of-run retrieval (the vessel hauled or drifting back on a
constant heading), or a sustained 180° yaw ambiguity in scan matching.
`calibration.mp4` and `trial_2.mp4` at t ≈ 20 s would settle which. The bridge
clips S2 at zero, so the vessel cannot have reversed under its own power.

§3's "no localisation jumps" is not contradicted — these are not jumps. §3's
"heading is consistent with direction of travel" does not hold for these
windows. **All three segments are in `train_runs`**, so the replay fit
integrates forward commands through windows where the vessel moved backwards,
residuals no hull parameter can absorb. How much that biased the yaw and sway
parameters — seven of which sit at their bounds (§4) — is unknown until the fit
is re-run with the windows trimmed.

### 3.6 `ship_model_v3.py`: "`rud_tau` sits at its lower bound"

Fitted `rud_tau` = 0.895 s in a bound of [0.01, 1.2] — 74 % of the way up. The
docstring is stale from an earlier fit.

---

## 4. Qualified

**V3's p = 0.016 is one-sided.** 1/64 is the exact one-sided Wilcoxon value with
all six differences positive; two-sided is 0.031. Still below 0.05 either way,
but it should say which. Two of the six wins are practical ties: runs 2 and 5
improve by 0.15° and 0.16° on RMSEs of about 13° and 6.6°. Honest phrasing is
four clear wins and two ties.

**V5's −0.79 m benefits from sign cancellation.** Run 1 is +2.91 m and the other
five are negative. Mean-absolute path-length error is **1.76 m** against v2's
3.89 m — still a clear improvement, but 2.2× the signed figure. Acceptance test
A6 guards `abs(mean(...))`, the cancellation-prone form.

**Seven of fifteen free parameters sit within 3 % of a bound**: `X_rr`,
`X_delta`, `Y_v`, `k_race`, `N_r`, `N_rr` at their lower bounds, `rud_rate` at
its upper. §8 discusses the wide intervals and the `rud_rate` saturation. It
does not mention **`N_uv` = −84.0 against a lower bound of −90** (1.8 % of the
range), with its 5 % interval at −86.8. `N_uv` belongs to the correlated yaw trio
§8 warns about, and its drift moment `−N_uv·u·v = +84·u·v` more than cancels the
Munk moment `(M11 − M22)·u·v = −59.1·u·v`. With the bound active, the bound is
shaping the solution. Worth refitting with it widened to see whether `N_uv`, `N_r`
and `k_R` move.

---

## 5. Corrections for `REPORT.md`

1. §6: "Only at 1 s does a naive predictor beat v3" → "at 1 s and 2 s"
2. §6: state that v3's free-run heading error (23.5°) is also worse than the
   freeze-heading baseline (22.9°)
3. §6: V6 — v3 under-drifts (4.78° vs measured 5.68°) and is further from the
   measurement than v2; the regression changed sign, it did not go away
4. §9 parity table: staleness is 40.8 %, not every frame; change
   `obs_delay_steps=1` to a per-frame probability of 0.41
5. §3 and `simulate.py`: the training-day drift outliers are not low-speed
   wrap-around; three training segments contain sustained motion against the
   heading at t ≈ 18–24 s
6. §6: V3 p-value is one-sided; two of the six wins are ties
7. §6: report V5 as mean-absolute error (1.76 m vs 3.89 m)
8. §8: note `N_uv` against its bound
9. `ship_model_v3.py` docstring: `rud_tau` is not at its lower bound

---

## 6. Recommendations

In priority order:

1. **Re-run the fit with the three retrieval windows trimmed** and compare the
   parameters and the holdout metrics. It is the cheapest test of how much
   un-self-propelled motion in the training set moved the yaw and sway
   subsystem. Check the two videos first.
2. **Widen the `N_uv` bound** in the same refit.
3. **Fix the bridge race**: wait for the frame's pose line before computing the
   action. Removes 41 % staleness from deployment for free.
4. **Add a crash-stop block to basin session 1** (`PART2_BASIN_PLAN.md` has none):
   full astern from settled speed at RPM {9, 12, 15}, three runs each, with
   thrust command logged at the highest available rate. It measures the two
   numbers the emergency stop's field viability rests on — reverse thrust
   efficiency and thrust transport delay — and nothing else in either session
   does.

---

## 7. What was integrated into the simulator

| Item | How | Default |
|---|---|---|
| Identified hull | `src/ship.py` **imports** `bluefin/` rather than copying it; forward trajectories asserted bit-identical to `ship_model_v3.ShipModel` | on |
| Substepping `≤ 0.05 s` | inside `ShipModel.update` | on |
| `U_REF` from the plant | `steady_speed(CRUISE_RPM)` = 1.116 m/s; replaces `THRUST_CAL` | on |
| Bridge rudder limiter, 50 %/s | the same true rate limit in `env.step` and the bridge; **off by default in both** — v3 predicts the eight July runs that sent raw commands at least as well as the limited ones (holdout heading 9.8° against 15.1° at 10 s) | off; `command_rate_limit=True` with `--rudder-limit` |
| Domain randomisation | `sample_params` bootstrap blends, seeded on a separate stream | off; `vessel_randomisation=<scale>` |
| 2 Hz decisions + 41 % staleness | **native to the environment since revision 7**: `UPDATE_RATE = 0.5`, physics and collision sub-stepped at 0.1 s, and a stale frame repeats every pose-derived observation and skips the tracker (the `DeploymentTiming` wrapper is gone) | on; staleness **0.0 since the bridge waits for each pose line**, 0.408 available |
| Reverse thrust | braking force by operator splitting, `REVERSE_THRUST_EFFICIENCY` = 0.5 (`TODO(05)`); the astern impulse the clip at zero discards is reported as `estop/reverse_dv_est_mps` | used only by the emergency stop |
| Emergency stop | `src/emergency_stop.py`, env-free; **the bridge now imports it** (revision 7) | on; trigger default in `PROJECT_STATE.md` revision 7 |
| Refit v4 (§8) | `refit_v4/refit_v4.py`, pre-registered | **not adopted** — the simulator keeps v3 |

Not integrated: `VesselSim` itself (the environment already owns stepping,
observation and actuation, and wrapping a second stepper inside it would
duplicate all three), `obs_delay_steps=1` (contradicted, §3.4), and the v2
MMG derivative block (inert, as §1.3 correctly says).

---

## 8. Revision 7 — the retrievals confirmed, and the refit

### 8.1 The three windows are retrievals

§3.5 left it at "most consistent with an un-trimmed retrieval" and pointed at the
videos. The July `.mp4` files turn out to be recordings of the **bridge's own
display**, not a camera, so they cannot show the vessel. They show its telemetry,
and that settles it anyway. At t = 21.0 s of `trial_2` the display reads surge
**u = −0.60 m/s** at heading +3° (course error −180°) while the bridge is
commanding **S2 = 62.5** — forward thrust at 15 rpm — and every forward LiDAR
sector reads 1.2–2.3 m. A vessel pushing ahead while travelling astern at
0.6 m/s is being hauled back. The 180° scan-matching alternative is not needed.

### 8.2 Refit v4

`refit_v4/refit_v4.py`. **Pre-registered in its docstring before it ran:** the
trim rule, `N_uv` widened to [−250, 250], `fit_final.py`'s pipeline otherwise
unchanged (pooled objective, three starts, 28 warm-started bootstrap
resamples), and the adoption rule — no holdout metric more than 5 % worse than
v3's, and acceptance tests A1–A5 passing on the new parameters and bootstrap.

**The trim rule** — three consecutive pose steps of at least 0.15 m whose course
is more than 90° off the heading — fired on exactly the three segments §3.5
found, all at t = 18.5 s, and on nothing in the holdout. Removing those 41
samples lowers v3's own training objective from 1.041 to 0.905.

**Holdout** (untrimmed 2026-07-03, the same six runs `REPORT.md` validates on),
heading RMSE in degrees:

| horizon | naive freeze | v3 | **v4** |
|---|---|---|---|
| 1 s | 4.10 | **1.31** | 1.45 |
| 2 s | 9.01 | 5.19 | **4.21** |
| 3 s | 12.87 | 7.90 | **6.75** |
| 5 s | 20.69 | 10.17 | **7.49** |
| 10 s | 20.13 | 13.82 | **11.82** |
| free run | **22.94** | 23.53 | 24.12 |

| | v3 | **v4** | measured |
|---|---|---|---|
| position RMSE, 10 s / free run | 1.25 / 3.53 m | **0.86 / 3.46 m** | — |
| free-run path length, mean absolute error | 1.76 m | **1.25 m** | — |
| turn-rate KS distance | 0.171 | 0.171 | — |
| drift-angle SD | 4.78° | 5.05° | 5.68° |

v4 is better than v3 at every windowed horizon from 2 s and in position, and at
2 s it now beats the constant-rate predictor (4.98°) as well. Free-run heading
is still worse than assuming the vessel never turns (§3.2 stands).

**The parameters moved wholesale.** `N_uv` left its bound as §4 suspected
(−84.0 → −45.5), but the yaw subsystem re-routed rather than settled: `N_r`
1.04 → 17.9, `k_R` 2.18 → 2.96, `N_rr` 0.017 → **59.6 against an upper bound of
60**, `Y_vv` 221 → 108, `X_vv` 50.6 → 59.7 (bound 60), and the actuator traded
lag for delay — `rud_tau` 0.895 → **0.010** (lower bound), `rud_delay`
0.73 → 1.12 s. Six parameters sit within 3 % of a bound.

**Verdict: not adopted.** All five holdout criteria pass; **A5 does not** — 11 of
12 domain-randomisation draws are usable. The simulator keeps v3.

**What it shows** is more useful than the parameters. Removing 41 samples of
hauled motion moved the yaw and actuator parameters by more than their own
bootstrap intervals, so the identification is not robust to data selection —
`REPORT.md` §8's "do not quote their individual values" is, if anything, too
mild. The lag-for-delay trade is exactly what S1-B's servo steps separate, and
the `N_r`/`N_rr`/`k_R` re-routing is what S1-D's turning circles constrain.

**A discrepancy found on the way.** `fit_final.py`'s docstring says "`N_rr` fixed
at zero" as unidentifiable against `N_r`; its `BOUNDS` allow [0, 60]. v3 happened
to land at 0.017. v4 goes to the bound.

### 8.3 Variant v4b — post hoc

Fixes `N_rr` at zero, as `fit_final.py` documents; everything else as v4,
including the adoption rule. **Post hoc** -- run because v4 failed -- so it is
weaker evidence than v4 whatever it shows.

| | v3 | v4 | **v4b** |
|---|---|---|---|
| heading RMSE 2 / 5 / 10 s | 5.19 / 10.17 / 13.82° | 4.21 / 7.49 / 11.82° | **3.80 / 6.75 / 9.33°** |
| heading RMSE, free run (naive freeze 22.94°) | 23.53° | 24.12° | **21.14°** |
| position RMSE, 10 s / free run | 1.25 / 3.53 m | 0.86 / 3.46 m | 0.92 / 3.26 m |
| turn-rate KS | 0.171 | 0.171 | **0.152** |
| path length, mean absolute error | 1.76 m | **1.25 m** | 2.41 m |
| parameters at a bound | 7 | 6 | **3** (`Y_v`, `k_race`, `rud_rate`) |
| A5 usable draws | 12/12 | 11/12 | 11/12 |

**The first model of the four to beat the freeze-heading baseline in free run**,
and the best at every heading horizon from 2 s. The actuator went back to v3's
decomposition (`rud_tau` 1.04 s, `rud_delay` 0.725 s), so v4's lag-for-delay
trade was an artefact of `N_rr` at its bound. The yaw damping now sits in `N_r`
(25.9) and `k_R` rose to 5.1, while `N_uv` held at -47 -- off its old bound in
both refits, so REVIEW §4's suspicion about -84 is confirmed.

**Not adopted, on two counts:** path-length error is 37 % worse than v3's,
beyond the 5 % tolerance, and A5 again passes 11 of 12 draws. The surge fit
moved too: `X_uu` 12.9 -> 14.1 puts steady cruise at 1.07 m/s, 6 % below the
logs' 1.14. It trades speed accuracy for heading accuracy, and the adoption rule
was written to catch exactly that trade.

**Recommendation:** keep v3 in the simulator until basin session 1. Take v4b's
structure -- `N_rr` fixed, as documented -- into the session-1 fit, where S1-C's
coast-downs pin `X_uu` independently and S1-D's turning circles pin the yaw
terms. That fit then has a pre-registered structure rather than a post-hoc one.

### 8.4 Recommendation 3 implemented

The bridge now holds each LiDAR frame until its own pose line arrives
(`udp_live_rl.PoseSync`). Replayed through all 19 July logs it releases 1,085
frames with **0 stale**, against the 443 (40.8 %) the unmodified decoder
produces on the same lines -- the §3.4 count, reproduced by a second method --
at 6.3 ms worst added latency. The rudder limiter the report's §9 counts as
part of the deployment plant was, in the bridge as found, overwritten to off.
It is now a true 50 %/s rate limit, **off by default**: §9 treated it as part of
the plant, but 8 of the 19 July runs never had it, and v3 predicts those runs at
least as well as the limited ones, so the identified model needs no stand-in
servo limit.
