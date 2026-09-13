# 05_VESSEL_MODEL_AND_SIM2REAL — Part 1

Vessel model identified from the 2026-07-02 / 2026-07-03 field logs, and the
simulator brought to a state where RL training and evaluation can start.

Fitted on **2026-07-02** (14 runs). Validated on **2026-07-03** (6 runs), held
back until the parameters were frozen.

**Status: 7/7 acceptance tests pass. Ready to train.**

Part 2 (basin validation) is outstanding, and one part of the model — the
sustained-turn regime — is currently prior-driven rather than data-driven. See §7.

---

## 1. Three errors in the existing toolchain

Found while building the replay. Each one alone invalidates an identification
attempt, and all three are silent failures.

**1.1 The transmitted rudder is `limited_rudder`, not `raw_rudder`.**
Each `#ACTION` line carries the literal `$CMD,<rudder>,<thrust>` string. Parsed
directly, the transmitted value equals `limited_rudder` in 100 % of samples
across every July run, and `raw_rudder` in only some runs. `udp_live_rl.py`
contains a line that disables the rate limiter
(`rudder_cmd = float(raw_rudder_cmd)`) and the logs span versions with and
without it active. The rate limiter was live for most runs, so the vessel
received a command ramped at 50 %/s, not the bang-bang policy output. Using the
wrong field drops the correlation between effective rudder and measured yaw rate
from **0.67 to 0.30**.

**1.2 Replaying a logged command needs the sign undone.**

```
policy action a0 --(rudder_scale)--> simulator rudder percent = +100*a0
policy action a0 --(rudder_sign=-1)-> transmitted $CMD percent = -100*a0
```

so `rud_sim = -cmd_transmitted`. With the sign the wrong way the model turns
opposite to the vessel and scores **worse than a model that never turns** —
which is what made the first fits collapse onto "don't turn at all". Worth
checking anywhere else in the toolchain that replays logs through the model.

**1.3 The MMG hull coefficients from `Blue02.m` are inert.**
At model scale (U = 1.2 m/s, v' = 0.15, r' = 0.35) the whole nondimensional
block produces about **0.09 N** of sway force and **0.07 N·m** of yaw moment,
against roughly **10 N** and **5 N·m** from the tunable linear terms beside it.
v2's behaviour came entirely from `TURN_COEF`, `LINEAR_*_DAMP` and the rudder
scales. The sign inconsistency between `Blue02.m` and `ship_model.py` is real
but numerically irrelevant. If that derivative table appears in the paper as if
it governs the model, the claim does not survive a reviewer checking magnitudes.

---

## 2. Five structural defects in the v2 model

Each was found by the fit refusing to behave, not by inspection.

| # | Defect | Symptom | Fix |
|---|---|---|---|
| 1 | Damping independent of forward speed (`-N_r·r`, `-Y_v·v`) | every fit drove `N_r` to zero and routed all yaw damping through the one speed-dependent term available | MMG form `-N_r·u·r`, `-Y_v·u·v`; objective improved 1.42 → 1.26 immediately |
| 2 | Cross-flow drag with odd powers (`-X_vv·v·\|v\|`) | becomes a **+116 N thrust** when the vessel drifts to port; surge diverges in a sustained turn at every timestep tested | even powers `-X_vv·v²`, `-X_rr·(L r)²` |
| 3 | Rudder pressure scaled by `(u_R² + v_R²)` | rudder force grows with yaw rate squared — positive feedback reaching ~200 N normal force on a 64 kg vessel | axial inflow only, as `Blue02.m` has it |
| 4 | Missing yaw Coriolis term | `+m_y·v·r` / `−m_x·u·r` is only energy-consistent with the Munk moment `(m_x − m_y)·u·v = −59.1·u·v` | added explicitly |
| 5 | Propeller race computed from `rpm/60` | `rpm` is a command unit on a 0–24 scale, not shaft RPM, so the race gave 0.07 m/s and the rudder lost all authority as the vessel slowed | `k_race` parameter, scaled by √(command ratio) |

Also: `delta_dot = clip(delta_cmd − delta, ±rate)` in v2 is a first-order lag
with an implicit 1 s time constant, not a rate limiter. v3 separates transport
delay, rate limit and lag, and integrates the rudder **outside** the RK4 stages
(exactly) — carrying a sub-step servo lag as an RK4 state is stiff and left the
rudder at 32.6° for a 40° command.

None of these were exposed by the field runs, which are 20–25 s of oscillating
rudder. They appear the moment rudder is held, which training episodes do.

---

## 3. What the data supports

20 usable runs, ~11 minutes of motion.

- Telemetry and control run at **2 Hz** (0.500 s), not the 10 Hz in the
  `log_parser.py` docstring.
- Pose is quantised to 0.1 m / 0.1° but the stream is **smooth**: residual about
  a local quadratic is **0.34°** in yaw and **0.026 m** in position, i.e. at the
  quantisation floor. Derived yaw rate is good to ~0.5°/s against an 8–20°/s
  signal, so it can initialise prediction windows (it is never a fitting target).
- `x_real`/`y_real`/`yaw_real` inside `#ACTION` are the decoder's *latched* pose
  and duplicate whenever a LiDAR line arrives before its pose line. The raw pose
  lines are clean at 2 Hz and are what the pipeline uses.
- `udp_live_rl.py` opens the log in append mode, so one file can hold several
  bridge launches with the pose frame reset at each. Segmented at clock gaps,
  there are **no localisation jumps** in the July data. (The 9.48 m/s "jump" was
  this artefact.)
- Heading is consistent with direction of travel: pooled median of
  course − heading is **+2.2°**, IQR 7.8°.
- **RPM was 12.0 in 18 of 20 runs.** No thrust-curve excitation.

---

## 4. Identification protocol

- **Session-level split.** Fit on 2026-07-02, hold out 2026-07-03 — different
  day, setup, battery and water state.
- **Objective.** Pooled over 3 s and 10 s prediction windows plus the full free
  run, with heading and position as the only residuals, plus two
  distribution-matching terms and a manoeuvring constraint (§5, §7).
- **Optimiser.** Differential evolution vectorised over runs *and* candidates,
  multiple independent starts converging to the same optimum.
- **Uncertainty.** 28 run-level bootstrap resamples.

---

## 5. Two findings about the objective itself

**A plain MSE fit systematically under-turns.** Against a partly unpredictable
response the MSE-optimal model shrinks: it produced mean |yaw rate| of **3.5°/s
against 8.7°/s measured** — worse than v2. That is the wrong bias here, because a
simulator that under-turns trains a policy that over-commands rudder, which is
the Paper 2 field symptom (wider turns, larger oscillation, doubled RMS
cross-track error). A penalty on the p90 of |yaw rate| costs ~2° of windowed
heading accuracy and buys a matching turn-rate distribution.

**Sway needs its own term.** Sway is observable only through the drift angle, and
an MSE fit leaves it unconstrained — the first v3 fit over-drifted (IQR 14°
against 8.5° measured) and was the single test where v3 scored worse than v2. A
drift-IQR term pins it down.

Both support a methodological claim worth making in the paper: **when a model's
purpose is policy transfer, distribution matching belongs in the identification
objective, not only in the validation.**

---

## 6. Results, holdout session (2026-07-03)

### V1 — prediction error vs horizon

| horizon | naive: freeze | naive: const. rate | v2 | **v3** | v2 pos | **v3 pos** |
|---|---|---|---|---|---|---|
| 1 s | 4.10° | 1.22° | 2.43° | **1.31°** | 0.123 m | **0.057 m** |
| 2 s | 9.01° | 4.98° | 5.90° | **5.19°** | 0.310 m | **0.141 m** |
| 3 s | 12.87° | 10.04° | 9.40° | **7.90°** | 0.501 m | **0.262 m** |
| 5 s | 20.69° | 22.93° | 12.29° | **10.17°** | 0.861 m | **0.541 m** |
| 10 s | 20.13° | 31.43° | 19.80° | **13.82°** | 1.803 m | **1.246 m** |
| free run | 22.94° | 30.32° | 33.17° | **23.53°** | 4.334 m | **3.530 m** |

v3 beats v2 at every horizon in both heading and position. Note the naive
references: v2's free-run heading error (33.2°) is **worse than assuming the
vessel never turns** (22.9°). Only at 1 s does a naive predictor beat v3.

### V3–V6

| test | v2 | **v3** | measured |
|---|---|---|---|
| V3 paired per-run heading RMSE at 3 s | — | better in **6/6**, Wilcoxon p = **0.016** | — |
| V4 turn-rate KS distance | 0.243 | **0.171** | — |
| V5 free-run path-length error | −3.89 m | **−0.79 m** | 20.9 m |
| V6 drift-angle sd | 5.39° | **4.78°** | 5.68° |

The V6 regression reported in the previous iteration is gone: constraining the
manoeuvring behaviour (§7) fixed it as a side effect, and also took the paired
test from 4/6 (p = 0.078) to 6/6 (p = 0.016).

---

## 7. The manoeuvring constraint — read this before quoting the model

The field runs contain **no sustained turn**: the policy oscillates the rudder
throughout. That regime is therefore entirely unconstrained by the data, and an
unregularised fit extrapolated into it badly — turn rate *fell* from 7.4°/s at
25 % helm to 2.2°/s at full helm, with the vessel stalling to 0.54 m/s. A policy
trained against that would learn to avoid sustained turns as a pure simulator
artefact.

The fit now carries explicit manoeuvring priors, as a constraint rather than a
soft term (at comparable weight the optimiser simply paid the penalty):

- turn rate grows monotonically with helm
- every non-zero helm turns the same way
- steady turning radius in 1.5–4 vessel lengths
- speed at full helm between 0.55 and 0.85 of straight-line speed

Result: **2.2 / 5.0 / 7.7 / 10.0 °/s** across 25–100 % helm, radius **3.05 L**,
speed ratio **0.82**.

These bands are **priors from standard manoeuvring expectations for a hull with a
2.7 % rudder area ratio — not measurements.** The steady-turn behaviour of this
model is an assumption. Basin session 1 (turning circles) replaces it. Do not
present it as an identified result.

---

## 8. Identified parameters

90 % intervals from 28 run-level bootstrap resamples.

| parameter | fit | 5 % | 95 % |
|---|---|---|---|
| `T12` (N) | 16.1 | 14.7 | 18.6 |
| `X_uu` | 12.9 | 11.3 | 15.1 |
| `X_vv` | 50.6 | 47.7 | 54.8 |
| `X_rr` | 0.058 | 0 | 13.4 |
| `X_delta` | 0.116 | 0 | 0.555 |
| `Y_v` | 0.924 | 0 | 5.96 |
| `Y_vv` | 221 | 206 | 236 |
| `k_R` | 2.18 | 1.14 | 2.55 |
| `k_race` (m/s) | 0.014 | 0 | 0.211 |
| `N_r` | 1.04 | 0.014 | 8.69 |
| `N_rr` | 0.017 | 0 | 5.85 |
| `N_uv` | −84.0 | −86.8 | −55.3 |
| `rud_rate` (°/s) | 2985 | 2676 | 2998 |
| `rud_tau` (s) | 0.895 | 0.872 | 0.982 |
| `rud_delay` (s) | 0.730 | 0.670 | 0.780 |

Cautions:

- Several intervals are **wider than the parameter's own value** (`X_rr`, `N_rr`,
  `Y_v`, `N_r`, `k_race`). Those parameters are not identified; the model works
  because other terms carry the behaviour.
- `k_R`, `N_r` and `N_uv` are strongly correlated — they all scale the yaw
  subsystem. **Do not quote their individual values as physical measurements.**
- The bootstrap resamples runs from **one session** and re-optimises locally from
  the full-data optimum, so the intervals capture within-day run-to-run variation
  only. They are a **lower bound** on true parameter uncertainty.
- `rud_rate` saturating at an effectively infinite value is interpretable, not a
  failure: the bridge already limits the command to 50 %/s ≈ 20°/s, so a further
  limit inside the model is redundant. **The physical servo rate limit cannot be
  identified from this data** unless it is slower than 20°/s.
- `rud_delay` is an **effective** delay lumping transport latency, the
  one-frame-stale pose the controller used, servo response and estimator lag. It
  is not a servo specification.

---

## 9. Simulator readiness

### Actuator and timing parity

A policy learns the actuator and timing environment it trains in, not just the
hull. `VesselSim` reproduces the deployment plant measured in the July logs:

| parity item | deployment | `VesselSim` |
|---|---|---|
| control interval | 2 Hz, dt = 0.500 s | `control_dt = 0.5` |
| command rate limit | bridge limits to 50 %/s before `$CMD` | `command_rate_pct_s = 50` |
| actuator delay | effective 0.73 s | in the model (`rud_delay`) |
| observation staleness | controller pose is one frame old | `obs_delay_steps = 1` |
| rudder sign | `rudder_sign = -1` | handled in `dynamics` |

The command rate limit matters most. `udp_live_rl.py` applies it before
transmitting, so the vessel never receives the bang-bang output the policy
produces. Train without it and the policy learns that full reversals are free;
deploy and they arrive as 2 s ramps. It is ablatable (`command_rate_pct_s=None`)
but that should be deliberate.

### Domain randomisation

`sample_params(rng, scale)` blends **whole bootstrap parameter vectors** rather
than sampling each parameter independently from its interval. Independent
sampling is wrong here: with correlated yaw parameters and intervals wider than
their own values, independent draws land off the solution manifold and produce
simulators that are unstable or turn the wrong way. Blending preserves the
correlation structure, and every draw starts from a set that satisfied the
manoeuvring constraints. `scale` > 1 is the honest setting given the caveats in
§8; re-run acceptance test A5 whenever it changes.

### Acceptance tests — run before every training job

| test | what it catches | result |
|---|---|---|
| A1 bounded over action envelope | divergence at held helm (defects 2–3) | PASS |
| A2 steady turn physical | non-monotonic or wrong-way turning | PASS — 2.2/5.0/7.7/10.0 °/s, 3.05 L, ratio 0.82 |
| A3 timestep consistency | stiff integration (the rudder-as-RK4-state bug) | PASS — max deviation 0.235 |
| A4 determinism | hidden state across resets | PASS |
| A5 randomisation safety | unusable domain-randomisation draws | PASS — 12/12 |
| A6 field regression guard | edits quietly undoing the identification | PASS — all holdout metrics still beat v2 |
| A7 actuator parity | training/deployment actuator mismatch | PASS |

`sub_dt ≤ 0.05` is required; 0.1 drifts about 0.5 m over 30 s of manoeuvring.

---

## 10. Files

| file | what it is |
|---|---|
| `ship_model_v3.py` | drop-in replacement for `ship_model.py` — same `update(rpm, rud, dt)` interface and rudder convention, identified values and bootstrap baked in, plus `sample_params()` |
| `vessel_sim.py` | **use this for training** — control-rate wrapper with deployment actuator parity |
| `acceptance.py` | the seven pre-training checks |
| `dynamics.py` | shared vectorised dynamics; imported by the two above |
| `build_dataset.py` | log parsing, segmentation, command recovery from `$CMD` |
| `simulate.py` | batch replay, objectives, distribution terms, manoeuvring priors |
| `baseline_v2.py` | replays the unmodified `ship_model.py` for a fair comparison |
| `fit_final.py`, `fit_model.py` | bounds, chunked optimiser, bootstrap |
| `validate.py`, `figures.py`, `make_model.py` | the six field tests, figures, model generation |
| `params_final.json`, `validation.json` | numbers behind every table above |

Reproduce: `fit_final start {0,1,2} 110` → `fit_final boot 0 28` → `fit_final` →
`validate` → `figures` → `make_model` → `acceptance`.

---

## 11. Outstanding for part 2

In priority order, set by what these logs cannot reach:

1. **Turning circles** at 25/50/75/100 % helm — replaces the §7 priors, which are
   currently the weakest part of the model.
2. **Thrust vs RPM sweep** — `T12` is anchored at one operating point and the
   square-law exponent is assumed.
3. **Direct servo measurement** with the bridge rate limiter disabled and logging
   faster than 2 Hz — separates `rud_rate`, `rud_tau` and `rud_delay`, which
   currently trade off freely.
