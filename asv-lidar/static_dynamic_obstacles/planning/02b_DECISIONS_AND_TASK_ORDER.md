# 02b — Decisions and Task Order

**Input:** `CONSTANTS_AND_SCALES.md` Rev 2.2 (42 unresolved constants),
`PORTING_MANIFEST.md` §5.7 F17, §6.3 F18, §6.4 F19, §6.7.
**Output:** every open item either decided, converted to a measurement, or explicitly deferred
with a reason. Nothing is left as "needs a call".
**Companion to:** `02a_REWARD_SPECIFICATION.md` Rev 2.2 — §1 below amends it.

Claude Code was right to refuse three unilateral changes (the `e_y` flip, the extra sweep
level, the terminal payoffs). All three are decided here.

---

## 1. Corrections to `02a` — the spec was wrong, the code was right

### C1. F18 stands. My §11.3 arithmetic was wrong; **do not add 6.25 m**

`02a §11.3` claims 7 m separates the crossing threshold from the centreline head-on. F18 is
correct that it does not: 6.02 m sits 2 cm *above* the 6 m sweep level, so that transition
coincides with a sample point and 6.52 m shares the (6, 7) bracket with it. My table placed
6.02 m in the "6 → 5" bracket, which only holds if read as exactly 12.0 B = 6.00 m.

**But do not add 6.25 m either.** Both thresholds derive from `d_abeam = 0.75·Lpp` and
`c_wall = 0.65 m`, and *both are `TODO(05)`*. Adding a sweep level to separate two numbers
that will move when the turning circle is identified is premature, and it would bake a level
into a suite that `04 §4.5` requires to be frozen and hashed.

**Decision:** hold the sweep at seven levels. Keep
`test_crossing_and_centreline_head_on_still_share_a_bracket` exactly as written — asserting
the collision as the documented state is the right instinct and it forces this to be
revisited rather than forgotten.

**Task:** make the four thresholds a **computed function of the domain constants**, not a
literal list. `PREDICTED_THRESHOLDS_M` becomes `predicted_thresholds(domain, c_wall) -> dict`,
so re-deriving after 05 is one call and the sweep levels can be re-chosen against fresh
numbers in one sitting. That is the actionable part of F18.

### C2. F19 stands, and it propagates further than the manifest says

The simulator's speed envelope is not merely inconsistent with `02a` — it is physically
implausible. At `LBP = 1.57 m`, Froude number `Fr = U/√(g·L)`:

| Source | U | Fr | Plausible for a 64.55 kg displacement hull? |
|---|---|---|---|
| Simulator @ 12 RPM | 1.77 m/s | **0.45** | No — semi-planing |
| Simulator @ 24 RPM | 3.13 m/s | **0.80** | No — full planing |
| `02a §1` | 0.80 m/s | 0.20 | Yes |
| Paper 2 field | 0.55 m/s | 0.14 | Yes, and it is a *measurement* |

**Decision: the field figure wins. The simulator is wrong, not `02a`.** `05 §2` already lists
"Paper 2 used thrust ∝ RPM²; verify" as a task; F19 is the evidence that it needs doing.

Provisional `U_REF = 0.55 m/s` — the measured field cruise speed, not my 0.80 assumption and
not the simulator's 1.77. Introduce a single calibration constant `THRUST_CAL` scaling the
thrust map such that steady `u` at `CRUISE_RPM` equals `U_REF`. The discrepancy then lives in
one number instead of propagating through every normaliser, and when the log-mining task
(T1) lands, one constant changes.

**One `U_REF`, consumed by everything.** The `ego` branch normaliser, both target-speed
features, `SPEED_SCALE`, every reward speed gate, `TARGET_SPEED_RANGE` and
`BEING_OVERTAKEN_SPEED_MARGIN` all read it. F19's saturating-`ego`-feature bug was two
constants disagreeing about the same physical quantity; one constant makes that impossible.

### C3. `r_prog` renormalised — this is a real spec change

`02a §5.5` defines `r_prog = clip((s_t − s_{t−1})/(U_ref·Δt), −1, 1)`, whose episode integral
is `L_path/(U_ref·Δt)`. That couples the whole §8.1 audit table to the unresolved speed
question: at `U_ref = 0.8` the integral is 250, at 0.55 it is 364, and `w_prog·Σr_prog` moves
from +75 to +109. The audit would have to be recomputed every time the speed estimate moved.

**Replace with a path-fraction form:**

```
r_prog = clip( N_ref · (s_t − s_{t−1}) / L_path , −1, +1 )        N_ref = 250
Σ r_prog = N_ref = 250, exactly, invariant to speed and to path length
```

`w_prog · Σr_prog = +75` as designed, permanently. The §8.1 table survives C2 unchanged, and
`R-9`'s telescoping property — the reason no progress carve-out is needed — is preserved and
in fact strengthened, because it no longer depends on `U_ref` at all.

### C4. `e_y` sign — flip to textbook, as F17.1 recommends

`path.py` returns positive-to-port; `02a §1` specifies positive-to-starboard. Claude Code's
recommendation is right and its reasoning is the reasoning: the reward's passing-side terms
condition on the sign, the classical comparators use textbook natively, and a reader will
assume it.

The decisive point is narrower than either document makes it. This paper's contribution is
COLREGs *geometry*. Carrying a non-standard lateral sign through `v_side`'s two opposite
branches (`02a §6.4`: head-on penalises `y_rel > 0`, overtaking penalises `y_rel < 0`) is
inviting exactly the class of error the paper is about. Flip it.

**Task T2** below specifies it as one atomic change with a regression gate.

---

## 2. Decisions on the unresolved constants

### Decided now — design calls, no measurement needed

| Constant | Decision | Reasoning |
|---|---|---|
| `R_COLLISION` / `R_GOAL` / `R_TIMEOUT` | **−300 / +100 / 0** | `02a §5.7`, `R-7`. `R_TIMEOUT = 0` is not cosmetic — see T4 |
| `DOMAIN_RADIUS_DCPA` | **Keep the lateral semi-axis for the observation feature only.** The reward uses directional `d_dom(β)` and the constant `d_req` | No conflict: `02a §5.3` already evaluates the domain at the target's bearing, and `ρ_t` is gated on `κ_eng·d_req`, a constant. Document that observation and reward normalise differently *on purpose* |
| CRI constants (`CRI_*`) | **Approve the sensor-horizon anchoring** | See §2.1 — the justification found is better than the one it replaces |
| `TCPA_CLIP` | **40 s**, was 60 | `T_engage = 25 s`, so anything past ~30 s is unactionable. 60 s wasted a third of the feature range on values the policy can never use |
| `DCPA_CLIP_DOMAINS` | **Replace with a clip at `LIDAR_RANGE` (16 m), normalised by `d_abeam`** | Removes a free constant. A DCPA beyond the sensor horizon is not an estimate, it is an extrapolation |
| `BEING_OVERTAKEN_SPEED_MARGIN` | **`0.15 · U_REF`**, not an absolute 0.10 m/s | At 1.77 m/s cruise, 0.10 was 5.6%; at 0.55 it is 18%. The same absolute number meant two different things. Must exceed the tracker's speed-estimation noise, which 05 measures |
| `TARGET_SPEED_RANGE` | **`(0.35, 1.35) · U_REF`** | F6.6 correctly widened it so `being_overtaken` is reachable, but derived it from the wrong cruise speed. Expressing it as a multiple survives T1 |
| `TRACK_GATE_DIST` | **`2.5 · U_REF · Δt`**, floor 0.30 m | Ties the association gate to maximum plausible inter-frame displacement so it rescales with the speed fix |
| `CLUSTER_EPS`, `CLUSTER_MIN_POINTS` | **Keep 0.35 m and 4** | The rope-rejection rationale is sound and `test_min_points_rejects_a_taut_suspension_line` pins it |
| `USE_RECURRENCE` | **False, and this closes `01`'s open item** | The headline architecture stays feedforward. Recurrence enters only as the RecurrentPPO comparator and, contingent on it ranking top-two in selection, the M7 frame-stacked ablation. Adding recurrence to the headline would confound N1: an occlusion result would no longer isolate perception from memory |
| `TARGET_COMPLIANT_SPAWN_PROB` | **Keep 0.5**, report realised | Roughly balanced is right for learning *when* to alter. 04 still owns stratification |
| `NO_TARGET_EPISODE_PROB` | **Keep 0.25** | Satisfies `04 §3.1`'s "meaningful fraction" |
| `REVERSE_AVAILABLE` | **Stay False** | `02a §10.5` — do not flip on a datasheet |

### 2.1 The CRI anchoring is a better argument than the one it replaces

Claude Code found that a faithful re-derivation in ship lengths gives a decay scale of 18.2 m
against a 16 m sensor horizon, so CRI would never decay and every target would read maximum
risk. Anchoring to the sensor horizon instead is approved, and the supporting observation is
worth putting in the paper:

> A 320 m ship with 2 NM of radar sees **11.6 hull lengths** ahead. The Bluefin with 16 m of
> LiDAR sees **10.2**. The perceptual horizon transfers almost exactly even though the
> absolute range does not — it is the *channel* that is small at model scale, not the sensing.

That is a stronger justification than ship-length scaling, and it generalises: **the invariant
to preserve under scaling is the perceptual horizon in ship lengths, not the sensing range in
ship lengths.** State it once in the problem formulation and it covers every other constant
that had to be re-derived.

Add one defusing sentence: because `R-6` removed CRI from the reward, these constants affect
an observation feature only. They are not safety-critical and a reviewer does not need to
audit them.

---

## 3. Three things nobody has flagged

### 3.1 The 1 m LiDAR dead zone constrains 05's ship-domain derivation

`CONSTANTS §4` records `LIDAR_MIN_RANGE = 1.0 m` as a confirmed sim-to-real gap. Nobody has
connected it to the ship domain.

`d_abeam = 0.75 · Lpp = 1.18 m`, which sits **0.18 m outside** the sensor blind zone. That is
currently a coincidence.

**If 05's turning-circle identification produces an abeam domain below 1.0 m, the ship domain
falls entirely inside the sensor's blind zone** — the agent would be penalised by `r_dom`
(ground truth, per `R-1`) for intrusions it is physically incapable of perceiving, in
simulation and in the field. `r_dom` would become an unlearnable term rather than a shaping
signal.

**Decision: `d_abeam ≥ LIDAR_MIN_RANGE + hull half-breadth = 1.25 m` is a hard floor on the
derived domain.** If the measured manoeuvring performance implies a smaller one, the domain is
floored at 1.25 m and the paper states why — a domain smaller than the sensor can resolve is
not a domain, it is a blind spot. Add to `05`'s derivation task and assert in the config
validator.

### 3.2 The aft-occlusion measurement gates the entire Rule 17 contribution

`CONSTANTS §4` is right that this one gates being-overtaken, and that settling it needs ten
minutes of static-spin recording. But it should not block for a basin booking.

**Decision: bound it now by sensitivity rather than waiting.** Train the reward-development
runs (`02a §8`, three seeds, one-third budget) at `LIDAR_AFT_MASK_HALF_DEG = 0°` and at a
plausible worst case of **30°** — bow-mounted sensor, 1.57 m hull, mast and superstructure.
If being-overtaken compliance survives the worst case, the measurement is a robustness
footnote. If it does not, the measurement is blocking and goes at the top of the next basin
session ahead of everything else.

Ten minutes of basin time either way, but this tells you *how much* it matters before you
spend it, and the sensitivity result is reportable regardless of which way it falls.

### 3.3 `03`'s corridor generator is on the critical path for three separate things

The manifest notes the boundary branch carries no information in a straight inset rectangle,
and correctly says do not ablate it yet. It is worse than that — three deliverables are
blocked on the same missing piece:

| Blocked | Why |
|---|---|
| The boundary-branch ablation | Port and starboard clearance are affine in `e_y`; a null result would be an artefact |
| **`R-8`'s regression test** (`02a §10.4` test 8) | With `κ = 0` everywhere, `r_path = 0` and `R-8` silently reduces to the absolute form. The term would look implemented and be untested |
| Study 1's interpretability | A width sweep in a straight corridor sweeps one number; the precedence table is about *local* admissibility |

After the reward itself, this is the highest-leverage build task. It is 03's, but it should be
sequenced immediately after 02 rather than in parallel with 04.

---

## 4. Task order for Claude Code

Dependencies are real; the ordering is not arbitrary.

### T1 — Mine the retained field logs *(no basin time, do first, ~half a day)*

Three extractions from the same logs, in one pass:

1. **Steady speed at commanded RPM** → settles C2. Commanded RPM is logged; scan-to-map
   registration gives speed over ground. Produces `U_REF` and `THRUST_CAL` as measurements
   rather than assumptions. **This is the highest-value half-day in the project** — every
   speed-normalised feature and every reward speed gate currently scales against a vessel
   that is not the one being simulated.
2. **rf2o pose drift** → `BOUNDARY_POSE_NOISE_XY` / `_HEADING_DEG` / `_WALK`. `CONSTANTS §6`
   calls these the single most consequential outstanding numbers and it is right: they are
   all 0.0, so the `01 §3.3` sim-to-real gap is wide open and no headline run should start.
3. **Black-wall return rate against bearing** → the `05` first action, already flagged.

All three are `05` deliverables that need no booking. Bundling them is one log-reading session
instead of three.

### T2 — Flip the `e_y` sign *(atomic, gated)*

One commit: `path.py` convention flip, propagated through `obstacles.py`'s lateral-offset
handling, `obs-v2` version bump.

**Gate:** re-run the 58 full-env rollout comparisons that established bit-identity. Expect
`path[0]` to change sign and **nothing else to move**. If any other observation index, reward
value, termination or `info` key changes, the flip has caught an unstated coupling — stop and
report rather than fixing forward. Add
`test_cross_track_sign_is_positive_to_starboard` asserting +1.01 for a vessel 1 m to starboard
of a due-north path, the exact case F17.1 measured.

### T3 — `r_path`, in `path.py`

```
r_path = u · κ_signed(s)          # κ from the reference path at the current projection
```

Sign convention matching `r > 0 = starboard`. Expose in `info` alongside `W_local`, and in the
`EncounterContext` (T4).

Implement now, but record in the manifest that **its test cannot pass until 03 delivers
bends** (§3.3). A term that is implemented, untested and silently inert is worse than one
that is absent, so the test should exist and be marked `xfail` with the reason, not omitted.

### T4 — Build the reward *(the main task)*

Per `02a` Rev 2.2, with C1–C4 applied. Order within the task:

1. `EncounterContext` (`02a §10.1`) — the single per-step object shared by observation, reward
   and metrics. `ObservationBuilder` currently computes class, crossing side and risk
   internally; move them here. The classifier stays one definition, which is what `01 §5.3`
   actually requires.
2. `reward/terms.py` — each term a pure function returning a value in its declared range.
3. `reward/reward.py` — weighted sum, group clipping to `[-1, 0]` before `w_COL`.
4. Config with the validators from `02a §9`, **plus** `assert d_abeam >= 1.25` (§3.1).
5. All eleven unit tests from `02a §10.4`, with test 8 `xfail` pending T5.

**`R_TIMEOUT = 0` is an implementation change, not a constant change.** The env must return
`truncated=True, terminated=False` at the step limit. If it currently terminates with −1000,
SB3 will not bootstrap, and the entire "no loitering incentive" argument in `02a §8.1` is
void — the agent would see timeout as a −1000 terminal and prefer almost anything to it.
Assert the flags directly in a test.

### T5 — `03`'s corridor generator

Variable width along the path, bends, deliberately off-centre reference paths. Unblocks the
boundary ablation, T3's test, and Study 1 (§3.3).

### T6 — Scale audit

`02a §8.2`, on the **stage-5** distribution. Stages 1–2 have no target, so the COLREGs group
reads exactly zero and the audit would look like a wiring bug.

The number to watch is the compliance-cost ratio, predicted ≈3.1. Below ~1.5 and `w_COL` is
too low for compliance to be learned, whatever the coefficient table says.

### T7 — Aft-occlusion sensitivity

§3.2. Runs alongside the reward-development seeds, no extra campaign.

---

## 5. Left open deliberately

| Item | Why it stays open |
|---|---|
| Ship domain final values | Output of 05's turning circle. Floored at `d_abeam ≥ 1.25 m` (§3.1) |
| `c_wall = 0.65 m` | `TODO(05)`. Drives all four sweep thresholds |
| Sweep levels | Deferred until the domain is final (C1). Re-choose in one sitting via `predicted_thresholds()` |
| `κ_δ` actuator rate limit | 05's calibrated actuator model |
| Reverse thrust | Needs identification, not a datasheet |
| Motion distortion | Needs a raw bag with `time_increment`; not extractable from the retained logs |
| Tier A / Tier B target-placement axis | 04's, and `02a §11.1` is the blocking hand-off |

---

## 6. What to say back to Claude Code

Three of its refusals were correct and should be confirmed explicitly, because a build agent
that is overruled on a good call learns to stop making them:

- **F18** — right, and the fix is not the one it proposed to defer to; it is to defer *entirely*
  and make the thresholds computable (C1).
- **F17.1** — right to stop. Now decided: flip (C4).
- **The terminal payoffs** — right to leave marked rather than silently changed. Now decided.

And one correction: **F19's `U_MAX_SURGE = 3.2 m/s` normaliser fixes the saturation symptom
but keeps the wrong vessel.** Normalising by a speed the real Bluefin cannot reach means the
`ego` surge feature will occupy the bottom fifth of its range once the thrust map is
corrected — the same dead-feature problem, moved rather than solved. Normalise by `U_REF`
after T1 and re-check the span.
