# 02a — Paper 3 Reward Function Specification

**Status:** design output of `02_REWARD_AND_COLREGS.md`. Implementation handover to Claude Code.
**Supersedes:** the six-term `REWARD_REDESIGN.md` spec (re-derived, not patched — D10).
**Revision 2.2** — reconciled against the full doc set including a live read of `04`, 2026-09-07.

---

## 0. Changelog against Revision 1

Six updates in the document set moved this specification. Read this section if you already
have Revision 1.

| # | Update | Effect here |
|---|---|---|
| 1 | `02 §3.2` precedence table now **structurally resolved**, with overtaking locked to **passing to port of the target** | `v_side` overtaking branch sign **inverted**; admissibility predicate for overtaking switches from "side with more room" to `A_port` |
| 2 | `02 §3.2` head-on rationale: **9(a) compliance satisfies Rule 14 without alteration** — an alteration is required only when the target is not where 9(a) says | `v_r8` reformulated from absolute to **deficit-based**. Previously it penalised inaction whenever engaged, which under the new rationale would punish the agent for correctly holding course |
| 3 | `02 §4.4` boundary conflict resolved: **slacken speed, do not violate the boundary** | **`R-3` withdrawn.** Suppressing the port-turn penalty near the starboard wall would license the wrong action. Replaced by path-relative yaw (`R-8`), which removes the unwinnable state at its source |
| 4 | `02 §4.2` new implementation trap: port turns are not globally bad — overtaking *requires* a port turn | Compliant turn sense is now **class-dependent**; two dedicated unit tests |
| 5 | `00` / `05 §4.7`: **IMU confirmed** | `r` is measured, not differentiated. `r_dead` set from gyro noise floor rather than guessed; the yaw-rate criterion is now cheap in the field too |
| 6 | `03 §6`: propulsion widening resolved; **reverse thrust capability unverified** | §10.5 softened to conditional; two variants specified |

**Also closed:** `00` lists "progress-penalty carve-out design and gating for Rule 8(e)" as
an open item owned by 02. §5.5 answers it — **no carve-out is needed**, because progress is
formulated as telescoping arclength. See §5.5 and `R-9`.

### Revision 2.2 — doc 04 verified live

`04_SCENARIOS_AND_EVALUATION.md` is now in the project context and has been read in full.
**All five Revision 2.1 findings are confirmed against the live text**, and it adds two more.

| New finding | Effect |
|---|---|
| `04 §4.1` runs "Around the Clock" in an **open-water** variant. Three reward terms are undefined without a boundary: `r_pf` normalises on `W_local`, `r_bnd` needs a boundary polygon, `A_stbd`/`A_port` need `d_bnd_*` | New decision **`R-10`** (§5.8). Without it the benchmark either crashes or silently runs different reward semantics from training, invalidating the comparison |
| `04 §4.3` Tier B has only **three** corridor-width levels, and `04 §9` asks for them to be defined numerically from the 02 precedence table. But §2.2 predicts three *per-class* transitions, so "wide / intermediate / narrow" cannot be class-agnostic — narrow for crossing is wide for overtaking | §11.3 now supplies a class-aware definition: levels indexed by **how many rules remain admissible**. This directly closes `04 §9`'s open item |

Two Revision 2.1 items also sharpen now that the live text is available: `04 §4.2` Tier A and
`04 §4.3` Tier B both lack a target-placement axis, so §11.1 is an **evaluation** gap as well
as a training gap; and `04 §3.2`'s stage-4 "reduced TCPA" needs a floor (§11.2).

### Revision 2.1 — doc 04 reconciliation

No change to any term, formula or coefficient. Reconciling against `04 §2–4` and
`PAPER3_DRAFT_SKELETON §5` changed the hand-off section only:

| Finding | Effect |
|---|---|
| **The generator never samples the case where holding course is correct.** `04 §2` step 5 solves backwards for a spawn position without sampling a spawn DCPA, so the target lands on OS's projected track and `Δy_req = d_req` always | `v_r8`'s zero branch — the whole point of the `02 §3.2` head-on rationale — is never exercised in training. Now a blocking hand-off, not a caution (§11.1) |
| The six sweep widths bracket all three predicted transitions, but 6.02 m and 6.52 m fall in adjacent brackets | Recommend one extra level at 7 m (14 B) to separate them (§11.3) |
| `04 §4.1` "Around the Clock" places both vessels meeting at the origin, so `DCPA = 0` and the geometry is unbounded | The benchmark exercises the *alteration-required* branch and the confined sweep exercises the *channel-keeping-suffices* branch. Complementary by construction — worth saying in the paper |
| The non-compliant evaluation stratum includes a target that alters to port in a head-on | Side-of-passing correctness will record an OS violation caused by the target. Must be reported conditioned on target compliance (§11.4) |
| `SKELETON §5.4`'s metric list omits the speed-reduction metric that `02 §4.4` introduced | Gap between 02 and 04. Keys already emitted (§10.3); the metric list needs updating |

*(Superseded by Revision 2.2 above — the live document confirms every item.)*

---

## 1. Conventions

Fossen body frame, consistent with the existing simulator.

| Symbol | Meaning | Sign convention |
|---|---|---|
| `u, v, r` | surge, sway, yaw rate | `r > 0` is a **starboard** turn |
| `r_path` | yaw rate required to track the reference path | same sign convention |
| `e_y` | cross-track error | positive when OS is to **starboard** of the path |
| `α` | relative bearing of TS from OS | clockwise from OS bow, `[0,360)`; `(0,180)` is starboard |
| `β` | relative bearing of OS from TS | clockwise from TS bow |
| `y_rel_CPA` | lateral offset of TS from OS at projected CPA, OS body frame | **positive = TS to starboard** |
| `W` | local channel width | metres; reported in breadths `B = 0.50 m` |

Constants: `Lpp = 1.57 m`, `B = 0.50 m`, `U_ref = 0.8 m/s`, `Δt = 0.1 s`.

**Ship domain** (provisional, `01 §5.2`; final values an output of 05):

```
d_ahead  = 2.00 · Lpp = 3.14 m
d_astern = 1.00 · Lpp = 1.57 m
d_abeam  = 0.75 · Lpp = 1.18 m

d_dom(β) = 1 / sqrt( (cos β / a(β))² + (sin β / d_abeam)² ),   a(β) = d_ahead if cos β ≥ 0 else d_astern
d_req    = 2 · d_abeam = 2.36 m          # required separation, two identical vessels abeam
```

---

## 2. Admissibility predicate

`02 §3.2` fixes the table structure and defers width thresholds to Study 1. The reward
therefore needs a **per-step geometric predicate**, not a width lookup. One predicate drives
every row of the table, and the width thresholds fall out of the sweep as results.

```
Δy_req  = max(0, d_req − DCPA)                   # lateral deficit
r_stbd  = d_bnd_stbd − (B/2) − c_wall            # usable room to starboard
r_port  = d_bnd_port − (B/2) − c_wall            # usable room to port

A_stbd  = (r_stbd ≥ Δy_req)                      # starboard alteration admissible
A_port  = (r_port ≥ Δy_req)                      # port alteration admissible
```

- `d_bnd_*`: OS centre to the boundary polygon abeam, from the **map** (ground truth),
  taken as the **minimum over the along-path interval from now to the projected CPA**. The
  instantaneous value would let the agent commit to a manoeuvre that stops fitting before
  the CPA arrives.
- `c_wall = 0.65 m` (hull half-breadth 0.25 + 0.40 margin).
- Hysteresis: ±0.15 m band on `r_* − Δy_req`.

**Invariant, assert at config build:** `d_safe < c_wall − B/2`. Otherwise the geometry that
defines a compliant narrow-channel manoeuvre would itself trigger the boundary penalty —
the reward would punish the behaviour the paper is trying to elicit.

### 2.1 Mapping to the `02 §3.2` table

| Encounter | Compliant action | Compliant turn sense | Fallback when inadmissible |
|---|---|---|---|
| Head-on | Alter to starboard **only if `Δy_req > 0`**; otherwise hold the starboard side | `+1` | Slacken speed (8(e)) |
| Crossing | Alter to starboard and/or slacken; never cross ahead | `+1` | Slacken speed or stop (8(e)) |
| Overtaking | Pass **to port of the target**; regain starboard side after | `−1` | Hold astern at reduced speed (8(e)) |
| Being overtaken | Hold course and speed, keep starboard | `0` | — |

`compliant_turn_sense ∈ {+1, −1, 0}` is read from this table by class. It appears in `v_port`
(§6.2), `v_side` (§6.4) and `v_r8` (§6.6). **This single field is the mechanism that prevents
the `02 §4.2` implementation trap** — there is no global "port turns are penalised" constant
anywhere in the code.

### 2.2 Predicted Study 1 thresholds

Subordinate to the table above — these are **predictions to be tested by the sweep**, not
inputs to the reward.

| Encounter | Predicted threshold | Derivation |
|---|---|---|
| Crossing (starboard alteration admissible) | `W ≥ 6.52 m` (13.0 B) | centreline path: `W/2 − 0.90 ≥ 2.36` |
| Head-on (alteration achievable) | `W ≥ 3.66 m` (7.3 B) if TS keeps its own starboard side; `W ≥ 6.02 m` (12.0 B) if TS holds the centreline | §2.4 |
| Overtaking (port pass fits) | `W ≥ 4.16 m` (8.3 B) geometric; `4.78 m` (9.6 B) with a 1.15 prudence factor | TS at `W/2 − 0.65`, OS at `TS − 2.36`, OS needs 0.65 to the port wall |

Predicted ordering: **crossing (13.0 B) > head-on (7.3–12.0 B) > overtaking (8.3 B)**.

Crossing is the *first* rule to become inadmissible as the channel narrows, not the last —
a vessel already keeping starboard under 9(a) has spent its starboard room before the Rule 14
alteration becomes tight. This is the quantitative form of N2 and it is not what open-water
intuition suggests. Study 1's width levels should be chosen to resolve all three transitions.

### 2.3 Why this reading is tidy

Rule 9(a) and Rule 14 **agree on the passing side** and differ only on whether an alteration
is required. Rule 9 never reverses a Rule 13–16 obligation in this domain; it removes the
manoeuvre and leaves the side intact. That is far easier to defend line by line than "Rule 9
overrides Rule 14", and it is exactly what `02 §3.2`'s head-on rationale says.

### 2.4 Constraint on the scenario generator

The head-on obligation has two regimes depending on the target's **lateral placement**:

| Target placement | OS must produce | Alteration required? |
|---|---|---|
| TS on its own starboard side (positionally 9(a)-compliant) | nothing — `Δy_req = 0` | **No.** Channel-keeping satisfies Rule 14 |
| TS on the channel centreline | the whole 2.36 m separation alone | Yes, and the threshold moves to 6.02 m |

Training targets are constant-velocity (D1) and never alter, so only initial placement
varies. `03 §5` currently brackets the head-on transition between 4 m and 3.5 m, which is the
3.66 m figure. **If the generator places targets on the centreline, that bracket is wrong by
a factor of ~1.6 and the sweep will miss the transition.** The reward is unaffected —
`Δy_req` is computed from the target's actual track — but Study 1's reported threshold is not.

---

## 3. Two structural decisions

### 3.1 `R-1` — ground truth vs perceived state

`01 §5.3` requires one classifier serving observation and reward. That settles the *module*,
not the *input*. Recommended split:

| Term group | Input | Reason |
|---|---|---|
| Physical consequence — collision, boundary, domain intrusion, path error, progress | **Ground truth** | Physical facts. The agent pays for hitting things whether or not it saw them |
| Rule regime — which COLREGs term is active, and its severity | **Perceived state**, identical tensor to the observation | Never penalise a role the agent was not shown |

Under Study 2's degradation sweep this makes COLREGs obligations degrade with perception
while physical safety obligations do not — which is the real situation at sea, and turns
Study 2 into a study of obligation under uncertainty rather than a robustness curve. It is
also what makes `04 §6`'s stated key result measurable: because the COLREGs reward follows
the *perceived* class, the agent is trained under classification noise, so "misclassifies and
turns the wrong way" is an observable failure mode rather than a confound.

**This does not breach the information-parity argument in `04 §8`.** The ground-truth inputs
are consumed by the *reward*, which exists only at training time. At evaluation the policy
reads the observation and nothing else, so it consumes the same tracked target state — errors
included — as the VO comparators. Say this explicitly in the methods; otherwise `R-1` reads
as the learned policy being handed privileged information, and it would undercut both the
parity claim and N1.

### 3.2 `R-6` — the reward does not depend on CRI

CRI constants are still being re-derived in ship lengths (`01` open item). Gating on CRI
would couple this spec to an unfinished one, and a later constant change would silently
re-weight the reward. Instead: engagement and severity are **geometric** (§6.1). CRI stays in
the observation exactly as `01 §6.1` specifies; it is simply unused here.

---

## 4. Architecture

```
r_t =  w_pf·r_pf + w_prog·r_prog + w_exist·r_exist + w_smooth·r_smooth
     + w_obs·r_obs + w_bnd·r_bnd + w_dom·r_dom + w_COL·r_col + r_term
```

**Every dense term is normalised to `[-1, 0]` before weighting**, except `r_prog ∈ [-1,+1]`.
The weight *is* the maximum per-step contribution, so the magnitude hierarchy holds by
construction and the §8 audit checks realised severity rather than hunting a hidden scale
factor. This is the direct fix for the Paper 2 failure.

---

## 5. Task and safety terms

### 5.1 `r_pf` — unified path following

```
ẽ_y  = e_y / (W_local / 2)                                          ∈ [-1, 1]
χ̃*   = ω_LA·χ̃_LA + (1 − ω_LA)·χ̃
q_t  = w_e·exp(−γ_e·ẽ_y²) + (1 − w_e)·(1 + cos χ̃*)/2                ∈ [0, 1]
g_u  = clip( max(u,0) / U_ref_eff , 0, 1 )
r_pf = −(1 − g_u · q_t)                                             ∈ [-1, 0]
```

`ω_LA = 0.25`, `w_e = 0.70`, `γ_e = 4.0`.

**Width normalisation is load-bearing.** Paper 2 used `exp(−0.05·|e_y|)`, inherited from a
60 × 150 m map; over a 10 m channel it varies by under 10% of its own value. Normalising by
local half-width fixes that *and* holds the term's range constant across the Study 1 sweep.
Without it the path-following gradient changes with corridor width and confounds Study 1.

**Penalty form** so the speed gate acts in the right direction: stopping gives `g_u = 0` and
therefore maximum penalty.

`U_ref_eff` (`R-2`) drops to `0.4 · U_ref` when a give-way obligation is active and the
compliant alteration is inadmissible, so a legal Rule 8(e) slowdown does not cost
path-following reward. Without this the agent structurally cannot learn 8(e).

### 5.2 `r_bnd` — channel boundary

```
r_bnd = −[ max(0, 1 − d_b / d_safe) ]²        ∈ [-1, 0]
```

`d_b` = hull polygon to boundary polygon, from the map, ground truth. `d_safe = 0.50 m`
(changed from Paper 2's 0.7 m — required by the §2 invariant).

The boundary stays a **hard constraint** per `02 §4.4`: `w_bnd` is the largest dense weight,
above the COLREGs group, so slackening speed always dominates violating the boundary.

### 5.3 `r_dom` — target ship-domain intrusion

```
r_dom = −[ max(0, 1 − d_TS / d_dom(β_TS)) ]²    ∈ [-1, 0]
```

Hull-to-hull, ground truth, asymmetric domain applied at the correct bearing.

**Addition beyond the doc's eleven terms, and not optional.** `00 §4.2` reports ship-domain
intrusion rate and depth, but nothing in the six carried terms or the five COLREGs terms
supplies a dense signal for target proximity — the only feedback would be the terminal
collision penalty. A reported metric with no corresponding reward signal is precisely the
pattern that produced Paper 2's concessions.

### 5.4 `r_obs` — static obstacle proximity

Shifted exponential: exponential in shape as specified, but exactly zero beyond a cut-off
rather than carrying a constant background.

```
ε_cut = exp(−d_cut / d_oa)
r_obs = −clip( (exp(−d_clear/d_oa) − ε_cut) / (1 − ε_cut), 0, 1 )     ∈ [-1, 0]
```

`d_oa = 0.60 m`, `d_cut = 2.00 m` → `ε_cut = 0.036`.
Values: `2.0 m → 0`; `1.0 → −0.16`; `0.5 → −0.41`; `0.2 → −0.71`; `0 → −1`.

`d_clear` = minimum hull-to-hull clearance over static obstacles **within ±135°**, matching
the `c_t` swath, so the agent is never charged for proximity it cannot observe and a passed
obstacle astern generates no signal against an action space with no reverse.

The unshifted form was rejected: at `d_oa = 0.8 m` it reads −0.08 at 2 m, integrating to
≈ −53 over 300 steps at `w_obs = 2.2` — larger than the path term, constant, and carrying no
gradient. Paper 2's failure mode in the opposite direction.

### 5.5 `r_prog` — progress, and the carve-out question

```
r_prog = clip( (s_t − s_{t−1}) / (U_ref · Δt), −1, 1 )      ∈ [-1, +1]
```

`s` = **along-path arclength**, not distance-to-goal (which penalises the outside of a bend).

**`R-9` — this closes the open item `00` assigns to 02.** `02 §4.4` flags that the progress
term and existence cost fight Rule 8(e), and asks for a class-conditional carve-out gated on
encounter class and CRI. It also flags the degenerate-policy risk that any such carve-out
creates: creep along slowly forever, avoid everything, never finish.

**No carve-out is needed.** `Σ_t r_prog` telescopes to `(s_T − s_0)/(U_ref·Δt)`, a constant
fixed by path length and independent of speed. Slowing down therefore costs **zero** progress
reward provided the agent still completes within the horizon. The only cost of a legal
slowdown is the extra steps' path penalty and existence cost — about 4 points for a 3-second
reduction, against roughly 90 for the violation avoided.

The clip makes telescoping exact only for `u ≤ U_ref`, which is the normal regime, and has
the useful side effect that speeding gains nothing.

This is strictly better than a gated carve-out: it removes the tension without introducing
the exploit, and it needs no CRI threshold, so it also stays consistent with `R-6`. The
acceptance check `02 §4.4` asks for — inspect the trained policy's speed profile in open
stretches — should still be run, but there is now no mechanism for the exploit to arise from.

### 5.5a `R-10` — open-water fallback for the "Around the Clock" benchmark

`04 §4.1` runs the benchmark in **two** variants: channel-constrained and open water. Three
terms above are undefined when there is no boundary polygon — `r_pf` normalises on
`W_local`, `r_bnd` measures to the boundary, and `A_stbd`/`A_port` read `d_bnd_*`.

```
if open_water_mode:
    W_local = W_ref_open_water = 10.0 m     # widest trained width
    r_bnd   = 0
    A_stbd  = A_port = True                 # no lateral constraint
    d_bnd_stbd = d_bnd_port = +inf          # boundary obs branch returns max range
```

`W_ref = 10.0 m` rather than an arbitrary large value: it keeps `ẽ_y` on the same scale as
the widest sweep condition, so the open-water score is directly comparable to the 20 B row of
Study 1. Any other choice silently changes the path-following gradient between the two
variants and makes the comparison meaningless.

**Also state the limitation.** The boundary observation branch saturates at max range in open
water, and training tops out at 10 m (20 B, O4), so the open-water variant is a **zero-shot
out-of-distribution transfer for that branch**. `04 §4.1` motivates the variant as
"comparability against the published literature", which is right, but the comparison is
against methods that were designed for open water while this policy was not. Report it as
such — it costs one sentence and is awkward if a reviewer raises it first.

### 5.6 `r_smooth` — action smoothness

```
r_smooth = −σ_t · clip( (Δa_δ/κ_δ)² + w_n·(Δa_n/κ_n)², 0, 1 )     ∈ [-1, 0]
```

`κ_δ` = the actuator's per-step rate limit in normalised units, so the term saturates at
exactly the physical limit and self-calibrates when 05 delivers the actuator model.
`κ_n = 0.30`, `w_n = 0.5`.

**`σ_t` resolves the Rule 8 tension (`02 §4.3`).** Rule 8(b) wants one large alteration and
forbids a succession of small ones; a plain smoothness penalty suppresses both.

```
σ_t = σ_enc  if  0 ≤ (t − t_engage) < N_free  else 1.0
σ_enc = 0.25,  N_free = 20 steps (2.0 s)
```

The first two seconds after engagement are cheap, so the committed alteration is affordable;
everything after is charged at full rate, so dithering is not. Directly testable —
first-action magnitude is already a reported metric.

### 5.7 `r_exist` and terminals

```
r_exist = −1  (constant)
r_term  = +100  goal  |  −300  collision (static / boundary / target)  |  0  timeout (bootstrap)
```

`R_collision = −300` rather than the −200 in `02 §4.1`. See `R-7`. Uniform across the three
collision types: they are reported separately as metrics but not weighted separately, so the
reward makes no claim about their relative severity that the paper would have to defend.

---

## 6. COLREGs terms

### 6.1 Encounter state machine and gates

```
IDLE     → ENGAGED   when class ≠ none ∧ TCPA ∈ (0, T_engage] ∧ DCPA < κ_eng · d_req
                     latch: class, ψ_engage, u_engage, t_engage, compliant_turn_sense
ENGAGED  → CLEARING  when TCPA < 0 ∧ range opening,  or  DCPA > κ_rel · d_req
CLEARING → IDLE      after N_clear steps
```

`T_engage = 25 s`, `κ_eng = 1.5`, `κ_rel = 2.5`, `N_clear = 30`, `N_switch = 10`.

**`κ_eng` is now scaled on `d_req`, not `d_dom`.** In Revision 1 it evaluated to 2.36 m —
exactly the compliant separation — putting the engagement threshold on a knife-edge at the
geometry the agent is supposed to achieve. At `1.5 · d_req = 3.54 m` engagement fires before
the obligation does, which is right: watch first, then act.

```
ρ_t = clip( 1 − DCPA / (κ_eng · d_req), 0, 1 )      ∈ [0, 1]     # proximity gate
```

At a compliant pass (`DCPA = d_req`) `ρ_t = 0.33`, but every severity below is zero there, so
no penalty accrues.

### 6.2 `v_port` — turning the wrong way while give-way

**`R-8` — measured relative to the path, not absolutely.**

```
r_err  = r − r_path                                          # yaw rate in excess of path tracking
v_port = ρ_t · clip( max(0, −s_c · r_err) / r_ref , 0, 1 )
         · 1[ class ∈ {head_on, crossing, overtaking} ] · 1[ ENGAGED ]
```

where `s_c = compliant_turn_sense` from §2.1 (`+1` head-on/crossing, `−1` overtaking).

Two things this fixes, both introduced by the Revision 2 doc updates:

1. **`02 §4.2`'s implementation trap.** Overtaking requires a port turn. With `s_c = −1` the
   same expression penalises a *starboard* turn during an overtake. There is no global "port
   is bad" constant to miscode.
2. **The bend problem, and the withdrawal of `R-3`.** Following a channel that bends to port
   requires a port turn that is path-following, not evasion. Subtracting `r_path` makes the
   term measure only the *excess* yaw rate, so a compliant bend costs nothing. This removes
   the unwinnable state that `R-3` was patching — and `R-3`'s carve-out is now actively
   wrong, because `02 §4.4` resolves the boundary conflict to *slacken speed*, not *turn to
   port*. Suppressing the penalty near the wall would license the wrong action.

`r_ref = 0.20 rad/s`, `r_dead = 0.02 rad/s`. With the IMU confirmed (`05 §4.7`), set `r_dead`
from the measured gyro noise floor rather than the nominal value, and note that the
yaw-rate-not-rudder criterion is now directly measurable in the field rather than inferred.

### 6.3 `v_bow` — crossing ahead of the target

Evaluated at the constant-velocity projected CPA, consistent with DCPA/TCPA.

```
σ_bow = clip( (cos β_CPA − cos β_bow) / (1 − cos β_bow), 0, 1 )        β_bow = 67.5°
v_bow = ρ_t · σ_bow · clip( 1 − DCPA/d_req, 0, 1 )
        · 1[ class ∈ {crossing, overtaking} ] · 1[ ENGAGED ]
```

Smooth in `β_CPA`, no discontinuity at the beam. For overtaking this catches cutting back
across the bow after the pass — the characteristic overtaking violation, and now more likely
given the locked port-side pass followed by a required return to the starboard side.

A realised version, measured at actual minimum range, is logged as the reported metric. Do
not use it in the reward; it is only available after the fact.

### 6.4 `v_side` — wrong-side passing

**Sign is class-dependent, and the two cases are opposite.** This is the `02 §4.2` trap in its
sharpest form.

```
head-on:      required TS to port      →  v_side = ρ_t · clip( +y_rel_CPA / d_req, 0, 1 )
overtaking:   required TS to starboard →  v_side = ρ_t · clip( −y_rel_CPA / d_req, 0, 1 )
```

Head-on: port-to-port means TS on OS's port, so `y_rel_CPA < 0` is correct and a positive
value is penalised. Overtaking: passing to port **of the target** puts OS on TS's port side,
which puts TS on OS's **starboard**, so `y_rel_CPA > 0` is correct and a negative value is
penalised. Opposite signs, same encounter geometry family — write both unit tests.

Overtaking additionally requires `A_port`; if `¬A_port` the correct behaviour is to hold
astern (§10.6), and `v_side` is evaluated only once a pass is actually attempted
(`|y_rel_CPA|` departing from the astern-station value).

Not applied to crossing — there the requirement is "pass astern", which `v_bow` covers.
Adding a side term would double-count.

### 6.5 `v_hold` — course-keeping while being overtaken (17(a)(i))

Penalty on deviation rather than reward for holding, so the whole group is a penalty group
and the §7 hierarchy is uniform. A positive hold reward would also duplicate `r_pf`, which
already rewards steady course, and would risk paying the agent to hold course into a
collision.

```
v_hold = ρ_t · clip( ((r − r_path)/r_hold)² + ((u − u_engage)/Δu_hold)², 0, 1 )
         · 1[ class = being_overtaken ] · 1[ ENGAGED ] · 1[ NOT in_extremis ]

in_extremis = (DCPA < d_req) ∧ (TCPA < T_extremis)
```

`r_hold = 0.05 rad/s`, `Δu_hold = 0.10 m/s`, `T_extremis = 5 s`.

**Path-relative yaw here too.** The `02 §3.2` table now adds "keep starboard" to the
being-overtaken row. An absolute-yaw formulation would penalise the corrective alteration
needed to regain the starboard side; `r − r_path` does not.

**`R-4` — the `in_extremis` suppression touches locked decision S5.** Rule 17(b) requires the
stand-on vessel to act when collision cannot be avoided by the give-way vessel alone. That is
a different provision from 17(a)(ii), which S5 puts out of scope. Suppression does not
*reward* release — it stops punishing it and leaves the collision and domain penalties to
dominate. The out-of-scope claim survives intact; without the carve-out the reward would
contain an instruction to hold course into a collision.

### 6.6 `v_r8` — Rule 8, deficit-based

**Reformulated in Revision 2.** `02 §3.2` now states that 9(a) compliance can satisfy Rule 14
without any alteration. A term that penalises inaction whenever engaged would punish the
agent for correctly holding course — the single most consequential change in this revision.

```
A_req   = clip( Δy_req / d_req, 0, 1 )                       # 0 when no action is required
Δψ_c    = max(0, s_c · (ψ_t − ψ_engage))  unwrapped          # compliant-sense heading change
Δu_red  = max(0, u_engage − u_t)
A_t     = (Δψ_c/Δψ_min if turn_admissible else 0) + Δu_red/Δu_min

urgency = clip( 1 − TCPA/T_act, 0, 1 )
v_r8    = urgency · clip( A_req − A_t, 0, 1 ) · 1[ class ∈ give-way ] · 1[ ENGAGED ]
```

`Δψ_min = 20° = 0.35 rad`, `Δu_min = 0.24 m/s` (30% of `U_ref`), `T_act = 15 s`.
`turn_admissible` = `A_stbd` for head-on/crossing, `A_port` for overtaking.

Reads directly:

- Target properly placed, `DCPA ≥ d_req` → `Δy_req = 0` → `A_req = 0` → **`v_r8 = 0`**.
  Channel-keeping satisfies Rule 14, exactly as `02 §3.2` requires.
- Target displaced → `A_req` scales with the deficit, and the obligation is proportionate
  rather than a fixed 20°.
- Alteration inadmissible → `A_t` counts speed reduction only, so Rule 8(e) discharges the
  obligation. This is `02 §4.4`'s decision expressed as reward structure.
- Counts **only compliant directions**, so a large wrong-way alteration does not discharge
  anything.

Time-to-first-action and first-action-magnitude fall out of `A_t` and are logged directly.

`T_act = 15 s` is derived: 2.36 m of lateral offset at 0.8 m/s with a 30° alteration needs
≈5.9 s of running plus ≈3.5 s of turn-in and turn-out, ≈10 s, plus margin. **Constraint for
04:** spawn TCPA must extend well above 15 s or the term saturates at spawn and carries no
gradient.

**State `T_act` non-dimensionally as well.** `03 §5` flags that a reviewer will ask what a
15 s TCPA on a 1.57 m model means at full scale. In ship lengths of advance,
`T_act · U_ref / Lpp = 15 × 0.8 / 1.57 ≈ 7.6 Lpp`. Report the Rule 8 timing threshold in
ship lengths, not seconds, and the Froude question answers itself.

### 6.7 Group aggregation

```
v_col = clip( 0.55·v_port + 0.55·v_bow + 0.40·v_side + 0.45·v_hold + 0.50·v_r8, 0, 1 )
r_col = −v_col                                                            ∈ [-1, 0]
```

Clipped to unit range **before** the group weight. No combination of violations can exceed
`w_COL` per step, so the group cannot silently outrank the safety terms the way Paper 2's
path term outranked its avoidance term.

Maxima before clipping: head-on `1.45`; crossing `1.60`; overtaking `1.50`; being overtaken
`0.45`. Two concurrent severe violations saturate; one does not.

---

## 7. Coefficients and hierarchy

| Term | Range | Weight | Max per-step |
|---|---|---|---|
| Terminal collision | one-shot | — | **300** |
| Terminal goal | one-shot | — | 100 |
| `r_bnd` boundary | `[-1,0]` | `3.0` | 3.0 |
| `r_dom` target domain | `[-1,0]` | `2.5` | 2.5 |
| `r_obs` static obstacle | `[-1,0]` | `2.2` | 2.2 |
| `r_col` COLREGs group | `[-1,0]` | `1.8` | 1.8 |
| `r_pf` path following | `[-1,0]` | `0.6` | 0.6 |
| `r_prog` progress | `[-1,1]` | `0.3` | 0.3 |
| `r_smooth` smoothness | `[-1,0]` | `0.10` | 0.10 |
| `r_exist` existence | `−1` | `0.05` | 0.05 |

```
300  ≫  3.0 > 2.5 > 2.2  >  1.8  >  0.6 > 0.3 > 0.10 > 0.05
collision ≫ ——— safety ———  >  COLREGs  >  ——— task ———
```

Satisfies `02 §5` exactly. Because every term is unit-normalised the ordering is a property
of the table, not something to discover empirically.

**Two checks, and the distinction matters.** `02 §5` and `02 §6` ask for different things and
can conflict, because terms have very different natural durations — a boundary excursion
lasts seconds, a COLREGs violation lasts a whole encounter.

1. **Instantaneous ordering** — the table above. Holds by construction. Unit test.
2. **Episode-integrated audit** — §8. Checks realised severity, and is *allowed* to show a
   different ordering: a rarely-active term should integrate small. Its job is to catch a
   term 10× off its intended share, not to reproduce the instantaneous ordering.

State this distinction in the paper. It pre-empts the obvious question about why Table R7's
ordering does not match the stated hierarchy.

---

## 8. Scale audit — pre-committed numbers

Mandatory per `02 §6`. Predictions to assert against, so a mismatch is diagnostic.

### 8.1 Predicted episode-integrated contributions

Design point: 300 steps, 20 m path, `U_ref = 0.8 m/s`.

| Term | Nominal success | Collision at step 150 | Max non-compliant, no collision |
|---|---|---|---|
| `w_prog·Σr_prog` | +75 (telescoping, fixed) | +37.5 | +75 |
| `w_pf·Σr_pf` | −27 | −13.5 | −40 |
| `w_obs·Σr_obs` | −26 | −13 | −26 |
| `w_exist·Σr_exist` | −15 | −7.5 | −17 |
| `w_smooth·Σr_smooth` | −1.5 | −0.8 | −4 |
| `w_bnd·Σr_bnd` | 0 | 0 | −15 |
| `w_dom·Σr_dom` | 0 | 0 | −45 |
| `w_COL·Σr_col` | 0 | 0 | −270 |
| Terminal | +100 | −300 | +100 |
| **Episode return** | **≈ +105** | **≈ −297** | **≈ −242** |

Three orderings to assert:

- `success (+105) > max-violation (−242) > collision (−297)`. A COLREGs-compliant collision is
  worse than a non-compliant near-miss (`02 §5`). Margin 55.
- Loitering to timeout ≈ −86 + bootstrap: worse than the goal, better than colliding. A
  cornered agent correctly prefers timeout to collision.
- **Compliance-cost ratio, the single most important number.** Over a 100-step encounter:
  compliance ≈ 29 (path deviation 25 + extra steps 4 — progress contributes **zero** because
  it telescopes, §5.5); violation at realised severity 0.5 ≈ 90. **Ratio ≈ 3.1.** Revision 1
  predicted 2.05 by wrongly charging 19 points of progress loss to compliance. If the audit
  returns below ~1.5, `w_COL` is too low and compliance will not be learned.

### 8.2 Procedure

1. Instrument `info` with per-term instantaneous and episode-integrated values (§10.3).
2. Run **1,000 random-policy** and **1,000 Paper 2 SAC** episodes through the new reward
   without training. Tabulate mean, median, 95th percentile per term.
   **Sample from the `04 §3.2` stage-5 distribution, not stage 1.** Stages 1–2 contain no
   dynamic target, so the whole COLREGs group reads exactly zero and the audit would look
   like a wiring bug. Run the audit per stage if you want the curriculum's reward profile,
   but the headline Table R7 must come from stage 5.
3. Check every term against §8.1 to a factor-of-3 tolerance. Outside that band means the
   coefficient is wrong regardless of what the ratios say on paper.
4. Check the three orderings explicitly.
5. Re-run after **any** coefficient change. Non-negotiable.
6. Report as Table R7.

The Paper 2 SAC policy is not COLREGs-aware, so it should show a large `Σr_col`. If it does
not, the classifier or the engagement gate is not firing and the audit has found a bug rather
than a scale problem.

---

## 9. Config schema

```python
@dataclass(frozen=True)
class RewardConfig:
    # weights
    w_pf: float = 0.60
    w_prog: float = 0.30
    w_exist: float = 0.05
    w_smooth: float = 0.10
    w_obs: float = 2.20
    w_bnd: float = 3.00
    w_dom: float = 2.50
    w_col: float = 1.80

    # terminal
    r_goal: float = 100.0
    r_collision: float = -300.0
    timeout_bootstrap: bool = True

    # path following
    gamma_e: float = 4.0
    w_e: float = 0.70
    omega_la: float = 0.25
    u_ref: float = 0.80
    u_ref_slow_factor: float = 0.40      # R-2

    # safety geometry
    d_safe: float = 0.50
    c_wall: float = 0.65
    d_oa: float = 0.60
    d_cut: float = 2.00
    obs_swath_deg: float = 135.0

    # ship domain (provisional; final from 05)
    dom_ahead_lpp: float = 2.00
    dom_astern_lpp: float = 1.00
    dom_abeam_lpp: float = 0.75

    # smoothness
    kappa_delta: float = None            # from actuator rate limit at build time
    kappa_n: float = 0.30
    w_n: float = 0.50
    sigma_enc: float = 0.25
    n_free: int = 20

    # encounter state machine
    t_engage: float = 25.0
    kappa_eng: float = 1.5               # scaled on d_req, not d_dom
    kappa_rel: float = 2.5
    n_clear: int = 30
    n_switch: int = 10

    # COLREGs sub-weights
    w_port: float = 0.55
    w_bow: float = 0.55
    w_side: float = 0.40
    w_hold: float = 0.45
    w_r8: float = 0.50

    # COLREGs thresholds (r_ref, r_dead from 05 turning circle + gyro noise floor)
    r_ref: float = 0.20
    r_dead: float = 0.02
    beta_bow_deg: float = 67.5
    r_hold: float = 0.05
    du_hold: float = 0.10
    t_extremis: float = 5.0
    dpsi_min_deg: float = 20.0
    du_min: float = 0.24
    t_act: float = 15.0

    # propulsion (03 §6 — reverse unverified)
    reverse_available: bool = False      # set True only if 05 confirms
    u_min_reachable: float = 0.20

    # open-water fallback (R-10; 04 §4.1 benchmark variant)
    open_water_mode: bool = False
    w_ref_open_water: float = 10.0       # = widest trained width, keeps e_y scale comparable

    # ablation switches (00 §4.3)
    colregs_terms_enabled: bool = True
    encounter_feature_enabled: bool = True
    colregs_term_mask: tuple = ("port", "bow", "side", "hold", "r8")

    log_per_term: bool = True
```

**Validator assertions — fail at construction, not at step 10,000:**

```
assert d_safe < c_wall - B/2
assert w_bnd > w_dom > w_obs > w_col > w_pf > w_prog > w_smooth > w_exist
assert abs(r_collision) > w_col * max_encounter_steps
assert t_act < t_engage
assert d_cut > d_oa
assert kappa_eng * d_req > d_req          # engagement fires before obligation
```

---

## 10. Implementation notes

### 10.1 One context object, three consumers

```python
@dataclass
class EncounterContext:
    # perceived — feeds observation AND colregs gating
    cls: EncounterClass
    alpha: float; ct: float
    dcpa: float; tcpa: float; cri: float
    y_rel_cpa: float; beta_cpa: float
    # latched
    state: EncounterState
    psi_engage: float; u_engage: float; t_engage: int
    compliant_turn_sense: int            # +1 / -1 / 0, from §2.1
    # admissibility (map, ground truth)
    a_stbd: bool; a_port: bool
    r_stbd: float; r_port: float; dy_req: float; a_req: float
    # path reference
    r_path: float                        # required yaw rate to track the path
    # ground truth — safety terms and metrics only
    d_ts_true: float; dcpa_true: float
```

Observation builder, reward and metrics logger all read this one object. Mechanical guarantee
behind `01 §5.3`'s "one module, two consumers" — if each recomputes, they will diverge at
sector boundaries eventually.

`r_path` is new in Revision 2 and is required by `v_port` and `v_hold`. Compute it from the
path curvature at the current along-track station and the current speed:
`r_path = κ_path(s) · u`.

### 10.2 Module layout

```
paper_pooling/src/
  colregs/
    classifier.py      # 5-class + hysteresis, head-on band ±10°  (owned by 01)
    geometry.py        # CPA, domain, A_stbd/A_port, dy_req, beta_cpa, y_rel_cpa
    context.py         # EncounterContext + state machine + compliant_turn_sense
  reward/
    terms.py           # each term a pure function returning a value in its declared range
    reward.py          # weighted sum, group clipping
    audit.py           # per-term accumulators, Table R7 generator
```

Pure functions of `(state, ctx, cfg)` make the leave-one-out ablation a mask over a dict
rather than `if` branches scattered through the step function, and make §10.4 trivial.

### 10.3 Logging

```
reward/term/<name>            instantaneous, pre-weight
reward/weighted/<name>        instantaneous, post-weight
reward/episode/<name>         running integral, post-weight
colregs/class, colregs/state, colregs/a_stbd, colregs/a_port
colregs/a_req, colregs/action_ratio
metrics/time_to_first_action, metrics/first_action_magnitude
metrics/min_dcpa, metrics/domain_intrusion_depth, metrics/passing_side_correct
metrics/speed_reduction_events        # NEW — 02 §4.4 requires timing + appropriateness
metrics/speed_reduction_appropriate   # was class active and was the alteration inadmissible?
```

`00 §4.2`'s metric set should be a **read** of these keys, not a separate computation. Paper 2's
concessions came from metrics that were not designed in before the campaign ran.

### 10.4 Unit tests before any training

1. Every term inside its declared range across 10⁵ random states.
2. Coefficient ordering assertion (§9).
3. `Σ r_prog` over a full traversal equals `L_path/(U_ref·Δt)` to float tolerance, and is
   **invariant to the speed profile** — this is the `R-9` claim, test it directly.
4. **Head-on, target positionally 9(a)-compliant, agent holds course → `v_r8 = 0`.** The
   `02 §3.2` rationale, as a regression test. If this fails the agent is being trained to
   manoeuvre unnecessarily.
5. **Head-on, port alteration → `v_port > 0`; overtaking, port alteration → `v_port = 0`.**
   The `02 §4.2` trap, both directions.
6. **`v_side` sign inversion:** head-on penalises `y_rel_CPA > 0`, overtaking penalises
   `y_rel_CPA < 0`. Assert both in one test so they cannot drift apart.
7. `v_port` fires when yaw rate crosses `r_dead`, **not** at the rudder reversal (locked
   principle 5 — the exact bug the principle exists to prevent).
8. Channel bending to port with no target → `v_port = 0` (the `r_path` subtraction).
9. `A_stbd` false at `W = 3.5 m`, true at `W = 10 m` for a centreline head-on.
10. `in_extremis` suppression: `v_hold → 0` when `DCPA < d_req ∧ TCPA < 5 s`.
11. `d_safe < c_wall − B/2` invariant.

### 10.5 Propulsion authority

`03 §6` resolves the widening. Two variants, because reverse thrust is unverified:

| | If reverse **is** available | If it is **not** |
|---|---|---|
| Reachable surge | `u ∈ [−0.15, 0.90] m/s` | `u ∈ [0.20, 0.90] m/s`, `n_min = 0` reachable |
| Rule 8(e) coverage | "take all way off" available | Slackening only; **state the limitation in the paper** rather than claiming stopping |
| Model requirement | Identification must cover reverse — this is a 05 deliverable, not a config flag | Current identification suffices |

Default `reverse_available = False`. Do not flip it on the basis of a datasheet; 05 must
identify the reverse regime or the simulator will extrapolate into an unmodelled envelope.

Stage the widening through the curriculum as in Paper 2, but stage 5 must expose the full
range. The Paper 2 frozen baseline (M1) runs with its own action mapping — not a
fair-comparison problem, but state it.

### 10.6 Narrow-channel overtaking — `R-5`

`02 §3.2` fixes the fallback as "hold astern at reduced speed". That behaviour is unreachable
under the default terms: no progress, full path penalty via the speed gate, existence cost
accruing, likely timeout.

```
if class == overtaking and not A_port:
    U_ref_eff   = max(u_TS, 0.2)      # matching the target's speed satisfies the gate
    w_exist_eff = 0.0                 # holding station is not wandering
```

`v_bow` and `v_side` still penalise attempting the pass. Without this the narrow overtaking
case is not a test of COLREGs reasoning but a test of whether the agent tolerates an
unwinnable reward — and it will resolve it by overtaking anyway.

Note this is the one place where the existence cost is suspended, and the suspension is
gated on a geometric predicate rather than on CRI, keeping it consistent with `R-6` and
avoiding the degenerate-policy risk `02 §4.4` warns about.

---

## 11. Sign-off and hand-offs

### Decisions

| # | Decision | Status | Consequence if rejected |
|---|---|---|---|
| **R-1** | Ground truth for physical terms, perceived state for COLREGs gating | Adopt | Either penalised for unseen roles, or not penalised for unseen collisions |
| **R-2** | `U_ref_eff` drops when speed reduction is the admissible action | Adopt | Rule 8(e) structurally unlearnable — the speed gate punishes the compliant action |
| **R-3** | ~~Suppress port-turn penalty near the starboard boundary~~ | **Withdrawn** | Superseded by `R-8`. `02 §4.4` resolves the boundary conflict to slacken speed; suppressing the penalty would license a port turn instead |
| **R-4** | `in_extremis` suppression of the course-keeping penalty (17(b), **not** 17(a)(ii)) | Adopt | The reward instructs the agent to hold course into a collision. Touches S5 — needs one sentence distinguishing 17(b) from 17(a)(ii) |
| **R-5** | Existence cost and speed gate relaxed for hold-astern in narrow overtaking | Adopt | The `02 §3.2` overtaking fallback is unreachable |
| **R-6** | Reward gated geometrically, not on CRI | Adopt | 02 blocks on 01's unfinished CRI re-derivation |
| **R-7** | `R_collision = −300`, not −200 | Adopt | At −200 the margin between a collision and a maximally non-compliant episode is 32 points, violating `02 §5` |
| **R-8** | All yaw-based COLREGs terms measured as `r − r_path` | **New** | Compliant bends and starboard-side recovery are penalised as COLREGs violations; the unwinnable state `R-3` was patching returns |
| **R-9** | No progress-penalty carve-out — telescoping arclength removes the tension | Adopt | The `00` open item stays open, and any gated carve-out reintroduces the creep exploit `02 §4.4` warns about |
| **R-10** | Open-water fallback: `W_local → 10.0 m`, `r_bnd = 0`, `A_stbd = A_port = True` | **New** | `04 §4.1`'s open-water benchmark variant crashes or runs different reward semantics from training, and the comparison against the published literature is void |

### Needs a number from elsewhere

- `r_ref`, and `r_dead` from the gyro noise floor — 05 (turning circle + IMU, both now confirmed)
- Final ship domain from measured advance and tactical diameter — 05
- `κ_δ` from the calibrated actuator rate limit — 05
- Reverse thrust capability — 03 open item, gates §10.5

### Lands on doc 04

**Verified against the live `04_SCENARIOS_AND_EVALUATION.md`.** Every item below cites text
that is present in the current document.

#### 11.1 Sample a spawn DCPA — **blocking**

`04 §2` parameterises the generator as: class → heading-intersection angle → target speed →
spawn TCPA → **solve backwards for spawn position** → corridor width, bends, path offset →
static obstacles. Step 6 samples the *own ship's* path offset. Nothing samples the
**target's** lateral station in the channel; it falls out of step 5 as a by-product.

The natural implementation of "solve backwards for the geometry at that TCPA" places the
target on OS's projected track, so `DCPA ≈ 0`, so `Δy_req = d_req`, so `A_req = 1` — **in
every single training episode**.

Consequence: `v_r8`'s zero branch never fires. The agent never sees the case that `02 §3.2`
identifies as the normal one — target correctly on its own side of the fairway, channel-keeping
satisfies Rule 14, holding course is the right answer. It would learn "always alter", which
is the behaviour the precedence table exists to avoid, and the M5 ablation could not
distinguish "learned to alter" from "learned *when* to alter".

**Fix:** add spawn DCPA as an explicit sampled parameter alongside spawn TCPA, with a
stratum where the target is positionally 9(a)-compliant (`DCPA ≥ d_req`, target on its own
starboard side). Report the realised distribution.

This also settles §2.4: with DCPA sampled, both head-on regimes appear in training and the
threshold Study 1 reports is a property of the sweep rather than an artefact of the spawner.

**It is an evaluation gap too, which the live document makes clear.** `04 §4.2` specifies
Tier A as "one per encounter class × corridor width condition"; `04 §4.3` strata are class ×
target behaviour × width × clutter. Neither has a target-placement axis. So even with the
generator fixed, the suite could not *report* the distinction — the two head-on regimes would
be pooled into one violation rate, and the `02 §3.2` rationale would be untestable. Head-on
needs **two** Tier A variants (target 9(a)-compliant / target displaced) and a fourth Tier B
stratum, or a documented decision not to.

#### 11.2 Spawn TCPA range — needs a floor

Must extend well above `T_act = 15 s` (≈7.6 Lpp of advance), or `v_r8` saturates at spawn and
carries no gradient. `04 §3.2`'s curriculum already treats spawn TCPA as an axis — stage 3
"generous", stage 4 "reduced" — so this is a bound on the sampled range, not a new parameter.

**Recommended:** training spawn TCPA ≥ `1.5 · T_act` = 22.5 s across stages 3–5, which also
sits below `T_engage = 25 s` so engagement fires shortly after spawn rather than at it.
Stage 4's "reduced TCPA" must not cross below `T_act`, or the agent is already late at spawn
and `v_r8` is pinned at maximum for the whole episode — teaching nothing except that the term
is unavoidable. Sub-`T_act` spawns are worth keeping, but as an **evaluation** stratum
("already late"), not a training stage.

#### 11.3 Add one width level at 7 m

The existing sweep — 10, 8, 6, 5, 4, 3.5 m — brackets all three predicted transitions, but
the two upper ones land in adjacent brackets and would be hard to separate:

| Predicted threshold | Current bracket |
|---|---|
| Crossing, 6.52 m (13.0 B) | 8 → 6 |
| Head-on with centreline target, 6.02 m (12.0 B) | 6 → 5 |
| Overtaking, 4.16–4.78 m (8.3–9.6 B) | 5 → 4 |
| Head-on with compliant target, 3.66 m (7.3 B) | 4 → 3.5 |

Adding 7 m (14 B) separates the crossing threshold from the head-on one cleanly. One extra
level in an existing sweep, and it is the level that carries N2's headline ordering result.

**Tier B's three levels cannot be class-agnostic — this closes `04 §9`.** `04 §4.3` uses
"wide / intermediate / narrow" and `04 §9` asks for those thresholds numerically from the 02
precedence table. But §2.2 predicts three *different* per-class transitions, so a single
width label means different things to different encounter classes: 10 B is wide for
overtaking and narrow for crossing. Define the levels instead by **how many rules remain
admissible**:

| Tier B level | Width | Admissible | Governing behaviour |
|---|---|---|---|
| **L1 — unconstrained** | `W ≥ 13.0 B` (6.5 m) | crossing, head-on, overtaking | Rules 13–16 apply as in open water |
| **L2 — partially constrained** | `9.6 B ≤ W < 13.0 B` (4.8–6.5 m) | head-on, overtaking | Crossing give-way falls to speed reduction, 8(e) |
| **L3 — Rule 9 dominant** | `W < 9.6 B` (< 4.8 m) | head-on only, and only with a positionally compliant target | Overtaking suppressed to hold-astern; head-on to channel-keeping |

Geometric, derived from the ship domain rather than from any method's performance, and it
satisfies `04 §4.4`'s requirement that difficulty never be defined by baseline behaviour. It
also makes L3 the natural home for `04 §4.4`'s deliberate-failure stratum.

Thresholds move with the ship domain, which is an output of 05 — recompute after the
turning-circle identification and before freezing the suite.

#### 11.4 Two metric corrections

- **Passing-side correctness must be conditioned on target compliance.** `03 §2.3`'s
  non-compliant stratum includes a target that alters to port in a head-on, forcing a
  starboard-to-starboard pass. `v_side` will score that as an OS violation, and the reported
  metric would conflate the target's fault with the policy's. Report the metric split by
  target behaviour, or the non-compliant stratum reads as a policy failure.
- **`SKELETON §5.4` omits the speed-reduction metric** that `02 §4.4` introduces. The keys
  are already emitted (§10.3); the metric list needs the entry.

#### 11.5 Two framing points, free

- **"Around the Clock" and the width sweep are complementary by construction.** The benchmark
  places both vessels meeting at the origin, so `DCPA = 0` and the geometry is unbounded: it
  exercises the *alteration-required* branch exclusively. The confined sweep exercises the
  *channel-keeping-suffices* branch. Together they cover both sides of the precedence table,
  and one of them is a published benchmark. Worth stating rather than leaving implicit.
- **Narrow-channel overtaking is a natural candidate for `SKELETON §5.3`'s deliberate
  failure stratum** — the one where no method passes cleanly. It is better than a merely
  hard case, because the correct answer (hold astern, do not overtake, §10.6) is a
  *behaviour* rather than a success rate, so failures there are interpretable.
