# 02a — Paper 3 Reward Function Specification

**Status:** design output of `02_REWARD_AND_COLREGS.md`. Implementation handover to Claude Code.
**Supersedes:** the six-term `REWARD_REDESIGN.md` spec (carried forward but re-derived, not patched — D10).
**Revision:** 1, drafted against document set Revision 2 (two-vessel scope).

This document is self-contained. It gives the Rule 9 precedence table (§2), the full reward
specification with normalisation ranges and coefficients (§4–§7), the pre-committed scale
audit with numeric assertions (§8), the config schema (§9), and the module structure (§10).

**Seven decisions marked `R-1`…`R-7` need Nam's sign-off before implementation.** They are
collected in §11. Everything else is settled by the existing decision log.

---

## 1. Conventions

Fossen body frame throughout, consistent with the existing simulator.

| Symbol | Meaning | Sign convention |
|---|---|---|
| `u, v, r` | surge, sway, yaw rate | `r > 0` is a **starboard** turn |
| `ψ` | heading | positive clockwise from north |
| `e_y` | cross-track error | positive when OS is to **starboard** of the path |
| `α` | relative bearing of TS from OS | clockwise from OS bow, `[0, 360)`; `(0,180)` is starboard |
| `CT` | heading intersection angle | `(ψ_TS − ψ_OS) mod 360` |
| `β` | relative bearing of OS from TS | clockwise from TS bow |
| `W` | local channel width | metres; also reported in breadths `B = 0.50 m` |
| `Δt` | control period | 0.1 s |

Vessel constants: `Lpp = 1.57 m`, `B = 0.50 m`, `LOA = 1.73 m`, `U_ref = 0.8 m/s`.

**Ship domain** (provisional, from `01 §5.2`; final values are an output of `05`):

```
d_ahead  = 2.00 · Lpp = 3.14 m
d_astern = 1.00 · Lpp = 1.57 m
d_abeam  = 0.75 · Lpp = 1.18 m

d_dom(β) = 1 / sqrt( (cos β / a(β))² + (sin β / d_abeam)² )
a(β)     = d_ahead  if cos β ≥ 0 else d_astern
```

Two joined half-ellipses. Continuous at `β = ±90°` because `cos β = 0` there.
Required separation between two identical vessels abeam: `d_req = 2 · d_abeam = 2.36 m`.

---

## 2. Rule 9 precedence table — the blocking deliverable

### 2.1 The single geometric predicate

The table below is driven by **one admissibility predicate**, evaluated every step from the
map boundary and the current encounter geometry. This replaces a fixed width threshold, and
it is the mechanism that makes the width thresholds a *reported result* of Study 1 rather
than an input (as `02 §3.2` requires).

```
Δy_req   = max(0, d_req − DCPA)                  # extra lateral offset needed
r_stbd   = d_bnd_stbd − (B/2) − c_wall           # usable room to starboard
r_port   = d_bnd_port − (B/2) − c_wall           # usable room to port

A_stbd   = (r_stbd ≥ Δy_req)                     # starboard alteration admissible
A_port   = (r_port ≥ Δy_req)                     # port alteration admissible (overtaking only)
```

- `d_bnd_stbd`, `d_bnd_port`: distance from OS centre to the boundary polygon abeam, taken
  from the **map** (ground truth), evaluated as the minimum over the along-path interval
  from the current position to the projected CPA. Using the minimum over the interval, not
  the instantaneous value, prevents the agent committing to a manoeuvre that stops fitting
  before the CPA is reached.
- `c_wall = 0.65 m` centre-to-wall clearance (hull half-breadth 0.25 m + 0.40 m margin).
  **Invariant that must be asserted in code:** `c_wall > d_safe` (§5.2), otherwise the
  geometry that defines a compliant manoeuvre would itself trigger the boundary penalty.
- `DCPA` is measured to the **ship domain**, per `01 §5.1`.

Hysteresis: `A_stbd` latches with a ±0.15 m band on `r_stbd − Δy_req` to prevent chatter.

### 2.2 The table

| Encounter | Wide channel (`A_stbd` true) | Narrow channel (`A_stbd` false) | Governing rule | Predicted threshold |
|---|---|---|---|---|
| **Head-on** | Alter to starboard, pass port-to-port. Alteration must be early and substantial | Hold the starboard side of the fairway; **slacken speed or stop**. Passing side unchanged: still port-to-port. No port alteration | 14 + 9(a) wide; 9(a) + 8(e) narrow | `W ≥ 3.66 m (7.3 B)` if TS holds its own starboard side; `W ≥ 6.02 m (12.0 B)` if TS holds the centreline — see §2.4 |
| **Crossing** (either side) | Give way: alter to starboard **and/or** slacken speed. Do not cross ahead | Give way by **speed reduction as the primary action**; hold the starboard side. Do not cross ahead | 15, 16, 9(b) throughout; 8(e) below threshold | `W ≥ 6.52 m (13.0 B)` for a centreline path; unavailable at any width once OS is already on the starboard limit |
| **Overtaking** | Overtake on the side with room; regain the starboard side after passing; do not cut back across the bow | **Do not initiate.** Fall in astern, match speed, hold station until the channel widens | 13, 16, 9(a) wide; 9(e) narrow | `W ≥ 4.16 m (8.3 B)` geometric minimum; `W ≥ 4.78 m (9.6 B)` with a 1.15 prudence factor |
| **Being overtaken** | Hold course and speed | Hold course and speed | 13, 17(a)(i) | — (no threshold; 17(a)(ii) out of scope per S5) |

### 2.3 The result that carries N2

The three thresholds are **different, and they are not ordered the way the open-water
literature would suggest**:

```
crossing (13.0 B)  >  head-on (7.3–12.0 B)  >  overtaking (8.3 B)
```

Crossing is the *first* rule to become inadmissible as the channel narrows, not the last.
The reason is Rule 9(a): a vessel already keeping to the starboard side of the fairway has
consumed its starboard room, so the Rule 15/16 starboard alteration is unavailable long
before the Rule 14 alteration is. This is the quantitative statement of N2 and it is a
genuinely non-obvious result. It should be foregrounded in the problem formulation, and
Study 1's width levels should be chosen to resolve all three transitions.

**A second clean statement worth making in the paper:** Rule 9(a) and Rule 14 *agree* on the
passing side (port-to-port) and differ only on whether an alteration is required. So Rule 9
never reverses a Rule 13–16 obligation in this domain — it only removes the manoeuvre while
leaving the passing side intact. That is a much tidier precedence relation than "Rule 9
overrides Rule 14", and it is defensible line by line.

### 2.4 Consequence for the scenario generator — flag to doc 04

The head-on threshold has **two values** depending on where the generator places the target
laterally in the channel:

| Target lateral placement | OS must produce | Threshold |
|---|---|---|
| TS on its own starboard side (positionally compliant) | half the separation | 3.66 m (7.3 B) |
| TS on the channel centreline | the **whole** separation alone | 6.02 m (12.0 B) |

Training targets are constant-velocity (D1) and therefore never alter, so OS produces the
entire separation in every case; only the target's *initial* placement varies. Doc 03 §5
currently brackets the transition between 4 m and 3.5 m, which is the 3.66 m figure. **If
the generator places targets on the centreline, that bracket is wrong by a factor of ~1.6
and the sweep will miss the transition entirely.**

Action: doc 04 must sample the target's lateral offset explicitly and report the realised
distribution. The reward is unaffected — `A_stbd` is computed from the target's actual track
— but Study 1's reported threshold depends on it, and that dependence must be stated.

### 2.5 The Rule 9(b) reading — state it, don't bury it

Rule 9(b) applies to a vessel under 20 m that "shall not impede the passage of a vessel that
can safely navigate only within a narrow channel". Both vessels here are model-scale. The
paper asserts OS is the non-impeding vessel and TS is channel-confined.

For crossing encounters this is deliberately conservative, because Rule 9(d) makes a vessel
*crossing* a narrow channel the one constrained not to impede. Under a strict reading a
crossing target would often be the one at fault. **Say so explicitly in the problem
formulation:** the own ship adopts the non-impeding posture unconditionally because an
autonomous vessel should not condition its own give-way obligation on an inference about
another vessel's manoeuvring constraints. That is a safety argument, not a COLREGs
loophole, and it is far stronger than leaving the asymmetry for a reviewer to find.

---

## 3. Two structural decisions before the terms

### 3.1 `R-1` — ground truth vs perceived state, term by term

`01 §5.3` requires one classifier serving observation and reward so the agent is never
"penalised for a role it was not shown". That is right, but it does not settle which *input*
the reward sees. Recommended split:

| Term group | Input | Reason |
|---|---|---|
| Physical consequence — collision, boundary, target-domain intrusion, path error, progress | **Ground truth** simulator geometry | These are physical facts. The agent should pay for hitting something whether or not it saw it |
| Rule regime — which COLREGs term is active, and its severity | **Perceived state**, identical tensor to the observation | Never penalise a role the agent was not shown |

This is also a paper point: under Study 2's degradation sweep, the agent's COLREGs
obligations degrade with its perception while its physical safety obligations do not. That
is the real situation at sea, and it makes Study 2 a study of *obligation under uncertainty*
rather than just a robustness curve.

### 3.2 `R-6` — the reward does not depend on CRI

CRI constants are still being re-derived in ship lengths (`01` open item). Gating COLREGs
terms on CRI would couple this specification to an unfinished one. Instead:

- **Engagement** is defined geometrically: `TCPA ∈ (0, T_engage]` and `DCPA < κ_eng · d_dom`.
- **Severity scaling** uses a proximity gate `ρ_t` built from the same two quantities.
- **CRI stays in the observation** exactly as specified in `01 §6.1`. It is simply not used
  in the reward.

This decouples 02 from 01's open item entirely, and removes a class of bug where a change to
the CRI constants silently re-weights the reward.

---

## 4. Reward architecture

```
r_t =   w_pf     · r_pf          # path following (penalty form)
      + w_prog   · r_prog        # progress (positive, telescoping)
      + w_exist  · r_exist       # existence cost
      + w_smooth · r_smooth      # action smoothness
      + w_obs    · r_obs         # static obstacle proximity
      + w_bnd    · r_bnd         # channel boundary
      + w_dom    · r_dom         # target ship-domain intrusion
      + w_COL    · r_col         # COLREGs group, clipped to [-1, 0]
      + r_term                   # terminal
```

**Every dense term is normalised to `[-1, 0]` before weighting**, except `r_prog` which is
`[-1, +1]`. This is the direct fix for the Paper 2 failure: the weight *is* the maximum
per-step contribution, so the magnitude hierarchy holds by construction and the audit in §8
is a check on realised severity rather than a hunt for a hidden scale factor.

**Why only progress is positive.** `Σ_t r_prog` telescopes to `(s_T − s_0)/(U_ref·Δt)`, a
constant fixed by path length regardless of episode duration. So the positive term cannot be
farmed by loitering: extra steps accrue only penalties. This removes the need for a large
existence cost to suppress stalling, which in Paper 2 was doing that job at the price of
fighting Rule 8(e).

---

## 5. Task and safety terms

### 5.1 `r_pf` — unified path following (replaces terms 2 + heading)

```
ẽ_y  = e_y / (W_local / 2)                              # width-normalised, ∈ [-1, 1]
χ̃*   = ω_LA · χ̃_LA + (1 − ω_LA) · χ̃
q_t  = w_e · exp(−γ_e · ẽ_y²) + (1 − w_e) · (1 + cos χ̃*)/2      ∈ [0, 1]
g_u  = clip( max(u, 0) / U_ref_eff , 0, 1 )
r_pf = −(1 − g_u · q_t)                                          ∈ [-1, 0]
```

`ω_LA = 0.25`, `w_e = 0.70`, `γ_e = 4.0`.

**Width normalisation is not cosmetic.** Paper 2 used `exp(−0.05·|e_y|)`, inherited from a
60 × 150 m map, which over a 10 m channel varies by under 10% of its own value — the term
was almost constant. Normalising by local half-width fixes that *and* keeps the term's range
identical across the Study 1 sweep, so reward scale is comparable at 10 m and at 3.5 m. Without
it, the effective path-following gradient changes with corridor width and confounds Study 1.

**Penalty form, not reward form.** In penalty form the speed gate works in the right
direction: stopping gives `g_u = 0` and therefore the maximum penalty.

### 5.2 `r_bnd` — channel boundary

```
r_bnd = −[ max(0, 1 − d_b / d_safe) ]²        ∈ [-1, 0]
```

`d_b` = minimum distance from the **hull polygon** to the boundary polygon, from the map,
ground truth. `d_safe = 0.50 m`.

Changed from Paper 2's 0.7 m. Required by the invariant in §2.1: the compliant narrow-channel
manoeuvre places OS centre 0.65 m from the wall (0.40 m hull-to-wall), so a 0.7 m hinge would
be permanently active during exactly the manoeuvre the paper is trying to elicit.
**Assert `d_safe < c_wall − B/2` in the config validator.**

### 5.3 `r_dom` — target ship-domain intrusion (new; not in the six carried terms)

```
r_dom = −[ max(0, 1 − d_TS / d_dom(β_TS)) ]²    ∈ [-1, 0]
```

`d_TS` = hull-to-hull distance to the target, ground truth. `β_TS` = bearing of the target
relative to the OS bow, so the asymmetric domain is applied correctly.

**This term is an addition to the doc's list and it is not optional.** The metric set in
`00 §4.2` reports ship-domain intrusion rate and depth, but the six carried terms plus the
five COLREGs terms contain no dense signal for proximity to the target — the only feedback
would be the terminal collision penalty. A reported metric with no corresponding reward
signal is the exact pattern that produced Paper 2's concessions.

### 5.4 `r_obs` — static obstacle proximity

Shifted exponential, so it is exponential in shape (as specified) but exactly zero beyond a
cut-off rather than carrying a constant background penalty:

```
ε_cut = exp(−d_cut / d_oa)
r_obs = −clip( (exp(−d_clear / d_oa) − ε_cut) / (1 − ε_cut), 0, 1 )     ∈ [-1, 0]
```

`d_oa = 0.60 m`, `d_cut = 2.00 m` → `ε_cut = 0.036`.
Values: `d_clear = 2.0 → 0`; `1.0 → −0.16`; `0.5 → −0.41`; `0.2 → −0.71`; `0 → −1`.

`d_clear` = minimum hull-to-hull clearance over static obstacles **within ±135° relative
bearing**, matching the `c_t` swath. The agent is never penalised for proximity it cannot
observe, and a passed obstacle astern does not generate a signal against an action space
with no reverse.

The unshifted exponential was rejected: with `d_oa = 0.8 m` it evaluates to −0.08 at 2 m,
which over a 300-step episode integrates to roughly −53 at `w_obs = 2.2` — larger than the
path-following term, and constant, so it carries no gradient. Same failure mode as Paper 2
in the opposite direction.

### 5.5 `r_prog` — progress

```
r_prog = clip( (s_t − s_{t−1}) / (U_ref · Δt), −1, 1 )      ∈ [-1, 1]
```

`s` = **along-path arclength**, not distance-to-goal. Arclength is correct in a bending
channel; distance-to-goal penalises the outside of a bend.

### 5.6 `r_smooth` — action smoothness

```
r_smooth = −σ_t · clip( (Δa_δ / κ_δ)² + w_n · (Δa_n / κ_n)² , 0, 1 )     ∈ [-1, 0]
```

`Δa` = per-step change in the normalised action. `κ_δ` = the actuator's per-step rate limit
in normalised units, so the term saturates at exactly the physical rate limit and is
self-calibrating when 05 delivers the actuator model. `κ_n = 0.30`, `w_n = 0.5`.

**`σ_t` resolves the Rule 8 tension (`02 §4.3`).** Rule 8(b) requires one large alteration
and forbids a succession of small ones; a plain smoothness penalty suppresses both. So:

```
σ_t = σ_enc   if  0 ≤ (t − t_engage) < N_free    else 1.0
σ_enc = 0.25,  N_free = 20 steps (2.0 s)
```

The first two seconds after engagement are cheap, so the large committed alteration is
affordable; everything after is charged at full rate, so dithering is not. This is the
narrowest intervention that makes the two rules compatible, and it is directly testable —
first-action magnitude is already a reported metric (`00 §4.2`).

### 5.7 `r_exist` and terminals

```
r_exist = −1                                       (constant)

r_term  = R_goal      = +100   on reaching the goal
        = R_collision = −300   on collision with static obstacle, boundary, or target
        = 0                    on timeout — truncation with value bootstrapping
```

`R_collision = −300` rather than the −200 in `02 §4.1`. See `R-7` in §11 for the arithmetic:
−200 leaves only a 32-point margin between a collision episode and a maximally non-compliant
one, which violates the §7 requirement that a compliant collision be worse than a
non-compliant near-miss.

Collision penalty is **uniform across the three collision types**. They are reported
separately as metrics but not weighted separately, so the reward makes no claim about their
relative severity that the paper would then have to defend.

---

## 6. COLREGs terms

### 6.1 Encounter state machine

The classifier in `01 §5.3` supplies a per-step class with internal hysteresis. The reward
adds a **latch** on top, because the Rule 8 accumulator needs a stable reference point.

```
IDLE      → ENGAGED   when class ≠ none  ∧  TCPA ∈ (0, T_engage]  ∧  DCPA < κ_eng · d_dom
                      latch: class, ψ_engage, u_engage, t_engage, TCPA_engage
ENGAGED   → CLEARING  when TCPA < 0 ∧ range opening,  or  DCPA > κ_rel · d_dom
CLEARING  → IDLE      after N_clear steps
```

`T_engage = 25 s`, `κ_eng = 2.0`, `κ_rel = 3.0`, `N_clear = 30 steps`.
A class change while ENGAGED requires the new class to persist `N_switch = 10` steps before
the latch updates. Only relevant in evaluation, where targets are reactive.

Proximity gate, shared by the geometry terms:

```
ρ_t = clip( 1 − DCPA / (κ_eng · d_dom), 0, 1 )      ∈ [0, 1]
```

### 6.2 `v_port` — port turn while give-way (head-on, crossing)

Yaw rate, not rudder angle (locked principle 5).

```
v_port = ρ_t · clip( max(0, −r) / r_ref , 0, 1 )
         · 1[ class ∈ {head_on, crossing} ]  · 1[ ENGAGED ]  · 1[ d_bnd_stbd ≥ d_safe ]
```

`r_ref = 0.20 rad/s` (≈11.5 °/s), `r_dead = 0.02 rad/s` deadband below which no turn is
registered. Both provisional pending the turning-circle identification in 05.

**The `d_bnd_stbd ≥ d_safe` suppression is `R-3`.** When OS is already hard against the
starboard boundary, a port turn is not a COLREGs choice, it is a boundary constraint. Without
the carve-out the agent is in an unwinnable state — penalised by `r_bnd` for staying and by
`v_port` for leaving — and unwinnable states in a shaped reward produce exactly the kind of
degenerate policy that Paper 2's staged repairs kept hitting.

The port-turn penalty applies to **both** crossing sub-cases. Working the geometry through:
for a target on the port bow and for a target on the starboard bow, the compliant actions are
identical — starboard alteration and/or speed reduction, never port, never crossing ahead.
That is what justifies collapsing port and starboard crossing into a single observation class
(`01 §5.3`, required modification 1), and it should be stated as a derived result rather than
asserted.

### 6.3 `v_bow` — crossing ahead of the target (crossing, overtaking)

Evaluated at the constant-velocity projected CPA, consistent with how DCPA/TCPA are computed.

```
β_CPA    = bearing of OS from TS at the projected CPA, relative to ψ_TS
σ_bow    = clip( (cos β_CPA − cos β_bow) / (1 − cos β_bow), 0, 1 )      β_bow = 67.5°
v_bow    = ρ_t · σ_bow · clip( 1 − DCPA / d_req, 0, 1 )
           · 1[ class ∈ {crossing, overtaking} ] · 1[ ENGAGED ]
```

Smooth in `β_CPA`, so there is no discontinuity at the beam. For overtaking this catches
cutting back across the bow after the pass, which is the characteristic overtaking violation.

A realised (rather than projected) version of the same quantity is computed at the actual
minimum range and logged as the reported metric. Do not use the realised version in the
reward — it is only available after the fact.

### 6.4 `v_side` — wrong-side passing (head-on, overtaking)

```
y_rel_CPA = starboard-positive lateral offset of TS from OS at the projected CPA, OS body frame

head-on:      v_side = ρ_t · clip( y_rel_CPA / d_req, 0, 1 )
              # port-to-port requires TS to port, i.e. y_rel_CPA < 0

overtaking:   side_ok = A_stbd on the chosen side (see §2.1)
              v_side  = ρ_t · (1 − side_ok) · clip( 1 − |y_rel_CPA| / d_req, 0, 1 )
              # preferred side = greater available room; tie broken to TS's port side
```

Not applied to crossing: there the required geometry is "pass astern", which `v_bow` already
covers. Adding a side term for crossing would double-count.

### 6.5 `v_hold` — course-keeping while being overtaken (Rule 17(a)(i))

Penalty on deviation rather than a reward for holding, so the whole COLREGs group is a
penalty group and the magnitude hierarchy in §7 is uniform. A positive hold reward would also
be partly redundant with `r_pf`, which already rewards steady course, and would risk paying
the agent to hold course into a collision.

```
v_hold = ρ_t · clip( (|r| / r_hold)² + (|u − u_engage| / Δu_hold)², 0, 1 )
         · 1[ class = being_overtaken ] · 1[ ENGAGED ] · 1[ NOT in_extremis ]

in_extremis = (DCPA < d_req) ∧ (TCPA < T_extremis)
```

`r_hold = 0.05 rad/s`, `Δu_hold = 0.10 m/s`, `T_extremis = 5 s`.

**The `in_extremis` suppression is `R-4` and it touches locked decision S5.** Rule 17(b)
requires the stand-on vessel to act when collision cannot be avoided by the give-way vessel
alone. This is a different provision from 17(a)(ii), which S5 puts out of scope. Suppressing
the hold penalty in extremis does not *reward* release — it stops punishing it and leaves the
collision and domain penalties to dominate. So the paper's claim that active release is out
of scope survives intact, while the agent is not trained to hold course into a collision.
Without this carve-out the reward contains an instruction the paper would not want to defend.

### 6.6 `v_r8` — Rule 8, unified late-and-insufficient term

One dense term replaces the two the doc sketches, because they share a gate and separating
them creates two coefficients where one suffices.

```
Δψ_stbd = max(0, ψ_t − ψ_engage) resolved to the starboard sense, unwrapped
Δu_red  = max(0, u_engage − u_t)

A_t     = Δψ_stbd / Δψ_min  +  Δu_red / Δu_min          # compliant action taken so far
urgency = clip( 1 − TCPA / T_act, 0, 1 )

v_r8    = urgency · clip(1 − A_t, 0, 1) · 1[ class ∈ give-way ] · 1[ ENGAGED ]
```

`Δψ_min = 20° = 0.35 rad`, `Δu_min = 0.24 m/s` (30% of `U_ref`), `T_act = 15 s`.
Give-way classes: `{head_on, crossing, overtaking}`.

Reads directly: the penalty is zero once sufficient compliant action has been taken, and
grows as time runs out if it has not. It counts **only compliant directions** — starboard
heading change and speed reduction — so a large port alteration does not discharge the Rule 8
obligation. Time-to-first-action and first-action-magnitude fall out as by-products of `A_t`
and are logged directly for the metric set.

`T_act = 15 s` is derived, not chosen: achieving 2.36 m of lateral offset at 0.8 m/s with a
30° alteration takes ≈5.9 s of running plus ≈3.5 s of turn-in and turn-out, so ≈10 s, plus
margin. **Constraint for doc 04:** the spawn TCPA distribution must extend well above 15 s or
the term is saturated at spawn and carries no gradient.

### 6.7 Group aggregation

```
v_col = clip( 0.55·v_port + 0.55·v_bow + 0.40·v_side + 0.45·v_hold + 0.50·v_r8, 0, 1 )
r_col = −v_col                                                          ∈ [-1, 0]
```

The group is clipped to unit range **before** the group weight is applied. This is what
guarantees the magnitude hierarchy by construction: no combination of COLREGs violations can
exceed `w_COL` per step, so the group cannot silently outrank the safety terms the way
Paper 2's path term outranked its avoidance term.

Maximum simultaneous, before clipping: head-on `port + side + r8 = 1.45`; crossing
`port + bow + r8 = 1.60`; overtaking `bow + side = 0.95`; being overtaken `hold = 0.45`.
Two concurrent severe violations saturate; one does not.

---

## 7. Coefficients and magnitude hierarchy

| Term | Range | Weight | Max per-step magnitude |
|---|---|---|---|
| Terminal collision | one-shot | — | **300** |
| Terminal goal | one-shot | — | 100 |
| `r_bnd` boundary | `[-1,0]` | `w_bnd = 3.0` | 3.0 |
| `r_dom` target domain | `[-1,0]` | `w_dom = 2.5` | 2.5 |
| `r_obs` static obstacle | `[-1,0]` | `w_obs = 2.2` | 2.2 |
| `r_col` COLREGs group | `[-1,0]` | `w_COL = 1.8` | 1.8 |
| `r_pf` path following | `[-1,0]` | `w_pf = 0.6` | 0.6 |
| `r_prog` progress | `[-1,1]` | `w_prog = 0.3` | 0.3 |
| `r_smooth` smoothness | `[-1,0]` | `w_smooth = 0.10` | 0.10 |
| `r_exist` existence | `−1` | `w_exist = 0.05` | 0.05 |

```
300  ≫  3.0 > 2.5 > 2.2  >  1.8  >  0.6 > 0.3 > 0.10 > 0.05
collision ≫ ——— safety ———  >  COLREGs  >  ——— task ———
```

This satisfies `02 §5` exactly, and because every term is unit-normalised the ordering is a
property of the coefficient table rather than something that has to be discovered empirically.

**The hierarchy is checked two ways, and the distinction matters.** `02 §5` and `02 §6` ask
for slightly different things and they can conflict, because the terms have very different
natural durations — a boundary excursion lasts a few seconds, a COLREGs violation lasts a
whole encounter. So:

1. **Instantaneous ordering** — the table above. Holds by construction. Assert in a unit test.
2. **Episode-integrated audit** — §8. Checks realised severity, and is allowed to show a
   different ordering, because a term that is rarely active *should* integrate small. The
   audit's job is to catch a term that is 10× off its design-intended share, not to reproduce
   the instantaneous ordering.

Stating this distinction in the paper pre-empts the obvious reviewer question about why
Table R7's ordering does not match the stated hierarchy.

---

## 8. Scale audit — pre-committed numbers

Mandatory per `02 §6`. These are **predictions to be asserted against**, computed here so
that a mismatch is diagnostic rather than a surprise.

### 8.1 Predicted episode-integrated contributions

Design point: 300-step episode, 20 m path, `U_ref = 0.8 m/s`.

| Term | Nominal success, no encounter | Collision at step 150 | Max non-compliant, no collision |
|---|---|---|---|
| `w_prog · Σr_prog` | +75 (telescoping, fixed) | +37.5 | +75 |
| `w_pf · Σr_pf` | −27 | −13.5 | −40 |
| `w_obs · Σr_obs` | −26 | −13 | −26 |
| `w_exist · Σr_exist` | −15 | −7.5 | −15 |
| `w_smooth · Σr_smooth` | −1.5 | −0.8 | −4 |
| `w_bnd · Σr_bnd` | 0 | 0 | −15 |
| `w_dom · Σr_dom` | 0 | 0 | −45 |
| `w_COL · Σr_col` | 0 | 0 | −270 |
| Terminal | +100 | −300 | +100 |
| **Episode return** | **≈ +105** | **≈ −297** | **≈ −240** |

Three orderings to assert:

- `success (+105) > max-violation (−240) > collision (−297)` — a COLREGs-compliant collision
  is worse than a non-compliant near-miss (`02 §5`). Margin 57.
- Loitering to timeout ≈ −86 + bootstrap, worse than reaching the goal (+105) and better than
  colliding (−297). A cornered agent correctly prefers timeout to collision.
- Cost of compliance vs cost of violation, over a 100-step encounter:
  compliance ≈ 44 (path 25 + progress 19), violation at realised severity 0.5 ≈ 90.
  **Ratio 2.05.** If the audit returns a ratio below ~1.5, `w_COL` is too low and compliance
  will not be learned — this is the single most important number in the table.

### 8.2 Procedure

1. Instrument `info` with per-term instantaneous and episode-integrated values (§10.3).
2. Run **1,000 random-policy episodes** and **1,000 Paper 2 SAC episodes** through the new
   reward without training. Tabulate mean, median and 95th percentile per term.
3. Check every term against §8.1 with a factor-of-3 tolerance. A term outside that band means
   the coefficient is wrong regardless of what the ratios say on paper.
4. Check the three orderings in §8.1 explicitly.
5. Re-run after **any** coefficient change. Non-negotiable.
6. Report as Table R7.

Note the Paper 2 SAC policy is not COLREGs-aware, so it should show a large `Σr_col`. If it
does not, the encounter classifier or the engagement gate is not firing and the audit has
found a bug rather than a scale problem.

---

## 9. Config schema

```python
@dataclass(frozen=True)
class RewardConfig:
    # --- weights ---
    w_pf: float = 0.60
    w_prog: float = 0.30
    w_exist: float = 0.05
    w_smooth: float = 0.10
    w_obs: float = 2.20
    w_bnd: float = 3.00
    w_dom: float = 2.50
    w_col: float = 1.80

    # --- terminal ---
    r_goal: float = 100.0
    r_collision: float = -300.0
    timeout_bootstrap: bool = True

    # --- path following ---
    gamma_e: float = 4.0
    w_e: float = 0.70
    omega_la: float = 0.25
    u_ref: float = 0.80

    # --- safety geometry ---
    d_safe: float = 0.50          # boundary hinge
    c_wall: float = 0.65          # centre-to-wall admissibility clearance
    d_oa: float = 0.60            # static obstacle exponential scale
    d_cut: float = 2.00           # static obstacle cut-off
    obs_swath_deg: float = 135.0

    # --- ship domain (provisional; final from 05) ---
    dom_ahead_lpp: float = 2.00
    dom_astern_lpp: float = 1.00
    dom_abeam_lpp: float = 0.75

    # --- smoothness ---
    kappa_delta: float = None     # set from actuator rate limit at build time
    kappa_n: float = 0.30
    w_n: float = 0.50
    sigma_enc: float = 0.25
    n_free: int = 20

    # --- encounter state machine ---
    t_engage: float = 25.0
    kappa_eng: float = 2.0
    kappa_rel: float = 3.0
    n_clear: int = 30
    n_switch: int = 10

    # --- COLREGs sub-weights ---
    w_port: float = 0.55
    w_bow: float = 0.55
    w_side: float = 0.40
    w_hold: float = 0.45
    w_r8: float = 0.50

    # --- COLREGs thresholds (provisional; r_ref from 05 turning circle) ---
    r_ref: float = 0.20           # rad/s, "clearly turning"
    r_dead: float = 0.02          # rad/s deadband
    beta_bow_deg: float = 67.5
    r_hold: float = 0.05
    du_hold: float = 0.10
    t_extremis: float = 5.0
    dpsi_min_deg: float = 20.0
    du_min: float = 0.24
    t_act: float = 15.0

    # --- ablation switches (00 §4.3) ---
    colregs_terms_enabled: bool = True
    encounter_feature_enabled: bool = True     # consumed by the obs builder
    colregs_term_mask: tuple = ("port", "bow", "side", "hold", "r8")   # leave-one-out

    # --- audit ---
    log_per_term: bool = True
```

**Validator assertions** (fail at construction, not at step 10,000):

```
assert d_safe < c_wall - B/2                 # §5.2 invariant
assert w_bnd > w_dom > w_obs > w_col > w_pf > w_prog > w_smooth > w_exist
assert abs(r_collision) > w_col * max_encounter_steps    # §8.1 ordering
assert t_act < t_engage
assert d_cut > d_oa
```

---

## 10. Implementation notes for Claude Code

### 10.1 One context object, three consumers

Build a single `EncounterContext` per step, before the observation and the reward:

```python
@dataclass
class EncounterContext:
    # perceived (feeds observation AND colregs reward gating)
    cls: EncounterClass
    alpha: float; ct: float
    dcpa: float; tcpa: float; cri: float
    y_rel_cpa: float; beta_cpa: float
    # latched
    state: EncounterState          # IDLE / ENGAGED / CLEARING
    psi_engage: float; u_engage: float; t_engage: int; tcpa_engage: float
    # admissibility (from map, ground truth)
    a_stbd: bool; a_port: bool
    r_stbd: float; r_port: float; dy_req: float
    # ground truth (feeds safety reward and metrics only)
    d_ts_true: float; dcpa_true: float
```

Observation builder, reward, and metrics logger all read from this one object. This is the
mechanical guarantee behind `01 §5.3`'s "one module, two consumers" requirement — if they
each recompute, they will diverge at sector boundaries eventually.

### 10.2 Module layout

```
paper_pooling/src/
  colregs/
    classifier.py      # 5-class + hysteresis  (owned by 01, consumed here)
    geometry.py        # CPA, domain, A_stbd/A_port, beta_cpa, y_rel_cpa
    context.py         # EncounterContext + state machine
  reward/
    terms.py           # each term as a pure function returning value ∈ [-1, 0]
    reward.py          # weighted sum, clipping, group aggregation
    audit.py           # per-term accumulators, Table R7 generator
```

Every term is a pure function of `(state, ctx, cfg)` returning a scalar in its declared range.
This makes the leave-one-out ablation a mask over a dict rather than a set of `if` branches
scattered through the step function, and it makes the unit tests in §10.4 trivial.

### 10.3 Logging

Emit in `info` every step:

```
reward/term/<name>          instantaneous, pre-weight
reward/weighted/<name>      instantaneous, post-weight
reward/episode/<name>       running integral, post-weight
colregs/class               int
colregs/state               int
colregs/a_stbd              bool
colregs/action_ratio        A_t
metrics/time_to_first_action, metrics/first_action_magnitude
metrics/min_dcpa, metrics/domain_intrusion_depth, metrics/passing_side_correct
```

The metric set in `00 §4.2` should be a **read** of these keys, not a separate computation.
Paper 2's concessions came from metrics that were not designed in before the campaign ran.

### 10.4 Unit tests to write before any training

1. Every term returns a value inside its declared range across 10⁵ random states.
2. Coefficient ordering assertion (§9).
3. `Σ r_prog` over a full traversal equals `L_path / (U_ref · Δt)` to within float tolerance.
4. Head-on with a compliant starboard alteration → `v_port = v_side = 0`, `v_r8 → 0` after
   the alteration completes.
5. Head-on with a port alteration → `v_port > 0` within `r_ref`-scaled steps of the yaw rate
   crossing `r_dead`, **not** at the rudder reversal (locked principle 5 — write this as an
   explicit regression test, it is the exact bug the principle exists to prevent).
6. `A_stbd` returns `False` at `W = 3.5 m` and `True` at `W = 10 m` for a centreline head-on.
7. `in_extremis` suppression: `v_hold → 0` when `DCPA < d_req ∧ TCPA < 5 s`.
8. `d_safe < c_wall − B/2` invariant.

### 10.5 Propulsion authority — resolves `02 §4.4`

**Widen it.** Rule 8(e) makes slackening speed a legal avoidance action, and the precedence
table in §2.2 makes it the *primary* action in three of four narrow-channel cases. If the
agent cannot slow down, the compliant behaviour is unreachable and the paper's central claim
is untestable.

- Reachable surge range at the final curriculum stage: `u ∈ [0.20, 0.90] m/s` (≈0.25–1.15 `U_ref`)
- `n_min = 0` reachable; **no reverse** — the model is not identified in reverse and the
  action space has none
- Staged widening in the curriculum as in Paper 2, but stage 5 must expose the full range
- `U_ref_eff` in the speed gate (§5.1) drops to `0.4 · U_ref` when a give-way obligation is
  active and `A_stbd` is false, so a legal slowdown does not cost path-following reward.
  This is `R-2` and without it the agent structurally cannot learn Rule 8(e)
- Note for doc 04: the Paper 2 frozen baseline (M1) runs with its own action mapping. The
  widened range is not a fair-comparison problem, but it must be stated

### 10.6 Narrow-channel overtaking — `R-5`

The correct behaviour when `class = overtaking ∧ ¬A_pass` is to fall in astern and hold
station. That behaviour is unreachable under the default terms: no progress, full path
penalty via the speed gate, existence cost accruing, likely timeout. So:

```
if class == overtaking and not (A_stbd or A_port):
    U_ref_eff = max(u_TS, 0.2)      # matching the target's speed satisfies the gate
    w_exist_eff = 0.0               # holding station is not wandering
```

The bow-crossing and side terms still penalise attempting the pass. Without this the narrow
overtaking case is not a test of COLREGs reasoning, it is a test of whether the agent can
tolerate an unwinnable reward — and it will resolve it by overtaking anyway.

---

## 11. Decisions for sign-off

| # | Decision | Recommendation | Consequence if rejected |
|---|---|---|---|
| **R-1** | Reward input: ground truth for physical terms, perceived state for COLREGs gating | Adopt | Either the agent is penalised for roles it was not shown, or it is not penalised for collisions it did not see |
| **R-2** | `U_ref_eff` drops when speed reduction is the admissible give-way action | Adopt | Rule 8(e) compliance is structurally unlearnable — the speed gate punishes the compliant action |
| **R-3** | Port-turn penalty suppressed when hard against the starboard boundary | Adopt | Unwinnable state; degenerate policy |
| **R-4** | `in_extremis` suppression of the course-keeping penalty (Rule 17(b), **not** 17(a)(ii)) | Adopt | The reward instructs the agent to hold course into a collision. Touches locked decision S5 — needs an explicit sentence in the paper distinguishing 17(b) from 17(a)(ii) |
| **R-5** | Existence cost and speed gate relaxed for "hold astern" in narrow overtaking | Adopt | Narrow overtaking becomes untestable |
| **R-6** | Reward gated geometrically, not on CRI | Adopt | 02 blocks on 01's unfinished CRI constant re-derivation |
| **R-7** | `R_collision = −300`, not −200 | Adopt | At −200 the margin between a collision episode and a maximally non-compliant one is 32 points, violating `02 §5` |

### Still needs a number from elsewhere

- `r_ref` (yaw rate reference) and `r_dead` — from the turning-circle identification in 05
- Final ship domain — from measured advance and tactical diameter, 05
- `κ_δ` — from the calibrated actuator rate limit, 05
- Head-on band width — 01 open item; the reward inherits whatever the classifier defines
- Target lateral offset distribution — 04, and it moves the head-on threshold by a factor of
  ~1.6 (§2.4)

### Downstream, now unblocked

- **01** — classifier definition and the five-class table are fixed by §6.1 and §2.2
- **04** — width sweep levels should resolve three thresholds at 13.0 B, 8.3 B and 7.3–12.0 B,
  not the single transition at 7.3 B currently in doc 03 §5; spawn TCPA must extend above
  `T_act = 15 s`
