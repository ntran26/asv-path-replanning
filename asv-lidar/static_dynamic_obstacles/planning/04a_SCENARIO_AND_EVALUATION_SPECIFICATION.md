# 04a — Scenario Generation and Evaluation Suite: Specification

**Revision 2.0** — first full specification. Expands `04_SCENARIOS_AND_EVALUATION.md` into an
implementable spec and closes its six open items.
**Handover target:** Claude Code (§3, §4, §9), Claude chat (§2 findings need your sign-off first)
**Depends on:** 02 §3.2 precedence structure, 01 §5.2–5.3 domain and classifier, 03 §2–§5 targets
and corridor
**Status of numbers:** every derived threshold below is a function of the ship domain and the
pose-noise characterisation. Both are provisional. All are written as formulae with the current
inputs substituted, so they recompute when 05 lands.

---

## 1. Four findings that change the design

These came out of writing the generator's feasibility constraints. Three of them change decisions
already recorded elsewhere. Read this section before §3.

### 1.1 The binding width threshold is crossing, not head-on — and it sits at 7.6 m

Document 04 §5 frames Study 1 as sweeping "down to the point at which the Rule 14 starboard
alteration no longer fits". Working the geometry through, Rule 14 is the *last* manoeuvre to become
inadmissible, not the first.

With own ship stationed under Rule 9(a) at the starboard quarter-width, the starboard room available
for an evasive alteration is `W/4 − w_wall`. A meaningful give-way alteration displaces the vessel by
at least one abeam domain semi-axis, so:

```
W_crossing  ≥  4 · (a_abeam + w_wall)  =  4 × (1.25 + 0.65)  =  7.60 m   (15.2 B)
```

Head-on and overtaking both reduce to two vessels abreast, which needs only:

```
W_headon    ≥  2 · a_abeam + 2 · w_wall  =  2.50 + 1.30      =  3.80 m   (7.6 B)
```

Overtaking adds an exposure margin, because the abreast configuration is held for
`t_exp = (a_ahead + a_astern + Lpp)/(U_OS − U_TS) ≈ 22.8 s` at a 0.5 speed ratio, over which pose
error accumulates:

```
W_overtake  ≈  W_headon + 2 · ρ_pose · t_exp  =  3.80 + 0.46  =  4.26 m   (8.5 B)
```

**Predicted ordering: crossing (7.60 m) ≫ overtaking (4.26 m) > head-on (3.80 m).**

Two consequences:

- The sweep in 03 §5 (10, 8, 6, 5, 4, 3.5 m) brackets the head-on transition well and steps straight
  over the crossing transition between 8 m and 6 m. **Add a 7 m and a 4.5 m level.**
- The headline Study 1 figure is the *crossing* transition — the width at which give-way stops being
  executed by alteration and starts being executed by speed reduction under 8(e). That is a
  behavioural switch, which is a stronger result than the success-rate decline the current framing
  implies, and it lands inside the sweep rather than at its edge.

**Flag:** an earlier note recorded the ordering as crossing > head-on > overtaking. The derivation
above puts overtaking above head-on, on the exposure-margin argument. Check it against 02a before
either number goes into the precedence table. If the exposure margin is dropped, head-on and
overtaking share a threshold exactly, which is also a defensible position — but it should be a
decision, not an accident.

### 1.2 The ship domain currently has two incompatible readings, worth 1.2 m of channel width

03 §5 computes the head-on minimum as `2 × 1.18 + 2 × 0.65 = 3.66 m`, which treats the ship domain
as governing vessel-to-vessel separation only, with a separate and smaller physical clearance to the
wall. The alternative reading — the domain is inviolable by *anything*, walls included — gives
`1.25 + 2.50 + 1.25 = 5.00 m`. That is one of the sweep levels, and it would move the head-on
transition out of the bracket entirely.

**Recommendation: adopt the two-constraint convention explicitly and defend it.** The ship domain
encodes the room needed to avoid a *moving* vessel whose next action is uncertain. A channel wall is
static and known from the chart, so the margin it requires is different in kind and smaller in
magnitude — dominated by localisation error, not by the other vessel's freedom of action. Define:

```
a_abeam  = max(0.75 · Lpp, d_lidar_deadzone + margin)   = 1.25 m   [vessel-to-vessel]
w_wall   = B/2 + 3σ_pose + control margin               = 0.65 m   [vessel-to-boundary]  TODO(04-1)
```

`w_wall` then becomes a *measured* quantity out of 05 rather than a chosen one, which is the better
position to be in. State the two-constraint convention in the methods section in one sentence; it is
cheap to declare and awkward if a reviewer derives the 5.00 m figure independently.

### 1.3 A 90° crossing is not realisable with a channel-confined target — Rule 9(d) fixes it

03 §3 requires the target to respect the channel and not pass through walls. Applied to a crossing
target in a straight corridor, that constraint is unsatisfiable: a vessel crossing a 4 m fairway at
90° runs into the far wall. Generating crossings on bends instead needs a heading change of at least
67.5° to reach the crossing band, and an L-bend of that angle does not fit a 10 m-wide basin.

The resolution is in the rulebook rather than the geometry. **Rule 9(d) is specifically about vessels
crossing a narrow channel** — "a vessel shall not cross a narrow channel or fairway if such crossing
impedes the passage of a vessel which can safely navigate only within such channel". The crossing
target is not a channel user. It is a small craft transiting *across* the fairway, which is precisely
the situation 9(d) addresses.

**Revise the constraint:** the *own ship* is confined to the channel in all classes. The *target* is
confined to the channel in head-on, overtaking, being-overtaken and null, and crosses it freely in
the crossing class.

This is field-reproducible without modification. The corridor is virtual — a map polygon enforced in
software (01 §3.4) — inside a 10 m basin. A target vessel can physically cross a 4 m virtual corridor
and continue into water the own ship treats as non-navigable. It also tightens the Rule 9 argument:
9(b) carries head-on and overtaking precedence, 9(d) carries crossing. Both are cited, neither is
stretched.

### 1.4 Spawning outside sensor range is only possible for head-on

03 §3 requires the target to spawn beyond `D_max` so that acquisition is part of the task. For
head-on that is comfortable. For every other class it is geometrically impossible in a 25 m basin,
because closing speed is a difference rather than a sum:

| Class | Closing speed | Required spawn range for a completed encounter | Feasible outside `D_max` = 12 m? |
|---|---|---|---|
| Head-on | `U_OS + U_TS` ≈ 1.10 m/s | 13.8 – 19 m | **Yes** |
| Crossing | ≈ 0.78 m/s at CT = 90° | target's whole track is 10 m (basin width) | No — bounded by basin width |
| Overtaking | `U_OS − U_TS` ≈ 0.28 m/s | `R_0/(1−k) ≤ 16 m` ⇒ `R_0 ≤ 8 m` at k = 0.5 | No |
| Being overtaken | `U_TS − U_OS` ≈ 0.44 m/s | `R_0 ≤ 6 m` (channel astern of own start) | No |

Forcing out-of-range spawning everywhere would need a target moving at under 0.16 m/s for
overtaking — below steerageway for a model hull — or a corridor two to three times the basin length,
which forfeits the physical-reproducibility argument that O4 was resolved to protect.

**Specify spawn range per class (§3.4) and report track acquisition range per class.** It will be
structurally different across classes, and that is a property of confined-water encounter geometry,
not a defect in the perception pipeline. One sentence in the results pre-empts the question.

**Consequence for the being-overtaken class:** the target must run at 1.5–2.2 × own-ship speed, i.e.
0.83–1.21 m/s. Whether the target platform can do that is open (05). If it cannot, run field
being-overtaken trials at a reduced own-ship reference speed of ≈0.35 m/s rather than dropping the
class. Acquisition range astern is 4–6 m and further degraded by the aft self-occlusion sector
(01 §2.3), so this is the class most likely to fail in the field for perception rather than policy
reasons — which makes it the most informative one to run.

---

## 2. Decisions closing 04 §9

| Open item | Decision |
|---|---|
| Episodes per Tier B cell | **25**, over 39 cells = 975 episodes per policy per seed. Evaluation is not the compute bottleneck (§8.3); do not cut this to buy training runs |
| Corridor width thresholds | **Derived in §1.1.** Wide ≥ 7.60 m, intermediate 4.26–7.60 m, narrow 3.50–4.26 m. Strata are defined by the predicted governing rule, not by round numbers |
| Tier A case list | **§5, 34 named cases** |
| Presentation format for multi-axis results | **One pre-registered primary endpoint — compliant success rate — plus a per-axis table and a two-panel Pareto. No weighted scalar** (§7.1) |
| Channel-constrained "Around the Clock" placement | **Main text, both variants in one figure.** The open-water/constrained contrast *is* the evidence for N2; splitting it across the appendix wastes it (§4.3) |
| Comparator list | **Protected core plus a pre-committed priority tail, gated on measured throughput** (§8) |

---

## 3. Scenario generator

One generator, three consumers (training distribution, frozen suite, sweeps), one seed namespace
scheme (§9.2).

### 3.1 Sampling order

1. Sample encounter class `k ∈ {head-on, crossing, overtaking, being overtaken, null, no-target}`
2. Sample corridor geometry: width profile, bend, path offset (§3.2)
3. Sample target kinematics: heading intersection angle `CT`, speed ratio, desired `DCPA`, desired
   spawn `TCPA` — all from class-conditional intervals (§3.4)
4. **Solve backwards** for the spawn position (§3.3)
5. Validate against class bands, channel containment, and range rules; reject and resample (§3.5)
6. Place static obstacles subject to non-interference constraints (§3.6)
7. Emit a scenario record (§9.1)

### 3.2 Corridor geometry

The boundary branch only earns its 7 dimensions when width varies, the path is off-centre, or the
channel bends (01 §3.3). This is a hard requirement on the generator, not a nicety.

| Parameter | Range | Notes |
|---|---|---|
| Nominal width `W` | 3.50 – 10.00 m (7 – 20 B) | Stratum-conditional in Tier B; swept in Study 1 |
| Width variation `W_max/W_min` | 1.0 – 1.8 | Piecewise-linear over 2–4 control points along `s` |
| Bend total heading change | 0 – 60° | **≥ 40% of episodes with \|Δψ_path\| ≥ 20°** |
| Path lateral offset | −0.30 – +0.30 of local half-width | Positive = toward starboard wall; **mean offset positive**, per Rule 9(a) |
| Corridor length | 25 m; usable spawn span 21 m | Matches basin (O4) |
| Reference path length | 20 m | Unchanged from Paper 2 |

**Assertion for the test suite:** in any batch of 1000 training episodes, the correlation between
`e_y` and each boundary ray must be below 0.9 in magnitude. If it is not, the corridor is effectively
constant-width and centred, and the branch is redundant regardless of what the config says.

### 3.3 Backward solve

Given own-ship pose `p_OS(0)`, path tangent `ĥ_OS`, and sampled `(CT, U_TS, DCPA d_0, TCPA T_0, side σ)`:

```
ψ_TS      = ψ_OS + CT
v_rel     = U_OS · ĥ_OS − U_TS · ĥ_TS ,     V = |v_rel|
n̂         = σ · perp(v_rel) / V                     σ ∈ {−1, +1}, passing side
p_TS(T_0) = p_OS(0) + U_OS · T_0 · ĥ_OS + d_0 · n̂
p_TS(0)   = p_TS(T_0) − U_TS · T_0 · ĥ_TS
R_0       = |p_TS(0) − p_OS(0)|
```

Then recompute `(α, CT, DCPA, TCPA)` forward from the resulting state and confirm they land in the
intended class bands (01 §5.3). This round-trip check is not optional — the sampled `CT` and the
realised `α` are independent, and a fraction of draws will produce a different class than intended,
particularly near band edges and on bends.

### 3.4 Class-conditional intervals

`U_OS,nom = 0.55 m/s` (field-measured, authoritative). `k = U_TS/U_OS`.

| Class | `CT` | `k` | Spawn `R_0` | `TCPA T_0` | `DCPA d_0` | Spawn vs `D_max` |
|---|---|---|---|---|---|---|
| Head-on | 180° ± 10° | 0.7 – 1.3 | 13.8 – 19 m | 12.5 – 17.3 s | 0 – 2.0 m | **Outside** |
| Crossing | 67.5 – 175° and 185 – 292.5° | 0.6 – 1.4 | basin-bounded, 5 – 10 m | 8 – 15 s | 0 – 2.5 m | Inside (declared) |
| Overtaking | ±67.5° | 0.25 – 0.55 | 6 – min(12, 16(1−k)) m | 20 – 32 s | 0 – 2.0 m | Inside |
| Being overtaken | ±67.5° | 1.5 – 2.2 | 4 – 6 m | 10 – 16 s | 0 – 2.0 m | Inside, astern |
| Null | ±20° | 0.85 – 1.15 | 8 – 15 m | — (no CPA in horizon) | > 4 m | Either |
| No target | — | — | — | — | — | — |

Head-on `T_0` bound: `R_0 = (U_OS + U_TS) · T_0`, `R_0 ≤ 19 m` so both vessels fit the 21 m usable
span, and `R_0 ≥ 1.15 · D_max` for out-of-range spawn. `D_max` is `TODO(04-2)` — RPLidar C1 nominal
is 12 m; verify against retained field logs, and note that the matte-black wall may cap effective
range on one side (05).

**Null class is mandatory.** A target on a similar course at a similar speed never emerges from a
class-conditional spawner but is common in practice, and it is the case where a policy that has
learned "target present ⇒ manoeuvre" will visibly overreact.

**No-target episodes: 15–20% of training.** Below that the static-only configuration drifts out of
distribution and the Paper 2 comparison degrades for the wrong reason (01 §6.2).

### 3.5 Rejection accounting is a result, not a diagnostic

Log rejections by `(class, width, reason)`. The rejection rate per cell is an analytic feasibility
measure — it says at what width each encounter class stops being constructible at all, entirely
independently of any policy's performance. Plotted against width alongside the Study 1 outcome
curves, it separates "the method fails here" from "the geometry is infeasible here", which is
exactly the distinction §4.4 of the parent document asks for and the strongest available answer to
"you designed the benchmark to produce the conclusion."

Cap at 200 attempts per scenario; record the cap-out rate.

### 3.6 Static obstacles

0–3 panels, placed after the target track is fixed.

- **Default:** no obstacle intersects either vessel's swept ship domain within `±0.4 · T_0` of CPA.
  Without this, clutter silently nullifies the encounter and the cell measures nothing.
- **Conflict cases (flagged):** exactly one obstacle placed inside the own ship's compliant manoeuvre
  corridor, forcing the Rule 8(e) fallback. Tagged `conflict=1` in the record.
- **Occlusion cases (flagged):** one obstacle placed on the OS→TS line of sight for a specified
  duration `t_occ ∈ {1, 2, 4} s`. Occlusion becomes a controlled variable rather than an accident,
  which is what makes it usable as a Study 2 axis and as the evidence for or against recurrence
  (01 §6.3).
- Minimum cluster size must exceed the suspension-line rejection threshold (03 §4a).

### 3.7 Curriculum parameters

| Stage | Width | Target | Spawn TCPA | Clutter | Steps |
|---|---|---|---|---|---|
| 1 | 8 – 10 m, constant | none | — | 0 – 1 | `TODO(04-3)` |
| 2 | 5 – 10 m, varying + bends | none | — | 0 – 3 | |
| 3 | 7 – 10 m | one, CV, head-on / null only | upper half of range | 0 – 1 | |
| 4 | 4.5 – 10 m | one, CV, all classes | full range | 0 – 2 | |
| 5 | 3.5 – 10 m | one, CV, all classes | full range | 0 – 3 | |

Stage 3 restricted to head-on and null deliberately: it is the only class pair where the compliant
response is available at every width, so the agent learns the encounter machinery before it meets a
geometry where the textbook manoeuvre is inadmissible.

---

## 4. Frozen evaluation suite

### 4.1 Episode horizon — increase to 900 steps

700 steps at 0.1 s is 70 s. Own ship at 0.55 m/s covers the 20 m path in 36 s, but Rule 8(e) is now a
*designated compliant behaviour* (02 §4.4): a correct narrow-channel give-way may involve slowing to
0.2 m/s for 20 s or more. A horizon tight enough to turn compliant slowing into a timeout puts the
horizon in direct conflict with the reward design.

**Set the horizon to 900 steps (90 s) for all cells, training and evaluation.** Fixed rather than
width-conditional, so timeout rates remain comparable across the sweep. Report timeout separately
from failure — under this reward they mean different things. Cost: +28.6% training wall-clock, which
is accounted for in §8.

### 4.2 Structure

| Component | Cases | Rollouts per case per seed | Purpose |
|---|---|---|---|
| Tier A — named | 34 | 10 | Trajectory figures, per-case commentary |
| Tier B — stratified holdout | 39 cells × 25 | 1 | Aggregate statistics with CIs |
| Around the Clock — open water | 24 | 10 | Literature comparability |
| Around the Clock — channel | 24 | 10 | N2 evidence |

Ten rollouts per named case despite a deterministic policy: the *environment* is stochastic once
perception noise is injected, which is the whole point of N1. A single rollout of a named case
reports one draw from the perception noise, not the behaviour.

### 4.3 Tier B strata

| Stratum | Levels |
|---|---|
| Encounter class | 5 — head-on, crossing, overtaking, being overtaken, null |
| Target behaviour | 3 — constant velocity, compliant reactive, non-compliant. Null takes CV only |
| Corridor width | 3 — wide `U[7.60, 10.00]`, intermediate `U[4.26, 7.60]`, narrow `U[3.50, 4.26]` |
| Static clutter | 0–3, sampled within cell, not a stratum |

`4 × 3 × 3 + 1 × 3 = 39` cells. At 25 episodes, 975 per policy per seed.

Width strata are cut at the derived thresholds (§1.1) rather than at round numbers, so each stratum
has a distinct predicted governing rule: wide = alteration admissible for all classes; intermediate =
crossing resolves under 8(e) while head-on and overtaking still alter; narrow = only head-on
channel-keeping and 8(e) remain. If the measured behaviour matches that partition, the precedence
table is validated by the data rather than asserted.

### 4.4 Around the Clock

24 constellations, `φ_TS,j = (j/25)·2π`, `j = 1…24`, both vessels set to meet at the origin
(Waltz & Okhrin). Run as-published in open water for comparability, then with corridor walls at each
of three widths (10, 6, 4.26 m).

**Report both variants in the main text as a single figure** — a 24-spoke polar layout with the
open-water outcome on the inner ring and the constrained outcomes on outer rings. The whole argument
of N2 is what changes between them; separating them across a section boundary makes the reader do
the comparison from memory. Re-derive the spawn radius in ship lengths against `Lpp = 1.57 m`
rather than adopting their 2 NM scaling.

### 4.5 The deliberate-failure stratum

Narrow width with a non-compliant target is the natural candidate and it must be *reported*, not
merely present. Two Tier A cases (§5, `A-FAIL-*`) and the narrow × non-compliant Tier B cells carry
it. Pre-commit in the claim ledger that this stratum is reported at whatever level it comes out.

Framing rule, applied throughout: no stratum is described as "challenging for classical methods".
Strata are described by geometry only. Degradation of any method is a result.

---

## 5. Tier A named cases

Width codes: `W` = 10.0 m (wide), `I` = 6.0 m (intermediate), `N` = 4.0 m (narrow),
`X` = 3.5 m (sub-threshold). Behaviour: `cv` constant velocity, `re` compliant reactive,
`nc` non-compliant. All cases constant-velocity unless marked.

| ID | Class | Width | Variant |
|---|---|---|---|
| A-HO-W / -I / -N | Head-on | W, I, N | Baseline, target compliant with 9(a) |
| A-CRS-W / -I / -N | Crossing from starboard | W, I, N | Target crosses fairway under 9(d) |
| A-CRP-W / -I / -N | Crossing from port | W, I, N | Same, port side. Class collapses in the observation; kept distinct here for the passing-side term |
| A-OT-W / -I / -N | Overtaking | W, I, N | `k = 0.4` |
| A-BO-W / -I / -N | Being overtaken | W, I, N | `k = 1.8`, 17(a)(i) hold |
| A-NU-W / -I / -N | Null | W, I, N | Similar course, no CPA |
| A-CLT-HO-I / -N | Head-on + conflict obstacle | I, N | Obstacle in the compliant corridor |
| A-CLT-CRS-I / -N | Crossing + conflict obstacle | I, N | |
| A-CLT-OT-I / -N | Overtaking + conflict obstacle | I, N | Obstacle on the port passing side |
| A-OCC-HO-I | Head-on, 2 s occlusion | I | Target lost then reacquired |
| A-OCC-CRS-I | Crossing, 4 s occlusion | I | Beyond tracker coast time |
| A-8E-HO-N | Head-on, target positionally non-compliant `nc` | N | Target on the wrong side; forces alteration or 8(e) |
| A-8E-CRS-N | Crossing, no room to alter | N | 8(e) is the only admissible response |
| A-BND-HO-I | Head-on on a bend | I | 40° bend at CPA |
| A-BND-CRS-I | Crossing on a bend | I | |
| A-OFF-CRP-I | Crossing from port, path offset to port wall | I | Boundary branch load-bearing |
| A-OFF-OT-I | Overtaking, path offset | I | |
| A-FAIL-CRS-X | Crossing, sub-threshold width, `nc` target | X | Expected failure |
| A-FAIL-HO-X | Head-on, sub-threshold width, `nc` target | X | Expected failure |

**34 cases.** Each generates a figure-ready record: trajectory overlay with ship domains, rudder and
propulsion traces, CPA-vs-time, encounter-class timeline, and the per-term reward decomposition from
the telemetry panel.

---

## 6. Study 1 — channel-width sweep

**Levels (8):** 10.0, 8.0, **7.0**, 6.0, 5.0, **4.5**, 4.0, 3.5 m — 20, 16, 14, 12, 10, 9, 8, 7 B.
The two added levels bracket the crossing transition at 7.60 m and the overtaking transition at
4.26 m, which the original six-level sweep stepped over.

**Base constellations:** 12, drawn from Tier A — one per class per behaviour for the four
non-null classes, held fixed across widths so the sweep is interpretable. No retraining: the policy
is trained across the full width range (§3.7 stage 5) and evaluated per width.

**Reported per width:**

- Compliant success rate (primary), and success rate ignoring compliance (secondary)
- Compliance rate per encounter class
- Minimum CPA distribution as a CDF
- Collision rate split static / boundary / target
- **Manoeuvre-mode share** — fraction of give-way events resolved by course alteration vs by speed
  reduction under 8(e). This is the headline curve
- Generator rejection rate per class (§3.5) on the same axis — the feasibility envelope
- Width at which each classical comparator becomes inadmissible

**Predicted result to pre-register:** the crossing manoeuvre-mode share crosses 50% between 8 m and
7 m; overtaking between 4.5 m and 4.0 m; head-on is resolved by channel-keeping at every width and
never crosses. Write this into the claim ledger before the first evaluation run. If the measured
transitions land elsewhere, that is a more interesting result than if they land where predicted, and
having predicted them is what makes it one.

---

## 7. Study 2 — perception degradation

### 7.1 Design

**Two training variants, both evaluated across the full sweep:**

| Variant | Training noise | Purpose |
|---|---|---|
| `P-noise` | Measured field noise (05) + domain randomisation | The method |
| `P-clean` | Noise-free perception | The contrast that makes N1 a claim rather than an assertion |

Without `P-clean` the study measures brittleness under distribution shift, which is interesting but
does not establish that injecting characterised noise helped. With it, the study establishes both.
Cost is 5 additional training runs and it is the single highest-value addition to the campaign.

**Axes, 5 levels each, as multiples of the measured nominal from 05:** `{0, 0.5, 1, 2, 4} × nominal`.

| Axis | Nominal source | Failure endpoint |
|---|---|---|
| Pose drift magnitude | rf2o characterisation, 05 §4 | Boundary raycast becomes unusable |
| Detection dropout rate | Returns-per-revolution analysis | Track loss |
| Occlusion duration | Scenario-controlled (§3.6) | Beyond tracker coast time |
| Velocity estimate noise | Scan-distortion analysis | Encounter misclassification |

Plus one joint corner at 2× on all four axes. **21 conditions.** Evaluated on the 12-constellation
Study 1 base set plus the narrow Tier B cells.

### 7.2 The reported quantity

The question is not how far performance drops but **where failures stop being conservative and start
being unsafe**. Define, per condition:

```
FMR  =  (conservative failures)  /  (unsafe failures)

conservative  =  timeouts, excessive slowing, aborted transits, over-large CPA
unsafe        =  wrong-side passing, bow crossing, domain intrusion, collision
```

Report `FMR` against each axis with the CI. A policy whose `FMR` stays above 1 under degradation is
deployable; one whose `FMR` falls below 1 misclassifies and turns the wrong way, and is not. This
single ratio is the deliverable of Study 2 and the clearest statement of what N1 buys.

---

## 8. Comparators and the compute gate

### 8.1 Protected core — runs regardless

| Configuration | Training runs |
|---|---|
| Full method (SAC, all terms, encounter feature) | 5 |
| Ablation 2×2, remaining three cells | 15 |
| `P-clean` noise-free training contrast (Study 2) | 5 |
| Domain-randomisation off (Study 3 / RQ4) | 5 |
| Paper 2 SAC, frozen, zero-shot | 0 |
| Classical: LOS-PID + DWA, COLREGs-VO, encounter-specific VO | 0 |
| **Total** | **30** |

### 8.2 Contingent tail — pre-committed priority order

Cut from the bottom, never from the middle, and never after seeing results.

| Rank | Configuration | Runs | Question it answers |
|---|---|---|---|
| 1 | TQC | 5 | Does distributional value estimation help under perception noise? |
| 2 | RecurrentPPO | 5 | Does recurrence help under occlusion? Gates the 01 recurrence decision |
| 3 | Leave-one-out COLREGs terms, **3 seeds** | 15 | Term-level attribution. Deviation from 5 seeds stated explicitly |
| 4 | PPO | 5 | On-policy reference, continuity with Paper 1 |
| 5 | Frame-stacked SAC | 5 | **Only if RecurrentPPO ranks top-two.** Isolates memory from algorithm |
| 6 | TD3 | 5 | Deterministic off-policy reference |
| 7 | NMPC with COLREGs constraints | 0 | Implementation cost, not compute |

### 8.3 The gate

Evaluation is not the bottleneck and must not be cut to buy training runs. One policy-seed's full
suite is ≈ (975 + 340 + 480) × 900 ≈ 1.6 M steps, roughly 50 minutes at 500 steps/s with no
gradients — call it 30 hours for twelve policies at five seeds, trivially parallel.

Training is the bottleneck, and the new environment is materially slower than Paper 2's: per step it
raycasts against a moving hull polygon, runs clustering and a Kalman filter, and raycasts the
boundary. **Measure it before committing.** Procedure:

1. Build the environment, then run 100k steps of SAC on stage-5 scenarios with a vectorised env
2. Record steps/s and extrapolate to the planned budget per run `TODO(04-4)`
3. `runs_affordable = (budget_hours − 30 × t_run) / t_run`
4. Include tail entries in rank order until exhausted

Report the measured throughput in the paper. It is also the honest answer to why the comparator list
is the length it is.

---

## 9. Freeze protocol

### 9.1 Scenario record

Every scenario serialises to canonical JSON with sorted keys: `suite_version`, `case_id`, `class`,
`width_profile`, `bend`, `path_offset`, `target_spawn_pose`, `target_velocity`, `target_behaviour`,
`obstacles`, `flags` (`conflict`, `occlusion`, `expected_failure`), `seed`, `generator_git_sha`,
`rejection_count`.

### 9.2 Seed namespaces — disjoint by construction

| Range | Use |
|---|---|
| `0 – 99,999` | Training distribution |
| `200,000 – 209,999` | Development suite — all algorithm and reward selection |
| `300,000 – 309,999` | Frozen evaluation suite |
| `400,000 – 409,999` | Study 1 width sweep |
| `500,000 – 509,999` | Study 2 degradation sweep |

Assert disjointness in a unit test. The development suite exists so that reward iteration and
algorithm selection never touch the frozen suite — using one suite for both introduces selection
bias that is invisible in the results and fatal if noticed.

### 9.3 Freeze checklist — all before the first headline training run

- [ ] Generator source committed, git SHA recorded
- [ ] Suite generated, serialised, SHA-256 per case and over the manifest
- [ ] `SUITE_MANIFEST.json` committed: version, hashes, seed ranges, generator SHA, constants snapshot
- [ ] `constants.py` snapshot with every `TODO(04-*)` resolved or explicitly deferred
- [ ] Empty result tables committed matching the draft skeleton
- [ ] Claim ledger committed, including the Study 1 predictions from §6
- [ ] Comparator priority order (§8.2) committed
- [ ] Regeneration test: regenerating from seed reproduces every hash

### 9.4 Release artefact

Generator source, seed ranges, the frozen suite as data, and the manifest, released as a citable
archive. With Imazu dropped this is not optional — it is the only structural defence against
"the authors built their own benchmark, then showed classical methods fail on it."

---

## 10. Metrics — suite-specific additions

Full list in `00` §4.2. New or refined here:

| Metric | Definition | Carries |
|---|---|---|
| **Compliant success** | Goal reached, no collision, no COLREGs violation. **Primary endpoint** | Everything |
| Manoeuvre-mode share | Give-way events resolved by alteration vs by 8(e) speed reduction | Study 1, N2 |
| Failure-mode ratio `FMR` | §7.2 | Study 2, N1 |
| Track acquisition range | Reported **per class** (§1.4) | N1 |
| Generator rejection rate | Per `(class, width)` | Study 1 feasibility envelope |
| Speed-reduction events | Count, timing, and whether class and geometry made them appropriate (02 §4.4) | Study 1, N2 |
| Timeout rate | Reported separately from failure (§4.1) | — |

### 10.1 Presentation

**One pre-registered primary endpoint: compliant success rate.** Conjunctive rather than weighted, so
there is no scalarisation to contest. Everything else is secondary and reported per axis.

Main comparison table: rows = methods, columns = per-axis metrics, cells = mean with bootstrap 95%
CI and the full seed spread. Plus a two-panel Pareto — safety (min-CPA 5th percentile) against
efficiency (path length ratio), and compliance against efficiency — with methods as points and seeds
as the scatter. A weighted scalar index will be contested by any reviewer who disagrees with the
weights, and there is no defensible way to choose them.

---

## 11. Open items after this document

| Tag | Item | Owner |
|---|---|---|
| `TODO(04-1)` | `w_wall` from measured pose drift | 05 |
| `TODO(04-2)` | `D_max` effective, including the black-wall side | 05 |
| `TODO(04-3)` | Curriculum steps per stage | 04 → after throughput measurement |
| `TODO(04-4)` | Training steps per run, total budget hours | 04 |
| — | Confirm the §1.1 threshold ordering against 02a | 02 |
| — | Confirm the two-constraint domain convention (§1.2) and restate 03 §5 arithmetic | 01 / 03 |
| — | Revise 03 §3 target-confinement rule for the crossing class (§1.3) | 03 |
| — | Target platform top speed — gates the being-overtaken class (§1.4) | 05 |
| — | Whether the sub-threshold width `X` = 3.5 m is also added to the Tier B narrow stratum | 04 |

### Changes this document requires elsewhere

1. **03 §3** — target confinement becomes class-conditional; crossing targets cross the fairway
   under Rule 9(d)
2. **03 §5** — width table gains 7.0 m and 4.5 m levels; head-on minimum recomputed to 3.80 m under
   the two-constraint convention; the 3.66 m figure is superseded
3. **03 §8** — episode horizon resolved: 900 steps
4. **02 §3.2** — precedence table gains numeric width thresholds and a Rule 9(d) row for crossing
5. **01 §6.3** — the recurrence decision is now gated by the Study 2 occlusion axis and the
   RecurrentPPO tail entry, both specified here
6. **00 §2** — "width thresholds" moves from *still open* to *derived, pending Study 1 confirmation*
