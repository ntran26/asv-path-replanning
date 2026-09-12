# 03a — Environment and Dynamic Target: Specification

**Revision 2.0** — first full specification. Expands `03_ENVIRONMENT_AND_TARGETS.md` into an
implementable spec, closes its five open items, and absorbs the three changes 04a requires of it.
**Handover target:** Claude Code (§4–§8), Claude chat (§1 needs your sign-off first)
**Depends on:** 01 §2–§5 (LiDAR, tracker, domain), 02 §4.4 (Rule 8(e) and propulsion),
04a §1, §3.2, §3.4 (corridor parameters, spawn rules), 05 (identified model — four TODOs)
**Supersedes in 03:** §3 target confinement, §5 width arithmetic, §8 horizon question

---

## 1. Three findings that change the environment

### 1.1 The simulator is running the vessel at roughly three times its real speed

02b flagged a Froude number of 0.45 in the simulator against an expected 0.14. Substituting
`Lpp = 1.57 m`, `√(g·Lpp) = 3.925 m/s`:

| | Froude | Speed | Time for the 20 m path |
|---|---|---|---|
| Simulator as configured | 0.451 | **1.77 m/s** | 11 s |
| Field-measured Bluefin | 0.140 | **0.55 m/s** | 36 s |

This is not a tuning discrepancy. It is a 3.2× error in the single quantity that sets every time
constant in the paper — TCPA ranges, spawn geometry, episode horizon, classification latency budget,
the Rule 8(a) "early action" metric, and the entire feasibility argument in 04a §1.4.

**Decision: `U_nom = 0.55 m/s`, Fr = 0.14, taken from the field measurement and treated as
authoritative until 05 delivers the identified model.** Everything in 04a was derived at 0.55 m/s and
therefore stands; the simulator is what has to move.

**This also answers the Froude-scaling question in 03 §5 outright,** and it answers it in the paper's
favour. At geometric scale λ = 50:

| Quantity | Model | Full scale (λ = 50) |
|---|---|---|
| `Lpp` | 1.57 m | 78.5 m |
| Speed at Fr = 0.14 | 0.55 m/s | 3.89 m/s ≈ 7.6 kn |
| Corridor, wide end | 10.0 m | 500 m |
| Corridor, narrow end | 3.5 m | 175 m |
| Spawn TCPA | 15 s | 106 s |
| Episode horizon | 90 s | 12.6 min |

Every one of those is a plausible restricted-water transit. Run the same table at Fr = 0.45 and the
full-scale speed is 24 kn, which is not a channel speed for any vessel. **The Froude argument is
therefore not merely a defence against a reviewer question — it is independent evidence that 0.55 m/s
is the correct operating point.** Put the λ = 50 column in the paper; it converts "a 1.73 m model
boat" into "a 78.5 m vessel in a 175–500 m fairway", which is the scale at which Rule 9 is actually
argued about.

Widths quoted in ship breadths are scale-invariant, so the sweep in breadths (20 B → 7 B) transfers
without qualification. Quote breadths first and metres second throughout.

**Consequence to check in the repository:** if Paper 2's trained policy was produced at 1.77 m/s and
transferred to a 0.55 m/s vessel, the observed 2× field cross-track error has a candidate explanation
that is not the control policy. Worth one paragraph — it is a free result and it strengthens the
sim-to-real framing rather than weakening Paper 2.

### 1.2 The boundary gate is a no-op in simulation and load-bearing in the field

01 §3.4 requires the identical gating operation in both pipelines. As currently specified the
simulated LiDAR returns only static panels and the target, so there is nothing for the gate to
remove: it passes everything through. In the field it discards facility-wall returns, operators
standing at scan height, and clutter beyond the pool edge, and a misconfigured margin either gates
out real obstacles or lets phantoms through.

That is a sim-to-real gap in exactly the component 01 §3 was written to remove one from.

**Decision: simulate the out-of-corridor world.** Place virtual facility walls at the basin envelope
plus 1.5 m and return them in the raw scan, then apply the same geometric gate. When the corridor is
narrower than the basin, the water between the corridor edge and the basin edge is open and returns
nothing — which is the real situation, since the corridor boundary is a map polygon and is invisible
to the sensor. Optionally inject transient off-corridor clutter (the operator model, §5.4).

The gate then does real work in training, its margin becomes a tunable with measurable failure modes
in both directions, and the N1 claim covers the whole perception stack rather than the part after the
gate.

### 1.3 Pose noise sets a floor on classification latency, and it squeezes the overtaking speed ratio

The static/dynamic decision (03 §4a, 01 §4 stage 6) is limited by the apparent velocity that ego-pose
error imparts to genuinely static objects:

```
σ_v  ≈  √2 · σ_p / T_w
```

with `σ_p` the per-scan pose error and `T_w` the effective velocity-estimation window. At `σ_p` =
0.02 m and a single 0.1 s interval, static panels appear to move at **0.28 m/s** — half a target's
speed. Velocity must therefore be estimated over a window of order seconds, and that window *is* the
classification latency. It is set by localisation quality, not by filter tuning, and it is a reported
metric and a Study 2 axis.

Requiring the threshold `v_hi = 5σ_v` to sit below 30% of the slowest target speed gives:

```
T_w  ≥  5 · √2 · σ_p / (0.3 · k_min · U_nom)
```

At `σ_p` = 0.02 m and `k_min` = 0.25 (the slow end of 04a's overtaking range) this demands
`T_w ≥ 3.4 s` — longer than several of the encounters are usable for.

**Decision: raise the overtaking speed-ratio floor from 0.25 to 0.40.** `T_w` then falls to ≈2.1 s,
which fits inside every class's acquisition-to-CPA budget.

This leaves the overtaking window squeezed from both ends by physics rather than preference:

```
k ≥ 0.40   from pose noise and the static/dynamic classifier   (this document)
k ≤ 0.55   from basin length and the completed-pass constraint (04a §1.4)
```

State the window and its two derivations in the methods. A reviewer asking why overtaking was tested
only over a narrow speed-ratio band gets a two-line answer from first principles.

---

## 2. Decisions closing 03 §8

| Open item | Decision |
|---|---|
| Reconcile `rl_env_dynamic.py` | Checklist in §9, executed by Claude Code — not assumed here |
| Propulsion authority | **`n ∈ [0, 1]` normalised thrust, no reverse by default**, with a config flag and a stated limitation. Rationale and stopping analysis in §4.3 |
| Target hull polygon and ship domain | §5.1 (hull) and §3.3 (domain), both provisional on 05 |
| Static/dynamic threshold and hysteresis | §6.3, derived from `σ_p` rather than chosen |
| Episode horizon | **900 steps / 90 s**, resolved in 04a §4.1 and re-verified at 0.55 m/s in §4.5 |

---

## 3. Corridor and boundary

### 3.1 Representation

A corridor is a centreline polyline `C(s)` with a width profile `W(s)`, both sampled from 04a §3.2.
The navigable polygon is the offset of `C` by `±W(s)/2`. Three distinct geometries, and they must not
be conflated:

| Polygon | Role | Visible to LiDAR? |
|---|---|---|
| **Corridor** | Hard constraint for the own ship; termination on breach | **No** — map-derived, invisible (01 §3.1) |
| **Basin envelope** | 10 × 25 m; limit of the physical water | No |
| **Facility walls** | Basin envelope + 1.5 m | **Yes** — returned, then gated (§1.2) |

### 3.2 Requirements the generator must satisfy

Restated from 04a §3.2 because they are enforced here: width varies along `s` (`W_max/W_min` up to
1.8), at least 40% of episodes carry a bend of ≥20°, and the reference path carries a lateral offset
with a positive mean (Rule 9(a) station). Without all three, port and starboard boundary rays are
affine in `e_y` and the 7-dimensional boundary branch is decorative.

**Assertion:** over 1000 sampled episodes, `|corr(e_y, b_i)| < 0.9` for every boundary ray `b_i`.

### 3.3 Ship domain

| Direction | Multiple | Metres |
|---|---|---|
| Ahead | 2.0 · Lpp | 3.14 |
| Astern | 1.0 · Lpp | 1.57 |
| Abeam, each side | `max(0.75·Lpp, a_floor)` | 1.18 or 1.25 — `TODO(03-1)` |

`TODO(03-1)`: 04a's width thresholds were derived with `a_abeam` = 1.25 m, taken from the sensor-floor
constraint flagged in 02b. Confirm that value and its origin against 02b before it propagates further.
The sensitivity is mild — `W_crossing = 4(a_abeam + w_wall)`, so the 0.07 m difference moves the
crossing threshold by 0.28 m and changes no ordering — but the number appears in the precedence table
and should be traceable.

**Two-constraint convention (04a §1.2):** the ship domain governs vessel-to-vessel separation only.
Clearance to the boundary is governed separately by `w_wall = B/2 + 3σ_p + control margin`,
`TODO(03-2)` pending 05. Domain intrusion is a **metric, not a termination** — the run continues and
the intrusion depth is logged.

---

## 4. Own ship

### 4.1 Dynamics

3-DOF Fossen model, unchanged in structure from Paper 2, with every coefficient replaced by the
identified set from 05 and randomised over its confidence interval (§8). Control period 0.1 s.

State: `η = [x, y, ψ]`, `ν = [u, v, r]`. The observation exposes `ν` and path-relative quantities
only — no global pose (01 §1). That framing is part of why Paper 2 transferred; do not regress it.

### 4.2 Action space

| Action | Range | Rate limit |
|---|---|---|
| Rudder `δ` | `[−δ_max, +δ_max]` | Matched to the real actuator, Paper 2 Table 2 |
| Thrust `n` | `[0, 1]` normalised | `TODO(03-3)` from 05 actuator identification |

Continuous, both. Actuator lag modelled as a first-order lag with time constant from 05 and
randomised (§8). **Time-sync caution from the field side:** a constant LiDAR/IMU offset appears in the
identification fit as actuator lag, so the lag constant handed over from 05 must come from a
synchronised fit or it will encode a clock error as a physical property.

### 4.3 Propulsion authority — decision and its justification

02 §4.4 widened propulsion authority so that Rule 8(e) speed reduction is available. The open
question was whether the vessel can reverse.

**Decision: model `n ∈ [0, 1]` — thrust reduction only, no reverse — as the default configuration,
with `allow_reverse` as a config flag.** Reasoning:

The demanding case is crossing give-way by passing astern. The own ship must shed enough along-track
distance for the target to clear:

```
Δt_required  ≈  (a_abeam,OS + a_abeam,TS + B) / U_TS  =  (1.25 + 1.25 + 0.50)/0.55  ≈  5.5 s
Δs_required  =  U_nom · Δt_required                    ≈  3.0 m
```

Losing 3.0 m of along-track position requires holding roughly 0.2 m/s for 10–15 s, i.e. decelerating
from 0.55 to 0.20 m/s within a few seconds. At 64.55 kg plus surge added mass, that is of order 5 N
of net decelerating force — comparable to the hull's own quadratic drag at 0.55 m/s. **Coasting alone
plausibly achieves it; reverse thrust is probably not required.**

`TODO(03-4)`: confirm against the identified surge drag from 05 by computing head reach from
`U_nom` to `0.2·U_nom`. **Acceptance criterion: head reach ≤ 1.5·Lpp (2.36 m).** If it exceeds that,
`allow_reverse` must be set and the platform's actual reverse capability verified — and if the
platform cannot reverse, "take all way off" is unavailable and the paper states the limitation rather
than claiming the manoeuvre.

Either way this is a one-sentence declaration in the methods. It is awkward only if a reviewer
notices that Rule 8(e) was claimed on a vessel that cannot execute it.

### 4.4 Termination

| Condition | Terminal | Recorded as |
|---|---|---|
| Hull polygon intersects a static panel | Yes | Collision — static |
| Hull polygon leaves the corridor polygon | Yes | Collision — boundary |
| Hull polygon intersects the target hull | Yes | Collision — target |
| Path progress ≥ 0.98 | Yes | Success |
| Step count = 900 | Yes (truncation) | **Timeout, reported separately** |
| Ship-domain intrusion | **No** | Metric: intrusion rate and depth |

Timeout is truncation, not termination — bootstrap the value function at the cut (00 §4.2). Under a
reward that designates slowing as compliant behaviour, treating a timeout as failure would penalise
the behaviour being taught.

### 4.5 Horizon re-verification at the corrected speed

900 steps = 90 s. Worst realistic case: 20 m path at `U_nom` = 36 s, plus a Rule 8(e) hold at 0.2 m/s
for 15 s costing ≈10 s of transit, plus a bend and an overtaking manoeuvre at reduced relative speed,
plus the being-overtaken hold. Total ≈65 s. 90 s leaves ~28% margin.

**Acceptance check:** timeout rate in curriculum stage 5 must sit below 5%. If it does not, the cause
is a degenerate slow-creep policy (02 §4.4) and the fix is the progress-penalty carve-out gating, not
a longer horizon.

---

## 5. Dynamic target

### 5.1 Hull geometry

Oriented polygon, not a circle — required for aspect-angle computation, ship-domain metrics, and
realistic LiDAR returns (03 §3). Body frame at midships, `x` forward, `y` starboard, metres:

```
( 0.865,  0.000)  ( 0.780,  0.130)  ( 0.600,  0.215)  ( 0.250,  0.250)
(−0.550,  0.250)  (−0.865,  0.230)  (−0.865, −0.230)  (−0.550, −0.250)
( 0.250, −0.250)  ( 0.600, −0.215)  ( 0.780, −0.130)
```

LOA 1.73 m, maximum breadth 0.50 m. `TODO(03-5)`: replace with traced offsets if hull lines exist;
centimetre-level accuracy matters only for the LiDAR return pattern, which feeds cluster-centroid
bias in the tracker.

Own ship and target share this polygon — they are the same class of vessel, which is the premise on
which S3 rejected the Rule 18 route.

### 5.2 Confinement — now class-conditional (04a §1.3)

| Class | Target confined to the corridor? | Governing rule |
|---|---|---|
| Head-on | Yes | 9(a), 9(b) |
| Overtaking | Yes | 9(a), 9(e) |
| Being overtaken | Yes | 9(a) |
| Null | Yes | — |
| **Crossing** | **No — crosses the fairway** | **9(d)** |

This supersedes 03 §3's blanket "the target must respect the channel". A crossing target confined to a
4 m corridor would have to pass through the far wall; a crossing target under Rule 9(d) is not a
channel user at all. The crossing target is spawned in open basin water outside the corridor, crosses
it, and continues out the other side. Physically reproducible, because the corridor is a virtual map
polygon inside a 10 m basin.

### 5.3 Behaviour models

| ID | Model | Used in |
|---|---|---|
| `T-CV` | Constant velocity, constant heading | Training (D1) and evaluation |
| `T-RE` | Compliant reactive — encounter-specific VO, Thyri & Breivik (2022) | Evaluation only |
| `T-NC1` | Stands on when it is the give-way vessel | Evaluation only |
| `T-NC2` | Alters to **port** in a head-on | Evaluation only |
| `T-NC3` | Positionally non-compliant — holds the wrong side of the fairway, violating 9(a) | Evaluation only; required by Tier A case `A-8E-HO-N` |

`T-NC3` is new relative to 03 §2.3, which specified only the two behavioural violations. It is needed
because the head-on precedence argument (02 §3.2) is that 9(a) channel-keeping satisfies Rule 14
*without* an alteration — so the only way to test whether the policy can still execute Rule 14 when
required is a target that is not where 9(a) says it should be. Without `T-NC3` the head-on class is
never exercised as an avoidance problem at all.

`T-RE` and the C3 comparator are one implementation (03 §2.2). Keep them one module; if they drift
apart, the reactive stratum and the VO baseline stop being comparable.

### 5.4 Operator model (optional, off by default)

A pedestrian-speed point obstacle outside the corridor at scan height, appearing for 5–20 s. Mirrors
the field hazard that made gating mandatory (01 §3.4). Off by default; enabled in a supplementary
robustness cell. Cheap to add now, impossible to add convincingly after a field trial produces a
phantom track.

---

## 6. Sensing simulation

### 6.1 Raw scan

| Parameter | Value |
|---|---|
| Beams per revolution | 720 at 0.5°, **`TODO(03-6)`** — verify against retained logs; the C1 at 10 Hz may deliver ≈500 (≈0.7°) |
| Rate | 10 Hz, one scan per control step |
| Range | `[r_min, D_max]`, `TODO(03-7)` from 05, black-wall side possibly capped |
| Aft self-occlusion | Masked bearing sector, width from field logs `TODO(03-6)` |
| Motion distortion | Beams cast from interpolated poses across the revolution |
| Range noise | Zero-mean Gaussian, σ from logs; randomised (§8) |
| Dropout | Per-beam Bernoulli, rate from logs; swept in Study 2 |

Ray-cast targets, in order: static panels, target hull polygon, facility walls. **Not** the corridor
boundary (§3.1).

Simulating finer than the sensor delivers is a self-inflicted sim-to-real gap, so resolve
`TODO(03-6)` before the suite is frozen, not after.

### 6.2 Occlusion

A target behind a panel disappears from the scan. Occlusion duration is logged per episode, is a
controlled variable in flagged scenarios (04a §3.6), is a Study 2 axis, and is the evidence that
decides the recurrence question in 01 §6.3. Do not shortcut it by injecting the target's true state
when occluded — that collapses N1.

### 6.3 Static/dynamic classification

Derived, not tuned (§1.3):

```
σ_v   =  √2 · σ_p / T_w
v_hi  =  5 · σ_v            promote static → dynamic
v_lo  =  3 · σ_v            demote dynamic → static
T_w   =  2.0 s              velocity-estimation window
```

Hysteresis: promote after 8 consecutive updates above `v_hi` (0.8 s); demote after 20 consecutive
updates below `v_lo` (2.0 s). Asymmetric and slow in both directions — a false promotion creates a
phantom give-way obligation with COLREGs consequences, and a premature demotion drops a real target
mid-encounter. Classification stability is a reported metric, so instability shows up in the results
rather than hiding in the policy.

`σ_p` comes from 05. Retighten the thresholds when scan-to-map registration replaces rf2o; the
threshold is a property of localisation quality, not of the obstacles (03 §4a).

### 6.4 Suspension lines

Minimum cluster size must reject taut-rope returns, which subtend one or two beams. Set
`min_cluster_points` above 2 at the ranges where lines cross the scan plane, and verify against a
retained log containing a known line crossing. Rejecting genuine small obstacles is the failure mode
in the other direction; both are worth measuring since the cluster stage is part of the N1 claim.

---

## 7. Observation assembly

Five branches as specified in 01 §6, assembled here. The environment's contract:

| Branch | Dim | Source |
|---|---|---|
| `lidar` | 27 | Pooled `c_t`, obstacles only, ±135°, post-gate |
| `boundary` | 7 | Virtual raycast at `{−90, −60, −30, 0, +30, +60, +90}°` from **estimated** pose |
| `ego` | 3 | `u, v, r` — with the field's differentiation error modelled (§8) |
| `path` | 3 | `e_y, χ̃, χ̃_LA` |
| `target` | 16 | 15 features + presence bit, from the tracker, never from ground truth |
| | **56** | |

**The boundary branch must be computed from the noisy estimated pose, not the true pose.** A
noiseless boundary scan in training creates a sim-to-real gap in the component introduced to remove
one (01 §3.3). Assert it in a test: with pose noise enabled, the boundary branch must differ from its
ground-truth counterpart.

Same for `ego`: in the field `u, v, r` are differentiated from noisy pose. The IMU now rescues yaw
rate and, via the accelerometer, surge — but the error does not vanish, and 05 will quantify what
remains.

---

## 8. Domain randomisation hooks

First-class config from the start (03 §7). Everything below is a named parameter with a nominal value
and a range, not a hardcoded constant.

```yaml
vessel:        # from 05 identification, randomised over ±CI
  mass, Iz, added_mass, linear_damping, quadratic_damping
actuator:
  rudder_max, rudder_rate, thrust_range, lag_tau, allow_reverse
sensing:
  beams_per_rev, angular_res, range_max, range_sigma, dropout_rate
  aft_occlusion_start, aft_occlusion_end, motion_distortion
localisation:
  pose_sigma, pose_drift_rate, heading_sigma
ego_estimate:
  u_sigma, v_sigma, r_sigma          # differentiation error
tracker:
  T_w, v_hi, v_lo, promote_n, demote_n, min_cluster_points, coast_time
obstacles:
  count_range, position_sigma        # suspended-panel placement uncertainty
gate:
  margin
```

Two of these are the sim-to-real gaps identified from the field side and exist for that reason:
`ego_estimate` (velocities differentiated from noisy pose) and `obstacles.position_sigma` (suspended
panels surveyed imprecisely between sessions). Neither existed in Paper 2's environment.

Study 2 sweeps `localisation.pose_sigma`, `sensing.dropout_rate`, occlusion duration, and tracker
velocity noise. Study 3 ablates the whole randomisation block. Both are impossible unless these are
config rather than constants — which is the entire reason for specifying them before implementation.

---

## 9. Reconciliation of `dynamic_obstacles/rl_env_dynamic.py`

The file (~22 KB) already contains early Paper 3 work and must be reconciled rather than replaced
(03 §1). This is Claude Code's task; the checklist is the deliverable expected back:

1. Does it inject the target as ground truth, or ray-cast it? Ground-truth injection is
   architecture-breaking, not a parameter — flag it rather than patching it
2. What nominal speed does it run at? Report the Froude number (§1.1)
3. Is the target a circle or an oriented polygon?
4. Does the corridor have variable width, bends, and path offset, or is it Paper 2's constant channel?
5. Does the boundary come from a map raycast or from the LiDAR?
6. What horizon, and is timeout truncation or termination?
7. Which of the §8 parameters exist as config and which are hardcoded?
8. Are any Paper 2 reward terms still wired in? They are all superseded by 02a
9. Line-by-line: reusable / adaptable / superseded

Return the checklist and a porting manifest before any code is written against this spec.

---

## 10. Acceptance tests

Committed with the environment, run before the first training run.

| # | Test |
|---|---|
| T1 | Froude number at `U_nom` is 0.14 ± 0.01 |
| T2 | Boundary branch differs from its ground-truth counterpart when pose noise is enabled |
| T3 | `\|corr(e_y, b_i)\| < 0.9` for all 7 boundary rays over 1000 episodes |
| T4 | Target is never visible in the observation when fully occluded by a panel |
| T5 | Facility-wall returns appear in the raw scan and are absent after the gate |
| T6 | Corridor boundary never appears in the raw scan |
| T7 | Crossing targets leave the corridor; all other classes never do |
| T8 | Static panels are classified dynamic in fewer than 1 in 10⁴ frames at nominal `σ_p` |
| T9 | Head reach from `U_nom` to `0.2·U_nom` ≤ 1.5·Lpp, or `allow_reverse` is set |
| T10 | Timeout rate below 5% on curriculum stage 5 |
| T11 | Encounter class from the environment matches the reward gate's class in every frame (one module, two consumers — 01 §5.3) |
| T12 | Domain intrusion does not terminate the episode |

T11 is the one most likely to fail silently and the most damaging if it does: a divergence at a class
boundary penalises the agent for a role it was never shown.

---

## 11. Open items and cross-document changes

| Tag | Item | Owner |
|---|---|---|
| `TODO(03-1)` | Confirm `a_abeam` floor and its origin against 02b | 02 |
| `TODO(03-2)` | `w_wall` from measured pose error (= `TODO(04-1)`) | 05 |
| `TODO(03-3)` | Actuator rate limits and lag from synchronised identification | 05 |
| `TODO(03-4)` | Head reach; decides `allow_reverse` | 05 |
| `TODO(03-5)` | Hull offsets, if available | — |
| `TODO(03-6)` | Beams per revolution, aft occlusion sector, motion distortion, from retained logs | 05a |
| `TODO(03-7)` | `D_max` effective, black-wall side (= `TODO(04-2)`) | 05 |

### Changes required elsewhere

1. **04a §3.4** — overtaking `k` range narrows from `[0.25, 0.55]` to `[0.40, 0.55]`, and `R_0` upper
   bound becomes `min(12, 16(1−k))` ≈ 7.2–9.6 m (§1.3)
2. **04a §5** — Tier A `A-OT-*` cases use `k = 0.45` rather than 0.40, mid-window
3. **04a §7.1** — add `sensing.dropout_rate` and tracker `T_w` to the Study 2 axis definitions; `T_w`
   is now a derived quantity with a floor, so its nominal is not free
4. **02a** — confirm `a_abeam`; add the `T-NC3` positionally non-compliant target to whatever the
   reward spec says about head-on, since it is the only case where Rule 14 alteration is exercised
5. **01 §5.2** — record the two-constraint domain convention explicitly
6. **00 §2** — move "reverse thrust capability" from open to *decided by acceptance test T9*; add the
   Froude correction to the decision log as a new entry
7. **PROJECT_BRIEF §2** — simulation speed corrected to 0.55 m/s, Fr = 0.14; add the λ = 50 scaling
   row
