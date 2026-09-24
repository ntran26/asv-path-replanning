# OBSERVATION SPEC — Paper 3

**Version:** `a25-v3-context` (observation revision 3; `OBSERVATION_SCHEMA_VERSION`
in `constants.py`). Supersedes `obs-v2` (56 dims, five branches) and `obs-v1`
(91 dims, three slots, six encounter classes). Revision 3 keeps every v2
feature in its original order and appends a sixth branch, `context` (F72).

**Status:** frozen in baseline-v1 (`configs/baseline_v1.json`, git tag
`baseline-v1`). Every checkpoint and every evaluation case depends on this
ordering. Changing any index is a version bump, not an edit; a checkpoint from
an earlier schema cannot be loaded or resumed against this one.

**Updated 2026-09-22** to match the code: this file described v2 until then.
Also changed since v2 without moving an index: the speed normaliser (§3), the
cross-track normaliser (§4), and nominal sensor noise, now on (§2, §3).

Total **70** dims across **6** branches, as a `gymnasium.spaces.Dict`. The
machine-readable copy of this layout is `src/observation.py`; the two must
agree, and `tests/test_observation.py` enforces the dimensions, the branch count
and the feature names.

| Branch | Contents | Dims | Range |
|---|---|---|---|
| `lidar` | `c_t` sector closeness, **static obstacles only** | 27 | [0, 1] |
| `boundary` | virtual boundary raycast, normalised | 7 | [0, 1] |
| `ego` | u, v, r | 3 | [-1, 1] |
| `path` | e_y, χ̃, χ̃_LA | 3 | [-1, 1] |
| `target` | 15 features + 1 presence bit | 16 | [-1, 1] |
| `context` | 12 encounter-state features + 2 previous-action values | 14 | [-1, 1] |
| | **Total** | **70** | |

All arrays are `float32`.

**Frame conventions.** World: +y north, +x east. Headings and bearings are
compass style — 0° is +y, increasing clockwise. Relative bearing `α` is measured
from the own ship's heading: 0° dead ahead, 90° abeam to starboard.

**Cross-track error is positive to STARBOARD** — the textbook LOS convention,
flipped from Paper 2's positive-to-port in 02b C4/T2. Paper 2's own
`verify_los_apf.py` flagged its convention as non-standard; this paper's
contribution is COLREGs geometry, and 02a §6.4's passing-side term has two
opposite branches keyed on the sign of the lateral offset, so carrying a
non-standard sign through them invites exactly the class of error the paper is
about. **Anything comparing numbers across the two papers must account for it.**

**Only perceived state enters the policy.** Target features come from the
tracker, the boundary from the map at the *estimated* pose, ego motion from
noisy sensors. Ground-truth target fields never enter any branch
(`test_context_inputs_do_not_depend_on_target_truth`); they are attached to the
encounter context for diagnostics only.

### Revisions

| | v1 | v2 | **v3 (current)** |
|---|---|---|---|
| Target slots | 3 + a 3-bit mask vector | 1 + a presence bit | 1 + a presence bit |
| Encounter classes | 6 (crossing split give-way / stand-on) | 5 (crossing collapsed) | 5, plus the latched **compliant turn sense** in `context` |
| Branches | 5 | 5 | **6** (`context` added) |
| Total dims | 91 | 56 | **70** |
| Cross-track normaliser | — | map size (25 m) | **local channel half-width on the vessel's side** |
| Architecture | shared encoder + DeepSets/attention flag | plain concatenation | plain concatenation; context joins the slot encoder |

v2 was driven by decisions S1, S3, S4 and S6: two-vessel encounters are the unit
of analysis, because Rules 13–16 are formulated pairwise; confined geometry
precludes simultaneous close-quarters conflicts, so encounters are sequential;
and every reported behaviour becomes physically reproducible in the basin.

v3 (A25, F72, ported from CODEX) addresses two measured problems: the reward
judged an encounter state — engaged or not, the compliant turn direction,
heading and speed change since engagement — that the policy could not see in a
single frame; and the cross-track error, scaled by 25 m, used about a tenth of
its range in a 5–10 m channel.

`N_MAX_TARGETS` remains a config parameter and the slot machinery is still
indexed, so multi-vessel extension costs a retrain rather than a redesign.
`tests/test_observation.py::test_slot_machinery_still_scales_past_one` exercises
that path so it cannot rot.

---

## 1. `lidar` — 27 dims

Pooled sector closeness, `1 - range / 16.0`, clipped to [0, 1]. **1 = touching,
0 = clear to max range.** The LiDAR is 720 beams at 0.5°, 16 m range, 1.0 m
minimum range.

Carries **static obstacles only**. Returns classified as moving by the
free-space consistency test (F37) go to the tracker and reach the policy through
`target`; the channel boundary is gated out and reaches the policy through
`boundary`.

Sectors run **port to starboard**, index 0 outboard on the port quarter.
Non-uniform allocation, ±135° swath, 540 of the 720 raw beams. The aft 90°
(135°–225°) is reserved for the tracker.

| Idx | Bearing span | Centre | Φ | Beams |
|---|---|---|---|---|
| 0 | [−135.00, −112.50) | −123.75 | 22.50 | 45 |
| 1 | [−112.50, −90.00) | −101.25 | 22.50 | 45 |
| 2 | [−90.00, −78.75) | −84.375 | 11.25 | 23 |
| 3 | [−78.75, −67.50) | −73.125 | 11.25 | 22 |
| 4 | [−67.50, −56.25) | −61.875 | 11.25 | 23 |
| 5 | [−56.25, −45.00) | −50.625 | 11.25 | 22 |
| 6 | [−45, −39) | −42 | 6.00 | 12 |
| 7 | [−39, −33) | −36 | 6.00 | 12 |
| 8 | [−33, −27) | −30 | 6.00 | 12 |
| 9 | [−27, −21) | −24 | 6.00 | 12 |
| 10 | [−21, −15) | −18 | 6.00 | 12 |
| 11 | [−15, −9) | −12 | 6.00 | 12 |
| 12 | [−9, −3) | −6 | 6.00 | 12 |
| **13** | **[−3, +3)** | **0 (dead ahead)** | 6.00 | 12 |
| 14 | [+3, +9) | +6 | 6.00 | 12 |
| 15 | [+9, +15) | +12 | 6.00 | 12 |
| 16 | [+15, +21) | +18 | 6.00 | 12 |
| 17 | [+21, +27) | +24 | 6.00 | 12 |
| 18 | [+27, +33) | +30 | 6.00 | 12 |
| 19 | [+33, +39) | +36 | 6.00 | 12 |
| 20 | [+39, +45) | +42 | 6.00 | 12 |
| 21 | [+45.00, +56.25) | +50.625 | 11.25 | 23 |
| 22 | [+56.25, +67.50) | +61.875 | 11.25 | 22 |
| 23 | [+67.50, +78.75) | +73.125 | 11.25 | 23 |
| 24 | [+78.75, +90.00) | +84.375 | 11.25 | 22 |
| 25 | [+90.00, +112.50) | +101.25 | 22.50 | 45 |
| 26 | [+112.50, +135.00) | +123.75 | 22.50 | 45 |

Sectors are half-open `[lo, hi)`, so no beam is counted twice.
Bow 15 + abeam 8 + quarter 4 = 27. Beam total 540 = 270° / 0.5°.

The 11.25° sectors hold **22.5 beams** at 0.5°, so they alternate 23/22/23/22
per side. This follows from the spec's own numbers, and the tests assert the
alternating pattern rather than a constant count.

## 2. `boundary` — 7 dims

Virtual range scan against the known channel or basin polygon, from the
**estimated** pose, normalised by the identical `closeness_from_ranges` used
for `c_t` (16 m).

| Idx | Body-frame bearing |
|---|---|
| 0 | −90° (abeam port) |
| 1 | −60° |
| 2 | −30° |
| 3 | 0° (ahead) |
| 4 | +30° |
| 5 | +60° |
| 6 | +90° (abeam starboard) |

This is an **architectural argument, not a workaround** (01 §3.1). In a real
narrow channel the navigable limit is usually not a physical structure either —
it is a charted depth contour, a buoyed line or a regulatory limit, none of
which a LiDAR can see. The basin reproduces that exactly: the sensor sits above
the pool edge and registers the facility walls 1–2 m beyond it, so the boundary
the vessel must respect is invisible to the sensor while what the sensor sees is
not the boundary.

Pose noise is injected before the raycast, so the branch inherits localisation
error as it will in the field: **0.03 m and 0.2°**, nominal (F46). Measured
magnitudes from 05 remain `TODO(05)` (B4).

## 3. `ego` — 3 dims

| Idx | Symbol | Quantity | Normaliser |
|---|---|---|---|
| 0 | u | surge velocity | `SPEED_SCALE` = `2 × U_REF` = **1.116 m/s** |
| 1 | v | sway velocity | `SPEED_SCALE` |
| 2 | r | yaw rate, deg/s | 180 |

`U_REF` is 0.558 m/s, the identified plant at `CRUISE_RPM = 6` (F24). The
throttle range is 0–12 RPM, whose top speed is about `2 × U_REF`, so the
normaliser covers the operating range without clipping or dead range.

> Normaliser history: `2 × U_CRUISE` = 1.10 m/s in an early revision, then
> `2 × U_REF` = 2.28 m/s while `U_REF` was the 1.14 m/s log median at 12 RPM
> (02b T1). The factor has been `2 × U_REF` throughout; `U_REF` changed with F24.

**An IMU is confirmed** (05 §4.7). `r` is measured by the gyro rather than
differentiated, so its residual is the sensor noise floor; `u` and `v` are
largely rescued by the accelerometer but remain fused rather than measured.
Nominal noise is on: **0.05 m/s on speed, 1.0 °/s on yaw rate** (F46), measured
values `TODO(05)`. The branch carries field error Paper 2's simulator did not
model — a sim-to-real gap in the *observation*, not just the dynamics (05 §6).

## 4. `path` — 3 dims

| Idx | Symbol | Quantity | Normaliser |
|---|---|---|---|
| 0 | e_y | cross-track error, signed, **+ = starboard** | **local channel half-width on the side the vessel is on** |
| 1 | χ̃ | course error, deg | 180 |
| 2 | χ̃_LA | look-ahead course error, deg | 180 |

Computed from the **perceived** pose. e_y is divided by the distance from the
path to the boundary on the vessel's side at its along-path position (in basin
mode the two sides differ, because paths are slanted), so ±1 means "at the
navigable edge" whatever the width. Until v3 it was divided by 25 m, and in a
5–10 m channel it used about a tenth of its range (F72).

Path-relative, not global. There is deliberately **no (x, y, ψ)** anywhere in
the observation: the path-relative framing is part of why Paper 2's transfer
worked, and regressing to global coordinates would give it up.

## 5. `target` — 16 dims

One slot. Indices 0–15 are the slot; index 15 is its presence bit.

Use `observation.split_target()` rather than re-deriving offsets. At
`N_MAX_TARGETS > 1` the branch becomes `16 · N`, slot *s* occupying
`16s … 16s+15`, with its presence bit at `16s + 15`.

| Idx | Feature | Encoding | Range |
|---|---|---|---|
| 0 | Distance to ship domain | `/ D_SCALE` (16 m), clipped | [0, 1] |
| 1 | Relative bearing α | sin | [−1, 1] |
| 2 | Relative bearing α | cos | [−1, 1] |
| 3 | Heading intersection CT | sin | [−1, 1] |
| 4 | Heading intersection CT | cos | [−1, 1] |
| 5 | Target speed | `/ SPEED_SCALE` (1.116 m/s), clipped | [0, 1] |
| 6 | Relative speed | `/ SPEED_SCALE`, clipped | [0, 1] |
| 7 | DCPA | `/ DOMAIN_RADIUS_DCPA` (1.25 m), clipped at `DCPA_CLIP_DOMAINS` (12.8), rescaled | [0, 1] |
| 8 | TCPA | clipped to ±`TCPA_CLIP` (40 s), `/ TCPA_CLIP` | [−1, 1] |
| 9 | CRI | already in [0, 1] | [0, 1] |
| 10 | class: none | one-hot | {0, 1} |
| 11 | class: head-on | one-hot | {0, 1} |
| 12 | class: crossing | one-hot | {0, 1} |
| 13 | class: overtaking | one-hot | {0, 1} |
| 14 | class: being overtaken | one-hot | {0, 1} |
| **15** | **presence** | 1 when a track is held | {0, 1} |

Angles are sin/cos so the wrap at ±180° is not a discontinuity the network has
to learn around.

Distance and DCPA are both measured to the **ship domain**, not to the hull.
DCPA is normalised in domain radii, not metres.

TCPA keeps its sign: **positive means the CPA is ahead**, negative means it is
already passed and the range is opening.

Every value is a read of the shared `EncounterContext`, the same object the
reward reads, so observation and reward cannot derive the same encounter by
different routes (01 §5.3).

### 5.1 The five encounter classes

The one-hot order is frozen and matches `constants.ENCOUNTER_CLASSES`. The
class is decided against the **path tangent** at the own ship, not the
instantaneous heading (A19), with 3° bearing hysteresis and a 0.8 s (2-step)
persistence requirement.

| Class | Governing rule | Own-ship obligation | Compliant turn sense (A17) |
|---|---|---|---|
| none | — | Follow path | — |
| head-on | 14 | Alter to starboard, subject to channel width | starboard (+1) |
| crossing | 15, 16, 9(b) | Give way **regardless of approach side** | toward the target's side: starboard from starboard (+1), port from port (−1) |
| overtaking | 13, 16, 9(e) | Keep clear of the vessel being overtaken | port (−1) |
| being overtaken | 13, 17(a)(i) | Hold course and speed | — |

**Port and starboard crossing are one class**, and the side is not a separate
one-hot entry. Rule 9(b) — a vessel under 20 m shall not impede a vessel that
can safely navigate only within a narrow channel — makes the own ship give way
from either side, so the side is not a different obligation. This deliberately
replaces the Rule 18 route used by Meyer et al., whose premise (own ship much
smaller than the vessels it meets) fails here: own ship and target are
similarly sized model vessels, and claiming that asymmetry in simulation and
then validating against an identical vessel is an inconsistency a reviewer will
find.

**What changed in v3:** the side still decides the *direction* of the
give-way turn (A17), and the reward charges a turn against it (`v_port`). Since
v3 that direction is observable: the `context` branch carries the latched
**compliant turn sense** (§6), so the policy is shown which way it is being
judged (`test_crossing_class_keeps_an_observable_turn_direction`). The
geometric side itself remains available as `ObservationBuilder.crossing_sides`
for the passing-side term.

**Being overtaken** has no equivalent in the source table — Waltz & Okhrin
assume linear deterministic targets and cover only give-way cases. Only Rule
17(a)(i) course-keeping is rewarded; an earlier release under 17(a)(ii) is open
(A30).

### 5.2 Presence-bit semantics

**Zero-padding alone is unsafe.** Zero is a legitimate value for bearing sin,
for TCPA and for relative speed, so an ungated empty slot decodes as a target
sitting on top of the vessel on a matching course. The presence bit is what
separates "no target" from "a target with zero-valued features".

Consumers must **gate before the encoder, not after**.
`ASVFeaturesExtractor` multiplies the slot's inputs by the presence bit before
the slot encoder; gating only the output would still let the encoder's bias
terms contribute. `tests/test_observation.py` asserts that an absent slot filled
with arbitrary garbage produces byte-identical extractor output, and that the
whole target half of the feature vector is exactly zero when no target is held.

**No-target coverage.** A meaningful fraction of training episodes carries no
target at all, so the static-only configuration is in distribution (no-target
class share 0.17).

### 5.3 Slot assignment

* **Track-ID persistence decides slot position.** The slot is bound on first
  acquisition and held until track loss, so observation discontinuities coincide
  with real events rather than with re-sorting.
* **CRI decides admission** when slots are contested. Moot at one target, but
  the hook stays for the extension path, and it is still well-defined: a
  newcomer takes the single slot only if it is genuinely riskier than the
  incumbent.

The encounter classifier's history for a track is dropped at the same time its
slot is released, so a re-used slot cannot inherit the previous occupant's held
class.

## 6. `context` — 14 dims (new in v3)

Observable memory of the obligation the reward is judging (A25, F72). Indices
0–11 are per slot (`12 · N` at `N_MAX_TARGETS = N`, following the target slots);
the last two are the previous executed action. An empty slot is all zeros.

The encounter **engages** when the class is not none, 0 < TCPA ≤ 25 s and
DCPA < 1.5 · `d_req` (`d_req` = 2.5 m); class, compliant sense, heading and speed
at engagement are then **latched** (A20) until the target is past and opening.
"Latched" below means engaged or clearing.

| Idx | Feature | Encoding | Range |
|---|---|---|---|
| 0 | engaged | 1 while the encounter is engaged | {0, 1} |
| 1 | clearing | 1 while the target is passing clear | {0, 1} |
| 2 | **compliant turn sense** | +1 starboard, −1 port, 0 none (§5.1) | {−1, 0, 1} |
| 3 | heading change since engagement | wrapped deg / 180; 0 unless latched | [−1, 1] |
| 4 | speed change since engagement | `(u − u_engage) / SPEED_SCALE`; 0 unless latched | [−1, 1] |
| 5 | turn admissible | 1 if the compliant alteration fits the channel | {0, 1} |
| 6 | slowdown clears | 1 if stopping would clear the target (A18/A23 stop test) | {0, 1} |
| 7 | admissibility known | 1 if the map geometry was available for 5–6 | {0, 1} |
| 8 | action required | the alteration still owed, `A_req` (as in `v_r8`) | [0, 1] |
| 9 | proximity gate | `ρ = clip(1 − DCPA / (1.5 · d_req), 0, 1)` | [0, 1] |
| 10 | in extremis | 1 when DCPA < `d_req` and TCPA < 5 s | {0, 1} |
| 11 | engagement age | seconds since engagement / 25 s; 0 unless latched | [0, 1] |
| 12 | previous rudder | executed rudder command / 100 % | [−1, 1] |
| 13 | previous throttle | `(RPM − 6) / 6` as executed | [−1, 1] |

The previous action is the **executed** command (after any supervisor override),
so the policy sees what the vessel actually did.

**Measured use** (F91): probing trained PPO policies at engagement, flipping
index 2 alone moves the commanded rudder by 0.15–1.03, so the policy does read
the turn sense. That it still opens crossings in one fixed direction is a
learning result, not an observability gap (F92–F94).

---

## 7. Architecture

The six branches feed a custom `ASVFeaturesExtractor` inside SB3's
`MultiInputPolicy`, identical for every learner (PPO, RecurrentPPO, SAC, TQC;
TD3 is implemented but not in the baseline):

- **Scene MLP:** `lidar` + `boundary` + `ego` + `path` + previous action
  (42 values) → 128.
- **Slot encoder:** each slot's 16 target values + its 12 context values
  (28) → 64 → 64 → 32, input gated by the presence bit, weights shared across
  slots.
- Features (128 + 32 = 160) → actor and critic heads of 256 × 256 ReLU;
  RecurrentPPO adds a 256-unit LSTM, separately for actor and critic.

The shared-encoder-plus-DeepSets comparison from v1 is **not built** —
superseded decision D3. Permutation invariance is meaningless at one target, and
the measured advantages of attention in the literature come from high-density
regimes this scope deliberately does not enter.

A custom extractor is used for two reasons only: the presence bit has to gate
the slot before encoding, and the target branch keeps a small shared-weight
encoder so the multi-vessel extension path exists.

**Pre-empt the scaling question by scope**, not by principle: restricted
waterway, sequential encounters, single target in deployment, `N_MAX_TARGETS`
configurable.

**Recurrence.** Resolved by construction rather than left open: the explicit
tracker carries target memory, and the `context` branch carries the encounter
memory the reward depends on. RecurrentPPO is one of the five baselines, so the
paper measures whether a learned memory adds anything on top.

---

## 8. What is deliberately absent

Three fields present in the Paper 2 observation are **dropped**, confirmed as a
decision rather than an oversight: `front_clearance`, `side_clearance_diff` and
`local_target_cte`. These were LiDAR-derived local-planner cues, not raw sensor
data. `local_target_cte` in particular was the engineered bypass side-choice cue
that Paper 2's `target_side` and `field_repair` curricula existed to repair. The
position taken here is that side choice should be learned from `lidar` plus
`boundary` rather than supplied. Worth one sentence in the methods, because a
reviewer comparing observation tables between the two papers will notice three
features disappear.

Paper 2's configurable border-visibility mode (`OBS_BORDER_MODE`) is removed
entirely, not ported. That concept no longer exists.
