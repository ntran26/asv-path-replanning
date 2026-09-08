# OBSERVATION SPEC — Paper 3

**Version:** `obs-v2` — two-vessel repositioning. Supersedes `obs-v1` (91 dims,
three slots, six encounter classes), which was never trained against.

**Status:** frozen. Every checkpoint and every evaluation case depends on this
ordering. Changing any index is a version bump, not an edit.

**Index order is unchanged since v2 was frozen.** Revision 2.2 changed one
*normaliser* (`SPEED_SCALE`, §3) and the head-on band feeding the class one-hot.
Neither moves an index, but both change what a trained checkpoint means, so
anything trained before this point would not be comparable. Nothing has been
trained.

Total **56** dims across **5** branches, as a `gymnasium.spaces.Dict`.
The machine-readable copy of this layout is `src/observation.py`; the two must
agree, and `tests/test_observation.py` enforces the dimensions.

| Branch | Contents | Dims | Range |
|---|---|---|---|
| `lidar` | `c_t` sector closeness, **static obstacles only** | 27 | [0, 1] |
| `boundary` | virtual boundary raycast, normalised | 7 | [0, 1] |
| `ego` | u, v, r | 3 | [-1, 1] |
| `path` | e_y, χ̃, χ̃_LA | 3 | [-1, 1] |
| `target` | 15 features + 1 presence bit | 16 | [-1, 1] |
| | **Total** | **56** | |

All arrays are `float32`.

**Frame conventions.** World: +y north, +x east. Headings and bearings are
compass style — 0° is +y, increasing clockwise. Relative bearing `α` is measured
from the own ship's heading: 0° dead ahead, 90° abeam to starboard. Positive
cross-track error is **port** of the path (Paper 2's convention, carried over).

### What changed from v1, and why

| | v1 | v2 |
|---|---|---|
| Target slots | 3 + a 3-bit mask vector | **1 + a presence bit** |
| Encounter classes | 6 (crossing split give-way / stand-on) | **5 (crossing collapsed)** |
| Total dims | 91 | **56** |
| Architecture | shared encoder + DeepSets/attention flag | **plain concatenation** |

Driven by decisions S1, S3, S4 and S6: two-vessel encounters are the unit of
analysis, because Rules 13–16 are formulated pairwise, confined geometry
precludes simultaneous close-quarters conflicts so encounters are sequential,
and every reported behaviour becomes physically reproducible in the basin.

`N_MAX_TARGETS` remains a config parameter and the slot machinery is still
indexed, so multi-vessel extension costs a retrain rather than a redesign.
`tests/test_observation.py::test_slot_machinery_still_scales_past_one` exercises
that path so it cannot rot.

---

## 1. `lidar` — 27 dims

Pooled sector closeness, `1 - range / 16.0`, clipped to [0, 1]. **1 = touching,
0 = clear to max range.**

Carries **static obstacles only**. The channel boundary is gated out and reaches
the policy through `boundary`; the dynamic target goes through `target`.

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

Virtual range scan against the known channel polygon, from the **estimated**
pose, normalised by the identical `closeness_from_ranges` used for `c_t`.

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

Pose noise is injected before the raycast so the branch inherits localisation
error, as it will in the field. Magnitudes are `TODO(05)` and currently 0.0.

## 3. `ego` — 3 dims

| Idx | Symbol | Quantity | Normaliser |
|---|---|---|---|
| 0 | u | surge velocity | `SPEED_SCALE` = `U_MAX_SURGE` (3.2 m/s) |
| 1 | v | sway velocity | `SPEED_SCALE` |
| 2 | r | yaw rate, deg/s | 180 |

> **Normaliser corrected in Revision 2.2.** `SPEED_SCALE` was `2 × U_CRUISE`
> = 1.10 m/s, which sits *below* the vessel's stage-1 operating range — the
> simulator reaches 1.35–2.16 m/s at 9–15 RPM. Index 0 was therefore pinned at
> 1.0 for roughly 45% of a plain straight run and carried no gradient at all.
> It is now tied to hull capability (`U_MAX_SURGE`, the steady speed at the
> widest curriculum stage) rather than to cruise, so the meaning does not shift
> when the curriculum widens the propulsion range mid-training.

**An IMU is confirmed** (05 §4.7). `r` is measured by the gyro rather than
differentiated, so its residual is the sensor noise floor; `u` and `v` are
largely rescued by the accelerometer but remain fused rather than measured. The
branch still carries field error Paper 2's simulator did not model — a
sim-to-real gap in the *observation*, not just the dynamics (05 §6) — but a
smaller one than before. `EGO_SPEED_NOISE` and `EGO_YAW_RATE_NOISE_DPS` are the
hooks; both are `TODO(05)` and currently 0.0.

## 4. `path` — 3 dims

| Idx | Symbol | Quantity | Normaliser |
|---|---|---|---|
| 0 | e_y | cross-track error, signed, + = port | `max(MAP_WIDTH, MAP_HEIGHT)` |
| 1 | χ̃ | course error, deg | 180 |
| 2 | χ̃_LA | look-ahead course error, deg | 180 |

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
| 0 | Distance to ship domain | `/ D_SCALE`, clipped | [0, 1] |
| 1 | Relative bearing α | sin | [−1, 1] |
| 2 | Relative bearing α | cos | [−1, 1] |
| 3 | Heading intersection CT | sin | [−1, 1] |
| 4 | Heading intersection CT | cos | [−1, 1] |
| 5 | Target speed | `/ SPEED_SCALE` (3.2 m/s), clipped | [0, 1] |
| 6 | Relative speed | `/ SPEED_SCALE`, clipped | [0, 1] |
| 7 | DCPA | `/ DOMAIN_RADIUS_DCPA`, clipped at `DCPA_CLIP_DOMAINS`, rescaled | [0, 1] |
| 8 | TCPA | clipped to ±`TCPA_CLIP`, `/ TCPA_CLIP` | [−1, 1] |
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

### 5.1 The five encounter classes

The one-hot order is frozen and matches `constants.ENCOUNTER_CLASSES`.

| Class | Governing rule | Own-ship obligation |
|---|---|---|
| none | — | Follow path |
| head-on | 14 | Alter to starboard, subject to channel width |
| crossing | 15, 16, 9(b) | Give way **regardless of approach side** |
| overtaking | 13, 16, 9(e) | Keep clear of the vessel being overtaken |
| being overtaken | 13, 17(a)(i) | Hold course and speed |

**Port and starboard crossing are one class.** Rule 9(b) — a vessel under 20 m
shall not impede a vessel that can safely navigate only within a narrow channel
— makes the own ship give way from either side, so the side is not a different
obligation and the observation does not carry it. This deliberately replaces the
Rule 18 route used by Meyer et al., whose premise (own ship much smaller than
the vessels it meets) fails here: own ship and target are similarly sized model
vessels, and claiming that asymmetry in simulation and then validating against
an identical vessel is an inconsistency a reviewer will find.

The geometric side **is** still computed, and is available as
`ObservationBuilder.crossing_sides` / `encounter.crossing_side()`, because 02's
passing-side reward term needs it. It is simply not observed.

**Being overtaken** has no equivalent in the source table — Waltz & Okhrin
assume linear deterministic targets and cover only give-way cases. Only Rule
17(a)(i) passive course-keeping is in scope; active release under 17(a)(ii) is
future work (S5).

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

**No-target coverage.** A meaningful fraction of training episodes must carry no
target at all, or the static-only configuration is out of distribution.
`NO_TARGET_EPISODE_PROB` is the hook; the distribution itself is 03/04's.

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

---

## 6. Architecture

Plain concatenation of the five branches into the SAC `MultiInputPolicy`
(01 §6.3). The shared-encoder-plus-DeepSets comparison from v1 is **not built** —
superseded decision D3. Permutation invariance is meaningless at one target, and
the measured advantages of attention in the literature come from high-density
regimes this scope deliberately does not enter.

A custom extractor is still used, for two reasons only: the presence bit has to
gate the slot before encoding, and the target branch keeps a small shared-weight
encoder so the multi-vessel extension path exists.

**Pre-empt the scaling question by scope**, not by principle: restricted
waterway, sequential encounters, single target in deployment, `N_MAX_TARGETS`
configurable.

**Open:** whether recurrence is added for occlusion. The explicit tracker already
carries memory (`max_coast` measures how long it survives one). Quantify
occlusion frequency in the scenario distribution before adding it — and note
that if recurrence is added, RecurrentPPO stops being a clean comparator.

---

## 7. What is deliberately absent

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
