# Claude Code Kickoff — Paper 3, Perception and Observation Rebuild

**Repo:** `asv-lidar`
**Reference (read-only):** `asv-lidar/static_obstacles/src/`
**Working directory:** `asv-lidar/static_dynamic_obstacles/`

**Attach alongside this file:**
`PROJECT_BRIEF.md`, `00_PAPER3_INDEX_AND_PROTOCOL.md`,
`01_PERCEPTION_AND_OBSERVATION.md`, `05_VESSEL_MODEL_AND_SIM2REAL.md`

---

## 0. What this session is

Paper 2 (`static_obstacles/`) handled path following plus **static** obstacle
avoidance and is published. Paper 3 extends to **dynamic targets and COLREGs
compliance in a narrow channel**.

`01_PERCEPTION_AND_OBSERVATION.md` is the **authoritative specification** for this
session. Where this file and 01 disagree, 01 wins. This file does not restate the
spec — read 01 in full before writing code.

This session builds the perception and observation layer only. **No reward work, no
training runs, no scenario generation.** Those are 02, 03, and 04.

---

## 1. Directory rules — read carefully

`static_obstacles/` is a **frozen published baseline**. The Paper 2 SAC policy is a
comparison baseline in Paper 3 and must stay reproducible for at least another
eighteen months.

- **Never write to, refactor, reformat, or delete anything under
  `static_obstacles/`.** Read only.
- All new code goes in `static_dynamic_obstacles/`.
- When reusing a Paper 2 file, **copy it across** and edit the copy. Do not import
  across the directory boundary — a later edit in Paper 3 must not be able to change
  Paper 2 behaviour.
- Before starting, confirm the working tree is clean and note the current commit SHA
  in your first report.

**Ignore the repo-root `README.md`.** It describes a much earlier iteration (63 beams,
270° swath, MultiDiscrete observation, 150 m ranges, discrete rudder). That is not the
Paper 2 system. `static_obstacles/src/` is the only ground truth for what Paper 2
actually does.

---

## 2. Step 0 — inventory before building

Read `static_obstacles/src/` and produce
`static_dynamic_obstacles/PORTING_MANIFEST.md`: every source file, one line on what it
does, and a bucket assignment (§3). Expected files include the environment (~1400
lines), `ship_model.py`, `asv_lidar.py`, and `lidar_pooling.py`, but **verify the
actual filenames** rather than assuming them.

Stop after the manifest and report. Do not start building until it is reviewed.

---

## 3. Bucket rules

### Bucket A — copy verbatim

Working, calibrated, and unaffected by the observation change. Copy across unmodified;
a recalibration pass on the hydrodynamic coefficients comes later under 05.

- 3-DOF Fossen ship model
- LOS path geometry: `e_y`, `χ̃`, `χ̃_LA`, look-ahead distance
- Action space (residual rudder + propulsion around nominal cruise)
- Geometric collision and termination checks
- SB3 training harness, curriculum machinery, logging, evaluation loop

### Bucket B — copy and modify, one stated change each

- **`asv_lidar.py`** — raycast engine unchanged. Output becomes **obstacles only**;
  border geometry moves out of the LiDAR channel entirely (01 §3).
- **`lidar_pooling.py`** — Algorithm 1 (feasibility sector pooling) is unchanged in
  substance, but must accept **per-sector angular span Φ** instead of assuming a
  constant interval. Per-beam resolution θ stays 0.5°; only the sector span varies.
  Verify the non-uniform allocation sums correctly: 15 + 8 + 4 = 27 sectors.

### Bucket C — rebuild from specification

For these, **build from `01_PERCEPTION_AND_OBSERVATION.md` and §5 below. Do not open
the Paper 2 equivalents for reference.** Decision D10 in the protocol document is that
the observation is redesigned from scratch, not patched. Reading the old version first
reliably produces a patched version.

- Observation assembly — `Box` becomes a 5-branch `Dict`
- Target tracking pipeline (gate → cluster → ego-motion compensate → associate →
  Kalman → static/dynamic split)
- CPA / DCPA / TCPA, CRI, ship domain
- Encounter classifier
- Boundary virtual raycast branch
- Custom SB3 features extractor with slot masking (`MlpPolicy` cannot do this)

The configurable border-visibility mode from Paper 2 is **removed**, not ported. That
concept no longer exists.

---

## 4. Build order

1. `PORTING_MANIFEST.md` — then stop and report
2. Bucket A copies; confirm the ported environment still steps end-to-end
3. `constants.py` + `CONSTANTS_AND_SCALES.md` (§5) — before any consumer is written
4. `lidar_pooling.py` with per-sector Φ; unit-test sector/beam counts
5. `boundary_raycast.py` — 7 rays at {−90, −60, −30, 0, +30, +60, +90}°, normalised
   identically to `c_t`, with a pose-noise injection hook
6. `tracking.py` — clustering through Kalman velocity estimation
7. `cpa_cri.py` — geometry and risk
8. `encounter.py` — **one module, two consumers.** 02 will import this exact function
   for the reward. Export a single pure function plus a hysteresis-holding wrapper;
   do not duplicate the logic anywhere.
9. `observation.py` — Dict assembly, slot management, valid mask
10. `features_extractor.py` — shared per-slot encoder φ, `aggregate` config flag
    (`concat` default, `sum` behind the flag)

---

## 5. Constants — single source of truth

Create `static_dynamic_obstacles/constants.py` with a documented dataclass or typed
config, mirrored in `CONSTANTS_AND_SCALES.md`.

Several values are **not yet decided**. Where a value is missing, define the symbol,
give a clearly labelled placeholder, and mark it `TODO(O#)` or `TODO(decision)`.
**Do not silently invent a value and bury it in a function body.** Every unresolved
constant must be visible in one file.

Known-unresolved, at minimum:

| Symbol | Status |
|---|---|
| CRI decay rates (pre- and post-CPA), bow-crossing factor | `TODO` — re-derive in ship lengths against LBP = 1.57 m |
| Ship domain fore/aft/lateral extent | `TODO` — Waltz & Okhrin use 3·Lpp fore-aft, which is 4.7 m in a 10 m channel and does not fit |
| Domain radius used to normalise DCPA | `TODO` — undefined for an asymmetric domain; decide the convention |
| `d_scale`, TCPA clip bounds, speed normalisers | `TODO` — depends on O4, final workspace size |
| Head-on band half-width | `TODO` — Waltz & Okhrin use ±5°; spec says widen toward ±6–10° |
| "Being overtaken" class thresholds | `TODO` — not in the source table, must be defined |
| Static/dynamic speed threshold and hysteresis band | `TODO` |
| Boundary raycast pose-noise magnitude | `TODO(05)` — default 0.0 for now, but the hook must exist |
| Kalman process and measurement noise | `TODO(05)` |

Placeholders must make the code run, not make it look finished. Anything you assume
goes in the report.

---

## 6. Observation contract — freeze it

Write `OBSERVATION_SPEC.md` giving the **explicit index order** of every element in
every branch: 27 + 7 + 3 + 3 + 51 = 91 dims, with the 51 broken out as 3 slots × 16
features + 3 mask bits, and the 16 per-slot features listed in fixed order.

Every checkpoint and every frozen evaluation case depends on this ordering. It must be
written down before the first training run, not inferred from the code later.

Mask semantics, explicitly: zero-padding alone is unsafe because zero is a legitimate
value for bearing and relative speed, so an unmasked empty slot reads as a target
sitting on top of the vessel on a matching course. Mask before pooling. With
sum-pooling, divide by the count of **valid** slots, not `N_max`.

---

## 7. Tests

Minimum, all runnable without a trained policy:

- Sector pooling: beam counts per sector, total = 27, no gaps or overlaps across ±135°
- Boundary raycast: known polygon, known pose, hand-checked ranges; correct behaviour
  at a bend and at varying width
- CPA: head-on gives TCPA > 0 and DCPA ≈ 0; already-passed geometry gives TCPA < 0;
  near-parallel courses give a large |TCPA| and must be caught by the Euclidean-distance
  risk term rather than by CPA
- Encounter classifier: one case per class including "being overtaken"; verify
  hysteresis prevents chatter at sector boundaries
- Observation: shape and dtype exactly match `observation_space`; 0-, 1-, 2- and
  3-target cases all produce finite values with correct mask bits
- Features extractor: a masked slot filled with arbitrary garbage produces byte-identical
  output to the same slot filled with zeros

---

## 8. Acceptance checks

Before reporting complete:

- `git status` shows no modifications anywhere under `static_obstacles/`
- No import in `static_dynamic_obstacles/` reaches into `static_obstacles/`
- Every unresolved constant appears in `constants.py` with a `TODO` marker, and
  nowhere else
- The encounter classifier exists in exactly one place
- No reward term names survive from Paper 2 into new code (`r_pf`, `r_oa`, `λ`, `g_u`,
  `w_χ`). Some may legitimately return in 02, but they must arrive by decision, not by
  inheritance — their presence now means Bucket C was patched rather than rebuilt.

---

## 9. Report back

1. Commit SHA and confirmation the tree was clean at the start
2. `PORTING_MANIFEST.md` bucket assignments
3. Every `TODO` constant, with the placeholder used
4. Any place where 01's specification was ambiguous or internally inconsistent, and
   what you assumed
5. Anything in the Paper 2 code that contradicts 01's description of it

Flag disagreements rather than resolving them silently. Unresolved decisions here are
deliberate, not oversights — several of them are waiting on a basin booking.
