# 01 — Perception and Observation Space

**Revision 2** — single dynamic target. Supersedes the three-slot version.
**Handover target:** Claude Code (implementation-heavy)
**Depends on:** 02 for the encounter classifier definition and the precedence table
**Consumed by:** 03 (tracker interface), 04 (Study 2 design)

---

## 1. Purpose

Replace the Paper 2 observation with one that supports a dynamic target ship and COLREGs
role reasoning, and move the channel boundary out of the LiDAR channel into the map.

**Paper 2 observation (superseded):**

```
o_t = [ c_t (M=25 sector closeness), u, v, r, e_y, χ̃, χ̃_LA ]
```

Already path-relative — no global (x, y, ψ). Do not regress; the path-relative framing is
part of why transfer worked.

---

## 2. LiDAR configuration

**Hardware (confirmed):** RPLidar C1, 360°, 720 beams, θ = 0.5° uniform, 10 Hz.
Localisation via `rf2o_laser_odometry`.

### 2.1 Raw scan — full 360°

Required so the tracker can detect a vessel overtaking from astern (Rule 13, and the
"being overtaken" class). Non-negotiable.

### 2.2 Pooled `c_t` — forward-biased, obstacles only

`c_t` carries **static obstacles only**. Borders are gated out (§3); the dynamic target
goes through the target branch (§5). A static obstacle directly astern has been passed,
and the action space has no reverse, so aft sectors would be dead input.

Swath **±135°**; aft 90° reserved for the tracker. 27 sectors:

| Span | Sector width | Beams / sector | Sectors |
|---|---|---|---|
| ±45° (bow) | 6° | 12 | 15 |
| 45–90° each side | 11.25° | 22–23 | 8 |
| 90–135° each side | 22.5° | 45 | 4 |

Algorithm 1 (feasibility sector pooling) carries over unchanged — it computes arc width
from θ, which stays at 0.5°; only the sector *span* varies. Pass Φ per sector rather than
assuming a constant interval. Paper 2 stated constant-width sectors, so this needs one
sentence in the methods.

### 2.3 Extract from existing Paper 2 field logs before freezing the simulator

All three are sim-to-real gaps being built deliberately if ignored, and all three feed
**Study 2** (perception degradation).

1. **Returns per revolution.** Verify 720 against the logs. The C1 at 10 Hz may deliver
   nearer ~500 points/rev (≈0.7°). Simulating finer than the sensor delivers is a
   self-inflicted gap.
2. **Aft self-occlusion sector.** A 360° scanner on a hull with superstructure has a blind
   or degraded aft arc. Model as a masked bearing range. If the tracker is trained to see
   astern and the mount cannot, the being-overtaken class fails in the field for reasons
   unrelated to the policy.
3. **Motion distortion.** Points in one revolution are captured at different poses; at
   10 Hz with meaningful yaw rate the sweep smears. Small at these speeds, but it lands
   directly on velocity estimation.

---

## 3. Boundary branch

### 3.1 Rationale

The LiDAR is mounted **higher than the basin wall** and cannot be repositioned. It does not
register the pool edge at all — it registers the facility walls 1–2 m beyond it.

**Frame this as an architectural argument, not a workaround.** In a real narrow channel the
navigable limit is usually not a physical structure either: it is a charted depth contour, a
buoyed line, or a regulatory limit, none of which a LiDAR can see. A boundary channel
supplied from the chart, with range sensing reserved for physical obstacles, is therefore
the architecturally correct split for the application. The basin reproduces that situation
exactly — the boundary the vessel must respect is invisible to the sensor, while what the
sensor sees is not the boundary. Put this in the problem formulation.

The practical consequence is unchanged: Since the tracker estimates velocity from clustered
returns, a person walking past at scan height becomes a phantom target with plausible
kinematics. Geometric gating is a prerequisite for the tracker, not a convenience.

Also removes an existing inconsistency: Paper 2 computed the obstacle-proximity reward
from a border-excluded scan while the observation used a configurable border-visibility
mode. And it matches the positioning — Rule 9 presupposes a known channel.

### 3.2 Encoding — virtual range scan

Ray-cast against the known boundary polygon from the estimated pose at fixed body-frame
bearings, normalised to closeness identically to `c_t`.

Bearings: `{−90°, −60°, −30°, 0°, +30°, +60°, +90°}` — 7 rays. Chosen over a simple
`[d_port, d_stbd]` pair because it generalises to bends and varying width without an
interface change.

### 3.3 Two constraints this creates

**Redundancy trap.** If the path runs down the centreline of a constant-width channel,
port and starboard clearances are affine functions of `e_y` and the branch carries no new
information. It only earns its place when width varies, the path is off-centre, or the
channel bends. **Hard requirement on the scenario generator (doc 03/04)** — and now doubly
so, because Study 1 sweeps channel width.

**Pose noise injection.** In the field the virtual scan is computed from map plus
estimated pose, inheriting localisation error. A noiseless boundary scan in training
creates a second sim-to-real gap in the place this change was meant to remove one.
Randomise pose error at a magnitude drawn from the rf2o drift characterisation in 05.

### 3.4 Field-side gating

Per beam: compute the endpoint in the map frame from the estimated pose, discard if
outside the boundary polygon plus a margin. Apply the **identical** operation in
simulation so the pipelines are equivalent.

Margin set from localisation uncertainty. Too tight gates out real obstacles near the
wall; too loose lets clutter through. Both failure modes are worth measuring — the gate is
a load-bearing component of the perception stack and part of the N1 claim.

**O5 resolved — software gating, not a physical barrier.** The facility walls carry fixed
geometric features (recessed doorways, protruding benches) that are the only available
along-track constraint for the scan-to-map localisation in 05. A barrier would occlude
them. Run localisation on the **full** scan including the walls, and apply the obstacle
gate only afterwards for the tracker — the walls are a liability for target tracking and an
asset for localisation, and the pipeline should treat them as both.

Gating is mandatory rather than merely preferable: during trials, operators standing on the
deck sit at scan height and move. Without a geometric gate they become tracked dynamic
targets with plausible kinematics.

**Unchanged:** the true collision boundary is still enforced geometrically for termination
and penalties, as in Paper 2.

---

## 4. Target tracking pipeline

Sector closeness is velocity-blind: a wall at 4 m and a vessel closing at 1 m/s at 4 m
produce identical `c_t`. Explicit tracking is chosen over frame-stacking a pooled scan
because it also delivers information parity with the VO comparators, which require the
same quantities.

**This pipeline is the headline contribution (N1).** Its noise characteristics are not an
implementation detail — they are the object of Study 2.

Stages:

1. **Gate** beyond-boundary returns (§3.4)
2. **Cluster** remaining returns (DBSCAN or adaptive breakpoint on range)
3. **Ego-motion compensate** using odometry. Scan-matching odometry is itself corrupted by
   moving objects in the scan, so drift produces false velocities on *static* objects.
   This couples 01 and 05 directly
4. **Associate** clusters to tracks — nearest-neighbour is sufficient at one target
5. **Estimate** velocity per track (constant-velocity Kalman filter)
6. **Classify** static vs dynamic by speed threshold with hysteresis. **The threshold is set
   by localisation quality, not by obstacle behaviour** — field obstacles are confirmed
   stable, so apparent motion of static objects comes almost entirely from ego-pose error,
   which affects every object in the scan identically. Set it from measured pose noise
   (05 §4) and retighten as registration improves. Bias toward under-detection: promoting a
   static panel to a target ship is a false positive with COLREGs consequences

Static clusters continue to feed `c_t`; the dynamic track feeds the target branch.

### 4.1 Degradation parameters (Study 2)

Expose as environment config so the sweep in 04 can drive them:

| Parameter | Source |
|---|---|
| Pose drift magnitude and character | rf2o characterisation (05) |
| Detection dropout rate | field logs |
| Occlusion duration | scenario geometry + aft self-occlusion |
| Velocity estimate noise | scan distortion + filter residual |

---

## 5. Derived target features

### 5.1 CPA / DCPA / TCPA

With relative position **p** = p_TS − p_OS and relative velocity **v** = v_TS − v_OS:

```
TCPA  = −(p · v) / |v|²
DCPA  = |p + v · TCPA|
```

TCPA > 0 means the CPA is ahead; TCPA < 0 means passed and opening.

**Known failure mode.** CPA assumes both vessels hold course and speed. Two ships close on
near-parallel courses have a CPA far in the past or future, so CPA-based risk reads low —
but a slight turn by either makes |TCPA| collapse. In a corridor, near-parallel geometry is
the *normal* case, so this matters more here than in the open-water literature.

### 5.2 Collision Risk Index

Follow Waltz & Okhrin (2023) §3.3:

```
CR = 1                    if TS inside OS ship domain
CR = max(CR_CPA, CR_ED)   otherwise
```

`CR_CPA` decays exponentially in DCPA and |TCPA| with different rates before and after the
CPA. `CR_ED` is a Euclidean-distance risk — the patch for the near-parallel failure mode,
not optional for a channel. DCPA measured to the **ship domain**, not the hull. A
bow-crossing factor inflates risk when the CPA would place the OS across the target's bow.

**Do not copy their constants.** Tuned for a 320 m KVLCC2 over a 14 NM box with decay
scaled to 2 NM. Re-derive in ship lengths against LBP = 1.57 m.

**Ship domain — compressed asymmetric (resolved, provisional).** Chun's 3·Lpp fore/aft and
1·Lpp abeam gives 4.7 m fore-aft at Lpp = 1.57 m, leaving almost no room in a 10 m channel.
Use instead:

| Direction | Multiple | Metres |
|---|---|---|
| Ahead | 2.0 · Lpp | 3.14 |
| Astern | 1.0 · Lpp | 1.57 |
| Abeam (each side) | 0.75 · Lpp | 1.18 |

Lateral footprint 2.36 m, about 24% of a 10 m channel.

**The principle matters more than the numbers.** Do not defend these as a scaled copy of
someone else's domain. Derive the final values from measured manoeuvring performance —
advance and tactical diameter from the turning-circle tests, stopping distance from the stop
test, all in 05. The domain is then "sized to this vessel's demonstrated ability to avoid",
which is the argument Thyri & Breivik make for confined water. Treat the table above as a
provisional input and the final values as an output of 05.
Szłapczyński & Szłapczyńska (2017) is the reference for justifying the compression.

### 5.3 Encounter classifier — five classes

**One module, two consumers.** The same function feeds the observation feature and the
reward gate in 02. If they diverge even at sector boundaries, the agent is penalised for a
role it was not shown. Hysteresis applied once, inside the module.

| Class | Governing rule | Own-ship obligation |
|---|---|---|
| None | — | Follow path |
| Head-on | 14 | Alter to starboard, subject to channel width |
| Crossing | 15, 16, 9(b) | Give way **regardless of approach side** |
| Overtaking | 13, 16, 9(e) | Keep clear of the vessel being overtaken |
| Being overtaken | 13, 17(a)(i) | Hold course and speed |

Baseline thresholds from Waltz & Okhrin Table 1 (after Xu et al. 2020), where α is
relative bearing OS→TS and CT the heading intersection angle:

| Class | Requirement |
|---|---|
| Head-on | α ∈ [0°,5°] ∪ [355°,360°); CT ∈ [175°,185°] |
| Crossing (starboard) | α ∈ [5°,112.5°]; CT ∈ [185°,292.5°] |
| Crossing (port) | α ∈ [247.5°,355°]; CT ∈ [67.5°,175°] |
| Overtaking | α_TS→OS ∈ [112.5°,247.5°]; CT ∈ [0°,67.5°] ∪ [292.5°,360°); U_OS > U_TS |
| None | otherwise |

**Three required modifications:**

1. **Port and starboard crossing collapse into one class** under Rule 9(b) — the own ship
   gives way either way. Keep the geometric distinction internally if the passing-side
   reward term needs it, but the observation one-hot has a single crossing class.
2. **Head-on band widened to ±10°** (resolved). The source value of ±5° is tight enough that
   a small heading error flips the classification; ±10° is within common practice and gives
   the hysteresis room to work.
3. **Add the "being overtaken" class** — the source table has no equivalent, because
   Waltz & Okhrin assume linear deterministic targets and cover only give-way cases.
   Mirror of the overtaking condition with U_TS > U_OS.

---

## 6. Observation specification

Dict observation, five branches, multi-input policy.

| Branch | Contents | Dim |
|---|---|---|
| `lidar` | `c_t` sector closeness, **obstacles only**, ±135° | 27 |
| `boundary` | virtual range scan, normalised, pose noise injected | 7 |
| `ego` | u, v, r | 3 |
| `path` | e_y, χ̃, χ̃_LA | 3 |
| `target` | 15 features + presence bit | 16 |
| | **Total** | **≈56** |

### 6.1 Target features (16)

| Feature | Encoding |
|---|---|
| Distance to ship domain | normalised by `d_scale` |
| Relative bearing α | sin, cos (2) |
| Heading intersection angle CT | sin, cos (2) |
| Target speed | normalised |
| Relative speed | normalised |
| DCPA | normalised by domain radius, **not** metres |
| TCPA | clipped and normalised |
| CRI | already ∈ [0,1] |
| Encounter class | one-hot (5) |
| Presence bit | 1 when a track is held |

Angles as sin/cos to avoid wraparound discontinuity.

### 6.2 Track and slot management

- **One target slot.** `N_max` is a config parameter and the branch is built as an indexed
  slot, so multi-vessel extension is a retrain rather than a redesign (S1)
- **Track-ID persistence** — the slot is bound on first acquisition and held until track
  loss, so observation discontinuities coincide with real events rather than with
  re-sorting. CRI-based ordering is moot at one target but the hook stays for extension
- **Presence bit rather than a mask vector.** Zero-padding alone is dangerous because zero
  is a legitimate value for bearing and relative speed, so a zero-padded empty slot looks
  like a target on top of you on a matching course
- **No-target coverage.** A meaningful fraction of training episodes must have no target at
  all, or the static-only configuration is out of distribution

### 6.3 Architecture

Plain concatenation of the five branches into the SAC multi-input policy. The shared
per-slot encoder and sum-pooling ablation from Revision 1 are **not needed at one target**
— keep the target branch behind a small encoder so the extension path exists, but do not
build or defend the DeepSets comparison.

**Pre-empt the scaling question by scope**, not by principle: restricted waterway,
sequential encounters, single target in deployment, `N_max` configurable.

**SB3 note.** Five-branch Dict observation requires `MultiInputPolicy`; a custom features
extractor is only needed if per-branch encoders are used.

**Open:** whether recurrence is added for occlusion. The explicit tracker carries some
memory already. Quantify occlusion frequency in the scenario distribution before adding
it — and note that if recurrence is added, RecurrentPPO stops being a clean comparator.

---

## 7. Literature grounding

| Source | Use |
|---|---|
| Waltz & Okhrin (2023) *Neural Networks* 165:634–653 | CPA/CRI (§3.3), encounter table (§4.3), "Around the Clock" benchmark. **Read in full before finalising the observation vector** |
| Villa, Aaltonen & Koskinen, *IEEE/ASME Trans. Mechatronics* | LiDAR-based path following and avoidance in harbour conditions — closest platform and setting match |
| Han et al. (2020) *J. Field Robotics* 37(6):987–1002 | LiDAR/radar track fusion feeding COLREGs manoeuvres, field-verified. Benchmark for the perception-to-action pipeline |
| Kim et al. (2022) *Ocean Engineering* | 2D LiDAR detection embedded in a physical catamaran ASV, simulation and experiment |
| Szłapczyński & Szłapczyńska (2017) *Ocean Eng.* 145:277–289 | Ship domain review — needed to justify compression at model scale |

**Two gaps to foreground as contribution:** Waltz & Okhrin assume linear deterministic
targets, covering only give-way cases; and their perception is AIS, so target course and
speed arrive for free. Estimating them from 2D LiDAR with drifting scan-matching odometry
is the harder problem and the basis of N1.

---

## 8. Open items

- ~~O5~~ — resolved: software gating, geometric against the pool polygon. Facility walls retained as the localisation reference (05 §4.2)
- Verify 720 beams against field logs; characterise aft occlusion and scan distortion
- Re-derive all CRI constants in ship lengths; finalise the ship domain from the identified turning circle (05)
- Define "being overtaken" thresholds
- Decide whether recurrence is added for occlusion
- Expose the Study 2 degradation parameters as environment config
- Set the clustering minimum-points threshold to reject suspension-line returns (thin, 1–2
  beams) without rejecting genuine small obstacles
- Set the static/dynamic threshold from measured pose noise once 05 §4 reports it
