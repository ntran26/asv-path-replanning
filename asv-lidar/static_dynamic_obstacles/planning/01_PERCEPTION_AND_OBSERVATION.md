# 01 — Perception and Observation Space

**Handover target:** Claude Code (implementation-heavy)
**Depends on:** nothing — start here
**Consumed by:** 02 (encounter classifier), 03 (LiDAR/tracker interface)

---

## 1. Purpose

Replace the Paper 2 observation with one that supports dynamic target ships and
COLREGs role reasoning, and move the channel boundary out of the LiDAR channel and
into the map.

**Paper 2 observation (superseded):**

```
o_t = [ c_t (M=25 sector closeness), u, v, r, e_y, χ̃, χ̃_LA ]
```

Note this is already path-relative — it does **not** contain global (x, y, ψ). Do not
regress to global coordinates; the path-relative framing is part of why transfer
worked.

---

## 2. LiDAR configuration

**Hardware (confirmed):** RPLidar C1, 360° swath, 720 beams, θ = 0.5° uniform,
10 Hz scan rate, scanning range 1-16m (out of range returns 0). 
Localisation via `rf2o_laser_odometry` (ROS2).

### 2.1 Raw scan — full 360°

Required so the tracker can detect overtaking vessels approaching from astern
(Rule 13). Non-negotiable.

### 2.2 Pooled `c_t` — forward-biased, obstacles only

`c_t` now carries **static obstacles only**. Borders are gated out (§3); dynamic
targets go through the target branch (§5). A static obstacle directly astern is one
already passed, and the action space has no reverse — so aft sectors would sit at max
range almost always and constitute dead input.

Swath: **±135°**. Aft 90° reserved for the tracker.

Non-uniform sector allocation, 27 sectors total:

| Span | Sector width | Beams / sector | Sectors |
|---|---|---|---|
| ±45° (bow) | 6° | 12 | 15 |
| 45–90° each side | 11.25° | 22–23 | 8 |
| 90–135° each side | 22.5° | 45 | 4 |

Angular precision matters far more ahead than abeam. This keeps the count close to
Paper 2's 25 while improving bow resolution.

**Implementation note.** Algorithm 1 (feasibility sector pooling) carries over
unchanged — it computes arc width from the per-beam angular resolution θ, which stays
constant at 0.5°; only the sector *span* varies. Pass Φ per sector rather than assuming
a constant interval. Paper 2 stated that sectors span a constant angular interval, so
this change needs one sentence in the methods.

### 2.3 Items to extract from the existing Paper 2 field logs

Do this **before freezing the simulator**. All three are sim-to-real gaps being built
deliberately if ignored.

1. **Returns per revolution.** Verify 720 against the logs. The C1's sample rate at
   10 Hz may deliver closer to ~500 points/rev (≈0.7°) than the nominal 0.5°.
   Simulating finer than the sensor delivers is a self-inflicted gap.
2. **Aft self-occlusion sector.** A 360° scanner on a hull with any superstructure has
   a blind or degraded aft arc. Find it, model it as a masked bearing range. If the
   tracker is trained to see astern and the real mount cannot, overtaking detection
   fails in the field for reasons unrelated to the policy.
3. **Motion distortion.** Points in one revolution are captured at different vessel
   poses; at 10 Hz with meaningful yaw rate during avoidance manoeuvres the sweep
   smears. Small at these speeds, but it lands directly on velocity estimation — the
   one thing the tracker exists to do. Either compensate with odometry or characterise
   the residual and inject it as tracker noise during training.

---

## 3. Boundary branch (new)

### 3.1 Rationale

The physical LiDAR sits **higher than the test basin boundary** and cannot be
repositioned, so the field scan contains returns from beyond the wall — equipment,
railings, people moving around the basin. Since the tracker estimates velocity from
clustered returns, a person walking past at scan height becomes a phantom target ship
with plausible kinematics. Geometric gating is therefore a prerequisite for the
tracker, not a convenience.

Two supporting arguments:

- It removes an existing inconsistency. Paper 2 already computed the obstacle-proximity
  reward from a border-excluded scan while the observation used a configurable
  border-visibility mode. Making the split total removes an asymmetry.
- It matches the positioning. Rule 9 presupposes a known channel; restricted waterways
  are surveyed by definition.

### 3.2 Encoding — virtual range scan

Ray-cast against the known boundary polygon from the estimated pose at fixed body-frame
bearings, then normalise to closeness identically to `c_t`.

Bearings: `{−90°, −60°, −30°, 0°, +30°, +60°, +90°}` — 7 rays.

Chosen over a simple `[d_port, d_stbd]` pair because it generalises to bends and
varying width without an interface change, and reuses the existing normalisation.

### 3.3 Two constraints this creates

**Redundancy trap.** If the reference path runs down the centreline of a
constant-width channel, port and starboard clearances are affine functions of `e_y` —
the branch carries no new information and a reviewer will notice. The boundary branch
only earns its place when width varies, the path is deliberately off-centre, or the
channel bends. **This is a hard requirement on the scenario generator (see 04).**

**Pose noise injection.** In the field, the virtual scan is computed from map + estimated
pose, so it inherits localisation error. A noiseless boundary scan in training creates a
second sim-to-real gap in exactly the place this change was meant to remove one.
Randomise pose error into the boundary raycast at a magnitude drawn from the rf2o drift
characterisation in 05.

### 3.4 Field-side gating

For each beam: compute the endpoint in the map frame from the estimated pose; discard
if it falls outside the boundary polygon plus a margin. Apply the **identical**
operation in simulation (ray-cast obstacles only, never borders) so the two pipelines
are equivalent.

Set the margin from localisation uncertainty. Too tight gates out real obstacles near
the wall; too loose lets beyond-wall clutter through. Both failure modes are worth
measuring and reporting — the gate is now a load-bearing component of the perception
stack.

**Check first (O5):** a physical barrier at scan height — corflute, shade cloth, foam
board — makes the LiDAR see a real wall and removes the problem *and* its localisation
dependency. Worth ten minutes with whoever manages the basin before engineering around
it.

**Unchanged:** the true collision boundary is still enforced geometrically for
termination and penalties, exactly as in Paper 2. Keep the separation between what the
policy sees and what counts as a collision.

---

## 4. Target tracking pipeline (new)

Sector closeness is velocity-blind: a wall at 4 m and a vessel closing at 1 m/s at 4 m
produce identical `c_t`. Explicit tracking is the chosen resolution (rather than
frame-stacking a pooled scan), because it also delivers information parity with the
VO and DWA baselines, which likewise require tracked target state.

Pipeline stages:

1. **Gate** beyond-boundary returns (§3.4).
2. **Cluster** remaining returns (DBSCAN or adaptive-breakpoint on range).
3. **Ego-motion compensate** using odometry. Critical: scan-matching odometry is itself
   corrupted by moving objects in the scan, so drift produces false velocities on
   *static* objects. This couples 01 and 05 directly.
4. **Associate** clusters to tracks (nearest-neighbour or JPDA; nearest-neighbour is
   almost certainly sufficient at N ≤ 3).
5. **Estimate** velocity per track (constant-velocity Kalman filter).
6. **Classify** static vs dynamic by speed threshold with hysteresis.

Static clusters continue to feed `c_t`; dynamic tracks feed the target branch.

---

## 5. Derived target features

### 5.1 CPA / DCPA / TCPA

With relative position **p** = p_TS − p_OS and relative velocity **v** = v_TS − v_OS:

```
TCPA  = −(p · v) / |v|²
DCPA  = |p + v · TCPA|
```

TCPA > 0 means the CPA is ahead; TCPA < 0 means already passed and opening.

**Known failure mode.** CPA assumes both vessels hold course and speed. Two ships close
on near-parallel courses have a CPA far in the past or future, so CPA-based risk reads
low — but a slight turn by either makes |TCPA| collapse and the situation becomes
urgent instantly. In a narrow channel, near-parallel geometry is the *normal* case, so
this matters more here than in the open-water literature it comes from.

### 5.2 Collision Risk Index

Follow Waltz & Okhrin (2023, *Neural Networks* 165:634–653), §3.3:

```
CR = 1                              if TS inside OS ship domain
CR = max(CR_CPA, CR_ED)             otherwise
```

- `CR_CPA` decays exponentially in DCPA and |TCPA|, with **different decay rates before
  and after** the CPA so risk drops quickly once passed.
- `CR_ED` is a plain Euclidean-distance risk. This is the patch for the near-parallel
  failure mode above — it is not optional for a channel.
- DCPA is measured to the **ship domain**, not the hull.
- A bow-crossing factor inflates risk when the CPA would place the OS across the
  target's bow.

**Do not copy their constants.** They are tuned for a 320 m KVLCC2 over a 14 NM box with
decay scaled to 2 NM (3704 m). The paper says explicitly that they must be adjusted for
a vessel with different characteristics. Re-derive everything in ship lengths against
LBP = 1.57 m.

**Ship domain arithmetic — check this early.** Their asymmetric domain (Chun et al.)
uses 3·Lpp fore and aft, 1·Lpp each side. At Lpp = 1.57 m that is 4.7 m fore-aft in a
10 m-wide channel, leaving almost no lateral room. A compressed domain is likely
necessary; justify it explicitly rather than silently shrinking it.

### 5.3 Encounter classifier

**One module, two consumers.** The classifier feeding the observation must be the exact
same function feeding the reward penalty in 02. If they diverge even at sector
boundaries, the agent is penalised for a role it was not shown. Apply hysteresis once,
inside the module.

Baseline thresholds (Waltz & Okhrin Table 1, after Xu et al. 2020), where
α = relative bearing OS→TS and CT = heading intersection angle:

| Class | Requirement |
|---|---|
| Head-on | α ∈ [0°,5°] ∪ [355°,360°); CT ∈ [175°,185°] |
| Starboard crossing (give-way) | α ∈ [5°,112.5°]; CT ∈ [185°,292.5°] |
| Port crossing (stand-on) | α ∈ [247.5°,355°]; CT ∈ [67.5°,175°] |
| Overtaking | α_TS→OS ∈ [112.5°,247.5°]; CT ∈ [0°,67.5°] ∪ [292.5°,360°); U_OS,R > U_TS |
| None | otherwise |

**Two required modifications:**

1. Their head-on band is ±5°, which is narrow relative to common practice (±6–10°).
   Widen and justify.
2. They have four classes; you need a fifth — **being overtaken** — because Rule 17
   stand-on behaviour is the strongest novelty angle and their formulation cannot
   express it. They assume linear deterministic targets and focus only on cases where
   the own ship is give-way.

---

## 6. Observation specification

Dict observation, five branches, multi-input policy.

| Branch | Contents | Dim |
|---|---|---|
| `lidar` | `c_t` sector closeness, **obstacles only** | 27 |
| `boundary` | virtual range scan, normalised | 7 |
| `ego` | u, v, r | 3 |
| `path` | e_y, χ̃, χ̃_LA | 3 |
| `targets` | 3 slots × 16 + 3 mask bits | 51 |
| | **Total** | **91** |

### 6.1 Per-slot target features (16)

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
| Encounter class | one-hot (6): none, head-on, crossing give-way, crossing stand-on, overtaking, being overtaken |

Angles as sin/cos to avoid wraparound discontinuity.

### 6.2 Slot management

- **N_max = 3** in simulation; 1 in field deployment.
- **Sort by CRI**, ascending. CRI is continuous and bounded in [0,1], so rank swaps
  occur where two targets genuinely have comparable risk, rather than at the arbitrary
  discontinuity where TCPA values cross. This is what Waltz & Okhrin do and it is
  strictly better than sorting by TCPA.
- **Track-ID persistence:** assign a slot on first acquisition, hold it until track
  loss. Re-sorting every step creates observation jumps when two targets swap rank —
  bad for SAC, because the replay buffer then holds transitions where `s` and `s'`
  differ with no action explaining it. With track IDs, discontinuities coincide with
  real events.
- **Valid mask:** binary, one bit per slot. Zero-padding alone is dangerous because
  zero is a legitimate value for bearing and relative speed, so a zero-padded empty
  slot looks like a target sitting on top of you on a matching course. Mask before
  pooling; set attention logits to −∞ if attention is ever used.
- **Masked-slot coverage:** field runs have one target and two masked slots. Ensure the
  training distribution contains plenty of 0- and 1-target episodes so that
  configuration is not out of distribution at deployment.

### 6.3 Architecture

Shared per-slot encoder φ (weights shared across slots), with aggregation as a config
flag:

- `aggregate = concat` → fixed-slot architecture **(headline)**
- `aggregate = sum` → DeepSets, permutation-invariant **(ablation)**
- `aggregate = attention` → only if 3-target results show ordering sensitivity

Rationale for concatenation as headline: N_max = 3 makes permutation invariance nearly
free of benefit; the measured advantages of attention in the literature come from
high-density regimes (dozens of aircraft, eight-ship encounters); and the novelty budget
is already committed to narrow-waterway COLREGs and Rule 17. Shipping the shared encoder
now makes the DeepSets comparison a config flag rather than a second architecture claim
to defend.

Weight sharing also softens the ordering problem on its own, since every slot learns the
same feature semantics.

**Pre-empt the scaling question** ("does this work beyond 3 targets?") by scope —
restricted waterway, single target in deployment — plus the pooling flag as evidence.
Do not defend the slot limit on principle.

**SB3 note.** Masking requires a custom features extractor; the default `MlpPolicy` will
not do it. With sum-pooling, divide by the number of *valid* targets, not N_max.

---

## 7. Literature grounding

| Source | Use |
|---|---|
| Waltz & Okhrin (2023) *Neural Networks* 165:634–653 | CPA/CRI formulation (§3.3), target feature vector and encounter table (§4.3), spatial-temporal recurrent architecture. **Read in full before finalising the observation vector.** |
| Everett, Chen & How (2018, 2021) | Origin of RNN-over-agents for variable neighbour counts |
| TU Delft (2025) *Eng. Appl. AI* — attention vs LSTM for separation management | Direct head-to-head under SAC. Additive attention best; LSTM order-sensitive; attention sequence-independent. Gaps are modest and density-driven. |
| Zaheer et al. (2017) Deep Sets | ρ(Σ φ(xᵢ)) permutation-invariance result |

**Two gaps between Waltz & Okhrin's setup and this one — foreground them as
contribution, do not hide them:**

1. They assume target ships move linearly and deterministically, so their work covers
   only give-way cases. That is exactly the Rule 17 hole.
2. Their perception is AIS — target course and speed arrive for free. Here they must be
   estimated from 2D LiDAR with drifting scan-matching odometry. That estimation step is
   a genuine difference, and its noise belongs in the domain randomisation.

---

## 8. Open items

- **O5** — physical barrier at basin edge vs software gating (check before building)
- Verify 720 beams against field logs; characterise aft occlusion and scan distortion
- Re-derive all CRI constants and the ship domain in ship lengths
- Widen the head-on band and justify
- Define the "being overtaken" class thresholds (not in the source table)
- Decide partial-observability handling: frame stacking vs recurrence, given that the
  explicit tracker already carries some memory. Occlusion of targets behind static
  obstacles in a channel argues for *some* memory; quantify before adding it.
