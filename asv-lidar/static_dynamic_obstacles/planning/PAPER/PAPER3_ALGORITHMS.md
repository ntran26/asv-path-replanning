# Paper 3 algorithm boxes (decision 2026-10-19, skeleton S17)

Pseudocode drafts for the algorithm boxes of Paper 3, written from the frozen baseline-v3 code
(the configuration every reported run used). Each box lists where it goes, its inputs and
outputs, the steps, its parameters and the source files. Symbols are the draft's and
`FORMULATION_EQUATIONS.md`'s (d_req, DCPA, TCPA, σ, ρ, ρ_pk, κ_eng, κ_rel, T_eng, N_clr, ψ_e, u_e,
t_e, c_wall, R_side, A, S, S_coast, D_stop, d_clr, B, LOA, U, T_0, c, 𝒞, w_s, c_k, d, θ; and, in
the comparator boxes, α, φ, r, v, e_y, U_nom, a_δ, a_n); every
other quantity is named in words, so no box introduces a symbol that already means something in
Sections 1–2 (k is the sector index, σ the required turn sense, a the action, **p** a position,
w with a subscript a reward weight).

| Box | Content | Section |
|---|---|---|
| Algorithm 1 | Encounter assessment: classification, engagement and latch, admissibility and stopping tests | 2.2.4 (end) |
| Algorithm 2 | Feasibility sector pooling (Paper 2, Algorithm 1, with the Paper 3 changes) | 2.3.1 |
| Algorithm 3 | Target detection and tracking | 2.3.1 |
| Algorithm 4 | Episode generation (one generator for the training, validation and test sets) | 2.3.5 |
| Algorithm A1 | Encounter-specific VO comparator (after Thyri and Breivik 2022; decision S18), drafted 2026-10-25 from comparators v2, `src/classical/encounter_vo.py` | Appendix, referenced from 3.3 (shared inputs: one paragraph ahead of it) |
| Algorithm A2 | LOS-DWA comparator (after Fox et al. 1997; no COLREGs) | Appendix, referenced from 3.3 |
| Algorithm A3 | Runtime safety layer: predictive filter and permanent emergency stop (decided 2026-10-25) | Appendix, referenced from 2.3.4 and 4.5 |

Not boxed: the test-set fairness check (oracle feasibility) stays a paragraph of the evaluation
design with its numbers (up to 574 manoeuvres per episode, 0.2 m clearance, 67 of 1,000 episodes
replaced); the reward (Eqs. 13–24), the curriculum (stage table), the training and
checkpoint-selection loop (one paragraph) and the seed ranges and set sizes (table).

---

## Algorithm 1. Encounter assessment (one decision step, per moving track)

**Placement.** End of Section 2.2.4; it gathers Eqs. 6–12 of Sections 2.2.2–2.2.4 into the order
in which they run. The track comes from Algorithm 3 (forward reference).

**Require:** estimated own-ship position **p**_OS, heading ψ, velocity **v**_OS and surge u; the
reference path's course ψ_path at the own ship's arc length s, and its present cross-track offset; the
map boundary polygon; a confirmed moving track (**p**_TS, **v**_TS, course χ_TS); the track's
latch (state, latched class, side, σ, ψ_e, u_e, t_e, clearing count, ρ_pk), initially idle.

**Ensure:** encounter class (one-hot), required turn sense σ, engagement state, turn admissible
A, slowdown clears S_coast, stop clears S, lateral deficit Δy_req, action required a_req,
proximity gate ρ and its peak ρ_pk, in extremis, and the heading and speed changes since
engagement (the context branch of the observation; the reward gates). The runtime safety layer
(Algorithm A3) does not read them: its filter and stop test the predicted geometry.

```
 1: 𝒞* ← CLASSIFY(p_OS, ψ_path, |v_OS|, p_TS, χ_TS, |v_TS|)              ▷ Eq. 6, bands widened 3° around the held class
 2: if 𝒞* differs from the held class for 2 consecutive steps then held class ← 𝒞*
 3: (DCPA, TCPA, range) ← CPA(p_OS, v_OS, p_TS, v_TS)
 4: Δy_req ← max(0, d_req − DCPA);  a_req ← clip(Δy_req / d_req, 0, 1)       ▷ Eq. 8
 5: run ← max(TCPA, 0) · max(u, 0)                                           ▷ along-path passage still to cover before the CPA
 6: for side ∈ {starboard, port} do                                          ▷ Eq. 10
 7:     d_side ← smallest beam-on map distance to the boundary over the arc lengths s to s + run,
                 measured from the path displaced by the own ship's present cross-track offset
 8:     R_side ← d_side − B/2 − c_wall
 9:     A_side ← 1 if R_side − Δy_req ≥ +0.15 m; 0 if ≤ −0.15 m; unchanged otherwise
                 (first value: R_side ≥ Δy_req)                              ▷ Eq. 11
10: end for
11: D_stop ← smallest distance between the target on its constant-velocity track and the own ship
             braking full astern along its heading, then lying stopped;  S ← [D_stop ≥ d_clr]   ▷ Eq. 12
12: repeat line 11 with the coasting profile (zero RPM, no reverse):  S_coast ← [D_stop ≥ d_clr]
13: if state = idle then
14:     if held class ≠ none and 0 < TCPA ≤ T_eng and DCPA < κ_eng d_req then              ▷ Eq. 7
15:         state ← engaged; latch class, crossing side, σ ← SENSE(class, side), ψ_e ← ψ, u_e ← u, t_e ← t
16: else if state = engaged then
17:     if TCPA < 0 or DCPA > κ_rel d_req then state ← clearing; clearing count ← 0
18: else if state = clearing then
19:     if not (TCPA < 0 or DCPA > κ_rel d_req) then state ← engaged                 ▷ the original latch is kept
20:     else if TCPA < 0 and range > d_req then
21:         clearing count ← clearing count + 1; if clearing count ≥ N_clr then state ← idle (release)
22:     else clearing count ← 0
23: end if
24: if state ∈ {engaged, clearing} then class, side, σ ← the latched values           ▷ no reclassification once engaged
25: else σ ← SENSE(held class, side)
26: A ← A_stbd if σ = +1;  A_port if σ = −1;  1 if σ = 0
27: ρ ← clip(1 − DCPA / (κ_eng d_req), 0, 1);  ρ_pk ← max(ρ_pk, ρ) while engaged             ▷ Eq. 9
28: in extremis ← [DCPA < d_req and 0 ≤ TCPA < 5 s]
```

SENSE(class, side): head-on +1 (starboard); crossing from starboard +1; crossing from port −1
(pass astern of the target, never ahead); overtaking −1 (pass to port of the target);
being overtaken 0 (hold course and speed); none 0.

| Parameter | Value |
|---|---|
| d_req = 2b | 2.5 m (b = 1.25 m, the domain's abeam semi-axis) |
| κ_eng, κ_rel | 1.5 (engage at DCPA < 3.75 m), 2.5 (clear at DCPA > 6.25 m) |
| T_eng; in-extremis horizon | 25 s; 5 s |
| N_clr | 6 steps (3 s) |
| Classification hold, bearing hysteresis | 2 steps (0.8 s rounds to 2 at 2 Hz), 3° |
| c_wall, admissibility band | 0.65 m, ±0.15 m |
| B (breadth) | 0.50 m |
| d_clr | LOA/2 + B/2 + 2 × 0.15 m + 0.35 m (safety distance) = 1.76 m (LOA 1.725 m) |

**Notes for the text.**
- Line 1 classifies against the path course ψ_path, not the instantaneous heading, so the own
  ship's own alteration cannot relabel the encounter (decision A19 in the design record).
- Line 24: the class, side and σ latched at engagement stand until release; a class switch while
  engaged does not re-latch (A20).
- The room of lines 6–10 comes from the map boundary only; perceived static obstacles do not
  bound it in baseline-v3 (the static-obstacle option was tested in v4.3 and not adopted).
- Lines 11–12: S (full astern) gates the reward's slowdown carve-out (R-2, with
  `R2_SLOWDOWN_TEST = "stop"`); S_coast is the observation's
  "slowdown clears" flag. The stop test reads the centroid track (the hull-fit view is off).
- Sources: `src/colregs/context.py` (`ContextManager.update`, `_attach_admissibility`,
  `_hysteretic`, `_advance_state`, `_attach_gates`), `src/encounter.py` (classifier and hold),
  `src/colregs/geometry.py` (`channel_room`, `usable_room`, `lateral_deficit`), `src/stopping.py`
  (`dcpa_over_stop`, modes "latch" and "coast"), `src/cpa_cri.py` (`cpa`).

---

## Algorithm 2. Feasibility sector pooling

**Placement.** Section 2.3.1, after the gating sentence. Cite Paper 2 (its Algorithm 1) and Meyer
et al. (2020a); state the changes listed below the box.

**Require:** the gated scan: beam ranges d_1, …, d_720 at 0.5° spacing (returns beyond the
boundary set to the maximum range by the map gate); the sector edges; the safe width w_s; the beam
spacing θ; the maximum range (16 m).

**Ensure:** the pooled distance of every sector k = 1, …, 27 and its closeness c_k (the lidar
branch).

```
 1: for k = 1, …, 27 do
 2:     D ← the ranges of the beams whose bearing lies in [edge_k, edge_k+1), in angular order
 3:     if D is empty then pooled_k ← maximum range; continue
 4:     pooled_k ← max(D)                                  ▷ default: clear to the farthest beam
 5:     for each level ξ ∈ D, in ascending order, with ξ > 0 do
 6:         arc ← θ ξ;  opening ← arc/2;  open ← false     ▷ arc between neighbouring beams at range ξ
 7:         for each d_j ∈ D, in angular order do
 8:             if d_j > ξ then opening ← opening + arc
 9:             else opening ← opening + arc/2             ▷ a blocking beam adds half an arc …
10:             if opening > w_s then open ← true; break
11:             if d_j ≤ ξ then opening ← 0                ▷ … then closes the opening
12:         end for
13:         if not open then pooled_k ← ξ; break           ▷ no opening wider than w_s beyond ξ
14:     end for
15:     c_k ← 1 − min(pooled_k, maximum range) / maximum range
16: end for
```

| Parameter | Value |
|---|---|
| Swath, sectors | ±135°, 27 sectors: port outer 2 × 22.5°, port mid 4 × 11.25°, bow 15 × 6°, starboard mid 4 × 11.25°, starboard outer 2 × 22.5° |
| Beams, θ | 720 per revolution, θ = 0.5° |
| w_s | B + 2 × 0.15 m = 0.80 m (the inflated hull, as in Paper 2) |
| Maximum range | 16 m (simulated sensor; the closeness normaliser) |

**Changes from Paper 2 (state below the box).**
1. The input is the map-gated scan, so the pooled sectors contain every gated return: static
   obstacles **and the target's hull**. A target therefore reaches the policy twice, as an
   obstacle at its range here and as a track with its velocity in the target branch.
2. Sectors are non-uniform (fine at the bow, coarse abeam), so beams are assigned to sectors by
   bearing against explicit edges rather than by index; Paper 2 used 25 uniform sectors.
3. θ is the sensor's true 0.5° beam spacing; Paper 2 used the swath divided by the beam count
   (1.205°). The pooling steps are unchanged, but pooled values are not comparable across the
   two papers.
- Sources: `src/lidar_pooling.py` (`feasibility_pool`, `pool_to_sectors`, `sector_assignment`),
  `src/constants.py` (`POOL_BANDS`, `LIDAR_SECTORS`), `src/asv_lidar.py` (`FEASIBILITY_SAFE_WIDTH`),
  `src/env.py` (`_perceive`: ray cast → gate → pool → segment → track).

---

## Algorithm 3. Target detection and tracking (one 2 Hz step)

**Placement.** Section 2.3.1, after the pipeline paragraph (replaces the prose list of steps or
sits beside it). Cite Borges and Aldon (2004) for the segmentation, Kalman (1960) and
Bar-Shalom, Li and Kirubarajan (2001) for the filter.

**Require:** the gated scan and the raw scan (720 beams each); the estimated pose (sensor origin
offset along the heading); the track set 𝒯 with each track's state [x, y, v_x, v_y], covariance,
hit and miss counts, motion flag and a window of its last observations.

**Ensure:** the confirmed moving tracks, which feed the target branch and Algorithm 1.

```
 1: if the pose estimate is stale then carry the elapsed time to the next update; return     ▷ deployment timing
 2: points ← world positions of the gated returns short of the maximum range, lifted at the sensor origin
 3: segment the points in angular order: consecutive returns join one cluster when their beams are
    adjacent and they lie within 0.35 m plus the beam arc θ × range of each other; merge the
    clusters on either side of the ±180° seam
 4: detections ← clusters of at least 4 points; each detection's position is its centroid
 5: for each track do PREDICT(track, elapsed time)          ▷ constant velocity, piecewise-constant white acceleration
 6: matches ← greedy global nearest neighbour: take track–detection pairs in ascending distance, skip a pair
    whose track or detection is already used, stop at the first pair beyond the 0.70 m gate
 7: for each matched (track, detection) do KALMAN-UPDATE(track, centroid); hits ← hits + 1; misses ← 0
 8: for each unmatched track do misses ← misses + 1
 9: for each unmatched detection do start a track at its centroid with zero velocity
10: record ← this revolution's free-space record from the raw scan (where every ray ended)
11: for each matched (track, detection) do                          ▷ free-space motion test
12:     (record_ref, points_ref) ← the track's oldest observation in its 2 s window
13:     appear ← the detection's points that record_ref saw through, with no return of record_ref near them
14:     vacate ← the points_ref that record saw through, with no return of record near them
15:     moved ← [appear + vacate ≥ 3 and ≥ 10 % of the points compared]
16:     if moved then moving run +1, quiet run ← 0 else quiet run +1, moving run ← 0
17:     if not moving and moving run ≥ 2 then moving ← true          ▷ 1 s of motion evidence
18:     if moving and quiet run ≥ 4 then moving ← false              ▷ 2 s without
19:     append (record, the detection's points) to the track's window
20: end for
21: remove tracks with more than 3 consecutive misses (1.5 s)
22: return the tracks with at least 2 hits that are moving
```

| Parameter | Value |
|---|---|
| Update period | 0.5 s (2 Hz) |
| Segmentation | break threshold 0.35 m plus the beam arc at that range; at least 4 points |
| Association gate | 0.70 m (2.5 × cruise speed × update period) |
| Process noise | 0.10 m/s² standard deviation (piecewise-constant white acceleration) |
| Measurement noise | 0.05 m standard deviation (position) |
| Initial velocity variance | 0.50 (m/s)² |
| Confirmation, deletion | 2 hits; dropped after more than 3 consecutive misses (1.5 s) |
| Motion window | 4 updates (2 s) |
| "Saw through" tolerance, "near" radius | 0.25 m (the larger of 0.25 m and 5√2 × the 0.03 m pose noise), 0.30 m |
| Promote, demote | after 2 updates with motion (1 s), after 4 without (2 s) |

**Notes for the text.**
- Line 10: the free-space record uses the raw scan, facility walls included, because a ray that
  ends on a wall has passed through everything in front of it; a no-return certifies nothing.
  "Saw through" requires the beam toward the point to have returned from at least the tolerance
  beyond it, its neighbours not to fall short of it, and the point to lie outside the 1 m dead
  zone.
- Lines 17–18: the hysteresis is biased toward under-detection, because calling a static panel a
  vessel has COLREGs consequences.
- The measurement is the cluster centroid (the hull-fit measurement exists but is off in
  baseline-v3). Detection dropout and velocity-estimate noise are zero in training.
- Static tracks are not used by the policy; the pooled sectors (Algorithm 2) carry all gated
  returns regardless of the motion test.
- Sources: `src/tracking.py` (`segment_scan`, `Tracker.update`, `_associate`, `Track.predict`,
  `Track.update`, `ScanFrame.passes_through`, `motion_evidence`, `Track.apply_evidence`),
  `src/constants.py` (CLUSTER_*, TRACK_*, KF_*, MOTION_*, DYNAMIC_* values), `src/env.py`
  (`_perceive`).

---

## Algorithm 4. Episode generation

**Placement.** Section 2.3.5 (curriculum), referenced from 2.1.4 where decoupled and coupled
scenarios are defined. The same generator produces the training episodes (on the fly, at the
current stage, from the training seed range) and the validation and test sets (from their own
disjoint seed ranges); the set sizes and composition go in a table beside the box.

**Require:** a seed from the set's range; the stage's parameters (classes and their weights,
channel share and width range, obstacle counts, CPA guard fraction, coupled-layout share,
varying-speed share, space-time solvability flag); for the test set's reference layouts, a fixed
layout (a leg and three panels).

**Ensure:** a layout (basin leg or channel), the own ship's start, a target (or none) with its
initial state and behaviour, and the static panels.

```
 1: if a fixed layout is given then return COUPLED-LAYOUT(); else if the stage has coupled layouts, then with
    the stage's coupled share return COUPLED-LAYOUT()   ▷ lines 22–27
 2: draw the class 𝒞 from the stage's classes and weights
 3: geometry ← a channel with the stage's channel share if 𝒞 ∈ {head-on, crossing, overtaking}, else the basin;
    a channel's nominal width is drawn from the stage range and varies along it (ratio of widest to
    narrowest ~ U[1.0, 1.8])
 4: own ship at the start of the reference path, on its heading, at U_OS = U_nom
 5: if 𝒞 = no target then go to line 15
 6: if 𝒞 = crossing then label it unescapable with probability 0.20                  ▷ once per episode
 7: repeat (at most 200 attempts; on cap-out redraw the seed, at most 20 times)
 8:     if 𝒞 = null then place the target: range U[8, 15] m, bearing 20–160° to either side, course within
        ±10° of the own ship's, speed ratio 0.85–1.15; accept only if its CPA is ≥ 4 m or after the episode
 9:     else draw the heading-crossing angle c, the speed ratio U_TS/U_OS, the time to CPA T_0 and the miss
        distance d_0 from the class bands (being overtaken: d_0 at least the contact-free miss distance
        plus 0.35 m) and a side ± at random; solve backward, with h the unit heading vectors:
            ψ_TS = ψ_OS + c;  v_rel = U_OS h_OS − U_TS h_TS;  n̂ = ± perp(v_rel)/|v_rel|
            p_TS(T_0) = p_OS(0) + U_OS T_0 h_OS + d_0 n̂;  p_TS(0) = p_TS(T_0) − U_TS T_0 h_TS
10:     reject if the spawn range lies outside the class band
11:     reject if CLASSIFY(the realized geometry) ≠ 𝒞                    ▷ the classifier of Algorithm 1
12:     reject if the target's track leaves the water
13:     if 𝒞 = crossing: reject unless the scripted escape check (hold 1.5 s, then coast to a stop or make a
        60° alteration in the required direction) agrees with the label
14: until accepted
15: draw the stage's number of 1.0 m panels along 25–70 % of the path with lateral offsets; drop any panel within
    the CPA guard (a fraction of T_0 U_OS of own-ship travel either side of the CPA) or on the target's spawn
16: A* on a 0.25 m grid, walls inflated 0.40 m and panels 0.45 m: a route must exist and be ≤ 2.25 × the leg;
    otherwise redraw the panels (up to 20 times), then thin them
17: if the stage requires it and 𝒞 ≠ no target and not SPACE-TIME-SOLVABLE() then redraw the episode (up to 30)
18: return the episode
19:
20: SPACE-TIME-SOLVABLE(): forward reachability on the A* grid; the own ship is a point moving up to 1.1 m/s in
    any direction or waiting; every 0.5 s the reachable set grows by 0.55 m and every cell within 1.3 m of the
    target's position at that time is removed; solvable if a cell within 0.5 m of the goal is reached in time
21:
22: COUPLED-LAYOUT(): the fixed layout's leg if one is given; otherwise a basin leg with start and goal
    x ∈ [2, 8] m (slant ≤ 14.1°)
23:     the fixed layout's three panels if one is given; otherwise three 1 m panels in one motif along 28–85 %
        of the leg, ≥ 0.9 m from the walls and ≥ 0.4 m apart: gate + on-path (0.40), slalom (0.30),
        side + on-path (0.30)
24:     the encounter by lines 2–14 on this leg, with the panels fixed and no CPA guard; the target starts inside
        the basin, keeps 0.20–1.10 m/s, stops short of the wall, and its straight track clears every panel
        by 0.30 m
25:     stage 7: with probability 0.30 the target changes speed once (one step, bounded acceleration)
26:     if the panels were drawn (no fixed layout) then reject unless SPACE-TIME-SOLVABLE()
27:     return the episode
```

Fixed layouts (added 2026-10-24): the 108 reference-layout episodes of test set v4 are drawn by the
same generator (`ScenarioGenerator.sample` at the stage-5 setting) with the layout's published leg
and three panels passed in as fixed (`src/paper2_set.py`, `_flags`; `tools/tiers/test_set.py`,
`_layout_episode`). They meet the field rules of line 24 (`FIELD_TARGET_SPEED_RANGE`, the stop
box, `TARGET_PANEL_CLEARANCE_M`); the generation does not filter them by the space-time check
(neither file calls it), and like every test episode they pass the test set's feasibility check
(Section 3.4.1).

CPA guard fraction: 0.4 in stages 1–5 (the decoupled scenarios), 0.2 in stage 6, none in stage 7.

| Class | c | U_TS/U_OS | T_0 (s) | d_0 (m) | Spawn range (m) |
|---|---|---|---|---|---|
| Head-on | 170–190° | 0.70–1.30 | 12.3–17.0 | U[0, 2.0] | 13.8–19 |
| Crossing | 67.5–175° or 185–292.5° (60 % from port in training) | 0.60–1.40 | 7.9–14.8 | U[0, 2.5] | 5–10 |
| Overtaking | −10–10° | 0.40–0.55 | 19.7–31.5 | U[0, 2.0] | 6 to min(12, 16 (1 − U_TS/U_OS)) |
| Being overtaken | −10–10° | 1.50–2.20 | 9.9–15.8 | U[f, max(2.0, f + 0.5)], f = contact-free + 0.35 | 4–6 astern (own ship starts 6.9 m along the leg) |
| Null | −10–10° | 0.85–1.15 | — (placed) | — | 8–15 |

**Notes for the text.**
- Sampling d_0 explicitly is what gives the policy encounters that need no alteration; solving
  backward without it puts the target on the own ship's track in every episode.
- The channel is drawn once per episode (line 3), so a rejected draw rejects the target's
  kinematics, not the width; resampling the channel per attempt would bias the width
  distribution.
- Training shares of the classes: head-on 0.20, crossing 0.22, overtaking 0.16, being overtaken
  0.14, null 0.11, no target 0.17 (stage limits apply); 15 % of training episodes start slow
  (half from rest, half up to half cruise).
- Training targets hold constant velocity and never give way. Reactive and non-compliant targets
  are evaluation behaviours set on the test-set episodes, not generator branches.
- Sources: `src/scenario.py` (`ScenarioGenerator.sample`, `_attempt`, `_backward_solve`,
  `_place_null`, `crossing_escape_feasible`, `contact_free_dcpa`, `seed_for`), `src/env.py`
  (`_load_generated`, `_obstacles_clear_of_encounter`), `src/feasibility.py` (A* route),
  `src/feasibility_st.py` (`solvable`), `src/field_training.py` (coupled layouts),
  `src/constants.py` (`CURRICULUM_STAGES`, class bands).

---

## Comparator inputs and shared steps (Algorithms A1 and A2)

**Placement.** Appendix, one paragraph ahead of Algorithm A1 (not boxed); Section 3.3 keeps one
sentence: both comparators see what the policy sees, and nothing else. Cite Fossen, Breivik and
Skjetne (2003) for LOS guidance.

Each 0.5 s decision builds one snapshot from onboard quantities; simulator truth never enters
action selection. The encounter context of Algorithm 1 is computed in every episode; the
encounter-specific VO reads it (decision S18), and LOS-DWA does not.

| Input | Content |
|---|---|
| Own ship | estimated pose (**p**_OS, ψ); measured surge u, sway v, yaw rate r; sideslip = clip(atan2(v, max(u, 0.15 m/s)), −20°, 20°) |
| Path | the reference path's course ψ_path at the nearest station; cross-track offset e_y (positive to starboard) |
| Map | the boundary polygon (known exactly) |
| Static points | gated returns short of the 16 m maximum range, placed from the sensor origin (LOA/2 ahead of the centre); returns within 1.8 m of a track removed as the target's own; the last 4 scans (2 s) kept, so the aft mask and the dead zone do not drop a passed obstacle; remembered returns within 2.0 m of a track's present position removed; thinned to one per 0.25 m cell |
| Tracks | the confirmed moving tracks of Algorithm 3: **p**_TS, **v**_TS and a hull heading (the hull-fit axis when the cluster gives a reliable fit, else the direction of **v**_TS when its speed exceeds 0.05 m/s, else 0°) |

Shared steps:
- **LOS course** = ψ_path + atan2(−e_y, 2.5 m), a ground course (look-ahead 2.5 m, about 1.6 hull lengths
  between perpendiculars).
- **Clearances** (hull level). Own hull: the LOA × B rectangle inflated by the 0.15 m hull margin,
  the collision hull of the environment. A track's hull: the same rectangle and margin, at the
  track's hull heading, moved along its constant-velocity track. Hull to hull: the
  separating-axis gap (a lower bound on the distance). Hull to point: the distance from the
  rectangle to the nearest static point. Hull to boundary: the distance from the centre to each
  polygon edge less the rectangle's extent along that edge's normal.
- **YAW-RATE-LOOP**(demand, °/s): a_δ = clip(demand/10.4 + 0.08 (demand − r), −1, 1).
- **SPEED-LOOP**(speed): rpm = speed/(0.093 m/s per rpm) + 6 (speed − u), 0 rpm for a zero
  demand, clipped to [0, 12] (propulsion stage 4); a_n = clip((rpm − 6)/6, −1, 1).
- The action a = [a_δ, a_n] has the policy's limits (rudder ±40°, 0–12 rpm, no astern).

---

## Algorithm A1. Encounter-specific VO comparator (one 2 Hz decision; comparators v2)

**Placement.** Appendix F, after the shared inputs; referenced from Section 3.3. Cite Thyri and
Breivik (2022) for the method and Fiorini and Shiller (1998) for velocity obstacles. The box is an
interpretation written for this study, not the authors' implementation (Section 6.2). Drafted
2026-10-25 from comparators v2 (decision S18, A35).

**Require:** the snapshot (shared inputs above); for each track, its encounter context from
Algorithm 1 (latched class 𝒞, required turn sense σ, engaged, in extremis); the course chosen at the
previous decision (none at the start).

**Ensure:** the action a = [a_δ, a_n]; for the log, whether the passing-side constraints were
released and whether the own ship stood on.

```
 1: LOS course ← ψ_path + atan2(−e_y, 2.5 m)                                  ▷ a ground course
 2: candidates ← courses ψ + {−90°, −85°, …, +90°} and the LOS course, each at speeds
                {1, 0.75, 0.5, 0.25, 0} × U_nom                                     ▷ 38 × 5 = 190
 3: for each candidate (course, speed) do
 4:     sweep the own ship from p_OS: hold ψ for 1 s, then turn toward the course at 6°/s; the speed
        moves from u toward the candidate's with time constant 60 s (slowing) or 20 s (speeding up);
        samples every 0.25 s to 20 s
 5:     first contact ← earliest sample at which the gap to any track's hull, moved at constant
        velocity, is < 0.20 m (∞ if none)
 6:     first static ← earliest sample within 10 s at which the clearance to a static point is
        < 0.15 m or to the boundary is < 0.05 m (∞ if none)
 7:     static penalty ← max(0, 0.60 m − smallest static clearance within 10 s) / 0.45 m
        (the boundary clearance enters raised by 0.10 m)
 8:     domain penalty ← the largest, over the tracks and the samples, of max(0, 1 − q), where q is
        the target's position in the swept own-ship frame measured in the own ship's domain scaled
        for the track's class 𝒞 (q = 1 on the domain's edge, 0 at its centre)
 9:     wrong side ← false
10:     for each track whose class 𝒞 is head-on, crossing or overtaking do
11:         if the track is engaged (Algorithm 1) and the candidate velocity, held from now,
            closes on it then                                     ▷ judged on the encounter (v2)
12:             CPA of the candidate velocity against the track's, TCPA clipped to [0, 20 s]
13:             head-on: wrong side ← true if the target lies to starboard at the CPA
                crossing: wrong side ← true if the own ship lies ahead of the target at the CPA
                          (no test if the target moves slower than 0.05 m/s)
                overtaking: wrong side ← true if the target lies on the side σ forbids
                          (σ = −1, a port pass, also when σ = 0: the target must lie to starboard)
14:     end for
15:     cost ← 1.0 |course − LOS course|/π + 0.5 max(0, U_nom − speed)/U_nom + 3.0 domain penalty
               + 0.8 |course − previous course|/π + 1.0 static penalty
16: end for
17: clear ← first contact = ∞ and first static = ∞
18: preferred ← the candidate nearest to (LOS course, U_nom)
19: if a track's class is being overtaken and min(first contact, first static) of preferred > 12 s
       and preferred is not on the wrong side then
20:     choice ← preferred                                      ▷ Rule 17(a)(i): hold course and speed
21: else
22:     allowed ← clear and not wrong side
23:     if a track is in extremis or no candidate is allowed then allowed ← clear
                                              ▷ release the passing side, never the clearance
24:     if some candidate is allowed then choice ← the allowed candidate of least cost
25:     else choice ← the candidate whose min(first contact, first static) is latest; ties by cost
26: previous course ← the choice's course
27: heading error ← wrap(the choice's course − sideslip − ψ)
28: a_δ ← YAW-RATE-LOOP(clip(0.3 s⁻¹ × heading error, −6°/s, 6°/s));  a_n ← SPEED-LOOP(the choice's speed)
29: return a
```

| Parameter | Value |
|---|---|
| Candidate courses | ψ + (−90° to +90° in 5° steps), plus the LOS course (38) |
| Candidate speeds | {1, 0.75, 0.5, 0.25, 0} × U_nom, U_nom = 0.558 m/s (6 rpm); 0 = zero rpm, coasting |
| Horizons, sample step | 20 s for tracks; 10 s for static points and the boundary; 0.25 s |
| Manoeuvre model of the sweep | 1 s before the turn starts, then 6°/s; speed time constants 60 s down, 20 s up |
| Hard gaps | 0.20 m between the inflated hulls; 0.15 m to a static point; 0.05 m to the boundary |
| Ship domain | the own ship's, 3.14 m ahead, 1.57 m astern, 1.25 m abeam; scaled (ahead, astern, abeam) by (1.0, 1.0, 1.2) head-on, (1.5, 1.0, 1.0) crossing, (1.0, 1.0, 1.2) overtaking, (1, 1, 1) otherwise |
| Passing side | head-on: target to port at the CPA; crossing: own ship astern of the target at the CPA; overtaking: the side σ names; binds every closing candidate while the encounter is engaged (Algorithm 1: engaged when 0 < TCPA ≤ 25 s and DCPA < 3.75 m, held until TCPA < 0 or DCPA > 6.25 m); released in extremis or when no clear candidate keeps it |
| Stand-on | being overtaken: the preferred candidate, while it stays clear for more than 12 s and keeps any passing side |
| Cost weights | course 1.0, speed 0.5, domain 3.0, course change 0.8, static 1.0 (course terms per π rad, speed per U_nom) |
| Static soft margin | penalty 1 at 0.15 m, 0 at 0.60 m (boundary raised by 0.10 m) |
| Course autopilot | yaw-rate demand 0.3 s⁻¹ × heading error, capped at ±6°/s, the sweep's turn rate |
| Tuning (comparators v2) | coordinate descent on the validation set's decoupled part, in the build the learners were selected on (search on 60 episodes, 10 per type; winner rescored on all 120; score goal − 2 × collision): horizon {12, 20, 30} s, hard gap {0.20, 0.35, 0.50} m, stand-on release {8, 12, 18} s, course-change weight {0, 0.3, 0.8}, domain weight {1.5, 3, 6}. Only the course-change weight moved, 0.3 → 0.8 (all 120: +0.350 → +0.475, success 0.783 → 0.825). Pinned in `configs/comparators_v2.json` (digest `617782d1483005d7`) |

**Notes for the text.**
- **Same rule set as the formulation.** The class, σ, engagement and in-extremis flag are those of
  Algorithm 1, so the comparator follows the narrow-channel convention (a crossing target from
  either side is passed astern) and Rule 9's overtaking side; it has no room or stopping test of
  its own (Eqs. 10–12): the boundary acts through lines 6–7.
- **Side rule judged on the encounter (A35, v2).** While the encounter is engaged, every candidate
  that closes on the target must keep the side; a candidate that does not close (slowing behind an
  overtaken vessel, opening the range) makes no pass and is free. Comparators v1 judged it per
  candidate (closing and that candidate's own DCPA < 2.5 m), so a candidate that opened the pass
  by itself was exempt: on a head-on 0.3 m to starboard of the reciprocal at 10 m, v1 turns to port and v2 to
  starboard (`tests/test_classical.py`, `test_vo_side_rule_is_judged_on_the_encounter`). On the validation set, the
  per-candidate rule with the v1 settings scored +0.475 and the encounter rule with them +0.350;
  re-tuning the course-change weight recovered the difference (v2: +0.475).
- **Adaptations for this problem**, as for Algorithm A2: the tracks of Algorithm 3 replace the
  published method's vessel states, a track's hull is the own ship's rectangle at the track's hull
  heading, static points and the map boundary are velocity obstacles over 10 s with a soft margin,
  and candidates are swept along a turn model (line 4) because this hull needs about 5 s to build
  its turn rate; the side rule uses the candidate velocity itself, which states intent, not the
  transient.
- **Test set v4 (2026-10-25):** success 0.675 (decoupled 0.774, coupled 0.479); collisions with the
  target 0.116, obstacles 0.130, boundary 0.079 (`planning/MASTER_ARTIFACT_RESULTS.md`).
- Sources: `src/classical/encounter_vo.py` (`select_velocity`, `_wrong_side`, `sweep`,
  `EncounterVOController.action`), `src/classical/common.py` (`Perception.snapshot`,
  `los_heading`, `course_rudder`, `yaw_rate_rudder`, `speed_rpm`, `point_clearance`,
  `boundary_clearance`), `src/reference_controller.py` (`hull_separation`),
  `src/constant_temp.py` (`CLASSICAL_VO_*`), `configs/comparators_v2.json`,
  `results/classical_tuning/encounter_vo/` (`best.json`, `validation.txt`, `side_rule_check.txt`),
  `tools/diagnostics/tune_classical.py`.

---

## Superseded: COLREGs-aware velocity-obstacle comparator (one 2 Hz decision)

*Not a Paper 3 comparator (decision S18, 2026-10-20): Algorithm A1 becomes the
encounter-specific VO, drafted from comparators v2. This box is kept because the test set's
reactive target runs the same rule from its own side, and the shared inputs above apply to the
new A1 unchanged.*

**Placement.** Appendix, referenced from Section 3.3. Cite Kuwata et al. (2014) for the method
and Fiorini and Shiller (1998) for velocity obstacles.

**Require:** the snapshot (above); the course chosen at the previous decision (none at the
start).

**Ensure:** the action a = [a_δ, a_n]; for the log, whether the COLREGs constraint was released
and whether the own ship stood on.

```
 1: LOS course ← ψ_path + atan2(−e_y, 2.5 m)
 2: for each track do                                                    ▷ open-water roles, Rules 13–17
 3:     α ← bearing of the track from ψ;  φ ← bearing of the own ship from the track's hull heading
        (both wrapped to ±180°)
 4:     role ← being overtaken                   if |α| ≥ 112.5°
               else overtaking                   if |φ| ≥ 112.5° and u > |v_TS|
               else head-on                      if |α| ≤ 22.5° and |φ| ≤ 22.5°
               else crossing, give-way           if α > 0 (target to starboard)
               else crossing, stand-on
 5: end for
 6: candidates ← courses ψ + {−90°, −85°, …, +90°} and the LOS course, each at speeds
                {1, 0.75, 0.5, 0.25, 0} × U_nom                                  ▷ 38 × 5 = 190
 7: for each candidate (course, speed) do
 8:     sweep the own ship from p_OS: hold ψ for 1 s, then turn toward the course at 6°/s; the speed
        moves from u toward the candidate's with time constant 60 s (slowing) or 20 s (speeding up);
        samples every 0.25 s to 20 s
 9:     first contact ← earliest sample at which the gap to any track's hull is < 0.20 m (∞ if none)
10:     first static ← earliest sample within 10 s at which the clearance to a static point is
        < 0.15 m or to the boundary is < 0.05 m (∞ if none)
11:     static penalty ← max(0, 0.60 m − smallest static clearance within 10 s) / 0.45 m
        (the boundary clearance enters raised by 0.10 m)
12:     forbidden ← false
13:     for each track whose role is head-on, crossing give-way or overtaking do   ▷ Kuwata's constraint
14:         v_OS ← the candidate's velocity, held from now;  DCPA, TCPA by Eq. 3, TCPA capped at 20 s
15:         live ← TCPA > 0 and DCPA < d_req                                   ▷ judged per candidate
16:         if live and (p_TS − p_OS) × (v_OS − v_TS) ≥ 0 then forbidden ← true
                ▷ the relative velocity must lie to starboard of the line of sight
17:     end for
18:     cost ← 1.0 |course − LOS course|/π + 0.5 max(0, U_nom − speed)/U_nom
               + 1.0 static penalty + 0.8 |course − previous course|/π
19: end for
20: safe ← first contact = ∞ and first static = ∞
21: preferred ← the candidate (LOS course, U_nom)
22: if a track's role is crossing stand-on or being overtaken
       and min(first contact, first static) of preferred > 12 s then
23:     choice ← preferred                                            ▷ Rule 17(a): hold course and speed
24: else if some candidate is safe and not forbidden then choice ← the safe, allowed candidate of least cost
25: else if some candidate is safe then choice ← the safe candidate of least cost
                                                ▷ Kuwata's fallback: drop the rule, not the clearance
26: else choice ← the candidate whose min(first contact, first static) is latest; ties by cost
27: previous course ← the choice's course
28: heading error ← wrap(the choice's course − sideslip − ψ)
29: a_δ ← YAW-RATE-LOOP(clip(0.3 s⁻¹ × heading error, −6°/s, 6°/s));  a_n ← SPEED-LOOP(the choice's speed)
30: return a
```

× is the scalar cross product x₁y₂ − y₁x₂ in the east–north frame; a negative value puts the
second vector to starboard of the first. "Alter to starboard and pass astern" in Kuwata's terms.

| Parameter | Value |
|---|---|
| Candidate courses | ψ + (−90° to +90° in 5° steps), plus the LOS course (38) |
| Candidate speeds | {1, 0.75, 0.5, 0.25, 0} × U_nom, U_nom = 0.558 m/s (6 rpm); 0 = zero rpm, coasting |
| Horizons, sample step | 20 s for tracks; 10 s for static points and the boundary; 0.25 s |
| Manoeuvre model of the sweep | 1 s before the turn starts, 6°/s; speed time constants 60 s down, 20 s up |
| Hard gaps | 0.20 m between the inflated hulls; 0.15 m to a static point; 0.05 m to the boundary |
| Classification | head-on: \|α\| and \|φ\| ≤ 22.5°; overtaking sector: \|α\| or \|φ\| ≥ 112.5° (more than 22.5° abaft the beam) |
| Live encounter | closing and DCPA < d_req = 2.5 m, TCPA capped at the 20 s horizon |
| Stand-on release | the preferred candidate meets a hard gap within 12 s (targets) or 10 s (static) |
| Cost weights | course 1.0, speed 0.5, static 1.0, course change 0.8 (course terms per π rad, speed per U_nom) |
| Static soft margin | penalty 1 at 0.15 m, 0 at 0.60 m |
| Course autopilot | yaw-rate demand 0.3 s⁻¹ × heading error, capped at ±6°/s |
| Tuning | course-change weight 0.3 → 0.8 on the validation set (goal − 2 × collision: +0.525 → +0.575, confirmed on all 120 episodes); every other value as built; pinned in `configs/comparators_v1.json` (digest `0dd8eab26bbf954f`) |

**Notes for the text.**
- **Open-water roles, by design.** The comparator applies the open-water role table: a crossing
  target to port makes the own ship stand on (lines 4 and 22). It has no narrow-channel crossing
  convention (the formulation's own convention gives way to a crossing target from either side
  and passes astern of it, Algorithm 1, SENSE), no Rule 9 precedence, and no room or stopping
  test (Eqs. 10–12): the boundary acts only through the clearance tests of lines 10–11. Where the
  starboard alteration does not fit, the two conventions part, and that difference is the
  measurement the Rule 9 claim rests on.
- **Classification differs from Algorithm 1**: it uses the instantaneous heading ψ, not ψ_path; it
  is recomputed every decision with no hold, hysteresis or latch; its head-on sector is ±22.5°
  (Algorithm 1: α within ±10° and c within 170–190°); the overtaking test needs only u > |v_TS|
  (Algorithm 1: a 0.15 U_ref margin). There is no engagement gate: the constraint acts while the
  encounter is live. No σ is computed.
- **Adaptations for this problem.** (1) Perception: the tracks of Algorithm 3 replace the
  published method's vessel states, and a track's hull is taken to be the own ship's rectangle
  (1.725 m × 0.50 m) at the track's hull heading. (2) Static points and the map boundary are added
  as velocity obstacles over 10 s, with a soft margin, for the confined workspace; the same
  treatment as Algorithm A2. (3) Reachability: a candidate is swept along a turn model (line 8)
  instead of being taken at once, because this hull needs about 5 s to build its turn rate; the side rule (line 16) uses the candidate velocity itself, which states intent, not
  the transient. (4) Stand-on means following the path at U_nom (the preferred candidate), released
  when that velocity meets a hard gap within 12 s (Rule 17(a)(ii) and (b)); there is no separate
  in-extremis rule.
- **Side rule judged per candidate (decision A35: option 1, decided 2026-10-19; built after TQC seeds 1–2).** Lines 14–16 decide "live" from each
  candidate's own velocity, so a candidate that opens the pass beyond d_req by itself, for
  instance a quick turn to port in a head-on, is exempt. On a head-on 0.3 m off the reciprocal
  the frozen comparator picks 160° (port); judged once on the encounter, as Kuwata et al. apply
  it, it picks 200° (starboard). A35 was decided for the published form; the paper's VO comparator
  is the encounter-specific VO of Algorithm A1 (comparators v2), whose side rule is judged on the
  engaged encounter. This superseded box keeps the per-candidate form of `colregs_vo.py`.
- The reactive target of the robustness set runs the same selection from its own side, with
  course-only candidates, the side rule judged on the encounter and a 0.75 m hard gap.
- Sources: `src/classical/colregs_vo.py` (`classify`, `_colregs_ok`, `select_velocity`,
  `ColregsVOController.action`), `src/classical/encounter_vo.py` (`sweep`, the manoeuvre model it
  shares), `src/classical/common.py` (`Perception.snapshot`, `los_heading`, `course_rudder`,
  `yaw_rate_rudder`, `speed_rpm`, `point_clearance`, `boundary_clearance`),
  `src/reference_controller.py` (`hull_separation`), `src/constant_temp.py` (`CLASSICAL_KVO_*`;
  the sweep reads `CLASSICAL_VO_TURN_LAG_S`, `CLASSICAL_VO_TURN_RATE_DPS`,
  `CLASSICAL_VO_SPEED_TAU_*`), `configs/comparators_v1.json`.

---

## Algorithm A2. LOS-DWA comparator (one 2 Hz decision)

**Placement.** Appendix, referenced from Section 3.3. Cite Fox, Burgard and Thrun (1997) for the
dynamic window approach.

**Require:** the snapshot (above); the controller's copy of the rudder delay line and servo,
advanced with every issued a_δ; the heading PID's integral; the previous choice (set-point or
LOS-PID, and speed fraction), none at the start.

**Ensure:** the action a = [a_δ, a_n].

```
 1: if no track is within 9 m, no static point is within 9 m unless more than 1 m abaft the
    own-ship centre, and the hull is more than 1.0 m from the boundary then
 2:     previous choice ← none
 3:     return LOS-PID(U_nom)                                            ▷ path following alone
 4: candidates ← yaw-rate set-points {−9, −6.75, …, +9}°/s × speeds {1, 0.5, 0} × U_nom
                × hold times {3, 6} s, plus LOS-PID at U_nom and 0.5 U_nom          ▷ 54 + 2 = 56
 5: for each candidate do                                ▷ the dynamic window, through the vessel model
 6:     simulate 16 s closed loop on the identified model (RK4, 0.125 s) from the measured u, v, r, ψ,
        the servo state and the commands still in the 0.73 s delay line; at each 0.5 s decision
            a_n ← SPEED-LOOP(the candidate's speed)
            a_δ ← YAW-RATE-LOOP(set-point) within the hold time, else the heading PD (no integral)
                  on the LOS course at the predicted position, less the predicted sideslip
 7:     travel window ← the samples within the first 5.5 m travelled (the first sample at least)
 8:     static clearance ← min(clearance to static points, boundary clearance + 0.15 m), in the travel window
 9:     target clearance ← gap to each track's hull, in the first 10 s
10:     clearance ← the smallest of both;  first violation ← earliest sample at which either is < 0.20 m (∞ if none)
11:     heading ← 1 − |wrap(LOS course at the window's last position − predicted heading there)| / π
12:     distance ← clip(clearance, 0, 1.5 m) / 1.5 m
13:     velocity ← clip(mean predicted speed in the first 10 s / U_nom, 0, 1)
14:     path ← 1 − clip(mean |e_y| in the travel window / 2 m, 0, 1)
15:     smoothness ← 0 if previous choice = none, else |speed fraction − previous fraction| plus
                     |set-point − previous set-point| / (18°/s) if both are set-points;
                     0 if both are LOS-PID; 1 if one of each
16:     score ← 1.0 heading + 1.0 distance + 0.5 velocity + 0.6 path − 0.25 smoothness
17: end for
18: if some candidate has first violation = ∞ then choice ← the admissible candidate of highest score
19: else choice ← the candidate whose first violation is latest; ties by clearance, then score
20: previous choice ← choice
21: if the choice is LOS-PID then return LOS-PID(the choice's speed)
22: reset the PID integral;  a_δ ← YAW-RATE-LOOP(the choice's set-point);  a_n ← SPEED-LOOP(the choice's speed)
23: return a
```

LOS-PID(speed): heading error e = wrap(LOS course − sideslip − ψ) in degrees; a_δ = clip(e/35 −
r[°/s]/18 + integral, −1, 1), the integral gaining e/600 per second only while |e| < 10° (held
within ±0.15, otherwise decaying by 10 % per decision); a_n ← SPEED-LOOP(speed). The heading PD
of line 6 is the same law without the integral.

| Parameter | Value |
|---|---|
| Engagement | a track or a static point within 9 m (the point no more than 1 m abaft the centre); hull within 1.0 m of the boundary |
| Yaw-rate set-points | −9 to +9°/s in 2.25°/s steps (9); demands to the yaw-rate loop, not achieved rates |
| Speeds | {1, 0.5, 0} × U_nom (0 = zero rpm, coasting); LOS-PID at {1, 0.5} × U_nom |
| Hold times | 3 s, 6 s; then the prediction hands back to LOS-PID |
| Prediction | 16 s, RK4 at 0.125 s on the identified model; rudder delay 0.73 s (6 samples) |
| Admissibility windows | static points and boundary: the first 5.5 m travelled (10 s at cruise); tracks: the first 10 s |
| Admissibility gaps | 0.20 m to a static point and between the inflated hulls; 0.05 m to the boundary |
| Objective weights | heading 1.0, distance 1.0, velocity 0.5, path 0.6, smoothness 0.25 |
| Scales | distance cap 1.5 m; path 2 m of mean cross-track offset; set-point change 18°/s |
| Yaw-rate loop | feed-forward 10.4°/s per unit rudder, proportional 0.08 per °/s |
| Heading PID | 1/35 per °, 1/18 per °/s of yaw rate, 1/600 per °·s inside 10° (limit 0.15) |
| Tuning | defaults kept: the 60-episode search's preferred values (14 s target window, 1.0 m cap) scored +0.650 on all 120 validation episodes, equal to the defaults; pinned in `configs/comparators_v1.json` |

**Notes for the text.**
- **No COLREGs.** DWA knows obstacles, not rules: a track enters only as a moving obstacle at
  constant velocity, with no classification, turn sense or stand-on. The comparator shows what
  collision avoidance without the rules does in these encounters, and there is no narrow-channel
  convention to state.
- **Adaptations of Fox et al. (1997).** (1) The window is the set of trajectories the vessel can
  fly, predicted through the identified dynamics (rudder delay, servo, surge and yaw lags), not a
  one-step (speed, yaw-rate) window of constant-curvature arcs: this hull needs about 5 s to build
  its yaw rate. A set-point is held for its hold time and then handed back to LOS-PID, which is
  what the controller does once the obstacle is passed; held for the whole 16 s, every arc ends
  far from the path and the heading term cannot rank them. (2) The goal is a path, not a point:
  the heading term is measured against the LOS course at the end of the travel window, and a path
  term and a smoothness term are added; without the path term a turn-away candidate is re-chosen
  every step and the return never comes. (3) Static points and the boundary are judged over equal
  travelled distance, not equal time, so a slow candidate cannot look clear by not arriving;
  tracks are judged over time. (4) No braking fallback: the hull coasts for tens of metres and
  loses rudder authority when it slows, so the fallback takes the latest first violation.
  (5) The boundary polygon is exact and held to 0.05 m; scan points are noisy and held to 0.20 m.
- The feed-forward 10.4°/s per unit rudder is a scale on the candidate set, not the plant's turn
  rate (about 5°/s at full rudder at cruise). The rollouts run the same law on the same model, so
  the trajectories scored are those flown; the nine set-points spread over about 9–36° of
  heading change in 6 s at cruise (check of 2026-10-10, `results/classical/dwa_yaw_gain_check.md`).
- Receding horizon: the choice is re-made every 0.5 s and only its first decision is executed.
- Sources: `src/classical/los_dwa.py` (`LosDwaController.action`, `_threat_in_reach`,
  `_clearance`, `_los_command`), `src/classical/common.py` (`Perception.snapshot`, `los_heading`,
  `HeadingPID`, `pd_rudder`, `yaw_rate_rudder`, `speed_rpm`, `throttle_for_rpm`, `Actuators`,
  `rollout`, `point_clearance`, `boundary_clearance`, `target_gap`),
  `src/reference_controller.py` (`hull_separation`), `src/constant_temp.py` (`CLASSICAL_DWA_*`,
  `CLASSICAL_PID_*`, the shared values), `configs/comparators_v1.json`.

---

## Algorithm A3. Runtime safety layer (one 2 Hz decision)

**Placement.** Appendix F, after Algorithm A2; referenced from Sections 2.3.4 and 4.5. Decided
2026-10-25: the one layer the paper reports. No other configuration (no other stop window) and
none of the layer's development versions is described; the text names neither.

**Purpose.** Fewer collisions of the own ship than the policy alone makes. A higher success rate
is welcome but is not the aim. A target that strikes the stopped own ship is counted separately,
not as an own-vessel collision.

**Require:** the policy's command a_SAC = [a_δ, a_n]; the layer's own snapshot (below), from
onboard estimates only; the actuator history; the stored plan, if any; the stop flag.

**Ensure:** the executed command; whether it differs from a_SAC; whether the stop is committed.

```
 1: if the stop is committed then return [0, full astern]             ▷ permanent: rudder centred
 2: if no hazard is in reach (no track within 7 m; no static return within 7 m and not more than
    1 m astern of the own ship's position; the boundary farther than 1.5 m) then return a_SAC
 3: first commands ← a_SAC; the grid rudder {−1, −0.5, 0, 0.5, 1} × propulsion {floor, cruise,
    ceiling}; a command that rejoins the path; with a track within 7 m, full astern at rudder
    {−1, 0, 1}
 4: recoveries ← hard port or hard starboard at cruise for 2 s, then the rudder centred; the rudder
    centred at the propulsion floor; with a track within 7 m, full astern at rudder {−1, 0, 1}
 5: for each first command and each recovery do                          ▷ FILTER
 6:     roll out on the identified vessel model (rudder delay and servo lag, from the actuator
        history): the first command for 1 s, then the recovery to 8 s; tracks at constant velocity
 7:     passes ← throughout, the hull inflated by its 0.15 m margin keeps 0.05 m from every static
        return, 0.02 m from the map boundary and 0.10 m from every target hull; and a hull still
        moving at 8 s has a clear run ahead of static returns (2.0 m, or 3 s at its final speed
        if shorter)
 8:     clearance ← the smallest of those gaps over the rollout
 9: end for
10: if a_SAC passes, with some recovery, with clearance ≥ 0.15 m then return a_SAC     ▷ untouched
11: if some first command passes then
12:     issue the passing first command nearest to a_SAC (rudder difference² + 2 × propulsion
        difference², full astern counted as −1.5), among those whose clearance is at least
        min(0.15 m, best clearance − 0.20 m); store its plan; return it
13: search for a checked plan by the cross-entropy method (3 rounds, 64 sampled sequences,
    8 elites); if a plan passes then issue its first command, store it and return it
14: STOP TEST (no checked plan at this decision):                          ▷ EMERGENCY STOP
15:     for each first command (a_SAC, the command the filter would issue, the grid of line 3 and
        the full-astern rows), each continuation of a fixed bank (hold the command; stop; turn,
        then stop; turn, cruise, then stop), over the window W = 2.5 s, under a weak reverse-thrust
        model (25 % efficiency, 0.75 s delay) and, for sequences that brake, a strong one (50 %,
        no delay) do
16:         time to contact ← the first time at which the hull inflated by 0.15 m, the geometry by
            which a collision is scored, leaves the map boundary or overlaps a target's hull on its
            constant-velocity track (∞ if neither happens within W; remembered static returns
            are not part of the test)
17:     if the longest time to contact over all sequences and both models is ≤ W then
18:         commit the stop; return [0, full astern]
19: return the filter's own decision (the next command of a stored plan while it holds, else a_SAC)
```

**The layer's perception** (serves the layer only; the policy's observation is unchanged): a
model-based observer of the own ship's motion (an issued-command model prior corrected by the
sensed motion); remembered static returns, cleared only where fresh rays pass through them; target
hypotheses admitted on observed motion and kept for a bounded time when a track is lost; and the
hull of a partly seen target completed from its axis of motion.

| Parameter | Value |
|---|---|
| Hazard in reach | a track within 7 m; a static return within 7 m and not more than 1 m astern of the own ship's position; the boundary within 1.5 m |
| Filter rollout | first command held 1 s, recovery to 8 s, on the identified model with rudder delay and servo lag |
| First commands | a_SAC; rudder {−1, −0.5, 0, 0.5, 1} × propulsion {floor, cruise, ceiling}; path rejoin; full astern at rudder {−1, 0, 1} with a track within 7 m |
| Recoveries | hard port or starboard at cruise for 2 s, then centred; centred at the floor; full astern at rudder {−1, 0, 1} with a track within 7 m |
| Hard gaps (inflated hull, 0.15 m margin) | 0.05 m to static returns, 0.02 m to the boundary, 0.10 m to target hulls |
| Terminal run-out | 2.0 m, or 3 s at the final speed, clear of static returns ahead of a hull moving faster than 0.10 m/s at 8 s |
| Policy kept | when its command passes with clearance ≥ 0.15 m |
| Selection | nearest to a_SAC by rudder² + 2 × propulsion², within min(0.15 m, best − 0.20 m) of clearance |
| Search | cross-entropy method, 3 rounds × 64 samples, 8 elites |
| Stop: when tested | only at a decision without a checked plan |
| Stop: window | W = 2.5 s (5 decisions) |
| Stop: continuations | hold; stop; turn then stop; turn, cruise, then stop |
| Stop: reverse thrust | weak 25 % with 0.75 s delay; strong 50 %, no delay (braking sequences) |
| Stop: contact | hull + 0.15 m outside the map boundary, or overlapping a target hull; static returns not used |
| Stop: action | full astern, rudder centred, for the rest of the episode |
| Evaluation convention after a stop | targets brake in a straight line at 0.471 m/s² to rest; the episode ends stopped when the own ship is still (≤ 0.05 m/s, ≤ 1°/s) and every target has stopped; a target that strikes the stopped ship is scored apart from own-vessel collisions |

**Notes for the text.**
- **Two parts.** A predictive filter that replaces the policy's command only when it fails the
  check, and a permanent stop tested only when the filter has no checked plan. The layer steers as
  well as stops, so it is not a pure "take all way off" layer; Rule 8(e)'s second layer (taking all
  way off) is its stop.
- **Not a guarantee.** The banks, horizons and models are finite; the layer is evaluated
  empirically (Section 4.5).
- **Selection.** Chosen on the validation set; the test set had been used during the layer's
  development (from 2026-10-10), so the test-set figures are not a fully held-out estimate for the
  layer. Both statements belong in the paper.
- **The target-strike count** depends on the target-braking convention above.
- **Results (deployed policy, test set v4, layer off → on):** own-vessel collisions 135 → 70
  (obstacle 70 → 45, boundary 12 → 17, target 53 → 8; decoupled 46 → 18, coupled 89 → 52); goals
  865 → 868; stopped 52; target strikes on the stopped ship 9; timeouts 0 → 1; 8.9 % of decisions
  and 36 % of episodes with a changed command (the held stop included). Paired: of the 135
  collision episodes, 39 reach the goal, 32 end stopped, 3 end with a target strike and 61 still
  collide; of the 865 goals, 829 stay goals and 20 become stops, 9 own-vessel collisions, 6 target
  strikes and 1 a timeout. Validation set (270): own-vessel collisions 53 → 24, goals 217 → 222,
  stops 23, changed decisions 13.5 % (4.6 % outside the episodes that end in a stop). The layer-off
  run reproduces the test-set run of the deployed policy exactly.
- **Field trials.** Which parts of the layer run on the vessel (the full layer, or only the
  operator-latched stop) is part of the pending field-trial design.
- Sources: `src/safety_v21.py` (`SafetyFilterV21EC(estop_window_s=2.5, use_static_points=False,
  braking_fallback=False)`, `collision_escape_time`, `collision_contact_times`), its filter lineage
  `src/safety_v16.py` back to `src/safety_v2.py` (constants), `src/safety_trajectory_search.py`,
  `src/safety_estop.py` (`continuation_bank`, `_stop`), `src/safety_v20_contingency.py` (`WEAK`,
  `STRONG`); `planning_records/SAFETY_V21_PLAN.md` section 15 and
  `planning_records/SAFETY_ESTOP_PROTOCOL.md` (evaluation convention); report
  `results/safety_dev/testset_v4_main/reports/ts4_final/`.
