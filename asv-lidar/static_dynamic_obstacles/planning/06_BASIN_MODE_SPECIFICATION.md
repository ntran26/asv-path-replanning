# 06 — Basin Mode: Navigable Geometry, Straight Legs, and Suite Restructure

**Revision 1.0** — introduces a second navigable-geometry mode and makes it the primary training
and evaluation geometry. Supersedes the single-corridor assumption in 03a §3 and 04a §3.2.
**Handover target:** Claude Code (§3–§7, §9), Claude chat (§2 decisions are signed off)
**Depends on:** 03a §3 (corridor), 04a §3 (generator), 01 §3 (boundary branch), 02a §3 (`r_pf`)
**Companion decision:** straight reference paths only (survey-leg framing). Bends are retained as a
config parameter fixed at zero, not deleted.

---

## 1. Why this change

Two problems, one fix.

**The boundary branch lost its justification.** 01 §3.3 rests the 7-ray boundary branch on three
decorrelation mechanisms: varying width, path offset, and bends. Removing bends leaves two, and in a
straight parallel-walled corridor both are weak: port and starboard clearances remain close to affine
functions of `e_y`. Acceptance test T3 (`|corr(e_y, b_i)| < 0.9`) then passes only marginally, and the
branch is decorative in exactly the regime the paper now claims to address.

**Corridor mode is not the field geometry.** The pool is a fixed 10 × 25 m rectangle. A survey leg run
across it at an angle — Paper 2's layout — has walls oblique to the path, clearance that changes along
the leg at constant heading, and corner geometry at the ends. None of that is reachable when the
navigable polygon is constructed as an offset of the path itself.

Basin mode supplies both: genuine boundary-ray decorrelation from straight legs alone, and a training
distribution that matches what the basin sessions will actually run. It also puts the Paper 2 policy
on its own map, which makes that comparator honest rather than re-tuned.

What basin mode does **not** replace: Rule 9 narrow-channel claims need parallel walls at a controlled
width. Channel mode is retained for Study 1 and for the intermediate and narrow N2 cells.

---

## 2. Decisions

| # | Decision |
|---|---|
| **M-1** | The generator emits two navigable-geometry modes: `basin` and `channel`. Mode is a scenario field, sampled per episode, recorded in the scenario record |
| **M-2** | **Basin mode is the primary geometry** for training (50% of stages 4–5, 100% of stages 1–2) and for the new Tier B basin stratum and the Tier A field-replicable cases |
| **M-3** | Channel mode is retained unchanged for Study 1, for the wide/intermediate/narrow Tier B strata, and for channel-constrained Around the Clock |
| **M-4** | Reference paths are straight in both modes. `bend_max` stays in the config, fixed at 0, with the clamp-and-record machinery intact |
| **M-5** | Path-relative normalisation becomes **side-specific**: `ẽ_y = e_y / h_side(s)`. In channel mode `h_+ = h_- = W/2`, so this reduces exactly to the current formula and no channel-mode result changes |
| **M-6** | Tier B is restructured to 48 cells × 20 episodes = 960 per policy seed, against 39 × 25 = 975. Narrow × null and narrow × being-overtaken are dropped as infeasible and reported through the rejection ledger |
| **M-7** | Tier A drops the two bend cases and adds six basin cases: 38 named cases |
| **M-8** | Scenarios reproducible in the physical basin carry `field_replicable = true`. All basin-mode cases qualify by construction; channel-mode cases qualify when the corridor fits the basin envelope |

---

## 3. Basin mode geometry

### 3.1 Polygons

| Polygon | Definition | Role |
|---|---|---|
| Basin envelope | 10 × 25 m rectangle | Limit of physical water |
| **Navigable polygon** `P_nav` | Envelope inset by `ι = d_safe + 0.05 = 0.40 m` → 9.20 × 24.20 m | Hard constraint; termination on breach |
| Facility walls | Envelope + 1.5 m | Returned by the LiDAR, then gated (01 §3.4) |

The inset matches the one the corridor generator already honours, so the two modes share the
termination test: *hull polygon leaves `P_nav`* (03a §4.4, wording generalised from "corridor polygon").

### 3.2 The reference path

A straight segment of length `L_path = 20 m`, defined by its midpoint `m` and slant `θ` measured from
the basin's long axis:

```
p_start = m − (L_path/2)·[cos θ, sin θ]
p_goal  = m + (L_path/2)·[cos θ, sin θ]
```

**Slant limit.** With a required free band of `w_clear` each side of the path, the segment fits when

```
L_path·cos θ + 2·w_clear ≤ 24.20            (long axis)
L_path·sin θ + 2·w_clear ≤ 9.20             (short axis)
θ_max = arcsin( (9.20 − 2·w_clear) / L_path )
```

At `w_clear = 1.50 m` this gives `θ_max = 18.1°`; at `w_clear = 1.25 m`, `20.7°`. Sample
`θ ~ U(−θ_max, θ_max)`, clamp, and record requested and realised slant exactly as the bend clamp did
(F25 pattern). Sign is symmetric — do **not** bias it, the Rule 9(a) station is carried by the lateral
offset, not by the slant.

**Endpoints.** Both endpoints must lie inside `P_nav` eroded by `e_clear = 1.50 m`, so no leg begins or
ends in a corner. Sample `m` inside the erosion of `P_nav` by `(L_path/2·cos θ + e_clear,
L_path/2·sin θ + e_clear)`, rejecting otherwise.

**Lateral offset.** After `m` and `θ` are fixed, offset the path perpendicular by
`o ~ U(−0.30, +0.30)·min_s h_min(s)` with positive mean, per Rule 9(a). Positive is toward the
starboard-side boundary in the direction of travel.

### 3.3 Clearance functions

For path arclength `s`, cast the path normal both ways to `P_nav`:

```
h_+(s) = distance from C(s) to P_nav on the starboard side
h_−(s) = distance from C(s) to P_nav on the port side
W_eff(s) = h_+(s) + h_−(s)                  effective clear width
```

For a slanted chord both are affine in `s` with opposite slopes — the ship runs obliquely toward one
boundary and away from the other. That is the property the boundary branch needs, and it is why
straight legs are sufficient once the walls are no longer parallel to the path.

Report `W_eff` at CPA as the scalar width for stratification and for the N2 precedence table. It is
not a constant in basin mode, so never report basin cases as if they had a single channel width.

### 3.4 Quantities that consume the clearance functions

| Quantity | Definition | Channel-mode reduction |
|---|---|---|
| `ẽ_y` (02a R-1, `r_pf`) | `e_y / h_side(s)`, `h_side` = clearance on the side of the deviation, clipped to `[0.60, 5.00] m` | `h_± = W/2` → identical to current |
| `r_bnd` (02a) | unchanged: hull-to-`P_nav` distance against `d_safe` | unchanged |
| `r_stbd`, `r_port` (admissibility, 02a §6) | `min_s (h_±(s) − B/2 − c_wall)` over the stretch to CPA | unchanged |
| Boundary branch (01 §3.2) | 7 virtual rays cast against `P_nav` from the estimated pose | unchanged |
| Width stratum | `W_eff` at CPA | `W` |

The clip on `h_side` matters: near the endpoints of a slanted leg the outward clearance grows toward
the far corner, and an unclipped normalisation would make `r_pf` insensitive there.

### 3.5 Targets in basin mode

The backward solve (04a §3.3) is unchanged — it never referenced the corridor. Only the validity
checks change:

- **Containment:** both hulls inside `P_nav` over `[0, min(T_horizon, t_exit)]`.
- **Band confinement, class-conditional:** head-on, overtaking, being-overtaken and null targets are
  confined to a band of half-width `min(h_+, h_−)` about the path, so they behave as channel traffic.
  Crossing targets are free within `P_nav`, which is the basin-mode reading of Rule 9(d).
- Acceptance test T7 becomes mode-conditional: *crossing targets leave the path band; all other
  classes never do.*

### 3.6 Static obstacles

Unchanged in rule, wider in placement: panels may now sit outside the path band but inside `P_nav`,
including near a wall the leg approaches obliquely. The non-interference constraint (no panel inside
either swept domain within `±0.4·T_0` of CPA) still applies, as do the `conflict` and `occlusion`
flagged variants.

---

## 4. Training distribution

`p_basin` is a per-stage parameter.

| Stage | `p_basin` | Basin slant | Channel width | Target | Clutter |
|---|---|---|---|---|---|
| 1 | 1.00 | `|θ| ≤ 6°` | — | none | 0 – 1 |
| 2 | 1.00 | full range | — | none | 0 – 3 |
| 3 | 0.70 | full range | 7 – 10 m | head-on, null | 0 – 1 |
| 4 | 0.50 | full range | 4.5 – 10 m | all classes | 0 – 2 |
| 5 | 0.50 | full range | 3.5 – 10 m | all classes | 0 – 3 |

Stage 1 restricts slant so the first stage is close to axis-aligned; everything else is unrestricted.
The 50% floor in stages 4–5 keeps the field geometry in distribution while leaving half the samples on
the narrow parallel-walled geometry that N2 depends on.

**This supersedes F25's consequence for stage 3.** With bends fixed at zero, the concern that stage 3
trains no `r_path` signal is resolved differently: `r_path` is zero everywhere by design, and the
boundary signal in stage 3 now comes from basin-mode slant rather than from a bend the width could
never carry.

---

## 5. Evaluation suite

### 5.1 Tier B — 48 cells, 20 episodes each

| Stratum group | Cells | Composition |
|---|---|---|
| **Basin** | 13 | 4 classes × 3 behaviours, plus null (CV only) |
| Channel, wide `[7.60, 10.00]` | 13 | same |
| Channel, intermediate `[4.26, 7.60]` | 13 | same |
| Channel, narrow `[3.50, 4.26]` | 9 | head-on, crossing, overtaking × 3 behaviours |

960 episodes per policy per seed, against 975 before, so the compute envelope is unchanged.

**Dropped cells.** Narrow × null and narrow × being-overtaken are removed: at 3.50–4.26 m a null
encounter is not an encounter, and a target overtaking in a channel of that width has nowhere to
pass. They are reported as infeasible with their generator rejection rates (0.99 and 0.96 at stage 5),
which is the §3.5 feasibility argument doing its job rather than a gap in the suite.

**CI note.** 20 episodes per cell widens each cell's interval by roughly 12% against 25. Cell-level
intervals were never the primary endpoint; the primary endpoint aggregates across cells and is
unaffected in practice.

### 5.2 Tier A — 38 named cases

| Change | Cases |
|---|---|
| Removed | `A-BND-HO-I`, `A-BND-CRS-I` — bends no longer exist |
| Added | `A-BSN-HO`, `A-BSN-CRS`, `A-BSN-CRP`, `A-BSN-OT`, `A-BSN-BO`, `A-BSN-CLT-CRS` |

Basin cases use `θ = 15°` and a starboard-side path offset, with the encounter placed at mid-leg where
the two clearances differ most. `A-BSN-CLT-CRS` places the conflict panel on the side the leg is
closing on, so the compliant alteration runs out of water for a reason the geometry makes visible.

All six carry `field_replicable = true` and form the basin-session trial list for 05.

### 5.3 Studies

| Study | Mode | Note |
|---|---|---|
| Study 1, width sweep | **channel only** | Width is the independent variable; parallel walls are the point. Unchanged |
| Study 2, perception degradation | **basin primary**, one channel-narrow level retained | The degradation claim is field-facing, so it belongs on the field geometry |
| Around the Clock | unchanged | Open water plus channel at 10, 6, 4.26 m. No basin ring — the contrast that carries N2 is open water against parallel walls |

---

## 6. Scenario record additions (04a §9.1)

```
geometry_mode           "basin" | "channel"
slant_requested_deg     float          basin only
slant_realised_deg      float          basin only, after clamp
path_midpoint           [x, y]         basin only
clearance_profile       {h_plus: [s0, s_mid, s1], h_minus: [...]}
w_eff_at_cpa            float
field_replicable        bool
```

`suite_version` increments to `3.0`. All hashes regenerate; the regeneration test must be re-run and
`SUITE_MANIFEST.json` recommitted.

---

## 7. Cross-document change list

| Document | Section | Change |
|---|---|---|
| **03a** | §3.1 | Add `P_nav` and the two modes to the polygon table; "corridor" becomes "navigable polygon" throughout |
| 03a | §3.2 | Replace the three decorrelation requirements with: basin mode carries slant, channel mode carries varying width and offset. T3 must pass **per mode** |
| 03a | §4.4 | Termination row: "leaves the corridor polygon" → "leaves the navigable polygon `P_nav`" |
| 03a | §5.2 | Confinement is band-based in basin mode (§3.5 here) |
| 03a | §10 | T7 mode-conditional; add T13–T15 (§8 here) |
| **04a** | §3.1 | Sampling order: mode is sampled at step 2, before geometry |
| 04a | §3.2 | Split the geometry table by mode; bend row fixed at 0; add slant row and `θ_max` formula |
| 04a | §3.3 | Backward solve unchanged; validity checks reference `P_nav` and the path band |
| 04a | §3.7 | Curriculum table gains `p_basin` (§4 here) |
| 04a | §4.2–4.3 | Tier B restructure (§5.1 here) |
| 04a | §5 | Tier A list (§5.2 here) |
| 04a | §6 | State explicitly: Study 1 is channel mode only |
| 04a | §7 | Study 2 runs on basin mode plus one channel-narrow level |
| 04a | §9.1 | Record fields (§6 here); `suite_version = 3.0` |
| **01** | §3.3 | Rewrite the redundancy argument: slant against fixed basin walls is now the primary decorrelation mechanism; width variation and offset are secondary; bends are gone |
| **02a** | R-1 / `r_pf` | Side-specific normalisation (M-5). State the channel-mode reduction explicitly so no earlier result is invalidated |
| 02a | §6 admissibility | `r_stbd`, `r_port` from `P_nav` clearances rather than from a constant half-width |
| **PROJECT_STATE** | §3.3 F25 | Mark the stage-3 consequence superseded by M-4 and §4 here |
| PROJECT_STATE | §2 blockers | Add: basin mode implementation and re-freeze; keep B3 and B5 open |
| **PAPER3_DRAFT_SKELETON** | problem formulation | Survey leg inside a laterally bounded basin; two geometries, one policy. Own ship does not claim Rule 3(g) status |

---

## 8. New acceptance tests

| # | Test |
|---|---|
| **T13** | Basin mode: both path endpoints lie inside `P_nav` eroded by 1.50 m; `|θ_realised| ≤ θ_max`; requested and realised slant both recorded |
| **T14** | Channel mode regression: with `h_+ = h_- = W/2`, `r_pf` is bitwise identical to the revision 8 implementation over a fixed 500-step replay |
| **T15** | Basin mode: both hulls remain inside the 10 × 25 envelope for the full horizon in 1000 sampled episodes |
| T3 (amended) | `|corr(e_y, b_i)| < 0.9` for all 7 rays, **evaluated separately for each mode** over 1000 episodes |
| T7 (amended) | Crossing targets leave the path band; all other classes never do |

T3 in basin mode is the test that decides whether this change achieved its purpose. If basin mode
alone does not clear it, the slant range is too narrow and `w_clear` should be reduced to 1.25 m
before anything else is reconsidered.

---

## 9. Implementation handoff — Claude Code

| Module | Work |
|---|---|
| `corridor.py` | Add `BasinGeometry`: `P_nav`, path sampling with slant, clamp-and-record, `h_±(s)`, `W_eff(s)`. Keep `ChannelGeometry` as-is behind a common interface (`nav_polygon()`, `clearances(s)`, `path()`) |
| `scenario.py` | Sample `geometry_mode` from `p_basin`; route validity checks by mode; extend the rejection ledger with `(mode, class, reason)`; new record fields |
| `env.py` | Termination against `nav_polygon()`; observation assembly unchanged; `W_local` replaced by `clearances(s)` |
| reward module | M-5 normalisation; admissibility from clearances; no other change |
| `suite.py` | Tier B 48 cells × 20; Tier A 38 cases; `field_replicable`; `suite_version = 3.0`; re-hash and regenerate the manifest |
| `test_acceptance.py` | T13–T15; amend T3 and T7 to be mode-conditional |
| `tools/scale_audit.py` | Re-run over both modes; confirm 02a §8.1 orderings hold in basin mode, where `h_side` varies along the leg |

**Order:** geometry and tests first, then the generator, then the suite rebuild. Do not re-freeze the
suite until T3 passes in basin mode, since a failing T3 changes the slant range and therefore every
hash.

---

## 10. Open items

| # | Item | Owner |
|---|---|---|
| M-a | `w_clear` default: 1.50 m (θ_max 18.1°) or 1.25 m (20.7°). Decide from the T3 result, not by preference | Claude Code measurement, then sign-off |
| M-b | Whether Study 2's channel-narrow level is 4.0 m or the 4.26 m stratum edge | open |
| M-c | Whether the Paper 2 policy is run on basin mode as a comparator, and under which action-space adapter | open, affects §8.2 priority tail |
| M-d | Basin-session trial list is now the six `A-BSN-*` cases; confirm against basin availability in 05 | open |

Unchanged and still open: B3 (throughput), B5 (curriculum budget, `TODO(04-3)`, `TODO(04-4)`),
`TODO(04-2)` (`D_max`).
