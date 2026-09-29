# Paper 3 — Section 2 equations: action, observation, encounter geometry and reward

**Written 2026-09-29** from the code of the frozen formulation baseline-v2
(`configs/baseline_v2.json`, formulation `3d697858e95e5adf`). Every expression
below is what the code computes, with the constant values in force during
training; source files are named per block. This is the companion to
`planning/FORMULATION_PLAN.md`: the plan gives structure and argument, this
file gives the exact equations for the MDP subsection (observation, action,
reward) in the style of Paper 2.

Notation. Own ship (OS) position **p**_OS, heading ψ (compass, clockwise from
north, degrees), surge u, sway v, yaw rate r. Target (TS) quantities carry
subscript TS. World frame: x east, y north. Relative bearing α ∈ [0°, 360°)
(0° dead ahead, 90° starboard); heading-crossing angle c = (ψ_TS − ψ) mod 360°
(180° reciprocal). `clip(x, a, b)` limits x to [a, b]. Δt = 0.5 s per decision.
Perceived quantities (tracker, estimated pose) carry a hat where the distinction
matters: \hat{·}.

---

## 1. Decision process and termination (`env.py`, `constants.py`)

- Objective: maximize E[Σ_{t=0}^{T−1} γ^t r_t], γ = 0.951 per decision
  (= 0.99^5 per 0.1 s physics step), T ≤ 180.
- **Goal** (terminal, +100): distance to the goal point ≤ 0.5 m, **or**
  remaining along-path distance ≤ 1.25 m with |e_y| ≤ 1.60 m (and not
  overshot beyond 1.60 m past the endpoint).
- **Collision** (terminal, −300): hull polygon (with a 0.15 m hull margin)
  contacts a static obstacle, the navigable boundary, or the target hull.
- **Timeout**: t = 180, no terminal reward (the learner bootstraps the value).

## 2. Action (`env.py`, `curriculum.py`, `ship.py`)

a_t = [a_δ, a_n] ∈ [−1, 1]²

- Rudder command: δ_c = 100 a_δ (% of full deflection), full deflection
  δ_max = 40°; the servo applies rate limit and lag in the vessel model.
- Propeller command: n_c = clip(6 + 6 a_n, 0, 12) RPM (propulsion stage 4).
  Cruise n = 6 RPM → U_ref = 0.558 m/s on the nominal model. No astern for the
  policy (the runtime layer alone can reverse).

## 3. Observation (`observation.py`, `lidar_pooling.py`, `boundary_raycast.py`; `OBSERVATION_SPEC.md`)

o_t = {lidar ∈ [0,1]^27, boundary ∈ [0,1]^7, ego ∈ [−1,1]^3, path ∈ [−1,1]^3,
target ∈ R^16, context ∈ R^14} — 70 values.

**3.1 lidar (27).** Static returns only, ±135° swath (the aft 90° is left to the
tracker). Sectors: 15 × 6° within ±45°, 8 × 11.25° abeam, 4 × 22.5° on the
quarters. Each sector's beams {ρ_j} are pooled by **feasibility pooling**
(Meyer et al., 2020a, Algorithm 1; unchanged from Paper 2): the pooled range
ρ̄_k is the largest range at which an opening wider than the safe width
w_s = B + 2·0.15 = 0.80 m exists within the sector, the arc between adjacent
beams at range ρ being Δθ·ρ with Δθ = 0.5°. Then

  c_k = clip(1 − ρ̄_k / R_max, 0, 1),  R_max = 16 m.

**3.2 boundary (7).** Map ray-cast of the navigable polygon from the
**estimated** pose at bearings β_i ∈ {−90°, −60°, …, +90°}, ranges d_i ≤ 16 m:

  b_i = clip(1 − d_i / 16, 0, 1).

**3.3 ego (3).** With speed scale U_s = 2 U_ref = 1.116 m/s:

  [clip(û/U_s, −1, 1), clip(v̂/U_s, −1, 1), clip(r̂[°/s]/180, −1, 1)].

**3.4 path (3).**

  [clip(ê_y / h_side, −1, 1), clip(χ̃/180°, −1, 1), clip(χ̃_LA/180°, −1, 1)]

e_y cross-track error (positive to starboard), h_side the distance from the path
to the boundary on the side the vessel is on (so ±1 marks the navigable edge),
χ̃ course error, χ̃_LA look-ahead course error, all from the estimated pose.

**3.5 target (16)** — one slot; zeros with presence 0 when no track is held.
From the tracked target and the estimated own ship:

| # | Feature | Encoding |
|---|---|---|
| 0 | distance to the target's domain d_dom (Eq. 5) | clip(d_dom / 16, 0, 1) |
| 1–2 | relative bearing α | sin α, cos α |
| 3–4 | heading-crossing angle c | sin c, cos c |
| 5 | target speed U_TS | clip(U_TS / U_s, 0, 1) |
| 6 | relative speed |**v**_TS − **v**_OS| | clip(v_rel / U_s, 0, 1) |
| 7 | DCPA measured to the domain, DCPA_dom = max(0, DCPA − R_dom(α)) | clip(DCPA_dom / 1.25, 0, 12.8) / 12.8 |
| 8 | TCPA | clip(TCPA, −40, 40) / 40 |
| 9 | CRI (Eq. 6) | ∈ [0, 1] |
| 10–14 | class one-hot (none, head-on, crossing, overtaking, being overtaken) | {0, 1} |
| 15 | presence | 1 |

**3.6 context (14)** — the latched encounter (Section 4.4), then the previous
executed action:

| # | Feature | Definition |
|---|---|---|
| 0, 1 | engaged, clearing | state indicators |
| 2 | compliant turn sense σ | +1, 0, −1 (Table 4 of the draft) |
| 3 | heading change since engagement | wrap(ψ̂ − ψ_e)/180° (0 when idle) |
| 4 | speed change since engagement | clip((û − u_e)/U_s, −1, 1) |
| 5 | turn admissible | A (Eq. 11) for the compliant side |
| 6 | slowdown clears | S_coast (Eq. 12, coasting profile) — see note |
| 7 | admissibility known | 1 when Eq. 11 was evaluated on a boundary |
| 8 | action required | a_req = clip(Δy_req / d_req, 0, 1) |
| 9 | proximity gate | ρ (Eq. 9) |
| 10 | in extremis | DCPA < d_req and 0 ≤ TCPA < 5 s |
| 11 | engagement age | clip((t − t_e)Δt / 25 s, 0, 1) |
| 12, 13 | previous rudder, throttle | executed a_{t−1} |

## 4. Encounter geometry (`cpa_cri.py`, `encounter.py`, `colregs/context.py`, `colregs/geometry.py`)

**4.1 CPA** (relative position **p** = **p**_TS − **p**_OS, velocity
**v** = **v**_TS − **v**_OS, from the tracker):

  TCPA = −(**p**·**v**)/|**v**|²,  DCPA = |**p** + **v** TCPA|   (Eq. 3)

(|**v**|² < 10⁻¹² ⇒ DCPA = |**p**|, TCPA = 0.)

**4.2 Ship domain** (asymmetric; fore a_f = 3.14 m, aft a_a = 1.57 m, abeam
b = 1.25 m), radius at relative bearing α:

  R_dom(α) = [ (cos α / a)² + (sin α / b)² ]^{−1/2},  a = a_f if cos α ≥ 0 else a_a   (Eq. 4)

  d_dom = max(0, |**p**| − R_dom(α)),  d_req = 2b = 2.5 m   (Eq. 5)

**4.3 Collision risk index** (after Waltz & Okhrin, 2023):

  CRI = 1 if d_dom = 0, otherwise
  CRI = min{1, max(CR_CPA, CR_ED)},
  CR_CPA = exp(−DCPA_dom / 4 m) · exp(−|TCPA| / T_s) · f_bow,
  T_s = 20 s (TCPA ≥ 0) or 6 s (TCPA < 0),
  CR_ED = exp(−d_dom / 5 m)   (Eq. 6)

f_bow = 1 + 0.3 (1 − φ/45°) when the own ship lies within φ < 45° of the
target's bow (φ = bearing of the OS from the TS's heading), else 1.

**4.4 Classification** (against the path tangent ψ_p at the own ship, not ψ;
3° bearing hysteresis; a new class must persist 2 steps). Tested in order:

1. c ∈ [−67.5°, 67.5°] (near-parallel): **overtaking** if the OS lies in the
   TS's stern arc [112.5°, 247.5°] and U_OS > U_TS + 0.15 U_ref;
   **being overtaken** if the TS lies in the OS's stern arc and
   U_TS > U_OS + 0.15 U_ref.
2. **head-on**: α ∈ [−10°, 10°] and c ∈ [170°, 190°].
3. **crossing from starboard**: α ∈ [10°, 112.5°] and c ∈ [190°, 292.5°];
   **crossing from port**: α ∈ [247.5°, 350°] and c ∈ [67.5°, 170°].
4. **none** otherwise.

**4.5 Engagement and latch** (κ_eng = 1.5, κ_rel = 2.5, T_eng = 25 s):

  engage ⇔ class ≠ none ∧ 0 < TCPA ≤ 25 s ∧ DCPA < 1.5 d_req (= 3.75 m)   (Eq. 7)

At engagement latch class, crossing side, σ, ψ_e = ψ̂, u_e = û, t_e.
Engaged → clearing when TCPA < 0 ∨ DCPA > 2.5 d_req (= 6.25 m); clearing →
engaged if that fails again; release after N_clr = 6 consecutive steps with
TCPA < 0 and range > d_req.

**4.6 Lateral deficit and room**

  Δy_req = max(0, d_req − DCPA)   (Eq. 8)

  ρ = clip(1 − DCPA / (κ_eng d_req), 0, 1),  ρ_pk = max ρ since engagement   (Eq. 9)

Room on each side: beam-on map distance from the own ship (current cross-track
offset carried forward) to the boundary, minimized over the along-path interval
from now to s + max(TCPA, 0)·û (sampled), less half-beam and wall clearance:

  R_side = min_{s ∈ passage} d_side(s) − B/2 − c_wall,  c_wall = 0.65 m   (Eq. 10)

  A_side = 1 if R_side − Δy_req ≥ +0.15 m; 0 if ≤ −0.15 m; unchanged otherwise
  (initial value: R_side ≥ Δy_req). turn admissible A = A_stbd for σ = +1,
  A_port for σ = −1, 1 for σ = 0.   (Eq. 11)

**4.7 Stop / slowdown tests** (`stopping.py`). With the target on its
constant-velocity track and the own ship following a braking profile s_b(τ)
along its heading from its current surge,

  D_stop = min_τ | **p**_TS(τ) − **p**_b(τ) |   (minimum over braking, then CPA
  with the own ship stopped)

  S = 1 ⇔ D_stop ≥ d_clr,  d_clr = LOA/2 + B/2 + 2·0.15 + d_safe = 1.76 m   (Eq. 12)

Two profiles: **full-astern stop** (the runtime layer's manoeuvre; used by the
reward's carve-out and the Rule 8 credit, `R2_SLOWDOWN_TEST = "stop"`) and
**coasting at zero RPM** (S_coast; the observation's "slowdown clears" flag).
**[Note for the text: the reward and the observation use different profiles in
baseline-v2; state it, or describe the flag as "coasting slowdown clears".]**

## 5. Reward (`reward/terms.py`, `reward/config.py`)

  r_t = w_pf r_pf + w_prog r_prog + w_exist κ_ex r_exist + w_smooth r_smooth
      + w_obs r_obs + w_bnd r_bnd + w_dom r_dom + w_col r_col + r_term   (Eq. 13)

| Term | w | Range |
|---|---|---|
| r_pf | 3.0 | [−1, 0] |
| r_prog | 1.5 | [−1, 1] |
| r_exist | 0.25 | −1 |
| r_smooth | 0.5 | [−1, 0] |
| r_obs | 11.0 | [−1, 0] |
| r_bnd | 15.0 | [−1, 0] |
| r_dom | 12.5 | [−1, 0] |
| r_col | 9.0 | [−1, 0] |
| r_term | +100 goal, −300 collision, 0 timeout | |

κ_ex = 0 only under carve-out R-5 (Eq. 15), else 1.

**5.1 Path following** (width-normalized, speed-gated):

  ẽ = clip(e_y / (W_loc/2), −1, 1),  χ* = 0.25 χ̃_LA + 0.75 χ̃
  q = 0.7 exp(−4 ẽ²) + 0.3 (1 + cos χ*)/2
  g_u = clip(max(u, 0)/U_eff, 0, 1) · g_over(u)
  g_over(u) = clip(1 − (u − 1.2 U_ref)/(0.5 U_ref), 0, 1)
  r_pf = −(1 − g_u q)   (Eq. 14)

U_eff is U_ref except under a carve-out (Eq. 15); g_u = 1 while the runtime
stop is latched.

**5.2 Speed-reference carve-outs** (while an encounter is engaged):

  R-5 (overtaking, port pass inadmissible): U_eff = max(U_TS, 0.2 m/s), κ_ex = 0
  R-2 (give-way, A = 0, S_stop = 1): U_eff = 0.4 U_ref   (Eq. 15)

**5.3 Progress**: r_prog = clip(N_ref (s_t − s_{t−1}) / L_path, −1, 1),
N_ref = L_ref / (U_ref Δt) = 71.7 (so Σ r_prog telescopes to a constant over a
completed path; slowing costs no progress reward).   (Eq. 16)

**5.4 Smoothness**:

  r_smooth = −σ_t clip((Δa_δ/0.25)² + 0.5 (Δa_n/0.30)², 0, 1),
  σ_t = 0.25 for the first 4 steps after engagement, else 1   (Eq. 17)

**5.5 Boundary, domain, obstacles** (ground truth):

  r_bnd = −[max(0, 1 − d_bnd/0.35)]²
  r_dom = −[max_TS max(0, 1 − |**p**_TS − **p**_OS| / R_dom(α))]²
  r_obs = −clip((e^{−d_clr/0.6} − e^{−2.0/0.6}) / (1 − e^{−2.0/0.6}), 0, 1)   (Eq. 18)

d_bnd hull-to-boundary distance; d_clr hull-to-hull clearance to the nearest
static obstacle within ±135°; r_obs = 0 beyond 2.0 m.

**5.6 COLREGs group** (perceived, latched encounter; zero unless engaged):

  v_col = clip(0.55 v_port + 0.55 v_bow + 0.40 v_side + 0.45 v_hold + 0.50 v_r8, 0, 1),
  r_col = −v_col   (Eq. 19)

With σ the compliant sense, r_p the yaw rate needed to track the path, and
headings on the perceived frame:

- **Wrong-way turn or heading** (give-way classes, σ ≠ 0):

  v_port = max(ρ, ρ_pk) · clip(max(m_r, m_ψ), 0, 1),
  m_r = (max(0, −σ(r − r_p)) − 0.02 rad/s) / 0.2 rad/s,
  m_ψ = (max(0, −σ wrap(ψ̂ − ψ_e)) − 5°) / 20°   (Eq. 20)

- **Bow crossing** (crossing, overtaking):

  v_bow = ρ · σ_bow(β_CPA) · clip(1 − DCPA/d_req, 0, 1),
  σ_bow(β) = clip((cos β − cos 67.5°)/(1 − cos 67.5°), 0, 1)   (Eq. 21)

  β_CPA: bearing of the OS from the TS at the projected CPA.

- **Wrong side** (y_CPA: target's lateral offset at the projected CPA in the OS
  body frame, positive to starboard):

  head-on: v_side = ρ clip(+y_CPA/d_req, 0, 1);
  overtaking (A_port = 1): v_side = ρ clip(−y_CPA/d_req, 0, 1)   (Eq. 22)

- **Hold course and speed** (being overtaken; 0 in extremis):

  v_hold = ρ [ clip(((r − r_p)/0.05)² + ((û − u_e)/0.10)², 0, 1)
              + clip((|û − u_e| − 0.10)/0.10, 0, 2) ]   (Eq. 23)

  (range [0, 3]: the speed part keeps rising past 0.10 m/s.)

- **Rule 8, late or insufficient action** (give-way classes):

  a_req = clip(Δy_req / d_req, 0, 1)
  Δψ_c = max(0, σ wrap(ψ̂ − ψ_e)),  Δu = max(0, u_e − û)
  A_t = Δu / 0.167 m/s + [A = 1 ∨ S_stop = 0] · Δψ_c / 20°
  v_r8 = clip(1 − TCPA/15 s, 0, 1) · clip(a_req − A_t, 0, 1)   (Eq. 24)

  (Δu_min = 0.30 U_ref = 0.167 m/s. Heading change counts when the turn is
  admissible, or when slowing cannot clear and turning is the only help.)

**5.7 Verified property (F94)** — a statement about Eqs. 13–24, not a result:
from one engagement state, a scripted 30° compliant crossing turn returns +61
(discounted) over the same turn made the wrong way, from either side.

## 6. Where Paper 2's equations still apply

The vessel model equations (3-DOF surge–sway–yaw with actuator dynamics) and
the feasibility pooling algorithm are as published in Paper 2 (Tran et al.,
2026); cite them rather than restating, and state only the changes (the
operating point, randomization scale, and the pooled swath and sector
layout). Paper 2's reward and observation are **not** carried over: the
width-normalized path term (Eq. 14) replaces exp(−0.05|e_y|), and every term is
bounded before weighting (Eq. 13).

## 7. Consistency notes for the drafts

- Formulation_v2 Table 3 crossing bands use the source's ±5° edges; the code
  uses the widened head-on band: α ∈ [10°, 112.5°], c ∈ [190°, 292.5°]
  (starboard) and α ∈ [247.5°, 350°], c ∈ [67.5°, 170°] (port).
- Formulation_v2 Eq. 8 (full-astern braking path for the reward's slowdown
  credit) is **correct** for baseline-v2 (`R2_SLOWDOWN_TEST = "stop"`); the
  observation's flag uses the coasting profile (Section 4.7 above).
- DCPA in the observation is measured to the domain and rescaled to [0, 1]
  (Table 3.5, row 7); Eq. 3's DCPA is centre to centre and is what Eqs. 7–9
  and 19–24 use.
- One LiDAR scan and one tracker update per decision step (2 Hz); physics and
  contact checks at 0.1 s.
