# 02 — Reward Function and COLREGs Encoding

**Revision 2** — Rules 13–16 with Rule 9 precedence; Rule 17 active release removed.
**Handover target:** Claude chat (design), then Claude Code
**Depends on:** nothing — **start here.** The precedence table produced in §3 gates the
encounter classifier in 01, the reward terms below, and the width sweep in 04
**Carry in:** `REWARD_REDESIGN.md` (the existing 6-term specification)

---

## 1. Purpose

Extend the 6-term reward redesign to cover a dynamic target and COLREGs Rules 8, 9 and
13–16, without recreating the scale failure found in Paper 2.

**Confirmed (D2):** compliance is enforced through **reward terms plus an explicit
encounter-class feature** in the observation — not shaping alone, not a separate
arbitration or safety-filter layer.

---

## 2. Why this mechanism

Pure continuous shaping is fragile because the rule is **discrete and conditional**: which
rule applies depends on relative bearing and heading-intersection sectors, so a smooth
reward cannot express a class switch cleanly.

A separate arbitration or safety-filter layer would give hard guarantees but breaks the
end-to-end framing and moves the interesting behaviour out of the learned policy.

**Chosen middle path:** classify the encounter deterministically each step (module defined
here, implemented in 01), then apply a **class-conditional** penalty. The discontinuity
lives in a classifier under your control rather than in learned features.

Precedent: Waltz & Okhrin include the encounter situation directly in the observation
vector *and* condition their COLREGs reward on it — useful cover when a reviewer asks
whether the agent is being handed the answer.

---

## 3. Rule precedence — the blocking deliverable

**This table is contribution N2 and it gates everything downstream.**

In a narrow channel Rule 9 modifies the others:

- 9(a) — keep to the starboard side of the fairway
- 9(b), 9(d) — do not impede vessels that can navigate only within the channel
- 9(e) — special overtaking provisions

Applying Rule 14's "both alter to starboard" in a channel too narrow to permit it will be
caught by any reviewer who knows COLREGs. Rule 15 crossing geometry is likewise partly
inapplicable where there is no room to cross.

### 3.1 The always-give-way simplification

**Own ship gives way in all crossing encounters, justified by Rule 9(b)** — a vessel under
20 m shall not impede the passage of a vessel that can safely navigate only within a
narrow channel.

This deliberately replaces the Rule 18 route used by Meyer et al. (2020), where the own
ship is always give-way because it is significantly smaller than the vessels encountered.
That premise fails here: own ship and target are similarly sized model vessels, so the
asymmetry Rule 18 requires does not exist, and claiming it in simulation then validating
against an identical vessel is an inconsistency a reviewer will find.

### 3.2 The table to produce

| Encounter | Wide channel | Narrow channel | Governing rule | Threshold |
|---|---|---|---|---|
| Head-on | `[TBC]` | `[TBC]` | `[TBC]` | `[TBC]` |
| Crossing | `[TBC]` | `[TBC]` | 9(b) throughout | — |
| Overtaking | `[TBC]` | `[TBC]` | `[TBC]` | `[TBC]` |
| Being overtaken | Hold course | Hold course | 13, 17(a)(i) | — |

Define "narrow" **geometrically** — in ship breadths, and in terms of the lateral excursion
the compliant manoeuvre requires — never by reference to where a method fails. The
threshold at which each transition occurs is itself a result, produced by the width sweep
(Study 1, doc 04).

Krasowski & Althoff (2024) is the model for formalising this into checkable
specifications.

---

## 4. Reward structure

### 4.1 Carried from `REWARD_REDESIGN.md`

Six terms — re-verify each against the new observation, since `c_t` no longer contains
borders:

1. Exponential clearance-based collision-avoidance term
2. Unified path-following term
3. Border penalty
4. Progress term
5. Action-smoothness penalty
6. Existence cost

Terminal: collision −200, goal +100, timeout via value bootstrapping.

**The border penalty now draws on the boundary branch and the geometric boundary, not on
LiDAR returns.** Confirm the term still behaves as designed.

### 4.2 COLREGs terms — five classes

| Term | Condition | Intent |
|---|---|---|
| Wrong-side passing | any class with a defined passing side | Penalise passing on the incorrect side |
| Port turn in head-on | head-on, TCPA > 0 | Penalise altering to port |
| Bow crossing | crossing, overtaking | Penalise crossing ahead of the target |
| Course-keeping hold | being overtaken | Reward holding course and speed |
| Late or insufficient action | give-way classes | Rule 8 |

**Removed in Revision 2:** the stand-on release term. Active release under Rule 17(a)(ii)
is out of scope (S5). Only 17(a)(i) passive course-keeping survives, as the
course-keeping hold term above.

**Use yaw rate, not rudder angle, as the turn criterion.** Ship dynamics delay the sign
change in yaw rate by several timesteps after a rudder reversal, so rudder angle is a poor
proxy for whether the vessel is actually turning — more so for an underactuated model
vessel than for a full-scale ship.

### 4.3 Rule 8 — early and substantial

Rule 8(b) requires alterations large enough to be readily apparent and that a succession of
small alterations be avoided. Encode as:

- Penalty on late first action, measured against TCPA at action onset
- Penalty on small first action magnitude
- **Check the tension:** the existing action-smoothness term discourages oscillation, but
  must not also suppress the large single alteration Rule 8 requires. These pull in
  opposite directions and the balance needs explicit verification

### 4.4 Speed as a legal action

Rule 8(e) permits slackening speed or stopping. Paper 2's curriculum restricted the
propulsion range. **If the agent cannot slow down, a lawful manoeuvre has been removed**
and a reviewer may notice. Widen propulsion authority, or justify the restriction
explicitly.

This matters more in a narrow channel than in open water, because speed reduction is often
the *only* admissible action when there is insufficient room for a course alteration —
which is itself an argument the precedence table should make.

---

## 5. Magnitude hierarchy

Enforced by design, not emergent from tuning:

```
collision  ≫  border  ≫  COLREGs violation  ≫  path following  ≫  smoothness
```

A COLREGs-compliant collision is worse than a non-compliant near-miss.

---

## 6. Scale audit protocol

**Mandatory. The direct lesson from Paper 2.**

The `r_oa` term was ~49× weaker than `r_pf` at contact distance, making the λ weighting
effectively 98/2 rather than the stated 50/50 — masked entirely by the weighting framing
and invisible until audited.

1. Normalise every term to a known, stated range before weighting
2. Instrument the environment to log **per-term episode-integrated contribution**, not
   instantaneous values
3. Run a random policy and the Paper 2 SAC baseline through the new reward; tabulate
   effective contribution per term
4. Verify the empirical ordering matches §5. If not, the coefficients are wrong regardless
   of what the ratios say on paper
5. Repeat after any coefficient change

Report the table in the paper (Table R7 in the draft skeleton).

---

## 7. Ablation design

**Not optional.** Paper 2 conceded the component-wise ablation because each variant would
have required a separate multi-seed campaign. The two-vessel cut roughly halves compute,
which is what makes this affordable now.

Primary 2 × 2:

|  | Encounter feature OFF | Encounter feature ON |
|---|---|---|
| **COLREGs terms OFF** | Avoidance only | Told, not rewarded |
| **COLREGs terms ON** | Learned from kinematics | Full method |

Leave-one-out on individual COLREGs terms, five seeds each.

**Framing note.** An explicit class feature means part of the behaviour is given rather
than learned. Get ahead of it: the classifier supplies the rule *regime*, the policy learns
the *manoeuvre*, and the 2 × 2 quantifies the split. This is also closer to how a
watchkeeper operates.

---

## 8. Open items

- **Produce the precedence table (§3.2)** — blocking for 01 and 04
- Define the geometric threshold for "narrow" per encounter class
- Re-verify the six carried-over terms against the new observation, especially the border
  penalty
- Decide whether propulsion authority widens (§4.4)
- Check the Rule 8 / smoothness tension explicitly (§4.3)
- Set per-term normalisation ranges before any coefficient tuning
