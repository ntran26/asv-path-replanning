# 02 — Reward Function and COLREGs Encoding

**Handover target:** Claude chat (design), then Claude Code (implementation)
**Depends on:** 01 — the encounter classifier and CRI are inputs here
**Carry in:** `REWARD_REDESIGN.md` (the existing 6-term specification)

---

## 1. Purpose

Extend the 6-term reward redesign to cover dynamic targets and COLREGs Rules 9 and
13–17, without recreating the scale failure found in Paper 2.

**Confirmed (D2):** COLREGs compliance is enforced through **reward terms plus an
explicit encounter-role feature** in the observation — not by reward shaping alone, and
not by a separate arbitration or safety-filter layer.

---

## 2. Why not pure shaping, and why not a separate layer

Pure continuous shaping is fragile because the rule is **discrete and conditional**:
which rule applies depends on relative bearing and heading-intersection sectors, so a
smooth reward cannot express a role switch cleanly.

A separate arbitration or safety-filter layer would give hard guarantees but breaks the
end-to-end framing that is the core of the contribution, and moves the interesting
behaviour out of the learned policy.

**The chosen middle path:** classify the encounter deterministically each step (module
from 01), then apply a **class-conditional** penalty. Still end-to-end, but the
discontinuity lives in a classifier under your control rather than in learned features.

Precedent: Waltz & Okhrin include the encounter situation σ directly in the observation
vector *and* condition their COLREGs reward on it. Useful cover when a reviewer asks
whether the agent is being handed the answer.

---

## 3. Rule precedence — Rule 9 vs Rules 13–17

**This is a liability if ignored and a genuine contribution if handled properly.**

In a narrow channel, Rule 9 modifies the others:

- 9(a) — keep to the starboard side of the fairway
- 9(b),(d) — do not impede vessels that can navigate only within the channel
- 9(e) — special overtaking provisions with sound signals

Applying Rule 14's "both alter to starboard" in a channel too narrow to permit it will
be caught by any reviewer who knows COLREGs. Equally, Rule 15 crossing geometry is
partly inapplicable where there is no room to cross.

**Deliverable: an explicit precedence table** stating, for each encounter class × channel
width condition, which rule governs and what action is expected. This table then defines
both the reward terms and the evaluation criteria in 04. Most COLREGs DRL work targets
open water and has nothing to say here — this is where the narrow-waterway positioning
earns its keep.

---

## 4. Reward structure

### 4.1 Carried from `REWARD_REDESIGN.md`

The existing six terms (re-verify each against the new observation, since `c_t` no
longer contains borders):

1. Exponential clearance-based collision-avoidance term
2. Unified path-following term
3. Border penalty
4. Progress term
5. Action-smoothness penalty
6. Existence cost

Terminal: collision −200, goal +100, timeout via value bootstrapping.

**Note:** the border penalty now draws on the boundary branch and the geometric
boundary, not on LiDAR returns. Confirm the term still behaves as designed.

### 4.2 New COLREGs terms

Class-conditional, gated on the classifier output:

| Term | Condition | Intent |
|---|---|---|
| Wrong-side passing | any class with a defined passing side | Penalise passing on the incorrect side |
| Port turn in head-on | class = head-on, TCPA > 0 | Penalise altering to port |
| Bow crossing | give-way classes | Penalise crossing ahead of the target |
| Stand-on hold | stand-on classes, TCPA above threshold | Reward holding course and speed |
| Stand-on release | stand-on, give-way vessel has not acted | Permit / reward evasive action |
| Late or small action | give-way classes | Penalise Rule 8 violations |

**Use yaw rate, not rudder angle, as the turn criterion.** Waltz & Okhrin make this
point explicitly: ship dynamics introduce a delay of several timesteps between a sign
change in rudder angle and a sign change in yaw rate, so rudder angle is a poor proxy
for whether the vessel is actually turning. This matters more for an underactuated model
vessel than for their tanker.

### 4.3 Rule 17 — the hard term and the strongest novelty

Two regimes:

- **Regime A (hold):** while stand-on and TCPA above threshold, reward holding course
  and speed. Penalise gratuitous manoeuvring.
- **Regime B (release):** once the give-way vessel has demonstrably failed to act,
  permit and then reward evasive action.

The contribution is making the **release condition principled rather than hand-tuned**.
Candidate triggers: CRI crossing a threshold; TCPA below a value scaled to stopping and
turning distance; observed absence of give-way action over a window. Whichever is
chosen, it must be justified against Rule 17(a)(ii) and 17(b), and it must be
*measurable* so that release timing becomes an evaluation metric.

This behaviour is nearly absent from the DRL literature — Waltz & Okhrin assume linear
deterministic targets and cover only give-way cases. It cannot be exercised at all
without the non-compliant target stratum specified in 03 and 04.

### 4.4 Rule 8 — early and substantial

Rule 8(b) requires alterations large enough to be readily apparent, and that a
succession of small alterations be avoided. Encode as:

- Penalty on late first action (measured against TCPA at action onset)
- Penalty on small first action magnitude
- The existing action-smoothness term already discourages oscillation — verify it does
  not *also* suppress the large single alteration Rule 8 requires. These pull in
  opposite directions and the balance needs checking explicitly.

### 4.5 Speed as a legal action

Rule 8(e) permits slackening speed or stopping. Paper 2's curriculum restricted the
propulsion range. **If the agent cannot slow down, a legal manoeuvre has been removed**
and reviewers may notice. Widen propulsion authority for Paper 3, or justify the
restriction explicitly.

---

## 5. Magnitude hierarchy

Strict ordering, enforced by design rather than emerging from tuning:

```
collision  ≫  border  ≫  COLREGs violation  ≫  path following  ≫  smoothness
```

COLREGs terms must never outweigh collision avoidance. A COLREGs-compliant collision is
worse than a non-compliant near-miss.

---

## 6. Scale audit protocol

**Mandatory. This is the direct lesson from Paper 2.**

The Paper 2 `r_oa` term was ~49× weaker than `r_pf` at contact distance, making the λ
weighting effectively 98/2 rather than the stated 50/50 — masked entirely by the
weighting framing, and invisible until audited.

Procedure:

1. Normalise every term to a known, stated range before weighting.
2. Instrument the environment to log **per-term episode-integrated contribution**, not
   just instantaneous values.
3. Run a random policy and the Paper 2 SAC baseline through the new reward; tabulate
   effective contribution per term.
4. Verify the empirical ordering matches §5. If it does not, the coefficients are wrong
   regardless of what the ratios say on paper.
5. Repeat the audit after any coefficient change.

Report the table in the paper. Paper 2's response to Reviewer 5.3 already committed to
consolidating reward settings; going further with an empirical contribution table is
cheap and pre-empts the question.

---

## 7. Ablation design

**Not optional.** Paper 2 conceded the component-wise ablation to Reviewers 1.3 and 2.2
because each variant would have required a separate multi-seed campaign. Budget for it
this time.

Primary 2 × 2 (from D2):

|  | Role feature OFF | Role feature ON |
|---|---|---|
| **COLREGs terms OFF** | Avoidance only | Told, not rewarded |
| **COLREGs terms ON** | Learned from kinematics | Full method |

This answers the question a sceptical reviewer will actually ask: is compliance learned
or handed over? Design the campaign around it from the start.

Leave-one-out on individual COLREGs terms, five seeds each.

**Framing note.** An explicit role feature means part of the COLREGs behaviour is given
rather than learned. Get ahead of it: the classifier supplies the rule *regime*, the
policy learns the *manoeuvre*, and the 2 × 2 quantifies the split. This is also closer
to how a watchkeeper actually operates.

---

## 8. Open items

- **O2** — give-way-only vs full give-way/stand-on reciprocity. Note that full
  reciprocity is what makes Rule 17 meaningful, so this is effectively decided by the
  Rule 17 contribution, but confirm.
- Produce the Rule 9 / Rules 13–17 precedence table (§3) — this gates the reward terms.
- Define the Rule 17 release condition and its justification.
- Re-verify the six carried-over terms against the new observation, especially the
  border penalty.
- Decide whether propulsion authority widens (§4.5).
- Check the Rule 8 / smoothness tension explicitly.
