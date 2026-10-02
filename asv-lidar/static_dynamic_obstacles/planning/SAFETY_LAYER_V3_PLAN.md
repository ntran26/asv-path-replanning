# Safety layer v3 — committed backup and recovery mode (plan, 2026-10-02)

**Status:** first version built (`src/safety_v3.py`, run-time `SAFETY_VERSION = 3`, wired into
`env.step`, `frozen_suite.py` and `paper2_suite.py` via `--safety-version 3`; tests
`tests/test_safety_v3.py`). It is being developed alongside the filter-free PPO baseline-v3 run
(`results/ppo_bl3_run.sh`). That PPO policy, once frozen, is the development policy. Until then,
development uses the kept 3 M SAC policy. Earlier work: `SAFETY_LAYER_V2_PLAN.md`.

## Why v3

v2 removed obstacle contacts on the development set (26 → 10–19 over 150 episodes) but gained no
goals: it traded the contacts for wall contacts and stalls. Two causes, both traced:

1. **No retained escape.** When no new candidate passed, v2 braked, or let the policy's own action
   stand. It discarded the escape it had certified one step earlier.
2. **Hand-back in unfamiliar states.** v2 returned the helm the moment the policy's action passed
   again. That was often when the boat was slow, off the path and next to a wall. The policy was
   trained without a safety layer and rarely saw such states.

The design follows the literature summary from Claude Chat (2026-10-02): model-predictive
shielding (Bastani et al.), backup control barrier functions (Gurriet, Ames et al.), the gatekeeper
idea of keeping a verified continuation, and predictive safety filters (Wabersich & Zeilinger).
Only the parts that fit our timeline and vessel are taken.

## What v3 does

- **Committed backup.** The chosen action's whole plan is stored: the action for 1 s, then its
  escape manoeuvre (2 s of turn, then rudder centred, to 8 s). Each step the plan, advanced one
  step, is re-checked from the new state as one more candidate, with a small continuity
  preference. When nothing passes, the stored plan is followed ("last certificate"), not a blind
  brake.
- **Recovery mode.** An override starts a recovery. During it, the reference action is a
  path-rejoin action (the LOS course autopilot at cruise), not the policy's action. The filter
  finishes passing the hazard and heads back to the path. Every action must still pass the same
  check. The helm goes back to the policy only when all of these hold, or when nothing is within
  7 m:
  - the policy's action passes with 0.30 m to spare;
  - speed is at least 0.30 m/s (steerage way);
  - heading is within 30° of the path;
  - the recovery has lasted at least 2 s.
- **Kept from v2:**
  - the prediction model: the identified hull, rudder delay and servo, remembered static LiDAR
    returns, the map polygon, and the tracked target at constant velocity;
  - braking only against traffic, on the pessimistic reverse model;
  - smoothed ego state;
  - the finite-mission run-out check (a clear run along the final heading).

## Limits (state these in the paper)

- **No invariant terminal set exists.** The basin is 10 m wide. The hull's steady turning circle
  is about 10.5 m across at the propulsion floor, at cruise and at the ceiling (identified model,
  2026-10-02). There is no astern motion. So no loiter circle or stop is safe forever. The filter
  is a **finite-horizon predictive filter with a finite-mission terminal condition**, evaluated
  empirically. It is not a certified one.
- **Assumed:**
  - the target holds constant velocity over the 8 s;
  - reverse thrust is unidentified (see the v2 plan's caveat);
  - perception is as the policy sees it, with no simulator truth.

## Next steps (from the literature summary, adapted)

1. **Development set:** v3 against v2 and safety off, on the kept 3 M SAC policy. Then on the
   frozen PPO v3 policy when it finishes.
2. **Base the margin on measured prediction error.** Log the filter's predicted positions against
   the realised ones, and set the margin from error quantiles. This replaces hand-set gaps.
3. **Oracle mode for debugging.** The filter sees true state and geometry, to separate logic bugs
   from perception effects.
4. **Final evaluation matrix**, on the frozen suite and the Paper 2 set (neither used for tuning):

   | ID | Controller | What it isolates |
   |---|---|---|
   | P0 | PPO alone | Nominal policy |
   | P1 | PPO + v2 | The v2 filter |
   | P2 | PPO + v3 without the recovery mode | The committed backup |
   | P3 | PPO + v3 | The recovery mode |
   | C0 | Classical controller alone (LOS-DWA / COLREGs-VO comparators) | Value of the learned policy |
   | S0 | SAC alone | SAC reference |
   | S1 | SAC + the same frozen v3 | Transfer without retuning |

   Report paired gains and losses on matched scenarios. When no collisions are observed, report
   the one-sided 95 % upper bound 1 − 0.05^(1/n).
5. **Regression suite** for quick checks: near-deployment layouts (as in the hand-back harvest),
   not the Paper 2 episodes.
6. Hand-back starts (`HANDBACK_STARTS_PLAN.md`) stay on hold until P3 is measured. If used, call
   the training "filter-informed".

## Development log (2026-10-02, kept 3 M SAC policy, 150-episode dev set)

| version | goals | obstacle | boundary | target | timeout | interventions / ep |
|---|---|---|---|---|---|---|
| safety off | 111 | 26 | 2 | 11 | 0 | – |
| v2 iteration 2 | 115 | 17 | 2 | 13 | 3 | 6.0 |
| v3 first version (rejoin reference) | 93 | 26 | 10 | 13 | 8 | 26.8 |
| v3, plans expire (max 2 s unchecked), rejoin reference | 95 | 30 | 14 | 11 | 0 | 16.9 |
| v3, plans expire, **policy reference (current default)** | **108** | 18 | 14 | 10 | 0 | 13.2 |
| … turn preference W_THROTTLE 0.5 | 101 | 18 | 20 | 11 | 0 | 14.5 |
| … no room slack | 108 | 19 | 16 | 7 | 0 | 15.9 |
| … horizon 16 s, escape turn 4 s | 98 | 17 | 22 | 13 | 0 | 21.2 |
| … horizon 16 s, escape turn 8 s | 82 | 23 | 29 | 15 | 1 | 21.3 |

What was learned:
* **Stale plans were the first version's main fault.** Following an expired, uncertified plan for 13–140
  steps drove the hull into obstacles and held brakes for whole episodes. A stored plan now lives only
  for its certified horizon, and is followed unchecked for at most 2 s.
* **Rejoining the path is the wrong recovery reference here.** The L1–L3 paths run through the panels.
* **Wall contacts (14 vs 2 for v2 iteration 2) remain.** Traces: after interventions the hull points at
  a wall, every candidate then fails, and the policy acts ("no escape"). A longer horizon did not help;
  it made the filter intervene more, and boundary contacts rose. In the traced cases the policy *nearly*
  escaped a state the filter judged hopeless. That suggests the prediction (inflated hull, gaps, the
  identified model against the simulator's dynamics) may be **pessimistic**, not too short-sighted.
* **Next, before any more parameter changes:** measure the filter's prediction error against what
  actually happens (step 2 above), and run an oracle-mode check (step 3). Tuning without knowing the
  model error has gone in circles.
