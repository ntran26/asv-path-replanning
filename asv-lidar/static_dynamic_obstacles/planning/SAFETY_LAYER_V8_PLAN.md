# Safety layer v8 track: trigger counterfactuals (2026-10-03)

**Question (user):** how to trigger the safety layer so that it does not fire when the policy
would solve the case without it.

**Status:**
- Phase 1 done: 429 episodes, 819 replays.
- Phase 2 (the remaining 721 test-set-v2 successes) running.
- Code:
  - `tools/diagnostics/safety/trigger_counterfactual.py` (shadow runs and replays)
  - `tools/diagnostics/safety/trigger_analysis.py` (analysis)
- Data: `results/safety_dev/trigger_counterfactual/`.
- Runs were authorised by the user for this plan ("continue improving the safety layer in this
  thread as planned"). The development set and test set v2 are development evidence for the
  safety layer (the user's designation).

## Method

- **Shadow mode.** SAC (kept 3 M) drives alone. Each step, copies of v4 (selected) and v7 decide
  on the policy's action, and the policy's action is executed regardless. The shadow filter then
  advances as if it had passed: perception and observer see the real trajectory; the actuator
  model is rewound and advanced with the executed command; any recovery it started is released.
- **Replays.** At the first would-fire step, and every 4th after it (at most 4; phase 2: the first
  only), the environment and filter are deep-copied and the episode is replayed with the filter
  acting. The copy takes about 4 ms and steps identically.
- **Key property.** Before its first fire a filter only passes the policy's actions, and the shadow
  state is exact there. So the first-fire replay **is** that filter's closed-loop outcome for the
  episode. Episodes with no fire keep the policy-alone outcome.
- **Fixed during this work:** the four perception wrappers (`safety_provisional_tracks`,
  `safety_track_history`, `safety_perception`, `safety_hull_perception`) recursed forever when
  deep-copied. Their delegating `__getattr__` now raises `AttributeError` until the delegate is
  set. Behaviour is unchanged, and 118 tests pass.

## Phase 1 results (429 episodes: DV3 150, test-set-v2 failures 128, matched controls 151)

**Trigger statistics** (episode level; a fire is unnecessary if the policy alone reaches the goal):

| | v4 | v7 |
|---|---|---|
| False-alarm rate: fires in episodes the policy alone solves | 72.5 % | **60.7 %** |
| Precision: P(policy fails, given a fire) | 0.44 | 0.49 |
| Recall: P(fire, given the policy fails) | 0.88 | 0.91 |
| First-fire replays: helpful / harmful | 68 / 23 | **79 / 13** |
| Net, current trigger / perfect (oracle) trigger | +45 / +68 | **+66 / +79** |

**Closed-loop outcomes:**

| Set | Policy alone | v4 | v7 |
|---|---|---|---|
| DV3 (150) | 111 | 123 (+19 / −7) | **128 (+21 / −4)** |
| Test-set-v2 failures (128): rescued | – | 49 | **58** (field 34/76, frozen 24/52) |
| Test-set-v2 matched controls (151): broken | – | 16 | **9** (field 3/56, frozen 6/95) |

These reproduce the safety chat's earlier DV3 numbers (v4 123), so the method is consistent with
direct evaluation.

**What it shows:**
1. **The trigger fires in most episodes the policy would solve alone** (v7: 61 %), but **most of
   those fires do no harm.** v7 breaks 4 of 75 false-alarm episodes on DV3 and 9 of 84 among the
   controls. The cost of over-triggering is about 1 in 10 false-alarm episodes, not every one.
2. **v7 is close to the oracle trigger.** A perfect trigger would add only 13 episodes (+79 against
   +66) across these 429. Better triggering is worth roughly 3 % of the failures; the filter's
   rescue capacity (58 of 128 test-set failures) matters more.
3. **A critic gate does not help at the first fire.** SAC's Q(s, π(s)) is lower in episodes the
   policy goes on to fail (median −5.2 against −2.9), but the distributions overlap. The best
   threshold, fitted on half the scenarios, gives on the other half 33 rescued / 6 broken against
   always-on 34 / 8: **+1 net, within noise.** The 2-s Q trend does not separate at all.
4. **The breaks have a pattern.** 6 of v7's 9 test-set breaks are **target collisions**, and 3 of
   them involve **non-compliant or reactive targets** (BAS-HO-NC-023, BAS-HO-NC-053, CH-CR-RE-038).
   The filter predicts targets at constant velocity, so its manoeuvre can turn into a target that
   then manoeuvres too. The other breaks are boundary or obstacle contacts after interventions in
   crossings (DV3-CRS-CV-04/12, P2-L2-CRS-FIX-19, P2-L2-CRP-VAR-16).

## Implications for the trigger question

- **Do not chase precision for its own sake.** Firing when the policy would succeed is common but
  usually harmless. What matters is the few interventions that cause a failure (13), set against
  the many that rescue one (79).
- **The breaks point to specific mechanisms**, not to over-triggering in general:
  - target-model mismatch (constant velocity against reactive or non-compliant targets);
  - interventions near walls in crossings.

  Candidate fixes:
  - (a) propagate a set of target behaviours (constant velocity, plus Rule 17-style give-way and
    stand-on manoeuvres) in the target check;
  - (b) a boundary-run-out check for interventions in crossings;
  - (c) withhold an intervention whose own predicted margin is not better than the policy's
    predicted margin (an "improvement" test).
- **The oracle bound limits the payoff.** Even a perfect trigger adds at most 13 on these 429
  episodes. More reach (rescuing more of the 70 test-set failures v7 cannot rescue) is the larger
  lever.

## Phase 2 result: closed loop on all 1,000 test-set-v2 episodes (2026-10-03, 08:25)

| Test set v2 | Success | Rescued | Broken | Fires in SAC successes / failures |
|---|---|---|---|---|
| SAC alone | 0.872 | – | – | – |
| SAC + v4 | 0.880 | 49 | 41 | 48 % / 88 % |
| **SAC + v7** | **0.904** | **58** | **26** | **37 % / 91 %** |

| Source | SAC alone | SAC + v4 | SAC + v7 |
|---|---|---|---|
| Frozen part (761) | 0.932 | 0.928 | 0.936 |
| Paper 2 part (239) | 0.682 | 0.728 | **0.803** |

- **Outcomes with v7:** 904 goals, 38 target, 31 boundary and 27 obstacle contacts. Among the 721
  successes outside the controls, v7 fires in 238 and breaks 17.
- **Weakest cells with v7:**

  | Cell | SAC alone | SAC + v7 |
  |---|---|---|
  | L2-HO | 0.00 | 0.27 |
  | L1-CRS | 0.44 | 0.50 |
  | L1-CRP | 0.41 | 0.65 |
  | L2-CRS | 0.47 | 0.76 |
  | L3-CRP | 0.58 | 0.79 |
  | L2-CRP | 0.60 | 0.80 |

- **The ceilings:**
  - **Perfect trigger:** 872 + 58 = 930 (0.930). Better triggering can add at most 2.6 points.
  - **Full rescue reach:** 70 failures stay unrescued. Better rescue is the larger lever.
- **Caveat:** test set v2 is safety-layer development evidence (the user's designation), so this
  is not an untouched test. No v7 setting was tuned on these runs: v7 was designed from saved
  summaries, and this is its first evaluation on them.

## v8 = v7 without hold-back (2026-10-03, 10:30)

**First-fire reasons, v7:**

| Reason | Rescued | Broken | Net |
|---|---|---|---|
| turn | 39 | 12 | +27 |
| brake | 15 | 4 | +11 |
| continue | 13 | 2 | +11 |
| searched escape | 5 | 0 | +5 |
| last certificate | 4 | 4 | 0 |
| **hold back** | 3 | 8 | **−5** |

Hold back is the filter acting when no candidate passes, picking whatever delays contact by at least
1 s. It is the only reason with a negative balance.

**Variant run:** `safety_v2.HOLD_BACK_GAIN_S = inf`, on the 549 episodes where v7 fired; the rest
cannot change. Results in `results/safety_dev/trigger_counterfactual/v7_nohold/`.

| Closed loop | SAC alone | v7 | **v7 without hold-back (= v8)** |
|---|---|---|---|
| Test set v2 (1,000) | 0.872 | 0.904 (+58 / −26) | **0.912 (+57 / −17)** |
| DV3 (150) | 0.740 | 0.853 (+21 / −4) | 0.847 (+20 / −4) |

- 15 episodes change between v7 and v8. Nine broken successes are fixed, all of them hold-back
  target contacts (BAS-HO-NC-023/053/080, BAS-HO-RE-099, BAS-CR-CV-047/063/075, BAS-HO-CV-064,
  P2-L3-CRP-FIX-09).
- Three rescues are lost: DV3-NT-CV-04, BAS-HO-RE-081, P2-L2-HO-FIX-11. One is gained:
  P2-L1-CRS-FIX-07.

**Implemented** as `src/safety_v8.py`:
- `SafetyFilterV8(SafetyFilterV7)` disables hold-back for the duration of each decision, so v2–v7
  are unchanged.
- Selected with `SAFETY_VERSION = 8`; `frozen_suite.py` and `paper2_suite.py` now accept versions
  6–8.
- Test `tests/test_safety_v8.py`: selected, never holds back, shared constant restored. v7 tests
  still pass.

**The rule in plain words:** with no certified escape, the policy's action stands. That is the
direct answer to "fire only when justified" that this data supports.

## Next steps

1. **Phase 2:** v4 and v7 on the 721 test-set-v2 successes not yet run (first-fire replay only).
   With phase 1, this gives the **closed-loop v7 result on all 1,000 test-set-v2 episodes**,
   against SAC alone at 0.872.
2. Look into v7's 9–13 breaks one by one (traces exist in `steps.csv`; replays can be re-run from
   saved states).
3. Prototype fix (a) or (c), then re-measure on the same 429 + 721 with the same method.
