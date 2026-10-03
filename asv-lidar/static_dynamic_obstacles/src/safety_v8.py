"""Safety layer v8: v7 that never overrides without a certified escape (2026-10-03).

`planning/SAFETY_LAYER_V8_PLAN.md`.  Every filter since v2 has a **hold-back**
fallback: when no candidate passes the check, it still replaces the policy's
action with whichever candidate delays the first predicted contact by at least
`HOLD_BACK_GAIN_S`.  The trigger counterfactuals (shadow runs plus replays, test
set v2 and the DV3 development set) measured what that buys:

* at v7's first fire, hold-back rescued 3 episodes and broke 8 (all target
  contacts) -- the only first-fire reason with a negative balance;
* closed loop with hold-back disabled: test set v2 912/1000 (v7 904; SAC alone
  872), 57 rescued / 17 broken (v7 58 / 26); DV3 127/150 (v7 128).

v8 therefore answers "fire only when it is justified" in its simplest form:
with no certified escape, the policy's action stands.  Everything else is v7's.
It is implemented by disabling hold-back for the duration of each decision, so
v2-v7 keep their behaviour.

Selected by `constants.SAFETY_VERSION = 8` at run time.
"""
from __future__ import annotations

import math

import safety_v2 as v2
import safety_v7 as v7


class SafetyFilterV8(v7.SafetyFilterV7):
    """V7 with the hold-back fallback disabled."""

    def _filter(self, env, action):
        saved = v2.HOLD_BACK_GAIN_S
        v2.HOLD_BACK_GAIN_S = math.inf          # no alternative ever "gains enough": the policy stands
        try:
            return super()._filter(env, action)
        finally:
            v2.HOLD_BACK_GAIN_S = saved
