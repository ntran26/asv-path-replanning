"""Safety layer v8: v7 with the hold-back fallback disabled (2026-10-03).

`planning/archive/safety/SAFETY_LAYER_V8_PLAN.md`.  Every filter since v2 has a **hold-back**
fallback: when no candidate passes the check, it still replaces the policy's
action with whichever candidate delays the first predicted contact by at least
`HOLD_BACK_GAIN_S`.  The trigger counterfactuals (shadow runs plus replays, test
set v2 and the DV3 development set) measured what that buys:

* at v7's first fire, hold-back rescued 3 episodes and broke 8 (all target
  contacts) -- the only first-fire reason with a negative balance;
* closed loop with hold-back disabled: test set v2 912/1000 (v7 904; SAC alone
  872), 57 rescued / 17 broken (v7 58 / 26); DV3 127/150 (v7 128).

V8 disables only hold-back. The inherited "last certificate" branch can still
execute a currently rejected continuation for up to four decisions if V7's
policy-prefix repair fails. Thus V8 does not require every override to have a
currently feasible backup, and passing the sampled model checks is not a
formal safety certificate. Everything else is V7's, including its predictive
filter method adaptation and limitations documented in safety_v7.py.

An instance-method override isolates this ablation without changing shared
module constants, including during nested or concurrent V2-V7 decisions.

Selected by `constants.SAFETY_VERSION = 8` at run time.
"""
from __future__ import annotations

import math

import safety_v7 as v7


class SafetyFilterV8(v7.SafetyFilterV7):
    """V7 with the hold-back fallback disabled."""

    def _hold_back_gain_s(self):
        return math.inf
