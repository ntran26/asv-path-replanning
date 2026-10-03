"""Experimental V14 with conditional, checked current-policy prefix search.

Extra search is restricted to a currently failing paired parent proposal, or
an ordinary passing proposal below the existing trigger margin. In the latter
case, the requested margin follows the parent's exact ordinary selection floor;
it is never negative. Already adequate parent plans retain V14's decision.

Wabersich & Zeilinger (2021), Sec. 4.1, Eq. (5a)-(5f), motivates preserving
the learning input subject to a feasible backup:
https://arxiv.org/html/1812.05506v4#S4.SS1 . The conditional admission rule here
is an engineering heuristic, not that paper's robust optimizer, terminal safe
set or theorem. The inherited finite search and nominal predictor give no
collision-avoidance, recursive-feasibility or goal-preservation guarantee.

No new numerical safety threshold is introduced. The complete inherited hard
checker still evaluates every candidate. A failed search keeps the parent's
action, plan and counters. No scenario IDs, outcomes or hidden state are read.
``conditional_prefix=False`` restores the V14 plus prefix-search request;
``policy_prefix_search=False`` retains V14's controller behavior.
"""
from __future__ import annotations

import math

import safety_v2 as v2
import safety_v14 as v14


CONDITIONAL_PREFIX = True
POLICY_PREFIX_SEARCH = True


def _number(value):
    """Accept real scalar diagnostic values, excluding booleans and strings."""
    if isinstance(value, (bool, str)) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


class SafetyFilterV15(v14.SafetyFilterV14):
    def __init__(self, *, conditional_prefix=None, policy_prefix_search=None,
                 **v14_options):
        self.conditional_prefix = (CONDITIONAL_PREFIX if conditional_prefix is None
                                   else bool(conditional_prefix))
        enabled = POLICY_PREFIX_SEARCH if policy_prefix_search is None else bool(policy_prefix_search)
        super().__init__(policy_prefix_search=enabled, **v14_options)

    def _prefix_search_request(self):
        if not self.conditional_prefix:
            return super()._prefix_search_request()
        details = self.last
        if details.get("v10_policy_preserved", False):
            return False, None, "already_policy_preserved"
        if details.get("v10_pair_checked") is not True:
            return False, None, "missing_paired_check"
        first = _number(details.get("v10_proposed_first_violation"))
        clear = _number(details.get("v10_proposed_clearance"))
        if (first is None or clear is None or math.isnan(first) or first < 0.
                or not math.isfinite(clear)):
            return False, None, "invalid_paired_scores"
        passing = math.isinf(first) and clear >= 0.
        if details.get("v10_proposed_currently_passing") is not passing:
            return False, None, "inconsistent_paired_status"
        if not passing:
            return True, 0., "currently_failing_parent"
        if clear >= v2.TRIGGER_MARGIN_M:
            return False, None, "adequate_parent_clearance"
        if details.get("selection_branch") != "ordinary":
            return False, None, "missing_ordinary_selection"
        floor = _number(details.get("selection_floor"))
        best = _number(details.get("selection_best_clearance"))
        if (floor is None or best is None or not math.isfinite(floor)
                or not math.isfinite(best) or best < 0.):
            return False, None, "invalid_ordinary_selection"
        expected = min(v2.TRIGGER_MARGIN_M, best - v2.ROOM_SLACK_M)
        # Both diagnostics originate in the same exact ordinary-pool expression;
        # reject rounded/stale alternatives instead of adding a numerical margin.
        if floor != expected:
            return False, None, "inconsistent_selection_floor"
        return True, max(0., floor), "ordinary_parent_below_trigger_margin"

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last["v15_conditional_prefix"] = self.conditional_prefix
        return result
