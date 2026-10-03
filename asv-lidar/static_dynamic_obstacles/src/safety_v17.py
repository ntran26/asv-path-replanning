"""Experimental V16 with independently selectable fresh measured yaw.

This is an observer-only ablation; the inherited rollout model and all safety
checks are unchanged. See safety_yaw_observer.py for scope and attribution.
``fresh_yaw_measurement=False`` keeps V16's original observer instance and
controller behavior. Neither setting establishes collision avoidance or
preservation of policy successes.
"""
from __future__ import annotations

import copy

import safety_v16 as v16
from safety_yaw_observer import FreshYawObserver


FRESH_YAW_MEASUREMENT = True


class SafetyFilterV17(v16.SafetyFilterV16):
    def __init__(self, *, fresh_yaw_measurement=None, **v16_options):
        super().__init__(**v16_options)
        self.fresh_yaw_measurement = (FRESH_YAW_MEASUREMENT if fresh_yaw_measurement is None
                                      else bool(fresh_yaw_measurement))
        if self.fresh_yaw_measurement:
            if self.observer is None:
                raise ValueError("fresh_yaw_measurement requires the inherited model ego observer")
            # Construction precedes any measurement or prediction, so no state
            # is lost. The option does not replace an observer mid-episode.
            self.observer = FreshYawObserver()

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last["v17_fresh_yaw_measurement"] = self.fresh_yaw_measurement
        self.last["yaw_observer"] = (copy.deepcopy(self.observer.last)
                                     if self.fresh_yaw_measurement else {})
        return result
