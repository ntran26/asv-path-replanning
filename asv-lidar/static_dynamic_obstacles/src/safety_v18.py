"""Experimental V16 with bounded same-source hull-heading memory.

See safety_heading_memory.py for the observation gates, primary modelling
reference and limitations. No observer, action-selection or clearance changes
are introduced. ``hull_heading_memory=False`` retains V16's perception object
and numerical behavior. This is not V17's rejected raw-yaw ablation.
"""
from __future__ import annotations

import copy

import safety_v16 as v16
from safety_heading_memory import HeadingMemoryPerception


HULL_HEADING_MEMORY = True


class SafetyFilterV18(v16.SafetyFilterV16):
    def __init__(self, *, hull_heading_memory=None, **v16_options):
        super().__init__(**v16_options)
        self.hull_heading_memory = (HULL_HEADING_MEMORY if hull_heading_memory is None
                                    else bool(hull_heading_memory))
        self.heading_memory_perception = None
        if self.hull_heading_memory:
            self.heading_memory_perception = HeadingMemoryPerception(self.perception)
            self.perception = self.heading_memory_perception

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last["v18_hull_heading_memory"] = self.hull_heading_memory
        self.last["hull_heading_memory"] = (
            copy.deepcopy(self.heading_memory_perception.last_heading_memory_stats)
            if self.heading_memory_perception is not None else {})
        return result
