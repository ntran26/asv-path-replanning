"""Experimental V16 with source-owned current-return transfer; see adapter citations.

This candidate is independent of V18. ``source_owned_points=False`` preserves
V16 perception and controller behavior. Thresholds, plans, checks and policy
observations remain inherited; only qualified current static points transfer
to an already published moving hypothesis.
"""
from __future__ import annotations

import copy

import safety_v16 as v16
from safety_source_points import SourcePointPerception


SOURCE_OWNED_POINTS = True


class SafetyFilterV19(v16.SafetyFilterV16):
    def __init__(self, *, source_owned_points=None, **v16_options):
        super().__init__(**v16_options)
        self.source_owned_points = (SOURCE_OWNED_POINTS if source_owned_points is None
                                    else bool(source_owned_points))
        self.source_point_perception = None
        if self.source_owned_points:
            self.source_point_perception = SourcePointPerception(self.perception, self.track_persistence)
            self.perception = self.source_point_perception

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last["v19_source_owned_points"] = self.source_owned_points
        self.last["source_points"] = (copy.deepcopy(self.source_point_perception.last_source_point_stats)
                                      if self.source_point_perception is not None else {})
        return result
