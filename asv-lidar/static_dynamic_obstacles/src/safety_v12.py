"""Experimental V11 with exact-source existing-track priority.

Persistent hypotheses act as fallback representations when the base snapshot
has lost their raw source ID. An existing base view may itself be coasting or
retained after dynamic demotion; its presence does not establish freshness or
superior accuracy. This isolated base-preservation ablation can suppress a more
accurate persistent prediction. While that exact ID remains represented, retain
the base view and suppress its synthetic alternative. The underlying measured
anchors remain available if the ID subsequently disappears.

See safety_track_fallback for the engineering rule and limitations, and
safety_track_persistence for the inherited method references. This does not
identify reappearing objects under different IDs, establish a perception bound,
or guarantee collision avoidance. All V11 policy/target-ensemble options and
existing geometric checks are inherited without modification.
"""
from __future__ import annotations

import safety_v11 as v11
from safety_track_fallback import TrackFallbackPerception


EXISTING_TRACK_PRIORITY = True


class SafetyFilterV12(v11.SafetyFilterV11):
    def __init__(self, *, existing_track_priority=None, **v11_options):
        super().__init__(**v11_options)
        self.existing_track_priority = (EXISTING_TRACK_PRIORITY if existing_track_priority is None
                                    else bool(existing_track_priority))
        if self.track_persistence is not None:
            self.track_persistence = TrackFallbackPerception(
                self.track_persistence, existing_track_priority=self.existing_track_priority)
            self.perception = self.track_persistence

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last["v12_existing_track_priority"] = self.existing_track_priority
        return result
