"""Experimental V11 with partial-shape continuity and single-source ownership.

The optional perception changes are implemented and attributed in
safety_consistent_tracks.py. Its projected geometric prior uses only previously
measured motion and the current onboard cluster; source ownership uses exact
raw track IDs and first-admission provenance. Neither mechanism uses simulator
truth or changes the shared tracker, policy observations, or static returns.

``geometry_prior=False, source_ownership=False`` preserves V11 perception and
control behavior for ablation. All V11 policy-prefix, target-ensemble and
admission/persistence settings remain independently configurable. This finite
prediction/filtering system has no general collision-avoidance guarantee.
"""
from __future__ import annotations

import safety_v11 as v11
from safety_consistent_tracks import ConsistentTrackPerception


GEOMETRY_PRIOR = True
SOURCE_OWNERSHIP = True


class SafetyFilterV13(v11.SafetyFilterV11):
    def __init__(self, *, geometry_prior=None, source_ownership=None, **v11_options):
        super().__init__(**v11_options)
        self.geometry_prior = GEOMETRY_PRIOR if geometry_prior is None else bool(geometry_prior)
        self.source_ownership = SOURCE_OWNERSHIP if source_ownership is None else bool(source_ownership)
        previous = self.track_persistence
        if previous is not None:
            self.track_persistence = ConsistentTrackPerception(
                previous.base_perception, geometry_prior=self.geometry_prior,
                source_ownership=self.source_ownership,
                enable_admission=previous.enable_admission,
                enable_persistence=previous.enable_persistence,
                max_coast_s=previous.max_coast_s,
                min_motion_observations=previous.min_motion_observations,
            )
            self.perception = self.track_persistence

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last.update(v13_geometry_prior=self.geometry_prior,
                         v13_source_ownership=self.source_ownership)
        return result
