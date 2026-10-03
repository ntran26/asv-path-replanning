"""Experimental V13 with exact-source persistent-estimate preference.

A saved-state diagnostic found that an existing centroid-based target view
rejected a recorded successful SAC continuation, while its measured full-motion
anchor with the partial-shape prior passed the same checks. This ablation uses
the persistent view whenever that exact source ID has a valid anchor, regardless
of whether the base or independent motion rule represented it first.

Related architectural inspiration: Granstrom, Baum & Reuter,
https://arxiv.org/abs/1604.00970 (extended-object shape/measurement modelling),
and Nuss et al., https://arxiv.org/abs/1605.02406 (dynamic prediction/update).
This deterministic estimate preference is an engineering heuristic, not either
paper's probabilistic algorithm, an accuracy ordering, or a safety guarantee.
A coasting anchor can be wrong when a target changes velocity or stops.

All original admission, bounded age and finite-ray contradiction safeguards
remain. No spatial association across IDs, hidden truth, scenario/outcome input,
or static-return deletion is added. ``prefer_persistent_estimates=False``
delegates to V13 unchanged. Other V13 options remain independently configurable.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import safety_v13 as v13
from safety_consistent_tracks import ConsistentTrackPerception
from safety_track_persistence import TrackPersistencePerception


PREFER_PERSISTENT_ESTIMATES = True


class PreferredPersistentPerception(ConsistentTrackPerception):
    """Choose one existing persistent view per source, without changing anchors."""

    def __init__(self, base_perception, *, prefer_persistent_estimates=True, **options):
        super().__init__(base_perception, **options)
        self.prefer_persistent_estimates = bool(prefer_persistent_estimates)

    def snapshot(self, env):
        if not self.prefer_persistent_estimates:
            # Keep V13's birth-ownership policy, values and ordering exactly.
            snap = super().snapshot(env)
            self.last_track_persistence_stats["prefer_persistent_estimates"] = False
            return snap

        # Reuse inherited admission/coasting/clearing. Dynamic dispatch still
        # calls ConsistentTrackPerception._motion for its optional shape prior;
        # only V13's subsequent birth-based selection is replaced here.
        snap = TrackPersistencePerception.snapshot(self, env)
        stats = copy.deepcopy(self.last_track_persistence_stats)
        generated = stats.get("hypotheses", [])
        synthetic_ids = {int(h["id"]) for h in generated}
        persistent_sources = {int(h["source_id"]) for h in generated}
        base_ids = {int(view.id) for view in snap.tracks if int(view.id) not in synthetic_ids}
        replaced_ids = persistent_sources & base_ids
        # Preserve first-admission provenance for diagnostics and opt-out state,
        # but do not use it to choose which estimate is published in this mode.
        for event in stats.get("updates", []):
            if event["kind"] == "admission":
                self._source_owners[int(event["source_id"])] = (
                    "base" if event["base_admitted"] else "persistent")
        self._source_owners = {key: value for key, value in self._source_owners.items()
                               if key in self._anchors}
        stats.update(
            geometry_prior=self.geometry_prior, source_ownership=self.source_ownership,
            prefer_persistent_estimates=True, selection_rule="persistent_when_valid",
            first_admission_owners=dict(self._source_owners),
            source_owners={source: "persistent" for source in sorted(persistent_sources)},
            base_track_ids=sorted(base_ids), generated_hypotheses=len(generated),
            suppressed_hypotheses=[], suppressed_source_ids=[],
            replaced_base_source_ids=sorted(replaced_ids),
            replaced_base_view_count=sum(int(view.id) in replaced_ids for view in snap.tracks),
            published_source_ids=sorted(persistent_sources),
            published_persistent_count=len(generated),
        )
        self.last_track_persistence_stats = stats
        if not replaced_ids:
            return snap
        return replace(snap, tracks=[view for view in snap.tracks
                                     if int(view.id) not in replaced_ids])


class SafetyFilterV14(v13.SafetyFilterV13):
    def __init__(self, *, prefer_persistent_estimates=None, **v13_options):
        super().__init__(**v13_options)
        self.prefer_persistent_estimates = (
            PREFER_PERSISTENT_ESTIMATES if prefer_persistent_estimates is None
            else bool(prefer_persistent_estimates))
        previous = self.track_persistence
        if previous is not None:
            self.track_persistence = PreferredPersistentPerception(
                previous.base_perception,
                prefer_persistent_estimates=self.prefer_persistent_estimates,
                geometry_prior=previous.geometry_prior,
                source_ownership=previous.source_ownership,
                enable_admission=previous.enable_admission,
                enable_persistence=previous.enable_persistence,
                max_coast_s=previous.max_coast_s,
                min_motion_observations=previous.min_motion_observations,
            )
            self.perception = self.track_persistence

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last["v14_prefer_persistent_estimates"] = self.prefer_persistent_estimates
        return result
