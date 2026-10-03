"""Use persistent target hypotheses only when their source ID is absent.

The underlying safety_track_persistence adapter still observes, refreshes and
ages its measured anchors. This wrapper only selects which hypotheses reach
the safety checks: an existing base view of the same raw track ID takes
priority over its synthetic alternative. It neither spatially associates
different IDs nor changes the tracker, policy, static points, or age limits.

Related architecture: Nuss et al., arXiv:1605.02406, dynamic occupancy
prediction/measurement updates; see safety_track_persistence for the full
extent and finite-ray method references. Exact-ID existing-view priority is a
separate engineering integration rule, not that paper's Bayesian update or
a guarantee that the base view is the more accurate prediction. An existing
base view need not be fresh, currently published as dynamic, or measured in this
frame: ProvisionalTrackPerception can retain a formerly dynamic raw ID after
demotion and carry its biased centroid state. The persistent hypothesis can be
more accurate. This is an isolated base-preservation ablation, not a quality
ranking, and it can discard a better prediction solely because IDs match.
"""
from __future__ import annotations

import copy
from dataclasses import replace


class TrackFallbackPerception:
    """Wrap TrackPersistencePerception without deleting its stored anchors.

With ``existing_track_priority=False`` the underlying snapshot is returned
unchanged. With priority enabled only synthetic views whose *exact* source ID
is already in the base snapshot are suppressed. Different-ID reacquisitions
remain independent hypotheses; there is no added association threshold.

``hypotheses`` and ``added_hypotheses`` in the copied diagnostics describe
published additions. ``updates`` still describes actual anchor measurements;
suppression does not erase or renew the underlying anchor.
    """

    def __init__(self, base_perception, *, existing_track_priority=True):
        self.base_perception = base_perception
        self.existing_track_priority = bool(existing_track_priority)
        self.last_track_persistence_stats = {}

    def __getattr__(self, name):
        delegate = self.__dict__.get("base_perception")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def snapshot(self, env):
        snap = self.base_perception.snapshot(env)
        stats = copy.deepcopy(self.base_perception.last_track_persistence_stats)
        generated = stats.get("hypotheses", [])
        synthetic_ids = {int(h["id"]) for h in generated}
        base_ids = {int(view.id) for view in snap.tracks if int(view.id) not in synthetic_ids}
        suppressed = ([h for h in generated if int(h["source_id"]) in base_ids]
                      if self.existing_track_priority else [])
        suppressed_ids = {int(h["id"]) for h in suppressed}
        published = [h for h in generated if int(h["id"]) not in suppressed_ids]
        stats.update(
            existing_track_priority=self.existing_track_priority,
            base_track_ids=sorted(base_ids),
            generated_hypotheses=len(generated),
            suppressed_hypotheses=suppressed,
            suppressed_source_ids=sorted({int(h["source_id"]) for h in suppressed}),
            hypotheses=published,
            added_hypotheses=len(published),
        )
        self.last_track_persistence_stats = stats
        if not suppressed_ids:
            return snap
        return replace(snap, tracks=[view for view in snap.tracks
                                     if int(view.id) not in suppressed_ids])
