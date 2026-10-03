"""Experimental partial-hull continuity and exact-source representation ownership.

For an already admitted motion anchor, a partial observed hull axis constrains
the possible centre to [observed_max-size/2, observed_min+size/2]. Project the
anchor's constant-velocity centre prediction into that interval. A substantially
full observed axis uses its midpoint. Admission, measured velocity, orientation,
extent tolerances, finite-ray clearing and maximum age remain inherited.

Related prior work: Granstrom, Baum & Reuter, "Extended Object Tracking:
Introduction, Overview and Applications", https://arxiv.org/abs/1604.00970
(shape and spatial measurement modelling); Nuss et al.,
https://arxiv.org/abs/1605.02406 (dynamic prediction/measurement updates).
Our deterministic interval projection and exact-ID ownership are engineering
adaptations, not those papers' probabilistic filters or uncertainty guarantees.

A source first admitted independently of the base snapshot uses its persistent
view while that anchor survives. A source already represented at first admission
keeps the base view whenever present. This avoids publishing two alternatives
for the same ID, but does not infer which is physically correct. Existing base
views can themselves be coasting or biased; different IDs are never associated.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math

import numpy as np

import constants as cfg
from safety_track_persistence import TrackPersistencePerception


def centre_with_shape_prior(points, heading, prior_centre):
    """Project an onboard centre prior into observed known-size hull intervals.

The caller already applied the inherited finite fit and maximum-extent gates.
No centre uncertainty bound is claimed. Return a detached centre and diagnostics.
    """
    forward = np.array([math.sin(heading), math.cos(heading)])
    axes = np.column_stack((forward, [forward[1], -forward[0]]))
    projected = np.asarray(points, dtype=float) @ axes
    low, high = projected.min(axis=0), projected.max(axis=0)
    extent = high - low
    size = np.array([cfg.LOA, cfg.BREADTH], dtype=float)
    prior = np.asarray(prior_centre, dtype=float) @ axes
    lower, upper = high - size / 2, low + size / 2
    full = extent >= size - float(cfg.TRACK_FIT_FULL_EXTENT_TOL_M)
    # A tolerated extent slightly larger than the physical dimension has an
    # inverted interval. It is already a full axis, so use the midpoint there.
    corrected = np.where(full, (low + high) / 2, np.clip(prior, lower, upper))
    centre = axes @ corrected
    return centre, {
        "prior_centre": np.asarray(prior_centre, dtype=float).tolist(),
        "corrected_centre": centre.tolist(), "observed_extents": extent.tolist(),
        "full_axes": full.tolist(), "centre_interval_lower": lower.tolist(),
        "centre_interval_upper": upper.tolist(),
    }


class ConsistentTrackPerception(TrackPersistencePerception):
    """Independent switches for shape-prior correction and source ownership.

Both switches False preserve V11's snapshot values and list ordering. Ownership
is fixed at the anchor's first admission, ages with that anchor, and is forgotten
when it expires or is contradicted. No sensor/tracker/base snapshot is modified.
    """

    def __init__(self, base_perception, *, geometry_prior=True, source_ownership=True,
                 **persistence_options):
        super().__init__(base_perception, **persistence_options)
        self.geometry_prior = bool(geometry_prior)
        self.source_ownership = bool(source_ownership)
        self._source_owners = {}

    def _motion(self, track):
        motion, reason = super()._motion(track)
        if motion is None or not self.geometry_prior:
            return motion, reason
        anchor = self._anchors.get(int(track.id))
        if anchor is None:
            return motion, reason
        age_s = (self._frame - anchor.frame) * float(cfg.UPDATE_RATE)
        # An anchor outside its existing coast budget is not a geometry prior.
        if age_s > self.max_coast_s:
            return motion, reason
        centre, velocity, heading, evidence = motion
        prior_centre = anchor.position + age_s * anchor.velocity
        points = np.asarray(track.history[-1][1], dtype=float)
        corrected, geometry = centre_with_shape_prior(points, heading, prior_centre)
        geometry.update(uncorrected_centre=centre.tolist(), anchor_age_s=age_s)
        return (corrected, velocity, heading, dict(evidence, geometry_prior=geometry)), None

    def snapshot(self, env):
        snap = super().snapshot(env)
        stats = copy.deepcopy(self.last_track_persistence_stats)
        generated = stats.get("hypotheses", [])
        synthetic_ids = {int(h["id"]) for h in generated}
        base_ids = {int(view.id) for view in snap.tracks if int(view.id) not in synthetic_ids}
        for event in stats.get("updates", []):
            source_id = int(event["source_id"])
            if event["kind"] == "admission":
                self._source_owners[source_id] = "base" if event["base_admitted"] else "persistent"
        self._source_owners = {key: value for key, value in self._source_owners.items()
                               if key in self._anchors}
        suppressed, replaced_ids = [], set()
        if self.source_ownership:
            for hypothesis in generated:
                source_id = int(hypothesis["source_id"])
                if source_id not in base_ids:
                    continue
                owner = self._source_owners[source_id]
                if owner == "base":
                    suppressed.append(hypothesis)
                else:
                    replaced_ids.add(source_id)
        suppressed_ids = {int(h["id"]) for h in suppressed}
        published = [h for h in generated if int(h["id"]) not in suppressed_ids]
        stats.update(
            geometry_prior=self.geometry_prior, source_ownership=self.source_ownership,
            source_owners=dict(self._source_owners), base_track_ids=sorted(base_ids),
            generated_hypotheses=len(generated), suppressed_hypotheses=suppressed,
            suppressed_source_ids=sorted({int(h["source_id"]) for h in suppressed}),
            replaced_base_source_ids=sorted(replaced_ids),
            hypotheses=published, added_hypotheses=len(published),
        )
        self.last_track_persistence_stats = stats
        if not suppressed_ids and not replaced_ids:
            return snap
        return replace(snap, tracks=[view for view in snap.tracks
                                     if int(view.id) not in suppressed_ids | replaced_ids])
