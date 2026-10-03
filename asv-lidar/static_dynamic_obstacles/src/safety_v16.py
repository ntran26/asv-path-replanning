"""Experimental V15 with independently selectable motion-axis hull geometry.

See safety_motion_axis.py for measured-evidence gates, citations and limitations.
``motion_axis_geometry=False`` keeps V15's perception and action behavior.
The adapter never alters V14's existing persistent views or static points.
"""
from __future__ import annotations

import copy

import safety_v15 as v15
from safety_motion_axis import MotionAxisPerception


MOTION_AXIS_GEOMETRY = True


class SafetyFilterV16(v15.SafetyFilterV15):
    def __init__(self, *, motion_axis_geometry=None, **v15_options):
        super().__init__(**v15_options)
        self.motion_axis_geometry = (MOTION_AXIS_GEOMETRY if motion_axis_geometry is None
                                     else bool(motion_axis_geometry))
        self.motion_axis_perception = None
        if self.motion_axis_geometry:
            self.motion_axis_perception = MotionAxisPerception(self.perception)
            self.perception = self.motion_axis_perception
        # Keep self.track_persistence pointing to the inner V14 adapter: inherited
        # filter diagnostics read it directly after the outer snapshot returns.

    def _filter(self, env, action):
        result = super()._filter(env, action)
        self.last["v16_motion_axis_geometry"] = self.motion_axis_geometry
        self.last["motion_axis_geometry"] = (copy.deepcopy(self.motion_axis_perception.last_motion_axis_stats)
            if self.motion_axis_perception is not None else {})
        return result
