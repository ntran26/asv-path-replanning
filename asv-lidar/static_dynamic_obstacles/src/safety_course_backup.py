"""Additional edge-parallel course backup with predicted sideslip compensation.

Fossen, Pettersen and Galeazzi (2015), IEEE TCST 23(2):820--827, Sec. II.B,
Eq. (14) and its discussion: beta = atan2(v, u), course = heading + beta.
https://doi.org/10.1109/TCST.2014.2338354
Primary manuscript: https://backend.orbit.dtu.dk/ws/files/129469382/FossenPettersenGaleazzi2014.pdf

Only this kinematic compensation is used. This is not the paper's adaptive
LOS guidance law and inherits no stability or safety guarantee. In particular,
low-speed drift can violate its small, constant sideslip assumptions. This
backup follows a boundary-parallel course rather than returning to the path.

The shared predictor, existing course-controller gains and cruise propulsion
are unchanged. The caller must apply its existing collision checks and decide
whether to admit this extra sequence; generating it does not certify safety.
"""
from __future__ import annotations

import numpy as np

from classical import common as cc
import safety_feedback


COURSE_NAMES = ("edge_parallel_sideslip",)


def _sideslip_rudder(target, state):
    """Track course using only each trajectory's predicted body velocity.

    The model's compass convention gives course = heading + atan2(v, u).
    Define sideslip as zero at exactly zero velocity, including signed zeros.
    """
    surge, sway = state[0], state[1]
    beta = np.arctan2(sway, surge)
    beta = np.where((surge == 0.0) & (sway == 0.0), 0.0, beta)
    error = cc.wrap_pi(target - state[3] - beta)
    return cc.course_rudder(error, state[2])


def feedback_bank(snap, act, candidates, commit_s):
    """Return ``(rollout, sequences, names)`` for one extra course per candidate.

    ``sequences`` has shape ``(len(candidates), 1, D, 2)``. Flatten the first
    two axes to replay it with ``safety_v3.rollout_seq``. The candidate prefix
    is preserved, including NaN-throttle astern. Thereafter sideslip is
    recomputed at each predicted decision and cruise propulsion is applied.
    Snapshot, actuator history and candidate arrays are not modified.
    """
    parallel = safety_feedback.desired_courses(snap)[0]
    return safety_feedback._feedback_bank(
        snap, act, candidates, commit_s, courses=np.array([parallel]),
        names=COURSE_NAMES, rudder_feedback=_sideslip_rudder)
