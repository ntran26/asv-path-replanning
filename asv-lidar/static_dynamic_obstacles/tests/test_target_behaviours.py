"""The target behaviour models (A34, F99): what the frozen suite's reactive and
non-compliant cells rely on.

Suite 3.1 labelled the cells `re` / `nc`, names no model recognised, so every
Tier B target ran at constant velocity.  Pinned here: the labels map to models,
each model moves the way its row of 03a §5.3 says, and a manoeuvring target
stopped by the fairway edge holds its heading instead of saw-toothing.
"""

import numpy as np
import pytest

import suite
import targets as tgt

U = 0.558


def _run(behaviour, x, y, heading, own_v=(0.0, U), seconds=30.0, cls="head_on"):
    """A target against an own ship holding (0, 0) -> north at cruise."""
    t = tgt.Target(x, y, heading, U, behaviour=behaviour, encounter_class=cls, confined=False)
    ox = oy = 0.0
    headings, ranges = [], []
    v = np.asarray(own_v, dtype=float)
    for _ in range(int(seconds / 0.1)):
        own = {"x": ox, "y": oy, "velocity": v,
               "heading": float(np.degrees(np.arctan2(v[0], v[1])) % 360.0)}
        t.step(0.1, own=own)
        ox, oy = ox + 0.1 * v[0], oy + 0.1 * v[1]
        headings.append(t.heading)
        ranges.append(float(np.hypot(t.x - ox, t.y - oy)))
    return np.array(headings), min(ranges)


def test_tier_b_behaviour_cells_map_to_target_models():
    """Option (a): head-on non-compliance alters to port, elsewhere a give-way
    target stands on; reactive is T-RE everywhere."""
    assert suite.target_model("cv", "crossing") == tgt.T_CV
    assert suite.target_model("re", "head_on") == tgt.T_RE
    assert suite.target_model("nc", "head_on") == tgt.T_NC2
    for cls in ("crossing", "overtaking", "being_overtaken"):
        assert suite.target_model("nc", cls) == tgt.T_NC1
    with pytest.raises(ValueError):
        suite.target_model("re-x", "head_on")


def test_suite_cells_are_built_with_recognised_models():
    built, short = suite.build_tier_b(cells=[1, 2])        # basin head-on: re, nc
    assert not short
    assert {b.target_behaviour for b in built} == {tgt.T_RE, tgt.T_NC2}
    assert all(b.target_behaviour in tgt.BEHAVIOURS for b in built)


def test_constant_velocity_and_stand_on_targets_never_turn():
    for behaviour in (tgt.T_CV, tgt.T_NC1):
        headings, _ = _run(behaviour, 0.3, 14.0, 180.0)
        assert np.all(headings == 180.0)


def test_reactive_head_on_alters_to_starboard_and_resumes_course():
    headings, closest = _run(tgt.T_RE, 0.3, 14.0, 180.0)
    assert headings.max() > 185.0 and headings.min() >= 180.0     # starboard only
    assert headings[-1] == pytest.approx(180.0, abs=1.0)           # back on course
    assert closest > 1.0                                            # a CV target: 0.30 m


def test_reactive_give_way_crossing_turns_to_starboard_and_keeps_clear():
    # From the own ship's port side the target has the own ship on its
    # starboard bow: it gives way, to starboard, passing astern.
    headings, closest = _run(tgt.T_RE, -7.0, 7.0, 90.0, cls="crossing")
    assert headings.max() > 100.0 and headings.min() >= 90.0
    assert closest > 1.5


def test_non_compliant_head_on_alters_to_port():
    headings, _ = _run(tgt.T_NC2, 0.3, 14.0, 180.0)
    assert headings.min() < 170.0 and headings.max() <= 180.0


def test_side_rule_judged_on_the_encounter_forbids_the_port_escape():
    """A35: per candidate, a port turn that opens the pass past the side-free
    distance counts as compliant; judged on the encounter it does not."""
    import math
    from classical import colregs_vo as kvo
    p, h = np.array([0.3, 14.0]), math.radians(180.0)
    other = kvo.Target(np.array([0.0, 0.0]), np.array([0.0, U]), 0.0, kvo.HEAD_ON)
    per_candidate = kvo.select_velocity(p, h, U, h, U, [other], speed_fractions=(1.0,))
    encounter = kvo.select_velocity(p, h, U, h, U, [other], speed_fractions=(1.0,),
                                    side_live=True)
    assert math.degrees(per_candidate.course) % 360.0 < 180.0      # the comparator as tuned
    assert math.degrees(encounter.course) % 360.0 > 180.0


def test_a_manoeuvre_stopped_by_the_fairway_edge_holds_its_heading():
    """No saw-tooth: once the clamp has stopped a target that manoeuvred, it
    keeps the edge heading."""
    t = tgt.Target(0.0, 10.0, 180.0, U, behaviour=tgt.T_RE, encounter_class="head_on")
    t._reacted, t._edge_hold, t._course0, t._plan_course = True, True, 180.0, 220.0
    own = {"x": 0.3, "y": 0.0, "velocity": np.array([0.0, U]), "heading": 0.0}
    for _ in range(20):
        t.step(0.1, own=own)
    assert t.heading == 180.0
