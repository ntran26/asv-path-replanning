"""Encounter classification: one case per class, plus hysteresis.

Revision 2: **five** classes.  Port and starboard crossing collapse into one
under Rule 9(b) -- the own ship gives way either way -- but the geometric side
stays available for 02's passing-side reward term.

Kickoff §7: "one case per class including 'being overtaken'; verify hysteresis
prevents chatter at sector boundaries".

Frame convention: +y north, +x east, headings compass (0 = +y, clockwise).
The own ship sits at (5, 10) heading north at cruise speed unless stated.
"""

import numpy as np
import pytest

import constants as cfg
import encounter as enc

OS_POS = (5.0, 10.0)
OS_HDG = 0.0
OS_SPD = 0.55


# ---------------------------------------------------------------------------
# One case per class
# ---------------------------------------------------------------------------
def test_head_on():
    """Target dead ahead on a reciprocal course."""
    assert enc.classify(OS_POS, OS_HDG, OS_SPD,
                        (5.0, 20.0), 180.0, 0.55) == enc.HEAD_ON


def test_crossing_from_starboard():
    """Target on the starboard bow crossing left-to-right."""
    assert enc.classify(OS_POS, OS_HDG, OS_SPD,
                        (12.0, 17.0), 270.0, 0.55) == enc.CROSSING


def test_crossing_from_port_is_the_same_class():
    """S3 / Rule 9(b): the own ship gives way regardless of approach side.

    Revision 1 split these into give-way and stand-on classes via Rule 18.
    That premise fails here -- own ship and target are similarly sized model
    vessels -- so both collapse to one class.
    """
    assert enc.classify(OS_POS, OS_HDG, OS_SPD,
                        (-2.0, 17.0), 90.0, 0.55) == enc.CROSSING


def test_crossing_side_is_still_recoverable():
    """Collapsed in the observation, but 02's passing-side term needs it."""
    assert enc.crossing_side(OS_POS, OS_HDG, (12.0, 17.0), 270.0) == enc.SIDE_STARBOARD
    assert enc.crossing_side(OS_POS, OS_HDG, (-2.0, 17.0), 90.0) == enc.SIDE_PORT
    assert enc.crossing_side(OS_POS, OS_HDG, (5.0, 20.0), 180.0) == enc.SIDE_NONE


def test_overtaking():
    """OS astern of the TS on the same course and faster: OS gives way (Rule 13)."""
    assert enc.classify(OS_POS, OS_HDG, 0.80,
                        (5.0, 16.0), 0.0, 0.30) == enc.OVERTAKING


def test_being_overtaken():
    """The fifth class, absent from the source table.

    TS astern of the OS on the same course and faster: the OS stands on and
    Rule 17 applies.  This is the geometry the whole Rule 17 contribution rests
    on, so it must classify cleanly.
    """
    assert enc.classify(OS_POS, OS_HDG, 0.30,
                        (5.0, 4.0), 0.0, 0.80) == enc.BEING_OVERTAKEN


def test_none_for_a_target_that_is_not_in_any_encounter():
    """Astern, same course, slower: opening, no encounter."""
    assert enc.classify(OS_POS, OS_HDG, 0.80,
                        (5.0, 2.0), 0.0, 0.20) == enc.NONE


def test_all_five_classes_are_reachable():
    """No class may be dead code in the observation's one-hot."""
    seen = {
        enc.classify(OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55),
        enc.classify(OS_POS, OS_HDG, OS_SPD, (12.0, 17.0), 270.0, 0.55),
        enc.classify(OS_POS, OS_HDG, 0.80, (5.0, 16.0), 0.0, 0.30),
        enc.classify(OS_POS, OS_HDG, 0.30, (5.0, 4.0), 0.0, 0.80),
        enc.classify(OS_POS, OS_HDG, 0.80, (5.0, 2.0), 0.0, 0.20),
    }
    assert seen == set(enc.CLASSES)
    assert len(enc.CLASSES) == 5


# ---------------------------------------------------------------------------
# The two required modifications
# ---------------------------------------------------------------------------
def test_head_on_band_is_wider_than_the_source_table():
    """01 §5.3 modification 2: widen from Waltz & Okhrin's +/-5 deg."""
    assert cfg.HEAD_ON_BEARING_HALF_DEG > 5.0
    assert 6.0 <= cfg.HEAD_ON_BEARING_HALF_DEG <= 10.0

    # A target 7 deg off the bow is head-on under the widened band and would
    # not have been under the original.
    import math
    d = 10.0
    a = math.radians(7.0)
    target = (OS_POS[0] + d * math.sin(a), OS_POS[1] + d * math.cos(a))
    assert enc.classify(OS_POS, OS_HDG, OS_SPD, target, 180.0, 0.55) == enc.HEAD_ON


def test_overtaking_and_being_overtaken_are_mirror_images():
    """Swapping the two vessels must swap the two roles."""
    fast, slow = 0.80, 0.30
    astern, ahead = (5.0, 4.0), (5.0, 16.0)

    assert enc.classify(OS_POS, OS_HDG, slow, astern, 0.0, fast) == enc.BEING_OVERTAKEN
    assert enc.classify(astern, OS_HDG, fast, OS_POS, 0.0, slow) == enc.OVERTAKING
    assert enc.classify(OS_POS, OS_HDG, fast, ahead, 0.0, slow) == enc.OVERTAKING
    assert enc.classify(ahead, OS_HDG, slow, OS_POS, 0.0, fast) == enc.BEING_OVERTAKEN


def test_overtaking_requires_a_speed_advantage():
    """Equal speeds astern is not an overtaking, however aligned the courses."""
    assert enc.classify(OS_POS, OS_HDG, 0.55, (5.0, 4.0), 0.0, 0.55) == enc.NONE
    assert enc.classify(OS_POS, OS_HDG, 0.55, (5.0, 16.0), 0.0, 0.55) == enc.NONE


def test_speed_margin_suppresses_a_marginal_overtaker():
    """A target barely faster must not flip the class on estimation noise."""
    barely = OS_SPD + 0.5 * cfg.BEING_OVERTAKEN_SPEED_MARGIN
    assert enc.classify(OS_POS, OS_HDG, OS_SPD, (5.0, 4.0), 0.0, barely) == enc.NONE

    clearly = OS_SPD + 2.0 * cfg.BEING_OVERTAKEN_SPEED_MARGIN
    assert enc.classify(OS_POS, OS_HDG, OS_SPD, (5.0, 4.0), 0.0, clearly) == enc.BEING_OVERTAKEN


def test_rule_13_takes_precedence_over_crossing():
    """An overtaking geometry must not be reported as a crossing."""
    # Target fine on the starboard quarter, same course, faster.
    result = enc.classify(OS_POS, OS_HDG, 0.30, (6.5, 5.0), 10.0, 0.80)
    assert result == enc.BEING_OVERTAKEN


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------
def test_one_hot_order_is_frozen():
    """Every checkpoint depends on this ordering."""
    assert enc.CLASSES == (
        "none", "head_on", "crossing", "overtaking", "being_overtaken",
    )
    for i, name in enumerate(enc.CLASSES):
        v = enc.one_hot(name)
        assert v.shape == (5,)
        assert v.dtype == np.float32
        assert v[i] == 1.0
        assert v.sum() == 1.0


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------
def test_classifier_matches_the_pure_function_on_first_sight():
    clf = enc.EncounterClassifier()
    assert clf.update(1, OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55) == enc.HEAD_ON


def test_hysteresis_prevents_chatter_at_a_sector_boundary():
    """A target oscillating across the head-on/crossing threshold.

    Without hysteresis the class would flip on every step, and 02's reward would
    penalise the agent for a role that never held long enough to act on.
    """
    import math
    clf = enc.EncounterClassifier()
    edge = cfg.HEAD_ON_BEARING_HALF_DEG

    def target_at(bearing_deg):
        a = math.radians(bearing_deg)
        return (OS_POS[0] + 10.0 * math.sin(a), OS_POS[1] + 10.0 * math.cos(a))

    # Settle on head-on just inside the band.
    for _ in range(3):
        clf.update(1, OS_POS, OS_HDG, OS_SPD, target_at(edge - 1.0), 180.0, 0.55)
    assert clf.held(1) == enc.HEAD_ON

    # Now jitter either side of the threshold by a degree.
    results = []
    for k in range(12):
        bearing = edge + (1.0 if k % 2 else -1.0)
        results.append(clf.update(1, OS_POS, OS_HDG, OS_SPD,
                                  target_at(bearing), 180.0, 0.55))

    # The held class must not chatter.
    assert set(results) == {enc.HEAD_ON}


def test_a_sustained_change_is_eventually_adopted():
    """Hysteresis must delay a real transition, not block it."""
    clf = enc.EncounterClassifier()
    for _ in range(3):
        clf.update(1, OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55)
    assert clf.held(1) == enc.HEAD_ON

    seen = [clf.update(1, OS_POS, OS_HDG, OS_SPD, (12.0, 17.0), 270.0, 0.55)
            for _ in range(cfg.ENCOUNTER_HOLD_STEPS + 2)]

    assert seen[0] == enc.HEAD_ON                        # not adopted immediately
    assert seen[-1] == enc.CROSSING                      # but adopted in the end
    assert seen.index(enc.CROSSING) == cfg.ENCOUNTER_HOLD_STEPS - 1


def test_an_interrupted_change_resets_the_counter():
    clf = enc.EncounterClassifier()
    clf.update(1, OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55)

    for _ in range(cfg.ENCOUNTER_HOLD_STEPS - 1):
        clf.update(1, OS_POS, OS_HDG, OS_SPD, (12.0, 17.0), 270.0, 0.55)
    # One step back to head-on wipes the pending transition...
    clf.update(1, OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55)
    # ...so a single crossing step must not flip it.
    assert clf.update(1, OS_POS, OS_HDG, OS_SPD, (12.0, 17.0), 270.0, 0.55) == enc.HEAD_ON


def test_state_is_per_track():
    """Two targets must not share a held class."""
    clf = enc.EncounterClassifier()
    clf.update(1, OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55)
    clf.update(2, OS_POS, OS_HDG, 0.30, (5.0, 4.0), 0.0, 0.80)
    assert clf.held(1) == enc.HEAD_ON
    assert clf.held(2) == enc.BEING_OVERTAKEN


def test_forget_clears_a_lost_track():
    """Slot re-use must not leak one target's history into another's."""
    clf = enc.EncounterClassifier()
    clf.update(1, OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55)
    assert clf.held(1) == enc.HEAD_ON
    clf.forget(1)
    assert clf.held(1) == enc.NONE


def test_reset_clears_everything():
    clf = enc.EncounterClassifier()
    clf.update(1, OS_POS, OS_HDG, OS_SPD, (5.0, 20.0), 180.0, 0.55)
    clf.reset()
    assert clf.held(1) == enc.NONE


# ---------------------------------------------------------------------------
# Single definition
# ---------------------------------------------------------------------------
def test_the_wrapper_delegates_to_the_pure_function():
    """01 §5.3: one module, two consumers.  The thresholds live in exactly one
    place, so the wrapper must not carry a second copy of the geometry."""
    import inspect
    source = inspect.getsource(enc.EncounterClassifier)
    for name in ("HEAD_ON_BEARING_HALF_DEG", "CROSSING_STBD_MAX_DEG",
                 "OVERTAKING_CT_HALF_DEG", "BEING_OVERTAKEN_BEARING_MIN_DEG"):
        assert name not in source, f"{name} is duplicated inside the wrapper"
    assert "classify(" in source
