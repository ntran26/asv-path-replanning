"""Observation assembly, slot management, and the features extractor.

Revision 2: one target slot with a presence bit, 56 dims, five encounter
classes.

Kickoff §7, restated for the new scope:
* shape and dtype exactly match `observation_space`; 0- and 1-target cases both
  produce finite values with the correct presence bit
* an absent slot filled with arbitrary garbage produces byte-identical extractor
  output to the same slot filled with zeros
"""

import numpy as np
import pytest
import torch

import constants as cfg
import encounter as enc
import observation as obs
from features_extractor import ASVFeaturesExtractor
from tracking import Track

OS_POS = (5.0, 10.0)
OS_VEL = (0.0, 0.55)
OS_HDG = 0.0


def make_track(x, y, vx, vy, *, confirmed=True):
    t = Track((x, y))
    t.state = np.array([x, y, vx, vy], dtype=np.float64)
    if confirmed:
        t.hits = cfg.TRACK_MIN_HITS
    return t


def base_kwargs(tracks=()):
    return dict(
        sector_closeness=np.zeros(cfg.LIDAR_SECTORS, dtype=np.float32),
        boundary_scan=np.zeros(cfg.BOUNDARY_RAYS, dtype=np.float32),
        u=0.55, v=0.0, yaw_rate_degps=0.0,
        cross_track_error=0.0, course_error_deg=0.0,
        lookahead_course_error_deg=0.0,
        tracks=tracks, p_os=OS_POS, v_os=OS_VEL, heading_os_deg=OS_HDG,
    )


# ---------------------------------------------------------------------------
# Space and dimensions
# ---------------------------------------------------------------------------
def test_total_dimension_is_56():
    """S6: observation reduced to ~56 dims by the two-vessel repositioning."""
    space = obs.observation_space()
    total = sum(int(np.prod(space[k].shape)) for k in space.spaces)
    assert total == 56
    assert obs.OBS_DIM == 56


def test_branch_dimensions_match_the_spec():
    space = obs.observation_space()
    assert space["lidar"].shape == (27,)
    assert space["boundary"].shape == (7,)
    assert space["ego"].shape == (3,)
    assert space["path"].shape == (3,)
    assert space["target"].shape == (16,)          # 15 features + presence


def test_there_are_exactly_five_branches():
    assert set(obs.observation_space().spaces) == {
        "lidar", "boundary", "ego", "path", "target"}


def test_one_target_slot_but_the_count_stays_configurable():
    """S1: N_MAX_TARGETS is a config parameter, so multi-vessel is a retrain."""
    assert cfg.N_MAX_TARGETS == 1
    assert cfg.TARGET_FEATURES == 16
    assert len(obs.SLOT_FEATURE_NAMES) == 16


def test_slot_machinery_still_scales_past_one():
    """The extension path must actually work, not merely be asserted."""
    manager = obs.SlotManager(n_slots=3)
    tracks = [make_track(5.0, 12.0 + i, 0.0, -0.5) for i in range(3)]
    assignment = manager.update(tracks, [0.9, 0.5, 0.1])
    assert len(assignment) == 3
    assert sorted(assignment.values()) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Contract with observation_space
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_targets", [0, 1])
def test_observation_matches_the_space(n_targets):
    tracks = [make_track(5.0, 16.0, 0.0, -0.4)][:n_targets]
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs(tracks))

    space = obs.observation_space()
    assert space.contains(o), {k: (v.shape, v.dtype, v.min(), v.max())
                               for k, v in o.items()}
    for key, box in space.spaces.items():
        assert o[key].shape == box.shape
        assert o[key].dtype == np.float32
        assert np.all(np.isfinite(o[key]))


@pytest.mark.parametrize("n_targets", [0, 1])
def test_presence_bit_reports_the_target(n_targets):
    tracks = [make_track(5.0, 16.0, 0.0, -0.4)][:n_targets]
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs(tracks))
    _, presence = obs.split_target(o["target"])

    assert presence.shape == (1,)
    assert float(presence.sum()) == float(n_targets)
    assert set(np.unique(presence)).issubset({0.0, 1.0})


def test_absent_slot_is_zero_filled_including_its_presence_bit():
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs())
    slots, presence = obs.split_target(o["target"])
    assert np.allclose(slots, 0.0)
    assert float(presence[0]) == 0.0


def test_empty_slot_would_be_ambiguous_without_the_presence_bit():
    """Why the presence bit exists: zero is a legitimate feature value.

    A zero-filled slot decodes to bearing sin=0/cos=0, relative speed 0 -- i.e.
    a target on top of the vessel on a matching course.  Only the presence bit
    separates that from an empty slot.
    """
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs())
    slots, _ = obs.split_target(o["target"])
    # The one-hot inside an empty slot is all-zero, which is not a valid class.
    assert float(slots[0][obs.CLASS_SLICE].sum()) == 0.0


def test_occupied_slot_sets_presence_and_a_class():
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs([make_track(5.0, 16.0, 0.0, -0.4)]))
    slots, presence = obs.split_target(o["target"])
    assert float(presence[0]) == 1.0
    assert np.any(slots[0] != 0.0)
    assert float(slots[0][obs.CLASS_SLICE].sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Feature values
# ---------------------------------------------------------------------------
def test_features_are_in_range():
    builder = obs.ObservationBuilder()
    for track in (make_track(5.0, 16.0, 0.0, -0.5),
                  make_track(2.0, 14.0, 0.3, 0.0),
                  make_track(8.0, 4.0, 0.0, 0.9)):
        builder.reset()
        o = builder.build(**base_kwargs([track]))
        slots, _ = obs.split_target(o["target"])
        assert np.all(slots >= -1.0) and np.all(slots <= 1.0)


def test_bearing_is_encoded_as_sin_cos():
    """Directly ahead: sin = 0, cos = 1."""
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs([make_track(5.0, 16.0, 0.0, -0.5)]))
    slots, _ = obs.split_target(o["target"])
    assert slots[0][1] == pytest.approx(0.0, abs=1e-6)     # bearing_sin
    assert slots[0][2] == pytest.approx(1.0, abs=1e-6)     # bearing_cos


def test_encounter_one_hot_is_five_wide_and_single():
    """S4: five classes, not six.  Port/starboard crossing collapsed."""
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs([make_track(5.0, 16.0, 0.0, -0.5)]))
    slots, _ = obs.split_target(o["target"])
    one_hot = slots[0][obs.CLASS_SLICE]
    assert one_hot.shape == (5,)
    assert one_hot.sum() == pytest.approx(1.0)
    assert one_hot[enc.CLASS_INDEX[enc.HEAD_ON]] == 1.0


def test_tcpa_sign_survives_normalisation():
    """A target already passed must give a negative TCPA feature."""
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs([make_track(5.0, 4.0, 0.0, -0.5)]))
    slots, _ = obs.split_target(o["target"])
    assert slots[0][8] < 0.0


def test_feature_names_match_the_frozen_layout():
    assert obs.SLOT_FEATURE_NAMES[obs.PRESENCE_INDEX] == "presence"
    assert obs.PRESENCE_INDEX == 15
    assert obs.SLOT_FEATURE_NAMES[obs.CLASS_SLICE] == (
        "class_none", "class_head_on", "class_crossing",
        "class_overtaking", "class_being_overtaken")


# ---------------------------------------------------------------------------
# Slot management
# ---------------------------------------------------------------------------
def test_slot_is_held_across_steps():
    """01 §6.2: track-ID persistence, so discontinuities are real events."""
    builder = obs.ObservationBuilder()
    track = make_track(5.0, 18.0, 0.0, -0.2)
    builder.build(**base_kwargs([track]))
    slot = builder.slots.assignments[track.id]

    for _ in range(5):
        track.state[1] -= 0.5
        builder.build(**base_kwargs([track]))
    assert builder.slots.assignments[track.id] == slot


def test_slot_is_released_when_the_track_is_lost():
    builder = obs.ObservationBuilder()
    track = make_track(5.0, 16.0, 0.0, -0.5)
    builder.build(**base_kwargs([track]))
    assert builder.slots.assignments[track.id] == 0

    builder.build(**base_kwargs([]))
    assert builder.slots.assignments == {}
    assert builder.slots.occupant(0) is None


def test_a_reused_slot_does_not_inherit_the_old_encounter_class():
    """Track history must not leak across slot re-use."""
    builder = obs.ObservationBuilder()
    a = make_track(5.0, 16.0, 0.0, -0.5)                # head-on
    for _ in range(3):
        builder.build(**base_kwargs([a]))
    assert builder.encounter_classes[a.id] == enc.HEAD_ON

    builder.build(**base_kwargs([]))                    # a is lost
    b = make_track(5.0, 4.0, 0.0, 0.9)                  # overtaking us from astern
    o = builder.build(**base_kwargs([b]))

    assert builder.encounter_classes[b.id] != enc.HEAD_ON
    slots, presence = obs.split_target(o["target"])
    assert float(presence[0]) == 1.0
    assert slots[0][obs.CLASS_SLICE][enc.CLASS_INDEX[enc.HEAD_ON]] == 0.0


def test_the_riskier_target_holds_the_single_slot():
    """Contention is the extension hook; at one slot it is still well-defined."""
    builder = obs.ObservationBuilder()
    far = make_track(9.5, 24.0, 0.0, 0.5)               # low risk
    builder.build(**base_kwargs([far]))
    assert builder.slots.assignments == {far.id: 0}

    near = make_track(5.0, 12.0, 0.0, -0.8)             # much higher risk
    builder.build(**base_kwargs([far, near]))
    assert builder.slots.assignments == {near.id: 0}


def test_zero_and_one_target_episodes_are_representable():
    """Field runs carry one target; a fraction of training episodes carry none."""
    builder = obs.ObservationBuilder()
    space = obs.observation_space()
    for tracks in ([], [make_track(5.0, 16.0, 0.0, -0.5)]):
        o = builder.build(**base_kwargs(tracks))
        assert space.contains(o)
        builder.reset()


def test_crossing_side_is_exposed_for_the_reward_but_not_observed():
    """Rule 9(b) collapses the class; 02's passing-side term still needs the side."""
    builder = obs.ObservationBuilder()
    track = make_track(12.0, 17.0, -0.5, 0.0)           # crossing from starboard
    o = builder.build(**base_kwargs([track]))

    assert builder.encounter_classes[track.id] == enc.CROSSING
    assert builder.crossing_sides[track.id] == enc.SIDE_STARBOARD
    # ...and the one-hot carries no side information at all.
    slots, _ = obs.split_target(o["target"])
    assert float(slots[0][obs.CLASS_SLICE].sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Features extractor
# ---------------------------------------------------------------------------
def _batch(o, n=1):
    return {k: torch.as_tensor(np.repeat(v[None, :], n, axis=0)) for k, v in o.items()}


def test_extractor_runs_and_has_the_declared_width():
    space = obs.observation_space()
    ex = ASVFeaturesExtractor(space)
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs([make_track(5.0, 16.0, 0.0, -0.5)]))
    out = ex(_batch(o, 4))
    assert out.shape == (4, ex.features_dim)
    assert torch.all(torch.isfinite(out))


def test_absent_slot_garbage_is_byte_identical_to_zeros():
    """The headline gating guarantee.

    Fill the absent slot with arbitrary values, leaving the presence bit at
    zero.  The extractor output must not move by a single bit.
    """
    torch.manual_seed(0)
    space = obs.observation_space()
    ex = ASVFeaturesExtractor(space)
    ex.eval()

    builder = obs.ObservationBuilder()
    clean = builder.build(**base_kwargs())          # no target: presence = 0
    assert float(clean["target"][obs.PRESENCE_INDEX]) == 0.0

    dirty = {k: v.copy() for k, v in clean.items()}
    rng = np.random.default_rng(0)
    dirty["target"][:obs.PRESENCE_INDEX] = rng.normal(size=obs.PRESENCE_INDEX).astype(np.float32)

    with torch.no_grad():
        a = ex(_batch(clean))
        b = ex(_batch(dirty))
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_extractor_gates_before_the_encoder_not_after():
    """Gating only the output would still let the encoder's biases through."""
    torch.manual_seed(0)
    space = obs.observation_space()
    ex = ASVFeaturesExtractor(space)
    ex.eval()
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs())

    with torch.no_grad():
        out = ex(_batch(o))
    # The whole target half of the feature vector must be exactly zero.
    target_half = out[:, cfg.SCENE_ENCODER_HIDDEN:]
    assert torch.equal(target_half, torch.zeros_like(target_half))


def test_extractor_handles_a_zero_target_observation():
    space = obs.observation_space()
    ex = ASVFeaturesExtractor(space)
    ex.eval()
    builder = obs.ObservationBuilder()
    o = builder.build(**base_kwargs())
    with torch.no_grad():
        out = ex(_batch(o))
    assert torch.all(torch.isfinite(out))


def test_slot_encoder_weights_are_shared_across_slots():
    """One encoder for every slot -- what makes the extension a retrain."""
    space = obs.observation_space()
    ex = ASVFeaturesExtractor(space)
    assert ex.slot_encoder[0].in_features == cfg.TARGET_FEATURES
    assert ex.n_slots == cfg.N_MAX_TARGETS


def test_no_aggregation_flag_survives_from_revision_1():
    """D3 is superseded: no DeepSets/attention comparison to build or defend."""
    space = obs.observation_space()
    ex = ASVFeaturesExtractor(space)
    assert not hasattr(ex, "aggregate")
    assert not hasattr(ex, "attn_query")
    assert not hasattr(cfg, "AGGREGATE")
