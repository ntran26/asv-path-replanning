"""Target tracking: clustering, association, velocity estimation, motion class.

Revision 2 adds the Study 2 degradation axes: detection dropout, velocity
estimate noise, and occlusion duration measured as track coast time.
"""

import numpy as np
import pytest

import constants as cfg
import lidar_pooling as lp
import tracking as trk


def synthetic_scan(objects, x=5.0, y=10.0, heading=0.0):
    """Ray-cast-free scan: place each object as an arc of returns.

    `objects` is a sequence of (world_x, world_y, half_width_m).
    """
    bearings = lp.beam_bearings()
    ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE, dtype=np.float64)
    for ox, oy, half in objects:
        dx, dy = ox - x, oy - y
        r = float(np.hypot(dx, dy))
        centre = (np.degrees(np.arctan2(dx, dy)) - heading + 180.0) % 360.0 - 180.0
        half_deg = np.degrees(np.arctan2(half, max(r, 1e-6)))
        hit = np.abs((bearings - centre + 180.0) % 360.0 - 180.0) <= half_deg
        ranges[hit] = np.minimum(ranges[hit], r)
    return ranges, bearings


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_empty_scan_yields_no_clusters():
    ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE)
    assert cluster(ranges) == []


def cluster(ranges, bearings=None, x=5.0, y=10.0, heading=0.0):
    if bearings is None:
        bearings = lp.beam_bearings()
    return trk.cluster_scan(ranges, bearings, x, y, heading)


def test_one_object_gives_one_cluster_at_its_position():
    ranges, bearings = synthetic_scan([(5.0, 14.0, 0.25)])
    clusters = cluster(ranges, bearings)
    assert len(clusters) == 1
    assert clusters[0][0] == pytest.approx(5.0, abs=0.3)
    assert clusters[0][1] == pytest.approx(14.0, abs=0.3)


def test_two_separated_objects_give_two_clusters():
    ranges, bearings = synthetic_scan([(3.0, 14.0, 0.25), (8.0, 14.0, 0.25)])
    assert len(cluster(ranges, bearings)) == 2


def test_min_points_rejects_a_single_stray_return():
    bearings = lp.beam_bearings()
    ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE)
    ranges[10] = 4.0                          # one lone beam
    assert cluster(ranges, bearings) == []


def test_adaptive_threshold_keeps_a_distant_object_together():
    """At 0.5 deg spacing, neighbouring beams are 0.017 m apart at 2 m but
    0.14 m at 16 m.  A fixed gap threshold would over-segment the far object."""
    near, _ = synthetic_scan([(5.0, 12.0, 0.25)])
    far, _ = synthetic_scan([(5.0, 24.0, 0.25)])
    assert len(cluster(near)) == 1
    assert len(cluster(far)) == 1


def test_scan_to_points_drops_no_returns():
    bearings = lp.beam_bearings()
    ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE)
    ranges[:10] = 3.0
    pts = trk.scan_to_points(ranges, bearings, 5.0, 10.0, 0.0)
    assert pts.shape == (10, 2)


def test_scan_to_points_respects_heading():
    """A return dead ahead lands north when heading 0, east when heading 90."""
    bearings = np.array([0.0])
    north = trk.scan_to_points([3.0], bearings, 5.0, 10.0, 0.0)[0]
    east = trk.scan_to_points([3.0], bearings, 5.0, 10.0, 90.0)[0]
    assert north == pytest.approx([5.0, 13.0])
    assert east == pytest.approx([8.0, 10.0])


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
def test_a_stationary_object_converges_to_zero_velocity():
    track = trk.Track((5.0, 14.0))
    for _ in range(40):
        track.predict()
        track.update((5.0, 14.0))
    assert track.speed < cfg.DYNAMIC_SPEED_OFF
    assert track.position == pytest.approx([5.0, 14.0], abs=1e-3)


def test_a_constant_velocity_target_converges_to_its_true_velocity():
    truth_v = np.array([0.0, -0.5])
    pos = np.array([5.0, 20.0])
    track = trk.Track(pos)
    for _ in range(60):
        pos = pos + truth_v * cfg.UPDATE_RATE
        track.predict()
        track.update(pos)
    assert track.velocity == pytest.approx(truth_v, abs=0.05)
    assert track.speed == pytest.approx(0.5, abs=0.05)


def test_course_is_reported_as_a_compass_bearing():
    track = trk.Track((5.0, 10.0))
    track.state = np.array([5.0, 10.0, 0.0, 1.0])       # due north
    assert track.course_deg == pytest.approx(0.0)
    track.state = np.array([5.0, 10.0, 1.0, 0.0])       # due east
    assert track.course_deg == pytest.approx(90.0)
    track.state = np.array([5.0, 10.0, 0.0, -1.0])      # due south
    assert track.course_deg == pytest.approx(180.0)


def test_filter_smooths_measurement_noise():
    rng = np.random.default_rng(0)
    truth_v = np.array([0.0, -0.5])
    pos = np.array([5.0, 20.0])
    track = trk.Track(pos)
    for _ in range(80):
        pos = pos + truth_v * cfg.UPDATE_RATE
        track.predict()
        track.update(pos + rng.normal(0.0, 0.03, 2))
    assert np.linalg.norm(track.velocity - truth_v) < 0.15


# ---------------------------------------------------------------------------
# Static / dynamic split
# ---------------------------------------------------------------------------
def test_static_object_stays_static():
    track = trk.Track((5.0, 14.0))
    for _ in range(40):
        track.predict()
        track.update((5.0, 14.0))
        track.update_motion_class()
    assert not track.is_dynamic


def test_moving_object_becomes_dynamic():
    pos = np.array([5.0, 20.0])
    track = trk.Track(pos)
    for _ in range(60):
        pos = pos + np.array([0.0, -0.6]) * cfg.UPDATE_RATE
        track.predict()
        track.update(pos)
        track.update_motion_class()
    assert track.is_dynamic


def test_hysteresis_needs_sustained_evidence_to_flip():
    """A single fast frame must not reclassify a static object."""
    track = trk.Track((5.0, 14.0))
    for _ in range(20):
        track.predict()
        track.update((5.0, 14.0))
        track.update_motion_class()
    assert not track.is_dynamic

    track.state[2:] = [0.0, 5.0]                 # one implausible frame
    track.update_motion_class()
    assert not track.is_dynamic                  # hold steps not yet met


def test_on_and_off_thresholds_are_separated():
    """A single threshold would chatter at the boundary."""
    assert cfg.DYNAMIC_SPEED_OFF < cfg.DYNAMIC_SPEED_ON


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
def test_tracker_creates_and_confirms_a_track():
    tracker = trk.Tracker()
    for _ in range(cfg.TRACK_MIN_HITS):
        tracker.update([np.array([5.0, 14.0])])
    assert len(tracker.confirmed_tracks()) == 1


def test_tracker_holds_a_track_id_across_steps():
    """Slot persistence (01 §6.2) depends on ids being stable."""
    tracker = trk.Tracker()
    pos = np.array([5.0, 20.0])
    ids = set()
    for _ in range(30):
        pos = pos + np.array([0.0, -0.5]) * cfg.UPDATE_RATE
        tracker.update([pos.copy()])
        ids |= {t.id for t in tracker.confirmed_tracks()}
    assert len(ids) == 1


def test_tracker_separates_two_targets():
    tracker = trk.Tracker()
    a = np.array([3.0, 20.0])
    b = np.array([8.0, 20.0])
    for _ in range(20):
        a = a + np.array([0.0, -0.5]) * cfg.UPDATE_RATE
        b = b + np.array([0.0, -0.4]) * cfg.UPDATE_RATE
        tracker.update([a.copy(), b.copy()])
    assert len(tracker.confirmed_tracks()) == 2


def test_tracker_drops_a_track_after_enough_misses():
    tracker = trk.Tracker()
    for _ in range(5):
        tracker.update([np.array([5.0, 14.0])])
    assert tracker.confirmed_tracks()

    for _ in range(cfg.TRACK_MAX_MISSES + 2):
        tracker.update([])
    assert tracker.confirmed_tracks() == []


def test_association_gate_rejects_an_implausible_jump():
    """A detection far from every track starts a new track, not an update."""
    tracker = trk.Tracker()
    for _ in range(5):
        tracker.update([np.array([5.0, 14.0])])
    first = tracker.confirmed_tracks()[0].id

    for _ in range(5):
        tracker.update([np.array([5.0, 14.0]), np.array([9.0, 20.0])])
    ids = {t.id for t in tracker.confirmed_tracks()}
    assert first in ids
    assert len(ids) == 2


def test_static_and_dynamic_tracks_are_partitioned():
    tracker = trk.Tracker()
    static = np.array([2.0, 14.0])
    moving = np.array([8.0, 22.0])
    for _ in range(60):
        moving = moving + np.array([0.0, -0.6]) * cfg.UPDATE_RATE
        tracker.update([static.copy(), moving.copy()])

    dynamic = tracker.dynamic_tracks()
    stationary = tracker.static_tracks()
    assert len(dynamic) == 1
    assert len(stationary) == 1
    assert dynamic[0].position[0] == pytest.approx(8.0, abs=0.3)
    assert set(dynamic) & set(stationary) == set()


def test_tracker_reset_clears_state():
    tracker = trk.Tracker()
    for _ in range(5):
        tracker.update([np.array([5.0, 14.0])])
    tracker.reset()
    assert tracker.tracks == []


# ---------------------------------------------------------------------------
# The 01 §4 coupling to 05
# ---------------------------------------------------------------------------
def test_pose_drift_creates_false_velocity_on_a_static_object():
    """01 §4 step 3, made concrete.

    A genuinely static object, observed from a pose that drifts, produces a
    non-zero estimated velocity.  This is why the static/dynamic threshold has
    to sit above the drift floor, and why 05's rf2o characterisation gates the
    value in `constants.py`.
    """
    bearings = np.array([0.0])
    tracker = trk.Tracker()
    drift_per_step = 0.02                       # 0.2 m/s of apparent motion
    for k in range(40):
        est_x = 5.0 + k * drift_per_step
        # The object is truly fixed at (5, 14), but we localise ourselves wrong.
        true_range = 4.0
        pts = trk.scan_to_points([true_range], bearings, est_x, 10.0, 0.0)
        tracker.update(list(pts))

    tracks = tracker.confirmed_tracks()
    assert tracks
    assert tracks[0].speed > cfg.DYNAMIC_SPEED_ON
    assert tracks[0].is_dynamic          # a static object misread as a target


# ---------------------------------------------------------------------------
# Study 2 degradation axes (04 §6)
# ---------------------------------------------------------------------------
def test_degradation_defaults_are_nominal():
    tracker = trk.Tracker()
    assert tracker.dropout_p == 0.0
    assert tracker.velocity_noise == 0.0
    assert tracker.dropped_detections == 0


def test_detection_dropout_prevents_a_track_forming():
    tracker = trk.Tracker(dropout_p=1.0, rng=np.random.default_rng(0))
    for _ in range(20):
        tracker.update([np.array([5.0, 14.0])])
    assert tracker.confirmed_tracks() == []
    assert tracker.dropped_detections == 20


def test_partial_dropout_still_tracks():
    tracker = trk.Tracker(dropout_p=0.3, rng=np.random.default_rng(0))
    pos = np.array([5.0, 20.0])
    for _ in range(60):
        pos = pos + np.array([0.0, -0.5]) * cfg.UPDATE_RATE
        tracker.update([pos.copy()])
    assert tracker.confirmed_tracks()
    assert tracker.dropped_detections > 0


def test_velocity_noise_perturbs_the_estimate():
    """Stands in for scan motion distortion plus filter residual."""
    clean = trk.Tracker(rng=np.random.default_rng(0))
    noisy = trk.Tracker(velocity_noise=0.20, rng=np.random.default_rng(0))
    pos = np.array([5.0, 20.0])
    speeds_clean, speeds_noisy = [], []
    for _ in range(40):
        pos = pos + np.array([0.0, -0.5]) * cfg.UPDATE_RATE
        clean.update([pos.copy()])
        noisy.update([pos.copy()])
        if clean.confirmed_tracks():
            speeds_clean.append(clean.confirmed_tracks()[0].speed)
            speeds_noisy.append(noisy.confirmed_tracks()[0].speed)
    assert np.std(speeds_noisy) > np.std(speeds_clean)


def test_max_coast_records_the_longest_occlusion():
    """Occlusion duration is a Study 2 axis and a reported metric."""
    tracker = trk.Tracker()
    for _ in range(5):
        tracker.update([np.array([5.0, 14.0])])
    assert tracker.max_coast == 0

    for _ in range(3):
        tracker.update([])                     # occluded
    assert tracker.max_coast == 3

    tracker.update([np.array([5.0, 14.0])])    # reacquired
    assert tracker.max_coast == 3              # the record survives


def test_min_points_rejects_a_taut_suspension_line():
    """03 §4a: a rope returns on one or two beams and must not be tracked."""
    assert cfg.CLUSTER_MIN_POINTS >= 3
    bearings = lp.beam_bearings()
    ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE)
    ranges[100:102] = 5.0                      # two-beam return
    assert cluster(ranges, bearings) == []

    ranges[100:106] = 5.0                      # a genuine small obstacle
    assert len(cluster(ranges, bearings)) == 1
