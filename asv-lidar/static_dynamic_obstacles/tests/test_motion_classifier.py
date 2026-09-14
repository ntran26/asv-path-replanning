"""F31's fix: static/dynamic by free-space consistency, not centroid speed.

Scenes are ray-cast with the simulator's own LiDAR against real polygons --
panels, a target hull and the facility walls -- because the property under test
is geometric: a ray that passes through a point of a solid's boundary must
already have hit the solid.  A synthetic arc of returns would not exercise it.
"""

import numpy as np
import pytest

import boundary_raycast as br
import constants as cfg
import corridor as corr
import targets as tgt
import tracking as trk
from asv_lidar import Lidar

BASIN = br.rectangle(cfg.MAP_WIDTH, cfg.MAP_HEIGHT)
WALLS = corr.facility_walls()


def panel(cx, cy, half=0.5):
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half)]


def observe(lidar, pos, heading, scene, pose_error=(0.0, 0.0)):
    """One revolution, gated and segmented, from a (possibly wrong) pose."""
    lidar.scan(pos, heading, obstacles=list(scene) + [WALLS])
    raw = lidar.ranges.copy()
    ex, ey = pos[0] + pose_error[0], pos[1] + pose_error[1]
    sx, sy = trk.sensor_origin(ex, ey, heading)
    gated = br.gate_beams(raw, lidar.bearings, sx, sy, heading, BASIN)
    clusters = trk.segment_scan(gated, lidar.bearings, sx, sy, heading)
    return clusters, trk.ScanFrame.from_scan(raw, lidar.bearings, (sx, sy), heading)


def drive_past(obstacles, *, x=5.0, y0=3.0, steps=30, speed=cfg.U_REF,
               classifier="free_space", targets=(), pose_noise=0.0, seed=0):
    """Own ship straight up the basin at `speed`; returns the tracker."""
    rng = np.random.default_rng(seed)
    lidar = Lidar()
    tracker = trk.Tracker(classifier=classifier)
    promoted = 0
    for k in range(steps):
        pos = (x, y0 + speed * cfg.UPDATE_RATE * k)
        hulls = []
        for target in targets:
            if k:
                target.step(cfg.UPDATE_RATE)
            hulls.append(target.hull())
        err = tuple(rng.normal(0.0, pose_noise, 2)) if pose_noise else (0.0, 0.0)
        clusters, scan = observe(lidar, pos, 0.0, list(obstacles) + hulls, err)
        tracker.update(clusters, cfg.UPDATE_RATE, scan=scan)
        promoted += int(bool(tracker.dynamic_tracks()))
    return tracker, promoted


# ---------------------------------------------------------------------------
# The phantom it exists to remove
# ---------------------------------------------------------------------------
def test_a_panel_passed_close_aboard_is_never_promoted():
    """The F31 geometry: the visible face changes as the vessel passes, so the
    centroid slides -- and nothing about the panel moved."""
    panels = [panel(6.6, 9.0), panel(3.2, 13.0), panel(6.9, 17.0)]
    _, promoted = drive_past(panels, steps=32)
    assert promoted == 0


def test_the_speed_classifier_is_fooled_by_the_same_pass():
    """Why a threshold could not fix it: the same frames through 01's classifier."""
    panels = [panel(6.6, 9.0), panel(3.2, 13.0), panel(6.9, 17.0)]
    _, promoted = drive_past(panels, steps=32, classifier="speed")
    assert promoted > 0


def test_pose_error_inside_the_tolerance_creates_no_motion():
    panels = [panel(6.6, 9.0), panel(3.2, 13.0)]
    sigma = cfg.MOTION_PASS_TOL_M / (5.0 * np.sqrt(2.0))
    for seed in range(3):
        _, promoted = drive_past(panels, steps=26, pose_noise=sigma, seed=seed)
        assert promoted == 0, seed


# ---------------------------------------------------------------------------
# ... without losing the vessels it exists to find
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, make", [
    ("head_on", lambda: tgt.Target(5.6, 18.0, 180.0, cfg.U_REF)),
    ("crossing, broadside", lambda: tgt.Target(8.5, 9.0, 270.0, cfg.U_REF, confined=False)),
    ("overtaken, slow ahead", lambda: tgt.Target(5.8, 8.0, 0.0, 0.45 * cfg.U_REF)),
])
def test_a_moving_target_is_promoted_within_two_seconds_of_tracking(name, make):
    target = make()
    lidar = Lidar()
    tracker = trk.Tracker()
    first_track = first_dynamic = None
    for k in range(20):
        if k:
            target.step(cfg.UPDATE_RATE)
        pos = (5.0, 3.0 + 0.6 * cfg.U_REF * cfg.UPDATE_RATE * k)
        clusters, scan = observe(lidar, pos, 0.0, [target.hull()])
        tracker.update(clusters, cfg.UPDATE_RATE, scan=scan)
        near = [t for t in tracker.confirmed_tracks()
                if np.hypot(*(t.position - [target.x, target.y])) <= cfg.TARGET_MATCH_RADIUS]
        if near and first_track is None:
            first_track = k
        if any(t.is_dynamic for t in near) and first_dynamic is None:
            first_dynamic = k
    assert first_dynamic is not None, f"{name}: never promoted"
    assert (first_dynamic - first_track) * cfg.UPDATE_RATE <= 2.0 + 1e-9, name


def test_a_target_that_stops_is_demoted_after_two_seconds():
    target = tgt.Target(5.5, 16.0, 180.0, cfg.U_REF)
    lidar = Lidar()
    tracker = trk.Tracker()
    for k in range(12):
        if k:
            target.step(cfg.UPDATE_RATE)
        clusters, scan = observe(lidar, (5.0, 4.0), 0.0, [target.hull()])
        tracker.update(clusters, cfg.UPDATE_RATE, scan=scan)
    assert tracker.dynamic_tracks()
    target.speed = 0.0
    quiet = 0
    for _ in range(cfg.MOTION_WINDOW_STEPS + cfg.DYNAMIC_DEMOTE_STEPS + 2):
        clusters, scan = observe(lidar, (5.0, 4.0), 0.0, [target.hull()])
        tracker.update(clusters, cfg.UPDATE_RATE, scan=scan)
        quiet += 1
        if not tracker.dynamic_tracks():
            break
    assert not tracker.dynamic_tracks()
    assert quiet >= cfg.DYNAMIC_DEMOTE_STEPS


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------
def test_a_ray_that_returned_beyond_a_point_passed_through_it():
    lidar = Lidar()
    lidar.scan((5.0, 5.0), 0.0, obstacles=[panel(5.0, 15.0, half=2.0), WALLS])
    scan = trk.ScanFrame.from_scan(lidar.ranges, lidar.bearings, lidar.pos, 0.0)
    ahead = np.array([[lidar.pos[0], lidar.pos[1] + 6.0]])      # short of the face at y = 13
    assert scan.passes_through(ahead, 0.25).all()
    behind = np.array([[lidar.pos[0], 14.0]])                     # inside the panel
    assert not scan.passes_through(behind, 0.25).any()


def test_a_no_return_is_not_evidence_of_free_space():
    """The C1 reports nothing for a surface inside its 1 m dead zone, so an
    empty beam cannot certify the space behind it."""
    lidar = Lidar()
    lidar.scan((5.0, 5.0), 0.0, obstacles=[])          # no walls: every beam empty
    scan = trk.ScanFrame.from_scan(lidar.ranges, lidar.bearings, lidar.pos, 0.0)
    ahead = np.array([[lidar.pos[0], lidar.pos[1] + 6.0]])
    assert not scan.passes_through(ahead, 0.25).any()


def test_a_point_inside_the_dead_zone_is_unknown():
    lidar = Lidar()
    lidar.scan((5.0, 5.0), 0.0, obstacles=[WALLS])
    scan = trk.ScanFrame.from_scan(lidar.ranges, lidar.bearings, lidar.pos, 0.0)
    close = np.array([[lidar.pos[0], lidar.pos[1] + 0.5]])
    assert not scan.passes_through(close, 0.25).any()


def test_an_edge_on_face_is_not_seen_through_under_a_small_sideways_error():
    """The second flaw T8 found, under 3 cm of pose noise.

    From a sensor almost in the plane of a panel's face, a face point displaced
    a few centimetres sideways lies on a ray that grazes past the panel's corner
    and returns from whatever is behind -- so the beam *toward* it certifies
    free space.  The neighbours within the tolerance's lateral width hit the
    panel short of the point, and they are what stop it counting.
    """
    lidar = Lidar()
    face = panel(6.6, 12.0)                      # west face at x = 6.1
    backstop = panel(6.5, 18.0, half=1.0)
    lidar.scan((5.6, 2.2), 0.0, obstacles=[face, backstop, WALLS])
    scan = trk.ScanFrame.from_scan(lidar.ranges, lidar.bearings, lidar.pos, 0.0)
    point = np.array([[6.1 - 0.03, 12.3]])      # on the face, 3 cm outward

    rel = point[0] - scan.origin
    rho = float(np.hypot(*rel))
    idx = int(np.rint((np.degrees(np.arctan2(rel[0], rel[1])) % 360.0) / cfg.LIDAR_BEAM_RES_DEG))
    tol = cfg.MOTION_PASS_TOL_M
    assert scan.ranges[idx] < cfg.LIDAR_RANGE - 1e-6 and scan.ranges[idx] > rho + tol, (
        "the geometry no longer reproduces the grazing ray the test is about")
    assert not scan.passes_through(point, tol).any()


def test_explains_finds_a_nearby_return():
    lidar = Lidar()
    lidar.scan((5.0, 5.0), 0.0, obstacles=[panel(5.0, 10.0), WALLS])
    scan = trk.ScanFrame.from_scan(lidar.ranges, lidar.bearings, lidar.pos, 0.0)
    face = np.array([[5.0, 9.6]])
    assert scan.explains(face, 0.30).all()
    assert not scan.explains(np.array([[2.0, 7.0]]), 0.30).any()


def test_points_without_a_scan_fall_back_to_the_speed_classifier():
    tracker = trk.Tracker()
    pos = np.array([5.0, 20.0])
    for _ in range(12):
        pos = pos + np.array([0.0, -0.6]) * cfg.UPDATE_RATE
        tracker.update([pos.copy()])
    assert tracker.dynamic_tracks()


def test_an_unknown_classifier_is_refused():
    with pytest.raises(ValueError):
        trk.Tracker(classifier="vibes")
