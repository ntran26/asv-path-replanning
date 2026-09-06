"""Boundary raycast: hand-checked ranges, bends, varying width, gating.

Kickoff §7: "known polygon, known pose, hand-checked ranges; correct behaviour
at a bend and at varying width".
"""

import numpy as np
import pytest

import boundary_raycast as br
import constants as cfg
from lidar_pooling import closeness_from_ranges


# ---------------------------------------------------------------------------
# Hand-checked geometry
# ---------------------------------------------------------------------------
def test_centred_in_a_rectangle_heading_north():
    """10 x 25 basin, vessel at (5, 12.5) heading 0 (+y).

    Beam bearings are body-frame, compass style: -90 is port (-x), +90 is
    starboard (+x), 0 is ahead (+y).
    """
    poly = br.rectangle(10.0, 25.0)
    r = br.boundary_ranges(5.0, 12.5, 0.0, poly)

    assert r.shape == (7,)
    assert r[0] == pytest.approx(5.0)                    # -90, port wall at x=0
    assert r[3] == pytest.approx(12.5)                   # 0, far wall at y=25
    assert r[6] == pytest.approx(5.0)                    # +90, stbd wall at x=10

    # -60 deg: dx = sin(-60) = -0.866.  Hits x=0 at t = 5 / 0.866 = 5.774.
    assert r[1] == pytest.approx(5.0 / np.sin(np.radians(60.0)), rel=1e-6)
    assert r[5] == pytest.approx(5.0 / np.sin(np.radians(60.0)), rel=1e-6)

    # -30 deg: dx = -0.5, so x=0 at t = 10.  dy = cos(30) = 0.866, so y=25 at
    # t = 12.5/0.866 = 14.43.  The side wall is nearer.
    assert r[2] == pytest.approx(10.0, rel=1e-6)
    assert r[4] == pytest.approx(10.0, rel=1e-6)


def test_off_centre_pose_is_asymmetric():
    poly = br.rectangle(10.0, 25.0)
    r = br.boundary_ranges(2.0, 12.5, 0.0, poly)
    assert r[0] == pytest.approx(2.0)                    # close to port wall
    assert r[6] == pytest.approx(8.0)                    # far from starboard
    assert r[0] < r[6]


def test_heading_rotates_the_scan():
    """Turning 90 deg to starboard swaps what the beam abeam sees."""
    poly = br.rectangle(10.0, 25.0)
    r = br.boundary_ranges(5.0, 12.5, 90.0, poly)
    # Now heading +x. Port beam (-90) looks toward +y, starboard toward -y.
    assert r[0] == pytest.approx(12.5)
    assert r[3] == pytest.approx(5.0)                    # ahead is the x=10 wall
    assert r[6] == pytest.approx(12.5)


def test_range_is_clipped_to_max():
    """A basin larger than the sensor horizon must not report past it."""
    poly = br.rectangle(200.0, 200.0)
    r = br.boundary_ranges(100.0, 100.0, 0.0, poly)
    assert np.all(r <= cfg.BOUNDARY_MAX_RANGE + 1e-6)
    assert np.allclose(r, cfg.BOUNDARY_MAX_RANGE)


# ---------------------------------------------------------------------------
# Varying width and bends -- the cases that justify 7 rays over [d_port, d_stbd]
# ---------------------------------------------------------------------------
def test_narrowing_channel_is_visible_ahead_before_it_is_abeam():
    """A channel that narrows downstream.

    This is the geometry 01 §3.3 says the branch needs in order to carry
    information a simple port/starboard pair could not.
    """
    # Width 10 at y=0, tapering to width 4 by y=25.
    poly = [(0.0, 0.0), (10.0, 0.0), (7.0, 25.0), (3.0, 25.0)]
    r = br.boundary_ranges(5.0, 5.0, 0.0, poly)

    # Abeam clearances are still wide here...
    assert r[0] > 3.0 and r[6] > 3.0
    # ...but the forward-oblique rays already register the taper, and do so
    # differently from the pure-abeam rays.  That difference is the information.
    assert r[1] < cfg.BOUNDARY_MAX_RANGE
    assert r[5] < cfg.BOUNDARY_MAX_RANGE
    assert not np.allclose(r[1], r[0])


def test_bend_produces_asymmetric_forward_rays():
    """An L-shaped channel bending to starboard."""
    poly = [(0.0, 0.0), (6.0, 0.0), (6.0, 20.0), (20.0, 20.0),
            (20.0, 26.0), (0.0, 26.0)]
    # Close enough to the bend that the starboard-oblique ray clears the corner
    # at (6, 20) instead of hitting the lower channel's right wall.
    r = br.boundary_ranges(3.0, 18.0, 0.0, poly)

    # Ahead, the outer wall of the bend at y=26 is what is hit.
    assert r[3] == pytest.approx(8.0)
    # -30 deg still hits the left wall at x=0, 6 m away.
    assert r[2] == pytest.approx(6.0)
    # +30 deg passes the corner and runs on to y=26.
    assert r[4] == pytest.approx(8.0 / np.cos(np.radians(30.0)), rel=1e-6)
    # The bend makes the two forward-oblique rays disagree.  A [d_port, d_stbd]
    # pair could not represent this.
    assert r[4] > r[2]


def test_constant_width_centreline_is_the_redundancy_trap():
    """01 §3.3: on a centreline in a constant-width channel the branch is
    an affine function of cross-track error and carries nothing new.

    This test documents the failure mode rather than guarding against it --
    it is a property of the *scenario*, and 04 owns fixing it.
    """
    poly = br.rectangle(10.0, 25.0)
    offsets = [3.0, 4.0, 5.0, 6.0, 7.0]
    port = [br.boundary_ranges(x, 12.5, 0.0, poly)[0] for x in offsets]
    stbd = [br.boundary_ranges(x, 12.5, 0.0, poly)[6] for x in offsets]

    # Perfectly affine in x, and perfectly anti-correlated.
    assert np.allclose(np.diff(port), 1.0)
    assert np.allclose(np.diff(stbd), -1.0)
    assert np.allclose(np.add(port, stbd), 10.0)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_scan_uses_the_same_normalisation_as_c_t():
    poly = br.rectangle(10.0, 25.0)
    ranges = br.boundary_ranges(5.0, 12.5, 0.0, poly)
    scan = br.boundary_scan(5.0, 12.5, 0.0, poly)
    assert np.allclose(scan, closeness_from_ranges(ranges, cfg.BOUNDARY_MAX_RANGE))
    assert scan.dtype == np.float32
    assert scan.shape == (cfg.BOUNDARY_RAYS,)
    assert np.all((scan >= 0.0) & (scan <= 1.0))


def test_touching_the_wall_gives_closeness_one():
    poly = br.rectangle(10.0, 25.0)
    scan = br.boundary_scan(0.0, 12.5, 0.0, poly)
    assert scan[0] == pytest.approx(1.0)                 # port ray, zero range


# ---------------------------------------------------------------------------
# Pose noise hook
# ---------------------------------------------------------------------------
def test_pose_noise_defaults_to_disabled_and_is_a_no_op():
    """TODO(05) values are 0.0 today, so training currently sees a clean map."""
    noise = br.PoseNoise(np.random.default_rng(0))
    assert not noise.enabled
    assert noise.perturb(5.0, 12.5, 30.0) == (5.0, 12.5, 30.0)

    poly = br.rectangle(10.0, 25.0)
    a = br.boundary_scan(5.0, 12.5, 0.0, poly, pose_noise=noise)
    b = br.boundary_scan(5.0, 12.5, 0.0, poly)
    assert np.allclose(a, b)


def test_pose_noise_perturbs_when_enabled():
    noise = br.PoseNoise(np.random.default_rng(0), sigma_xy=0.10,
                         sigma_heading_deg=2.0, walk=0.0)
    assert noise.enabled
    poly = br.rectangle(10.0, 25.0)
    scans = [br.boundary_scan(5.0, 12.5, 0.0, poly, pose_noise=noise) for _ in range(20)]
    assert np.std([s[0] for s in scans]) > 0.0


def test_pose_noise_walk_accumulates():
    noise = br.PoseNoise(np.random.default_rng(1), sigma_xy=0.0,
                         sigma_heading_deg=0.0, walk=0.05)
    early = [abs(noise.perturb(5.0, 12.5, 0.0)[0] - 5.0) for _ in range(10)]
    for _ in range(200):
        noise.perturb(5.0, 12.5, 0.0)
    late = [abs(noise.perturb(5.0, 12.5, 0.0)[0] - 5.0) for _ in range(10)]
    assert np.mean(late) > np.mean(early)

    # reset() clears accumulated drift.  The next call still takes one walk
    # step, so the offset is small again rather than exactly zero.
    noise.reset()
    fresh = [abs(noise.perturb(5.0, 12.5, 0.0)[0] - 5.0) for _ in range(10)]
    assert np.mean(fresh) < np.mean(late)


# ---------------------------------------------------------------------------
# Field-side gating
# ---------------------------------------------------------------------------
def test_point_in_polygon():
    poly = br.rectangle(10.0, 25.0)
    assert br.point_in_polygon(5.0, 12.5, poly)
    assert not br.point_in_polygon(-1.0, 12.5, poly)
    assert not br.point_in_polygon(11.0, 12.5, poly)
    assert not br.point_in_polygon(5.0, 30.0, poly)


def test_gate_discards_beyond_wall_returns():
    """The phantom-target case: a return from past the wall must be dropped."""
    poly = br.rectangle(10.0, 25.0)
    bearings = np.array([-90.0, 0.0, 90.0])
    # Port beam sees something 8 m away -- that is 3 m outside the x=0 wall.
    ranges = np.array([8.0, 5.0, 3.0])
    gated = br.gate_beams(ranges, bearings, 5.0, 12.5, 0.0, poly, margin=0.3)

    assert gated[0] == pytest.approx(cfg.LIDAR_RANGE)    # discarded
    assert gated[1] == pytest.approx(5.0)                # inside, kept
    assert gated[2] == pytest.approx(3.0)                # inside, kept


def test_gate_keeps_returns_just_inside_the_wall():
    poly = br.rectangle(10.0, 25.0)
    bearings = np.array([-90.0])
    gated = br.gate_beams(np.array([4.9]), bearings, 5.0, 12.5, 0.0, poly, margin=0.3)
    assert gated[0] == pytest.approx(4.9)


def test_gate_margin_admits_returns_just_outside():
    """The margin is what stops localisation error gating out a real obstacle."""
    poly = br.rectangle(10.0, 25.0)
    bearings = np.array([-90.0])
    # 5.2 m to port is 0.2 m outside the wall -- inside a 0.3 m margin.
    tight = br.gate_beams(np.array([5.2]), bearings, 5.0, 12.5, 0.0, poly, margin=0.0)
    loose = br.gate_beams(np.array([5.2]), bearings, 5.0, 12.5, 0.0, poly, margin=0.5)
    assert tight[0] == pytest.approx(cfg.LIDAR_RANGE)    # gated out
    assert loose[0] == pytest.approx(5.2)                # admitted


def test_gate_leaves_no_return_beams_alone():
    poly = br.rectangle(10.0, 25.0)
    bearings = np.array([-90.0, 0.0])
    ranges = np.array([cfg.LIDAR_RANGE, cfg.LIDAR_RANGE])
    gated = br.gate_beams(ranges, bearings, 5.0, 12.5, 0.0, poly)
    assert np.allclose(gated, cfg.LIDAR_RANGE)


def test_gate_is_a_no_op_on_a_simulated_obstacle_only_scan():
    """Equivalence of the two pipelines (01 §3.4).

    In simulation the raycast never sees the border, so every return is already
    inside the polygon and gating must change nothing.
    """
    rng = np.random.default_rng(3)
    poly = br.rectangle(10.0, 25.0)
    bearings = np.linspace(-180.0, 179.5, 720)
    # Returns that all land well inside the basin.
    ranges = rng.uniform(0.5, 2.0, size=720)
    gated = br.gate_beams(ranges, bearings, 5.0, 12.5, 0.0, poly)
    assert np.allclose(gated, ranges)
