"""Path geometry: the cross-track sign convention (T2) and `r_path` (T3)."""

import numpy as np
import pytest

import constants as cfg
from env import ASVLidarEnv
from path import ReferencePath, straight_points


def north_path():
    """A due-north reference path down the middle of the basin."""
    return ReferencePath(straight_points(5.0, 0.0, 5.0, 20.0))


# ---------------------------------------------------------------------------
# T2 -- cross-track error is positive to starboard
# ---------------------------------------------------------------------------
def test_cross_track_sign_is_positive_to_starboard():
    """02b C4 / F17.1, the exact case that was measured.

    Paper 2 used positive-to-port, which its own `verify_los_apf.py` flagged as
    non-standard.  Flipped because 02a §6.4's passing-side term has two opposite
    branches keyed on the sign of the lateral offset.
    """
    p = north_path()
    assert p.project(6.0, 10.0, 0.0).cross_track_error == pytest.approx(1.01, abs=0.02)
    assert p.project(4.0, 10.0, 0.0).cross_track_error == pytest.approx(-1.01, abs=0.02)


def test_cross_track_is_zero_on_track():
    assert north_path().project(5.0, 10.0, 0.0).cross_track_error == pytest.approx(0.0, abs=1e-6)


def test_cross_track_sign_follows_the_path_not_the_world():
    """A south-bound path must give the opposite world-frame sign."""
    south = ReferencePath(straight_points(5.0, 20.0, 5.0, 0.0))
    # +x is to *port* of a south-bound vessel.
    assert south.project(6.0, 10.0, 180.0).cross_track_error < 0.0


def test_the_observation_carries_the_flipped_sign():
    env = ASVLidarEnv(render_mode=None, pose_noise=False)
    env.forced_num_obs = 0
    env.reset(seed=0)
    env.asv_x = env.path.points[env.closest_idx][0] + 1.0     # 1 m to starboard
    env._update_path_errors(env.asv_h)
    assert env.cross_track_error > 0.0
    assert env._get_obs()["path"][0] > 0.0


def test_goal_acceptance_is_unaffected_by_the_flip():
    """It compares |cte|, so the flip must not move the goal region."""
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=0)
    env.asv_x, env.asv_y = env.goal_x, env.goal_y
    env.distance_to_goal = 0.0
    assert env._reached_goal()


# ---------------------------------------------------------------------------
# T3 -- r_path
# ---------------------------------------------------------------------------
def arc(radius: float, starboard: bool, n: int = 200):
    """A circular arc starting due north; curvature is exactly 1/radius."""
    sgn = 1.0 if starboard else -1.0
    th = np.linspace(0.0, 0.6, n)
    x = sgn * radius * (1.0 - np.cos(th))
    y = radius * np.sin(th)
    return ReferencePath(np.column_stack([x, y]).astype(np.float32))


def test_straight_path_has_zero_curvature():
    assert north_path().curvature(20) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("radius", [5.0, 10.0, 25.0])
def test_curvature_matches_a_known_arc(radius):
    """Tolerance is set by float32 vertex storage, not by the formula.

    `ReferencePath.points` is float32 (Paper 2's choice), so differencing
    closely-spaced vertices loses relative precision -- and it gets *worse* with
    more points, not better.  5e-3 covers the range of sampling densities the
    generator produces.
    """
    assert arc(radius, starboard=True).curvature(100) == pytest.approx(1.0 / radius, rel=5e-3)


def test_straight_path_curvature_is_exactly_zero():
    """The deadband exists so float32 noise cannot leak into `r_path`.

    Without it a nominally straight path carries ~2e-5 1/m -- a 49 km turn
    radius -- and `R-8`'s `r - r_path` would be non-zero everywhere for no
    physical reason.
    """
    for sx, gx in ((5.0, 5.0), (3.0, 7.0), (7.0, 2.5)):
        p = ReferencePath(straight_points(sx, 2.0, gx, 22.0))
        assert max(abs(p.curvature(i)) for i in range(1, len(p.points) - 1)) == 0.0


def test_curvature_is_positive_to_starboard():
    """Matching `r > 0` = starboard, so `r - r_path` is a like-for-like difference."""
    assert arc(10.0, starboard=True).curvature(100) > 0.0
    assert arc(10.0, starboard=False).curvature(100) < 0.0


def test_r_path_is_speed_times_curvature():
    p = arc(10.0, starboard=True)
    for speed in (0.0, 0.5, cfg.U_REF):
        assert p.yaw_rate_for_tracking(100, speed) == pytest.approx(speed * 0.1, rel=1e-3)


def test_r_path_is_reported_in_radians_alongside_the_yaw_rate():
    """02a `R-8` differences the two; mixed units would be silent and wrong."""
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert "r_path_radps" in info
    assert "yaw_rate_radps" in info
    assert info["yaw_rate_radps"] == pytest.approx(np.radians(info["yaw_rate_dps"]))


def test_r_path_is_zero_on_a_straight_corridor():
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=0)
    for _ in range(20):
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["r_path_radps"] == pytest.approx(0.0, abs=1e-9)
        if term or trunc:
            break


@pytest.mark.xfail(reason="needs 03's corridor generator: with kappa = 0 everywhere, "
                          "R-8 silently reduces to the absolute form and the term "
                          "would look implemented while being untested (02b §3.3)",
                   strict=True)
def test_r_path_is_nonzero_somewhere_in_the_scenario_distribution():
    """The `R-8` regression test 02a §10.4 asks for.

    Deliberately failing rather than absent: a term that is implemented,
    untested and silently inert is worse than one that is missing.
    """
    env = ASVLidarEnv(render_mode=None)
    seen = 0.0
    for seed in range(20):
        env.reset(seed=seed)
        for _ in range(60):
            _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            seen = max(seen, abs(info["r_path_radps"]))
            if term or trunc:
                break
    # 1e-3 rad/s at cruise is a ~1 km turn radius: far above the float32 noise
    # floor the deadband suppresses, far below any bend that fits in the basin.
    assert seen > 1e-3, "no episode in the distribution bends the path"
