"""End-to-end environment: the ported env steps, and the observation contract holds.

Revision 2: one dynamic target, 56-dim observation, corridor width as a Study 1
parameter, Study 2 degradation axes as constructor arguments.
"""

import numpy as np
import pytest

import constants as cfg
import observation as obs
from env import ASVLidarEnv, TargetShip


def make_env(**kwargs):
    return ASVLidarEnv(render_mode=None, **kwargs)


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------
def test_reset_returns_a_valid_observation():
    env = make_env()
    o, info = env.reset(seed=0)
    assert env.observation_space.contains(o)
    assert info == {}


def test_env_steps_end_to_end():
    """Build-order step 2: the ported environment still runs a full episode."""
    env = make_env()
    env.reset(seed=0)
    steps = 0
    for _ in range(cfg.MAX_EPISODE_STEPS + 10):
        o, r, terminated, truncated, info = env.step(np.array([0.0, 0.0], np.float32))
        steps += 1
        assert env.observation_space.contains(o)
        assert np.isfinite(r)
        if terminated or truncated:
            break
    assert steps > 1
    assert terminated or truncated


def test_a_full_random_rollout_stays_finite():
    env = make_env()
    rng = np.random.default_rng(0)
    for seed in range(5):
        env.reset(seed=seed)
        for _ in range(120):
            action = rng.uniform(-1, 1, 2).astype(np.float32)
            o, r, term, trunc, info = env.step(action)
            for key, value in o.items():
                assert np.all(np.isfinite(value)), key
            assert np.isfinite(r)
            if term or trunc:
                break


def test_seeded_resets_reproduce_the_same_layout():
    a, b = make_env(), make_env()
    a.reset(seed=42)
    b.reset(seed=42)
    assert a.obstacles == b.obstacles
    assert (a.start_x, a.start_y) == (b.start_x, b.start_y)


def test_action_space_is_the_paper_2_two_channel_box():
    env = make_env()
    assert env.action_space.shape == (2,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


# ---------------------------------------------------------------------------
# Observation contract
# ---------------------------------------------------------------------------
def test_observation_dimension_is_56():
    env = make_env()
    o, _ = env.reset(seed=0)
    assert sum(int(v.size) for v in o.values()) == 56


def test_lidar_branch_is_obstacle_only():
    """D5: the basin walls must be invisible to `c_t`.

    An empty basin with no obstacles must pool to all-clear, even though the
    vessel is only a few metres from a wall.
    """
    env = make_env()
    env.forced_num_obs = 0
    env.reset(seed=1)
    assert np.allclose(env.sector_closeness, 0.0), env.sector_closeness.max()


def test_boundary_branch_does_see_the_walls():
    """The complement of the test above: the walls reach the policy, but
    through the map, not through the sensor."""
    env = make_env()
    env.forced_num_obs = 0
    env.reset(seed=1)
    assert np.any(env.boundary_closeness > 0.0)


def test_an_obstacle_shows_up_in_the_lidar_branch():
    env = make_env()
    env.forced_num_obs = 0
    env.reset(seed=1)
    # Drop a box squarely ahead of the vessel.
    env.obstacles = [[(env.asv_x - 0.5, env.asv_y + 3.0), (env.asv_x + 0.5, env.asv_y + 3.0),
                      (env.asv_x + 0.5, env.asv_y + 4.0), (env.asv_x - 0.5, env.asv_y + 4.0)]]
    env._perceive()
    assert np.max(env.sector_closeness) > 0.0


def test_no_target_gives_a_zero_presence_bit():
    env = make_env(no_target_prob=1.0)
    o, _ = env.reset(seed=0)
    assert env.targets == []
    _, presence = obs.split_target(o["target"])
    assert np.allclose(presence, 0.0)


# ---------------------------------------------------------------------------
# Target ships
# ---------------------------------------------------------------------------
def test_target_ship_moves_at_constant_velocity():
    """D1: constant velocity for training."""
    t = TargetShip(5.0, 20.0, 180.0, 0.5)
    for _ in range(10):
        t.step(cfg.UPDATE_RATE)
    assert t.y == pytest.approx(20.0 - 0.5 * 10 * cfg.UPDATE_RATE)
    assert t.x == pytest.approx(5.0)


def test_target_ship_velocity_follows_its_heading():
    assert TargetShip(0, 0, 0.0, 1.0).velocity == pytest.approx([0.0, 1.0])
    assert TargetShip(0, 0, 90.0, 1.0).velocity == pytest.approx([1.0, 0.0])
    assert TargetShip(0, 0, 180.0, 1.0).velocity == pytest.approx([0.0, -1.0])


def test_a_target_ship_is_eventually_tracked_and_occupies_the_slot():
    """The whole perception chain, end to end: hull -> raycast -> cluster ->
    track -> Kalman -> dynamic -> slot."""
    env = make_env(pose_noise=False)
    env.forced_num_obs = 0
    env.reset(seed=3)
    env.obstacles = []
    # Head-on target, closing from ahead well inside the sensor horizon.
    env.targets = [TargetShip(env.asv_x, env.asv_y + 9.0, 180.0, 0.6)]

    for _ in range(60):
        env.step(np.array([0.0, 0.0], np.float32))
        if env.tracks:
            break

    assert env.tracks, "target ship was never tracked"
    o = env._get_obs()
    _, presence = obs.split_target(o["target"])
    assert float(presence[0]) == 1.0
    assert env.acquisition_range is not None


def test_target_collision_is_reported_separately():
    env = make_env()
    env.reset(seed=0)
    env.obstacles = []
    env.targets = [TargetShip(env.asv_x, env.asv_y, 180.0, 0.0)]   # sitting on us
    kind = env.collision_kind(env.hull_polygon())
    assert kind == "target"


def test_boundary_collision_is_reported_separately():
    env = make_env()
    env.reset(seed=0)
    env.obstacles = []
    env.targets = []
    env.asv_x = -1.0
    assert env.collision_kind(env.hull_polygon()) == "boundary"


def test_obstacle_collision_is_reported_separately():
    env = make_env()
    env.reset(seed=0)
    env.targets = []
    env.obstacles = [[(env.asv_x - 0.5, env.asv_y - 0.5), (env.asv_x + 0.5, env.asv_y - 0.5),
                      (env.asv_x + 0.5, env.asv_y + 0.5), (env.asv_x - 0.5, env.asv_y + 0.5)]]
    assert env.collision_kind(env.hull_polygon()) == "obstacle"


# ---------------------------------------------------------------------------
# Reward placeholder
# ---------------------------------------------------------------------------
def test_reward_is_sparse_terminal_only():
    """02 owns the reward.  Nothing dense may appear here yet."""
    env = make_env()
    env.reset(seed=0)
    _, r, _, _, _ = env.step(np.array([0.0, 0.0], np.float32))
    assert r == 0.0


def test_collision_returns_the_terminal_penalty():
    env = make_env()
    env.reset(seed=0)
    assert env._reward("obstacle", False, False) == cfg.R_COLLISION
    assert env._reward(None, True, False) == cfg.R_GOAL
    assert env._reward(None, False, True) == cfg.R_TIMEOUT


def test_no_paper_2_reward_terms_survive_in_info():
    """Kickoff §8 acceptance check, enforced from the outside."""
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.0, 0.0], np.float32))
    for banned in ("r_pf", "r_oa", "lam", "g_u", "w_chi", "r_heading",
                   "r_border", "r_progress", "r_slow", "r_thrust",
                   "r_cte_recovery", "r_wrong_side", "gamma_e_eff",
                   "block_alpha", "local_target_cte", "side_clearance_diff",
                   "front_clearance"):
        assert banned not in info, banned


# ---------------------------------------------------------------------------
# Pose noise wiring
# ---------------------------------------------------------------------------
def test_pose_noise_is_wired_but_currently_disabled():
    """TODO(05): the hook is on the path; the magnitude is still 0.0."""
    env = make_env()
    env.reset(seed=0)
    assert env._pose_noise is not None
    assert not env._pose_noise.enabled          # TODO(05) values are 0.0
    assert env.estimated_pose() == (env.asv_x, env.asv_y, env.asv_h)


def test_pose_noise_can_be_switched_off_entirely():
    env = make_env(pose_noise=False)
    env.reset(seed=0)
    assert env._pose_noise is None
    assert env.estimated_pose() == (env.asv_x, env.asv_y, env.asv_h)


# ---------------------------------------------------------------------------
# SB3 integration
# ---------------------------------------------------------------------------
def test_sb3_can_build_a_policy_over_this_space():
    """The custom extractor is required: MlpPolicy cannot mask."""
    from stable_baselines3 import SAC
    from features_extractor import policy_kwargs

    env = make_env()
    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs(),
                buffer_size=1000, learning_starts=10, verbose=0)
    o, _ = env.reset(seed=0)
    action, _ = model.predict(o, deterministic=True)
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


def test_sb3_learns_a_few_steps_without_error():
    from stable_baselines3 import SAC
    from features_extractor import policy_kwargs

    env = make_env()
    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs(),
                buffer_size=1000, learning_starts=20, batch_size=8, verbose=0)
    model.learn(total_timesteps=60)


# ---------------------------------------------------------------------------
# Study 1 -- corridor width
# ---------------------------------------------------------------------------
def test_corridor_width_defaults_to_the_basin():
    """O4 resolved: simulation matches the basin, so every width is reproducible."""
    env = make_env()
    assert env.corridor_width == cfg.MAP_WIDTH
    assert env.corridor_breadths == pytest.approx(20.0)


@pytest.mark.parametrize("width", cfg.CORRIDOR_WIDTHS_M)
def test_every_sweep_width_runs(width):
    env = make_env(corridor_width=width)
    env.reset(seed=0)
    info = {}
    for _ in range(20):
        o, r, term, trunc, info = env.step(np.array([0.0, 0.0], np.float32))
        assert env.observation_space.contains(o)
        if term or trunc:
            break
    assert info["corridor_width"] == pytest.approx(width)
    assert info["corridor_breadths"] == pytest.approx(width / cfg.BREADTH)


def test_widths_in_breadths_are_the_declared_sweep():
    """03 §5: 20, 16, 12, 10, 8, 7 B, bracketing the head-on threshold."""
    assert cfg.widths_in_breadths() == (20.0, 16.0, 12.0, 10.0, 8.0, 7.0)


def test_a_narrow_corridor_actually_narrows_the_navigable_space():
    narrow = make_env(corridor_width=4.0)
    lo, hi = narrow.corridor_bounds_x()
    assert hi - lo == pytest.approx(4.0)
    assert lo == pytest.approx(3.0)          # centred in a 10 m basin

    narrow.reset(seed=0)
    narrow.asv_x = lo - 0.5                  # outside the channel
    assert narrow.collision_kind(narrow.hull_polygon()) == "boundary"


def test_boundary_branch_reflects_the_corridor_width():
    """A narrower channel must read as closer walls, not the basin walls."""
    wide = make_env(corridor_width=10.0)
    tight = make_env(corridor_width=4.0)
    wide.forced_num_obs = 0
    tight.forced_num_obs = 0
    wide.reset(seed=1)
    tight.reset(seed=1)
    for env in (wide, tight):
        lo, hi = env.corridor_bounds_x()
        env.asv_x = 0.5 * (lo + hi)
        env.asv_y = 12.0
        env.asv_h = 0.0
        env._perceive()
    assert np.max(tight.boundary_closeness) > np.max(wide.boundary_closeness)


# ---------------------------------------------------------------------------
# Study 2 -- degradation axes
# ---------------------------------------------------------------------------
def test_degradation_axes_default_to_nominal():
    env = make_env()
    assert env.tracker.dropout_p == 0.0
    assert env.tracker.velocity_noise == 0.0
    assert env.lidar.dropout_p == 0.0
    assert env.lidar.aft_mask_half_deg == 0.0
    assert env.ego_speed_noise == 0.0
    assert env.ego_yaw_rate_noise_dps == 0.0


def test_detection_dropout_is_swept_through_the_constructor():
    env = make_env(detection_dropout_p=1.0, pose_noise=False)
    env.forced_num_obs = 0
    env.reset(seed=3)
    env.obstacles = []
    env.targets = [TargetShip(env.asv_x, env.asv_y + 8.0, 180.0, 0.6)]
    for _ in range(40):
        env.step(np.array([0.0, 0.0], np.float32))
    # Everything dropped, so no track can ever form.
    assert env.tracks == []
    assert env.tracker.dropped_detections > 0


def test_lidar_dropout_removes_returns():
    env = make_env(lidar_dropout_p=1.0)
    env.forced_num_obs = 0
    env.reset(seed=1)
    env.obstacles = [[(env.asv_x - 0.5, env.asv_y + 3.0), (env.asv_x + 0.5, env.asv_y + 3.0),
                      (env.asv_x + 0.5, env.asv_y + 4.0), (env.asv_x - 0.5, env.asv_y + 4.0)]]
    env._perceive()
    assert np.allclose(env.sector_closeness, 0.0)


def test_aft_mask_blinds_the_stern_arc():
    """The arc that gates the being-overtaken class."""
    env = make_env(aft_mask_half_deg=30.0)
    masked = env.lidar.aft_mask
    assert masked.sum() > 0
    astern = np.abs(env.lidar.bearings) >= 150.0
    assert np.array_equal(masked, astern)


def test_ego_noise_perturbs_the_ego_branch():
    """No IMU: u, v and r are differentiated from a noisy pose (05 §6)."""
    env = make_env(ego_speed_noise=0.05, ego_yaw_rate_noise_dps=2.0)
    env.reset(seed=0)
    env.u_body, env.v_body, env.asv_w = 0.5, 0.0, 0.0
    samples = [env._measured_ego() for _ in range(30)]
    assert np.std([s[0] for s in samples]) > 0.0
    assert np.std([s[2] for s in samples]) > 0.0


def test_perception_metrics_are_reported():
    """04 §7: acquisition range, occlusion duration, track uptime."""
    env = make_env(pose_noise=False)
    env.forced_num_obs = 0
    env.reset(seed=3)
    env.obstacles = []
    env.targets = [TargetShip(env.asv_x, env.asv_y + 9.0, 180.0, 0.6)]
    info = {}
    for _ in range(40):
        _, _, term, trunc, info = env.step(np.array([0.0, 0.0], np.float32))
        if term or trunc:
            break
    for key in ("acquisition_range", "max_coast_steps", "dropped_detections",
                "steps_target_visible", "steps_target_tracked", "encounter_class"):
        assert key in info, key
    assert info["steps_target_visible"] > 0
