"""The quick-start harness: helm behaviour, named geometries, headless smoke test."""

import math

import numpy as np
import pytest

import constants as cfg
import encounter as enc
import play


# ---------------------------------------------------------------------------
# Helm
# ---------------------------------------------------------------------------
class FakeKeys(dict):
    """Stand-in for `pygame.key.get_pressed()`."""

    def __getitem__(self, key):
        return self.get(key, False)


@pytest.fixture
def keys():
    pygame = pytest.importorskip("pygame")
    return pygame


def test_helm_starts_at_cruise(keys):
    """The vessel must make way from step one, so steering is all you do."""
    helm = play.Helm()
    assert helm.throttle == 0.0
    rpm = cfg.CRUISE_RPM + cfg.RPM_DELTA * helm.throttle
    assert rpm == pytest.approx(cfg.CRUISE_RPM)


def test_right_arrow_puts_the_helm_to_starboard(keys):
    helm = play.Helm()
    for _ in range(5):
        action = helm.update(FakeKeys({keys.K_RIGHT: True}))
    assert action[0] == pytest.approx(5 * play.RUDDER_RATE)
    assert action[0] > 0.0


def test_left_arrow_puts_the_helm_to_port(keys):
    helm = play.Helm()
    for _ in range(5):
        action = helm.update(FakeKeys({keys.K_LEFT: True}))
    assert action[0] < 0.0


def test_the_helm_holds_when_the_key_is_released(keys):
    """Held, not spring-centred -- a steady rate of turn is what you test."""
    helm = play.Helm()
    for _ in range(4):
        helm.update(FakeKeys({keys.K_RIGHT: True}))
    held = helm.rudder
    for _ in range(10):
        action = helm.update(FakeKeys())
    assert action[0] == pytest.approx(held)


def test_throttle_moves_and_is_clamped_to_the_rpm_authority(keys):
    helm = play.Helm()
    for _ in range(100):
        helm.update(FakeKeys({keys.K_UP: True}))
    assert helm.throttle == pytest.approx(1.0)
    assert cfg.CRUISE_RPM + cfg.RPM_DELTA * helm.throttle == pytest.approx(cfg.RPM_CEIL)

    for _ in range(200):
        helm.update(FakeKeys({keys.K_DOWN: True}))
    assert helm.throttle == pytest.approx(-1.0)
    assert cfg.CRUISE_RPM + cfg.RPM_DELTA * helm.throttle == pytest.approx(cfg.RPM_FLOOR)


def test_helm_action_is_always_in_the_action_space(keys):
    from env import ASVLidarEnv
    env = ASVLidarEnv(render_mode=None)
    helm = play.Helm()
    rng = np.random.default_rng(0)
    for _ in range(200):
        pressed = FakeKeys({k: bool(rng.integers(2)) for k in
                            (keys.K_LEFT, keys.K_RIGHT, keys.K_UP, keys.K_DOWN)})
        action = helm.update(pressed)
        assert env.action_space.contains(action)


def test_centre_and_cruise_reset_the_helm(keys):
    helm = play.Helm()
    for _ in range(6):
        helm.update(FakeKeys({keys.K_RIGHT: True, keys.K_UP: True}))
    assert helm.rudder != 0.0 and helm.throttle != 0.0
    helm.centre()
    helm.cruise()
    assert helm.rudder == 0.0 and helm.throttle == 0.0


# ---------------------------------------------------------------------------
# Named encounter geometries
# ---------------------------------------------------------------------------
def _built(target, **overrides):
    argv = ["--mode", "random", "--no-render", "--obstacles", "0", "--target", target]
    for k, v in overrides.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    args = play.parse_args(argv)
    env = play.build_env(args)
    play.reset_env(env, args, 1)
    return env


@pytest.mark.parametrize("kind,expected", [
    ("head_on", enc.HEAD_ON),
    ("crossing_stbd", enc.CROSSING),
    ("crossing_port", enc.CROSSING),
    ("overtaking", enc.OVERTAKING),
    ("being_overtaken", enc.BEING_OVERTAKEN),
])
def test_named_geometry_classifies_as_intended(kind, expected):
    """The --target flag has to actually produce the encounter it names."""
    env = _built(kind)
    target = env.targets[0]
    got = enc.classify((env.asv_x, env.asv_y), env.asv_h, cfg.U_CRUISE,
                       (target.x, target.y), target.heading, target.speed)
    assert got == expected


def test_crossing_geometries_differ_only_by_side():
    """Rule 9(b) collapses the class; the side is still recoverable for 02."""
    stbd = _built("crossing_stbd")
    port = _built("crossing_port")
    for env, side in ((stbd, enc.SIDE_STARBOARD), (port, enc.SIDE_PORT)):
        t = env.targets[0]
        assert enc.crossing_side((env.asv_x, env.asv_y), env.asv_h,
                                 (t.x, t.y), t.heading) == side


def test_target_none_spawns_nothing():
    assert _built("none").targets == []


# ---------------------------------------------------------------------------
# Headless smoke test
# ---------------------------------------------------------------------------
def test_random_mode_runs_and_reports_a_clean_contract(capsys):
    code = play.main(["--mode", "random", "--no-render", "--episodes", "2", "--seed", "0"])
    assert code == 0
    assert "observation contract held" in capsys.readouterr().out


def test_the_contract_check_would_catch_a_bad_observation():
    """The check has to be able to fail, or it is decoration."""
    from env import ASVLidarEnv
    env = ASVLidarEnv(render_mode=None)
    obs, _ = env.reset(seed=0)
    assert play.check_observation(env, obs, 0) == []

    obs["lidar"] = obs["lidar"] * np.float32(5.0)      # out of [0, 1]
    assert play.check_observation(env, obs, 1)

    obs, _ = env.reset(seed=0)
    obs["ego"] = obs["ego"].astype(np.float64)
    assert play.check_observation(env, obs, 2)

    obs, _ = env.reset(seed=0)
    del obs["target"]
    assert play.check_observation(env, obs, 3)


@pytest.mark.parametrize("width", cfg.CORRIDOR_WIDTHS_M)
def test_every_sweep_width_runs_through_the_cli(width):
    code = play.main(["--mode", "random", "--no-render", "--episodes", "1",
                      "--seed", "2", "--corridor-width", str(width)])
    assert code == 0


# ---------------------------------------------------------------------------
# Study 2 knobs reach the pipeline
# ---------------------------------------------------------------------------
def test_degradation_flags_reach_the_env():
    args = play.parse_args(["--no-render", "--pose-noise", "0.1",
                            "--detection-dropout", "0.4", "--velocity-noise", "0.2",
                            "--lidar-dropout", "0.3", "--aft-mask", "30"])
    env = play.build_env(args)
    assert env._pose_noise.sigma_xy == pytest.approx(0.1)
    assert env._pose_noise.enabled
    assert env.tracker.dropout_p == pytest.approx(0.4)
    assert env.tracker.velocity_noise == pytest.approx(0.2)
    assert env.lidar.dropout_p == pytest.approx(0.3)
    assert env.lidar.aft_mask_half_deg == pytest.approx(30.0)


def test_pose_noise_degrades_tracking_of_a_head_on_target():
    """Study 2's first axis, visible end to end."""
    clean = _built("head_on")
    noisy = _built("head_on", pose_noise=0.10)
    for env in (clean, noisy):
        for _ in range(60):
            _, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
            if term or trunc:
                break
    assert clean.steps_target_tracked > noisy.steps_target_tracked


def test_the_aft_mask_blinds_the_being_overtaken_class():
    """Why `LIDAR_AFT_MASK_HALF_DEG` is not a cosmetic constant.

    A vessel overtaking from astern sits squarely in the masked arc.  Mask it
    and the target is never acquired at all -- so Rule 17 behaviour would fail
    in the field for reasons that have nothing to do with the policy, which is
    exactly what 01 §2.3 warns about.
    """
    seen = _built("being_overtaken")
    blind = _built("being_overtaken", aft_mask=45.0)
    for env in (seen, blind):
        for _ in range(80):
            _, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
            if term or trunc:
                break
    assert seen.acquisition_range is not None
    assert blind.acquisition_range is None
    assert blind.steps_target_tracked == 0
