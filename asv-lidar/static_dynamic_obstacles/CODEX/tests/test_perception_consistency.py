"""Regressions for a single synchronized perceived state per decision."""

import numpy as np
import pytest

import boundary_raycast as br
from env import ASVLidarEnv, _normalised_indices
from metrics import EpisodeRecorder


SCENE = {"start": (5., 3.), "goal": (5., 22.), "obstacles": [], "targets": [],
         "path": [[5., y] for y in np.linspace(3., 22., 191)]}


def make_env(**kwargs):
    return ASVLidarEnv(vessel_randomisation=None, pose_stale_prob=0,
                       facility_walls=False, **kwargs)


def test_one_pose_sample_drives_boundary_and_observation(monkeypatch):
    env = make_env(pose_noise=True)
    calls = []
    def perturb(x, y, h):
        calls.append((x, y, h))
        return x + .2, y + .1, h + 4.
    monkeypatch.setattr(env._pose_noise, "perturb", perturb)
    obs, _ = env.reset(seed=12, options={"scenario": SCENE})
    assert len(calls) == 1
    pose = env.estimated_pose()
    context_step = env.observer.contexts.step_index
    np.testing.assert_allclose(env.boundary_closeness,
                               br.boundary_scan(*pose, env.boundary_polygon))
    np.testing.assert_allclose(env.perceived_path_state.cross_track_error, .2, atol=.01)
    for _ in range(3):
        again = env._get_obs()
        for key in obs:
            np.testing.assert_array_equal(again[key], obs[key])
        again["ego"][0] = -123  # callers cannot mutate the cached observation
    assert len(calls) == 1
    assert env.observer.contexts.step_index == context_step
    assert env._get_obs()["ego"][0] != -123
    env.step(np.zeros(2, np.float32))
    assert len(calls) == 2
    assert env.observer.contexts.step_index == context_step + 1


def test_observer_and_tracker_receive_the_same_estimated_origin(monkeypatch):
    env = make_env(pose_noise=True)
    env.reset(seed=13, options={"scenario": SCENE})
    received = {}
    build = env.observer.build
    def capture(**kwargs):
        received.update(kwargs)
        return build(**kwargs)
    monkeypatch.setattr(env.observer, "build", capture)
    env.step(np.zeros(2, np.float32))
    x, y, h = env.estimated_pose()
    assert received["p_os"] == (x, y)
    assert received["heading_os_deg"] == h
    np.testing.assert_array_equal(received["v_os"], env.perceived_velocity)
    assert received["u"] == env._measured_ego()[0]


def test_nondefault_target_slots_match_environment_space():
    env = make_env(n_max_targets=2)
    obs, _ = env.reset(seed=14, options={"scenario": SCENE})
    assert obs["target"].shape == (32,)
    assert obs["context"].shape == (26,)
    assert env.observation_space.contains(obs)
    _, _, _, _, info = env.step(np.zeros(2, np.float32))
    assert info is not None


def test_recovery_start_is_reproducible_and_rejects_invalid_geometry():
    env = make_env(pose_noise=False, ego_speed_noise=0, ego_yaw_rate_noise_dps=0)
    options = {"scenario": SCENE, "initial_lateral_offset_m": .5,
               "recovery_heading_deg": 12., "initial_speed": .2}
    obs1, _ = env.reset(seed=5, options=options)
    assert env.asv_x == pytest.approx(5.5)
    assert env.asv_h == pytest.approx(12.)
    assert env.u_body == pytest.approx(.2)
    obs2, _ = env.reset(seed=5, options=options)
    for key in obs1:
        np.testing.assert_array_equal(obs1[key], obs2[key])
    with pytest.raises(ValueError, match="recovery start intersects"):
        env.reset(seed=5, options={"scenario": SCENE,
                                  "initial_lateral_offset_m": 20.})


def test_context_health_excludes_indicator_and_action_limits():
    np.testing.assert_array_equal(_normalised_indices("context", 14), [3, 4, 11])


def test_goal_waits_for_return_to_path_after_avoidance():
    env = make_env(pose_noise=False)
    env.reset(seed=7, options={"scenario": SCENE})
    env.asv_x, env.asv_y = 6.0, 21.0
    env._update_path_errors(0.0)
    assert not env._reached_goal(), "one metre lateral error is unfinished recovery"
    env.asv_x = 5.5
    env._update_path_errors(0.0)
    assert env._reached_goal()
    env.asv_x, env.asv_y = 5.0, 23.0
    env._update_path_errors(0.0)
    assert not env._reached_goal(), "passing beyond the endpoint is not goal capture"


@pytest.mark.parametrize("kind", ["boundary", "obstacle", "target"])
def test_metrics_preserve_physics_collision_kind(kind):
    env = make_env()
    env.reset(seed=7, options={"scenario": SCENE})
    recorder = EpisodeRecorder(env, {"case_id": 1})
    recorder.last_info = {"collided": True, "collision_kind": kind}
    row = recorder.finish(truncated=True, hit_max_steps=True)
    assert row["term_reason"] == ("border" if kind == "boundary" else kind)
    assert row["target_collision"] == int(kind == "target")
    assert row["obstacle_collision"] == int(kind == "obstacle")
    assert row["border_collision"] == int(kind == "boundary")
    assert not row["timeout"]
