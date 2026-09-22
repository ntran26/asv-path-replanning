"""The five learners share one trainer, one evaluation and one actor (F76)."""

import numpy as np
import pytest
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv

import curriculum
import train_formulation as tf
from env import ASVLidarEnv
from features_extractor import ASVFeaturesExtractor


def _vec():
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    return DummyVecEnv([lambda: ASVLidarEnv(render_mode=None, scenario_stage=5)])


def test_the_baseline_set_is_complete():
    assert set(tf.ALGORITHMS) == {"ppo", "recurrent_ppo", "td3", "sac", "tqc"}
    assert set(tf.OFF_POLICY) == {"td3", "sac", "tqc"}


@pytest.mark.parametrize("algo", ["ppo", "recurrent_ppo", "td3", "sac", "tqc"])
def test_every_learner_builds_and_acts_on_the_dict_observation(algo):
    vec = _vec()
    extractor = dict(features_extractor_class=ASVFeaturesExtractor, activation_fn=nn.ReLU)
    if algo in tf.OFF_POLICY:
        model = tf.ALGORITHMS[algo]("MultiInputPolicy", vec, device="cpu", learning_starts=10,
                                    buffer_size=500, batch_size=16,
                                    policy_kwargs=dict(extractor, net_arch=[32, 32]))
    elif algo == "recurrent_ppo":
        model = tf.ALGORITHMS[algo]("MultiInputLstmPolicy", vec, device="cpu", n_steps=16, batch_size=16,
                                    policy_kwargs=dict(extractor, lstm_hidden_size=16))
    else:
        model = tf.ALGORITHMS[algo]("MultiInputPolicy", vec, device="cpu", n_steps=16, batch_size=16,
                                    policy_kwargs=extractor)
    env = ASVLidarEnv(render_mode=None, scenario_stage=5)
    obs, _ = env.reset(seed=0)
    actor = tf.EpisodeActor(model)
    action = actor(obs)
    assert np.asarray(action).shape == (2,)
    assert env.action_space.contains(np.asarray(action, dtype=np.float32))


def test_recurrent_evaluation_carries_the_lstm_state_through_the_episode():
    vec = _vec()
    model = tf.ALGORITHMS["recurrent_ppo"](
        "MultiInputLstmPolicy", vec, device="cpu", n_steps=16, batch_size=16,
        policy_kwargs=dict(features_extractor_class=ASVFeaturesExtractor, lstm_hidden_size=16))
    env = ASVLidarEnv(render_mode=None, scenario_stage=5)
    obs, _ = env.reset(seed=0)
    actor = tf.EpisodeActor(model)
    assert actor.recurrent and actor.state is None
    actor(obs)
    first = [np.array(s, copy=True) for s in actor.state]
    obs, *_ = env.step(np.zeros(2, dtype=np.float32))
    actor(obs)
    assert actor.state is not None and not actor.start[0]
    assert any(not np.allclose(a, b) for a, b in zip(first, actor.state))
    actor.reset()
    assert actor.state is None and actor.start[0]


def test_replay_buffers_live_outside_the_repository_and_only_the_latest_is_kept(tmp_path):
    """Your call (2026-09-22): ~0.5 GB buffers never enter the repository."""
    from pathlib import Path
    import gymnasium as gym
    from stable_baselines3 import SAC
    import train_formulation as tf
    repo = Path(__file__).resolve().parents[1]
    default = tf.REPLAY_BUFFER_DIR.resolve()
    assert repo not in default.parents and tf.REPO_ROOT.resolve() not in default.parents
    model = SAC("MlpPolicy", gym.make("Pendulum-v1"), buffer_size=64, learning_starts=0, device="cpu")
    cb = tf.ReplayBufferCheckpoint(tmp_path / "runs" / "sac_x", "sac", save_freq=1, base=tmp_path / "buf")
    cb.init_callback(model)
    for steps in (250, 500):
        model.num_timesteps = steps
        cb.save()
    kept = sorted(p.name for p in (tmp_path / "buf" / "sac_x").iterdir())
    assert kept == ["sac_replay_buffer_500_steps.pkl"]
