"""The ported Paper 2 harness still runs against the Paper 3 environment.

Build-order step 2.  These modules were assigned to Bucket A ("copy verbatim"),
but their logging schema turned out to be coupled to Paper 2's reward terms and
to the three dropped observation fields, so each needed a stated change.  These
tests are what stops that coupling reappearing silently.
"""

import numpy as np
import pytest

import constants as cfg
import rollout
import train
from env import ASVLidarEnv, TargetShip
from metrics import EpisodeRecorder


class ZeroController:
    def predict(self, obs, deterministic=True, **kwargs):
        return np.zeros(2, dtype=np.float32), None


@pytest.fixture
def env():
    return ASVLidarEnv(render_mode=None)


# ---------------------------------------------------------------------------
# rollout.py
# ---------------------------------------------------------------------------
def test_run_episode_completes(env):
    ep = rollout.run_episode(ZeroController(), env, max_steps=80)
    assert ep.steps > 0
    assert ep.reason in {"goal", "timeout", "obstacle", "boundary", "target", "terminated"}


def test_run_episode_tolerates_missing_info_keys(env):
    """02 will add reward terms to `info`; a missing key must not raise.

    The Paper 2 loop indexed `info[key]` directly, so the first key that went
    away took the whole harness with it.
    """
    ep = rollout.run_episode(ZeroController(), env, max_steps=30)
    for key in rollout._INFO_SERIES:
        assert key in ep.series


def test_tracked_info_keys_are_all_emitted_by_the_env(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    for key in rollout._INFO_SERIES:
        assert key in info, f"rollout tracks {key!r} but the env does not emit it"


def test_forward_beam_stats_uses_the_new_bearing_attribute(env):
    """Paper 2's Lidar exposed `.angles`; the Paper 3 one exposes `.bearings`."""
    env.reset(seed=0)
    stats = rollout.forward_beam_stats(env)
    assert set(stats) == {"min_lidar_all", "p10_front"}
    assert np.isfinite(stats["min_lidar_all"])


def test_termination_reason_distinguishes_the_three_collision_kinds(env):
    env.reset(seed=0)
    for kind in ("boundary", "obstacle", "target"):
        reason = rollout.termination_reason(
            env, {"collided": True, "collision_kind": kind}, False, False)
        assert reason == kind


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------
def test_episode_metrics_matches_its_own_header(env):
    ep = rollout.run_episode(ZeroController(), env, max_steps=60)
    row = train.episode_metrics(ep, env)
    # The caller supplies these three; everything else must come from the row.
    supplied_by_caller = {"timesteps", "eval_group", "eval_episode"}
    missing = [k for k in train.DETAIL_HEADER if k not in row and k not in supplied_by_caller]
    assert not missing, missing
    assert not [k for k in row if k not in train.DETAIL_HEADER]


def test_no_paper_2_reward_columns_remain_in_the_log_schema():
    banned = {"mean_r_pf", "mean_r_oa", "mean_r_local", "mean_r_center",
              "mean_r_border", "mean_r_exist", "mean_lambda", "has_reward_info",
              "reward_info_rate", "min_front_clearance", "max_block_alpha",
              "mean_abs_local_target_cte", "mean_sector_pen", "p10_sector_range"}
    assert not banned & set(train.DETAIL_HEADER)
    assert not banned & set(train.SUMMARY_HEADER)


def test_perception_columns_are_present_in_the_log_schema():
    for column in ("min_sector_range", "max_boundary_closeness",
                   "min_border_clearance", "max_tracks"):
        assert column in train.DETAIL_HEADER, column


def test_the_paper_2_side_choice_guard_is_gone():
    """It read all three dropped observation fields and hard-coded a side
    choice Paper 3 expects the policy to learn."""
    assert not hasattr(train, "side_path_guard")
    assert not hasattr(train, "USE_SIDE_PATH_GUARD")


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------
def test_episode_recorder_runs_and_summarises(env):
    env.reset(seed=0)
    recorder = EpisodeRecorder(env, {"case_id": 0, "num_obstacles": len(env.obstacles)})
    truncated = False
    info = {}
    for _ in range(40):
        _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        recorder.observe(np.zeros(2), reward, info)
        if terminated or truncated:
            break
    summary = recorder.finish(truncated, False)
    assert summary["episode_id"] == 0
    assert np.isfinite(summary["ep_reward"])
    assert "min_front_clearance" not in summary


# ---------------------------------------------------------------------------
# Every ported module imports
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", [
    "ship", "path", "obstacles", "rollout", "metrics", "curriculum", "compare",
    "train", "train_sac_baseline", "train_ppo_baseline",
    "constants", "asv_lidar", "lidar_pooling", "boundary_raycast",
    "tracking", "cpa_cri", "encounter", "observation", "features_extractor", "env",
])
def test_module_imports(module):
    __import__(module)


def test_no_module_reaches_into_the_paper_2_tree():
    """Kickoff §1: a later edit in Paper 3 must not be able to change Paper 2."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "src"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "static_obstacles" not in text, path.name
        assert "paper_pooling" not in text, path.name
