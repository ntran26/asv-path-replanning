"""The ported Paper 2 modules still run against the Paper 3 environment.

Build-order step 2.  These modules were assigned to Bucket A ("copy verbatim"),
but their logging schema turned out to be coupled to Paper 2's reward terms and
to the three dropped observation fields, so each needed a stated change.  These
tests are what stops that coupling reappearing silently.

The tests of `rollout.py` and `train.py` moved with those modules to `archive/` (2026-11-20,
`archive/tests/test_ported_harness_rollout_train.py`).
"""

import numpy as np
import pytest

from env import ASVLidarEnv
from metrics import EpisodeRecorder


@pytest.fixture
def env():
    return ASVLidarEnv(render_mode=None)


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
    "ship", "path", "obstacles", "metrics", "curriculum",
    "constants", "asv_lidar", "lidar_pooling", "boundary_raycast",
    "tracking", "cpa_cri", "encounter", "observation", "features_extractor", "env",
])
def test_module_imports(module):
    __import__(module)


def test_no_module_reaches_into_the_paper_2_tree():
    """Kickoff section 1: a later edit in Paper 3 must not be able to change Paper 2."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "src"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "static_obstacles" not in text, path.name
        assert "paper_pooling" not in text, path.name
