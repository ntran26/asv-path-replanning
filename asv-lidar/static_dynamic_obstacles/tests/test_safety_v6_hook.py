"""Native safety dispatch only; sentinel filters stop before any plant update.

The version6 case requires the native env hook planned after the frozen broad
evaluation. No environment constructor, reset, dynamics or policy is invoked.
"""
import sys
from types import SimpleNamespace

import pytest

import constants as cfg
from env import ASVLidarEnv


class StopBeforePlant(RuntimeError):
    pass


@pytest.mark.parametrize("version", [2, 3, 4, 5, 6])
def test_native_dispatch_selects_one_factory_and_reuses_cached_filter(monkeypatch, version):
    factories, filter_calls, instances = [], [], {}

    class SentinelFilter:
        def __init__(self, selected_version):
            self.selected_version = selected_version

        def filter(self, env, action):
            filter_calls.append((self.selected_version, env, action))
            raise StopBeforePlant("Dispatch observed; deliberately stop before dynamics")

    for selected in range(2, 7):
        def factory(selected_version=selected):
            factories.append(selected_version)
            instance = SentinelFilter(selected_version)
            instances[selected_version] = instance
            return instance

        monkeypatch.setitem(sys.modules, f"safety_v{selected}",
                            SimpleNamespace(**{f"SafetyFilterV{selected}": factory}))

    monkeypatch.setattr(cfg, "SAFETY_VERSION", version, raising=False)
    env = ASVLidarEnv.__new__(ASVLidarEnv)
    env.elapsed_time = 0.0
    env.estop_enabled = True
    action = object()  # Any post-filter command conversion would fail this test.

    for _ in range(2):
        with pytest.raises(StopBeforePlant, match="before dynamics"):
            env.step(action)

    assert factories == [version]
    assert filter_calls == [(version, env, action), (version, env, action)]
    assert env._safety_v2 is instances[version]
    assert env.elapsed_time == 2 * cfg.UPDATE_RATE
