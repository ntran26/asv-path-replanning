"""Version dispatch only; never initialize workers, load a model or run a case."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/trigger_counterfactual.py"
SPEC = importlib.util.spec_from_file_location("trigger_dispatch_test_module", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("version", [4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
def test_requested_filter_is_not_silently_replaced_by_v7(monkeypatch, version):
    sentinel = object()
    monkeypatch.setitem(sys.modules, f"safety_v{version}",
                        SimpleNamespace(**{f"SafetyFilterV{version}": lambda: sentinel}))
    assert runner._version_number(f"v{version}") == version
    assert runner._new_filter(f"v{version}") is sentinel


@pytest.mark.parametrize("name", ["v1", "v5", "v99", "", "safety_v9", "v9,v7"])
def test_unknown_versions_fail_before_a_filter_can_run(name):
    with pytest.raises(ValueError):
        runner._new_filter(name)
