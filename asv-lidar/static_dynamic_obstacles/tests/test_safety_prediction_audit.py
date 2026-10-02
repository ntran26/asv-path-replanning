"""Prediction-audit contracts; deterministic checks, no policy or episode runs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "prediction_audit_under_test", _ROOT / "tools/diagnostics/safety/prediction_audit.py")
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("carried_actuators", [False, True])
def test_executed_command_replay_matches_filter_sequence_model(carried_actuators):
    """A changed future command is replayed at its own decision, including brakes."""
    cc, v2, v3 = audit.cc, audit.v2, audit.v3
    snap = cc.Snapshot(2., 3., .3, .7, .02, -.04, np.array([0., 1.]),
                       np.array([1., 0.]), np.zeros(2), 0., 0., 10., np.empty((0, 2)))
    act = cc.Actuators()
    if carried_actuators:
        act.buffer = np.linspace(-.25, .3, cc.DELAY_STEPS).tolist()
        act.servo = .17
    original = copy.deepcopy(act.__dict__)
    controls = np.array([[.5, 0.], [-1., -.5], [1., np.nan], [1., np.nan],
                         [0., np.nan], [0., np.nan], [-.5, .3], [0., 0.],
                         [-1., np.nan], [-1., np.nan], [1., np.nan], [1., np.nan]])
    expected = v3.rollout_seq(snap, act, controls[None])
    commands = list(zip(controls[:, 0], v2._rpm(controls[:, 1])))
    position, heading = audit.replay_commands(snap, act, commands)
    np.testing.assert_allclose(position, expected.positions[:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(heading, expected.headings[:, 0], rtol=0, atol=1e-12)
    assert act.__dict__ == original, "Audit must not advance the live filter's actuators"
    held_position, _ = audit.replay_commands(snap, act, [commands[0]] * len(commands))
    assert np.linalg.norm(position[-1] - held_position[-1]) > .05


def test_zero_thrust_remains_distinct_from_clipped_forward_floor(monkeypatch):
    """The policy-space throttle -1 cannot substitute for actual zero-thrust RPM."""
    # Some curricula deliberately set the floor to zero. Exercise the positive
    # floor configuration explicitly, independent of earlier curriculum tests.
    monkeypatch.setattr(audit.cfg, "RPM_FLOOR", 4.0)
    cc = audit.cc
    snap = cc.Snapshot(0., 0., 0., .7, 0., 0., np.array([0., 1.]),
                       np.array([1., 0.]), np.zeros(2), 0., 0., 10., np.empty((0, 2)))
    act = cc.Actuators()
    zero, _ = audit.replay_commands(snap, act, [(0., 0.)] * 8)
    floor, _ = audit.replay_commands(snap, act, [(0., audit.cfg.RPM_FLOOR)] * 8)
    assert floor[-1, 1] > zero[-1, 1]


def test_selective_scene_cache_preserves_authoritative_indices_seeds_and_kwargs(tmp_path, monkeypatch):
    """Mock only costly sampling; compare the real authoritative 150-case loop."""
    import field_training as ft
    import formulation_v3 as fv

    calls = []

    def fake_sample(rng, **kw):
        generator = kw["generator"]
        signature = (kw["namespace"], kw["encounter"], kw["varying"],
                     int(rng.integers(0, 2 ** 31)), kw["seed_fn"](0), kw["seed_fn"](7),
                     kw["solvable_only"], kw["near"], generator.stage, generator.seed_namespace)
        calls.append(signature)
        return SimpleNamespace(signature=signature)

    monkeypatch.setattr(ft, "sample", fake_sample)
    expected = fv.field_development_set()
    assert len(expected) == 150
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    selected = {"DV3-OT-CV-16", "DV3-BO-CV-04"}
    calls.clear()
    cases = audit.load_development_cases(selected)
    assert len(cases) == len(calls) == 2
    assert [index for index, _ in cases] == [113, 127]
    for index, built in cases:
        assert built.case_id == expected[index].case_id
        assert built.signature == expected[index].signature
    calls.clear()
    cached = audit.load_development_cases(selected)
    assert not calls, "Cache hits must not regenerate scenes or consume scenario RNG"
    assert [(i, b.case_id, b.signature) for i, b in cases] == [(i, b.case_id, b.signature) for i, b in cached]
    monkeypatch.setattr(audit, "DEVELOPMENT_CACHE_SCHEMA", audit.DEVELOPMENT_CACHE_SCHEMA + 1)
    audit.load_development_cases(selected)
    assert len(calls) == 2, "Changing the cache schema invalidates old entries"


def test_cache_tracks_generator_imports_but_ignores_env_and_line_endings(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    generator = src / "formulation_v3.py"
    generator.write_bytes(b"import scene_helper\n")
    helper = src / "scene_helper.py"
    helper.write_bytes(b"VALUE = 1\n")
    env = src / "env.py"
    env.write_bytes(b"FILTER_VERSION = 3\n")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    initial = audit.development_cache_digest()
    env.write_bytes(b"FILTER_VERSION = 4\n")
    generator.write_bytes(b"import scene_helper\r\n")
    helper.write_bytes(b"VALUE = 1\r\n")
    assert audit.development_cache_digest() == initial
    helper.write_bytes(b"VALUE = 2\n")
    assert audit.development_cache_digest() != initial


def test_explicit_cache_import_preserves_sources_and_refuses_conflicts(tmp_path, monkeypatch):
    from scenario import Scenario
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    source = tmp_path / "results/safety_dev/scenario_cache/known_legacy"
    source.mkdir(parents=True)
    built = Scenario(case_id="DV3-OT-CV-16", encounter_class="overtaking", seed=123)
    original = pickle.dumps(built)
    (source / f"{built.case_id}.pkl").write_bytes(original)
    manifest_path, manifest = audit.import_known_cache(source, rationale="Test fixture with known identical scene.")
    assert manifest_path.exists() and manifest["imported"] == 1
    assert manifest["cases"][0]["scenario_sha256"] == built.digest()
    assert (source / f"{built.case_id}.pkl").read_bytes() == original
    _, repeated = audit.import_known_cache(source, rationale="Same fixture imported twice.")
    assert repeated["already_present"] == 1 and repeated["imported"] == 0
    target = Path(manifest["destination"]) / f"{built.case_id}.pkl"
    built.seed = 999
    changed = pickle.dumps(built)
    target.write_bytes(changed)
    with pytest.raises(ValueError, match="Conflicting current-cache"):
        audit.import_known_cache(source, rationale="Conflict must preserve both copies.")
    assert target.read_bytes() == changed
    assert (source / f"{built.case_id}.pkl").read_bytes() == original


@pytest.mark.parametrize(("last", "expected"), [
    ({"mode": "idle"}, "idle"),
    ({"mode": "filter", "why": "nominal"}, "nominal"),
    ({"mode": "filter", "why": "no escape"}, "no escape"),
    ({"mode": "filter", "any_safe": False}, "no escape"),
    ({"mode": "recovery", "why": "last certificate"}, "last certificate"),
    ({"mode": "recovery", "why": "continue"}, "recovery"),
])
def test_contact_mode_classification_retains_idle_and_failed_escape(last, expected):
    assert audit.classify_mode(last) == expected


def test_v4_override_uses_literal_values(monkeypatch):
    fake_v4 = SimpleNamespace(GATE=1.5, ENABLED=True)
    monkeypatch.setitem(sys.modules, "safety_v4", fake_v4)
    monkeypatch.delenv("V2_SET", raising=False)
    monkeypatch.delenv("V3_SET", raising=False)
    monkeypatch.delenv("V5_SET", raising=False)
    monkeypatch.setenv("V4_SET", "GATE=2.25,ENABLED=False")
    assert audit.runtime_overrides() == {"V4_SET:GATE": 2.25, "V4_SET:ENABLED": False}
    assert fake_v4.GATE == 2.25 and fake_v4.ENABLED is False
    monkeypatch.setenv("V4_SET", "GATE=__import__('os').getcwd()")
    with pytest.raises((ValueError, SyntaxError)):
        audit.runtime_overrides()


def test_v5_factory_and_overrides_route_to_new_filter_without_running_episode(monkeypatch):
    expected = object()
    fake_v5 = SimpleNamespace(RECOVERY_ENABLED=False, SafetyFilterV5=lambda: expected)
    monkeypatch.setitem(sys.modules, "safety_v5", fake_v5)
    for key in ("V2_SET", "V3_SET", "V4_SET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("V5_SET", "RECOVERY_ENABLED=True")
    assert audit.runtime_overrides() == {"V5_SET:RECOVERY_ENABLED": True}
    assert fake_v5.RECOVERY_ENABLED is True
    assert audit.make_filter("v5") is expected
    assert audit.make_filter("oracle5") is expected


def test_existing_run_artifacts_are_never_overwritten(tmp_path):
    audit.reserve_run(tmp_path, "run", {"source": "original"})
    with pytest.raises(FileExistsError):
        audit.reserve_run(tmp_path, "run", {"source": "replacement"})
    assert json.loads((tmp_path / "run_metadata.json").read_text()) == {"source": "original"}
    orphan = tmp_path / "old_errors.csv"
    orphan.write_text("evidence\n")
    with pytest.raises(FileExistsError):
        audit.reserve_run(tmp_path, "old", {})
    assert orphan.read_text() == "evidence\n"
    assert not (tmp_path / "old_metadata.json").exists()


def test_metadata_retains_startup_source_hashes(tmp_path, monkeypatch):
    """Source edits after reservation must not rewrite the run's provenance."""
    source_files = ["src/safety_v2.py", "src/safety_v3.py", "src/classical/common.py", "src/env.py", "src/ship.py",
                    "src/constants.py", "src/constant_temp.py", "src/train_formulation.py",
                    "bluefin/dynamics.py", "bluefin/ship_model_v3.py", "tools/diagnostics/safety/prediction_audit.py"]
    for name in source_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# initial source\n")
    model = tmp_path / "policy" / "model.zip"
    model.parent.mkdir()
    model.write_bytes(b"retained checkpoint")
    (model.parent / "config.json").write_text('{"algo": "sac"}')
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    metadata = audit.run_metadata(SimpleNamespace(model=model, stride=4), ["v3"], {})
    output = tmp_path / "output"
    audit.reserve_run(output, "audit", metadata)
    (tmp_path / "src/safety_v3.py").write_text("edited while audit was running\n")
    persisted = json.loads((output / "audit_metadata.json").read_text())
    assert persisted["source_sha256"]["src/safety_v3.py"] == hashlib.sha256(b"# initial source\n").hexdigest()
    assert persisted["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert persisted["model_config_sha256"] == hashlib.sha256((model.parent / "config.json").read_bytes()).hexdigest()
