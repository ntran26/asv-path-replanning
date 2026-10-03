"""Saved-cache identity checks only; no environment, policy or generator calls."""
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/diagnostics/safety/v9_paired_pilot.py"
spec = importlib.util.spec_from_file_location("ts3_scene_loader_test", SCRIPT)
loader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = loader
spec.loader.exec_module(loader)


class SavedScene:
    def __init__(self, digest):
        self.value = digest

    def digest(self):
        return self.value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def save_ts3(root, entries=None):
    entries = entries or [dict(test_id="SHARED", episode_seed=31, built=SavedScene("3" * 64)),
                          dict(test_id="ONLY3", episode_seed=32, built=SavedScene("4" * 64))]
    directory = root / "results/test_set/v3"
    directory.mkdir(parents=True, exist_ok=True)
    digests = {row["test_id"]: row["built"].digest() for row in entries}
    metadata = dict(version="3.0", episodes=len(entries), manifest_digest=loader.sha(
        json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()))
    write_json(directory / "definition.json", metadata)
    (directory / "set_v3.0.pkl").write_bytes(pickle.dumps((entries, metadata)))
    with (directory / "definition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_id", "episode_seed", "digest"])
        writer.writeheader()
        writer.writerows(dict(test_id=r["test_id"], episode_seed=r["episode_seed"],
                              digest=r["built"].digest()) for r in entries)
    return directory


def selection(case="TS3:SHARED", seed=31, digest="3" * 64):
    return dict(case=case, dataset=case.split(":")[0].lower(), seed=seed, scenario_sha256=digest)


def save_legacy(root):
    ts2 = root / "results/test_set/v2/set_v2.0.pkl"
    ts2.parent.mkdir(parents=True)
    ts2.write_bytes(pickle.dumps(([dict(test_id="SHARED", episode_seed=21,
                                      built=SavedScene("2" * 64))], {})))
    inventory = root / "results/safety_dev/suites/inventory_v5/manifest.json"
    write_json(inventory, dict(cases=[dict(suite="dev_field", case="A", seed=41,
                                          scenario_sha256="5" * 64)]))
    cached = root / "results/safety_dev/scenario_cache/frozen/A.pkl"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(pickle.dumps(SavedScene("5" * 64)))
    return ts2, cached


def test_ts3_only_reads_its_three_saved_files(tmp_path, monkeypatch):
    save_ts3(tmp_path)
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    scenes, hashes = loader.load_scenes([selection()])
    assert scenes["TS3:SHARED"].digest() == "3" * 64
    assert set(hashes) == {"results/test_set/v3/" + name for name in
                           ["set_v3.0.pkl", "definition.json", "definition.csv"]}
    assert all(loader.sha((tmp_path / name).read_bytes()) == digest for name, digest in hashes.items())


def test_namespaced_v2_v3_and_dv3_keep_exact_seed_geometry_and_legacy_provenance(tmp_path, monkeypatch):
    save_ts3(tmp_path)
    ts2, cached = save_legacy(tmp_path)
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    legacy = [selection("TS2:SHARED", 21, "2" * 64), selection("DV3:A", 41, "5" * 64)]
    before, hashes = loader.load_scenes(legacy)
    assert hashes == {p.relative_to(tmp_path).as_posix(): loader.sha(p.read_bytes()) for p in [ts2, cached]}
    mixed, mixed_hashes = loader.load_scenes(legacy + [selection()])
    assert {case: scene.digest() for case, scene in before.items()} == {
        case: mixed[case].digest() for case in before}
    assert mixed["TS3:SHARED"].digest() != mixed["TS2:SHARED"].digest()
    assert all(mixed_hashes[name] == value for name, value in hashes.items())


@pytest.mark.parametrize("change", [dict(seed=21), dict(scenario_sha256="2" * 64), dict(dataset="ts2")])
def test_ts3_rejects_selected_v2_identity_even_when_bare_id_matches(tmp_path, monkeypatch, change):
    save_ts3(tmp_path)
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    row = selection() | change
    with pytest.raises(ValueError, match="mismatch"):
        loader.load_scenes([row])


@pytest.mark.parametrize("field,value", [("episode_seed", "99"), ("digest", "0" * 64), ("test_id", "UNKNOWN")])
def test_ts3_rejects_changed_definition_rows(tmp_path, monkeypatch, field, value):
    directory = save_ts3(tmp_path)
    path = directory / "definition.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        loader.load_scenes([selection()])


def test_ts3_rejects_duplicate_ids(tmp_path, monkeypatch):
    entry = dict(test_id="SHARED", episode_seed=31, built=SavedScene("3" * 64))
    save_ts3(tmp_path, [entry, entry])
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        loader.load_scenes([selection()])


@pytest.mark.parametrize("kind", ["metadata", "manifest"])
def test_ts3_rejects_metadata_and_manifest_drift(tmp_path, monkeypatch, kind):
    directory = save_ts3(tmp_path)
    path = directory / "definition.json"
    data = json.loads(path.read_text())
    data["manifest_digest"] = "0" * 64
    write_json(path, data)
    if kind == "manifest":
        path = directory / "set_v3.0.pkl"
        entries, _ = pickle.loads(path.read_bytes())
        path.write_bytes(pickle.dumps((entries, data)))
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="metadata|manifest"):
        loader.load_scenes([selection()])


def test_ts3_missing_cache_refuses_regeneration_and_fixed_pilot_defaults_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_scenes([selection()])
    assert loader.OUT.name == "v9_paired_pilot"
    assert loader.MODES == ("off", "v8", "v9")


@pytest.mark.parametrize("cases,match", [([selection(), selection()], "Duplicate"),
                                         ([selection("TS4:SHARED")], "namespace")])
def test_rejects_duplicate_selection_and_unknown_namespace(tmp_path, monkeypatch, cases, match):
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    with pytest.raises(ValueError, match=match):
        loader.load_scenes(cases)
