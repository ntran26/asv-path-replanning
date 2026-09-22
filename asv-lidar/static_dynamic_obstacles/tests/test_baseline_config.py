"""F93: the frozen baseline formulation (`configs/baseline_v1.json`)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import baseline_config as bc  # noqa: E402
import constants as cfg  # noqa: E402
import train_formulation as tf  # noqa: E402


def test_code_matches_the_frozen_baseline():
    """A formulation change after the freeze must be a new config version, not a
    silent drift under the campaign."""
    assert bc.check(cfg, tf) == []


def test_run_12_switches_are_off_in_the_baseline():
    switches = bc.load()["formulation"]["switches"]
    assert switches["V_PORT_HEADING_NEEDS_ADMISSIBLE"] is False     # F91 off (F92)
    assert switches["STAGE3_WEIGHTS"] is None                       # A31 off (F92)
    assert switches["V_PORT_LATCHED_RHO"] and switches["V_HOLD_GROWS"]


def test_a_changed_constant_is_caught(monkeypatch):
    monkeypatch.setattr(cfg, "V_HOLD_CAP", cfg.V_HOLD_CAP + 1.0)
    assert any("V_HOLD_CAP" in p for p in bc.check(cfg, tf))


def test_every_campaign_learner_has_its_settings():
    learners = bc.load()["learners"]
    assert set(learners) == set(tf.ALGORITHMS) == set(bc.ALGOS)
    for spec in learners.values():
        assert spec["hyperparameters"]["gamma"] == pytest.approx(tf.PPO_HYPERPARAMS["gamma"])


@pytest.mark.parametrize("seed", [0, 1])
def test_run_11_is_the_baseline_formulation(seed):
    run = ROOT / "runs" / f"ppo_formulation_seed{seed}_v11"
    if not (run / "config.json").exists():
        pytest.skip("run 11 not present")
    assert bc.verify_run(run, bc.load()) == []
