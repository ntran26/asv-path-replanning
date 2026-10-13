"""train_asv.py: what each request runs (train_formulation.py arguments), without training."""
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("train_asv", Path(__file__).resolve().parents[1] / "train_asv.py")
train_asv = importlib.util.module_from_spec(_spec)
sys.modules["train_asv"] = train_asv      # the dataclass looks its module up
_spec.loader.exec_module(train_asv)


def request(**kw):
    base = dict(algo="ppo", seed=0, timesteps=None, resume=False, replay_buffer=None, tag="bl3", smoke=False)
    return Namespace(**{**base, **kw})


def value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def make_run(runs, name, budget, checkpoints, finished=False, algo=None):
    run = runs / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"timesteps": budget}))
    algo = algo or name.split("_formulation")[0]
    for steps in checkpoints:
        (run / f"{algo}_{steps}_steps.zip").write_bytes(b"")
    if finished:
        (run / "final_model.zip").write_bytes(b"")
    return run


def test_new_off_policy_run_trains_to_3m_on_the_2p5m_schedule(tmp_path):
    p = train_asv.plan(request(algo="sac", seed=1), runs_dir=tmp_path, pin=True)
    assert p.run_dir.name == "sac_formulation_seed1_bl3" and p.target == 3_000_000
    assert value(p.argv, "--train-to") == "3000000"
    assert value(p.argv, "--extra-evals") == "2500000"
    assert value(p.argv, "--torch-threads") == "4" and p.pinned
    assert value(p.argv, "--eval-safety-override") == "off"
    assert value(p.argv, "--config").replace("\\", "/") == "configs/baseline_v3.json"


def test_on_policy_runs_unpinned_without_the_extra_evaluation(tmp_path):
    p = train_asv.plan(request(algo="recurrent_ppo"), runs_dir=tmp_path, pin=True)
    assert not p.pinned and "--torch-threads" not in p.argv and "--extra-evals" not in p.argv


def test_budget_run_and_shorter_run(tmp_path):
    assert "--train-to" not in train_asv.plan(request(timesteps=2_500_000), runs_dir=tmp_path).argv
    short = train_asv.plan(request(timesteps=1_000_000), runs_dir=tmp_path)
    assert value(short.argv, "--train-to") == "1000000" and short.notes


def test_an_existing_run_is_not_overwritten(tmp_path):
    make_run(tmp_path, "ppo_formulation_seed0_bl3", 3_000_000, [250_000])
    with pytest.raises(SystemExit, match="--resume"):
        train_asv.plan(request(), runs_dir=tmp_path)
    assert train_asv.plan(request(tag="trial"), runs_dir=tmp_path).run_dir.name == "ppo_formulation_seed0_trial"


def test_resume_continues_from_the_latest_checkpoint_with_an_empty_buffer(tmp_path):
    make_run(tmp_path, "sac_formulation_seed1_bl3", 3_000_000, [1_500_000, 1_750_000])
    p = train_asv.plan(request(algo="sac", seed=1, resume=True), runs_dir=tmp_path, pin=False)
    assert p.resume_from == 1_750_000 and p.target == 3_000_000
    assert value(p.argv, "--resume").replace("\\", "/").endswith("sac_formulation_seed1_bl3")
    assert value(p.argv, "--replay-buffer") == "none" and "--extend-to" not in p.argv


def test_resume_with_a_given_replay_buffer(tmp_path):
    make_run(tmp_path, "tqc_formulation_seed0_bl3", 3_000_000, [1_000_000])
    buffer = tmp_path / "buffer.pkl"
    buffer.write_bytes(b"")
    p = train_asv.plan(request(algo="tqc", resume=True, replay_buffer=str(buffer)), runs_dir=tmp_path, pin=False)
    assert value(p.argv, "--replay-buffer") == str(buffer)


def test_a_finished_run_is_extended_only_past_its_budget(tmp_path):
    make_run(tmp_path, "ppo_formulation_seed2_bl3", 3_000_000, [2_750_000, 3_000_000], finished=True)
    with pytest.raises(SystemExit, match="above 3,000,000"):
        train_asv.plan(request(seed=2, resume=True), runs_dir=tmp_path)
    p = train_asv.plan(request(seed=2, resume=True, timesteps=3_500_000), runs_dir=tmp_path)
    assert value(p.argv, "--extend-to") == "3500000" and "--replay-buffer" not in p.argv
    with pytest.raises(SystemExit, match="cannot be shortened"):
        train_asv.plan(request(seed=2, resume=True, timesteps=2_000_000), runs_dir=tmp_path)


def test_replay_buffer_requests_that_cannot_apply(tmp_path):
    make_run(tmp_path, "ppo_formulation_seed0_bl3", 3_000_000, [250_000])
    with pytest.raises(SystemExit, match="no replay buffer"):
        train_asv.plan(request(resume=True, replay_buffer="saved"), runs_dir=tmp_path)
    with pytest.raises(SystemExit, match="with --resume"):
        train_asv.plan(request(algo="sac", seed=5, replay_buffer="saved"), runs_dir=tmp_path)


def test_running_trainers_are_recognised_from_their_command_lines():
    assert train_asv._run_name(["python", "src/train_formulation.py", "--config", "c.json", "--algo", "sac",
                                "--seed", "1", "--tag", "bl3"]) == "sac_formulation_seed1_bl3"
    assert train_asv._run_name(["python", "src/train_formulation.py", "--resume",
                                "runs/tqc_formulation_seed0_bl3"]) == "tqc_formulation_seed0_bl3"
    assert train_asv._run_name(["python", "train_asv.py", "--algo", "ppo", "--seed", "2"]) == "ppo_formulation_seed2_bl3"
    assert train_asv._run_name(["bash", "results/run_bl3_3m.sh", "sac", "1"]) is None
