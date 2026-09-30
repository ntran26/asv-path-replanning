"""baseline-v3 (prepared 2026-09-28, F107): the overlay, its stages, and feasibility.

Pinned: the space-time check separates solvable from unsolvable encounters;
the overlay adds stages 6-7 without touching baseline-v2 (its digest and check);
v3 stage-7 episodes mix field layouts, relax the CPA guard, weight 3-panel
clutter, and are all solvable.  Anything that installs the overlay runs in a
subprocess, so it cannot leak into tests that assume baseline-v2.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
import feasibility_st as fs
import formulation_v3 as fv

ROOT = Path(__file__).resolve().parents[1]


def _box(x, y, h=0.5):
    return [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]


def test_space_time_check_separates_solvable_from_unsolvable():
    assert fs.solvable((5, 2), (5, 22), [])["solvable"]
    wall = [_box(x + 0.5, 12) for x in np.arange(0, 10, 1.0)]
    assert not fs.solvable((5, 2), (5, 22), wall)["solvable"]
    gate = [_box(x, 12) for x in (0.5, 1.5, 2.5, 3.5, 6.5, 7.5, 8.5, 9.5)]
    parked = SimpleNamespace(encounter_class="head_on", target_spawn=(5.0, 12.0), target_heading=180.0,
                             target_speed=0.0, flags={})
    res = fs.solvable((5, 2), (5, 22), gate, parked)
    assert res["static_route"] and not res["solvable"]          # A* alone would pass it
    passing = SimpleNamespace(encounter_class="head_on", target_spawn=(5.0, 20.0), target_heading=180.0,
                              target_speed=0.5, flags={"target_stop_box": [0.5, 9.5, 0.5, 24.5]})
    res = fs.solvable((5, 2), (5, 22), gate, passing)
    assert res["solvable"] and res["t_goal_s"] > 20.0           # it waits for the target


def test_overlay_stages_leave_v2_untouched():
    st = fv.stages(cfg)                                         # a copy; cfg is not changed
    assert set(st) == {1, 2, 3, 4, 5, 6, 7}
    for k in (1, 2, 3, 4):
        assert st[k] == cfg.CURRICULUM_STAGES[k]
    assert st[6]["cpa_guard"] == 0.2 and st[7]["cpa_guard"] == 0.0
    assert st[6]["field_share"] == 0.25 and st[7]["field_share"] == 0.45
    assert st[7]["field_varying_share"] > 0 and st[6]["field_varying_share"] == 0
    assert st[6]["st_feasibility"] and st[7]["st_feasibility"]
    assert set(cfg.CURRICULUM_STAGES) == {1, 2, 3, 4, 5}         # the live module is v2's
    assert "clutter_weights" not in cfg.CURRICULUM_STAGES[5]
    # v3 stages 1-5 start at v2's absolute step counts (2 M budget vs 2.5 M)
    v2 = dict((s, f * 2_000_000) for f, s in cfg.CURRICULUM_STAGE_FRACTIONS)
    v3 = dict((s, f * fv.TIMESTEPS) for f, s in fv.STAGE_FRACTIONS)
    for s in (1, 2, 3, 4, 5):
        assert v3[s] == pytest.approx(v2[s])


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_both_configs_check_in_fresh_processes():
    assert "code matches baseline-v2" in _run(
        "import subprocess,sys; r=subprocess.run([sys.executable,'src/baseline_config.py','--check'],"
        "capture_output=True,text=True); print(r.stdout)")
    assert "code matches baseline-v3" in _run(
        "import subprocess,sys; r=subprocess.run([sys.executable,'src/baseline_config.py','--check',"
        "'--config','configs/baseline_v3.json'],capture_output=True,text=True); print(r.stdout)")
    v2 = json.loads((ROOT / "configs" / "baseline_v2.json").read_text())
    assert v2["formulation_digest"] == "3d697858e95e5adf"


def test_stage_seven_episodes_are_mixed_relaxed_and_solvable():
    code = """
import json, numpy as np
import constants as cfg, curriculum, formulation_v3 as fv, feasibility_st as fs, train_formulation as tf
fv.apply(cfg); curriculum.apply_stage(tf.PROPULSION_STAGE)
from env import ASVLidarEnv
env = ASVLidarEnv(render_mode=None, scenario_stage=7)
env.reset(seed=1)
env.set_scenario_stage(7)
field, n, three, unsolvable = 0, 40, 0, 0
for i in range(n):
    env.reset(seed=1000 + i)
    b = env.scenario
    field += (b.flags or {}).get("set") == "field_training"
    three += len(env.obstacles) == 3
    if b.encounter_class != "no_target":
        unsolvable += not fs.solvable((env.start_x, env.start_y), (env.goal_x, env.goal_y), env.obstacles, b)["solvable"]
print(json.dumps({"field": field, "n": n, "three": three, "unsolvable": unsolvable,
                  "mix": env.field_mix, "guard": env._stage_param("cpa_guard", None)}))
"""
    r = json.loads(_run(code).strip().splitlines()[-1])
    assert r["mix"] == 0.45 and r["guard"] == 0.0
    assert 8 <= r["field"] <= 30                       # ~45 % of 40
    assert r["three"] >= 15                            # 3 panels dominate
    assert r["unsolvable"] == 0


def test_clutter_weights_are_honoured():
    code = """
import numpy as np, constants as cfg, formulation_v3 as fv, scenario as scn
fv.apply(cfg)
g = scn.ScenarioGenerator(stage=7, seed_namespace="training")
rng = np.random.default_rng(0)
counts = [g._sample_obstacle_count(rng) for _ in range(4000)]
print({k: counts.count(k) / 4000 for k in (0, 1, 2, 3)})
"""
    share = eval(_run(code).strip().splitlines()[-1])
    assert share[0] == 0.0 and abs(share[3] - 0.5) < 0.04 and abs(share[1] - 0.2) < 0.04


def test_near_deployment_layouts_are_close_but_never_the_tested_layout():
    import field_training as ft
    import paper2_set as p2
    rng = np.random.default_rng(11)
    got = [l for l in (ft.sample_near_layout(rng) for _ in range(300)) if l is not None]
    assert len(got) > 150
    for l in got:
        d = min(ft.layout_distance(l["start"], l["goal"], l["centres"], lay) for lay in p2.LAYOUTS.values())
        assert ft.NEAR_DISTANCE_M[0] <= d <= ft.NEAR_DISTANCE_M[1]
        assert not ft._near_deployment_layout(l["start"], l["goal"], l["centres"])
    st = fv.stages(cfg)
    assert st[6]["field_near_share"] == st[7]["field_near_share"] == fv.NEAR_SHARE > 0
    built = ft.sample(np.random.default_rng(2), encounter="HO", varying=False, near=True, solvable_only=True)
    assert built.flags["motif"].startswith("near-")


def test_stage8_failure_replay_and_spec_stage():
    code = """
import json, numpy as np
import constants as cfg, curriculum, formulation_v3 as fv, field_training as ft, train_formulation as tf
fv.apply(cfg); curriculum.apply_stage(tf.PROPULSION_STAGE)
from env import ASVLidarEnv
env = ASVLidarEnv(render_mode=None, scenario_stage=7)
env.reset(seed=3)
env.define_stage(8, {**cfg.CURRICULUM_STAGES[7], "field_share": 0.0, "failure_replay": 1.0})
built = ft.sample(np.random.default_rng(4), encounter="NT", varying=False)
env._failure_pool = [built]
env.reset(seed=5)
replayed = env.scenario is built and not env._failure_pool
env.define_stage(9, {**cfg.CURRICULUM_STAGES[7], "field_share": 0.0})
env._failure_pool = [built]
env.reset(seed=6)
print(json.dumps({"replayed": replayed, "stage": env.scenario_stage,
                  "untouched_without_key": env.scenario is not built}))
"""
    r = json.loads(_run(code).strip().splitlines()[-1])
    assert r["replayed"] and r["stage"] == 9 and r["untouched_without_key"]
