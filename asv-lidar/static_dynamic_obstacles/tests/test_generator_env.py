"""C1 and A8 (revision 8): the scenario generator drives the environment, and
the emergency stop has its own reward treatment."""

import numpy as np
import pytest

import constants as cfg
import emergency_stop as es
import scenario as scn
from env import ASVLidarEnv
from reward import RewardConfig
from reward import terms as T


# ---------------------------------------------------------------------------
# The generator in the environment
# ---------------------------------------------------------------------------
def test_generated_episodes_cover_the_stage_5_classes_and_start_at_cruise():
    env = ASVLidarEnv(render_mode=None, scenario_stage=5)
    seen = set()
    for seed in range(40):
        env.reset(seed=seed)
        cls = env.scenario.encounter_class
        seen.add(cls)
        assert env.u_body == pytest.approx(cfg.U_NOM)
        assert abs(((env.asv_h - env.scenario.own_heading) + 180.0) % 360.0 - 180.0) < 1e-6
        assert (len(env.targets) == 0) == (cls == "no_target")
        if env.targets:
            assert env.targets[0].encounter_class == cls
            assert env.targets[0].confined == (cls != "crossing")
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["scenario_class"] == cls
    assert {"head_on", "crossing", "overtaking", "being_overtaken"} <= seen


def test_generated_episodes_are_reproducible_from_the_seed():
    a, b = (ASVLidarEnv(render_mode=None, scenario_stage=4) for _ in range(2))
    for seed in (3, 17):
        a.reset(seed=seed)
        b.reset(seed=seed)
        assert a.scenario.digest() == b.scenario.digest()
        assert a.obstacles == b.obstacles


def test_the_stage_follows_set_scenario_stage():
    env = ASVLidarEnv(render_mode=None, scenario_stage=1)
    env.reset(seed=0)
    assert env.scenario.encounter_class == "no_target"
    env.set_scenario_stage(5)
    classes = set()
    for seed in range(20):
        env.reset(seed=seed)
        classes.add(env.scenario.encounter_class)
    assert len(classes) > 1


def test_a_supplied_scenario_is_used_as_given():
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    built = generator.sample(scn.seed_for("development", 7), encounter_class="head_on", width=6.0)
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=0, options={"generated": built})
    assert env.scenario is built
    assert env.corridor_width == pytest.approx(built.channel.nominal_width)
    assert (env.targets[0].x, env.targets[0].y) == built.target_spawn


def test_the_being_overtaken_goal_is_clear_of_the_corridor_end():
    """F49: the path end keeps the spawn inset at the far edge too, and a fast
    straight run to the goal ends in a goal, not a boundary collision."""
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    env = ASVLidarEnv(render_mode=None)
    env.forced_num_obs = 0
    checked = 0
    for index in range(40):
        built = generator.sample(scn.seed_for("development", 50_000 + index),
                                 encounter_class="being_overtaken", width=10.0)
        if built is None:
            continue
        env.reset(seed=index, options={"generated": built})
        assert env.path_start_s + env.path.length <= \
            built.channel.length - cfg.GOAL_END_INSET_M + 0.05
        env.targets = []
        while True:
            _, _, term, trunc, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
            if term or trunc:
                break
        assert info["reached_goal"] and not info["collided"]
        checked += 1
        if checked == 5:
            break
    assert checked == 5


def test_the_being_overtaken_dcpa_is_floored_with_no_draws_below():
    """A15: most draws pass at or above the floor; about a fifth are labelled
    below it, and the label agrees with the drawn DCPA."""
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    below, n = 0, 0
    for index in range(150):
        built = generator.sample(scn.seed_for("development", 80_000 + index),
                                 encounter_class="being_overtaken")
        if built is None:
            continue
        n += 1
        # A21: the floor is hull clearance for the draw, plus D_SAFE.
        assert built.dcpa_floor_m == pytest.approx(
            scn.contact_free_dcpa(built.ct_deg, built.speed_ratio) + cfg.BEING_OVERTAKEN_FLOOR_MARGIN)
        if built.dcpa_below_floor:
            below += 1
            assert built.dcpa_m < built.dcpa_floor_m
        else:
            assert built.dcpa_m >= built.dcpa_floor_m
    assert n >= 100
    # baseline-v2 (your call, 2026-09-23): nothing below the floor, so the Rule
    # 17(b) case S5 put out of scope is in neither training nor the suite.
    assert below == 0


def test_contact_free_dcpa_matches_the_hull_geometry():
    """A21: parallel hulls clear at their combined half-breadths; oblique ones
    need more, which is why a 1.0 m centre floor still collided."""
    parallel = scn.contact_free_dcpa(0.0, 1.85)
    assert 0.55 <= parallel <= 0.75
    assert scn.contact_free_dcpa(10.0, 1.5) > parallel
    assert scn.contact_free_dcpa(45.0, 1.85) > 1.4


def test_confined_targets_keep_the_channel_until_cpa():
    """A21: no confined target breaches the corridor before its CPA, so the
    clamp never re-draws a generated encounter."""
    import targets as tgt
    env = ASVLidarEnv(render_mode=None)
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    checked = 0
    for cls in ("overtaking", "being_overtaken", "null", "head_on"):
        for index in range(6):
            built = generator.sample(scn.seed_for("development", 60_000 + 100 * index), encounter_class=cls)
            if built is None:
                continue
            if cls in cfg.CONFINED_CT_CLASSES:
                assert abs(((built.ct_deg + 180.0) % 360.0) - 180.0) <= cfg.CONFINED_CT_HALF_DEG + 1e-9
            env.forced_num_obs = 0
            env.reset(seed=index, options={"generated": built})
            horizon = built.tcpa_s if cls != "null" else cfg.NULL_TRACK_CHECK_S
            for _ in range(int(horizon / cfg.UPDATE_RATE)):
                t = env.targets[0]
                assert tgt.confinement_violation(t, env.channel, env.boundary_polygon) is None
                env.targets[0].step(cfg.UPDATE_RATE)
            checked += 1
    assert checked >= 15


def test_the_clamp_nudges_rather_than_teleports():
    """A21: a target breaching the wall moves inward by about the breach, not
    to the centreline."""
    import corridor as corr
    import targets as tgt
    channel = corr.rectangle(8.0)
    centre_x = float(channel.centre[len(channel.centre) // 2][0])
    target = tgt.Target(centre_x + 3.9, 12.0, 20.0, 0.8, confined=True)
    assert tgt.confinement_violation(target, channel) is not None
    tgt.clamp_to_corridor(target, channel)
    assert abs(target.x - centre_x) > 2.5
    assert target.heading == pytest.approx(0.0, abs=1e-6)


def test_crossings_are_escapable_except_a_labelled_fraction():
    """A22: every crossing carries the label, the label agrees with the escape
    check, and about a fifth are drawn unescapable."""
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    labels = []
    for index in range(80):
        built = generator.sample(scn.seed_for("development", 70_000 + index), encounter_class="crossing")
        if built is None:
            continue
        assert built.crossing_escapable in (True, False)
        if index < 12:
            solved = {"x": built.target_spawn[0], "y": built.target_spawn[1],
                      "heading": built.target_heading, "speed": built.target_speed,
                      "ct": built.ct_deg, "tcpa": built.tcpa_s}
            assert scn.crossing_escape_feasible(np.array(built.own_spawn), built.own_heading,
                                                solved, built.channel) == built.crossing_escapable
        labels.append(built.crossing_escapable)
    assert len(labels) >= 60
    unescapable = labels.count(False) / len(labels)
    assert 0.08 <= unescapable <= 0.35


def test_obstacles_keep_clear_of_the_encounter():
    """04a §3.6: the CPA stretch of the own ship's track stays clear."""
    env = ASVLidarEnv(render_mode=None, scenario_stage=5)
    env.forced_num_obs = 3
    checked = 0
    for seed in range(30):
        env.reset(seed=seed)
        built = env.scenario
        if built.encounter_class in ("no_target", "null") or built.tcpa_s <= 0:
            continue
        guard = cfg.OBSTACLE_CPA_GUARD_FRAC * built.tcpa_s * cfg.U_NOM
        s_cpa = built.tcpa_s * cfg.U_NOM
        for poly in env.obstacles:
            cx, cy = np.mean(poly, axis=0)
            assert abs(env.path.project(cx, cy, 0.0).s_along - s_cpa) > guard
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# The emergency stop in the reward (A8)
# ---------------------------------------------------------------------------
def test_the_speed_gate_is_suspended_while_the_latch_holds():
    cfgr = RewardConfig()
    stopped = T.RewardState(u=0.0, w_local=10.0, l_path=20.0)
    held = T.RewardState(u=0.0, w_local=10.0, l_path=20.0, estop_active=True)
    assert T.r_pf(stopped, {}, cfgr) == pytest.approx(-1.0)
    assert T.r_pf(held, {}, cfgr) == pytest.approx(0.0)
    assert T.effective_speed_reference(held, {}, cfgr)["rule"] == "8(e) stop"


def test_a_stop_costs_once_and_never_as_much_as_a_collision():
    assert RewardConfig().r_collision < cfg.R_ESTOP < 0.0
    with pytest.raises(ValueError, match="r_estop"):
        RewardConfig(r_estop=-400.0)

    env = ASVLidarEnv(render_mode=None, no_target_prob=1.0)
    env.forced_num_obs = 0
    env.reset(seed=0)
    for _ in range(cfg.steps_for(6.0)):
        env.step(np.zeros(2, dtype=np.float32))
    env.request_emergency_stop("test")
    charged = []
    for _ in range(cfg.steps_for(6.0)):
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        charged.append(info["reward/intervention"])
        if term or trunc:
            break
    assert charged[0] == pytest.approx(cfg.R_ESTOP)
    assert all(c == 0.0 for c in charged[1:])


def test_a27_training_crossings_favour_port_and_other_namespaces_do_not(monkeypatch):
    """A27 option 2 (F81): the port share applies to training draws only, so the
    development and frozen sets -- and every comparison on them -- are unchanged."""
    import constants as cfg
    train = scn.ScenarioGenerator(stage=5, seed_namespace="training")
    sides = [b.ct_deg < 180.0 for b in
             (train.sample(scn.seed_for("training", 61_000 + i), encounter_class="crossing")
              for i in range(160)) if b is not None]
    assert abs(np.mean(sides) - cfg.CROSSING_PORT_SHARE_TRAINING) < 0.12
    dev = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    seed = scn.seed_for("development", 10_000 * 2 + 3)
    before = dev.sample(seed, encounter_class="crossing")
    monkeypatch.setattr(cfg, "CROSSING_PORT_SHARE_TRAINING", 0.99)
    after = scn.ScenarioGenerator(stage=5, seed_namespace="development").sample(
        seed, encounter_class="crossing")
    assert before.ct_deg == after.ct_deg


def test_a27_stage_3_teaches_crossings():
    import constants as cfg
    assert "crossing" in cfg.CURRICULUM_STAGES[3]["classes"]


def test_a31_stage_3_draws_crossings_as_often_as_head_ons():
    """A31: a stage's own `weights` govern its class draw (off in baseline-v1)."""
    import constants as cfg
    generator = scn.ScenarioGenerator(stage=3, seed_namespace="training")
    stage = dict(cfg.CURRICULUM_STAGES[3], weights=cfg.STAGE3_CROSSING_WEIGHTS)
    classes = [generator._sample_class(np.random.default_rng(s), stage)
               for s in range(400)]
    share = {c: classes.count(c) / len(classes) for c in set(classes)}
    assert abs(share.get("crossing", 0) - share.get("head_on", 0)) < 0.08
    assert share.get("crossing", 0) > 0.25
