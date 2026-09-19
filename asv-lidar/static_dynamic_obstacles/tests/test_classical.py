"""The classical comparators (B8): LOS-PID + DWA and encounter-specific VO.

What is pinned here is what makes them fair comparators rather than
strawmen or oracles: they read only onboard state, their internal model
predicts the simulator, they follow a clear path cleanly, and the VO's
encounter-specific constraints point the way the rules say.
"""

import math

import numpy as np
import pytest

import constants as cfg
import curriculum
import scenario as scn
from classical import common as cc
from classical import encounter_vo as vo
from classical.encounter_vo import EncounterVOController, Obstacle, select_velocity
from classical.los_dwa import LosDwaController
from env import ASVLidarEnv

PROPULSION_STAGE = 4          # train_formulation.PROPULSION_STAGE, without importing SB3
CONTROLLERS = (LosDwaController, EncounterVOController)


@pytest.fixture(scope="module")
def env():
    curriculum.apply_stage(PROPULSION_STAGE)
    return ASVLidarEnv(render_mode=None, emergency_stop=False)


def _scenario(cls: str, index: int = 0):
    gen = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    for j in range(200):
        built = gen.sample(scn.seed_for("development", 800_000 + 1000 * index + j), encounter_class=cls)
        if built is not None:
            return built
    raise RuntimeError(f"no {cls} scenario")


# ---------------------------------------------------------------------------
# Onboard information only
# ---------------------------------------------------------------------------
TRUTH = ("targets", "asv_x", "asv_y", "asv_h", "asv_w", "u_body", "v_body", "speed_mps",
         "cross_track_error", "obstacles")


class TruthGuard:
    """The environment with simulator truth removed."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name in TRUTH:
            raise AssertionError(f"a classical controller read simulator truth: env.{name}")
        return getattr(self._env, name)


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_controllers_read_only_onboard_state(env, controller):
    obs, _ = env.reset(seed=11, options={"generated": _scenario("head_on")})
    ctl, guarded = controller(), TruthGuard(env)
    for _ in range(25):
        action = ctl.action(guarded, obs)
        assert action.shape == (2,) and np.all(np.abs(action) <= 1.0)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break


# ---------------------------------------------------------------------------
# The internal model predicts the simulator
# ---------------------------------------------------------------------------
def test_rollout_predicts_the_simulator(env):
    """Both comparators plan on `cc.rollout`; it must turn and slow like the hull."""
    env.reset(seed=3, options={"generated": _scenario("no_target")})
    perception, act = cc.Perception(), cc.Actuators()
    for _ in range(4):
        perception.snapshot(env)
        act.issue(env, 0.0)
        env.step(np.zeros(2, dtype=np.float32))
    snap = perception.snapshot(env)
    # Measured from the true start: the snapshot's pose carries the nominal
    # pose noise, and the hull is randomised per episode, so the tolerances
    # are a model check, not an identity.
    start, start_h = np.array([env.asv_x, env.asv_y]), env.asv_h
    commands = [(0.6, -0.5)] * 12
    ro = cc.rollout(snap, act, 1, 6.0, lambda k, s: (
        np.array([cfg.CRUISE_RPM + cfg.RPM_DELTA * commands[k][1]]), np.array([commands[k][0]])))
    for rudder, throttle in commands:
        act.issue(env, rudder)
        env.step(np.array([rudder, throttle], dtype=np.float32))
    turned_env = (env.asv_h - start_h + 180.0) % 360.0 - 180.0
    turned_pred = math.degrees(float(cc.wrap_pi(ro.headings[-1, 0] - snap.heading)))
    assert turned_env > 15.0, "the probe should be a real turn"
    # Planned on the nominal identified hull; the episode's hull is randomised.
    assert abs(turned_pred - turned_env) < 0.25 * turned_env
    travelled_env = np.linalg.norm(np.array([env.asv_x, env.asv_y]) - start)
    travelled_pred = np.linalg.norm(ro.positions[-1, 0] - snap.position)
    assert abs(travelled_pred - travelled_env) < 0.15
    assert abs(ro.speeds[-1, 0] - env.u_body) < 0.05


# ---------------------------------------------------------------------------
# A clear path is followed cleanly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_clear_path_is_followed_to_the_goal(env, controller):
    env.forced_num_obs = 0
    try:
        obs, _ = env.reset(seed=5, options={"generated": _scenario("no_target", 1)})
        ctl, cte = controller(), []
        for _ in range(cfg.MAX_EPISODE_STEPS):
            obs, _, term, trunc, info = env.step(ctl.action(env, obs))
            cte.append(env.cross_track_error)
            if term or trunc:
                break
    finally:
        env.forced_num_obs = None
    assert info["reached_goal"], info.get("collision_kind")
    assert float(np.sqrt(np.mean(np.square(cte)))) < 0.3


# ---------------------------------------------------------------------------
# Encounter-specific VO: the constraints point the way the rules say
# ---------------------------------------------------------------------------
U = 0.55


def _vessel(x, y, heading_deg, speed, cls, sense=0):
    a = math.radians(heading_deg)
    return Obstacle(np.array([x, y]), speed * np.array([math.sin(a), math.cos(a)]), a, cls, sense)


def _pass(choice, ob):
    """Target from own at the CPA of the chosen straight-line velocity."""
    v = choice.speed * np.array([math.sin(choice.course), math.cos(choice.course)])
    p, w = ob.position, ob.velocity - v
    t = max(0.0, -(w @ p) / max(w @ w, 1e-9))
    return p + t * w


def test_vo_holds_the_preferred_velocity_when_clear():
    choice = select_velocity(np.zeros(2), 0.0, U, 0.0, U, [])
    assert choice.feasible and abs(choice.course) < 1e-9 and choice.speed == pytest.approx(U)


def test_vo_head_on_passes_port_to_port():
    ob = _vessel(0.0, 10.0, 180.0, 0.5, "head_on")
    choice = select_velocity(np.zeros(2), 0.0, U, 0.0, U, [ob])
    assert choice.feasible and not choice.released
    assert choice.course > 0.0, "alteration to starboard"
    stbd = np.array([math.cos(choice.course), -math.sin(choice.course)])
    assert _pass(choice, ob) @ stbd < 0.0, "target passes down the port side"


@pytest.mark.parametrize("side, x0, heading", [("starboard", 5.0, 270.0), ("port", -5.0, 90.0)])
def test_vo_crossing_passes_astern_from_either_side(side, x0, heading):
    """A17: give way from either side, and giving way means passing astern."""
    ob = _vessel(x0, 6.0, heading, 0.5, "crossing")
    choice = select_velocity(np.zeros(2), 0.0, U, 0.0, U, [ob])
    assert choice.feasible and not choice.released
    t_hat = ob.velocity / np.linalg.norm(ob.velocity)
    assert (-_pass(choice, ob)) @ t_hat < 0.0, f"own ship crosses ahead of a {side} crosser"
    assert not vo._wrong_side(ob, np.zeros(2), np.array([[math.sin(choice.course), math.cos(choice.course)]]) * choice.speed,
                              np.array([choice.course]))[0]


def test_vo_overtaking_passes_on_the_named_side():
    ob = _vessel(0.0, 4.0, 0.0, 0.2, "overtaking", sense=-1)
    choice = select_velocity(np.zeros(2), 0.0, U, 0.0, U, [ob])
    assert choice.feasible and not choice.released
    stbd = np.array([math.cos(choice.course), -math.sin(choice.course)])
    assert _pass(choice, ob) @ stbd > 0.0, "a port pass leaves the target to starboard"


def test_vo_stands_on_when_being_overtaken_clear():
    ob = _vessel(2.2, -5.0, 0.0, 1.0, "being_overtaken")
    choice = select_velocity(np.zeros(2), 0.0, U, 0.0, U, [ob])
    assert choice.stand_on
    assert choice.course == pytest.approx(0.0) and choice.speed == pytest.approx(U)


def test_vo_stand_on_releases_when_holding_would_collide():
    """Rule 17(b): the overtaker is coming straight up the own track."""
    ob = _vessel(0.0, -6.5, 0.0, 1.0, "being_overtaken")
    choice = select_velocity(np.zeros(2), 0.0, U, 0.0, U, [ob])
    assert not choice.stand_on
    assert choice.feasible
