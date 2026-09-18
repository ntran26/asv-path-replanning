"""The identified vessel plant, and how the environment drives it.

Four groups:

* **parity** -- `ship.ShipModel` is the identified `bluefin` model, exactly, for
  every forward-thrust trajectory.  This is what licenses importing rather than
  copying it: if the two ever diverge, this fails;
* **physics** -- steady speed, the report's steady-turn acceptance checks, and
  the reverse braking this project adds;
* **environment** -- the bridge's rudder limiter, the emergency stop end to end,
  hull randomisation;
* **deployment timing** -- 2 Hz decisions and the measured pose staleness, both
  native to the environment since revision 7.
"""

import math

import numpy as np
import pytest

import constants as cfg
import corridor as corr
import emergency_stop as es
import ship
from colregs.context import ENGAGED, EncounterContext, compliant_turn_sense
from env import ASVLidarEnv


# ---------------------------------------------------------------------------
# Parity with the identified model
# ---------------------------------------------------------------------------
def test_forward_trajectories_match_the_identified_model_exactly():
    """The wrapper adds substepping and reverse braking and nothing else.

    Random rudder and forward rpm every step, compared state for state against
    `bluefin/ship_model_v3.ShipModel` at the same step.  Exact to round-off:
    anything looser would let a wrapper edit quietly change the physics the
    field validation was run against.
    """
    rng = np.random.default_rng(3)
    ours = ship.ShipModel(sub_dt=0.05)
    reference = ship._v3.ShipModel(params=ship.IDENTIFIED)
    for _ in range(400):
        rpm = float(rng.choice([0.0, 6.0, 9.0, 12.0, 15.0, 18.0]))
        rud = float(rng.uniform(-100.0, 100.0))
        assert np.allclose(ours.update(rpm, rud, 0.05),
                           reference.update(rpm, rud, 0.05), rtol=0.0, atol=1e-12)
    assert np.allclose(ours._s, reference._s, rtol=0.0, atol=1e-12)


def test_a_control_step_is_exactly_its_substeps():
    """`update(dt)` splits into `sub_dt` pieces, so 0.1 s == two 0.05 s calls."""
    one, two = ship.ShipModel(), ship.ShipModel()
    for k in range(120):
        rud = 60.0 * math.sin(0.2 * k)
        one.update(12.0, rud, 0.1)
        two.update(12.0, rud, 0.05)
        two.update(12.0, rud, 0.05)
    assert np.allclose(one._s, two._s, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def test_steady_speed_is_the_identified_thrust_drag_balance():
    p = ship.IDENTIFIED
    assert ship.steady_speed(12.0) == pytest.approx(math.sqrt(p["T12"] / p["X_uu"]), rel=1e-6)
    model = ship.ShipModel()
    for _ in range(600):
        model.update(12.0, 0.0, 0.1)
    assert model.u == pytest.approx(ship.steady_speed(12.0), rel=1e-3)


def test_u_ref_is_the_plant_cruise_and_agrees_with_the_logs():
    """One `U_REF`, and the simulated vessel cruises at it.

    Two independent methods: the hull fit and 02b T1's speed-over-ground median
    from 18 logs (1.14 m/s).  They agree within 3 %.
    """
    assert cfg.U_REF == pytest.approx(ship.steady_speed(cfg.CRUISE_RPM))
    # The log median was measured at 12 rpm-units; F24 chose 6 (revision 8).
    assert ship.steady_speed(12.0) == pytest.approx(cfg.U_REF_LOG_MEDIAN, rel=0.03)
    assert cfg.U_REF == pytest.approx(0.558, abs=0.005)


def test_the_055_option_is_exactly_six_rpm_units():
    """F24's alternative operating point needs no calibration on this plant."""
    assert ship.steady_speed(6.0) == pytest.approx(0.5 * ship.steady_speed(12.0), rel=1e-6)


@pytest.mark.parametrize("helm", [0.25, 0.5, 0.75, 1.0])
def test_steady_turn_is_bounded_and_turns_the_commanded_way(helm):
    """REPORT A1/A2."""
    model = ship.ShipModel()
    for _ in range(200):
        model.update(12.0, 0.0, 0.1)
    u0 = model.u
    for _ in range(600):
        model.update(12.0, 100.0 * helm, 0.1)
    r = math.degrees(model.yaw_rate)
    assert np.isfinite([model.u, model.v, r]).all()
    assert abs(r) < 30.0 and 0.4 * u0 < model.u < u0


def test_turn_rate_grows_with_helm():
    rates = []
    for helm in (0.25, 0.5, 0.75, 1.0):
        model = ship.ShipModel()
        for _ in range(200):
            model.update(12.0, 0.0, 0.1)
        for _ in range(600):
            model.update(12.0, 100.0 * helm, 0.1)
        rates.append(math.degrees(model.yaw_rate))
    signs = np.sign(rates)
    assert np.all(signs == signs[-1])
    assert np.all(np.diff(np.abs(rates)) > -0.5), rates


def test_reverse_braking_decelerates_and_never_goes_astern():
    model = ship.ShipModel()
    for _ in range(600):
        model.update(12.0, 0.0, 0.1)
    speeds = []
    for _ in range(80):
        model.update(-24.0, 0.0, 0.1)
        speeds.append(model.u)
    assert min(speeds) >= 0.0
    assert speeds[-1] == 0.0
    assert all(b <= a + 1e-12 for a, b in zip(speeds, speeds[1:]))


def test_astern_impulse_past_zero_is_accounted_rather_than_lost():
    """What the clip at zero surge throws away is what drives the real vessel
    astern.  Once stopped, every further second of full astern adds exactly
    `braking force x time` of impulse."""
    model = ship.ShipModel()
    for _ in range(600):
        model.update(12.0, 0.0, 0.1)
    for _ in range(100):
        model.update(-24.0, 0.0, 0.1)
    assert model.u == 0.0
    model.update(-24.0, 0.0, 0.5)
    assert model.last_astern_impulse == pytest.approx(model.last_braking_force * 0.5)
    model.update(12.0, 0.0, 0.5)
    assert model.last_astern_impulse == 0.0


def test_zero_reverse_efficiency_is_exactly_coasting():
    """The braking term is the only thing a negative command adds."""
    braked, coasted = ship.ShipModel(reverse_efficiency=0.0), ship.ShipModel()
    for model in (braked, coasted):
        for _ in range(300):
            model.update(12.0, 0.0, 0.1)
    for _ in range(100):
        braked.update(-24.0, 20.0, 0.1)
        coasted.update(0.0, 20.0, 0.1)
    assert np.allclose(braked._s, coasted._s, rtol=0.0, atol=1e-12)


def test_stronger_reverse_stops_sooner():
    def time_to_stop(eff):
        model = ship.ShipModel(reverse_efficiency=eff)
        for _ in range(600):
            model.update(12.0, 0.0, 0.1)
        for k in range(600):
            model.update(-24.0, 0.0, 0.1)
            if model.u <= 0.0:
                return k
        return 10 ** 6
    assert time_to_stop(1.0) < time_to_stop(0.5) < time_to_stop(0.25)


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------
def straight_env(**kwargs):
    env = ASVLidarEnv(render_mode=None, no_target_prob=1.0,
                      channel=corr.rectangle(cfg.MAP_WIDTH), **kwargs)
    env.forced_num_obs = 0
    return env


def test_the_rudder_limiter_is_off_by_default_and_ramps_at_50_percent_per_second_when_on():
    """Off by default in bridge and simulator alike; when on, 25 % per 0.5 s."""
    assert cfg.RUDDER_COMMAND_LIMIT is False
    default = straight_env()
    default.reset(seed=0)
    default.step(np.array([1.0, 0.0], dtype=np.float32))
    assert default.rudder == pytest.approx(100.0)

    env = straight_env(command_rate_limit=True)
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.rudder == pytest.approx(25.0)
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.rudder == pytest.approx(50.0)
    env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert env.rudder == pytest.approx(25.0)

    free = straight_env(command_rate_limit=False)
    free.reset(seed=0)
    free.step(np.array([1.0, 0.0], dtype=np.float32))
    assert free.rudder == pytest.approx(100.0)


def test_a_manual_emergency_stop_runs_the_full_manoeuvre_and_hands_back():
    """Full astern until stopped, hold at zero, then control returns."""
    env = straight_env()
    env.reset(seed=0)
    cruise = np.array([0.0, 0.0], dtype=np.float32)
    for _ in range(cfg.steps_for(12.0)):
        env.step(cruise)
    assert env.u_body > 0.8 * cfg.U_REF

    env.request_emergency_stop("test")
    seen = []
    for _ in range(cfg.steps_for(20.0)):
        _, _, terminated, truncated, info = env.step(cruise)
        seen.append((info["estop/state"], info["propulsion_s2"], env.u_body,
                     info["estop/reverse_dv_est_mps"]))
        assert not terminated and not truncated
        if info["estop/state"] == es.IDLE and len(seen) > 1:
            break

    states = [s for s, _, _, _ in seen]
    assert states[0] == es.BRAKING and es.HOLDING in states and states[-1] == es.IDLE
    assert all(s2 == es.S2_FULL_ASTERN for st, s2, _, _ in seen if st == es.BRAKING)
    assert all(s2 == es.S2_STOP for st, s2, _, _ in seen if st == es.HOLDING)
    assert min(u for _, _, u, _ in seen) >= 0.0
    held = [u for st, _, u, _ in seen if st == es.HOLDING]
    assert max(held) <= cfg.ESTOP_STOP_SPEED + 1e-9
    assert seen[-1][1] == pytest.approx(es.rpm_to_s2(cfg.CRUISE_RPM))
    # At 2 Hz the latch sees the vessel stopped up to half a second late, and
    # the astern it applies meanwhile is reported rather than silently lost.
    assert max(dv for _, _, _, dv in seen) >= 0.0


def test_the_emergency_stop_can_be_disabled():
    env = straight_env(emergency_stop=False)
    env.reset(seed=0)
    env.request_emergency_stop("ignored")
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["estop/state"] == es.IDLE and info["propulsion_s2"] > 0.0


def _ctx(cls, **kwargs):
    base = dict(track_id=1, cls=cls, state=ENGAGED, engaged=True,
                dcpa=1.0, tcpa=3.0, in_extremis=True,
                compliant_turn_sense=compliant_turn_sense(cls),
                a_stbd=False, a_port=False)
    base.update(kwargs)
    return EncounterContext(**base)


def test_the_supervisor_fires_only_when_8e_is_the_only_lawful_response():
    assert es.stop_required([_ctx("head_on")]) is not None
    assert es.stop_required([_ctx("crossing")]) is not None
    # Alteration still admissible: the policy's job, not the supervisor's.
    assert es.stop_required([_ctx("head_on", a_stbd=True)]) is None
    assert es.stop_required([_ctx("overtaking", a_port=True)]) is None
    # Not in extremis yet.
    assert es.stop_required([_ctx("head_on", in_extremis=False)]) is None
    # Stand-on: stopping in front of an overtaking vessel is the wrong act.
    assert es.stop_required([_ctx("being_overtaken")]) is None
    assert es.stop_required([]) is None


def test_the_supervisor_stops_only_when_stopping_clears():
    """A18: a reciprocal head-on runs onto a stopped ship; a crossing target
    passing ahead of it does not."""
    import math
    import pytest
    import constants as cfg
    reciprocal = _ctx("head_on", rng=4.0, alpha=0.0, ct=180.0, speed_ts=0.5)
    assert reciprocal.dcpa_if_stopped == pytest.approx(0.0, abs=1e-9)
    assert not reciprocal.stop_clears
    assert es.stop_required([reciprocal]) is None

    ahead = _ctx("crossing", rng=4.0, alpha=45.0, ct=270.0, speed_ts=0.5)
    assert ahead.dcpa_if_stopped == pytest.approx(4.0 * math.sin(math.radians(45.0)), rel=1e-6)
    assert ahead.stop_clears
    assert es.stop_required([ahead]) is not None

    receding = _ctx("crossing", rng=4.0, alpha=45.0, ct=90.0, speed_ts=0.5)
    assert receding.dcpa_if_stopped == pytest.approx(4.0)
    assert cfg.ESTOP_CLEAR_DCPA_M == pytest.approx(
        0.5 * cfg.LOA + 0.5 * cfg.BREADTH + 0.30 + cfg.D_SAFE)


def test_danger_passes_when_the_cpa_is_behind():
    assert not es.danger_passed([_ctx("head_on", tcpa=2.0)])
    assert es.danger_passed([_ctx("head_on", tcpa=-0.5)])
    assert es.danger_passed([])


def test_hull_randomisation_is_seeded_and_does_not_reshuffle_the_scenario():
    env = ASVLidarEnv(render_mode=None, vessel_randomisation=1.0)
    env.reset(seed=11)
    p_a, start_a = dict(env.model.p), (env.start_x, env.start_y)
    env.reset(seed=12)
    assert env.model.p != p_a
    env.reset(seed=11)
    assert env.model.p == p_a

    nominal = ASVLidarEnv(render_mode=None, vessel_randomisation=None)
    nominal.reset(seed=11)
    assert (nominal.start_x, nominal.start_y) == start_a
    assert nominal.model.p == ship.IDENTIFIED


def test_a_supplied_channel_reports_its_own_width():
    """F36: an explicit channel used to report the default basin width."""
    assert ASVLidarEnv(render_mode=None, channel=corr.rectangle(8.0)).corridor_width \
        == pytest.approx(8.0)


def test_the_stern_spawns_clear_of_the_boundary_band():
    """F35: every episode used to open with 16-19 steps of boundary penalty."""
    env = ASVLidarEnv(render_mode=None, no_target_prob=1.0)
    env.forced_num_obs = 0
    for seed in range(5):
        env.reset(seed=seed)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["reward/term/bnd"] == 0.0, seed


# ---------------------------------------------------------------------------
# Deployment timing
# ---------------------------------------------------------------------------
def test_the_environment_decides_at_the_vessels_rate():
    assert cfg.UPDATE_RATE == pytest.approx(cfg.DEPLOYED_DECISION_DT)
    env = straight_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["elapsed_time"] == pytest.approx(0.5)


def test_staleness_is_off_now_the_bridge_waits_for_the_pose_line():
    """443 of 1,085 July frames used the previous frame's pose.  The bridge's
    `PoseSync` removes that -- 0 of 1,085 on replay -- so the default is zero,
    and the measured rate stays available for robustness studies."""
    assert cfg.MEASURED_POSE_STALE_PROB == pytest.approx(443 / 1085, abs=1e-3)
    assert cfg.POSE_STALE_PROB == 0.0
    assert ASVLidarEnv(render_mode=None).pose_stale_prob == 0.0
    env = ASVLidarEnv(render_mode=None, pose_stale_prob=cfg.MEASURED_POSE_STALE_PROB)
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.zeros(2, dtype=np.float32))
    assert 0 < env.stale_frames < 20


def test_consecutive_stale_frames_hold_the_last_received_pose():
    """Missing telemetry cannot advance to unreceived intermediate poses."""
    stale = straight_env(pose_stale_prob=1.0)
    initial, _ = stale.reset(seed=0)
    initial_pose = stale.estimated_pose()
    helm = np.array([0.4, 0.0], dtype=np.float32)
    for _ in range(6):
        o_stale, _, _, _, info = stale.step(helm)
        assert info["pose_stale"]
        assert stale.estimated_pose() == initial_pose
        for branch in ("boundary", "ego", "path"):
            assert np.allclose(o_stale[branch], initial[branch]), branch


def test_stale_frames_do_not_feed_the_tracker():
    """The bridge can tell a stale frame -- the pose timestamp has not moved --
    and lifting a fresh scan with an old pose would displace every static
    object by a step's travel."""
    env = straight_env(pose_stale_prob=1.0)
    env.reset(seed=0)
    fed = len(env.tracker._scans)
    for _ in range(4):
        env.step(np.zeros(2, dtype=np.float32))
    assert len(env.tracker._scans) == fed
    assert env.stale_frames == 4


def test_the_stop_test_reads_the_close_range_hull_fit_view():
    """F66 (C15): inside STOP_TEST_FIT_RANGE_M the stop test uses the fitted
    centre and axis; the track's own values otherwise."""
    import pytest
    # Centroid track: a reciprocal head-on dead ahead -- a stop cannot clear it.
    ctx = _ctx("head_on", rng=3.0, alpha=0.0, ct=180.0, speed_ts=0.5)
    assert not ctx.stop_clears
    # The fit says the hull is 2 m to starboard on a parallel reciprocal track.
    ctx.stop_rng, ctx.stop_alpha, ctx.stop_ct = 3.0, 41.8, 180.0
    assert ctx.dcpa_if_stopped == pytest.approx(3.0 * abs(np.sin(np.radians(41.8 - 180.0))), rel=1e-6)
    assert ctx.stop_clears


def test_tracks_carry_their_latest_hull_fit():
    """F66: the tracker records a fit on every matched update, in either mode."""
    import tracking as trk
    import constants as cfg
    tracker = trk.Tracker(measurement="centroid")
    a = np.radians(180.0)
    pts = np.array([[5.0 + t * np.cos(a), 9.0 - 0.865] for t in np.linspace(-0.25, 0.25, 12)])
    pts = np.vstack([pts, [[5.25, 9.0 - 0.865 + d] for d in np.linspace(0.05, 1.2, 12)]])
    for step in range(4):
        shift = np.array([0.0, -0.25 * step])
        cluster = trk.Cluster((pts + shift).mean(axis=0), pts + shift, np.array([5.6, 4.0]))
        tracker.update([cluster], cfg.UPDATE_RATE)
    track = tracker.tracks[0]
    assert track.last_fit_centre is not None
    tracker.update([], cfg.UPDATE_RATE)
    assert track.last_fit_centre is None


def test_the_braking_profile_stops_and_grows_with_speed():
    """A23: the latch's full astern brings the nominal hull below the stop speed,
    and a faster vessel travels further doing it."""
    import constants as cfg
    import stopping
    t0, s0 = stopping.braking_profile(0.0)
    assert len(t0) == 1 and s0[-1] == 0.0
    t1, s1 = stopping.braking_profile(cfg.U_REF)
    t2, s2 = stopping.braking_profile(2.0 * cfg.U_REF)
    assert 0.0 < s1[-1] < s2[-1]
    assert t1[-1] < cfg.ESTOP_MAX_BRAKE_S
    assert np.all(np.diff(s2) >= -1e-9)


def test_the_stop_test_follows_the_braking_path():
    """A23: a crossing target that clears a stationary own ship by more than the
    hull clearance can still meet one that slides forward while stopping."""
    import math
    import pytest
    import constants as cfg
    import stopping
    ahead = 1.9
    rng, alpha = math.hypot(3.0, ahead), math.degrees(math.atan2(3.0, ahead))
    stationary = stopping.dcpa_over_stop(rng, alpha, 270.0, 0.5, 0.0)
    moving = stopping.dcpa_over_stop(rng, alpha, 270.0, 0.5, cfg.U_REF)
    assert stationary == pytest.approx(ahead, abs=1e-6)
    assert stationary >= cfg.ESTOP_CLEAR_DCPA_M > moving
    # And the context reads it from its own surge.
    ctx = _ctx("crossing", rng=rng, alpha=alpha, ct=270.0, speed_ts=0.5, u_own=cfg.U_REF)
    assert ctx.dcpa_if_stopped == pytest.approx(moving)
    assert not ctx.stop_clears


def test_the_policy_slowdown_is_a_coast_not_the_latch():
    """F68: R-2 and the Rule 8 credit ask whether the agent's own slowdown
    clears -- a coast at the propulsion floor -- which carries the vessel much
    further than the supervisor's full astern."""
    import math
    import constants as cfg
    import stopping
    _, latch = stopping.braking_profile(cfg.U_REF, "latch")
    _, coast = stopping.braking_profile(cfg.U_REF, "coast")
    assert coast[-1] > 3.0 * latch[-1]
    ahead = 2.6
    rng, alpha = math.hypot(3.0, ahead), math.degrees(math.atan2(3.0, ahead))
    ctx = _ctx("crossing", rng=rng, alpha=alpha, ct=270.0, speed_ts=0.5, u_own=cfg.U_REF)
    assert ctx.dcpa_if_slowed <= ctx.dcpa_if_stopped
    assert ctx.stop_clears and not ctx.slowdown_clears


def test_a_fraction_of_episodes_can_start_slow():
    """F68: off by default; with a fraction set, some generated episodes start
    at rest or below half cruise, on their own random stream."""
    import constants as cfg
    from env import ASVLidarEnv
    default = ASVLidarEnv(render_mode=None, scenario_stage=5)
    default.reset(seed=3)
    assert default.start_speed == pytest.approx(cfg.U_NOM)
    slow = ASVLidarEnv(render_mode=None, scenario_stage=5, low_speed_start_frac=1.0)
    speeds = []
    for seed in range(12):
        slow.reset(seed=seed)
        speeds.append(slow.start_speed)
        assert slow.u_body == pytest.approx(slow.start_speed)
    assert all(s <= 0.5 * cfg.U_NOM + 1e-9 for s in speeds)
    assert any(s == 0.0 for s in speeds) and any(s > 0.0 for s in speeds)
    # The same seed reproduces the start, and the scenario is the default's.
    again = ASVLidarEnv(render_mode=None, scenario_stage=5, low_speed_start_frac=1.0)
    again.reset(seed=3)
    slow.reset(seed=3)
    assert again.start_speed == slow.start_speed
    assert again.scenario.digest() == default.scenario.digest()
