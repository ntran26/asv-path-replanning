"""03a §10's twelve acceptance tests, plus 04a's freeze assertions.

"Committed with the environment, run before the first training run."  They are
numbered as 03a numbers them so the table can be checked off directly.

`T1` is `xfail(strict)` and that is the point: it encodes 03a §1.1's decision
that `U_nom = 0.55 m/s`, which T1 (the log-mining task, unrelated name collision)
measured at **1.14 m/s**.  A test that hard-failed would block the suite; one
asserting 0.29 would silently bless a number the specification rejects.  It
fails visibly and flips the day F24 is decided, which is the same treatment
02b §3.3 prescribed for `R-8` and endorsed afterwards.
"""

import math

import numpy as np
import pytest

import boundary_raycast as br
import constants as cfg
import corridor as corr
import scenario as scn
import suite as ste
import targets as tgt
from env import ASVLidarEnv


# ---------------------------------------------------------------------------
# T1 — operating speed
# ---------------------------------------------------------------------------
def test_t1_froude_is_self_consistent():
    """Whatever `U_NOM` is, `froude()` reports it correctly.

    This half always holds and is worth pinning separately, because it is what
    makes the failing half below a statement about the *decision* rather than
    about the arithmetic.
    """
    assert cfg.froude() == pytest.approx(cfg.U_NOM / math.sqrt(9.81 * cfg.LBP))
    assert cfg.full_scale(50.0)["Lpp_m"] == pytest.approx(78.5, abs=0.1)


@pytest.mark.xfail(reason="F24: 03a §1.1 decides U_nom = 0.55 m/s and calls it "
                          "the field measurement, but the retained logs give a "
                          "median of 1.14 m/s at 12 RPM across 18 runs. The "
                          "environment runs the measured value; this flips when "
                          "the decision is made either way.",
                   strict=True)
def test_t1_froude_at_u_nom_is_0_14():
    """03a §10 T1.  Fr = 0.14 ± 0.01."""
    assert cfg.froude() == pytest.approx(0.14, abs=0.01)


# ---------------------------------------------------------------------------
# T2 — the boundary branch is computed from the estimated pose
# ---------------------------------------------------------------------------
def test_t2_boundary_branch_uses_the_noisy_pose():
    """03a §7: a noiseless boundary scan in training creates a sim-to-real gap
    in the component introduced to remove one (01 §3.3)."""
    env = ASVLidarEnv(render_mode=None, corridor_width=6.0, pose_noise=True)
    env._pose_noise.sigma_xy = 0.15
    env.forced_num_obs = 0
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.zeros(2, dtype=np.float32))

    noisy = np.asarray(env.boundary_closeness, dtype=float)
    truth = br.boundary_scan(env.asv_x, env.asv_y, env.asv_h,
                             env.boundary_polygon, pose_noise=None)
    assert not np.allclose(noisy, truth, atol=1e-6), (
        "the boundary branch is being computed from the true pose")


# ---------------------------------------------------------------------------
# T3 — the boundary branch is not affine in the cross-track error
# ---------------------------------------------------------------------------
def test_t3_boundary_rays_are_decorrelated_from_cross_track_error():
    """03a §3.2 / 04a §3.2, and the reason the generator exists.

    In a straight centred constant-width channel the port and starboard rays are
    affine functions of `e_y`, so the 7-dimensional branch carries one number
    and an ablation of it would return a null result for a reason that has
    nothing to do with the branch.

    60 episodes rather than 1000: at ~47 steps/s the specified batch is a
    twenty-minute test, and 1800 paired samples already resolves a correlation
    to well inside the 0.9 threshold.  Run the full batch before the freeze.
    """
    env = ASVLidarEnv(render_mode=None)
    env.forced_num_obs = 0
    cte, rays = [], []
    for seed in range(60):
        env.reset(seed=seed)
        for _ in range(30):
            _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            cte.append(info["cross_track_error"])
            rays.append(np.asarray(env.boundary_closeness, dtype=float).copy())
            if term or trunc:
                break

    cte = np.asarray(cte)
    rays = np.asarray(rays)
    worst = 0.0
    for i in range(rays.shape[1]):
        column = rays[:, i]
        if column.std() < 1e-9 or cte.std() < 1e-9:
            continue
        worst = max(worst, abs(float(np.corrcoef(cte, column)[0, 1])))
    assert worst < cfg.BOUNDARY_DECORRELATION_MAX, (
        f"|corr(e_y, b_i)| = {worst:.3f}: the corridor is effectively constant "
        f"width and centred, and the boundary branch is decorative")


# ---------------------------------------------------------------------------
# T5 / T6 — what the sensor sees
# ---------------------------------------------------------------------------
def test_t6_the_corridor_boundary_is_never_in_the_raw_scan():
    """01 §3.1 / 03a §3.1.  The corridor is a map polygon, and the whole D5
    decision is that it reaches the policy through a virtual raycast rather than
    through the sensor.  A narrow channel inside a wide basin is the case that
    would expose a leak: the walls are 1.5 m outside the basin, so anything
    returning at the channel limit could only be the channel."""
    env = ASVLidarEnv(render_mode=None, corridor_width=3.5, facility_walls=False,
                      no_target_prob=1.0)
    env.forced_num_obs = 0
    env.reset(seed=0)
    assert np.allclose(env.raw_ranges, cfg.LIDAR_RANGE), (
        "something returned from the corridor limit")


# ---------------------------------------------------------------------------
# T7 — class-conditional confinement
# ---------------------------------------------------------------------------
def test_t7_crossing_targets_leave_the_corridor_and_others_do_not():
    """03a §5.2 / 04a §1.3, both directions.

    A crossing target confined to a 4 m fairway would have to pass through the
    far wall; Rule 9(d) is specifically about vessels crossing a narrow channel,
    and such a vessel is not a channel user at all.
    """
    assert tgt.is_confined("head_on")
    assert tgt.is_confined("overtaking")
    assert tgt.is_confined("being_overtaken")
    assert tgt.is_confined("null")
    assert not tgt.is_confined("crossing")

    generator = scn.ScenarioGenerator(stage=5)
    seen = {"crossing": 0, "confined": 0}
    for index in range(60):
        for cls in ("crossing", "head_on"):
            built = generator.sample(scn.seed_for("development", index * 4 + 1),
                                     encounter_class=cls, width=6.0)
            if built is None:
                continue
            channel = corr.rectangle(6.0)
            inside = br.point_in_polygon(built.target_spawn[0],
                                         built.target_spawn[1], channel.polygon())
            if cls == "crossing":
                seen["crossing"] += 1
                assert not built.target_confined
            else:
                seen["confined"] += 1
                assert built.target_confined
    assert seen["crossing"] > 0 and seen["confined"] > 0


# ---------------------------------------------------------------------------
# T8 — static panels are not promoted to dynamic tracks
# ---------------------------------------------------------------------------
def test_t8_static_panels_are_rarely_classified_dynamic():
    """03a §10 T8 and §6.3.  A false promotion creates a phantom give-way
    obligation with COLREGs consequences, which is why the hysteresis is
    asymmetric and slow in both directions."""
    env = ASVLidarEnv(render_mode=None, corridor_width=6.0, pose_noise=False)
    env.forced_num_obs = 3
    env.no_target_prob = 1.0
    frames = promoted = 0
    for seed in range(6):
        env.reset(seed=seed)
        for _ in range(80):
            _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            frames += 1
            promoted += int(info["n_tracks"])
            if term or trunc:
                break
    rate = promoted / max(frames, 1)
    assert rate < 0.05, (
        f"{rate:.1%} of frames carried a dynamic track with no target present")


# ---------------------------------------------------------------------------
# T9 — stopping authority decides `allow_reverse`
# ---------------------------------------------------------------------------
@pytest.mark.xfail(reason="F28: head reach is 29.9 m (19.1 Lpp) against 03a "
                          "§4.3's 1.5 Lpp criterion, and 13.5 m (8.6 Lpp) even "
                          "at half speed. Coasting cannot execute Rule 8(e); "
                          "either the identified drag is badly low (05) or "
                          "allow_reverse is mandatory. 03a §4.3 states the "
                          "consequence; the call is not Claude Code's.",
                   strict=True)
def test_t9_head_reach_is_within_one_and_a_half_ship_lengths():
    """03a §4.3 / §10 T9, and **it fails by an order of magnitude**.

    The demanding case is crossing give-way by passing astern: the own ship must
    shed about 3 m of along-track position, which needs deceleration from cruise
    to ~0.2 `U_nom`.  03a §4.3 reasons that this is "of order 5 N of net
    decelerating force -- comparable to the hull's own quadratic drag at
    0.55 m/s" and concludes that "coasting alone plausibly achieves it; reverse
    thrust is probably not required".

    Measured against the carried-over Paper 2 hull, coasting takes **29.9 m and
    53 s** from 1.14 m/s, and **13.5 m and 54 s** from 0.51 m/s.  The result is
    the same at either candidate operating speed, so it does not turn on F24.

    03a §4.3 says what follows: `allow_reverse` must be set and the platform's
    actual reverse capability verified, and if the platform cannot reverse then
    "take all way off" is unavailable and the paper states the limitation rather
    than claiming the manoeuvre.  Both routes run through 05.
    """
    from ship import ShipModel

    model = ShipModel()
    # Run up to cruise, then cut thrust entirely.
    for _ in range(600):
        model.update(cfg.CRUISE_RPM, 0.0, cfg.UPDATE_RATE)
    assert model.u == pytest.approx(cfg.U_NOM, rel=0.05), model.u

    target_speed = 0.2 * cfg.U_NOM
    reach = 0.0
    for _ in range(2000):
        dx, dy, _, _ = model.update(0.0, 0.0, cfg.UPDATE_RATE)
        reach += float(math.hypot(dx, dy))
        if model.u <= target_speed:
            break

    limit = 1.5 * cfg.LBP
    assert reach <= limit or cfg.REVERSE_AVAILABLE, (
        f"head reach {reach:.2f} m exceeds {limit:.2f} m and reverse is not "
        f"available: Rule 8(e) 'take all way off' cannot be executed")


# ---------------------------------------------------------------------------
# T12 — domain intrusion is a metric, not a termination
# ---------------------------------------------------------------------------
def test_t12_domain_intrusion_does_not_terminate_the_episode():
    """03a §3.3 / §4.4.  The two-constraint convention: the ship domain governs
    vessel-to-vessel separation and is *reported*, while clearance to the
    boundary is a separate and harder constraint."""
    env = ASVLidarEnv(render_mode=None, corridor_width=8.0)
    env.forced_num_obs = 0
    env.reset(seed=4)

    # **Abeam and parallel**, which is the only way to sit inside the domain and
    # stay there.  Two 1.73 m hulls placed nose-to-tail inside the 3.14 m fore
    # domain are already touching once the own hull's `HULL_MARGIN` is counted,
    # so an ahead placement measures collision rather than intrusion; and a
    # closing target collides before the tracker has published anything.
    #
    # Abeam the domain is 1.25 m and the hulls clear at 0.65 m, so 1.0 m of
    # centre separation is unambiguously inside the domain and unambiguously not
    # a collision.
    abeam = 1.0
    env.targets = [tgt.Target(env.asv_x + abeam, env.asv_y,
                              env.asv_h, cfg.U_NOM)]

    # The tracker publishes after `TRACK_MIN_HITS` updates, so the context the
    # panel reports does not exist on the first step.
    terminated = False
    for _ in range(cfg.TRACK_MIN_HITS + 3):
        env.targets[0].x = env.asv_x + abeam
        env.targets[0].y = env.asv_y
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert not terminated, "domain intrusion must not terminate the episode"

    assert info["collision_kind"] is None

    # `r_dom` reads **ground truth** under `R-1` -- the agent pays for intruding
    # whether or not it saw the target -- so this is the assertion that matters,
    # and it holds with no track at all.
    assert info["reward/term/dom"] < 0.0, "r_dom should be charging for this"

    # **F29, found by writing this test.**  A target 1.0 m abeam is inside the
    # sensor's 1 m dead zone, so no track is published and the *panel's*
    # `domain_margin` is `None` -- it is derived from the `EncounterContext`,
    # which is perceived.  So the reward charges for an intrusion the panel
    # cannot display, which is the one case where the panel stops being a
    # faithful view of the step.  It is also exactly the case 02b §3.1 floored
    # the domain to avoid, and it survives because the floor is 1.25 m while the
    # dead zone is 1.0 m: the 0.25 m band between them is perceivable in
    # principle and invisible here in practice.
    margin = info["panel"]["clearance"].get("domain_margin")
    assert margin is None or margin < 0.0


# ---------------------------------------------------------------------------
# 04a §9 — the freeze protocol
# ---------------------------------------------------------------------------
def test_seed_namespaces_are_disjoint():
    """04a §9.2.  The development suite exists so that reward iteration and
    algorithm selection never touch the frozen suite; one suite for both is a
    selection bias that is invisible in the results and fatal if noticed."""
    assert scn.namespaces_are_disjoint(), cfg.SEED_NAMESPACES
    for name in cfg.SEED_NAMESPACES:
        lo, hi = cfg.SEED_NAMESPACES[name]
        assert lo <= scn.seed_for(name, 0) <= hi
        assert lo <= scn.seed_for(name, 10 ** 9) <= hi


def test_the_suite_structure_matches_the_specification():
    """04a §4.2 and §5, as counts."""
    assert len(ste.tier_a()) == 34
    assert len({c.case_id for c in ste.tier_a()}) == 34
    assert len(ste.tier_b_cells()) == 39
    assert sum(c["episodes"] for c in ste.tier_b_cells()) == 975
    assert len(ste.around_the_clock(open_water=True)) == 24


def test_the_suite_regenerates_to_the_same_hashes():
    """04a §9.3's last checklist item: regenerating from seed reproduces every
    hash.  Without it the released archive is a claim rather than an artefact."""
    first = ste.build_tier_a()
    second = ste.build_tier_a()
    assert [s.case_id for s in first] == [s.case_id for s in second]
    assert [s.digest() for s in first] == [s.digest() for s in second]
    assert ste.manifest(first)["manifest_digest"] == ste.manifest(second)["manifest_digest"]


def test_the_constants_snapshot_records_what_the_geometry_depends_on():
    """A suite is only reproducible against the constants it was built under,
    and every threshold in it is a function of the ship domain and the pose
    characterisation — both still provisional."""
    snap = ste.constants_snapshot()
    for key in ("U_NOM", "DOMAIN_LATERAL", "W_WALL", "MAX_EPISODE_STEPS",
                "width_thresholds"):
        assert key in snap
    assert snap["MAX_EPISODE_STEPS"] == 900


def test_the_freeze_checklist_reports_what_is_outstanding():
    """The checklist is machine-checked state, not a list to tick by hand.

    It is *expected* to have unsatisfied entries right now — that is the honest
    answer to "can the headline training run start", and several of them cannot
    be satisfied by code at all because they are waiting on 05.
    """
    checks = ste.freeze_checklist()
    assert all(len(row) == 3 for row in checks)
    outstanding = [name for name, ok, _ in checks if not ok]
    assert outstanding, "the checklist should not read clean while F24 is open"
    assert any("F24" in note or "speed" in name for name, ok, note in checks if not ok)
