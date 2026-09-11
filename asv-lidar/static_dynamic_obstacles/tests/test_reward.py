"""The reward: 02a §10.4's eleven tests, and the decisions T4 had to make.

The eleven are numbered in their docstrings so they can be checked off against
the specification.  Test 8 is `xfail(strict)` pending 03's corridor generator,
per 02b §3.3 -- a term that is implemented, untested and silently inert is worse
than one that is missing, so the test exists and fails rather than being omitted.
"""

import math

import numpy as np
import pytest

import boundary_raycast as br
import constants as cfg
import encounter as enc
from colregs import geometry as geo
from colregs.context import ContextManager, EncounterContext
from env import ASVLidarEnv
from path import ReferencePath, straight_points
from reward import RewardConfig, RewardFunction
from reward import terms as T
from reward.audit import TermAudit

CFG = RewardConfig()
D_REQ = CFG.d_req


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FakeTrack:
    """The minimum a `ContextManager` needs, without a perception pipeline."""

    def __init__(self, x, y, course_deg, speed, track_id=1):
        self.id = track_id
        self.position = np.array([float(x), float(y)])
        self.course_deg = float(course_deg)
        self.speed = float(speed)

    @property
    def velocity(self):
        a = math.radians(self.course_deg)
        return np.array([self.speed * math.sin(a), self.speed * math.cos(a)])


def ctx_for(cls, **kwargs) -> EncounterContext:
    """An engaged context in `cls`, with everything else benign.

    `rho` defaults to 1 so a term's own severity is what is being measured
    rather than the proximity gate multiplying it away.
    """
    from colregs.context import ENGAGED, compliant_turn_sense
    base = dict(track_id=1, cls=cls, state=ENGAGED, engaged=True, rho=1.0,
                dcpa=0.5 * D_REQ, tcpa=10.0,
                compliant_turn_sense=compliant_turn_sense(cls),
                a_stbd=True, a_port=True, r_stbd=5.0, r_port=5.0,
                admissibility_known=True)
    base.update(kwargs)
    return EncounterContext(**base)


def state_for(**kwargs) -> T.RewardState:
    base = dict(u=cfg.U_REF, w_local=10.0, l_path=20.0)
    base.update(kwargs)
    return T.RewardState(**base)


# ---------------------------------------------------------------------------
# 1. Every term inside its declared range
# ---------------------------------------------------------------------------
def test_1_every_term_stays_in_range_across_random_states():
    """02a §10.4 test 1.  The declared range is what makes the weight the
    maximum per-step contribution, and therefore what makes the §7 hierarchy a
    property of the coefficient table rather than something to discover."""
    rng = np.random.default_rng(0)
    n = 20_000
    classes = list(cfg.ENCOUNTER_CLASSES)

    for i in range(n):
        state = T.RewardState(
            u=float(rng.uniform(-0.5, 3.0)),
            v=float(rng.uniform(-1.0, 1.0)),
            r=float(rng.uniform(-1.5, 1.5)),
            heading_deg=float(rng.uniform(0.0, 360.0)),
            e_y=float(rng.uniform(-8.0, 8.0)),
            chi=float(rng.uniform(-math.pi, math.pi)),
            chi_la=float(rng.uniform(-math.pi, math.pi)),
            w_local=float(rng.uniform(3.0, 12.0)),
            r_path=float(rng.uniform(-0.5, 0.5)),
            ds=float(rng.uniform(-0.5, 0.5)),
            l_path=float(rng.uniform(5.0, 40.0)),
            d_bnd=float(rng.uniform(0.0, 5.0)),
            d_clear=float(rng.uniform(0.0, 6.0)),
            d_rudder=float(rng.uniform(-2.0, 2.0)),
            d_throttle=float(rng.uniform(-2.0, 2.0)),
            step_index=int(rng.integers(0, 700)),
        )
        ctx = ctx_for(
            classes[int(rng.integers(0, len(classes)))],
            rho=float(rng.uniform(0.0, 1.0)),
            dcpa=float(rng.uniform(0.0, 20.0)),
            tcpa=float(rng.uniform(-40.0, 40.0)),
            y_rel_cpa=float(rng.uniform(-10.0, 10.0)),
            beta_cpa=float(rng.uniform(0.0, 360.0)),
            alpha=float(rng.uniform(0.0, 360.0)),
            psi_engage=float(rng.uniform(0.0, 360.0)),
            u_engage=float(rng.uniform(0.0, 2.0)),
            t_engage=int(rng.integers(0, 700)),
            d_ts_true=float(rng.uniform(0.0, 20.0)),
            a_stbd=bool(rng.integers(0, 2)),
            a_port=bool(rng.integers(0, 2)),
            a_req=float(rng.uniform(0.0, 1.0)),
            dy_req=float(rng.uniform(0.0, D_REQ)),
            speed_ts=float(rng.uniform(0.0, 2.0)),
            engaged=bool(rng.integers(0, 2)),
            in_extremis=bool(rng.integers(0, 2)),
        )
        for name, fn in T.DENSE_TERMS.items():
            lo, hi = T.TERM_RANGE[name]
            value = fn(state, {1: ctx}, CFG)
            assert lo - 1e-9 <= value <= hi + 1e-9, f"{name} = {value} at draw {i}"

        for name, fn in T.COLREGS_TERMS.items():
            value = fn(state, ctx, CFG)
            assert 0.0 <= value <= 1.0 + 1e-9, f"v_{name} = {value} at draw {i}"


# ---------------------------------------------------------------------------
# 2. Coefficient ordering
# ---------------------------------------------------------------------------
def test_2_coefficient_ordering_holds_and_is_enforced():
    """02a §7 and §9.  The hierarchy is a property of the table, and the config
    refuses to be built if it is not."""
    assert (CFG.w_bnd > CFG.w_dom > CFG.w_obs > CFG.w_col
            > CFG.w_pf > CFG.w_prog > CFG.w_smooth > CFG.w_exist)
    with pytest.raises(ValueError, match="coefficient ordering"):
        RewardConfig(w_col=5.0)


def test_2b_a_collision_outranks_a_whole_non_compliant_encounter():
    """02 §5 / `R-7`: a COLREGs-compliant collision must be worse than a
    maximally non-compliant episode that avoids one.  At -200 the margin was 32
    points, which is why the payoff moved to -300."""
    assert abs(CFG.r_collision) > CFG.w_col * CFG.max_encounter_steps
    with pytest.raises(ValueError, match="r_collision"):
        RewardConfig(r_collision=-100.0)


# ---------------------------------------------------------------------------
# 3. Progress telescopes, and is invariant to the speed profile
# ---------------------------------------------------------------------------
def _traverse(l_path: float, speeds) -> float:
    """Sum `r_prog` over a full traversal driven at the given speeds."""
    total, s = 0.0, 0.0
    while s < l_path:
        step = min(float(speeds(s)) * cfg.UPDATE_RATE, l_path - s)
        total += T.r_prog(state_for(ds=step, l_path=l_path), {}, CFG)
        s += step
    return total


def test_3_progress_telescopes_to_n_ref():
    """02a §10.4 test 3, and the direct test of `R-9`'s claim.

    `Sum r_prog` is a constant fixed by the path, so a legal Rule 8(e) slowdown
    costs **zero** progress reward.  That is the whole reason no carve-out is
    needed, and why the creep exploit 02 §4.4 warns about has no mechanism to
    arise from.
    """
    for l_path in (20.0, 20.2, 20.42):
        assert _traverse(l_path, lambda s: cfg.U_REF) == pytest.approx(
            CFG.n_ref_prog, rel=3e-3)


def test_3d_the_generator_must_keep_path_length_near_the_design_point():
    """**A constraint on 03's generator, pinned rather than left as a comment.**

    C3's fixed `N_ref` buys an episode integral independent of path length, and
    the price is that the clip now binds at `U_REF * L_path / L_REF` rather than
    at `U_REF`.  Over the lengths this environment produces (20.00-20.42 m) that
    is 1.140-1.164 m/s, at or just above cruise, so `R-9` holds.  On a path much
    *shorter* than the 20 m design point it would bind below cruise and `R-9`
    would invert -- slowing down would start to pay, which is F22 returning by
    another route.

    Half the design length is the failure, so it is asserted as one.  If 03's
    corridor generator produces shorter paths, `L_REF_PATH` moves with it.
    """
    env = ASVLidarEnv(render_mode=None)
    lengths = []
    for seed in range(40):
        env.reset(seed=seed)
        lengths.append(env.path.length)
    for length in lengths:
        binding = length / (CFG.n_ref_prog * cfg.UPDATE_RATE)
        assert binding >= cfg.U_REF - 1e-6, (
            f"a {length:.2f} m path binds the progress clip at {binding:.3f} m/s, "
            f"below cruise {cfg.U_REF:.2f}: R-9 inverts and slowing down pays")

    # And the failure it is guarding against is real, not hypothetical.
    assert _traverse(0.5 * cfg.L_REF_PATH, lambda s: cfg.U_REF) < \
        0.6 * CFG.n_ref_prog


def test_3b_progress_is_invariant_to_the_speed_profile():
    """The claim that matters: slowing down must not cost progress reward."""
    steady = _traverse(20.0, lambda s: cfg.U_REF)
    half = _traverse(20.0, lambda s: 0.5 * cfg.U_REF)
    crawl = _traverse(20.0, lambda s: 0.2 * cfg.U_REF)
    # A slowdown in the middle third, which is the 8(e) manoeuvre itself.
    slowed = _traverse(20.0, lambda s: cfg.U_REF * (0.35 if 7.0 < s < 13.0 else 1.0))
    for other in (half, crawl, slowed):
        assert other == pytest.approx(steady, rel=2e-3)


def test_3c_n_ref_is_derived_from_the_measured_cruise_speed():
    """F22.  A literal `N_ref = 250` would bind the clip at 0.80 m/s -- 02a's
    *assumed* cruise, not the 1.14 m/s T1 measured -- so every step at cruise
    would clip and slowing down would start to *increase* the integral.  That is
    `R-9` running backwards, and it is the creep exploit arriving through the
    term that exists to remove it."""
    binding_speed = cfg.L_REF_PATH / (CFG.n_ref_prog * cfg.UPDATE_RATE)
    assert binding_speed == pytest.approx(cfg.U_REF, rel=1e-6)
    assert _traverse(20.0, lambda s: 1.6 * cfg.U_REF) < 0.95 * CFG.n_ref_prog


# ---------------------------------------------------------------------------
# 4. Holding course is correct when Rule 9(a) already satisfies Rule 14
# ---------------------------------------------------------------------------
def test_4_compliant_head_on_target_holding_course_costs_nothing():
    """02a §10.4 test 4, and the `02 §3.2` rationale as a regression test.

    Target positionally 9(a)-compliant, so the projected pass already clears
    `d_req` and nothing is owed.  **If this fails the agent is being trained to
    manoeuvre unnecessarily** -- and the M5 ablation could not then distinguish
    "learned to alter" from "learned when to alter".
    """
    ctx = ctx_for(enc.HEAD_ON, dcpa=1.2 * D_REQ, a_req=0.0, dy_req=0.0,
                  psi_engage=0.0, u_engage=cfg.U_REF, tcpa=8.0)
    state = state_for(heading_deg=0.0, r=0.0, u=cfg.U_REF)
    assert T.v_r8(state, ctx, CFG) == 0.0
    assert T.v_port(state, ctx, CFG) == 0.0
    assert T.r_col(state, {1: ctx}, CFG) == 0.0


def test_4b_a_displaced_target_does_owe_an_alteration():
    """The other branch: `A_req` scales with the deficit, so the obligation is
    proportionate rather than a fixed 20 degrees."""
    ctx = ctx_for(enc.HEAD_ON, dcpa=0.2 * D_REQ, a_req=0.8, dy_req=0.8 * D_REQ,
                  psi_engage=0.0, u_engage=cfg.U_REF, tcpa=5.0)
    state = state_for(heading_deg=0.0, r=0.0, u=cfg.U_REF)
    assert T.v_r8(state, ctx, CFG) > 0.0


def test_4c_rule_8e_discharges_the_obligation_when_the_turn_does_not_fit():
    """02 §4.4 as reward structure: with no starboard room, slackening speed is
    the compliant action and must be able to discharge `v_r8` on its own."""
    common = dict(dcpa=0.2 * D_REQ, a_req=0.8, dy_req=0.8 * D_REQ,
                  psi_engage=0.0, u_engage=cfg.U_REF, tcpa=5.0,
                  a_stbd=False, r_stbd=-0.2)
    ctx = ctx_for(enc.HEAD_ON, **common)
    holding = state_for(heading_deg=0.0, u=cfg.U_REF)
    slowed = state_for(heading_deg=0.0, u=cfg.U_REF - 1.2 * CFG.du_min)
    assert T.v_r8(holding, ctx, CFG) > 0.0
    assert T.v_r8(slowed, ctx, CFG) == 0.0


# ---------------------------------------------------------------------------
# 5-6. The `02 §4.2` trap, both directions
# ---------------------------------------------------------------------------
def test_5_a_port_turn_is_a_violation_head_on_and_compliant_overtaking():
    """02a §10.4 test 5.  **Overtaking requires a port turn.**

    The same expression penalises opposite turns in the two classes, because
    `compliant_turn_sense` carries the sign.  There is no global "port turns are
    penalised" constant anywhere in the tree, which is what makes this trap
    impossible to code rather than merely documented.
    """
    turning_port = state_for(r=-0.25)
    assert T.v_port(turning_port, ctx_for(enc.HEAD_ON), CFG) > 0.0
    assert T.v_port(turning_port, ctx_for(enc.CROSSING), CFG) > 0.0
    assert T.v_port(turning_port, ctx_for(enc.OVERTAKING), CFG) == 0.0

    turning_stbd = state_for(r=+0.25)
    assert T.v_port(turning_stbd, ctx_for(enc.HEAD_ON), CFG) == 0.0
    assert T.v_port(turning_stbd, ctx_for(enc.OVERTAKING), CFG) > 0.0


def test_6_v_side_inverts_between_head_on_and_overtaking():
    """02a §10.4 test 6.  Both directions in one test so they cannot drift apart.

    Head-on: port-to-port puts the target on the own ship's PORT at the CPA, so
    `y_rel_CPA > 0` is the violation.  Overtaking: passing to port *of the
    target* puts the target on the own ship's STARBOARD, so `y_rel_CPA < 0` is
    the violation.  Opposite signs, same geometry family.
    """
    state = state_for()
    stbd, port = +0.8 * D_REQ, -0.8 * D_REQ

    assert T.v_side(state, ctx_for(enc.HEAD_ON, y_rel_cpa=stbd), CFG) > 0.0
    assert T.v_side(state, ctx_for(enc.HEAD_ON, y_rel_cpa=port), CFG) == 0.0

    assert T.v_side(state, ctx_for(enc.OVERTAKING, y_rel_cpa=port), CFG) > 0.0
    assert T.v_side(state, ctx_for(enc.OVERTAKING, y_rel_cpa=stbd), CFG) == 0.0


def test_6b_v_side_is_not_applied_to_crossing():
    """There the requirement is "pass astern", which `v_bow` covers; a side term
    would double-count it."""
    state = state_for()
    for y in (-2.0, -0.5, 0.5, 2.0):
        assert T.v_side(state, ctx_for(enc.CROSSING, y_rel_cpa=y), CFG) == 0.0


def test_6c_overtaking_side_is_not_charged_when_the_port_pass_does_not_fit():
    """`R-5`: the correct behaviour is to hold astern, and penalising the side
    of a pass that is not being attempted would charge the agent for complying."""
    ctx = ctx_for(enc.OVERTAKING, y_rel_cpa=-0.8 * D_REQ, a_port=False, r_port=-0.3)
    assert T.v_side(state_for(), ctx, CFG) == 0.0


# ---------------------------------------------------------------------------
# 7. Yaw rate, not rudder
# ---------------------------------------------------------------------------
def test_7_v_port_fires_on_yaw_rate_crossing_r_dead_not_on_rudder():
    """02a §10.4 test 7, and locked principle 5.

    A rudder movement that never develops into a turn is not an alteration of
    course.  `v_port` reads the yaw rate and nothing else -- the rudder command
    is not even an input to it, which is the strongest available form of this
    guarantee.
    """
    ctx = ctx_for(enc.HEAD_ON)
    assert T.v_port(state_for(r=-0.5 * CFG.r_dead), ctx, CFG) == 0.0
    assert T.v_port(state_for(r=-0.999 * CFG.r_dead), ctx, CFG) == 0.0
    assert T.v_port(state_for(r=-2.0 * CFG.r_dead), ctx, CFG) > 0.0

    # Hard over with no yaw developed yet still costs nothing.
    hard_over = state_for(r=0.0, d_rudder=-1.0)
    assert T.v_port(hard_over, ctx, CFG) == 0.0


def test_7b_v_port_saturates_at_r_ref():
    state = state_for(r=-(CFG.r_ref + CFG.r_dead + 0.05))
    assert T.v_port(state, ctx_for(enc.HEAD_ON), CFG) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. The `r_path` subtraction -- blocked on 03's bends
# ---------------------------------------------------------------------------
def test_8_a_compliant_port_bend_costs_nothing_in_the_term():
    """The unit half of test 8, which does not need a bent corridor.

    Following a channel that bends to port requires a port turn that is
    path-following, not evasion.  Subtracting `r_path` measures only the excess.
    """
    ctx = ctx_for(enc.HEAD_ON, r_path=-0.25)
    assert T.v_port(state_for(r=-0.25), ctx, CFG) == 0.0
    assert T.v_port(state_for(r=-0.45), ctx, CFG) > 0.0
    # Without the subtraction the same vessel would be scored a violator.
    assert T.v_port(state_for(r=-0.25), ctx_for(enc.HEAD_ON, r_path=0.0), CFG) > 0.0


@pytest.mark.xfail(reason="needs 03's corridor generator: with kappa = 0 everywhere "
                          "the environment cannot produce a bend, so R-8 is only "
                          "tested against a hand-set r_path and never end to end "
                          "(02b §3.3)",
                   strict=True)
def test_8b_the_environment_produces_a_bend_that_exercises_r_path():
    """02a §10.4 test 8, end to end.

    Deliberately failing rather than absent: a term that is implemented,
    untested and silently inert is worse than one that is missing.  It will
    start passing -- and flag itself as XPASS -- the moment T5 lands.
    """
    env = ASVLidarEnv(render_mode=None)
    seen = 0.0
    for seed in range(20):
        env.reset(seed=seed)
        for _ in range(60):
            _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            seen = max(seen, abs(info["r_path_radps"]))
            if term or trunc:
                break
    assert seen > 1e-3, "no episode in the distribution bends the path"


# ---------------------------------------------------------------------------
# 9. Admissibility against channel width
# ---------------------------------------------------------------------------
def _head_on_admissibility(width: float) -> dict:
    """Run one step of the context machinery in a channel of the given width."""
    inset = 0.5 * (cfg.MAP_WIDTH - width)
    polygon = br.rectangle(width, cfg.MAP_HEIGHT, x0=inset, y0=0.0)
    centre = inset + 0.5 * width
    path = ReferencePath(straight_points(centre, 0.0, centre, cfg.MAP_HEIGHT))

    manager = ContextManager()
    track = FakeTrack(centre, 12.0, 180.0, cfg.U_REF)
    contexts = manager.update(
        tracks=[track], p_os=(centre, 4.0), v_os=(0.0, cfg.U_REF),
        heading_os_deg=0.0, u_os=cfg.U_REF, path=path,
        boundary_polygon=polygon, s_along=4.0, cross_track=0.0)
    return contexts[track.id]


def test_9_starboard_alteration_fits_at_10_m_and_not_at_3_5_m():
    """02a §10.4 test 9.  The per-step geometric predicate, which is what makes
    Study 1 a measurement: the width thresholds fall out of the sweep as results
    rather than going in as a lookup table."""
    wide = _head_on_admissibility(10.0)
    narrow = _head_on_admissibility(3.5)
    assert wide.admissibility_known and narrow.admissibility_known
    assert wide.a_stbd is True
    assert narrow.a_stbd is False
    assert narrow.r_stbd < wide.r_stbd


def test_9b_the_predicate_matches_the_hand_derivation():
    """`r_stbd = d_bnd_stbd - B/2 - c_wall`, on a centreline vessel."""
    ctx = _head_on_admissibility(10.0)
    assert ctx.d_bnd_stbd == pytest.approx(5.0, abs=0.05)
    assert ctx.r_stbd == pytest.approx(5.0 - 0.5 * cfg.BREADTH
                                       - cfg.HEAD_ON_WALL_CLEARANCE, abs=0.05)


def test_9c_admissibility_is_hysteretic_at_the_margin():
    """Without the band a vessel at the width where a manoeuvre just fits would
    flip `A_stbd` every step, and the Rule 8 obligation would appear and vanish
    at the exact geometry Study 1 is trying to resolve."""
    manager = ContextManager()
    assert manager._hysteretic(1, "stbd", +0.30) is True
    assert manager._hysteretic(1, "stbd", +0.05) is True      # inside the band, held
    assert manager._hysteretic(1, "stbd", -0.05) is True      # still held
    assert manager._hysteretic(1, "stbd", -0.30) is False     # cleanly out
    assert manager._hysteretic(1, "stbd", -0.05) is False     # held the other way


# ---------------------------------------------------------------------------
# 10. `in_extremis` suppression
# ---------------------------------------------------------------------------
def test_10_in_extremis_releases_the_course_keeping_penalty():
    """02a §10.4 test 10, and `R-4`.

    Rule 17(b) requires the stand-on vessel to act when collision cannot be
    avoided by the give-way vessel alone.  That is a different provision from
    17(a)(ii), which S5 puts out of scope.  Without the carve-out the reward
    would contain a literal instruction to hold course into a collision.
    """
    deviating = state_for(r=0.4, u=cfg.U_REF - 0.5)
    ctx = ctx_for(enc.BEING_OVERTAKEN, u_engage=cfg.U_REF, dcpa=0.3 * D_REQ,
                  tcpa=2.0, in_extremis=False)
    assert T.v_hold(deviating, ctx, CFG) > 0.0

    released = ctx_for(enc.BEING_OVERTAKEN, u_engage=cfg.U_REF, dcpa=0.3 * D_REQ,
                       tcpa=2.0, in_extremis=True)
    assert T.v_hold(deviating, released, CFG) == 0.0


def test_10b_the_in_extremis_predicate_is_dcpa_and_tcpa():
    manager = ContextManager()
    close = ctx_for(enc.BEING_OVERTAKEN, dcpa=0.5 * D_REQ, tcpa=3.0)
    manager._attach_gates(close, D_REQ)
    assert close.in_extremis

    for far in (ctx_for(enc.BEING_OVERTAKEN, dcpa=1.5 * D_REQ, tcpa=3.0),
                ctx_for(enc.BEING_OVERTAKEN, dcpa=0.5 * D_REQ, tcpa=20.0)):
        manager._attach_gates(far, D_REQ)
        assert not far.in_extremis


# ---------------------------------------------------------------------------
# 11. The §2 invariant
# ---------------------------------------------------------------------------
def test_11_d_safe_is_below_c_wall_minus_half_breadth():
    """02a §2's invariant.  Otherwise the geometry that *defines* a compliant
    narrow-channel manoeuvre would itself trigger the boundary penalty -- the
    reward would punish the behaviour the paper exists to elicit.

    02a §5.2's own `d_safe = 0.50 m` breaches it (the ceiling is 0.40 m), which
    is why `constants.D_SAFE` is 0.35 and carries a `TODO(decision)`.
    """
    assert CFG.d_safe < CFG.c_wall - 0.5 * cfg.BREADTH
    with pytest.raises(ValueError, match="d_safe"):
        RewardConfig(d_safe=0.50)


def test_11b_the_domain_floor_is_asserted_at_construction():
    """02b §3.1, and the assertion T4 step 4 asks for.  A domain inside the
    sensor's blind zone makes `r_dom` unlearnable, because `R-1` evaluates
    intrusion on ground truth."""
    assert CFG.dom_abeam >= cfg.DOMAIN_ABEAM_FLOOR
    with pytest.raises(ValueError, match="d_abeam"):
        RewardConfig(dom_abeam=0.75 * cfg.LBP)


# ---------------------------------------------------------------------------
# Group aggregation, and the two speed carve-outs
# ---------------------------------------------------------------------------
def test_the_colregs_group_is_clipped_before_the_group_weight():
    """02a §6.7.  No combination of violations may exceed `w_COL` in one step,
    or the group could silently outrank the safety terms -- which is exactly how
    Paper 2's path term came to outrank its avoidance term."""
    ctx = ctx_for(enc.CROSSING, y_rel_cpa=-3.0, beta_cpa=0.0, dcpa=0.0,
                  a_req=1.0, dy_req=D_REQ, tcpa=0.0, psi_engage=0.0,
                  u_engage=cfg.U_REF)
    group = T.colregs_group(state_for(r=-1.0, heading_deg=0.0), {1: ctx}, CFG)
    assert group["pre_clip"] > 1.0, "this geometry should over-saturate"
    assert group["v_col"] == pytest.approx(1.0)
    assert T.r_col(state_for(r=-1.0, heading_deg=0.0), {1: ctx}, CFG) == pytest.approx(-1.0)


def test_a_single_severe_violation_does_not_saturate_the_group():
    """02a §6.7's stated design: two concurrent severe violations saturate, one
    does not.  If one did, the group would stop distinguishing severities."""
    ctx = ctx_for(enc.HEAD_ON, y_rel_cpa=0.0)
    group = T.colregs_group(state_for(r=-1.0), {1: ctx}, CFG)
    assert 0.0 < group["v_col"] < 1.0


def test_r2_drops_the_speed_reference_when_the_alteration_does_not_fit():
    """`R-2`.  Without it the path term charges full penalty for the compliant
    8(e) action, and slackening speed is structurally unlearnable."""
    boxed_in = ctx_for(enc.HEAD_ON, a_stbd=False, r_stbd=-0.4)
    speed = T.effective_speed_reference(state_for(), {1: boxed_in}, CFG)
    assert speed["rule"] == "R-2"
    assert speed["u_ref_eff"] == pytest.approx(CFG.u_ref * CFG.u_ref_slow_factor)

    slow = state_for(u=0.5 * cfg.U_REF)
    assert T.r_pf(slow, {1: boxed_in}, CFG) > T.r_pf(slow, {}, CFG)


def test_r5_makes_holding_astern_affordable_in_a_narrow_overtaking():
    """`R-5`.  Without it the narrow overtaking case is not a test of COLREGs
    reasoning but a test of whether the agent tolerates an unwinnable reward --
    and it would resolve it by overtaking anyway."""
    ctx = ctx_for(enc.OVERTAKING, a_port=False, r_port=-0.3, speed_ts=0.4 * cfg.U_REF)
    speed = T.effective_speed_reference(state_for(), {1: ctx}, CFG)
    assert speed["rule"] == "R-5"
    assert speed["u_ref_eff"] == pytest.approx(0.4 * cfg.U_REF)
    assert speed["w_exist_scale"] == 0.0

    matching = state_for(u=0.4 * cfg.U_REF)
    assert T.r_pf(matching, {1: ctx}, CFG) == pytest.approx(
        T.r_pf(state_for(u=cfg.U_REF), {}, CFG), abs=1e-9)


def test_the_existence_cost_is_suspended_only_by_r5():
    fn = RewardFunction(CFG)
    holding = ctx_for(enc.OVERTAKING, a_port=False, r_port=-0.3,
                      speed_ts=0.4 * cfg.U_REF)
    assert fn(state_for(), {1: holding}).weighted["exist"] == 0.0
    fn.reset()
    assert fn(state_for(), {}).weighted["exist"] == pytest.approx(-cfg.W_EXIST)


# ---------------------------------------------------------------------------
# The dense terms individually
# ---------------------------------------------------------------------------
def test_r_pf_is_width_normalised_so_study_1_is_not_confounded():
    """02a §5.1.  The same *relative* offset must cost the same in a 10 m
    channel and a 4 m one, or the path-following gradient changes with corridor
    width and Study 1 measures the reward instead of the geometry."""
    wide = T.r_pf(state_for(e_y=2.5, w_local=10.0), {}, CFG)
    narrow = T.r_pf(state_for(e_y=1.0, w_local=4.0), {}, CFG)
    assert wide == pytest.approx(narrow, rel=1e-9)


def test_r_pf_penalises_a_stopped_vessel_sitting_on_the_line():
    """The penalty form and the speed gate: `g_u = 0` gives maximum penalty, so
    a stationary vessel cannot collect a perfect path score."""
    assert T.r_pf(state_for(u=0.0, e_y=0.0), {}, CFG) == pytest.approx(-1.0)
    assert T.r_pf(state_for(u=cfg.U_REF, e_y=0.0), {}, CFG) == pytest.approx(0.0, abs=1e-9)


def test_r_obs_is_exactly_zero_beyond_the_cut_off():
    """02a §5.4.  The shift is the point: the unshifted form reads -0.08 at 2 m
    and integrates to about -53 over an episode -- larger than the path term,
    constant, and carrying no gradient."""
    assert T.r_obs(state_for(d_clear=cfg.D_CUT), {}, CFG) == 0.0
    assert T.r_obs(state_for(d_clear=5.0), {}, CFG) == 0.0
    assert T.r_obs(state_for(d_clear=float("inf")), {}, CFG) == 0.0
    assert T.r_obs(state_for(d_clear=0.0), {}, CFG) == pytest.approx(-1.0)
    for near, far in ((0.2, 0.5), (0.5, 1.0), (1.0, 1.9)):
        assert T.r_obs(state_for(d_clear=near), {}, CFG) < \
               T.r_obs(state_for(d_clear=far), {}, CFG)


def test_r_bnd_reaches_full_penalty_outside_the_channel():
    assert T.r_bnd(state_for(d_bnd=cfg.D_SAFE), {}, CFG) == 0.0
    assert T.r_bnd(state_for(d_bnd=0.0), {}, CFG) == pytest.approx(-1.0)
    assert T.r_bnd(state_for(d_bnd=0.5 * cfg.D_SAFE), {}, CFG) == pytest.approx(-0.25)


def test_r_bnd_is_zero_in_open_water():
    """`R-10`.  Three terms are undefined without a boundary polygon, and the
    04 §4.1 benchmark runs an open-water variant."""
    assert T.r_bnd(state_for(d_bnd=0.0, open_water=True), {}, CFG) == 0.0


def test_r_pf_uses_the_reference_width_in_open_water():
    """`W_ref = 10 m` rather than an arbitrary large value, so `e_y` stays on the
    same scale as the widest sweep condition and the open-water score is
    directly comparable to the 20 B row of Study 1."""
    open_water = T.r_pf(state_for(e_y=2.5, w_local=3.5, open_water=True), {}, CFG)
    trained = T.r_pf(state_for(e_y=2.5, w_local=cfg.W_REF_OPEN_WATER), {}, CFG)
    assert open_water == pytest.approx(trained)


def test_r_dom_is_evaluated_on_ground_truth_at_the_right_bearing():
    """02a §5.3, and the asymmetry is the point: the same range ahead and abeam
    is not the same intrusion."""
    abeam = ctx_for(enc.CROSSING, alpha=90.0, d_ts_true=0.5 * cfg.DOMAIN_LATERAL)
    ahead = ctx_for(enc.HEAD_ON, alpha=0.0, d_ts_true=0.5 * cfg.DOMAIN_LATERAL)
    assert T.r_dom(state_for(), {1: abeam}, CFG) == pytest.approx(-0.25)
    # The same range dead ahead is deeper inside a domain that reaches further.
    assert T.r_dom(state_for(), {1: ahead}, CFG) < T.r_dom(state_for(), {1: abeam}, CFG)

    clear = ctx_for(enc.HEAD_ON, alpha=0.0, d_ts_true=3.0 * cfg.DOMAIN_FORE)
    assert T.r_dom(state_for(), {1: clear}, CFG) == 0.0


def test_r_smooth_saturates_at_the_actuator_rate_limit():
    """02a §5.6.  `kappa_delta` is derived from the actuator, so the term
    saturates at exactly the physical limit and self-calibrates when 05 delivers
    the actuator model."""
    assert T.r_smooth(state_for(d_rudder=0.0), {}, CFG) == 0.0
    assert T.r_smooth(state_for(d_rudder=CFG.kappa_delta), {}, CFG) == pytest.approx(-1.0)
    assert T.r_smooth(state_for(d_rudder=0.5 * CFG.kappa_delta), {}, CFG) == \
        pytest.approx(-0.25)


def test_r_smooth_is_cheap_in_the_free_window_after_engagement():
    """`sigma_t` resolves the Rule 8 tension: 8(b) wants ONE large alteration
    and forbids a succession of small ones, and a plain smoothness penalty
    suppresses both."""
    ctx = ctx_for(enc.HEAD_ON, t_engage=100)
    committed = state_for(d_rudder=CFG.kappa_delta, step_index=105)
    dithering = state_for(d_rudder=CFG.kappa_delta, step_index=100 + CFG.n_free + 5)
    assert T.r_smooth(committed, {1: ctx}, CFG) == pytest.approx(-CFG.sigma_enc)
    assert T.r_smooth(dithering, {1: ctx}, CFG) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------
def test_engagement_fires_before_the_obligation_does():
    """02a §6.1.  In Revision 1 `kappa_eng` evaluated to exactly the compliant
    separation, putting the threshold on a knife-edge at the geometry the agent
    is supposed to achieve.  Watch first, then act."""
    assert CFG.kappa_eng * D_REQ > D_REQ
    manager = ContextManager()
    track = FakeTrack(5.0, 14.0, 180.0, cfg.U_REF)
    ctx = manager.update(tracks=[track], p_os=(5.0, 4.0), v_os=(0.0, cfg.U_REF),
                         heading_os_deg=0.0, u_os=cfg.U_REF)[track.id]
    assert ctx.cls == enc.HEAD_ON
    assert ctx.state == "engaged"
    assert ctx.psi_engage == pytest.approx(0.0)
    assert ctx.u_engage == pytest.approx(cfg.U_REF)


def test_rho_is_a_third_at_a_compliant_pass_and_every_severity_is_zero_there():
    """02a §6.1's own worked number.  The gate being non-zero at a compliant
    pass is fine precisely because nothing it multiplies is non-zero there."""
    manager = ContextManager()
    ctx = ctx_for(enc.HEAD_ON, dcpa=D_REQ)
    manager._attach_gates(ctx, D_REQ)
    assert ctx.rho == pytest.approx(1.0 - 1.0 / CFG.kappa_eng, abs=1e-9)
    assert ctx.rho == pytest.approx(0.33, abs=0.01)


def test_a_cleared_encounter_returns_to_idle():
    """`ENGAGED -> CLEARING -> IDLE`, so a passed target stops gating anything."""
    manager = ContextManager()
    approaching = FakeTrack(5.0, 14.0, 180.0, cfg.U_REF)
    assert manager.update(tracks=[approaching], p_os=(5.0, 4.0),
                          v_os=(0.0, cfg.U_REF), heading_os_deg=0.0,
                          u_os=cfg.U_REF)[1].state == "engaged"

    # Now astern and opening: TCPA goes negative.
    passed = FakeTrack(5.0, 2.0, 180.0, cfg.U_REF)
    state = manager.update(tracks=[passed], p_os=(5.0, 4.0), v_os=(0.0, cfg.U_REF),
                           heading_os_deg=0.0, u_os=cfg.U_REF)[1].state
    assert state == "clearing"
    for _ in range(CFG.n_clear + 1):
        ctx = manager.update(tracks=[passed], p_os=(5.0, 4.0), v_os=(0.0, cfg.U_REF),
                             heading_os_deg=0.0, u_os=cfg.U_REF)[1]
    assert ctx.state == "idle"


def test_the_context_is_the_only_place_the_class_is_decided():
    """01 §5.3, mechanically.  The observation's class and the reward's gate are
    the same field of the same object on the same step, not two derivations that
    happen to agree today."""
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=1)
    for _ in range(30):
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        contexts = env.encounter_contexts
        assert info["encounter_classes"] == {t: c.cls for t, c in contexts.items()}
        if term or trunc:
            break


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------
def test_the_audit_flags_a_constant_offset_but_not_an_inactive_term():
    """The `range(ep)` column, which is the direct detector for the Paper 2
    scale bug.  A term at a non-zero constant is broken; a term at zero is
    simply not active this episode."""
    audit = TermAudit(CFG)
    for _ in range(60):
        audit.record(_fake_breakdown({"pf": -0.42, "bnd": 0.0, "prog": 0.0}))
    rows = {row["name"]: row for row in audit.rows()}
    assert rows["pf"]["flat"] is True
    assert rows["bnd"]["flat"] is False


def test_the_audit_does_not_flag_a_term_that_varies():
    audit = TermAudit(CFG)
    for i in range(60):
        audit.record(_fake_breakdown({"pf": -0.1 - 0.01 * i}))
    assert {row["name"]: row for row in audit.rows()}["pf"]["flat"] is False


def _fake_breakdown(term):
    from reward.reward import RewardBreakdown
    return RewardBreakdown(term=dict(term),
                           weighted={k: v * 0.6 for k, v in term.items()})


def test_the_pre_committed_episode_orderings_hold():
    """02a §8.1's three orderings, on 02a's own pre-committed integrals.

    These are the reward's actual contract, and they are the reason
    `R_collision` moved to -300: **a COLREGs-compliant collision must be worse
    than a maximally non-compliant episode that avoids one** (02 §5).  At -200
    the margin was 32 points.

    The `prog` row is +52.6 rather than the tabulated +75 (F22).  It is the only
    row F22 moves, and the orderings survive it with 44 points to spare.  These
    remain *predictions*: T6 replaces them with the measured Table R7, and a
    mismatch there is diagnostic rather than a reason to edit this test.
    """
    from reward.audit import PREDICTED_NOMINAL

    nominal = sum(PREDICTED_NOMINAL.values()) + cfg.R_GOAL
    collided = (0.5 * sum(PREDICTED_NOMINAL.values())) + cfg.R_COLLISION
    violating = (PREDICTED_NOMINAL["prog"] - 40.0 - 26.0 - 17.0 - 4.0
                 - 15.0 - 45.0 - 270.0) + cfg.R_GOAL

    assert nominal == pytest.approx(83.1, abs=1.0)
    assert violating == pytest.approx(-264.4, abs=1.0)
    assert collided == pytest.approx(-308.5, abs=1.0)

    assert nominal > violating > collided
    assert violating - collided > 20.0, "the 02 §5 margin has gone thin"


def test_a_cornered_agent_prefers_timeout_to_collision():
    """02a §8.1's second ordering -- and it has to be tested *discounted*.

    **F23.**  Undiscounted, this ordering does not hold at the environment's
    actual step limit.  A vessel that stops dead in clear water pays
    `w_pf + w_exist = 0.65` every step (`r_pf` is a penalty form, so `g_u = 0`
    gives maximum penalty by design), and over `MAX_EPISODE_STEPS = 700` that is
    -455 against a collision's -300.  02a §8.1's "-86" assumes its own 300-step
    design point; the environment's 700-step cap is a Paper 2 carry-over that
    was never reconciled with it.

    The ordering is nonetheless sound, because the undiscounted sum is not what
    the agent optimises.  At the headline SAC discount the same behaviour is
    worth about -65, comfortably above the collision payoff, and the other two
    orderings compare episodes of similar length so discounting does not reorder
    them.

    Worth watching rather than fixing: it is marginal for the **PPO comparator**
    at `gamma = 0.999`, where the truncated geometric sum reaches about -327 and
    crosses -300.  If that comparator ever prefers a collision to holding
    station in a corner, this is why, and the fix is the step limit rather than
    a coefficient.
    """
    stopped_per_step = cfg.W_PF + cfg.W_EXIST
    horizon = cfg.MAX_EPISODE_STEPS

    def discounted(gamma):
        return -stopped_per_step * (1.0 - gamma ** horizon) / (1.0 - gamma)

    sac = discounted(0.99)          # train.py: SAC, the headline architecture
    assert sac > cfg.R_COLLISION, (
        f"loitering is worth {sac:.0f} against a collision at {cfg.R_COLLISION:.0f}")
    assert sac == pytest.approx(-65.0, abs=2.0)

    # The undiscounted statement, pinned so the margin cannot quietly erode.
    undiscounted = -stopped_per_step * horizon
    assert undiscounted < cfg.R_COLLISION, (
        "if this ever passes, the step limit and 02a §8.1's design point have "
        "been reconciled and this test should assert the ordering directly")

    # And the PPO comparator's margin, so a regression is visible as a number.
    ppo = discounted(0.999)
    assert ppo == pytest.approx(-327.0, abs=5.0)


def test_the_step_limit_leaves_room_for_a_detour():
    """The other half of F23: the cap has to be generous enough to be a *cap*.

    A 20 m path at the measured cruise takes ~175 steps, so 700 is four times
    the traversal.  That is what makes timeout a genuine fallback rather than a
    routine outcome -- and it is also why the loiter sum gets large.
    """
    traversal = cfg.L_REF_PATH / (cfg.U_REF * cfg.UPDATE_RATE)
    assert traversal == pytest.approx(175.4, abs=1.0)
    assert cfg.MAX_EPISODE_STEPS > 2.0 * traversal
