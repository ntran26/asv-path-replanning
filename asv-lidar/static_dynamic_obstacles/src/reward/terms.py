"""Every reward term, as a pure function returning a value in its declared range.

02a §4:

```
r_t =  w_pf*r_pf + w_prog*r_prog + w_exist*r_exist + w_smooth*r_smooth
     + w_obs*r_obs + w_bnd*r_bnd + w_dom*r_dom + w_COL*r_col + r_term
```

**Every dense term is normalised to `[-1, 0]` before weighting**, except
`r_prog` which is `[-1, +1]`.  The weight *is* the maximum per-step
contribution, so the §7 magnitude hierarchy holds by construction and the §8
audit checks realised severity rather than hunting a hidden scale factor.  That
is the direct fix for the Paper 2 failure, where a path term out-scaled the
avoidance term through a factor nobody had computed.

Pure functions of `(state, ctx, cfg)` rather than branches inside a step
function, for two reasons 02a §10.2 gives and one it does not: the leave-one-out
ablation becomes a mask over a dict; the §10.4 unit tests become trivial; and a
term that is wrong can be read in isolation, which is not true of forty lines of
accumulating `reward +=`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

import constants as cfg_mod
import cpa_cri as cc
from colregs import context as ctxmod
from colregs import geometry as geo


@dataclass
class RewardState:
    """The own ship and the scene, as one step's worth of facts.

    Everything a term needs that is *not* about a specific target.  Assembled by
    the environment, which is the only object holding both the simulated truth
    and the map.
    """

    # --- own-ship motion (ground truth; the physical terms read it) --------
    u: float = 0.0                     # surge, m/s
    v: float = 0.0                     # sway, m/s
    r: float = 0.0                     # yaw rate, **rad/s**
    heading_deg: float = 0.0

    # --- path errors -------------------------------------------------------
    e_y: float = 0.0                   # cross-track error, m, +ve to starboard
    chi: float = 0.0                   # course error, rad
    chi_la: float = 0.0                # lookahead course error, rad
    w_local: float = cfg_mod.MAP_WIDTH  # local channel width, m
    r_path: float = 0.0                # yaw rate required to track the path, rad/s

    # --- progress ----------------------------------------------------------
    ds: float = 0.0                    # along-path advance this step, m
    l_path: float = 1.0                # total path length, m

    # --- clearances (ground truth, from the map and the true geometry) -----
    d_bnd: float = float("inf")        # hull to channel boundary, m
    d_clear: float = float("inf")      # hull to nearest static obstacle in swath, m

    # --- action ------------------------------------------------------------
    d_rudder: float = 0.0              # change in normalised rudder command
    d_throttle: float = 0.0            # change in normalised throttle command

    # --- bookkeeping -------------------------------------------------------
    step_index: int = 0
    open_water: bool = False


# ---------------------------------------------------------------------------
# Speed reference:  `R-2` and `R-5`
# ---------------------------------------------------------------------------
def effective_speed_reference(state: RewardState, contexts, cfg) -> dict:
    """`U_ref_eff` and the existence-cost scale, with the reason for each.

    Two carve-outs, and both exist because the default terms make the compliant
    action unaffordable rather than merely costly:

    * **`R-2`** -- a give-way obligation whose compliant alteration is
      inadmissible discharges under Rule 8(e) by slackening speed.  The path
      term's speed gate would charge full penalty for exactly that action, so
      8(e) would be structurally unlearnable.  `U_ref_eff` drops to
      `0.4 * U_ref` and the legal slowdown costs nothing.
    * **`R-5`** -- narrow-channel overtaking where the port pass does not fit.
      `02 §3.2` fixes the fallback as "hold astern at reduced speed", which
      under the default terms earns no progress, full path penalty and an
      accruing existence cost, then times out.  Matching the target's speed
      satisfies the gate and the existence cost is suspended, because holding
      station is not wandering.

    Without `R-5` the narrow overtaking case is not a test of COLREGs reasoning
    but a test of whether the agent tolerates an unwinnable reward -- and it
    would resolve it by overtaking anyway.  This is the one place the existence
    cost is suspended, and the suspension is gated on a *geometric* predicate
    rather than on CRI, which keeps it consistent with `R-6` and avoids the
    degenerate-policy risk 02 §4.4 warns about.
    """
    result = {"u_ref_eff": float(cfg.u_ref), "w_exist_scale": 1.0,
              "reason": "nominal", "rule": ""}

    for ctx in _iter(contexts):
        if not ctx.engaged:
            continue
        if ctx.cls == ctxmod.enc.OVERTAKING and not ctx.a_port:
            return {"u_ref_eff": max(float(ctx.speed_ts), float(cfg.u_min_reachable)),
                    "w_exist_scale": 0.0,
                    "reason": "hold astern, port pass does not fit",
                    "rule": "R-5"}

    for ctx in _iter(contexts):
        if ctx.engaged and ctx.gives_way and not ctx.turn_admissible:
            result = {"u_ref_eff": float(cfg.u_ref) * float(cfg.u_ref_slow_factor),
                      "w_exist_scale": 1.0,
                      "reason": "8(e) slowdown, alteration inadmissible",
                      "rule": "R-2"}
            break
    return result


# ---------------------------------------------------------------------------
# 1.  Task and safety terms  (02a §5)
# ---------------------------------------------------------------------------
def r_pf(state: RewardState, contexts, cfg, u_ref_eff: float = None) -> float:
    """Unified path following, `[-1, 0]` (02a §5.1).

    ```
    e~_y = e_y / (W_local/2)
    chi~* = omega_LA*chi~_LA + (1 - omega_LA)*chi~
    q_t  = w_e*exp(-gamma_e * e~_y^2) + (1 - w_e)*(1 + cos chi~*)/2
    g_u  = clip(max(u,0)/U_ref_eff, 0, 1)
    r_pf = -(1 - g_u * q_t)
    ```

    **Width normalisation is load-bearing**, and it is the term's whole reason
    for existing in this form.  Paper 2 used `exp(-0.05*|e_y|)`, inherited from
    a 60 x 150 m map; over a 10 m channel that varies by under 10% of its own
    value -- a constant offset wearing a shaping term's costume.  Normalising by
    the local half-width fixes that *and* holds the term's range constant across
    the Study 1 sweep, so the path-following gradient does not change with
    corridor width and confound the study.

    **Penalty form, so the speed gate pushes the right way**: stopping gives
    `g_u = 0` and therefore maximum penalty, rather than a stationary vessel
    collecting a perfect path score for sitting exactly on the line.
    """
    if u_ref_eff is None:
        u_ref_eff = effective_speed_reference(state, contexts, cfg)["u_ref_eff"]

    width = float(cfg.w_ref_open_water) if state.open_water else float(state.w_local)
    half = max(0.5 * width, 1e-6)
    e_tilde = float(np.clip(state.e_y / half, -1.0, 1.0))

    chi_star = cfg.omega_la * state.chi_la + (1.0 - cfg.omega_la) * state.chi
    q = (cfg.w_e * math.exp(-cfg.gamma_e * e_tilde * e_tilde)
         + (1.0 - cfg.w_e) * 0.5 * (1.0 + math.cos(chi_star)))

    g_u = float(np.clip(max(state.u, 0.0) / max(u_ref_eff, 1e-6), 0.0, 1.0))
    return float(np.clip(-(1.0 - g_u * q), -1.0, 0.0))


def r_bnd(state: RewardState, contexts, cfg) -> float:
    """Channel boundary, `[-1, 0]` (02a §5.2).

    `r_bnd = -[max(0, 1 - d_b/d_safe)]^2`, with `d_b` the hull-to-boundary
    distance from the map -- ground truth, per `R-1`, because running aground is
    a physical fact and not a matter of what the sensor saw.

    The boundary stays a **hard constraint** per `02 §4.4`: `w_bnd` is the
    largest dense weight, above the COLREGs group, so slackening speed always
    dominates violating the boundary.  `R-10` zeroes the term in open water,
    where there is no boundary to violate.
    """
    if state.open_water:
        return 0.0
    if not np.isfinite(state.d_bnd):
        return 0.0
    excess = max(0.0, 1.0 - float(state.d_bnd) / max(float(cfg.d_safe), 1e-9))
    return float(np.clip(-(excess ** 2), -1.0, 0.0))


def r_dom(state: RewardState, contexts, cfg) -> float:
    """Target ship-domain intrusion, `[-1, 0]` (02a §5.3).

    `r_dom = -[max(0, 1 - d_TS/d_dom(beta_TS))]^2`, evaluated on **ground
    truth** and with the asymmetric domain taken at the target's actual bearing.

    **Addition beyond the doc's eleven terms, and not optional.**  `00 §4.2`
    reports ship-domain intrusion rate and depth, but nothing in the six carried
    terms or the five COLREGs terms supplies a dense signal for target
    proximity: the only feedback would be the terminal collision penalty.  A
    reported metric with no corresponding reward signal is precisely the pattern
    that produced Paper 2's concessions.

    **Measured centre-to-centre, not hull-to-hull as §5.3's wording says.**  The
    same document defines `d_req = 2*d_abeam` as the centre separation of two
    vessels passing abeam, and `kappa_eng * d_req` gates engagement against
    `cpa()`, which is centre-to-centre throughout.  Reading `d_dom` as a
    standoff from the hull in this one term would put the domain on a different
    datum from `d_req`, `rho_t` and the observation's `distance_to_domain`, and
    the four would disagree about what a compliant pass is.  One datum, and it
    is the one the rest of the specification already uses.
    """
    worst = 0.0
    for ctx in _iter(contexts):
        d_ts = float(ctx.d_ts_true)
        if not np.isfinite(d_ts):
            continue
        beta = _bearing_from_ctx(ctx)
        d_dom = cc.domain_scale(beta, fore=cfg.dom_fore, aft=cfg.dom_aft,
                                lateral=cfg.dom_abeam)
        worst = max(worst, max(0.0, 1.0 - d_ts / max(d_dom, 1e-9)))
    return float(np.clip(-(worst ** 2), -1.0, 0.0))


def r_obs(state: RewardState, contexts, cfg) -> float:
    """Static obstacle proximity, `[-1, 0]` (02a §5.4).

    A **shifted** exponential: exponential in shape as the source spec asks, but
    exactly zero beyond a cut-off rather than carrying a constant background.

    ```
    eps_cut = exp(-d_cut/d_oa)
    r_obs   = -clip( (exp(-d_clear/d_oa) - eps_cut) / (1 - eps_cut), 0, 1 )
    ```

    The unshifted form was rejected on arithmetic: at `d_oa = 0.8 m` it reads
    -0.08 at 2 m, integrating to about -53 over 300 steps at `w_obs = 2.2` --
    larger than the path term, constant, and carrying no gradient.  That is
    Paper 2's failure mode running in the opposite direction, and the `range(ep)`
    column of the audit exists to catch exactly it.

    `d_clear` is the minimum hull-to-hull clearance over static obstacles
    **within +/-135 deg**, matching the `c_t` swath, so the agent is never
    charged for proximity it cannot observe and an obstacle passed astern
    generates no signal against an action space with no reverse.
    """
    d = float(state.d_clear)
    if not np.isfinite(d):
        return 0.0
    eps_cut = math.exp(-float(cfg.d_cut) / max(float(cfg.d_oa), 1e-9))
    raw = (math.exp(-max(d, 0.0) / max(float(cfg.d_oa), 1e-9)) - eps_cut) / (1.0 - eps_cut)
    return float(np.clip(-raw, -1.0, 0.0))


def r_prog(state: RewardState, contexts, cfg) -> float:
    """Progress along the path, `[-1, +1]` (02a §5.5, 02b C3).

    ```
    r_prog = clip( N_ref * (s_t - s_{t-1}) / L_path, -1, +1 )
    ```

    `s` is **along-path arclength**, not distance-to-goal, which would penalise
    the outside of a bend -- the agent would be charged for taking the geometry
    the channel actually has.

    **`R-9`: this is why no progress carve-out is needed.**  `Sum r_prog`
    telescopes to `N_ref`, a constant fixed by nothing but the path, so slowing
    down costs **zero** progress reward provided the agent still finishes.  The
    only cost of a legal 8(e) slowdown is the extra steps' path penalty and
    existence cost -- about 4 points for a 3-second reduction, against roughly
    90 for the violation avoided.  That is strictly better than the gated
    carve-out `02 §4.4` asks for, because it removes the tension without
    introducing the creep exploit the same section warns about, and it needs no
    CRI threshold, so it stays consistent with `R-6` too.

    The clip makes the telescoping exact only for `u <= U_ref`, which is the
    normal regime, and has the useful side effect that speeding gains nothing.
    See `constants.py` §13.4 (F22) for why `N_ref` must be *derived* from
    `U_REF` rather than written down: at a fixed 250 the clip would bind below
    cruise, and slowing down would start to *increase* the integral.
    """
    l_path = max(float(state.l_path), 1e-6)
    return float(np.clip(cfg.n_ref_prog * float(state.ds) / l_path, -1.0, 1.0))


def r_exist(state: RewardState, contexts, cfg) -> float:
    """Existence cost, constant `-1` (02a §5.7).

    Scaled to zero by `R-5` while holding astern in a narrow overtaking, which
    is handled by the weight rather than here so that the term keeps its
    declared range and the ablation mask stays a mask.
    """
    return -1.0


def r_smooth(state: RewardState, contexts, cfg) -> float:
    """Action smoothness, `[-1, 0]` (02a §5.6).

    ```
    r_smooth = -sigma_t * clip( (Da_d/kappa_d)^2 + w_n*(Da_n/kappa_n)^2, 0, 1 )
    ```

    `kappa_delta` is the actuator's per-step rate limit in normalised units, so
    the term saturates at exactly the physical limit and self-calibrates when 05
    delivers the actuator model.  If `Dact` never approaches the limit in a
    render frame, the term is inert and the panel says so.

    **`sigma_t` resolves the Rule 8 tension** (`02 §4.3`).  Rule 8(b) wants one
    large alteration and forbids a succession of small ones; a plain smoothness
    penalty suppresses both.  The first `N_free` steps after engagement are
    charged at `sigma_enc`, so the committed alteration is affordable, and
    everything after is charged at full rate, so dithering is not.  Directly
    testable -- first-action magnitude is already a reported metric.
    """
    quad = ((float(state.d_rudder) / max(float(cfg.kappa_delta), 1e-9)) ** 2
            + float(cfg.w_n) * (float(state.d_throttle) / max(float(cfg.kappa_n), 1e-9)) ** 2)
    return float(np.clip(-smoothness_scale(state, contexts, cfg) * np.clip(quad, 0.0, 1.0),
                         -1.0, 0.0))


def smoothness_scale(state: RewardState, contexts, cfg) -> float:
    """`sigma_t`: `sigma_enc` inside the free window after engagement, else 1."""
    for ctx in _iter(contexts):
        if not ctx.engaged or ctx.t_engage < 0:
            continue
        since = int(state.step_index) - int(ctx.t_engage)
        if 0 <= since < int(cfg.n_free):
            return float(cfg.sigma_enc)
    return 1.0


# ---------------------------------------------------------------------------
# 2.  COLREGs terms  (02a §6)
# ---------------------------------------------------------------------------
def v_port(state: RewardState, ctx, cfg) -> float:
    """Turning the wrong way while give-way, `[0, 1]` (02a §6.2).

    ```
    r_err  = r - r_path
    v_port = rho_t * clip( (max(0, -s_c*r_err) - r_dead) / r_ref, 0, 1 )
    ```

    `s_c` is the compliant turn sense: `+1` for head-on and crossing, `-1` for
    overtaking.  Two things this shape fixes, both of them traps the documents
    name explicitly:

    1. **`02 §4.2`'s implementation trap.**  Overtaking *requires* a port turn.
       With `s_c = -1` the same expression penalises a *starboard* turn during
       an overtake, so there is no global "port is bad" constant to miscode.
    2. **The bend problem, and the withdrawal of `R-3`.**  Following a channel
       that bends to port requires a port turn that is path-following, not
       evasion.  Subtracting `r_path` measures only the *excess* yaw rate, so a
       compliant bend costs nothing -- which removes the unwinnable state `R-3`
       was patching.  `R-3` is now actively wrong: `02 §4.4` resolves the
       boundary conflict to *slacken speed*, so suppressing the penalty near the
       wall would license a port turn instead.

    `r_dead` is subtracted rather than used as a gate, so the term is continuous
    where it starts.  It fires on **yaw rate crossing `r_dead`, not on rudder
    reversal** -- locked principle 5, and the exact bug that principle exists to
    prevent, since a rudder movement that never develops into a turn is not an
    alteration of course.
    """
    if not (ctx.engaged and ctx.gives_way):
        return 0.0
    s_c = int(ctx.compliant_turn_sense)
    if s_c == 0:
        return 0.0
    wrong_way = max(0.0, -s_c * (float(state.r) - float(ctx.r_path)))
    severity = (wrong_way - float(cfg.r_dead)) / max(float(cfg.r_ref), 1e-9)
    return float(ctx.rho) * float(np.clip(severity, 0.0, 1.0))


def v_bow(state: RewardState, ctx, cfg) -> float:
    """Crossing ahead of the target, `[0, 1]` (02a §6.3).

    ```
    v_bow = rho_t * sigma_bow(beta_CPA) * clip(1 - DCPA/d_req, 0, 1)
    ```

    Evaluated at the constant-velocity projected CPA, consistently with
    DCPA/TCPA, and smooth in `beta_CPA` so there is no discontinuity at the
    beam.  For overtaking it catches cutting back across the bow after the pass
    -- the characteristic overtaking violation, and more likely here given the
    locked port-side pass followed by a required return to the starboard side.
    """
    if not (ctx.engaged and ctx.cls in (ctxmod.enc.CROSSING, ctxmod.enc.OVERTAKING)):
        return 0.0
    closeness = float(np.clip(1.0 - float(ctx.dcpa) / max(cfg.d_req, 1e-9), 0.0, 1.0))
    return float(ctx.rho) * geo.sigma_bow(ctx.beta_cpa, cfg.beta_bow_deg) * closeness


def v_side(state: RewardState, ctx, cfg) -> float:
    """Wrong-side passing, `[0, 1]` (02a §6.4).

    **The sign is class-dependent, and the two cases are opposite.**  This is
    the `02 §4.2` trap in its sharpest form:

    ```
    head-on:     required TS to port      ->  clip( +y_rel_CPA / d_req, 0, 1 )
    overtaking:  required TS to starboard ->  clip( -y_rel_CPA / d_req, 0, 1 )
    ```

    Head-on: port-to-port means the target ends on the own ship's port side, so
    `y_rel_CPA < 0` is correct and a positive value is the violation.
    Overtaking: passing to port **of the target** puts the own ship on the
    target's port side, which puts the target on the own ship's **starboard**,
    so `y_rel_CPA > 0` is correct and a negative value is the violation.
    Opposite signs, same geometry family -- which is why one test asserts both.

    Not applied to crossing: there the requirement is "pass astern", which
    `v_bow` already covers, and a side term would double-count it.

    Overtaking additionally requires `A_port`.  Where the port pass does not
    fit, the correct behaviour is to hold astern (`R-5`), and penalising the
    side of a pass that is not being attempted would charge the agent for
    complying.
    """
    if not ctx.engaged:
        return 0.0
    if ctx.cls == ctxmod.enc.HEAD_ON:
        signed = +float(ctx.y_rel_cpa)
    elif ctx.cls == ctxmod.enc.OVERTAKING:
        if not ctx.a_port:
            return 0.0
        signed = -float(ctx.y_rel_cpa)
    else:
        return 0.0
    return float(ctx.rho) * float(np.clip(signed / max(cfg.d_req, 1e-9), 0.0, 1.0))


def v_hold(state: RewardState, ctx, cfg) -> float:
    """Failing to hold course and speed while being overtaken, `[0, 1]` (17(a)(i)).

    ```
    v_hold = rho_t * clip( ((r - r_path)/r_hold)^2 + ((u - u_engage)/Du_hold)^2, 0, 1 )
    ```

    A penalty on deviation rather than a reward for holding, so the whole group
    stays a penalty group and the §7 hierarchy is uniform.  A positive hold
    reward would also duplicate `r_pf`, which already rewards a steady course,
    and would risk paying the agent to hold course into a collision.

    **Path-relative yaw here too.**  `02 §3.2` adds "keep starboard" to the
    being-overtaken row; an absolute-yaw formulation would penalise the
    corrective alteration needed to regain the starboard side, and `r - r_path`
    does not.

    **`R-4` -- the `in_extremis` suppression touches locked decision S5.**  Rule
    17(b) requires the stand-on vessel to act when collision cannot be avoided by
    the give-way vessel alone.  That is a different provision from 17(a)(ii),
    which S5 puts out of scope.  Suppression does not *reward* the release; it
    stops punishing it, and leaves the collision and domain penalties to
    dominate.  Without it the reward would contain a literal instruction to hold
    course into a collision.
    """
    if not (ctx.engaged and ctx.cls == ctxmod.enc.BEING_OVERTAKEN):
        return 0.0
    if ctx.in_extremis:
        return 0.0
    yaw = (float(state.r) - float(ctx.r_path)) / max(float(cfg.r_hold), 1e-9)
    surge = (float(state.u) - float(ctx.u_engage)) / max(float(cfg.du_hold), 1e-9)
    return float(ctx.rho) * float(np.clip(yaw * yaw + surge * surge, 0.0, 1.0))


def v_r8(state: RewardState, ctx, cfg) -> float:
    """Rule 8: late or insufficient action, deficit-based, `[0, 1]` (02a §6.6).

    ```
    A_req   = clip(Dy_req / d_req, 0, 1)                  # 0 when nothing is owed
    Dpsi_c  = max(0, s_c * (psi_t - psi_engage))          # compliant-sense heading change
    Du_red  = max(0, u_engage - u_t)
    A_t     = (Dpsi_c/Dpsi_min if turn_admissible else 0) + Du_red/Du_min
    urgency = clip(1 - TCPA/T_act, 0, 1)
    v_r8    = urgency * clip(A_req - A_t, 0, 1)
    ```

    **Reformulated in Revision 2, and it is the single most consequential change
    in that revision.**  `02 §3.2` now states that Rule 9(a) compliance can
    satisfy Rule 14 without any alteration at all.  A term penalising inaction
    whenever engaged would punish the agent for correctly holding course.

    It reads directly off the geometry:

    * Target properly placed, `DCPA >= d_req` -> `Dy_req = 0` -> `A_req = 0` ->
      **`v_r8 = 0`**.  Channel-keeping satisfies Rule 14, exactly as the
      precedence table requires.
    * Target displaced -> `A_req` scales with the deficit, so the obligation is
      proportionate rather than a fixed 20 degrees.
    * Alteration inadmissible -> `A_t` counts the speed reduction only, so Rule
      8(e) discharges the obligation.  That is `02 §4.4`'s decision expressed as
      reward structure rather than as a special case.
    * Counts **only compliant directions**, so a large wrong-way alteration
      discharges nothing.

    Time-to-first-action and first-action-magnitude fall out of `A_t` and are
    logged from it rather than recomputed.
    """
    return r8_parts(state, ctx, cfg)["value"]


def r8_parts(state: RewardState, ctx, cfg) -> dict:
    """`v_r8` and every intermediate, so the panel and the term cannot disagree.

    `A_req` is the health check on the deficit reformulation: **if `A_req` reads
    1.00 in every head-on episode, the spawn-DCPA gap (02a §11.1) is back** and
    the agent is being trained on "always alter" rather than "when to alter".
    That is invisible in the term's value alone, which is why the parts are
    exported rather than recomputed for display.
    """
    parts = {"a_req": float(ctx.a_req), "dpsi_c": 0.0, "du_red": 0.0,
             "a_t": 0.0, "urgency": 0.0, "value": 0.0}
    if not (ctx.engaged and ctx.gives_way):
        return parts
    s_c = int(ctx.compliant_turn_sense)
    if s_c == 0:
        return parts

    dpsi = _wrap180(float(state.heading_deg) - float(ctx.psi_engage))
    parts["dpsi_c"] = max(0.0, s_c * dpsi)
    parts["du_red"] = max(0.0, float(ctx.u_engage) - float(state.u))

    a_t = parts["du_red"] / max(float(cfg.du_min), 1e-9)
    if ctx.turn_admissible:
        a_t += parts["dpsi_c"] / max(float(cfg.dpsi_min_deg), 1e-9)
    parts["a_t"] = a_t

    parts["urgency"] = float(np.clip(
        1.0 - float(ctx.tcpa) / max(float(cfg.t_act), 1e-9), 0.0, 1.0))
    parts["value"] = parts["urgency"] * float(np.clip(parts["a_req"] - a_t, 0.0, 1.0))
    return parts


def explain_colregs(state: RewardState, ctx, cfg) -> Dict[str, str]:
    """Why each sub-term holds the value it holds (RENDER_PANEL_SPEC §1).

    When `v_side` reads 0.000 you cannot tell from the number whether that is
    correct -- wrong class for this term -- or a bug: gate stuck, `rho_t`
    collapsed, class never latched.  This is the difference between a panel you
    glance at and a panel you debug with, and it costs one string per term.

    Every string is a statement about the context.  Nothing here re-evaluates a
    term, so the reason and the value cannot drift apart.
    """
    cls = ctx.cls
    if not ctx.engaged:
        common = f"not engaged ({ctx.state})"
        return {name: common for name in COLREGS_TERMS}

    sense = {1: "STBD", -1: "PORT", 0: "none"}[int(ctx.compliant_turn_sense)]
    out = {}

    if cls not in GIVE_WAY:
        out["port"] = f"n/a for {cls}"
    elif ctx.compliant_turn_sense == 0:
        out["port"] = "no compliant sense"
    else:
        excess = -ctx.compliant_turn_sense * (float(state.r) - float(ctx.r_path))
        if excess <= cfg.r_dead:
            out["port"] = f"turning {sense} or holding, compliant"
        else:
            out["port"] = f"turning against {sense} at {excess:+.3f} rad/s"

    if cls not in (ctxmod.enc.CROSSING, ctxmod.enc.OVERTAKING):
        out["bow"] = f"n/a for {cls}"
    elif ctx.dcpa >= cfg.d_req:
        out["bow"] = f"DCPA {ctx.dcpa / max(cfg.d_req, 1e-9):.2f} dreq, clear"
    else:
        out["bow"] = (f"beta_CPA {ctx.beta_cpa:.0f}, "
                      f"DCPA {ctx.dcpa / max(cfg.d_req, 1e-9):.2f} dreq")

    if cls == ctxmod.enc.HEAD_ON:
        side = "STBD" if ctx.y_rel_cpa > 0 else "PORT"
        out["side"] = f"TS to {side} at CPA, needs PORT"
    elif cls == ctxmod.enc.OVERTAKING:
        if not ctx.a_port:
            out["side"] = "port pass inadmissible, hold astern (R-5)"
        else:
            side = "STBD" if ctx.y_rel_cpa > 0 else "PORT"
            out["side"] = f"TS to {side} at CPA, needs STBD"
    else:
        out["side"] = f"n/a for {cls}"

    if cls != ctxmod.enc.BEING_OVERTAKEN:
        out["hold"] = f"wrong class ({cls})"
    elif ctx.in_extremis:
        out["hold"] = "in extremis, 17(b) release"
    else:
        out["hold"] = f"holding from psi {ctx.psi_engage:.0f} u {ctx.u_engage:.2f}"

    if cls not in GIVE_WAY:
        out["r8"] = f"n/a for {cls}"
    else:
        parts = r8_parts(state, ctx, cfg)
        if parts["a_req"] <= 1e-6:
            out["r8"] = "A_req 0: 9(a) compliance satisfies 14"
        elif not ctx.turn_admissible:
            out["r8"] = f"8(e) only: no {sense} room, A_t {parts['a_t']:.2f}"
        elif parts["a_t"] >= parts["a_req"]:
            out["r8"] = f"A_t {parts['a_t']:.2f} >= A_req {parts['a_req']:.2f}, discharged"
        else:
            out["r8"] = (f"A_t {parts['a_t']:.2f} < A_req {parts['a_req']:.2f}, "
                         f"urgency {parts['urgency']:.2f}")
    return out


GIVE_WAY = ctxmod.GIVE_WAY_CLASSES

COLREGS_TERMS = {
    "port": v_port,
    "bow": v_bow,
    "side": v_side,
    "hold": v_hold,
    "r8": v_r8,
}


def colregs_group(state: RewardState, contexts, cfg) -> dict:
    """The weighted, clipped COLREGs group and its parts (02a §6.7).

    ```
    v_col = clip(0.55*v_port + 0.55*v_bow + 0.40*v_side + 0.45*v_hold + 0.50*v_r8, 0, 1)
    r_col = -v_col
    ```

    **Clipped to unit range before the group weight.**  No combination of
    violations can exceed `w_COL` in one step, so the group cannot silently
    outrank the safety terms the way Paper 2's path term outranked its avoidance
    term.  Pre-clip maxima are 1.45 head-on, 1.60 crossing, 1.50 overtaking and
    0.45 being overtaken: two concurrent severe violations saturate, one does not.

    With one target the outer aggregation is a no-op.  At `N_MAX_TARGETS > 1`
    the *worst* target governs rather than the sum, so meeting two vessels
    cannot exceed the penalty for meeting one badly -- the group weight has to
    stay the maximum per-step contribution or the §7 hierarchy stops being a
    property of the table.
    """
    sub_w = {"port": cfg.w_port, "bow": cfg.w_bow, "side": cfg.w_side,
             "hold": cfg.w_hold, "r8": cfg.w_r8}
    best = {"v_col": 0.0, "parts": {k: 0.0 for k in COLREGS_TERMS},
            "pre_clip": 0.0, "track_id": None}

    if not cfg.colregs_terms_enabled:
        return best

    for ctx in _iter(contexts):
        parts = {}
        for name, fn in COLREGS_TERMS.items():
            parts[name] = fn(state, ctx, cfg) if name in cfg.colregs_term_mask else 0.0
        pre_clip = sum(sub_w[name] * value for name, value in parts.items())
        v_col = float(np.clip(pre_clip, 0.0, 1.0))
        if v_col >= best["v_col"]:
            best = {"v_col": v_col, "parts": parts, "pre_clip": float(pre_clip),
                    "track_id": ctx.track_id}
    return best


def r_col(state: RewardState, contexts, cfg) -> float:
    """The COLREGs group as a penalty in `[-1, 0]`."""
    return -colregs_group(state, contexts, cfg)["v_col"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DENSE_TERMS = {
    "pf": r_pf,
    "prog": r_prog,
    "exist": r_exist,
    "smooth": r_smooth,
    "obs": r_obs,
    "bnd": r_bnd,
    "dom": r_dom,
    "col": r_col,
}

# Declared range per term, asserted by 02a §10.4 test 1.
TERM_RANGE = {name: (-1.0, 0.0) for name in DENSE_TERMS}
TERM_RANGE["prog"] = (-1.0, 1.0)


def _iter(contexts):
    """Iterate contexts whether they arrive as a dict, a sequence or one object."""
    if contexts is None:
        return ()
    if isinstance(contexts, dict):
        return tuple(contexts.values())
    if hasattr(contexts, "track_id"):
        return (contexts,)
    return tuple(contexts)


def _bearing_from_ctx(ctx) -> float:
    """Relative bearing to use for the domain radius.

    `alpha` is the perceived bearing; the domain is evaluated on ground truth.
    At the ranges where `r_dom` is non-zero the two differ by far less than the
    domain's own asymmetry, so the perceived bearing is used rather than
    carrying a second bearing through the context for a sub-degree correction.
    """
    return float(ctx.alpha)


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0
