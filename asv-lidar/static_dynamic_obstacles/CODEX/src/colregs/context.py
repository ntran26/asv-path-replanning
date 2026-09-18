"""`EncounterContext`: one object per target per step, three consumers.

02a §10.1 asks for a single per-step context read by the observation builder,
the reward and the metrics logger.  The mechanical reason is `01 §5.3`'s "one
module, two consumers": if each recomputes the class, the risk and the CPA
products from the same inputs, they will eventually disagree at a threshold --
and an agent penalised for a role it was never shown is close to undiagnosable
from a training curve.

`R-1` runs through every field here, and the dataclass is laid out to make the
split visible rather than to be tidy:

| Group | Input | Why |
|---|---|---|
| Rule regime -- class, DCPA/TCPA, the gates | **Perceived** | Never penalise a role the agent was not shown |
| Physical consequence -- true range, true DCPA | **Ground truth** | The agent pays for hitting things whether or not it saw them |
| Admissibility -- `d_bnd_*`, `A_stbd`, `A_port` | **Map** | Where the vessel fits is a physical fact, not an estimate |

Under Study 2's degradation sweep this makes COLREGs obligations degrade with
perception while physical safety obligations do not -- which is the real
situation at sea, and turns Study 2 into a study of obligation under
uncertainty rather than a robustness curve.

**This is not a breach of the information-parity argument in `04 §8`.** The
ground-truth fields are consumed by the *reward*, which exists only at training
time.  At evaluation the policy reads the observation and nothing else, so it
consumes the same tracked target state -- errors included -- as the comparators.
Say that explicitly in the methods, or `R-1` reads as the learned policy being
handed privileged information, and it undercuts both the parity claim and N1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

import constants as cfg
import cpa_cri as cc
import encounter as enc
from colregs import geometry as geo

# Engagement states (02a §6.1).
IDLE = "idle"
ENGAGED = "engaged"
CLEARING = "clearing"

# The three classes that carry a give-way obligation.  `being_overtaken` is
# stand-on; `none` is nothing.
GIVE_WAY_CLASSES = (enc.HEAD_ON, enc.CROSSING, enc.OVERTAKING)

# `02 §3.2`'s precedence table, as a lookup rather than as branching.  **This
# single field is the mechanism that prevents the `02 §4.2` implementation
# trap** -- overtaking requires a PORT turn, so there is no global "port turns
# are penalised" constant anywhere in this tree, and nothing to miscode.
_TURN_SENSE = {
    enc.HEAD_ON: +1,          # alter to starboard, subject to width
    enc.CROSSING: +1,         # from starboard; -1 from port (A17): pass astern, never ahead
    enc.OVERTAKING: -1,       # pass to PORT of the target, then regain starboard
    enc.BEING_OVERTAKEN: 0,   # hold course and speed
    enc.NONE: 0,
}


def compliant_turn_sense(encounter_class: str, crossing_side: str = enc.SIDE_NONE) -> int:
    """`+1` starboard, `-1` port, `0` no alteration required (02a §2.1).

    **A17 (decided): a crossing's sense depends on the side it crosses from.**
    S3 keeps the own ship give-way either way, and giving way to a crossing
    vessel means passing astern of it.  From starboard that is a starboard turn;
    from port it is a port turn.  A starboard turn there carries the own ship
    along the target's track: run 2 collided in 0.75 of port crossings at
    DCPA > 1.4 m doing exactly that, because `v_port` paid for it.
    """
    if encounter_class == enc.CROSSING and crossing_side == enc.SIDE_PORT:
        return -1
    return int(_TURN_SENSE.get(encounter_class, 0))


@dataclass
class EncounterContext:
    """Everything the reward, the observation and the metrics need about one target."""

    track_id: int

    # --- perceived: feeds the observation AND the COLREGs gating -----------
    cls: str = enc.NONE
    alpha: float = 0.0                 # relative bearing OS->TS, deg
    ct: float = 0.0                    # heading intersection angle, deg
    dcpa: float = float("inf")
    tcpa: float = 0.0
    cri: float = 0.0
    y_rel_cpa: float = 0.0             # +ve = TS to starboard at the CPA
    beta_cpa: float = 0.0              # bearing of OS from TS at the CPA, deg
    rng: float = float("inf")          # present range, perceived
    speed_ts: float = 0.0
    # Both measured to the ship domain rather than to the hull (01 §5.2), and
    # both perceived.  Held here so `observation.slot_features` is a pure
    # function of the context and cannot drift from the reward's view of the
    # same encounter.
    d_domain: float = float("inf")     # range from the domain boundary to the TS
    dcpa_domain: float = float("inf")  # DCPA less the domain radius at the bearing
    v_rel: float = 0.0                 # relative speed magnitude, m/s
    # The A18 stop test's view at close range (F66): range, bearing and
    # heading-intersection from the hull-fitted centre and axis.  None beyond
    # `STOP_TEST_FIT_RANGE_M`, or without a fit, when the track's own values are used.
    stop_rng: Optional[float] = None
    stop_alpha: Optional[float] = None
    stop_ct: Optional[float] = None
    # Own-ship surge when the context was built: where the braking path of
    # the A23 stop test starts from.
    u_own: float = 0.0

    # --- latched on engagement --------------------------------------------
    state: str = IDLE
    psi_engage: float = 0.0            # OS heading when the encounter engaged, deg
    u_engage: float = 0.0              # OS surge when it engaged, m/s
    t_engage: int = -1                 # step index at engagement
    compliant_turn_sense: int = 0

    # --- admissibility: from the map, ground truth ------------------------
    a_stbd: bool = True
    a_port: bool = True
    r_stbd: float = float("inf")
    r_port: float = float("inf")
    d_bnd_stbd: float = float("inf")
    d_bnd_port: float = float("inf")
    dy_req: float = 0.0
    a_req: float = 0.0
    admissibility_known: bool = False

    # --- path reference (`R-8`) -------------------------------------------
    r_path: float = 0.0                # yaw rate required to track the path, rad/s

    # --- ground truth: safety terms and metrics only ----------------------
    d_ts_true: float = float("inf")
    dcpa_true: float = float("inf")
    cls_true: str = enc.NONE
    crossing_side: str = enc.SIDE_NONE

    # --- gates ------------------------------------------------------------
    rho: float = 0.0                   # proximity gate, [0, 1]
    engaged: bool = False
    in_extremis: bool = False

    @property
    def turn_admissible(self) -> bool:
        """Is the *compliant* alteration for this class admissible?

        `A_stbd` for head-on and crossing, `A_port` for overtaking (02a §6.6).
        Read off `compliant_turn_sense` rather than off the class name, which
        keeps the `02 §4.2` trap closed in one more place.
        """
        if self.compliant_turn_sense > 0:
            return bool(self.a_stbd)
        if self.compliant_turn_sense < 0:
            return bool(self.a_port)
        return True

    @property
    def misclassified(self) -> bool:
        """Perceived class differs from the true one.

        `04 §6` names turning the wrong way after a misclassification as the
        failure that matters, so it is counted rather than inferred.
        """
        return self.cls != self.cls_true

    @property
    def gives_way(self) -> bool:
        return self.cls in GIVE_WAY_CLASSES

    @property
    def dcpa_if_stopped(self) -> float:
        """The target's closest approach if the own ship stopped now, perceived.

        A18 evaluated this with the own ship stationary where it is.  **A23
        (decided, option 1)** evaluates it along the own ship's braking path --
        the supervisor latch's full astern from its present surge `u_own`, then
        stopped (`stopping.dcpa_over_stop`).  At `u_own` below the stop speed
        the two agree exactly.
        """
        import stopping
        # F66 (C15): at close range the stop test can read the hull-fitted
        # centre and axis instead of the centroid track.
        use_fit = self.stop_rng is not None
        return stopping.dcpa_over_stop(
            self.stop_rng if use_fit else self.rng,
            self.stop_alpha if use_fit else self.alpha,
            self.stop_ct if use_fit else self.ct,
            self.speed_ts, self.u_own)

    @property
    def dcpa_if_slowed(self) -> float:
        """The target's closest approach if the policy itself slowed now (F68).

        The same test as `dcpa_if_stopped`, on the policy's own slowdown -- a
        coast at the propulsion floor, with no reverse -- rather than the
        supervisor's full astern.  This is what R-2's carve-out and the Rule 8
        alteration credit ask about: whether *the agent's* slowdown helps.
        """
        import stopping
        return stopping.dcpa_over_stop(self.rng, self.alpha, self.ct,
                                       self.speed_ts, self.u_own, mode="coast")

    @property
    def slowdown_clears(self) -> bool:
        """Would the policy's own slowdown let the target pass clear?  (F68)"""
        return self.dcpa_if_slowed >= cfg.ESTOP_CLEAR_DCPA_M

    @property
    def stop_clears(self) -> bool:
        """Would stopping, alone, let the target pass clear?  (A18)"""
        return self.dcpa_if_stopped >= cfg.ESTOP_CLEAR_DCPA_M


class ContextManager:
    """Builds one `EncounterContext` per track per step, and owns the latches.

    Stateful in three ways, all deliberate and all here rather than spread
    across the consumers:

    * the **encounter classifier's** hysteresis, delegated to `encounter.py` so
      that module stays the single definition of the angular bands;
    * the **engagement state machine** and what it latches;
    * the **admissibility hysteresis**, a +/-0.15 m band on `r_* - Dy_req` so
      the predicate does not chatter when the vessel sits at the width where
      the manoeuvre just fits.
    """

    def __init__(self, *, hold_steps: int = cfg.ENCOUNTER_HOLD_STEPS,
                 t_engage: float = cfg.T_ENGAGE,
                 kappa_eng: float = cfg.KAPPA_ENG,
                 kappa_rel: float = cfg.KAPPA_REL,
                 n_clear: int = cfg.N_CLEAR_STEPS,
                 n_switch: int = cfg.N_SWITCH_STEPS,
                 t_extremis: float = cfg.T_EXTREMIS,
                 admissibility_band: float = 0.15) -> None:
        self.classifier = enc.EncounterClassifier(hold_steps=hold_steps)
        self.t_engage = float(t_engage)
        self.kappa_eng = float(kappa_eng)
        self.kappa_rel = float(kappa_rel)
        self.n_clear = int(n_clear)
        self.n_switch = int(n_switch)
        self.t_extremis = float(t_extremis)
        self.band = float(admissibility_band)
        self.reset()

    def reset(self) -> None:
        self.classifier.reset()
        self.step_index = 0
        self._latch: Dict[int, dict] = {}
        self._held_adm: Dict[int, Dict[str, bool]] = {}
        self.contexts: Dict[int, EncounterContext] = {}

    def forget(self, track_id: int) -> None:
        self.classifier.forget(track_id)
        self._latch.pop(track_id, None)
        self._held_adm.pop(track_id, None)

    # ------------------------------------------------------------------
    def update(self, *, tracks: Sequence, p_os, v_os, heading_os_deg: float,
               u_os: float = 0.0, r_path: float = 0.0,
               path=None, boundary_polygon=None,
               s_along: Optional[float] = None, cross_track: float = 0.0,
               true_targets: Sequence = (), open_water: bool = False,
               advance_clock: bool = True) -> Dict[int, EncounterContext]:
        """Build this step's contexts.  Returns `{track_id: EncounterContext}`.

        `path`, `boundary_polygon` and `s_along` are what the admissibility
        predicate needs.  When any is absent -- a unit test exercising only the
        perceived half, or `R-10`'s open-water benchmark variant -- the
        predicate degrades to "no lateral constraint", and
        `admissibility_known` records that it did, so a permissive answer can
        never be mistaken for a measured one.
        """
        if advance_clock:
            self.step_index += 1

        speed_os = float(np.linalg.norm(np.asarray(v_os, dtype=np.float64)))
        d_required = geo.d_req()

        live = {t.id for t in tracks}
        for tid in [tid for tid in list(self._latch) + list(self._held_adm)
                    if tid not in live]:
            self.forget(tid)

        # A19: the encounter is classified against the course being kept, not
        # the momentary heading, so the own ship's alteration cannot re-label it.
        heading_cls = self._path_heading(path, s_along, heading_os_deg)

        contexts: Dict[int, EncounterContext] = {}
        for track in tracks:
            ctx = self._build(track, p_os, v_os, heading_os_deg, speed_os,
                              u_os, r_path, true_targets, d_required,
                              heading_cls_deg=heading_cls)
            self._attach_admissibility(ctx, path, boundary_polygon, s_along,
                                       cross_track, u_os, open_water, d_required)
            self._advance_state(ctx, heading_os_deg, u_os, d_required)
            self._attach_gates(ctx, d_required)
            contexts[track.id] = ctx

        self.contexts = contexts
        return contexts

    def attach_truth(self, *, tracks: Sequence, p_os, v_os,
                     heading_os_deg: float, true_targets: Sequence,
                     path=None, s_along: Optional[float] = None) -> None:
        """Attach diagnostics using the physical own-ship pose and velocity.

        Call after ``update(..., true_targets=())`` when perception and physical
        state differ.  Pairing still uses each perceived track's position, but
        all ``*_true`` geometry uses physical own-ship and target states.  This
        never changes the perceived obligation or any policy input.
        """
        speed_os = float(np.linalg.norm(np.asarray(v_os, dtype=np.float64)))
        heading_cls = self._path_heading(path, s_along, heading_os_deg)
        for track in tracks:
            ctx = self.contexts.get(track.id)
            if ctx is None:
                continue
            ctx.d_ts_true = ctx.dcpa_true = float("inf")
            ctx.cls_true = enc.NONE
            self._attach_truth(ctx, track.position, p_os, v_os, heading_cls,
                               speed_os, true_targets)

    # ------------------------------------------------------------------
    @staticmethod
    def _path_heading(path, s_along, heading_os_deg: float) -> float:
        """The reference path's course at the own ship, degrees (A19).

        **A19 (decided, option 1).**  Classifying against the instantaneous
        heading let the own ship's compliant starboard alteration move a head-on
        out of its band and re-label it a crossing from port, which A17 then
        paid to turn back toward: 67 % of the run 2 model's head-on episodes
        latched a port sense (F55).  The path tangent is the course the own ship
        is keeping, and an alteration does not move it.  Without a path --
        `R-10`'s open water, or a unit test of the perceived half -- the
        instantaneous heading is all there is.
        """
        if path is None or s_along is None:
            return float(heading_os_deg)
        idx = int(np.clip(np.searchsorted(path.s, float(s_along)), 0, len(path.points) - 1))
        tangent = path.tangent(idx)
        return float(np.degrees(np.arctan2(float(tangent[0]), float(tangent[1]))) % 360.0)

    def _build(self, track, p_os, v_os, heading_os_deg, speed_os, u_os,
               r_path, true_targets, d_required,
               heading_cls_deg: Optional[float] = None) -> EncounterContext:
        p_ts = track.position
        v_ts = track.velocity
        heading_ts = track.course_deg
        # Class and side from the kept course (A19); CPA products from the real
        # heading and velocity, which is what the collision geometry is.
        heading_cls = float(heading_os_deg if heading_cls_deg is None else heading_cls_deg)

        cls = self.classifier.update(track.id, p_os, heading_cls, speed_os,
                                     p_ts, heading_ts, track.speed)
        products = geo.cpa_products(p_os, v_os, heading_os_deg,
                                    p_ts, v_ts, heading_ts)

        ctx = EncounterContext(
            track_id=track.id,
            cls=cls,
            alpha=products["alpha"],
            ct=products["ct"],
            dcpa=products["dcpa"],
            tcpa=products["tcpa"],
            cri=cc.cri(p_os, v_os, heading_os_deg, p_ts, v_ts, heading_ts),
            y_rel_cpa=products["y_rel_cpa"],
            beta_cpa=products["beta_cpa"],
            rng=products["range"],
            speed_ts=float(track.speed),
            u_own=float(u_os),
            r_path=float(r_path),
            crossing_side=enc.crossing_side(p_os, heading_cls, p_ts, heading_ts),
            d_domain=cc.distance_to_domain(p_os, heading_os_deg, p_ts),
            dcpa_domain=max(0.0, products["dcpa"]
                            - cc.domain_scale(products["alpha"])),
            v_rel=float(np.linalg.norm(np.asarray(v_ts, dtype=np.float64)
                                       - np.asarray(v_os, dtype=np.float64))),
        )
        self._attach_stop_view(ctx, track, p_os, heading_os_deg)
        self._attach_truth(ctx, p_ts, p_os, v_os, heading_cls, speed_os,
                           true_targets)
        return ctx

    @staticmethod
    def _attach_stop_view(ctx, track, p_os, heading_os_deg: float) -> None:
        """F66 (C15): the stop test's close-range view from the hull fit.

        Only the stop test reads it.  Feeding the fit into the track state
        (F64) cut close-range error but, as a time-varying correction, read as
        velocity at engagement range; a view that only a boolean test consumes
        has no such effect.
        """
        centre = getattr(track, "last_fit_centre", None)
        if not cfg.STOP_TEST_USES_HULL_FIT or centre is None:
            return
        rng = float(np.linalg.norm(np.asarray(centre, dtype=np.float64)
                                   - np.asarray(p_os, dtype=np.float64)))
        if rng > float(cfg.STOP_TEST_FIT_RANGE_M):
            return
        ctx.stop_rng = rng
        ctx.stop_alpha = cc.relative_bearing_deg(p_os, heading_os_deg, centre)
        fit_heading = getattr(track, "last_fit_heading_deg", None)
        ctx.stop_ct = (cc.heading_intersection_deg(heading_os_deg, fit_heading)
                       if fit_heading is not None else ctx.ct)

    # ------------------------------------------------------------------
    def _attach_truth(self, ctx, p_ts, p_os, v_os, heading_os_deg, speed_os,
                      true_targets) -> None:
        """Pair the track with the nearest true target: the physical facts.

        Attributed by proximity rather than by list index, for the same reason
        the perception metrics are: a static return promoted to a dynamic track
        by pose drift is a false positive, and pairing it with a real target by
        position in a list would quietly launder that into the ground truth.
        Unpaired tracks keep their `inf` defaults, so a safety term reading them
        contributes nothing rather than something wrong.
        """
        if not true_targets:
            return

        px, py = float(p_ts[0]), float(p_ts[1])
        best, best_gap = None, float("inf")
        for target in true_targets:
            gap = float(np.hypot(float(target.x) - px, float(target.y) - py))
            if gap < best_gap:
                best, best_gap = target, gap
        if best is None or best_gap > cfg.TARGET_MATCH_RADIUS:
            return

        p_true = (float(best.x), float(best.y))
        ctx.d_ts_true = float(np.hypot(p_true[0] - float(p_os[0]),
                                       p_true[1] - float(p_os[1])))
        ctx.dcpa_true = float(cc.cpa(p_os, v_os, p_true, best.velocity)[0])
        ctx.cls_true = enc.classify(p_os, heading_os_deg, speed_os,
                                    p_true, best.heading_deg, best.speed)

    # ------------------------------------------------------------------
    def _attach_admissibility(self, ctx, path, polygon, s_along, cross_track,
                              u_os, open_water, d_required) -> None:
        ctx.dy_req = geo.lateral_deficit(ctx.dcpa, d_required)
        ctx.a_req = float(np.clip(ctx.dy_req / d_required, 0.0, 1.0))

        if open_water or path is None or polygon is None or s_along is None:
            # `R-10`: with no boundary there is no lateral constraint, both
            # alterations are admissible and the room is unbounded.
            ctx.a_stbd = ctx.a_port = True
            ctx.r_stbd = ctx.r_port = float("inf")
            ctx.d_bnd_stbd = ctx.d_bnd_port = float("inf")
            ctx.admissibility_known = bool(open_water)
            return

        # The passage over which the room is minimised: from here to the
        # projected CPA.  Once the CPA is behind, the interval collapses to the
        # present station, which is the right answer -- there is no future
        # passage left to fit into.
        run = max(0.0, float(ctx.tcpa)) * max(float(u_os), 0.0)
        d_stbd, d_port = geo.channel_room(path, polygon, float(s_along),
                                          float(s_along) + run, float(cross_track))
        ctx.d_bnd_stbd, ctx.d_bnd_port = d_stbd, d_port
        ctx.r_stbd = geo.usable_room(d_stbd)
        ctx.r_port = geo.usable_room(d_port)
        ctx.a_stbd = self._hysteretic(ctx.track_id, "stbd", ctx.r_stbd - ctx.dy_req)
        ctx.a_port = self._hysteretic(ctx.track_id, "port", ctx.r_port - ctx.dy_req)
        ctx.admissibility_known = True

    def _hysteretic(self, track_id: int, side: str, margin: float) -> bool:
        """Latch admissibility across a +/-`band` metre dead zone on the margin.

        Without it a vessel holding station at the width where a manoeuvre just
        fits would flip `A_stbd` every step, and `v_r8` would alternate between
        counting the turn and not counting it -- so the Rule 8 obligation would
        appear and vanish at the exact geometry Study 1 is trying to resolve.
        """
        held = self._held_adm.setdefault(track_id, {})
        current = held.get(side)
        if margin >= self.band:
            current = True
        elif margin <= -self.band:
            current = False
        elif current is None:
            current = margin >= 0.0
        held[side] = current
        return bool(current)

    # ------------------------------------------------------------------
    def _advance_state(self, ctx, heading_os_deg, u_os, d_required) -> None:
        latch = self._latch.get(ctx.track_id)
        state = latch["state"] if latch else IDLE

        if state == IDLE:
            if (ctx.cls != enc.NONE and 0.0 < ctx.tcpa <= self.t_engage
                    and ctx.dcpa < self.kappa_eng * d_required):
                latch = self._engage(ctx, heading_os_deg, u_os)

        elif state == ENGAGED:
            # `TCPA < 0` is exactly "range opening": TCPA is
            # `-(p . v)/|v|^2`, so a negative value and a positive range rate
            # are the same statement about the same two vectors.
            if ctx.tcpa < 0.0 or ctx.dcpa > self.kappa_rel * d_required:
                latch["state"] = CLEARING
                latch["clear_steps"] = 0
            # **A20 (decided, option 1): no re-engagement on a class switch.**
            # COLREGs decides the situation when risk of collision first
            # develops.  Re-deciding it whenever a different class persisted for
            # `N_SWITCH_STEPS` let close-range bearing drift and tracker course
            # noise turn a head-on into a crossing from port at a median 3.6 m,
            # and, since A17, flip the turn sense (F56).  The class, side and
            # sense latched at engagement stand until the encounter clears.

        elif state == CLEARING:
            # Clearing is a confirmation window, not a grace period in which
            # the vessel may turn back into the encounter without obligation.
            # Require continuous safe geometry.  If risk returns, restore the
            # original latch rather than reclassifying from close-range angles
            # or resetting the heading/speed against which action is judged.
            remains_clear = (ctx.tcpa < 0.0
                             or ctx.dcpa > self.kappa_rel * d_required)
            if not remains_clear:
                latch["state"] = ENGAGED
                latch["clear_steps"] = 0
            else:
                # A predicted safe DCPA is not yet a completed passage.  Keep
                # the original obligation available while closing; release
                # only after continuously opening outside required separation.
                passed_clear = ctx.tcpa < 0.0 and ctx.rng > d_required
                latch["clear_steps"] = (latch.get("clear_steps", 0) + 1
                                        if passed_clear else 0)
                if passed_clear and latch["clear_steps"] >= self.n_clear:
                    self._latch.pop(ctx.track_id, None)
                    latch = None

        if latch is not None:
            self._latch[ctx.track_id] = latch
            ctx.state = latch["state"]
            # A20: the latched class is *the* class while the encounter is
            # engaged or clearing, so the observation's one-hot and every
            # reward gate still read one field (01 §5.3).
            ctx.cls = latch["cls"]
            ctx.crossing_side = latch.get("crossing_side", ctx.crossing_side)
            ctx.psi_engage = latch["psi_engage"]
            ctx.u_engage = latch["u_engage"]
            ctx.t_engage = latch["t_engage"]
            ctx.compliant_turn_sense = latch["turn_sense"]
        else:
            ctx.state = IDLE
            ctx.compliant_turn_sense = compliant_turn_sense(ctx.cls, ctx.crossing_side)

        ctx.engaged = ctx.state == ENGAGED

    def _engage(self, ctx, heading_os_deg, u_os) -> dict:
        latch = {
            "state": ENGAGED,
            "cls": ctx.cls,
            "psi_engage": float(heading_os_deg),
            "u_engage": float(u_os),
            "t_engage": int(self.step_index),
            "turn_sense": compliant_turn_sense(ctx.cls, ctx.crossing_side),
            "crossing_side": ctx.crossing_side,
            "switch_steps": 0,
            "clear_steps": 0,
        }
        self._latch[ctx.track_id] = latch
        return latch

    # ------------------------------------------------------------------
    def _attach_gates(self, ctx, d_required) -> None:
        # `rho_t` -- the proximity gate every COLREGs severity multiplies by.
        # At a compliant pass (DCPA = d_req) it reads 0.33, but every severity
        # is zero there, so nothing accrues at the geometry the agent is being
        # asked to produce.
        ctx.rho = float(np.clip(1.0 - ctx.dcpa / (self.kappa_eng * d_required), 0.0, 1.0))
        ctx.in_extremis = bool(ctx.dcpa < d_required and 0.0 <= ctx.tcpa < self.t_extremis)
