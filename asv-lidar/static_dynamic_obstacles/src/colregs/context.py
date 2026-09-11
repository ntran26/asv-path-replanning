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
    enc.CROSSING: +1,         # alter to starboard and/or slacken; never cross ahead
    enc.OVERTAKING: -1,       # pass to PORT of the target, then regain starboard
    enc.BEING_OVERTAKEN: 0,   # hold course and speed
    enc.NONE: 0,
}


def compliant_turn_sense(encounter_class: str) -> int:
    """`+1` starboard, `-1` port, `0` no alteration required (02a §2.1)."""
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

        contexts: Dict[int, EncounterContext] = {}
        for track in tracks:
            ctx = self._build(track, p_os, v_os, heading_os_deg, speed_os,
                              u_os, r_path, true_targets, d_required)
            self._attach_admissibility(ctx, path, boundary_polygon, s_along,
                                       cross_track, u_os, open_water, d_required)
            self._advance_state(ctx, heading_os_deg, u_os, d_required)
            self._attach_gates(ctx, d_required)
            contexts[track.id] = ctx

        self.contexts = contexts
        return contexts

    # ------------------------------------------------------------------
    def _build(self, track, p_os, v_os, heading_os_deg, speed_os, u_os,
               r_path, true_targets, d_required) -> EncounterContext:
        p_ts = track.position
        v_ts = track.velocity
        heading_ts = track.course_deg

        cls = self.classifier.update(track.id, p_os, heading_os_deg, speed_os,
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
            r_path=float(r_path),
            crossing_side=enc.crossing_side(p_os, heading_os_deg, p_ts, heading_ts),
            d_domain=cc.distance_to_domain(p_os, heading_os_deg, p_ts),
            dcpa_domain=max(0.0, products["dcpa"]
                            - cc.domain_scale(products["alpha"])),
            v_rel=float(np.linalg.norm(np.asarray(v_ts, dtype=np.float64)
                                       - np.asarray(v_os, dtype=np.float64))),
        )
        self._attach_truth(ctx, p_ts, p_os, v_os, heading_os_deg, speed_os,
                           true_targets)
        return ctx

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
            elif ctx.cls != latch["cls"]:
                # A genuinely different class must persist before the Rule 8
                # accumulator is re-based.  A one-step flicker that reset
                # `psi_engage` would make the agent's committed alteration stop
                # counting toward `A_t`, and the obligation would reappear
                # after it had already been discharged.
                latch["switch_steps"] = latch.get("switch_steps", 0) + 1
                if latch["switch_steps"] >= self.n_switch:
                    latch = self._engage(ctx, heading_os_deg, u_os)
            else:
                latch["switch_steps"] = 0

        elif state == CLEARING:
            latch["clear_steps"] = latch.get("clear_steps", 0) + 1
            if latch["clear_steps"] >= self.n_clear:
                self._latch.pop(ctx.track_id, None)
                latch = None

        if latch is not None:
            self._latch[ctx.track_id] = latch
            ctx.state = latch["state"]
            ctx.psi_engage = latch["psi_engage"]
            ctx.u_engage = latch["u_engage"]
            ctx.t_engage = latch["t_engage"]
            ctx.compliant_turn_sense = latch["turn_sense"]
        else:
            ctx.state = IDLE
            ctx.compliant_turn_sense = compliant_turn_sense(ctx.cls)

        ctx.engaged = ctx.state == ENGAGED

    def _engage(self, ctx, heading_os_deg, u_os) -> dict:
        latch = {
            "state": ENGAGED,
            "cls": ctx.cls,
            "psi_engage": float(heading_os_deg),
            "u_engage": float(u_os),
            "t_engage": int(self.step_index),
            "turn_sense": compliant_turn_sense(ctx.cls),
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
