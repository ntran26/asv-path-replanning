"""The scenario generator (04a §3): one generator, three consumers.

The training distribution, the frozen evaluation suite and the sweeps all come
out of here, under one seed-namespace scheme (04a §9.2).  A second generator for
evaluation would be the single easiest way to make the two distributions differ
in a way nobody noticed.

Sampling order (04a §3.1):

1. encounter class
2. corridor geometry — width profile, bend, path offset
3. target kinematics — `CT`, speed ratio, desired `DCPA`, desired spawn `TCPA`
4. **solve backwards** for the spawn position
5. validate against class bands, containment and range rules; reject and resample
6. place static obstacles subject to non-interference
7. emit a scenario record

**Step 5 is not optional and step 3 is why.**  The sampled `CT` and the realised
relative bearing `α` are independent: a fraction of draws land in a different
encounter class than the one requested, particularly near band edges and on
bends.  Without the round-trip check the generator would quietly mislabel them,
and every per-class result in the paper would be computed over a contaminated
partition.

**Rejection accounting is a result, not a diagnostic** (04a §3.5).  The rejection
rate per `(class, width)` is an analytic feasibility measure: it says at what
width each encounter class stops being constructible *at all*, independently of
any policy.  Plotted against the Study 1 outcome curves it separates "the method
fails here" from "the geometry is infeasible here" — which is the strongest
available answer to "you designed the benchmark to produce the conclusion".
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import constants as cfg
import corridor as corr
import cpa_cri as cc
import encounter as enc
import targets as tgt
from ship import HULL_MARGIN, VESSEL_LENGTH, VESSEL_WIDTH, ShipModel


@dataclass
class Scenario:
    """One generated episode, serialisable to the 04a §9.1 record."""

    case_id: str
    encounter_class: str
    seed: int
    # corridor
    nominal_width: float = 0.0
    width_ratio: float = 1.0
    bend_deg: float = 0.0
    bend_requested_deg: float = 0.0
    path_offset_frac: float = 0.0
    # target
    target_spawn: Tuple[float, float] = (0.0, 0.0)
    target_heading: float = 0.0
    target_speed: float = 0.0
    target_behaviour: str = tgt.T_CV
    target_confined: bool = True
    speed_ratio: float = 1.0
    ct_deg: float = 0.0
    dcpa_m: float = 0.0
    tcpa_s: float = 0.0
    spawn_range_m: float = 0.0
    # own ship
    own_spawn: Tuple[float, float] = (0.0, 0.0)
    own_heading: float = 0.0
    # scene
    n_obstacles: int = 0
    obstacles: tuple = ()
    flags: Dict[str, object] = field(default_factory=dict)
    rejection_count: int = 0
    suite_version: str = cfg.SUITE_VERSION
    # geometry mode (06 §6)
    geometry_mode: str = "channel"
    slant_requested_deg: float = 0.0
    slant_realised_deg: float = 0.0
    path_midpoint: Tuple[float, float] = (0.0, 0.0)
    clearance_profile: Dict[str, list] = field(default_factory=dict)
    w_eff_at_cpa: float = 0.0
    field_replicable: bool = True

    def to_record(self) -> dict:
        """Canonical JSON-ready dict with sorted keys (04a §9.1)."""
        record = asdict(self)
        record["obstacles"] = [[list(map(float, p)) for p in poly]
                               for poly in self.obstacles]
        return record

    def digest(self) -> str:
        """SHA-256 over the canonical serialisation, for the freeze manifest."""
        blob = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class RejectionLog:
    """Rejections by `(class, width_bucket, reason)` — 04a §3.5's result."""

    counts: Counter = field(default_factory=Counter)
    attempts: Counter = field(default_factory=Counter)
    capped: Counter = field(default_factory=Counter)

    def record(self, encounter_class: str, width: float, reason: str) -> None:
        self.counts[(encounter_class, _width_bucket(width), reason)] += 1

    def attempt(self, encounter_class: str, width: float) -> None:
        self.attempts[(encounter_class, _width_bucket(width))] += 1

    def cap_out(self, encounter_class: str, width: float) -> None:
        self.capped[(encounter_class, _width_bucket(width))] += 1

    def rate(self, encounter_class: str = None) -> float:
        rejects = sum(n for (c, _w, _r), n in self.counts.items()
                      if encounter_class in (None, c))
        tries = sum(n for (c, _w), n in self.attempts.items()
                    if encounter_class in (None, c))
        return 0.0 if tries == 0 else rejects / tries

    def summary(self) -> dict:
        return {
            "by_class_width_reason": {f"{c}|{w}|{r}": n
                                      for (c, w, r), n in sorted(self.counts.items())},
            "attempts": {f"{c}|{w}": n for (c, w), n in sorted(self.attempts.items())},
            "cap_outs": {f"{c}|{w}": n for (c, w), n in sorted(self.capped.items())},
            "overall_rate": self.rate(),
        }


def _width_bucket(width) -> str:
    """04a §4.3's three strata, cut at the §1.1 derived thresholds.

    06: a basin draw has no single width, so the generator passes the string
    "basin" and the ledger keys it as its own stratum.
    """
    if isinstance(width, str):
        return width
    w_cross, w_over = width_thresholds()["crossing"], width_thresholds()["overtaking"]
    if width >= w_cross:
        return "wide"
    if width >= w_over:
        return "intermediate"
    return "narrow"


# ---------------------------------------------------------------------------
# Width thresholds  (04a §1.1)
# ---------------------------------------------------------------------------
def width_thresholds(a_abeam: float = None, w_wall: float = None,
                     u_nom: float = None, k: float = 0.5) -> dict:
    """The three predicted governing-rule transitions, as formulae.

    **04a §1.1 supersedes 02a §2.2 for crossing**, and by 0.8 m.  02a assumes the
    own ship has `W/2` of starboard room; 04a observes that a vessel stationed at
    the starboard quarter-width under Rule 9(a) has only `W/4`, so

        W_crossing >= 4 * (a_abeam + w_wall)

    That is 02a's own N2 insight carried one step further: a vessel already
    keeping starboard has *spent* its starboard room before the Rule 14
    alteration becomes tight.  Both derivations are reported by
    `constants.predicted_thresholds()`; 04a §11 leaves reconciling them as an
    open item owned by 02, and it is not this module's to close.

    The overtaking margin is an **exposure** term: the abreast configuration is
    held for `t_exp = (a_ahead + a_astern + Lpp)/(U_OS - U_TS)`, over which pose
    error accumulates at `rho_pose`.  It therefore shrinks as the operating speed
    rises, which is one more quantity F24 moves.
    """
    a = cfg.DOMAIN_LATERAL if a_abeam is None else float(a_abeam)
    w = cfg.W_WALL if w_wall is None else float(w_wall)
    u = cfg.U_NOM if u_nom is None else float(u_nom)

    head_on = 2.0 * a + 2.0 * w
    crossing = 4.0 * (a + w)
    closing = max((1.0 - float(k)) * u, 1e-6)
    t_exp = (cfg.DOMAIN_FORE + cfg.DOMAIN_AFT + cfg.LBP) / closing
    overtaking = head_on + 2.0 * cfg.RHO_POSE_DRIFT * t_exp

    return {"head_on": head_on, "overtaking": overtaking, "crossing": crossing,
            "t_exp_s": t_exp}


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------
class ScenarioGenerator:
    """Samples scenarios, validates them, and keeps the rejection ledger."""

    def __init__(self, *, stage: int = 5, seed_namespace: str = "training",
                 rejections: Optional[RejectionLog] = None) -> None:
        self.stage = int(stage)
        self.seed_namespace = str(seed_namespace)
        self.rejections = rejections if rejections is not None else RejectionLog()
        self._bend_debt = 0.0

    # ------------------------------------------------------------------
    def sample(self, seed: int, *, case_id: str = "",
               encounter_class: Optional[str] = None,
               width: Optional[float] = None,
               behaviour: str = tgt.T_CV,
               flags: Optional[dict] = None,
               geometry_mode: Optional[str] = None) -> Optional[Scenario]:
        """Generate one scenario, or `None` if it capped out (04a §3.5).

        `geometry_mode` forces "basin" or "channel"; otherwise the stage's
        `p_basin` decides (06 M-1, §4).  A forced `width` implies a channel.
        """
        rng = np.random.default_rng(int(seed))
        stage = cfg.CURRICULUM_STAGES[self.stage]
        cls = encounter_class or self._sample_class(rng, stage)
        # A22: drawn once per sample, not per attempt -- per attempt, rejection
        # would shrink the realised unescapable share from 20 % to about 8 %.
        self._want_unescapable = (cls == "crossing"
                                  and bool(rng.uniform() < cfg.CROSSING_UNESCAPABLE_FRAC))

        # **The corridor is sampled once.**  04a §3.1 puts the geometry at step 2
        # and the target kinematics at step 3, so a rejected draw rejects the
        # *kinematics*, not the channel.  Resampling the corridor per attempt
        # would also bias the width distribution toward whatever widths happen
        # to admit a target easily -- which is exactly the contamination the
        # rejection ledger exists to measure rather than to hide.
        flags = dict(flags or {})
        channel = self._sample_geometry(rng, stage, width, cls, geometry_mode, flags)
        key = _ledger_key(channel)

        for attempt in range(cfg.GENERATOR_MAX_ATTEMPTS):
            self.rejections.attempt(cls, key)
            scenario = self._attempt(rng, cls, channel, case_id, int(seed),
                                     behaviour, flags, attempt)
            if scenario is not None:
                scenario.rejection_count = attempt
                return scenario

        self.rejections.cap_out(cls, key)
        return None

    def _sample_geometry(self, rng, stage, width, cls, geometry_mode, flags=None):
        """Basin or channel for this episode (06 §4).

        Basin draws Paper 2's leg.  Channel draws keep 03a's corridor, with the
        narrow stratum reserved for the classes whose rules the width decides
        (06 M-6): null and being-overtaken start at the narrow edge.
        """
        mode = geometry_mode
        if mode is None:
            p_basin = (float(stage.get("p_basin", 1.0)) if cls in cfg.CHANNEL_CLASSES
                       else 1.0)
            mode = ("channel" if width else
                    "basin" if rng.uniform() < p_basin else "channel")
        flags = flags or {}
        if mode == "basin":
            # F75: a named basin case fixes its leg (06 §5.2).
            if flags.get("basin_leg"):
                start, goal = flags["basin_leg"]
                return corr.build_basin(tuple(start), tuple(goal))
            return corr.sample_basin(rng, slant_max_deg=stage.get("slant_max"))
        if not width:
            lo, hi = stage["width"]
            lo = max(float(lo), float(cfg.CHANNEL_MIN_WIDTH_BY_CLASS.get(cls, 0.0)))
            stage = dict(stage, width=(min(lo, hi), hi))
        channel = self._sample_corridor(rng, stage, width)
        if "offset" in flags:
            # F75: the named offset cases (A-OFF-*) set the Rule 9(a) station.
            channel.offset_frac = float(flags["offset"])
        return channel

    # ------------------------------------------------------------------
    def _sample_class(self, rng, stage) -> str:
        allowed = stage["classes"]
        # A31: a stage may weight its classes itself (stage 3 draws crossings as
        # often as head-ons); otherwise the global shares apply.
        table = stage.get("weights") or cfg.CLASS_SAMPLE_WEIGHTS
        weights = np.array([table[c] for c in allowed], dtype=float)
        return str(rng.choice(allowed, p=weights / weights.sum()))

    def _sample_corridor(self, rng, stage, width):
        """Corridor for this episode, honouring the stage and the bend quota.

        The bend quota is tracked as a running debt rather than drawn
        independently, because 04a asks for a *fraction* of episodes to carry a
        bend and an independent Bernoulli would meet that only in expectation --
        and F25 then clamps an unpredictable share of them to zero, so the
        realised fraction would fall below the requirement without anything
        reporting that it had.
        """
        want_bend = stage["bend"] and self._bend_debt >= 0.0
        channel = corr.sample(
            rng,
            width_range=(width, width) if width else stage["width"],
            vary_width=stage["vary"],
            allow_bend=stage["bend"],
            force_bend=want_bend,
        )
        self._bend_debt += (cfg.CORRIDOR_BEND_FRACTION - (1.0 if channel.has_bend else 0.0))
        return channel

    # ------------------------------------------------------------------
    def _attempt(self, rng, cls, channel, case_id, seed, behaviour, flags,
                 attempt) -> Optional[Scenario]:
        width = _ledger_key(channel)
        start_s = own_start_s(cls, channel)
        path_points = channel.reference_path_points(start_s=start_s)
        if len(path_points) < 2:
            self.rejections.record(cls, width, "degenerate_path")
            return None

        own = path_points[0].astype(float)
        tangent = path_points[1] - path_points[0]
        own_heading = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))

        scenario = Scenario(
            case_id=case_id or f"gen-{seed}", encounter_class=cls, seed=seed,
            nominal_width=channel.nominal_width, width_ratio=channel.width_ratio,
            bend_deg=channel.bend_deg,
            bend_requested_deg=channel.bend_requested_deg,
            path_offset_frac=channel.offset_frac,
            own_spawn=(float(own[0]), float(own[1])), own_heading=own_heading,
            target_behaviour=behaviour, flags=dict(flags),
        )
        # The corridor itself, for the environment.  An attribute, not a field,
        # so the 04a §9.1 record and its hash are unchanged.
        scenario.channel = channel
        _record_geometry(scenario, channel, start_s)

        if cls == "no_target":
            scenario.n_obstacles = self._sample_obstacle_count(rng)
            return scenario

        solved = (self._place_null(rng, own, own_heading) if cls == "null"
                  else self._backward_solve(rng, cls, own, own_heading, channel, flags))
        if solved is None:
            self.rejections.record(cls, width, "spawn_unsolvable")
            return None

        # --- the round-trip check (04a §3.3) ------------------------------
        # 04a's "null" is the classifier's `none`: a target present, tracked and
        # observed, but in no encounter geometry.  Two vocabularies for one
        # state, so the check maps between them rather than comparing strings --
        # which it was doing, and which rejected every null scenario ever drawn.
        expected = enc.NONE if cls == "null" else cls
        realised = self._classify(own, own_heading, solved)
        if realised != expected:
            self.rejections.record(cls, width, f"class_mismatch_{realised}")
            return None

        if not self._containment_ok(cls, solved, channel):
            self.rejections.record(cls, width, "containment")
            return None

        escapable = None
        if cls == "crossing":
            escapable = crossing_escape_feasible(own, own_heading, solved, channel)
            if escapable == bool(getattr(self, "_want_unescapable", False)):
                self.rejections.record(cls, width, "escape_label_mismatch")
                return None

        scenario.target_spawn = (solved["x"], solved["y"])
        scenario.target_heading = solved["heading"]
        scenario.target_speed = solved["speed"]
        scenario.target_confined = tgt.is_confined(cls)
        scenario.speed_ratio = solved["k"]
        scenario.ct_deg = solved["ct"]
        scenario.dcpa_m = solved["dcpa"]
        scenario.tcpa_s = solved["tcpa"]
        scenario.spawn_range_m = solved["range"]
        # A15's label.  An attribute, like `channel`, so the record hash is unchanged.
        scenario.dcpa_below_floor = solved.get("below_floor")
        scenario.dcpa_floor_m = solved.get("floor")
        # A22's label: can a lawful escape clear this crossing?  None otherwise.
        scenario.crossing_escapable = escapable
        scenario.n_obstacles = self._sample_obstacle_count(rng)
        # 06 §3.3: the clear width where the encounter happens.
        scenario.w_eff_at_cpa = float(channel.width_at_s(
            start_s + cfg.U_NOM * max(float(solved.get("tcpa", 0.0)), 0.0)))
        return scenario

    # ------------------------------------------------------------------
    def _backward_solve(self, rng, cls, own, own_heading, channel,
                        flags: Optional[dict] = None) -> Optional[dict]:
        """04a §3.3: solve for the spawn that produces the sampled `(DCPA, TCPA)`.

        ```
        psi_TS    = psi_OS + CT
        v_rel     = U_OS * h_OS - U_TS * h_TS,   V = |v_rel|
        n_hat     = sigma * perp(v_rel) / V
        p_TS(T_0) = p_OS(0) + U_OS * T_0 * h_OS + d_0 * n_hat
        p_TS(0)   = p_TS(T_0) - U_TS * T_0 * h_TS
        ```

        **Sampling `DCPA` explicitly is what 02a §11.1 calls a blocking
        hand-off.**  Solving backwards without it puts the target on the own
        ship's projected track in every episode, so `Dy_req = d_req` always,
        `A_req = 1` always, and `v_r8`'s zero branch — the whole point of the
        02 §3.2 head-on rationale — never fires.  The agent would learn "always
        alter" rather than "when to alter", and the M5 ablation could not tell
        the two apart.
        """
        flags = flags or {}
        # F75: named cases fix the crossing side and the speed ratio they name.
        port_share = (cfg.CROSSING_PORT_SHARE_TRAINING
                      if self.seed_namespace == "training" else None)
        ct = _sample_ct(rng, cls, side=flags.get("side"), port_share=port_share)
        k = (float(flags["speed_ratio"]) if flags.get("speed_ratio") is not None
             else float(rng.uniform(*cfg.CLASS_SPEED_RATIO[cls])))
        u_os, u_ts = cfg.U_NOM, k * cfg.U_NOM

        tcpa_lo, tcpa_hi = cfg.class_tcpa_range(cls)
        t0 = float(rng.uniform(tcpa_lo, tcpa_hi)) if tcpa_hi > 0 else 0.0

        dcpa_max = cfg.CLASS_DCPA_MAX[cls]
        below_floor = None
        if dcpa_max is None:
            d0 = float(rng.uniform(cfg.NULL_MIN_DCPA, cfg.NULL_MIN_DCPA + 3.0))
        elif cls == "being_overtaken":
            # A15, as amended by A21: floored at hull clearance for this draw's
            # angle and speed ratio, with a labelled Rule 17(b) fraction below.
            floor = contact_free_dcpa(ct, k) + float(cfg.BEING_OVERTAKEN_FLOOR_MARGIN)
            upper = max(float(dcpa_max), floor + float(cfg.BEING_OVERTAKEN_ABOVE_FLOOR_SPAN))
            below_floor = bool(rng.uniform() < cfg.BEING_OVERTAKEN_BELOW_FLOOR_FRAC)
            d0 = float(rng.uniform(0.0, floor) if below_floor
                       else rng.uniform(floor, upper))
        else:
            d0 = float(rng.uniform(0.0, dcpa_max))
        side = float(rng.choice([-1.0, 1.0]))

        psi_ts = (own_heading + ct) % 360.0
        h_os = _unit(own_heading)
        h_ts = _unit(psi_ts)

        v_rel = u_os * h_os - u_ts * h_ts
        speed_rel = float(np.linalg.norm(v_rel))
        if speed_rel < 1e-6:
            return None
        n_hat = side * np.array([v_rel[1], -v_rel[0]]) / speed_rel

        at_cpa = own + u_os * t0 * h_os + d0 * n_hat
        spawn = at_cpa - u_ts * t0 * h_ts
        r0 = float(np.linalg.norm(spawn - own))

        lo, hi = cfg.class_spawn_range(cls, k)
        if not (lo <= r0 <= hi):
            return None

        return {"x": float(spawn[0]), "y": float(spawn[1]), "heading": psi_ts,
                "speed": u_ts, "k": k, "ct": ct, "dcpa": d0, "tcpa": t0,
                "range": r0, "below_floor": below_floor,
                "floor": floor if cls == "being_overtaken" else None}

    # ------------------------------------------------------------------
    def _place_null(self, rng, own, own_heading) -> Optional[dict]:
        """The null class is **placed**, not solved backwards.

        **F26 -- 04a §3.4's null row is internally inconsistent.**  It asks for
        no CPA in the horizon *and* a spawn range of 8-15 m.  With no CPA there
        is no `T_0` to solve backwards from, so the backward solve degenerates to
        `R_0 = DCPA`, which 04a caps at 4-7 m: the two windows are disjoint and
        the class could never be constructed at all.  Every draw capped out at
        200 attempts.

        The resolution is that "no CPA in the horizon" *is* the specification:
        a target on a similar course at a similar speed, placed at range, whose
        projected CPA simply never arrives.  So place it at a sampled range and
        near-parallel course, then **verify** the CPA is absent rather than
        solving for one.

        The class is mandatory and this is why: a similar-course target never
        emerges from a class-conditional spawner but is common in practice, and
        it is the case where a policy that has learned "target present =>
        manoeuvre" will visibly overreact.
        """
        ct = _sample_ct(rng, "null")
        k = float(rng.uniform(*cfg.CLASS_SPEED_RATIO["null"]))
        lo, hi = cfg.class_spawn_range("null")
        r0 = float(rng.uniform(lo, hi))

        # Anywhere but dead ahead or dead astern, so it is a genuine neighbour
        # rather than a degenerate head-on or overtaking geometry.
        bearing = float(rng.uniform(20.0, 160.0)) * float(rng.choice([-1.0, 1.0]))
        psi_ts = (own_heading + ct) % 360.0
        direction = _unit(own_heading + bearing)
        spawn = np.asarray(own, dtype=float) + r0 * direction

        u_os, u_ts = cfg.U_NOM, k * cfg.U_NOM
        dcpa, tcpa = cc.cpa(tuple(own), u_os * _unit(own_heading),
                            tuple(spawn), u_ts * _unit(psi_ts))

        # The defining property: either the CPA is beyond the horizon, or it is
        # comfortably wide.  Either way the correct behaviour is to hold course.
        horizon_s = cfg.MAX_EPISODE_STEPS * cfg.UPDATE_RATE
        if dcpa < cfg.NULL_MIN_DCPA and 0.0 < tcpa < horizon_s:
            return None

        return {"x": float(spawn[0]), "y": float(spawn[1]), "heading": psi_ts,
                "speed": u_ts, "k": k, "ct": ct, "dcpa": float(dcpa),
                "tcpa": float(tcpa), "range": r0}

    # ------------------------------------------------------------------
    def _classify(self, own, own_heading, solved) -> str:
        """Forward-recompute the class from the realised state (04a §3.3)."""
        return enc.classify(
            (float(own[0]), float(own[1])), own_heading, cfg.U_NOM,
            (solved["x"], solved["y"]), solved["heading"], solved["speed"])

    def _containment_ok(self, cls, solved, channel) -> bool:
        """Confined classes spawn inside; crossing spawns outside (03a §5.2)."""
        import boundary_raycast as br
        # 06 §3.5: in basin mode confined traffic keeps the path band, and a
        # crossing target is anything that starts off it.  In a channel the
        # band is the channel.
        band = confinement_geometry(cls, channel)
        poly = band.polygon()
        inside = br.point_in_polygon(solved["x"], solved["y"], poly)
        if tgt.is_confined(cls):
            # A21: the whole hull, along its whole track to CPA.  A point check
            # let 75 % of being-overtaken targets breach the channel on the first
            # step, and the clamp then re-drew the encounter (F60).
            return bool(inside) and self._track_inside(cls, solved, band, poly)
        # A crossing target under Rule 9(d) is not a channel user: it must start
        # in open basin water outside the corridor, or it is not crossing.
        in_basin = (0.0 <= solved["x"] <= cfg.MAP_WIDTH
                    and 0.0 <= solved["y"] <= cfg.MAP_HEIGHT)
        # **Revision 7: the corridor is the basin**, so there is no water outside
        # it and that rule would reject every crossing draw.  A crossing target
        # then starts anywhere in the basin and stays unconfined -- it crosses
        # the fairway rather than entering it from outside.
        # Basin mode: the navigable polygon is the whole basin, so a crossing
        # target starts anywhere in it -- and, 06 T15, its hull stays in the
        # water up to CPA rather than crossing through a wall to get there.
        if channel.mode == "basin":
            envelope = br.rectangle(cfg.MAP_WIDTH, cfg.MAP_HEIGHT)
            return bool(in_basin and self._track_inside(cls, solved, channel, envelope))
        spans_basin = channel.nominal_width >= cfg.MAP_WIDTH - 1e-6
        return bool(in_basin and (spans_basin or not inside))

    @staticmethod
    def _track_inside(cls, solved, channel, poly) -> bool:
        horizon = (float(cfg.NULL_TRACK_CHECK_S) if cls == "null"
                   else max(float(solved.get("tcpa", 0.0)), 0.0))
        heading = float(solved["heading"])
        v = float(solved["speed"]) * _unit(heading)
        for t in np.arange(0.0, horizon + 1e-9, float(cfg.CONFINED_TRACK_CHECK_DT_S)):
            probe = tgt.Target(float(solved["x"] + v[0] * t), float(solved["y"] + v[1] * t),
                               heading, float(solved["speed"]), confined=True)
            if tgt.confinement_violation(probe, channel, poly) is not None:
                return False
        return True

    def _sample_obstacle_count(self, rng) -> int:
        lo, hi = cfg.CURRICULUM_STAGES[self.stage]["clutter"]
        return int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def own_start_s(encounter_class: str, channel) -> float:
    """How far along the corridor the own ship starts, metres.

    **F27 -- being-overtaken does not fit the corridor as specified.**  04a §3.4
    puts the target 4-6 m astern of the own ship and "inside" the channel, while
    §3.2 sets the corridor at 25 m and the reference path at 20 m.  Starting the
    own ship at the corridor mouth leaves 5 m astern, so a 6 m spawn is outside
    the water and every draw was rejected on containment.  20 + 6 = 26 > 25.

    The own ship therefore starts **`astern_room` metres along** for this class,
    which costs nothing — the reference path still has its full 20 m — and the
    spawn window is capped at what the channel actually provides.  The class is
    the one 04a §1.4 already identifies as most likely to fail in the field for
    perception rather than policy reasons, so losing it to a metre of geometry
    would be an expensive accident.
    """
    if getattr(channel, "mode", "channel") == "basin":
        # 06: the leg starts at Paper 2's start y, already clear of the wall.
        # Being overtaken starts further along it, for water astern.
        if encounter_class != "being_overtaken":
            return 0.0
        return float(min(cfg.BASIN_BEING_OVERTAKEN_START_S, 0.5 * channel.length))
    # F35: never closer to the corridor's end edge than `SPAWN_INSET_M`, or the
    # stern starts inside the boundary penalty band.
    if encounter_class != "being_overtaken":
        return float(cfg.SPAWN_INSET_M)
    # F49: the far end needs an inset too.  Taking all of `length - 20` as
    # astern room put the path end -- and so the goal -- on the corridor's end
    # edge: the bow crossed it inside the step that reached the goal, and 34 %
    # of being-overtaken training episodes scored a boundary collision there.
    # `GOAL_END_INSET_M` rather than `SPAWN_INSET_M`: the full spawn inset
    # leaves too little astern room and cut the class's yield from 200 to 129
    # in 200 draws.
    spare = max(0.0, channel.length - cfg.REF_PATH_LENGTH_M - cfg.GOAL_END_INSET_M)
    return float(max(cfg.SPAWN_INSET_M,
                     min(spare, cfg.class_spawn_range("being_overtaken")[1])))


def confinement_geometry(encounter_class: str, channel):
    """The water a confined target keeps to (06 §3.5, as amended in F74).

    Channel mode: the channel.  Basin mode: **the whole navigable basin**, which
    at 10 m is already narrow water -- except head-on traffic, which keeps the
    path band, because Rule 9(a) with Rule 14 is about each vessel keeping to
    its own side of the fairway the leg defines.  06 §3.5 banded every confined
    class; on a 2-3 m band null traffic could not be placed at all (37 of 40
    draws capped out) and overtaking traffic had nowhere to pass.
    """
    if getattr(channel, "mode", "channel") == "basin" and str(encounter_class) != "head_on":
        return channel
    return channel.band()


def _ledger_key(channel):
    """Rejection-ledger stratum: a width for a channel, "basin" for a basin."""
    return "basin" if getattr(channel, "mode", "channel") == "basin" else channel.nominal_width


def _record_geometry(scenario: Scenario, channel, start_s: float) -> None:
    """06 §6's record fields, and 06 M-8's field-replicable flag."""
    scenario.geometry_mode = getattr(channel, "mode", "channel")
    if scenario.geometry_mode == "basin":
        scenario.slant_requested_deg = float(channel.slant_requested_deg)
        scenario.slant_realised_deg = float(channel.slant_realised_deg)
        mid = 0.5 * (channel.centre[0] + channel.centre[-1])
        scenario.path_midpoint = (float(mid[0]), float(mid[1]))
        scenario.clearance_profile = channel.clearance_profile()
        scenario.field_replicable = True
    else:
        scenario.field_replicable = bool(channel.fits_basin(0.0))
    scenario.w_eff_at_cpa = float(channel.width_at_s(start_s))


def _unit(heading_deg: float) -> np.ndarray:
    a = math.radians(float(heading_deg))
    return np.array([math.sin(a), math.cos(a)])


@lru_cache(maxsize=4096)
def _contact_free_dcpa_cached(ct_rounded: float, k_rounded: float) -> float:
    own_half_l = 0.5 * (VESSEL_LENGTH + 2.0 * HULL_MARGIN)
    own_half_w = 0.5 * (VESSEL_WIDTH + 2.0 * HULL_MARGIN)
    own = [(own_half_w, own_half_l), (-own_half_w, own_half_l),
           (-own_half_w, -own_half_l), (own_half_w, -own_half_l)]
    h_ts = _unit(ct_rounded)
    v_rel = k_rounded * h_ts - np.array([0.0, 1.0])
    speed = float(np.linalg.norm(v_rel))
    if speed < 1e-9:
        return 0.0
    v = v_rel / speed
    n = np.array([v[1], -v[0]])
    reach = VESSEL_LENGTH + 2.0

    def touches(d: float) -> bool:
        for s in np.arange(-reach, reach, 0.02):
            p = d * n + s * v
            if _convex_overlap(own, tgt.hull_polygon(float(p[0]), float(p[1]), ct_rounded)):
                return True
        return False

    lo, hi = 0.0, 4.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if touches(mid):
            lo = mid
        else:
            hi = mid
    return hi


def contact_free_dcpa(ct_deg: float, k: float) -> float:
    """Smallest centre DCPA at which an overtaker's hull never touches the own
    ship's (collision hulls, margins included), for relative course `ct_deg`
    and speed ratio `k`.  A21.  Cached on 0.5 deg and 0.05 of speed ratio,
    rounded up in the conservative direction of the angle."""
    ct = ((float(ct_deg) + 180.0) % 360.0) - 180.0
    ct_r = math.copysign(math.ceil(abs(ct) * 2.0) / 2.0, ct)
    k_r = round(float(k) * 20.0) / 20.0
    return float(_contact_free_dcpa_cached(ct_r, k_r))


def _convex_overlap(poly_a, poly_b) -> bool:
    """Separating-axis test for two convex polygons (as `env._overlaps`)."""
    for poly in (poly_a, poly_b):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            ax, ay = -(y2 - y1), x2 - x1
            a = [p[0] * ax + p[1] * ay for p in poly_a]
            b = [p[0] * ax + p[1] * ay for p in poly_b]
            if max(a) < min(b) or max(b) < min(a):
                return False
    return True


def crossing_escape_feasible(own, own_heading: float, solved: dict, channel) -> bool:
    """A22: does either lawful escape clear this crossing in the own ship's physics?

    The own ship starts where the environment starts it -- on the path, at
    `U_NOM`, nominal hull -- holds course for `CROSSING_ESCAPE_DELAY_S`, then
    either coasts to a stop or makes a committed `CROSSING_ESCAPE_TURN_DEG`
    alteration in the A17 compliant sense (starboard for a target from
    starboard, port from port).  The target is constant-velocity and
    unconfined, as a crossing target is.  An escape counts only if the hulls
    never touch and the own hull stays inside the corridor, up to
    `CROSSING_ESCAPE_TAIL_S` past the drawn TCPA.
    """
    import boundary_raycast as br
    full = channel.polygon()
    # The wall test runs every physics step; a straight corridor's outline at
    # every 8th station (0.4 m) is exact to well under a centimetre.  Basin
    # mode's `P_nav` has four corners and is used as it is.
    poly = (list(full[::8]) + ([full[-1]] if (len(full) - 1) % 8 else [])
            if len(full) > 16 else list(full))
    sense = -1.0 if float(solved["ct"]) < 180.0 else +1.0
    horizon = max(float(solved.get("tcpa", 0.0)), 0.0) + float(cfg.CROSSING_ESCAPE_TAIL_S)
    dt = float(cfg.CROSSING_ESCAPE_DT_S)
    half_l = 0.5 * (VESSEL_LENGTH + 2.0 * HULL_MARGIN)
    half_w = 0.5 * (VESSEL_WIDTH + 2.0 * HULL_MARGIN)
    t_heading = float(solved["heading"])
    t_vel = float(solved["speed"]) * _unit(t_heading)

    def own_hull(x, y, heading):
        a = math.radians(heading)
        s, c = math.sin(a), math.cos(a)
        return [(x + f * s - l * c, y + f * c + l * s)
                for f, l in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]

    for response in ("stop", "turn"):
        model = ShipModel()
        model._s[3, 0] = math.radians(float(own_heading))
        model._s[0, 0] = float(cfg.U_NOM)
        x, y, heading = float(own[0]), float(own[1]), float(own_heading)
        clear = True
        per_decision = max(1, int(round(float(cfg.UPDATE_RATE) / dt)))
        rpm, rudder, last_range = float(cfg.CRUISE_RPM), 0.0, float("inf")
        for i in range(int(math.ceil(horizon / dt))):
            t = (i + 1) * dt
            if i % per_decision == 0:
                # Commands at the environment's decision rate, held for the step.
                responding = t > float(cfg.CROSSING_ESCAPE_DELAY_S)
                want = float(own_heading)
                rpm = float(cfg.CRUISE_RPM)
                if responding and response == "turn":
                    want = float(own_heading) + sense * float(cfg.CROSSING_ESCAPE_TURN_DEG)
                if responding and response == "stop":
                    rpm = float(cfg.CROSSING_ESCAPE_STOP_RPM)
                error = ((want - heading) + 180.0) % 360.0 - 180.0
                rudder = float(np.clip(error / 20.0, -1.0, 1.0)) * 100.0
            dx, dy, heading, _ = model.update(rpm, rudder, dt)
            x += dx
            y += dy
            hull = own_hull(x, y, heading)
            tx = float(solved["x"]) + float(t_vel[0]) * t
            ty = float(solved["y"]) + float(t_vel[1]) * t
            if _convex_overlap(hull, tgt.hull_polygon(tx, ty, t_heading)):
                clear = False
                break
            rng_now = math.hypot(tx - x, ty - y)
            if t > float(solved.get("tcpa", 0.0)) and rng_now > 3.0 and rng_now > last_range:
                break                        # passed and opening: nothing left to hit
            last_range = rng_now
            xs = np.array([p[0] for p in hull])
            ys = np.array([p[1] for p in hull])
            if not np.all(br.points_in_polygon(xs, ys, poly)):
                clear = False
                break
        if clear:
            return True
    return False


def _sample_ct(rng, cls: str, side: Optional[str] = None,
               port_share: Optional[float] = None) -> float:
    if cls in cfg.CONFINED_CT_CLASSES:
        # A21: channel users run near-parallel to a straight fairway.
        half = float(cfg.CONFINED_CT_HALF_DEG)
        return float(rng.uniform(-half, half))
    band = cfg.CLASS_CT_DEG[cls]
    if cls == "crossing":
        # The first band (CT < 180) crosses from port, the second from starboard.
        # A27 option 2: training draws port at `port_share`; every other
        # namespace keeps the even integer draw, so its scenarios are unchanged.
        index = (int(rng.integers(0, 2)) if port_share is None
                 else (0 if rng.uniform() < float(port_share) else 1))
        if side in ("port", "starboard"):
            index = 0 if side == "port" else 1
        chosen = band[index]
        return float(rng.uniform(*chosen))
    return float(rng.uniform(*band))


def seed_for(namespace: str, index: int) -> int:
    """A seed inside a namespace, so the five uses can never collide (04a §9.2)."""
    lo, hi = cfg.SEED_NAMESPACES[namespace]
    span = hi - lo + 1
    return int(lo + (int(index) % span))


def namespaces_are_disjoint() -> bool:
    spans = sorted(cfg.SEED_NAMESPACES.values())
    return all(a[1] < b[0] for a, b in zip(spans, spans[1:]))
