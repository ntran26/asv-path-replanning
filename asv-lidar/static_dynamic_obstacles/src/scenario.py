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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import constants as cfg
import corridor as corr
import cpa_cri as cc
import encounter as enc
import targets as tgt


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


def _width_bucket(width: float) -> str:
    """04a §4.3's three strata, cut at the §1.1 derived thresholds."""
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
               flags: Optional[dict] = None) -> Optional[Scenario]:
        """Generate one scenario, or `None` if it capped out (04a §3.5)."""
        rng = np.random.default_rng(int(seed))
        stage = cfg.CURRICULUM_STAGES[self.stage]
        cls = encounter_class or self._sample_class(rng, stage)

        # **The corridor is sampled once.**  04a §3.1 puts the geometry at step 2
        # and the target kinematics at step 3, so a rejected draw rejects the
        # *kinematics*, not the channel.  Resampling the corridor per attempt
        # would also bias the width distribution toward whatever widths happen
        # to admit a target easily -- which is exactly the contamination the
        # rejection ledger exists to measure rather than to hide.
        channel = self._sample_corridor(rng, stage, width)

        for attempt in range(cfg.GENERATOR_MAX_ATTEMPTS):
            self.rejections.attempt(cls, channel.nominal_width)
            scenario = self._attempt(rng, cls, channel, case_id, int(seed),
                                     behaviour, flags or {}, attempt)
            if scenario is not None:
                scenario.rejection_count = attempt
                return scenario

        self.rejections.cap_out(cls, channel.nominal_width)
        return None

    # ------------------------------------------------------------------
    def _sample_class(self, rng, stage) -> str:
        allowed = stage["classes"]
        weights = np.array([cfg.CLASS_SAMPLE_WEIGHTS[c] for c in allowed],
                           dtype=float)
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
        width = channel.nominal_width
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
            nominal_width=width, width_ratio=channel.width_ratio,
            bend_deg=channel.bend_deg,
            bend_requested_deg=channel.bend_requested_deg,
            path_offset_frac=channel.offset_frac,
            own_spawn=(float(own[0]), float(own[1])), own_heading=own_heading,
            target_behaviour=behaviour, flags=dict(flags),
        )

        if cls == "no_target":
            scenario.n_obstacles = self._sample_obstacle_count(rng)
            return scenario

        solved = (self._place_null(rng, own, own_heading) if cls == "null"
                  else self._backward_solve(rng, cls, own, own_heading, channel))
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

        scenario.target_spawn = (solved["x"], solved["y"])
        scenario.target_heading = solved["heading"]
        scenario.target_speed = solved["speed"]
        scenario.target_confined = tgt.is_confined(cls)
        scenario.speed_ratio = solved["k"]
        scenario.ct_deg = solved["ct"]
        scenario.dcpa_m = solved["dcpa"]
        scenario.tcpa_s = solved["tcpa"]
        scenario.spawn_range_m = solved["range"]
        scenario.n_obstacles = self._sample_obstacle_count(rng)
        return scenario

    # ------------------------------------------------------------------
    def _backward_solve(self, rng, cls, own, own_heading, channel) -> Optional[dict]:
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
        ct = _sample_ct(rng, cls)
        k = float(rng.uniform(*cfg.CLASS_SPEED_RATIO[cls]))
        u_os, u_ts = cfg.U_NOM, k * cfg.U_NOM

        tcpa_lo, tcpa_hi = cfg.class_tcpa_range(cls)
        t0 = float(rng.uniform(tcpa_lo, tcpa_hi)) if tcpa_hi > 0 else 0.0

        dcpa_max = cfg.CLASS_DCPA_MAX[cls]
        d0 = (float(rng.uniform(cfg.NULL_MIN_DCPA, cfg.NULL_MIN_DCPA + 3.0))
              if dcpa_max is None else float(rng.uniform(0.0, dcpa_max)))
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
                "range": r0}

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
        poly = channel.polygon()
        inside = br.point_in_polygon(solved["x"], solved["y"], poly)
        if tgt.is_confined(cls):
            return bool(inside)
        # A crossing target under Rule 9(d) is not a channel user: it must start
        # in open basin water outside the corridor, or it is not crossing.
        in_basin = (0.0 <= solved["x"] <= cfg.MAP_WIDTH
                    and 0.0 <= solved["y"] <= cfg.MAP_HEIGHT)
        return bool(not inside and in_basin)

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
    if encounter_class != "being_overtaken":
        return 0.0
    spare = max(0.0, channel.length - cfg.REF_PATH_LENGTH_M)
    return float(min(spare, cfg.class_spawn_range("being_overtaken")[1]))


def _unit(heading_deg: float) -> np.ndarray:
    a = math.radians(float(heading_deg))
    return np.array([math.sin(a), math.cos(a)])


def _sample_ct(rng, cls: str) -> float:
    band = cfg.CLASS_CT_DEG[cls]
    if cls == "crossing":
        chosen = band[int(rng.integers(0, 2))]
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
