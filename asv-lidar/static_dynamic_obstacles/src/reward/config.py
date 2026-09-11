"""`RewardConfig`: the coefficients, and the assertions that fail at build time.

02a §9 is explicit about *when* these should fail: "fail at construction, not at
step 10,000".  A reward whose hierarchy is silently violated does not crash --
it trains, for a week, and produces a policy whose behaviour nobody can explain.

Defaults come from `constants.py` §13, which is the single source of truth.
Nothing here writes a number down that lives there; a config that duplicated the
values would be a second place for them to drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple

import constants as cfg


@dataclass(frozen=True)
class RewardConfig:
    """Weights, geometry and thresholds for one reward instance.

    Frozen, so an ablation is `replace(cfg, colregs_terms_enabled=False)` rather
    than a mutation that leaks between runs.
    """

    # --- weights (02a §7) --------------------------------------------------
    w_pf: float = cfg.W_PF
    w_prog: float = cfg.W_PROG
    w_exist: float = cfg.W_EXIST
    w_smooth: float = cfg.W_SMOOTH
    w_obs: float = cfg.W_OBS
    w_bnd: float = cfg.W_BND
    w_dom: float = cfg.W_DOM
    w_col: float = cfg.W_COL

    # --- terminal ----------------------------------------------------------
    r_goal: float = cfg.R_GOAL
    r_collision: float = cfg.R_COLLISION
    r_timeout: float = cfg.R_TIMEOUT
    timeout_bootstrap: bool = True

    # --- path following ----------------------------------------------------
    gamma_e: float = cfg.PF_GAMMA_E
    w_e: float = cfg.PF_W_E
    omega_la: float = cfg.PF_OMEGA_LA
    u_ref: float = cfg.U_REF
    u_ref_slow_factor: float = cfg.U_REF_SLOW_FACTOR

    # --- safety geometry ---------------------------------------------------
    d_safe: float = cfg.D_SAFE
    c_wall: float = cfg.HEAD_ON_WALL_CLEARANCE
    d_oa: float = cfg.D_OA
    d_cut: float = cfg.D_CUT
    obs_swath_deg: float = cfg.OBS_SWATH_HALF_DEG
    breadth: float = cfg.BREADTH

    # --- ship domain (provisional; final from 05) --------------------------
    dom_fore: float = cfg.DOMAIN_FORE
    dom_aft: float = cfg.DOMAIN_AFT
    dom_abeam: float = cfg.DOMAIN_LATERAL

    # --- progress ----------------------------------------------------------
    n_ref_prog: float = cfg.N_REF_PROG

    # --- smoothness --------------------------------------------------------
    kappa_delta: float = cfg.KAPPA_DELTA
    kappa_n: float = cfg.KAPPA_N
    w_n: float = cfg.SMOOTH_W_N
    sigma_enc: float = cfg.SIGMA_ENC
    n_free: int = cfg.N_FREE_STEPS

    # --- encounter state machine -------------------------------------------
    t_engage: float = cfg.T_ENGAGE
    kappa_eng: float = cfg.KAPPA_ENG
    kappa_rel: float = cfg.KAPPA_REL
    n_clear: int = cfg.N_CLEAR_STEPS
    n_switch: int = cfg.N_SWITCH_STEPS

    # --- COLREGs sub-weights -----------------------------------------------
    w_port: float = cfg.V_PORT_W
    w_bow: float = cfg.V_BOW_W
    w_side: float = cfg.V_SIDE_W
    w_hold: float = cfg.V_HOLD_W
    w_r8: float = cfg.V_R8_W

    # --- COLREGs thresholds ------------------------------------------------
    r_ref: float = cfg.R_REF
    r_dead: float = cfg.R_DEAD
    beta_bow_deg: float = cfg.BETA_BOW_DEG
    r_hold: float = cfg.R_HOLD
    du_hold: float = cfg.DU_HOLD
    t_extremis: float = cfg.T_EXTREMIS
    dpsi_min_deg: float = cfg.DPSI_MIN_DEG
    du_min: float = cfg.DU_MIN
    t_act: float = cfg.T_ACT

    # --- propulsion (03 §6 -- reverse unverified) --------------------------
    reverse_available: bool = cfg.REVERSE_AVAILABLE
    u_min_reachable: float = cfg.U_MIN_REACHABLE

    # --- open-water fallback (`R-10`) --------------------------------------
    open_water_mode: bool = False
    w_ref_open_water: float = cfg.W_REF_OPEN_WATER

    # --- ablation switches (00 §4.3) ---------------------------------------
    colregs_terms_enabled: bool = True
    encounter_feature_enabled: bool = True
    colregs_term_mask: Tuple[str, ...] = ("port", "bow", "side", "hold", "r8")
    dense_term_mask: Tuple[str, ...] = ("pf", "prog", "exist", "smooth",
                                        "obs", "bnd", "dom", "col")

    log_per_term: bool = True

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    @property
    def d_req(self) -> float:
        """Required separation for two identical vessels abeam."""
        return 2.0 * self.dom_abeam

    @property
    def max_encounter_steps(self) -> int:
        return int(cfg.MAX_ENCOUNTER_STEPS)

    def with_(self, **changes) -> "RewardConfig":
        """A modified copy.  Re-validates, so an ablation cannot smuggle in a
        combination the assertions would have rejected."""
        return replace(self, **changes)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """02a §9's assertions, plus 02b §3.1's floor.  Raises, deliberately."""
        # 02b §3.1.  The domain must not sit inside the sensor's blind zone:
        # `r_dom` is evaluated on ground truth per `R-1`, so a domain the
        # sensor cannot resolve would penalise the agent for intrusions it is
        # physically incapable of perceiving, and the term stops being a
        # shaping signal.
        if self.dom_abeam < cfg.DOMAIN_ABEAM_FLOOR:
            raise ValueError(
                f"d_abeam {self.dom_abeam:.3f} m is below the sensor-resolution "
                f"floor {cfg.DOMAIN_ABEAM_FLOOR:.3f} m "
                f"(= LIDAR_MIN_RANGE + B/2); r_dom would be unlearnable (02b §3.1)")

        # 02a §2.  Otherwise the geometry that *defines* a compliant
        # narrow-channel manoeuvre would itself trigger the boundary penalty --
        # the reward would punish the behaviour the paper exists to elicit.
        ceiling = self.c_wall - 0.5 * self.breadth
        if not self.d_safe < ceiling:
            raise ValueError(
                f"d_safe {self.d_safe:.3f} m must be below c_wall - B/2 = "
                f"{ceiling:.3f} m (02a §2 invariant)")

        # The 02a §7 hierarchy, as an assertion rather than as a table anyone
        # has to keep in their head.
        order = [("w_bnd", self.w_bnd), ("w_dom", self.w_dom),
                 ("w_obs", self.w_obs), ("w_col", self.w_col),
                 ("w_pf", self.w_pf), ("w_prog", self.w_prog),
                 ("w_smooth", self.w_smooth), ("w_exist", self.w_exist)]
        for (n_hi, hi), (n_lo, lo) in zip(order, order[1:]):
            if not hi > lo:
                raise ValueError(f"coefficient ordering violated: {n_hi}={hi} "
                                 f"must exceed {n_lo}={lo} (02a §7)")

        # 02 §5: a COLREGs-compliant collision must be worse than a maximally
        # non-compliant episode that avoids one.
        if not abs(self.r_collision) > self.w_col * self.max_encounter_steps:
            raise ValueError(
                f"|r_collision| {abs(self.r_collision)} must exceed "
                f"w_col * max_encounter_steps = "
                f"{self.w_col * self.max_encounter_steps} (02 §5, `R-7`)")

        if not self.t_act < self.t_engage:
            raise ValueError(f"t_act {self.t_act} must be below t_engage "
                             f"{self.t_engage}: the obligation cannot become "
                             f"urgent before the encounter has engaged")
        if not self.d_cut > self.d_oa:
            raise ValueError(f"d_cut {self.d_cut} must exceed d_oa {self.d_oa}")

        # Engagement must fire before the obligation does -- watch, then act.
        if not self.kappa_eng > 1.0:
            raise ValueError(f"kappa_eng {self.kappa_eng} must exceed 1.0, or "
                             f"engagement fires at the compliant separation "
                             f"itself (02a §6.1)")

        if self.kappa_delta is None or self.kappa_delta <= 0.0:
            raise ValueError("kappa_delta must come from the actuator rate "
                             "limit; None means 05 has not delivered it")
        if not self.kappa_rel > self.kappa_eng:
            raise ValueError(f"kappa_rel {self.kappa_rel} must exceed kappa_eng "
                             f"{self.kappa_eng}, or an encounter clears at the "
                             f"same range it engages and the state machine chatters")

        unknown = set(self.colregs_term_mask) - {"port", "bow", "side", "hold", "r8"}
        if unknown:
            raise ValueError(f"unknown COLREGs terms in the mask: {sorted(unknown)}")
        unknown = set(self.dense_term_mask) - {"pf", "prog", "exist", "smooth",
                                               "obs", "bnd", "dom", "col"}
        if unknown:
            raise ValueError(f"unknown dense terms in the mask: {sorted(unknown)}")


DEFAULT = RewardConfig()
