"""The weighted sum, the group clipping, and the per-step breakdown.

```
r_t =  w_pf*r_pf + w_prog*r_prog + w_exist*r_exist + w_smooth*r_smooth
     + w_obs*r_obs + w_bnd*r_bnd + w_dom*r_dom + w_COL*r_col + r_term
```

The object returned by one call carries every intermediate value, not just the
scalar.  That is not instrumentation added afterwards -- `02 §6` makes the scale
audit mandatory and `04 §7` reads its metric set off these keys, so a reward
that could only report its total would have to be rebuilt to be measured.
`RENDER_PANEL_SPEC` then makes the same struct a view rather than a second
computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import constants as cfg_mod
from reward import terms as T
from reward.audit import TermAudit
from reward.config import RewardConfig


@dataclass
class RewardBreakdown:
    """One step's reward, decomposed.  Every panel field is a read of this."""

    total: float = 0.0
    dense: float = 0.0
    terminal: float = 0.0

    term: Dict[str, float] = field(default_factory=dict)        # pre-weight
    weighted: Dict[str, float] = field(default_factory=dict)    # post-weight
    colregs: Dict[str, float] = field(default_factory=dict)     # sub-terms, pre-weight
    colregs_pre_clip: float = 0.0
    colregs_track: Optional[int] = None

    u_ref_eff: float = 0.0
    u_ref_rule: str = ""
    u_ref_reason: str = ""
    w_exist_scale: float = 1.0
    sigma_smooth: float = 1.0

    dominant: str = ""

    def as_info(self, prefix: str = "reward") -> Dict[str, float]:
        """Flatten to the `02a §10.3` logging keys."""
        out: Dict[str, float] = {}
        for name, value in self.term.items():
            out[f"{prefix}/term/{name}"] = float(value)
        for name, value in self.weighted.items():
            out[f"{prefix}/weighted/{name}"] = float(value)
        for name, value in self.colregs.items():
            out[f"colregs/v_{name}"] = float(value)
        out[f"{prefix}/total"] = float(self.total)
        out[f"{prefix}/dense"] = float(self.dense)
        out[f"{prefix}/terminal"] = float(self.terminal)
        out["colregs/pre_clip"] = float(self.colregs_pre_clip)
        out["colregs/u_ref_eff"] = float(self.u_ref_eff)
        return out


class RewardFunction:
    """The whole reward, as one callable, with its episode accumulators.

    Stateful only in the audit: the terms themselves are pure, so a leave-one-out
    ablation is a mask over `dense_term_mask` rather than an `if` in the middle
    of the step function.  Reset it when the episode does.
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.cfg = config if config is not None else RewardConfig()
        self.audit = TermAudit(self.cfg)
        self.reset()

    def reset(self) -> None:
        self.audit.reset()
        self.last: Optional[RewardBreakdown] = None

    # ------------------------------------------------------------------
    def __call__(self, state: T.RewardState, contexts, *,
                 collision: Optional[str] = None, reached_goal: bool = False,
                 truncated: bool = False, record: bool = True) -> RewardBreakdown:
        cfg = self.cfg
        speed = T.effective_speed_reference(state, contexts, cfg)
        group = T.colregs_group(state, contexts, cfg)

        raw = {
            "pf": T.r_pf(state, contexts, cfg, u_ref_eff=speed["u_ref_eff"]),
            "prog": T.r_prog(state, contexts, cfg),
            "exist": T.r_exist(state, contexts, cfg),
            "smooth": T.r_smooth(state, contexts, cfg),
            "obs": T.r_obs(state, contexts, cfg),
            "bnd": T.r_bnd(state, contexts, cfg),
            "dom": T.r_dom(state, contexts, cfg),
            "col": -group["v_col"],
        }
        # `R-5` suspends the existence cost while holding astern, through the
        # weight rather than the term, so `r_exist` keeps its declared range and
        # the ablation mask stays a mask.
        weights = {
            "pf": cfg.w_pf, "prog": cfg.w_prog,
            "exist": cfg.w_exist * speed["w_exist_scale"],
            "smooth": cfg.w_smooth, "obs": cfg.w_obs, "bnd": cfg.w_bnd,
            "dom": cfg.w_dom, "col": cfg.w_col,
        }

        weighted = {}
        for name, value in raw.items():
            active = name in cfg.dense_term_mask
            weighted[name] = weights[name] * value if active else 0.0
            if not active:
                raw[name] = 0.0

        dense = float(sum(weighted.values()))
        terminal = self._terminal(collision, reached_goal, truncated)

        out = RewardBreakdown(
            total=dense + terminal,
            dense=dense,
            terminal=terminal,
            term=raw,
            weighted=weighted,
            colregs=dict(group["parts"]),
            colregs_pre_clip=float(group["pre_clip"]),
            colregs_track=group["track_id"],
            u_ref_eff=float(speed["u_ref_eff"]),
            u_ref_rule=str(speed["rule"]),
            u_ref_reason=str(speed["reason"]),
            w_exist_scale=float(speed["w_exist_scale"]),
            sigma_smooth=T.smoothness_scale(state, contexts, cfg),
            dominant=max(weighted, key=lambda k: abs(weighted[k])) if weighted else "",
        )
        if record:
            self.audit.record(out)
        self.last = out
        return out

    # ------------------------------------------------------------------
    def _terminal(self, collision, reached_goal, truncated) -> float:
        """The one-shot payoffs (02a §5.7, 02b §2).

        Uniform across the three collision types: they are reported separately
        as metrics but not weighted separately, so the reward makes no claim
        about their relative severity that the paper would have to defend.

        `r_timeout = 0` is only sound if the environment truncates rather than
        terminating at the step limit, so SB3 bootstraps the value of the final
        state.  That is asserted in the environment's own tests, not here --
        a reward function cannot check how it is being called.
        """
        if collision is not None:
            return float(self.cfg.r_collision)
        total = 0.0
        if reached_goal:
            total += float(self.cfg.r_goal)
        if truncated:
            total += float(self.cfg.r_timeout)
        return float(total)
