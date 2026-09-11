"""Per-term accumulators and the Table R7 generator (02a §8).

`02 §6` makes the scale audit mandatory and `02a §8` pre-commits the numbers, so
that a mismatch is diagnostic rather than something to explain away afterwards.
This module is what makes that cheap: the accumulators run during every episode,
so the audit is a read of state that already exists.

**The range column is the point.**  A term whose episode range is `[-0.44,
-0.41]` varies by less than 10% of its own value: it is a constant offset
wearing a shaping term's costume, it carries no gradient, and it is invisible in
the instantaneous, weighted and integrated columns alike.  That is the exact
failure that cost Paper 2 the most, and `flat` below is the detector for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import constants as cfg_mod
from reward.terms import TERM_RANGE

# A term is "flat" when its realised episode range spans less than this fraction
# of its declared span.  5% is 02a §8's figure, and it is deliberately generous:
# a term at 6% is not healthy either, but at 5% it is certainly broken.
FLAT_FRACTION = 0.05

# Minimum steps before `flat` means anything.  A term cannot have shown its
# range in the first few steps of an episode, and flagging it there would put a
# permanent red marker on the panel for the first second of every run.
FLAT_MIN_STEPS = 30

# A term contributing less than this per step on average is inactive, not
# out of order.  Without the floor the check fires on every clean run --
# `r_exist` is a constant 0.05 and `r_smooth` is near zero whenever the helm is
# steady, so "exist outweighs smooth" would be permanently on screen.  A warning
# that is always showing is a warning nobody reads.
HIERARCHY_FLOOR = 0.01


@dataclass
class TermStats:
    """Running statistics for one term across an episode."""

    name: str
    total_weighted: float = 0.0
    total_raw: float = 0.0
    lo: float = float("inf")
    hi: float = float("-inf")
    steps: int = 0
    history: List[float] = field(default_factory=list)

    def update(self, raw: float, weighted: float, keep: int) -> None:
        self.total_raw += float(raw)
        self.total_weighted += float(weighted)
        self.lo = min(self.lo, float(raw))
        self.hi = max(self.hi, float(raw))
        self.steps += 1
        self.history.append(float(raw))
        if len(self.history) > keep:
            del self.history[:-keep]

    @property
    def span(self) -> float:
        return 0.0 if self.steps == 0 else self.hi - self.lo

    @property
    def flat(self) -> bool:
        """Realised range under 5% of the declared span (02a §8.1).

        **A term that is identically zero is inactive, not flat.**  `r_bnd` on
        an episode that never approached a wall should read `0.000` with no
        flag; flagging it would put a permanent warning on three terms in every
        clean run and train the reader to ignore the column that matters.  What
        `flat` is for is a term sitting at a *non-zero* constant -- the offset
        wearing a shaping term's costume.

        `r_exist` is the deliberate exception and reads `[flat]` always, which
        is correct: it is a constant by construction, and the panel mockup in
        `RENDER_PANEL_SPEC` shows it flagged.
        """
        if self.steps < FLAT_MIN_STEPS:
            return False
        if self.lo == 0.0 and self.hi == 0.0:
            return False
        lo, hi = TERM_RANGE.get(self.name, (-1.0, 0.0))
        declared = hi - lo
        return declared > 0.0 and self.span < FLAT_FRACTION * declared

    @property
    def mean_abs_weighted(self) -> float:
        if self.steps == 0:
            return 0.0
        return abs(self.total_weighted) / self.steps


class TermAudit:
    """Episode accumulators for every term, plus the live hierarchy check.

    `keep` bounds the per-term history so a long run cannot grow without limit;
    100 steps is what the panel's sparklines need and nothing reads further back.
    """

    def __init__(self, cfg, *, keep: int = 100) -> None:
        self.cfg = cfg
        self.keep = int(keep)
        self.reset()

    def reset(self) -> None:
        self.stats: Dict[str, TermStats] = {}
        self.steps = 0
        self.episode_total = 0.0
        self.terminal_total = 0.0
        self.colregs_steps = 0
        self.engaged_steps = 0

    # ------------------------------------------------------------------
    def record(self, breakdown) -> None:
        self.steps += 1
        self.episode_total += float(breakdown.total)
        self.terminal_total += float(breakdown.terminal)
        if breakdown.term.get("col", 0.0) < 0.0:
            self.colregs_steps += 1
        for name, raw in breakdown.term.items():
            stat = self.stats.setdefault(name, TermStats(name))
            stat.update(raw, breakdown.weighted.get(name, 0.0), self.keep)

    # ------------------------------------------------------------------
    def episode_dominant(self) -> str:
        """The term with the largest absolute episode integral."""
        if not self.stats:
            return ""
        return max(self.stats, key=lambda n: abs(self.stats[n].total_weighted))

    def hierarchy_violations(self) -> List[str]:
        """Where the *realised* per-step magnitudes invert the §7 ordering.

        **This is allowed to fire, and that is the point.**  `02 §5` and `02 §6`
        ask for different things and can conflict, because terms have very
        different natural durations -- a boundary excursion lasts seconds, a
        COLREGs violation lasts a whole encounter.  A rarely-active term
        *should* integrate small.  What this catches is a term running 10x off
        its intended share, which is a coefficient error rather than a duty
        cycle.  Cheap insurance that the hierarchy survives contact; not a test
        to make green.
        """
        order = ["bnd", "dom", "obs", "col", "pf", "prog", "smooth", "exist"]
        present = [n for n in order
                   if n in self.stats and self.stats[n].steps
                   and self.stats[n].mean_abs_weighted > HIERARCHY_FLOOR]
        problems = []
        for hi, lo in zip(present, present[1:]):
            a = self.stats[hi].mean_abs_weighted
            b = self.stats[lo].mean_abs_weighted
            if b > a:
                problems.append(f"{lo} ({b:.3f}) outweighs {hi} ({a:.3f})")
        return problems

    # ------------------------------------------------------------------
    def rows(self) -> List[dict]:
        """One row per term: the four columns the panel and Table R7 both use."""
        out = []
        for name in ("pf", "prog", "exist", "smooth", "obs", "bnd", "dom", "col"):
            stat = self.stats.get(name)
            if stat is None:
                out.append({"name": name, "sum": 0.0, "lo": 0.0, "hi": 0.0,
                            "flat": False, "steps": 0})
                continue
            out.append({
                "name": name,
                "sum": stat.total_weighted,
                "lo": 0.0 if stat.steps == 0 else stat.lo,
                "hi": 0.0 if stat.steps == 0 else stat.hi,
                "flat": stat.flat,
                "steps": stat.steps,
            })
        return out

    def summary(self) -> dict:
        """The episode record `metrics.py` writes and Table R7 aggregates."""
        return {
            "steps": self.steps,
            "episode_return": self.episode_total,
            "terminal": self.terminal_total,
            "dominant": self.episode_dominant(),
            "colregs_active_steps": self.colregs_steps,
            "hierarchy_violations": self.hierarchy_violations(),
            **{f"sum_{row['name']}": row["sum"] for row in self.rows()},
            **{f"flat_{row['name']}": row["flat"] for row in self.rows()},
        }


# ---------------------------------------------------------------------------
# Table R7
# ---------------------------------------------------------------------------
# 02a §8.1's pre-committed episode-integrated contributions, at the nominal
# success design point.  `prog` is +52.6 rather than the tabulated +75: see
# `constants.py` §13.4 (F22), where `N_ref` had to be derived from the measured
# cruise speed rather than fixed at 250.
PREDICTED_NOMINAL = {
    "prog": +52.6,
    "pf": -27.0,
    "obs": -26.0,
    "exist": -15.0,
    "smooth": -1.5,
    "bnd": 0.0,
    "dom": 0.0,
    "col": 0.0,
}

# Outside a factor of 3 means the coefficient is wrong regardless of what the
# ratios say on paper (02a §8.2 step 3).
AUDIT_TOLERANCE = 3.0


def check_against_prediction(sums: Dict[str, float],
                             predicted: Dict[str, float] = None,
                             tolerance: float = AUDIT_TOLERANCE) -> List[str]:
    """Compare realised episode integrals against `02a §8.1`.

    Terms predicted to be exactly zero are skipped rather than compared: a
    factor-of-3 band around zero is empty, and `bnd`, `dom` and `col` are all
    predicted zero on a nominal success by construction -- a compliant episode
    that never approaches anything.  Their being non-zero is informative, but it
    is the *orderings* below, not this band, that judge them.
    """
    predicted = PREDICTED_NOMINAL if predicted is None else predicted
    problems = []
    for name, want in predicted.items():
        if abs(want) < 1e-9:
            continue
        got = float(sums.get(name, 0.0))
        if got == 0.0 or (got > 0.0) != (want > 0.0):
            problems.append(f"{name}: {got:+.1f} against a predicted {want:+.1f} "
                            f"(sign or zero)")
            continue
        ratio = abs(got / want)
        if ratio > tolerance or ratio < 1.0 / tolerance:
            problems.append(f"{name}: {got:+.1f} is {ratio:.1f}x the predicted "
                            f"{want:+.1f}")
    return problems


def compliance_cost_ratio(violation_return: float, compliance_cost: float) -> float:
    """`02a §8.1`'s single most important number.

    Cost of complying against cost of violating, over one encounter.  Predicted
    about 3.1.  **Below roughly 1.5 and `w_COL` is too low for compliance to be
    learned**, whatever the coefficient table says on paper -- the agent will
    take the violation because it is cheaper, and the ablation could not tell
    that apart from a classifier that never fired.
    """
    if abs(compliance_cost) < 1e-9:
        return float("inf")
    return abs(violation_return) / abs(compliance_cost)
