"""The Paper 3 reward (02a Rev 2.2, with 02b C1-C4 applied).

```
r_t =  w_pf*r_pf + w_prog*r_prog + w_exist*r_exist + w_smooth*r_smooth
     + w_obs*r_obs + w_bnd*r_bnd + w_dom*r_dom + w_COL*r_col + r_term
```

Four modules, and the split is the one 02a §10.2 asks for:

* `config`  -- coefficients and the assertions that fail at *construction*
* `terms`   -- every term as a pure function of `(state, ctx, cfg)`
* `reward`  -- the weighted sum, the group clipping, the per-step breakdown
* `audit`   -- episode accumulators and Table R7

Redesigned rather than patched (D10).  No Paper 2 reward term name appears
anywhere in this tree, which is a deliberate constraint and not a stylistic one:
carrying a name across carries its scale assumptions with it, and that is how
Paper 2's path term came to outrank its avoidance term.
"""

from reward.audit import TermAudit, compliance_cost_ratio  # noqa: F401
from reward.config import DEFAULT, RewardConfig  # noqa: F401
from reward.reward import RewardBreakdown, RewardFunction  # noqa: F401
from reward.terms import RewardState  # noqa: F401
