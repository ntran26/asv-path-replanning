"""COLREGs geometry and the per-step encounter context (02a §10.1-10.2).

Three modules, and the split is by *who owns the definition*:

* `classifier` -- re-exported from the top-level `encounter` module, which 01
  owns.  It is not copied here: `01 §5.3` requires exactly one definition of the
  encounter thresholds, and a re-export keeps the import path 02a §10.2 names
  without creating a second place the bands could drift.
* `geometry` -- CPA products, the admissibility predicate, the ship domain
  margins.  Pure functions of geometry, no state.
* `context` -- `EncounterContext`, the engagement state machine, and the latches.
  The only stateful part.

The point of the package is the context object.  Observation, reward and metrics
all read one per-step `EncounterContext` rather than each recomputing the class,
the risk and the CPA products from the same inputs.  Recomputation is not merely
wasteful: two consumers that disagree at a sector boundary would penalise the
agent for a role it was never shown, and that failure is close to undiagnosable
from a training curve.
"""

from encounter import (  # noqa: F401
    BEING_OVERTAKEN,
    CLASSES,
    CROSSING,
    HEAD_ON,
    NONE,
    OVERTAKING,
    EncounterClassifier,
    classify,
    crossing_side,
    one_hot,
)

from colregs.context import (  # noqa: F401
    CLEARING,
    ENGAGED,
    GIVE_WAY_CLASSES,
    IDLE,
    ContextManager,
    EncounterContext,
    compliant_turn_sense,
)
