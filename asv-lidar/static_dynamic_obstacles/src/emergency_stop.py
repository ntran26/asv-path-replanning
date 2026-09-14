"""Emergency stop: reverse propulsion as a latched manoeuvre, not an action.

Rule 8(e): "If necessary to avoid collision or allow more time to assess the
situation, a vessel shall slacken her speed or take all way off by stopping or
reversing her means of propulsion."  Rule 17(b) asks the same of a stand-on
vessel in extremis.

**Why a latch rather than a wider action space.**  The vessel accepts a
propulsion command `S2` in `[-100, +100]`; the policy's action space covers only
the forward half.  Opening the reverse half to the policy would let it learn to
reverse routinely -- a behaviour nobody wants on the water, with a reverse
thrust nobody has measured -- and every speed-scaled term in the reward would
have to be re-derived for astern motion.  A latch keeps the learned action
space forward-only and makes reverse a rare, auditable event with a single
entry condition.

The manoeuvre (as specified for this project):

    IDLE --request--> BRAKING   S2 = -100 until speed <= stop_speed
                      HOLDING   S2 =    0 to stay still
                      IDLE      control returns to the policy

**Pure and dependency-free on purpose.**  This module imports nothing from the
simulator, so the deployment bridge (`field_deployment/udp_live_rl.py`) can run
the identical state machine against telemetry.  The same code in both places is
the only way the stop the policy was trained around is the stop the vessel
performs.

Three things the specification left open, and what is done here:

* **Release.**  "Switch back to 0 propulsion to stay still" has no exit, and a
  vessel that stays still forever times out every episode.  HOLDING returns to
  the policy once the caller reports the danger has passed (`release_ok`) after
  at least `min_hold_s`, or unconditionally after `max_hold_s` -- the second
  exit exists because a target that stops dead ahead would otherwise pin the
  latch for the rest of the episode.
* **Braking without a result.**  If speed never reaches `stop_speed` within
  `max_brake_s` (reverse thrust weaker than modelled, or a telemetry fault
  reading speed high), BRAKING gives up into HOLDING rather than commanding
  full astern indefinitely.
* **The speed threshold is not zero.**  In the field, speed comes from rf2o
  pose at 2 Hz with 0.1 m quantisation, so a literal `speed <= 0` test either
  fires early on quantisation or never fires on noise.  `stop_speed` should sit
  at the speed-estimate noise floor, which is a 05 measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

IDLE = "idle"
BRAKING = "braking"
HOLDING = "holding"

S2_FULL_ASTERN = -100.0
S2_STOP = 0.0


@dataclass
class StopEvent:
    """One completed or in-progress stop, for metrics and the panel."""

    reason: str
    t_request: float
    speed_at_request: float
    t_stopped: Optional[float] = None       # speed first <= stop_speed
    t_released: Optional[float] = None
    distance_braking_m: float = 0.0         # travelled from request to stopped
    gave_up: bool = False                   # braking timed out

    @property
    def time_to_stop_s(self) -> Optional[float]:
        if self.t_stopped is None:
            return None
        return self.t_stopped - self.t_request


@dataclass
class EmergencyStop:
    """The stop latch.  Call `update` once per control step."""

    stop_speed: float = 0.05
    min_hold_s: float = 2.0
    max_hold_s: float = 10.0
    max_brake_s: float = 8.0

    state: str = IDLE
    events: List[StopEvent] = field(default_factory=list)
    _t_state: float = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = IDLE
        self.events = []
        self._t_state = 0.0

    @property
    def active(self) -> bool:
        return self.state != IDLE

    @property
    def current(self) -> Optional[StopEvent]:
        return self.events[-1] if self.events and self.state != IDLE else None

    # ------------------------------------------------------------------
    def request(self, reason: str, t: float, speed: float) -> bool:
        """Ask for a stop.  Returns True if this started a new one.

        A request while already stopping is absorbed, not restarted: re-entering
        BRAKING from HOLDING would reset the hold timer every step the danger
        persists, and the `max_hold_s` exit would never be reached.
        """
        if self.state != IDLE:
            return False
        self.state = BRAKING
        self._t_state = 0.0
        self.events.append(StopEvent(reason=str(reason), t_request=float(t),
                                     speed_at_request=float(speed)))
        return True

    def update(self, *, t: float, dt: float, speed: float,
               distance_step_m: float = 0.0,
               release_ok: bool = False) -> Optional[float]:
        """Advance one control step.  Returns the S2 override, or None.

        `None` means the latch is idle and the policy's own propulsion command
        stands.  `speed` is the forward speed the controller believes, which in
        the field is an estimate -- see the module docstring on `stop_speed`.
        """
        self._t_state += float(dt)
        event = self.current

        if self.state == BRAKING:
            event.distance_braking_m += float(distance_step_m)
            if speed <= self.stop_speed:
                event.t_stopped = float(t)
                self._enter(HOLDING)
            elif self._t_state >= self.max_brake_s:
                event.gave_up = True
                self._enter(HOLDING)

        elif self.state == HOLDING:
            held = self._t_state
            if (held >= self.min_hold_s and release_ok) or held >= self.max_hold_s:
                event.t_released = float(t)
                self._enter(IDLE)
                return None

        if self.state == BRAKING:
            return S2_FULL_ASTERN
        if self.state == HOLDING:
            return S2_STOP
        return None

    def _enter(self, state: str) -> None:
        self.state = state
        self._t_state = 0.0


def s2_to_rpm(s2: float, rpm_max: float = 24.0, s2_max: float = 100.0) -> float:
    """Propulsion command `S2` to the simulator's rpm-unit scale.

    The bridge maps `S2 = rpm / 24 * 100` for forward motion
    (`udp_live_rl.rpm_to_s2_cmd`); the same linear map is extended through zero
    so full astern, `S2 = -100`, is `-24` rpm-units.  Reverse *thrust* at a given
    command is a separate, unmeasured property of the propeller -- see
    `ship.REVERSE_THRUST_EFFICIENCY`.
    """
    return float(s2) / float(s2_max) * float(rpm_max)


def rpm_to_s2(rpm: float, rpm_max: float = 24.0, s2_max: float = 100.0) -> float:
    return float(rpm) / float(rpm_max) * float(s2_max)


# ---------------------------------------------------------------------------
# When to stop, and when the danger has passed
# ---------------------------------------------------------------------------
# Duck-typed over anything carrying the `EncounterContext` fields, so this stays
# importable by the bridge without the simulator.

def stop_required(contexts) -> Optional[str]:
    """The supervisor trigger.  Returns a reason string, or None.

    **Fires only when a give-way encounter is in extremis AND the compliant
    alteration is inadmissible** -- the one situation in which the precedence
    table leaves Rule 8(e) as the only lawful response.  Three things it
    deliberately does not do:

    * **pre-empt a policy that could still turn.**  If the compliant alteration
      is admissible, avoiding the collision is the policy's job, and a
      supervisor that stopped the vessel there would train the policy to leave
      its manoeuvres late and be rescued;
    * **fire when being overtaken.**  Stopping dead in front of an overtaking
      vessel is the wrong act under 17(b) -- it removes the stand-on vessel's
      way at the moment the overtaker is relying on it;
    * **read ground truth.**  The contexts are perceived.  A controller on the
      water has nothing else, and a supervisor that read truth in training
      would hand the evaluated system information the comparators never get
      (04 §8).

    `in_extremis` is 02a's own 17(b) predicate -- DCPA inside `d_req` with TCPA
    under 5 s -- reused rather than duplicated with a new free constant.  The
    stop takes 1.1-3.7 s across plausible reverse efficiencies, so a 5 s window
    leaves time for it.
    """
    for ctx in contexts:
        if (getattr(ctx, "engaged", False) and getattr(ctx, "gives_way", False)
                and getattr(ctx, "in_extremis", False)
                and not getattr(ctx, "turn_admissible", True)):
            return (f"8(e) in extremis: {ctx.cls}, compliant alteration "
                    f"inadmissible (DCPA {ctx.dcpa:.2f} m, TCPA {ctx.tcpa:.1f} s)")
    return None


def danger_passed(contexts) -> bool:
    """True when no engaged encounter is still in extremis and closing.

    `TCPA < 0` means the CPA is behind: the range is opening and holding
    station no longer buys anything.  With no contexts at all -- the target lost
    or never seen -- there is nothing to hold for.
    """
    return not any(getattr(ctx, "engaged", False) and getattr(ctx, "in_extremis", False)
                   and getattr(ctx, "tcpa", -1.0) >= 0.0 for ctx in contexts)

