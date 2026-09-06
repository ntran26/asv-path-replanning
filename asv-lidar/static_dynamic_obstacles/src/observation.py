"""Observation assembly: five branches, 56 dims, one dynamic target.

**Revision 2** — one target slot with a presence bit.  Supersedes the
three-slot-plus-mask-vector version.

The index order of every element is frozen in `OBSERVATION_SPEC.md`.  Every
checkpoint and every frozen evaluation case depends on it, so change it only
with a deliberate version bump -- never by editing a loop here.

    branch      contents                                   dims
    lidar       c_t sector closeness, obstacles only         27
    boundary    virtual boundary raycast, normalised          7
    ego         u, v, r                                       3
    path        e_y, chi_tilde, chi_tilde_LA                  3
    target      15 features + 1 presence bit                 16
                                                       total 56

Slot management (01 §6.2)
-------------------------
* **One target slot.**  `N_MAX_TARGETS` is a config parameter and the branch is
  built as an indexed slot, so a multi-vessel extension costs a retrain rather
  than a redesign (S1).  The machinery below is parameterised but not exercised
  at 1.
* **Track-ID persistence.**  The slot is bound on first acquisition and held
  until track loss, so observation discontinuities coincide with real events
  rather than with re-sorting.  CRI-based ordering is moot at one target, but
  the hook stays for the extension path -- when slots are contested, the
  highest-CRI targets hold them.
* **Presence bit, not a mask vector.**  Zero-padding alone is unsafe: zero is a
  legitimate value for bearing and for relative speed, so a zero-padded empty
  slot reads as a target sitting on top of the vessel on a matching course.

**No-target coverage.** A meaningful fraction of training episodes must carry no
target at all, or the static-only configuration is out of distribution.  That is
enforced by the scenario distribution (`cfg.NO_TARGET_EPISODE_PROB`), not here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

import constants as cfg
import cpa_cri as cc
import encounter as enc
from tracking import Track

# Per-slot feature layout.  Documented in OBSERVATION_SPEC.md; this tuple is the
# machine-readable copy and the two must agree.
SLOT_FEATURE_NAMES = (
    "distance_to_domain",        # 0
    "bearing_sin",               # 1
    "bearing_cos",               # 2
    "ct_sin",                    # 3
    "ct_cos",                    # 4
    "target_speed",              # 5
    "relative_speed",            # 6
    "dcpa",                      # 7
    "tcpa",                      # 8
    "cri",                       # 9
    "class_none",                # 10
    "class_head_on",             # 11
    "class_crossing",            # 12
    "class_overtaking",          # 13
    "class_being_overtaken",     # 14
    "presence",                  # 15
)
assert len(SLOT_FEATURE_NAMES) == cfg.TARGET_FEATURES
PRESENCE_INDEX = SLOT_FEATURE_NAMES.index("presence")
CLASS_SLICE = slice(10, 15)

TARGET_DIM = cfg.N_MAX_TARGETS * cfg.TARGET_FEATURES
OBS_DIM = cfg.LIDAR_SECTORS + cfg.BOUNDARY_RAYS + 3 + 3 + TARGET_DIM
assert OBS_DIM == 56, OBS_DIM


def observation_space() -> DictSpace:
    """The frozen observation space.  Five branches, 56 dims."""
    return DictSpace({
        "lidar": Box(0.0, 1.0, shape=(cfg.LIDAR_SECTORS,), dtype=np.float32),
        "boundary": Box(0.0, 1.0, shape=(cfg.BOUNDARY_RAYS,), dtype=np.float32),
        "ego": Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        "path": Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        "target": Box(-1.0, 1.0, shape=(TARGET_DIM,), dtype=np.float32),
    })


# ---------------------------------------------------------------------------
# Per-slot features
# ---------------------------------------------------------------------------
def slot_features(track: Track, p_os, v_os, heading_os_deg: float,
                  speed_os: float, encounter_class: str) -> np.ndarray:
    """The 16 values for one occupied slot, in the frozen order.

    Angles go in as sin/cos so the wraparound at +/-180 deg is not a
    discontinuity the network has to learn around.
    """
    p_ts = track.position
    v_ts = track.velocity
    heading_ts = track.course_deg

    bearing = cc.relative_bearing_deg(p_os, heading_os_deg, p_ts)
    ct = cc.heading_intersection_deg(heading_os_deg, heading_ts)
    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)

    # Distance and DCPA are both measured to the ship domain, not the hull.
    distance = cc.distance_to_domain(p_os, heading_os_deg, p_ts)
    dcpa_domain = max(0.0, dcpa - cc.domain_scale(bearing))

    relative_speed = float(np.linalg.norm(np.asarray(v_ts) - np.asarray(v_os)))
    risk = cc.cri(p_os, v_os, heading_os_deg, p_ts, v_ts, heading_ts)

    a = math.radians(bearing)
    c = math.radians(ct)

    kinematics = np.array([
        np.clip(distance / cfg.D_SCALE, 0.0, 1.0),
        math.sin(a),
        math.cos(a),
        math.sin(c),
        math.cos(c),
        np.clip(track.speed / cfg.SPEED_SCALE, 0.0, 1.0),
        np.clip(relative_speed / cfg.SPEED_SCALE, 0.0, 1.0),
        np.clip(dcpa_domain / cfg.DOMAIN_RADIUS_DCPA, 0.0, cfg.DCPA_CLIP_DOMAINS)
        / cfg.DCPA_CLIP_DOMAINS,
        np.clip(tcpa, -cfg.TCPA_CLIP, cfg.TCPA_CLIP) / cfg.TCPA_CLIP,
        np.clip(risk, 0.0, 1.0),
    ], dtype=np.float32)

    presence = np.ones(1, dtype=np.float32)
    return np.concatenate([kinematics, enc.one_hot(encounter_class), presence]).astype(np.float32)


# ---------------------------------------------------------------------------
# Slot management
# ---------------------------------------------------------------------------
class SlotManager:
    """Maps track ids to fixed observation slots, holding assignments stable.

    At `N_MAX_TARGETS = 1` this reduces to "the first track acquired keeps the
    slot until it is lost".  The contention path below is the extension hook
    (S1) and is not exercised in the two-vessel scope.
    """

    def __init__(self, n_slots: int = cfg.N_MAX_TARGETS) -> None:
        self.n_slots = int(n_slots)
        self.reset()

    def reset(self) -> None:
        self._slot_of: Dict[int, int] = {}
        self._track_in: List[Optional[int]] = [None] * self.n_slots

    @property
    def assignments(self) -> Dict[int, int]:
        return dict(self._slot_of)

    def occupant(self, slot: int) -> Optional[int]:
        return self._track_in[slot]

    def update(self, tracks: Sequence[Track], risks: Sequence[float]) -> Dict[int, int]:
        """Assign slots for this step and return {track_id: slot}.

        Held assignments survive; free slots go to the highest-risk unassigned
        targets; a full board is contested only when a newcomer out-ranks the
        weakest occupant.
        """
        live = {t.id for t in tracks}

        # Release slots whose track is gone.
        for tid in [tid for tid in self._slot_of if tid not in live]:
            self._track_in[self._slot_of.pop(tid)] = None

        risk_of = {t.id: float(r) for t, r in zip(tracks, risks)}
        unassigned = sorted((t for t in tracks if t.id not in self._slot_of),
                            key=lambda t: risk_of[t.id], reverse=True)

        for track in unassigned:
            if None in self._track_in:
                slot = self._track_in.index(None)
            else:
                # Contest the weakest occupant, and take the slot only if this
                # target is genuinely riskier.
                weakest = min(self._slot_of, key=lambda tid: risk_of.get(tid, 0.0))
                if risk_of[track.id] <= risk_of.get(weakest, 0.0):
                    continue
                slot = self._slot_of.pop(weakest)
                self._track_in[slot] = None

            self._slot_of[track.id] = slot
            self._track_in[slot] = track.id

        return dict(self._slot_of)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
class ObservationBuilder:
    """Builds the Dict observation and owns the slot and encounter state.

    The encounter classifier held here is the same module 02 imports for the
    reward gate (01 §5.3).  There is one classifier per environment instance, so
    the role shown to the policy and the role the reward conditions on are by
    construction the same value on the same step.
    """

    def __init__(self, n_slots: int = cfg.N_MAX_TARGETS) -> None:
        self.slots = SlotManager(n_slots)
        self.encounters = enc.EncounterClassifier()
        self.n_slots = int(n_slots)
        self._last_classes: Dict[int, str] = {}
        self._last_sides: Dict[int, str] = {}

    def reset(self) -> None:
        self.slots.reset()
        self.encounters.reset()
        self._last_classes = {}
        self._last_sides = {}

    @property
    def encounter_classes(self) -> Dict[int, str]:
        """{track_id: class} as of the last `build`.  02's reward gate reads this."""
        return dict(self._last_classes)

    @property
    def crossing_sides(self) -> Dict[int, str]:
        """{track_id: "port"|"starboard"|"none"} as of the last `build`.

        Not in the observation -- Rule 9(b) makes the own ship give way from
        either side -- but 02's passing-side reward term needs the geometry.
        """
        return dict(self._last_sides)

    def build(self, *, sector_closeness, boundary_scan, u: float, v: float,
              yaw_rate_degps: float, cross_track_error: float,
              course_error_deg: float, lookahead_course_error_deg: float,
              tracks: Sequence[Track] = (), p_os=(0.0, 0.0), v_os=(0.0, 0.0),
              heading_os_deg: float = 0.0) -> Dict[str, np.ndarray]:
        """Assemble one observation."""
        speed_os = float(np.linalg.norm(v_os))

        # Encounter class per track, through the shared hysteretic classifier.
        classes: Dict[int, str] = {}
        sides: Dict[int, str] = {}
        risks: List[float] = []
        for track in tracks:
            classes[track.id] = self.encounters.update(
                track.id, p_os, heading_os_deg, speed_os,
                track.position, track.course_deg, track.speed,
            )
            sides[track.id] = enc.crossing_side(p_os, heading_os_deg,
                                                track.position, track.course_deg)
            risks.append(cc.cri(p_os, v_os, heading_os_deg,
                                track.position, track.velocity, track.course_deg))
        self._last_classes = classes
        self._last_sides = sides

        # Drop history for tracks that have gone, so slot re-use is clean.
        live = {t.id for t in tracks}
        for tid in list(self.encounters._held):
            if tid not in live:
                self.encounters.forget(tid)

        assignment = self.slots.update(tracks, risks)
        by_id = {t.id: t for t in tracks}

        slot_block = np.zeros((self.n_slots, cfg.TARGET_FEATURES), dtype=np.float32)
        for tid, slot in assignment.items():
            track = by_id[tid]
            slot_block[slot] = slot_features(track, p_os, v_os, heading_os_deg,
                                             speed_os, classes[tid])

        span = max(cfg.MAP_WIDTH, cfg.MAP_HEIGHT)
        return {
            "lidar": np.asarray(sector_closeness, dtype=np.float32),
            "boundary": np.asarray(boundary_scan, dtype=np.float32),
            "ego": np.array([
                np.clip(u / cfg.SPEED_SCALE, -1.0, 1.0),
                np.clip(v / cfg.SPEED_SCALE, -1.0, 1.0),
                np.clip(yaw_rate_degps / 180.0, -1.0, 1.0),
            ], dtype=np.float32),
            "path": np.array([
                np.clip(cross_track_error / span, -1.0, 1.0),
                np.clip(course_error_deg / 180.0, -1.0, 1.0),
                np.clip(lookahead_course_error_deg / 180.0, -1.0, 1.0),
            ], dtype=np.float32),
            "target": slot_block.reshape(-1).astype(np.float32),
        }


def split_target(target) -> tuple:
    """Split the `target` branch into (slot features, presence bits).

    The single place that knows the layout, so the environment, the features
    extractor and any analysis script all agree.
    """
    arr = np.asarray(target, dtype=np.float32)
    slots = arr.reshape(*arr.shape[:-1], cfg.N_MAX_TARGETS, cfg.TARGET_FEATURES)
    return slots, slots[..., PRESENCE_INDEX]
