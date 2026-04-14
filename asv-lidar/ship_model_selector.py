"""Single-point ship-model selection for the ASV RL stack.

Change `SHIP_MODEL_VARIANT` below to switch the active model used by the main
environment and training script.
"""

from __future__ import annotations

import math
import numpy as np

# SHIP_MODEL_VARIANT = "standard_3dof"
SHIP_MODEL_VARIANT = "bluefin_4dof"

if SHIP_MODEL_VARIANT == "standard_3dof":
    from ship_model import (  # type: ignore
        ShipModel,
        THRUST_COEF,
        DRAG_COEF,
        MAX_RUD_ANGLE,
        MAX_SURGE_SPEED,
        MAX_SWAY_SPEED,
        VESSEL_LENGTH,
        VESSEL_WIDTH,
        HULL_MARGIN,
        HULL_FORWARD_SHIFT,
    )

    MODEL_RPM_MAX = 24.0
    MODEL_U_MAX = float(np.sqrt(THRUST_COEF / DRAG_COEF) * MODEL_RPM_MAX)
    MODEL_COMMAND_SCALE = 1.0
    MODEL_INTERNAL_RPM_MAX = MODEL_RPM_MAX

    def model_u_body(model: ShipModel) -> float:
        return float(model._v)

    def model_v_body(model: ShipModel) -> float:
        return float(model._v_sway)

    def model_rudder_deg(model: ShipModel) -> float:
        return float(model.state_dict()["rudder_deg"])

elif SHIP_MODEL_VARIANT == "bluefin_4dof":
    from ship_model_bluefin_4dof import (  # type: ignore
        ShipModel as _RawShipModel,
        MAX_RUD_ANGLE,
        MAX_SURGE_SPEED,
        MAX_SWAY_SPEED,
        VESSEL_LENGTH,
        VESSEL_WIDTH,
        HULL_MARGIN,
        HULL_FORWARD_SHIFT,
        RPM_COMMAND_SCALE,
        RECOMMENDED_COMMAND_RPM_MAX,
        RECOMMENDED_PROP_RPM_MAX,
        RECOMMENDED_PEAK_SPEED_MPS,
    )

    # The raw 4DOF adapter follows the MATLAB heading convention internally:
    # 0 deg along +x, CCW-positive. The rest of the repo expects:
    # 0 deg along +y, clockwise-positive. Wrap the model here so callers such
    # as rl_env.py keep the historical repo convention without needing edits.
    def _repo_heading_deg_to_raw(repo_heading_deg: float) -> float:
        return (90.0 - float(repo_heading_deg)) % 360.0

    def _raw_heading_deg_to_repo(raw_heading_deg: float) -> float:
        return (90.0 - float(raw_heading_deg)) % 360.0

    def _startup_rudder_gain(model: _RawShipModel) -> float:
        # The raw 4DOF model is calibrated around underway manoeuvres and can
        # become numerically aggressive if a controller commands full rudder
        # from rest. Fade rudder authority in with surge speed so the runtime
        # env remains robust while preserving the calibrated underway behaviour.
        speed_mps = math.hypot(float(model._u), float(model._v))
        return float(np.clip(speed_mps / 0.25, 0.0, 1.0))


    class ShipModel:
        def __init__(self) -> None:
            object.__setattr__(self, "_impl", _RawShipModel())
            self._apply_repo_frame_defaults()

        def _apply_repo_frame_defaults(self) -> None:
            # Repo convention: heading 0 deg means pointing along +y.
            self._impl._psi = math.radians(_repo_heading_deg_to_raw(0.0))
            self._impl._h = 0.0
            self._impl._w = 0.0

        def reset(self) -> None:
            self._impl.reset()
            self._apply_repo_frame_defaults()

        def update(self, rpm: float, rud: float, dt: float, *, thruster_rpm: float = 0.0):
            effective_rud = float(rud) * _startup_rudder_gain(self._impl)
            dx, dy, raw_heading_deg, raw_yaw_rate_degps = self._impl.update(
                rpm, effective_rud, dt, thruster_rpm=thruster_rpm
            )
            repo_heading_deg = _raw_heading_deg_to_repo(raw_heading_deg)
            repo_yaw_rate_degps = -float(raw_yaw_rate_degps)

            # Keep compatibility fields aligned with the repo-facing convention.
            self._impl._h = math.radians(repo_heading_deg)
            self._impl._w = math.radians(repo_yaw_rate_degps)
            return dx, dy, repo_heading_deg, repo_yaw_rate_degps

        def state_dict(self):
            state = self._impl.state_dict()
            state["heading_deg"] = _raw_heading_deg_to_repo(state["heading_deg"])
            state["heading_rad"] = math.radians(state["heading_deg"])
            state["yaw_rate_degps"] = -float(state["yaw_rate_degps"])
            state["yaw_rate_radps"] = math.radians(state["yaw_rate_degps"])
            return state

        def __getattr__(self, name: str):
            return getattr(self._impl, name)

        def __setattr__(self, name: str, value) -> None:
            if name == "_impl":
                object.__setattr__(self, name, value)
            else:
                setattr(self._impl, name, value)

    MODEL_RPM_MAX = float(RECOMMENDED_COMMAND_RPM_MAX)
    MODEL_U_MAX = float(RECOMMENDED_PEAK_SPEED_MPS)
    MODEL_COMMAND_SCALE = float(RPM_COMMAND_SCALE)
    MODEL_INTERNAL_RPM_MAX = float(RECOMMENDED_PROP_RPM_MAX)

    def model_u_body(model: ShipModel) -> float:
        return float(model._u)

    def model_v_body(model: ShipModel) -> float:
        return float(model._v)

    def model_rudder_deg(model: ShipModel) -> float:
        return float(model.state_dict()["rudder_deg"])

else:
    raise ValueError(
        f"Unsupported SHIP_MODEL_VARIANT: {SHIP_MODEL_VARIANT!r}. "
        "Use 'standard_3dof' or 'bluefin_4dof'."
    )
