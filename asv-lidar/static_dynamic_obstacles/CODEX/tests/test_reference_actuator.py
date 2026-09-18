"""Compare onboard command-history estimates to the independent plant FIFO."""

from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from path import ReferencePath
from reference_controller import ReferenceController
from ship import COMMAND_RATE_PCT_S, ShipModel


@pytest.mark.parametrize("limited", [False, True])
def test_reference_delay_history_matches_executed_commands_through_reversals(limited):
    # A changing measured lateral error forces command reversals. The
    # controller receives only this measurement view; the test independently
    # advances a plant using the issued action and the configured bridge rule.
    pose = [4.0, 5.0, 0.0]
    inputs = SimpleNamespace(
        estimated_pose=lambda: tuple(pose),
        _measured_ego=lambda: (cfg.U_REF, 0.0, 0.0),
        path=ReferencePath([[5.0, y] for y in np.linspace(4.0, 22.0, 91)]),
        boundary_polygon=[(0, 0), (10, 0), (10, 25), (0, 25)],
        lidar=SimpleNamespace(ranges=np.full(720, cfg.LIDAR_RANGE),
                              bearings=np.arange(720) * 0.5),
        tracks=[], encounter_contexts={}, command_rate_limit=limited,
    )
    controller = ReferenceController(horizon_s=0.5)
    plant = ShipModel()
    applied = 0.0
    requested = []
    for lateral in (-1.0, -1.0, 1.0, 1.0, -1.0, 1.0):
        pose[0] = 5.0 + lateral
        action = controller.action(inputs, {})
        requested.append(float(action[0]))
        if limited:
            limit = COMMAND_RATE_PCT_S / 100.0 * cfg.UPDATE_RATE
            applied += float(np.clip(float(action[0]) - applied, -limit, limit))
        else:
            applied = float(action[0])
        plant.update(cfg.CRUISE_RPM, 100.0 * applied, cfg.UPDATE_RATE)
        assert controller.executed_rudder == pytest.approx(applied, abs=3e-8)
        assert controller.servo_angle == pytest.approx(float(plant._s[4, 0]), abs=3e-8)
        np.testing.assert_allclose(controller.actuator_buffer, list(plant._cmd_buf),
                                   rtol=0.0, atol=3e-8)
    assert min(requested) < 0.0 < max(requested)
