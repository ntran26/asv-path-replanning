"""Controller information boundary and independent behaviour contracts.

The slower, complete deterministic scenes live in tools/validate_behaviors.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from path import ReferencePath
from reference_controller import ReferenceController
from ship import ShipModel


class OnboardInputs:
    """Only the measurements and map available on the vessel are exposed."""
    def __init__(self, x=5.0, tracks=()):
        self.pose = (x, 5.0, 0.0)
        self.path = ReferencePath([[5.0, y] for y in np.linspace(4, 22, 91)])
        self.boundary_polygon = [(0, 0), (10, 0), (10, 25), (0, 25)]
        self.lidar = SimpleNamespace(ranges=np.full(720, cfg.LIDAR_RANGE), bearings=np.arange(720)*0.5)
        self.tracks = tracks
        self.encounter_contexts = {}

    def estimated_pose(self):
        return self.pose

    def _measured_ego(self):
        return cfg.U_REF, 0.0, 0.0

    def __getattr__(self, name):
        if name in ("targets", "obstacles", "model", "asv_x", "asv_y", "asv_h", "u_body"):
            raise AssertionError(f"controller accessed simulation truth: {name}")
        raise AttributeError(name)


def test_clear_path_cruise_uses_only_onboard_inputs():
    command = ReferenceController().action(OnboardInputs(), {})
    assert command == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("offset", [-1.5, 1.5])
def test_recovery_steers_towards_path(offset):
    command = ReferenceController().action(OnboardInputs(5.0 + offset), {})
    assert offset * command[0] < 0.0
    assert np.isfinite(command).all() and np.max(np.abs(command)) <= 1.0


def test_head_on_prediction_commits_to_starboard_without_truth():
    track = SimpleNamespace(id=1, position=np.array([5.0, 15.0]), velocity=np.array([0.0, -cfg.U_REF]))
    measurements = OnboardInputs(tracks=[track])
    measurements.encounter_contexts[1] = SimpleNamespace(gives_way=True, tcpa=9.0, dcpa=0.0, compliant_turn_sense=1)
    controller = ReferenceController()
    command = controller.action(measurements, {})
    assert controller.offset > 0.0
    assert command[0] > 0.0


def test_controller_actuator_estimate_matches_issued_command_history():
    """Reversing requests must not reset the delayed servo to the new command."""
    controller = ReferenceController()
    model = ShipModel()
    measurements = OnboardInputs()
    for x in (6.5, 3.5, 6.5, 3.5):
        measurements.pose = (x, 5.0, 0.0)
        action = controller.action(measurements, {})
        model.update(cfg.CRUISE_RPM, 100.0 * float(action[0]), cfg.UPDATE_RATE)
        assert controller.servo_angle == pytest.approx(np.radians(model.rudder_deg), abs=2e-7)


def test_safe_stand_on_preserves_course_and_cruise_under_noisy_measurements():
    """Recorded nominal-noise inputs previously selected discretionary slowing."""
    track = SimpleNamespace(id=18, position=np.array([2.596, 13.445]), velocity=np.array([-0.049, 0.889]))
    measurements = OnboardInputs(tracks=[track])
    measurements.pose = (4.956, 13.476, 4.098)
    measurements._measured_ego = lambda: (0.614, -0.032, 0.230)
    measurements.encounter_contexts[18] = SimpleNamespace(cls="being_overtaken", state="clearing", gives_way=False,
        tcpa=-1.720, dcpa=2.311, compliant_turn_sense=0, in_extremis=False)
    controller = ReferenceController()
    action = controller.action(measurements, {})
    assert controller.offset == 0.0
    assert action[1] == 0.0
    assert controller.last_diagnostics["stand_on_active"]
    assert not controller.last_diagnostics["stand_on_override"]
    assert not controller.last_diagnostics["fallback"]


def test_unsafe_stand_on_prediction_retains_avoidance_and_reports_override():
    """A stand-on label cannot force an unsafe hold when the track is ahead."""
    track = SimpleNamespace(id=1, position=np.array([5.0, 10.0]), velocity=np.array([0.0, 0.20]))
    measurements = OnboardInputs(tracks=[track])
    measurements.encounter_contexts[1] = SimpleNamespace(cls="being_overtaken", state="engaged", gives_way=False,
        tcpa=8.0, dcpa=0.0, compliant_turn_sense=0, in_extremis=True)
    controller = ReferenceController()
    command = controller.action(measurements, {})
    assert controller.last_diagnostics["stand_on_override"]
    assert not controller.last_diagnostics["fallback"]
    assert controller.offset != 0.0 or command[1] != 0.0
