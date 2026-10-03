"""Pure saved-data helper checks: no environments, policies, or episodes.

These tests check declared conditioning and time/measurement provenance.  They
do not turn Gaussian confidence boxes or sampled arithmetic into certificates.
"""

from copy import deepcopy
import math

import numpy as np
import pytest

from safety_reachability import EnclosureFailure, IntervalBounds
from tools.diagnostics.safety import reachability_containment as audit


def constants():
    return {
        "BOUNDARY_POSE_NOISE_XY": 0.03,
        "BOUNDARY_POSE_NOISE_HEADING_DEG": 0.2,
        "BOUNDARY_POSE_NOISE_WALK": 0.0,
        "EGO_SPEED_NOISE": 0.05,
        "EGO_YAW_RATE_NOISE_DPS": 1.0,
    }


def frame():
    return {
        "step": 1, "elapsed_time": 0.0, "fresh": True,
        "pose": [2.0, 4.0, 359.0], "ego": [0.5, -0.02, 2.0],
        "command": (20.0, 12.0), "tracks": [],
    }


def raw_row(step=1):
    return {
        "step": step,
        "rudder_command": 0.2,
        "signed_rpm_command": -24.0,
        "diagnostic_before": {
            "elapsed_time": (step - 1) * 0.5,
            "onboard": {
                "pose_hold_xy_heading_deg": [2.0, 4.0, 359.0],
                "ego_hold_u_v_yaw_deg_s": [0.5, -0.02, 2.0],
                "pose_stale": False,
            },
            "truth_scoring_only": {
                "own": {"asv_x": 2.1, "asv_y": 4.2, "asv_h": 0.5,
                        "u_body": 0.51, "v_body": -0.01, "asv_w": 1.5},
                "targets": [{"x": 8.0, "y": 9.0, "heading": 90.0, "speed": 0.5}],
            },
        },
        "diagnostic_decision": {"snapshot": {"tracks": []}},
        # Terminal/post-step truth is not a synchronized next onboard frame.
        "post_state": {"x": 1e9, "y": -1e9, "heading": 12345.0},
        "outcome": "collision:target",
    }


def assert_covers(bound, low, high):
    assert bound.lo <= low + 1e-12
    assert bound.hi >= high - 1e-12
    assert bound.lo == pytest.approx(low, abs=1e-12)
    assert bound.hi == pytest.approx(high, abs=1e-12)


def test_conditioned_parameter_box_includes_blends_scaled_about_mean_and_jitter():
    bank = np.array([[2.0, 0.0, 0.5, -8.0], [4.0, 0.0, 0.5, -4.0]])
    names = ("T12", "t_boost", "u_boost", "N_uv")
    original = bank.copy()
    box = audit.conditional_parameter_bounds(
        bank, names, scale=2.0, jitter_fraction=0.05, sigma_multiple=3.0)
    # Independent endpoints: scaled pair hull [1,5], sigma jitter .3; negative
    # coordinate scaled hull [-10,-2], sigma jitter .6.  Not CI_LOW/CI_HIGH.
    assert_covers(box["T12"], 0.7, 5.3)
    assert_covers(box["N_uv"], -10.6, -1.4)
    assert_covers(box["t_boost"], 0.0, 0.0)
    assert_covers(box["u_boost"], 0.5, 0.5)
    np.testing.assert_array_equal(bank, original)


def test_parameter_sign_clamp_includes_zero_mass_at_clipped_tail():
    bank = np.array([[0.0, 0.0], [0.1, -0.1]])
    box = audit.conditional_parameter_bounds(
        bank, ("positive", "negative"), jitter_fraction=1.0, sigma_multiple=3.0)
    assert_covers(box["positive"], 0.0, 0.25)
    assert_covers(box["negative"], -0.25, 0.0)


@pytest.mark.parametrize("bank,names", [
    (np.array([[1.0, math.nan]]), ("a", "b")),
    (np.array([[1.0, math.inf]]), ("a", "b")),
    (np.empty((0, 2)), ("a", "b")),
    (np.array([1.0, 2.0]), ("a", "b")),
    (np.array([[1.0, 2.0]]), ("a",)),
    (np.array([[1.0, 2.0]]), ("a", "a")),
])
def test_invalid_parameter_support_does_not_silently_make_a_box(bank, names):
    with pytest.raises((ValueError, TypeError)):
        audit.conditional_parameter_bounds(bank, names)


@pytest.mark.parametrize("kwargs", [
    {"scale": -1.0}, {"scale": math.inf},
    {"jitter_fraction": -0.1}, {"jitter_fraction": math.nan},
    {"sigma_multiple": -1.0}, {"sigma_multiple": math.inf},
])
def test_invalid_conditional_support_controls_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        audit.conditional_parameter_bounds(np.array([[1.0], [2.0]]), ("a",), **kwargs)


def test_initial_bounds_use_fresh_raw_sensor_units_not_filtered_or_true_state():
    row = frame()
    box = audit.initial_state_bounds(row, constants(), sigma_multiple=3.0)
    assert_covers(box["x"], 1.91, 2.09)
    assert_covers(box["y"], 3.91, 4.09)
    assert_covers(box["heading"], math.radians(358.4), math.radians(359.6))
    assert_covers(box["u"], 0.35, 0.65)
    assert_covers(box["v"], -0.17, 0.13)
    assert_covers(box["r"], math.radians(-1.0), math.radians(5.0))


def test_initial_bounds_ignore_current_command_future_and_scoring_poison():
    base = frame()
    expected = audit.initial_state_bounds(base, constants())
    poisoned = deepcopy(base)
    poisoned.update(command=(math.nan, math.nan), tracks=[{"truth": math.nan}],
                    outcome="goal", future=[{"u": 1e100}],
                    truth_scoring_only={"own": {"x": math.nan}},
                    filtered_state={"x": 1e100, "u": -1e100})
    assert audit.initial_state_bounds(poisoned, constants()) == expected


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(fresh=False),
    lambda r: r.pop("fresh"),
    lambda r: r.pop("pose"),
    lambda r: r.pop("ego"),
    lambda r: r.update(pose=[1.0, 2.0]),
    lambda r: r.update(ego=[math.nan, 0.0, 0.0]),
    lambda r: r.update(pose=[1.0, math.inf, 0.0]),
])
def test_missing_stale_or_nonfinite_causal_sensor_data_rejected(mutation):
    row = frame()
    mutation(row)
    with pytest.raises((ValueError, KeyError, TypeError)):
        audit.initial_state_bounds(row, constants())


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("EGO_YAW_RATE_NOISE_DPS"),
    lambda c: c.update(BOUNDARY_POSE_NOISE_WALK=0.001),
    lambda c: c.update(EGO_SPEED_NOISE=-0.01),
    lambda c: c.update(BOUNDARY_POSE_NOISE_XY=math.inf),
])
def test_undeclared_or_unsupported_measurement_uncertainty_rejected(mutation):
    effective = constants()
    mutation(effective)
    with pytest.raises((ValueError, KeyError, TypeError)):
        audit.initial_state_bounds(frame(), effective)


@pytest.mark.parametrize("low,high", [(0.0, 0.0), (0.024, 0.026), (0.124, 0.126),
                                      (0.125, 0.125), (0.175, 0.175), (0.58, 0.80)])
def test_delay_branches_cover_rounding_ties_and_both_adjacent_sides(low, high):
    branches = audit.delay_step_branches(IntervalBounds(low, high), step_s=0.05)
    assert branches == sorted(set(branches))
    assert all(isinstance(n, int) and n >= 0 for n in branches)
    probes = np.linspace(low, high, 101).tolist()
    for n in range(20):
        tie = (n + 0.5) * 0.05
        probes.extend(x for x in (np.nextafter(tie, -np.inf), tie,
                                   np.nextafter(tie, np.inf)) if low <= x <= high)
    assert {round(float(x) / 0.05) for x in probes} <= set(branches)


@pytest.mark.parametrize("step", [0.0, -0.05, math.nan, math.inf])
def test_invalid_delay_grid_rejected(step):
    with pytest.raises((ValueError, TypeError)):
        audit.delay_step_branches(IntervalBounds(0.1, 0.2), step_s=step)


def test_periodic_heading_matches_unwrapped_enclosure_without_widening_other_state():
    box = IntervalBounds(math.radians(358.0), math.radians(362.0))
    assert audit.score_interval(math.radians(1.0), box, periodic=True)
    assert audit.score_interval(math.radians(-1.0), box, periodic=True)
    assert audit.score_interval(math.radians(721.0), box, periodic=True)
    assert not audit.score_interval(math.radians(180.0), box, periodic=True)
    assert not audit.score_interval(math.radians(1.0), box)
    assert audit.score_interval(box.lo, box)
    assert audit.score_interval(box.hi, box)


def test_truth_scoring_nonfinite_value_is_not_reported_as_contained():
    for value in (math.nan, math.inf, -math.inf):
        try:
            contained = audit.score_interval(value, IntervalBounds(-1.0, 1.0), periodic=True)
        except (ValueError, TypeError):
            continue
        assert contained is False


def test_trace_stripping_separates_scoring_and_keeps_signed_actual_command():
    raw = raw_row()
    saved = deepcopy(raw)
    causal, scoring = audit.strip_trace_rows([raw])
    assert raw == saved
    assert len(causal) == len(scoring) == 1
    assert set(causal[0]) == {"step", "elapsed_time", "pose", "ego", "fresh", "command", "tracks"}
    assert causal[0]["command"] == (20.0, -24.0)
    assert causal[0]["pose"] == [2.0, 4.0, 359.0]
    assert causal[0]["ego"] == [0.5, -0.02, 2.0]
    assert scoring[0]["own"]["heading"] == pytest.approx(math.radians(0.5))
    assert scoring[0]["own"]["r"] == pytest.approx(math.radians(1.5))
    # Changing truth/outcome/post-state cannot change the inputs to propagation.
    raw["diagnostic_before"]["truth_scoring_only"]["own"].update(asv_x=1e6, asv_h=179.0)
    raw["diagnostic_before"]["truth_scoring_only"]["targets"] = []
    raw["post_state"] = {"x": -1e99}
    raw["outcome"] = "goal"
    changed_causal, changed_scoring = audit.strip_trace_rows([raw])
    assert changed_causal == causal
    assert changed_scoring != scoring


def test_trace_stripping_copies_causal_lists_before_truth_scoring():
    rows = [raw_row()]
    causal, scoring = audit.strip_trace_rows(rows)
    rows[0]["diagnostic_before"]["onboard"]["pose_hold_xy_heading_deg"][0] = 99.0
    rows[0]["diagnostic_before"]["truth_scoring_only"]["own"]["asv_x"] = 123.0
    assert causal[0]["pose"][0] == 2.0
    assert scoring[0]["own"]["x"] == 2.1


class ToyTube:
    """Small deterministic stand-in to test runner data flow, not vessel math."""

    instances = []
    failing_branch = None

    def __init__(self, parameters, state, delay_steps, command_history=None):
        self.state = dict(state, servo=IntervalBounds(0.0, 0.0))
        self.branch = delay_steps
        self.history = deepcopy(command_history)
        self.calls = []
        self.diagnostics = {"toy": True}
        self.instances.append(self)

    def bounds(self):
        return dict(self.state)

    def advance(self, rudder, rpm, dt):
        self.calls.append((rudder, rpm, dt))
        if self.branch == self.failing_branch:
            raise EnclosureFailure("synthetic missing delay branch")
        assert math.isfinite(rudder) and math.isfinite(rpm)
        shift = rpm * dt * 0.01
        old = self.state["x"]
        self.state["x"] = IntervalBounds(old.lo + shift, old.hi + shift)
        return self.bounds()


@pytest.fixture
def toy_tubes(monkeypatch):
    ToyTube.instances, ToyTube.failing_branch = [], None
    monkeypatch.setattr(audit, "ConditionalPlantTube", ToyTube)
    return ToyTube


def test_runner_stops_at_last_predecision_frame_not_terminal_posttruth(toy_tubes):
    causal, scoring = audit.strip_trace_rows([raw_row(i) for i in (1, 2, 3)])
    causal[-1]["command"] = (math.nan, math.nan)  # unavailable endpoint: must not consume
    result = audit.run_containment(causal, 1, 2.0,
                                  {"rud_delay": IntervalBounds(0.1, 0.1)}, constants())
    assert result["available_prestate_horizon_s"] == 1.0
    assert [item["decision"] for item in result["endpoints"]] == [1, 2, 3]
    assert [item["horizon_s"] for item in result["endpoints"]] == [0.0, 0.5, 1.0]
    assert result["status"] == "unknown"
    assert result["terminal_poststate_used"] is False
    assert len(toy_tubes.instances[0].calls) == 2
    scored = audit.score_containment(result, scoring)
    assert scored["mission_safety_certificate"] is False
    assert "scoring_truth" not in result["endpoints"][0]  # scorer copies input


def test_only_past_commands_build_actuators_future_measurements_cannot_shrink_tube(toy_tubes):
    causal, _ = audit.strip_trace_rows([raw_row(i) for i in (1, 2, 3, 4)])
    parameters = {"rud_delay": IntervalBounds(0.1, 0.1)}
    expected = audit.run_containment(causal, 2, 1.0, parameters, constants())
    assert toy_tubes.instances[0].history == [causal[0]["command"]]
    poisoned = deepcopy(causal)
    for item in poisoned[2:]:
        item.update(pose=[math.nan] * 3, ego=[math.nan] * 3, fresh=False,
                    truth_scoring_only={"x": 1e100}, outcome="goal")
    changed = audit.run_containment(poisoned, 2, 1.0, parameters, constants())
    assert changed["initial_bounds"] == expected["initial_bounds"]
    assert changed["endpoints"] == expected["endpoints"]
    assert changed["status"] == expected["status"]
    assert toy_tubes.instances[-1].history == [causal[0]["command"]]


def test_one_failed_delay_branch_never_becomes_an_enclosed_union(toy_tubes):
    causal, scoring = audit.strip_trace_rows([raw_row(i) for i in (1, 2, 3)])
    toy_tubes.failing_branch = 3
    result = audit.run_containment(causal, 1, 1.0,
                                  {"rud_delay": IntervalBounds(0.1, 0.15)}, constants())
    assert result["delay_branches"] == [2, 3]
    assert result["status"] == "unknown"
    assert len(result["endpoints"]) == 2
    failed = result["endpoints"][-1]
    assert failed["status"] == "unknown_incomplete_branch_bank"
    assert failed["bounds"] is None and failed["branches_completed"] == 1
    assert audit.score_containment(result, scoring)["endpoints"][-1]["truth_contained"] is None


def test_branch_budget_rejects_whole_bank_instead_of_verifying_subset(toy_tubes):
    causal, _ = audit.strip_trace_rows([raw_row(i) for i in (1, 2)])
    result = audit.run_containment(causal, 1, 0.5,
                                  {"rud_delay": IntervalBounds(0.1, 0.15)}, constants(),
                                  max_branches=1)
    assert result["status"] == "unknown"
    assert result["endpoints"] == []
    assert toy_tubes.instances == []


def test_width_exhaustion_is_unknown_and_never_marked_contained(toy_tubes):
    causal, scoring = audit.strip_trace_rows([raw_row(i) for i in (1, 2)])
    result = audit.run_containment(causal, 1, 0.5,
                                  {"rud_delay": IntervalBounds(0.1, 0.1)}, constants(),
                                  max_position_width_m=0.01)
    assert result["endpoints"][-1]["status"] == "unknown_width_limit"
    assert result["status"] == "unknown"
    assert audit.score_containment(result, scoring)["endpoints"][-1]["truth_contained"] is None


@pytest.mark.parametrize("rudder", [1.1, -1.1, math.nan])
def test_trace_stripping_rejects_unsupported_normalized_command(rudder):
    row = raw_row()
    row["rudder_command"] = rudder
    with pytest.raises(ValueError):
        audit.strip_trace_rows([row])
