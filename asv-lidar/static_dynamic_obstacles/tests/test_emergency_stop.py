"""The emergency-stop latch, as a pure state machine.

Environment integration and the 03a T9 stopping-distance test live in
`test_vessel_model.py` and `test_acceptance.py`; this file checks only the
latch, because the latch is the part the deployment bridge will share.
"""

import pytest

import emergency_stop as es


def run_until(latch, speeds, dt=0.1, release_ok=False):
    """Feed a speed sequence; return the S2 override at every step."""
    out, t = [], 0.0
    for speed in speeds:
        t += dt
        out.append(latch.update(t=t, dt=dt, speed=speed, release_ok=release_ok))
    return out


def test_idle_latch_leaves_propulsion_to_the_policy():
    latch = es.EmergencyStop()
    assert latch.update(t=0.1, dt=0.1, speed=1.1) is None
    assert not latch.active


def test_a_request_commands_full_astern_until_the_vessel_stops():
    """The specified manoeuvre: S2 = -100 until speed <= threshold, then 0."""
    latch = es.EmergencyStop(stop_speed=0.05)
    assert latch.request("test", t=0.0, speed=1.1)
    overrides = run_until(latch, [1.0, 0.6, 0.2, 0.04, 0.0, 0.0])
    assert overrides[:3] == [es.S2_FULL_ASTERN] * 3
    assert overrides[3:] == [es.S2_STOP] * 3
    assert latch.state == es.HOLDING
    assert latch.events[-1].t_stopped == pytest.approx(0.4)


def test_the_threshold_is_not_zero():
    """In the field speed is a 2 Hz pose estimate quantised to 0.1 m; a literal
    `speed <= 0` fires on quantisation or never fires on noise."""
    latch = es.EmergencyStop(stop_speed=0.05)
    latch.request("test", t=0.0, speed=1.0)
    assert run_until(latch, [0.049])[-1] == es.S2_STOP


def test_a_repeated_request_is_absorbed_not_restarted():
    """Re-entering BRAKING every step the danger persists would reset the hold
    timer forever, and `max_hold_s` would never release the vessel."""
    latch = es.EmergencyStop(stop_speed=0.05)
    assert latch.request("first", t=0.0, speed=1.0)
    run_until(latch, [0.0])
    assert latch.state == es.HOLDING
    assert not latch.request("second", t=0.2, speed=0.0)
    assert len(latch.events) == 1


def test_holding_releases_only_after_the_minimum_hold_and_a_clear_signal():
    latch = es.EmergencyStop(stop_speed=0.05, min_hold_s=1.0, max_hold_s=10.0)
    latch.request("test", t=0.0, speed=1.0)
    run_until(latch, [0.0])                                     # -> HOLDING
    assert run_until(latch, [0.0] * 5, release_ok=True)[-1] == es.S2_STOP
    assert latch.state == es.HOLDING                            # 0.5 s held
    assert run_until(latch, [0.0] * 6, release_ok=True)[-1] is None
    assert latch.state == es.IDLE
    assert latch.events[-1].t_released is not None


def test_holding_releases_after_the_maximum_even_if_danger_persists():
    """A target stopped dead ahead would otherwise pin the latch for the rest of
    the episode, and every such episode would time out."""
    latch = es.EmergencyStop(stop_speed=0.05, min_hold_s=1.0, max_hold_s=2.0)
    latch.request("test", t=0.0, speed=1.0)
    run_until(latch, [0.0])
    overrides = run_until(latch, [0.0] * 25, release_ok=False)
    assert overrides[-1] is None
    assert latch.state == es.IDLE


def test_braking_gives_up_rather_than_commanding_astern_forever():
    """If reverse thrust is weaker than modelled, or speed telemetry reads high,
    full astern must not continue indefinitely."""
    latch = es.EmergencyStop(stop_speed=0.05, max_brake_s=1.0)
    latch.request("test", t=0.0, speed=1.0)
    overrides = run_until(latch, [0.8] * 12)
    assert es.S2_STOP in overrides
    assert latch.events[-1].gave_up


def test_braking_distance_accumulates_until_stopped():
    latch = es.EmergencyStop(stop_speed=0.05)
    latch.request("test", t=0.0, speed=1.0)
    t = 0.0
    for speed, step in ((0.8, 0.09), (0.4, 0.06), (0.0, 0.02), (0.0, 0.0)):
        t += 0.1
        latch.update(t=t, dt=0.1, speed=speed, distance_step_m=step)
    assert latch.events[-1].distance_braking_m == pytest.approx(0.17)


def test_s2_maps_through_zero_to_rpm_units():
    """The bridge sends S2 = rpm / 24 * 100 forward; full astern is -24 units."""
    assert es.s2_to_rpm(50.0) == pytest.approx(12.0)
    assert es.s2_to_rpm(-100.0) == pytest.approx(-24.0)
    assert es.rpm_to_s2(12.0) == pytest.approx(50.0)
    assert es.rpm_to_s2(es.s2_to_rpm(-37.5)) == pytest.approx(-37.5)


def test_reset_clears_the_latch_and_its_history():
    latch = es.EmergencyStop()
    latch.request("test", t=0.0, speed=1.0)
    latch.reset()
    assert latch.state == es.IDLE and latch.events == []
