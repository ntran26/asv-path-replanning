"""Behavioural regressions for encounter release and effective Rule 8 action."""

import pytest
import numpy as np
from types import SimpleNamespace

import constants as cfg
import encounter as enc
from colregs.context import CLEARING, ENGAGED, IDLE, ContextManager, EncounterContext
from reward import RewardConfig
from reward import terms


def _advance(manager, *, tcpa, dcpa, rng=5.0, cls=enc.HEAD_ON, heading=0.0, u=0.55,
             crossing_side=enc.SIDE_NONE):
    manager.step_index += 1
    context = EncounterContext(track_id=1, cls=cls, tcpa=tcpa, dcpa=dcpa, rng=rng,
                               crossing_side=crossing_side)
    manager._advance_state(context, heading, u, d_required=1.0)
    return context


@pytest.mark.parametrize("safe_tcpa,safe_dcpa", [(-1.0, 0.2), (4.0, 3.0)])
def test_renewed_risk_during_clearing_restores_original_obligation(safe_tcpa, safe_dcpa):
    manager = ContextManager(n_clear=3)
    engaged = _advance(manager, tcpa=8.0, dcpa=0.2, heading=10.0)
    assert engaged.state == ENGAGED
    clearing = _advance(manager, tcpa=safe_tcpa, dcpa=safe_dcpa, heading=30.0)
    assert clearing.state == CLEARING
    # Geometry drift would now call this a port crossing.  It must not erase
    # the original head-on obligation or refresh the action reference.
    restored = _advance(manager, tcpa=2.0, dcpa=0.1, cls=enc.CROSSING,
                        crossing_side=enc.SIDE_PORT, heading=40.0, u=0.1)
    assert restored.state == ENGAGED and restored.engaged
    assert restored.cls == enc.HEAD_ON
    assert restored.compliant_turn_sense == 1
    assert restored.psi_engage == 10.0
    assert restored.u_engage == 0.55
    assert restored.t_engage == engaged.t_engage


def test_clearance_confirmation_requires_consecutive_safe_frames():
    manager = ContextManager(n_clear=3)
    _advance(manager, tcpa=8.0, dcpa=0.2)
    _advance(manager, tcpa=-1.0, dcpa=0.2)
    _advance(manager, tcpa=-1.0, dcpa=0.2)
    _advance(manager, tcpa=1.0, dcpa=0.2)
    assert _advance(manager, tcpa=-1.0, dcpa=0.2).state == CLEARING
    for _ in range(2):
        assert _advance(manager, tcpa=-1.0, dcpa=0.2).state == CLEARING
    assert _advance(manager, tcpa=-1.0, dcpa=0.2).state == IDLE


@pytest.mark.parametrize("tcpa,dcpa,rng", [(4.0, 3.0, 5.0), (-0.1, 0.2, 0.5)])
def test_release_requires_opening_outside_required_separation(tcpa, dcpa, rng):
    manager = ContextManager(n_clear=3)
    first = _advance(manager, tcpa=8.0, dcpa=0.2)
    for _ in range(6):
        context = _advance(manager, tcpa=tcpa, dcpa=dcpa, rng=rng)
        assert context.state == CLEARING
        assert context.t_engage == first.t_engage
    for _ in range(3):
        context = _advance(manager, tcpa=-2.0, dcpa=0.2, rng=2.0)
    assert context.state == IDLE


def test_slowing_on_reciprocal_collision_course_does_not_discharge_rule8():
    config = RewardConfig()
    context = EncounterContext(
        track_id=1, cls=enc.HEAD_ON, state=ENGAGED, engaged=True,
        compliant_turn_sense=1, a_stbd=False, a_req=1.0,
        psi_engage=0.0, u_engage=cfg.U_REF, u_own=0.0,
        rng=4.0, alpha=0.0, ct=180.0, speed_ts=cfg.U_REF, tcpa=2.0)
    holding = terms.RewardState(heading_deg=0.0, u=cfg.U_REF)
    stopped = terms.RewardState(heading_deg=0.0, u=0.0)
    assert not context.slowdown_clears
    before = terms.r8_parts(holding, context, config)
    after = terms.r8_parts(stopped, context, config)
    assert after["du_red"] > config.du_min
    assert after["slowdown_credit"] == 0.0
    assert after["value"] == pytest.approx(before["value"])
    assert after["value"] > 0.0


def test_effective_crossing_slowdown_can_discharge_rule8():
    config = RewardConfig()
    context = EncounterContext(
        track_id=1, cls=enc.CROSSING, state=ENGAGED, engaged=True,
        compliant_turn_sense=1, a_stbd=False, a_req=1.0,
        psi_engage=0.0, u_engage=cfg.U_REF, u_own=0.0,
        rng=4.0, alpha=45.0, ct=270.0, speed_ts=cfg.U_REF, tcpa=2.0)
    assert context.slowdown_clears
    holding = terms.r8_parts(terms.RewardState(u=cfg.U_REF), context, config)
    slowed = terms.r8_parts(terms.RewardState(u=0.0), context, config)
    assert holding["value"] > 0.0
    assert slowed["slowdown_credit"] > 1.0
    assert slowed["value"] == 0.0


def test_rule8_and_standon_compare_perceived_motion_to_perceived_latches():
    config = RewardConfig()
    context = EncounterContext(
        track_id=1, cls=enc.HEAD_ON, state=ENGAGED, engaged=True,
        compliant_turn_sense=1, a_stbd=True, a_req=1.0,
        psi_engage=0.0, u_engage=cfg.U_REF, u_own=cfg.U_REF,
        rng=4.0, alpha=0.0, ct=180.0, speed_ts=cfg.U_REF, tcpa=2.0)
    state = terms.RewardState(heading_deg=90.0, u=0.0,
                              perceived_heading_deg=0.0, perceived_u=cfg.U_REF)
    parts = terms.r8_parts(state, context, config)
    assert parts["dpsi_c"] == 0.0
    assert parts["du_red"] == 0.0
    assert parts["value"] > 0.0
    context.cls = enc.BEING_OVERTAKEN
    context.rho = 1.0
    assert terms.v_hold(state, context, config) == 0.0
    # Physical speed scoring continues to see the stopped physical vessel.
    assert terms.r_pf(state, {}, config) == -1.0


def test_truth_diagnostics_use_physical_own_ship_without_changing_obligation():
    manager = ContextManager()
    track = SimpleNamespace(id=1, position=np.array([5.0, 10.0]),
                            velocity=np.array([0.0, -1.0]), course_deg=180.0, speed=1.0)
    target = SimpleNamespace(x=5.0, y=10.0, velocity=np.array([0.0, -1.0]),
                             heading_deg=180.0, speed=1.0)
    context = manager.update(tracks=[track], p_os=(7.0, 2.0), v_os=(0.0, 1.0),
                              heading_os_deg=0.0, u_os=1.0)[1]
    perceived = (context.rng, context.alpha, context.dcpa, context.cls)
    manager.attach_truth(tracks=[track], p_os=(5.0, 0.0), v_os=(0.0, 1.0),
                         heading_os_deg=0.0, true_targets=[target])
    assert context.d_ts_true == pytest.approx(10.0)
    assert context.dcpa_true == pytest.approx(0.0)
    assert context.cls_true == enc.HEAD_ON
    assert (context.rng, context.alpha, context.dcpa, context.cls) == perceived
