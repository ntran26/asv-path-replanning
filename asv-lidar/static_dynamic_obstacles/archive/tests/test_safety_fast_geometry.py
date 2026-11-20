"""Numerical identity of tiled checks, including terminal constraints."""
from types import SimpleNamespace
import numpy as np
import pytest
from classical import common as cc
import safety_fast_geometry as fast
import safety_v2 as v2


@pytest.mark.parametrize("reach", [None,cc.HALF_L+1.])
@pytest.mark.parametrize("points", [0,1,17,257])
def test_point_formula_is_identical(monkeypatch,reach,points):
    rng = np.random.default_rng(123)
    pos,hdg,pts = rng.normal(size=(7,11,2)),rng.normal(size=(7,11)),rng.normal(size=(points,2))*4.
    monkeypatch.setattr(fast,"MAX_BLOCK_ELEMENTS",7*11*3)
    np.testing.assert_array_equal(fast.point_clearance(pos,hdg,pts,reach),cc.point_clearance(pos,hdg,pts,reach))


@pytest.mark.parametrize("targets", [False,True])
@pytest.mark.parametrize("terminal", [False,True])
def test_complete_check_is_identical(monkeypatch,targets,terminal):
    rng = np.random.default_rng(441)
    pos = rng.normal(size=(6,15,2))+[4.,4.]
    headings = rng.normal(size=(6,15))
    ro = cc.Rollout(pos,headings,rng.uniform(0.,1.,size=(6,15)),np.linspace(.125,.75,6))
    edges = np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    snap = SimpleNamespace(points=rng.uniform(0.,10.,size=(41,2)),edges_a=edges,
        edges_b=np.roll(edges,-1,axis=0),tracks=[cc.TrackView(1,np.array([7.,7.]),np.array([.1,-.2]),.3)] if targets else [])
    monkeypatch.setattr(v2,"TERMINAL_BOUNDARY",terminal)
    monkeypatch.setattr(fast,"MAX_BLOCK_ELEMENTS",6*15*2)
    original = v2.SafetyFilterV2._evaluate(None,snap,ro)
    got = fast.evaluate(snap,ro)
    np.testing.assert_array_equal(got[0],original[0])
    np.testing.assert_array_equal(got[1],original[1])


def test_no_reach_points_preserves_infinity():
    pos,hdg=np.zeros((2,3,2)),np.zeros((2,3))
    np.testing.assert_array_equal(fast.point_clearance(pos,hdg,np.array([[100.,100.]]),1.),np.full((2,3),np.inf))
