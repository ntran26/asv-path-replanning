"""Basin mode (06), as amended by your calls in F74.

T13-T15 from 06 §8, T3 and T7 per mode, the default geometry, the
side-specific normalisation, and the static-feasibility filter.
"""

import math

import numpy as np
import pytest

import boundary_raycast as br
import constants as cfg
import corridor as corr
import feasibility as feas
import scenario as scn
import targets as tgt
from env import ASVLidarEnv
from reward import RewardConfig
from reward import terms as T

CLASSES = ("head_on", "crossing", "overtaking", "being_overtaken", "null", "no_target")


def _basin_draws(n, *, cls=None, stage=5, base=0):
    generator = scn.ScenarioGenerator(stage=stage)
    out = []
    for i in range(n):
        built = generator.sample(scn.seed_for("training", base + i), encounter_class=cls,
                                 geometry_mode="basin")
        if built is not None:
            out.append(built)
    return out


# ---------------------------------------------------------------------------
# T13 -- the leg
# ---------------------------------------------------------------------------
def test_t13_basin_legs_start_and_end_clear_of_the_walls_and_record_their_slant():
    nav = corr.nav_polygon()
    eroded = corr.nav_polygon(inset=cfg.BASIN_NAV_INSET_M + 1.5)
    slant_cap = math.degrees(math.atan2(cfg.BASIN_X_RANGE[1] - cfg.BASIN_X_RANGE[0],
                                        cfg.BASIN_GOAL_Y - cfg.BASIN_START_Y))
    for seed in range(200):
        leg = corr.sample_basin(np.random.default_rng(seed))
        for p in (leg.centre[0], leg.centre[-1]):
            assert br.point_in_polygon(float(p[0]), float(p[1]), eroded)
        assert leg.centre[0][1] == pytest.approx(cfg.BASIN_START_Y)
        assert leg.centre[-1][1] == pytest.approx(cfg.BASIN_GOAL_Y)
        assert abs(leg.slant_realised_deg) <= slant_cap + 1e-9
        assert leg.polygon() == nav


def test_t13_a_stage_slant_cap_clamps_and_records_both_slants():
    capped = [corr.sample_basin(np.random.default_rng(s), slant_max_deg=6.0) for s in range(100)]
    assert all(abs(leg.slant_realised_deg) <= 6.0 + 1e-9 for leg in capped)
    clamped = [leg for leg in capped if abs(leg.slant_requested_deg) > 6.0 + 1e-9]
    assert clamped, "no draw exceeded the cap, so the clamp was not exercised"
    assert all(abs(leg.slant_realised_deg) == pytest.approx(6.0) for leg in clamped)


def test_clearances_are_affine_with_opposite_slopes_on_a_slanted_leg():
    """06 §3.3: the property the boundary branch needs."""
    leg = corr.build_basin((2.5, cfg.BASIN_START_Y), (7.5, cfg.BASIN_GOAL_Y))
    s = np.linspace(2.0, leg.length - 2.0, 5)
    hp = np.array([leg.clearances_at_s(v)[0] for v in s])
    hm = np.array([leg.clearances_at_s(v)[1] for v in s])
    assert np.all(np.diff(hp) < 0) and np.all(np.diff(hm) > 0)
    assert np.allclose(np.diff(hp, 2), 0.0, atol=1e-6)
    assert np.allclose(hp + hm, [leg.width_at_s(v) for v in s])


# ---------------------------------------------------------------------------
# T14 -- channel mode is unchanged
# ---------------------------------------------------------------------------
def test_t14_channel_normalisation_is_exactly_the_half_width():
    channel = corr.rectangle(6.0)
    for s in (2.0, 10.0, 20.0):
        for e_y in (-1.0, 0.0, 1.0):
            assert channel.half_width_on_side(s, e_y) == 0.5 * channel.width_at_s(s)
            assert channel.clearances_at_s(s) == (0.5 * channel.width_at_s(s),) * 2


def test_t14_channel_env_reports_the_channel_width_to_the_reward():
    env = ASVLidarEnv(render_mode=None, channel=corr.rectangle(6.0))
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.array([0.3, 0.0], dtype=np.float32))
        assert env.local_channel_width() == pytest.approx(6.0)


def test_basin_normalises_on_the_side_of_the_deviation():
    leg = corr.build_basin((2.5, cfg.BASIN_START_Y), (7.5, cfg.BASIN_GOAL_Y))
    s = 5.0
    h_plus, h_minus = leg.clearances_at_s(s)
    assert leg.half_width_on_side(s, +0.5) == pytest.approx(np.clip(h_plus, *cfg.BASIN_H_SIDE_CLIP))
    assert leg.half_width_on_side(s, -0.5) == pytest.approx(np.clip(h_minus, *cfg.BASIN_H_SIDE_CLIP))
    assert leg.half_width_on_side(s, +0.5) != pytest.approx(leg.half_width_on_side(s, -0.5))


# ---------------------------------------------------------------------------
# T15 -- both hulls stay in the water
# ---------------------------------------------------------------------------
def test_t15_basin_targets_stay_inside_the_basin_envelope():
    """06 §8 asks for 1000 episodes; 120 draws cover every class here."""
    envelope = br.rectangle(cfg.MAP_WIDTH, cfg.MAP_HEIGHT)
    draws = []
    for cls in CLASSES[:-1]:
        draws += _basin_draws(24, cls=cls, base=hash(cls) % 10_000)
    assert len(draws) >= 100
    for built in draws:
        # The generator's guarantee is to CPA (null: its own check window);
        # after that the environment's clamp keeps confined traffic in, which
        # the episode test below covers.
        horizon = (float(cfg.NULL_TRACK_CHECK_S) if built.encounter_class == "null"
                   else max(float(built.tcpa_s), 0.0))
        v = float(built.target_speed) * np.array([math.sin(math.radians(built.target_heading)),
                                                  math.cos(math.radians(built.target_heading))])
        for t in np.arange(0.0, horizon + 1e-9, 0.5):
            x, y = built.target_spawn[0] + v[0] * t, built.target_spawn[1] + v[1] * t
            hull = np.asarray(tgt.hull_polygon(x, y, built.target_heading))
            assert np.all(br.points_in_polygon(hull[:, 0], hull[:, 1], envelope)), (
                built.encounter_class, built.seed, t)


def test_t15_confined_basin_traffic_stays_in_the_basin_through_an_episode():
    envelope = br.rectangle(cfg.MAP_WIDTH, cfg.MAP_HEIGHT)
    env = ASVLidarEnv(render_mode=None, scenario_stage=5)
    seen = 0
    for seed in range(40):
        env.reset(seed=seed)
        if env.channel.mode != "basin" or not env.targets or not env.targets[0].confined:
            continue
        seen += 1
        for _ in range(80):
            _, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
            # The side walls, not the ends: traffic running with the leg leaves
            # the basin through its far end, as it leaves a channel, and is
            # gated out of the scan when it does.
            hull = np.asarray(env.targets[0].hull())
            if hull[:, 1].min() > cfg.MAP_HEIGHT or hull[:, 1].max() < 0.0:
                break
            assert hull[:, 0].min() >= 0.0 and hull[:, 0].max() <= cfg.MAP_WIDTH, seed
            if term or trunc:
                break
    assert seen >= 5


# ---------------------------------------------------------------------------
# T3 per mode (06 §8, amended)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["basin", "channel"])
def test_t3_boundary_rays_decorrelate_in_each_mode(mode):
    env = ASVLidarEnv(render_mode=None, geometry_mode=mode)
    env.forced_num_obs = 0
    cte, rays = [], []
    rng = np.random.default_rng(0)
    for seed in range(40):
        env.reset(seed=seed)
        rudder = float(rng.uniform(-0.3, 0.3))
        for _ in range(30):
            _, _, term, trunc, info = env.step(np.array([rudder, 0.0], dtype=np.float32))
            cte.append(info["cross_track_error"])
            rays.append(np.asarray(env.boundary_closeness, dtype=float).copy())
            if term or trunc:
                break
    cte, rays = np.asarray(cte), np.asarray(rays)
    worst = max(abs(float(np.corrcoef(cte, rays[:, i])[0, 1]))
                for i in range(rays.shape[1]) if rays[:, i].std() > 1e-9)
    assert worst < cfg.BOUNDARY_DECORRELATION_MAX, (mode, worst)


# ---------------------------------------------------------------------------
# T7 per mode, and the confinement rule you set
# ---------------------------------------------------------------------------
def test_t7_basin_head_on_traffic_keeps_the_band_and_the_rest_keep_the_basin():
    leg = corr.build_basin((3.0, cfg.BASIN_START_Y), (6.0, cfg.BASIN_GOAL_Y))
    assert scn.confinement_geometry("head_on", leg) is leg.band()
    for cls in ("overtaking", "being_overtaken", "null", "crossing"):
        assert scn.confinement_geometry(cls, leg) is leg
    channel = corr.rectangle(6.0)
    for cls in CLASSES:
        assert scn.confinement_geometry(cls, channel) is channel


def test_t7_basin_head_on_spawns_inside_the_band():
    for built in _basin_draws(20, cls="head_on", base=300):
        band = built.channel.band()
        assert br.point_in_polygon(built.target_spawn[0], built.target_spawn[1], band.polygon())


# ---------------------------------------------------------------------------
# The default geometry and the class mix
# ---------------------------------------------------------------------------
def test_basin_is_the_default_geometry():
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=0)
    assert env.channel.mode == "basin"
    assert ASVLidarEnv(render_mode=None, corridor_width=6.0).channel.mode == "channel"


def test_channel_draws_carry_only_the_width_governed_rules():
    generator = scn.ScenarioGenerator(stage=5)
    modes = {}
    for i in range(240):
        built = generator.sample(scn.seed_for("training", 20_000 + i))
        if built is not None:
            modes.setdefault(built.encounter_class, set()).add(built.geometry_mode)
    for cls in ("being_overtaken", "null", "no_target"):
        assert modes.get(cls, {"basin"}) == {"basin"}, cls
    assert any("channel" in modes.get(c, set()) for c in cfg.CHANNEL_CLASSES)


def test_every_class_can_be_drawn_in_basin_mode():
    for cls in CLASSES:
        assert _basin_draws(6, cls=cls, base=40_000), cls


def test_basin_records_carry_the_06_fields():
    built = _basin_draws(1, cls="head_on", base=50)[0]
    record = built.to_record()
    for key in ("geometry_mode", "slant_requested_deg", "slant_realised_deg",
                "path_midpoint", "clearance_profile", "w_eff_at_cpa", "field_replicable"):
        assert key in record
    assert record["geometry_mode"] == "basin" and record["field_replicable"]
    assert record["suite_version"] == "3.0"


# ---------------------------------------------------------------------------
# Static feasibility
# ---------------------------------------------------------------------------
def test_a_gate_narrower_than_the_hull_has_no_route_and_a_wide_one_does():
    nav = corr.nav_polygon()
    def gate(gap):
        half = 0.5 * gap
        return [[(0.3, 11.0), (5.0 - half, 11.0), (5.0 - half, 12.0), (0.3, 12.0)],
                [(5.0 + half, 11.0), (9.7, 11.0), (9.7, 12.0), (5.0 + half, 12.0)]]
    assert not feas.layout_feasible((5.0, 2.0), (5.0, 22.0), nav, gate(0.8))
    assert feas.layout_feasible((5.0, 2.0), (5.0, 22.0), nav, gate(1.6))


def test_thinning_removes_the_blocking_panel_first():
    nav = corr.nav_polygon()
    blocking = [[(0.3, 11.0), (9.7, 11.0), (9.7, 12.0), (0.3, 12.0)]]
    aside = [[(8.5, 5.0), (9.3, 5.0), (9.3, 6.0), (8.5, 6.0)]]
    path = np.column_stack([np.full(50, 5.0), np.linspace(2.0, 22.0, 50)])
    kept = feas.thin_to_feasible((5.0, 2.0), (5.0, 22.0), nav, aside + blocking, path)
    assert kept == aside


def test_every_generated_layout_leaves_a_route():
    env = ASVLidarEnv(render_mode=None, scenario_stage=5)
    for seed in range(40):
        env.reset(seed=seed)
        assert feas.layout_feasible((env.start_x, env.start_y), (env.goal_x, env.goal_y),
                                    env.boundary_polygon, env.obstacles), seed
