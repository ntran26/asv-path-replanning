"""The left telemetry panel (`RENDER_PANEL_SPEC`).

Two things are worth testing about a panel, and neither is how it looks:

* **it is a view on `info`, never a second computation** -- every field it draws
  has to be a key the step already emitted, or the screenshot and the analysis
  can disagree;
* **it renders every block without raising**, on a scene with a target and on
  one without, because a panel that crashes on the empty case is a panel nobody
  leaves switched on.

Rendering runs against SDL's dummy driver, so this needs no display.
"""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

import constants as cfg
from env import ASVLidarEnv, TargetShip
from observation import NORMALISED_SLOT_INDICES, SLOT_FEATURE_NAMES


@pytest.fixture(scope="module")
def render_mod():
    return pytest.importorskip("render")


def head_on_env(steps: int = 40, width: float = 8.0) -> ASVLidarEnv:
    """A displaced head-on, which is the regime that lights every block up."""
    env = ASVLidarEnv(render_mode=None, corridor_width=width)
    env.forced_num_obs = 0
    env.reset(seed=7)
    point, tangent, _ = env.path.frame_at_frac(0.62)
    course = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))
    env.targets = [TargetShip(float(point[0]) + 0.4, float(point[1]),
                              (course + 180.0) % 360.0, 0.9 * cfg.U_CRUISE)]
    env._perceive()
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(np.array([0.12, 0.0], dtype=np.float32))
        if term or trunc:
            break
    return env


# ---------------------------------------------------------------------------
# The panel is a view on `info`
# ---------------------------------------------------------------------------
def test_the_panel_is_carried_in_info_and_is_the_same_object():
    """RENDER_PANEL_SPEC §0.  Same principle as the single `EncounterContext`:
    two consumers, one source, or they diverge."""
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert "panel" in info
    assert info["panel"] is env.last_panel


def test_every_block_is_present_and_populated():
    env = head_on_env()
    panel = env.last_panel
    for block in ("run", "ego", "colregs", "reward", "obs_health", "clearance",
                  "perception"):
        assert block in panel, block
        assert panel[block] is not None, block
    assert len(panel["reward"]["rows"]) == 8
    assert {row["name"] for row in panel["obs_health"]} == set(
        env.observation_space.spaces)


def test_the_panel_agrees_with_the_flat_logging_keys():
    """The display schema and `02a §10.3`'s logging schema are the same step."""
    env = head_on_env()
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    rows = {row["name"]: row for row in info["panel"]["reward"]["rows"]}
    for name, row in rows.items():
        assert row["inst"] == pytest.approx(info[f"reward/term/{name}"])
        assert row["xw"] == pytest.approx(info[f"reward/weighted/{name}"])
    assert info["panel"]["reward"]["step_total"] == pytest.approx(info["reward"])


def test_the_colregs_block_explains_every_sub_term():
    """§1's one idea: show the gate, not just the value.

    When `v_side` reads 0.000 the number alone cannot say whether that is
    correct or a stuck gate.
    """
    env = head_on_env()
    block = env.last_panel["colregs"]
    for name in ("port", "bow", "side", "hold", "r8"):
        assert name in block["terms"]
        assert block["why"].get(name), f"v_{name} has no reason string"
    assert block["sense"] in ("STBD", "PORT", "none")
    assert block["cls"] == "head_on"
    assert block["state"] in ("idle", "engaged", "clearing")


def test_the_admissibility_numbers_explain_the_predicate():
    """§3's point: the three numbers behind `A_stbd` say *why* it flipped, which
    is what is wanted when the agent does something odd near a wall."""
    env = head_on_env(width=8.0)
    block = env.last_panel["colregs"]
    assert block["known"] is True
    expected = block["r_stbd"] >= block["dy_req"] - 0.15
    assert block["a_stbd"] == expected or abs(
        block["r_stbd"] - block["dy_req"]) < 0.15


# ---------------------------------------------------------------------------
# Clip accounting
# ---------------------------------------------------------------------------
def test_clip_accounting_ignores_the_structurally_bounded_dimensions():
    """The panel found this one in itself on its first frame.

    `bearing_cos` reads 1.0 whenever the target is dead ahead, `ct_cos` reads
    -1.0 on a reciprocal course, and the class one-hot and presence bit are
    indicators.  Counting them reported 55% on a branch whose normalised
    features were healthy -- and buried the one that was not.
    """
    excluded = {"bearing_sin", "bearing_cos", "ct_sin", "ct_cos", "presence"}
    excluded |= {n for n in SLOT_FEATURE_NAMES if n.startswith("class_")}
    kept = {SLOT_FEATURE_NAMES[i] for i in NORMALISED_SLOT_INDICES}
    assert kept.isdisjoint(excluded)
    assert kept == {"distance_to_domain", "target_speed", "relative_speed",
                    "dcpa", "tcpa", "cri"}


def test_a_saturating_dimension_is_named():
    """§6 asks for a drill-down, because one saturating dimension inside a
    27-dim branch will not move the branch aggregate much.

    70 steps rather than the fixture default: `cri` pins at 1.0 only once the
    target is inside the domain, so a shorter run sits just under the warning
    threshold and would make this test about the fixture length instead of the
    attribution.
    """
    env = head_on_env(steps=70)
    rows = {row["name"]: row for row in env.last_panel["obs_health"]}
    target = rows["target"]
    assert target["clip"] > 0.10, "this scenario should saturate `cri`"
    assert target["worst"] == "cri"
    assert target["worst_clip"] == pytest.approx(target["clip"], abs=0.05)


def test_a_healthy_branch_reports_no_clipping():
    env = head_on_env()
    rows = {row["name"]: row for row in env.last_panel["obs_health"]}
    assert rows["ego"]["clip"] == 0.0
    assert rows["path"]["clip"] == 0.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_every_block_renders_without_raising(render_mod):
    env = head_on_env()
    renderer = render_mod.Renderer(env.map_width, env.map_height)
    renderer.blocks = set(render_mod.BLOCKS)
    renderer.draw(env)
    assert renderer.window_size[0] > render_mod.PANEL_WIDTH
    renderer.close()


def test_the_panel_renders_with_no_target_at_all(render_mod):
    """The empty case: a panel that crashes here is one nobody leaves on."""
    env = ASVLidarEnv(render_mode=None, no_target_prob=1.0)
    env.forced_num_obs = 0
    env.reset(seed=1)
    for _ in range(5):
        env.step(np.zeros(2, dtype=np.float32))
    assert env.last_panel["colregs"] is None
    assert env.last_panel["perception"] is None

    renderer = render_mod.Renderer(env.map_width, env.map_height)
    renderer.blocks = set(render_mod.BLOCKS)
    renderer.draw(env)
    renderer.close()


def test_the_field_is_offset_by_the_panel_width(render_mod):
    """The field moved right rather than shrinking, so the metres-per-pixel
    scale is unchanged and 04's screenshots stay comparable to Paper 2's."""
    env = ASVLidarEnv(render_mode=None)
    renderer = render_mod.Renderer(env.map_width, env.map_height)
    assert renderer.to_screen((0.0, env.map_height))[0] == render_mod.PANEL_WIDTH
    assert renderer.scale == cfg.RENDER_SCALE
    renderer.close()


def test_blocks_toggle_and_default_to_the_reward_pair(render_mod):
    """§5: default to [4] and [5] only, so the panel is readable in a short
    window and the rest is opt-in."""
    env = ASVLidarEnv(render_mode=None)
    renderer = render_mod.Renderer(env.map_width, env.map_height)
    assert renderer.blocks == {"colregs", "reward"}
    renderer.toggle_index(3)
    assert "perception" in renderer.blocks
    renderer.toggle_index(3)
    assert "perception" not in renderer.blocks
    renderer.toggle_index(99)          # out of range, ignored
    renderer.close()


def test_scrub_walks_the_ring_buffer_and_clamps(render_mod):
    """§5's step-back key.  Nearly every question worth asking about an
    encounter is "what was the state four seconds ago"."""
    env = head_on_env(steps=25)
    renderer = render_mod.Renderer(env.map_width, env.map_height)
    for _ in range(25):
        env.step(np.zeros(2, dtype=np.float32))
        renderer.draw(env)

    assert len(renderer.history) > 10
    renderer.scrub_by(+5)
    assert renderer.scrub == 5
    renderer.draw(env)                     # renders the scrubbed frame
    renderer.scrub_by(-99)
    assert renderer.scrub == 0
    renderer.scrub_by(+10_000)
    assert renderer.scrub == len(renderer.history) - 1
    renderer.close()


def test_the_history_is_bounded(render_mod):
    env = ASVLidarEnv(render_mode=None)
    env.reset(seed=0)
    renderer = render_mod.Renderer(env.map_width, env.map_height)
    assert renderer.history.maxlen == render_mod.HISTORY
    renderer.close()


def test_angle_errors_are_wrapped_on_the_circle(render_mod):
    """A bearing estimate of -7.2 against a truth of 330.7 is a 22 degree error,
    not a 338 degree one, and the naive subtraction made the tracker look broken
    on the first frame the panel was ever looked at."""
    assert render_mod._wrap180(-7.2 - 330.7) == pytest.approx(22.1, abs=0.1)
    assert render_mod._wrap180(0.0) == 0.0
    assert render_mod._wrap180(181.0) == pytest.approx(-179.0)


def test_infinite_clearances_render_as_inf(render_mod):
    """`target hull inf` is the honest reading when no track is paired with a
    true target; formatting it as a number would invent a measurement."""
    assert render_mod._fin(float("inf")).strip() == "inf"
    assert render_mod._fin(2.345) == "2.35"
