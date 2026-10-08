"""The test set: the frozen suite and the Paper 2 deployment-layout set merged,
twins and near-duplicates trimmed, 1,000 episodes (decision, 2026-10-02).

Sources (neither used for training or tuning):

* the **frozen suite** (suite 3.4): 800 constant-velocity headline scenarios in
  8 cells (basin: head-on, crossing, overtaking, being overtaken, null;
  channel: head-on, crossing, overtaking), plus the robustness set -- the same
  scenarios and seeds rerun with a reactive target (700) and, in head-ons, a
  non-compliant one (200);
* the **Paper 2 set** (revision 2.0): layouts L1-L3 x 5 encounters x 20, each a
  fixed-speed scenario with its varying-speed twin (600), plus 10 no-target runs
  per layout (30), which differ only in sensor noise.

Trimming, in two steps:

1. **Twins collapse.** A robustness variant is its headline scenario with only
   the target's behaviour changed; a varying-speed twin is its fixed-speed
   scenario with only the target's speed profile changed; the no-target runs of
   a layout share their geometry.  Each underlying scenario is kept once, with
   its variant rotating across the cell (head-on: CV, RE, NC; other target
   classes: CV, RE; Paper 2: FIX, VAR), so every behaviour and speed stays
   represented.  2,330 episodes -> 1,103 distinct scenarios (800 + 300 + 3).
2. **Near-duplicates go.** Each scenario becomes a feature vector (own start
   pose, path midpoint, slant and bend, channel width; target spawn, heading,
   speed, DCPA, TCPA, crossing angle; obstacle centres), standardised over the
   pool.  The scenario whose nearest neighbour *in its own cell* is closest is
   removed, repeatedly, until 1,000 remain; no cell loses more than
   `MAX_CELL_CUT` of its scenarios.

Every episode keeps its original episode seed, so a test-set result can be
checked against the policy's earlier frozen-suite and Paper 2 rows.

**Version 2 (2026-10-02).** v1 held 15 episodes no policy could pass: in the
basin a being-overtaken own ship starts `BASIN_BEING_OVERTAKEN_START_S` (6.9 m)
along its leg -- on L1 that is on the (5, 8) panel, so every L1-BO episode
collided at its first step (the Paper 2 set's 40 L1-BO episodes too).  v2 is v1
with every episode whose own hull starts in contact with a panel replaced
(regenerated ones must start `START_GAP_M` clear): the L1-BO cell is regenerated with the start fixed by
the scenario (`flags["own_start_s"]` = `L1_BO_START_S`, honoured by
`scenario.own_start_s`) under the Paper 2 field rules plus the start check,
rotated FIX/VAR and trimmed to the same count.  All other episodes are v1's.
Results in `results/test_set/v2/<tag>/`.

**Version 3 (2026-10-03, decision: field deployment runs boustrophedon
survey lanes, so straight legs are the common case; vary the obstacle
arrangements more, keep some slanted legs).** v3 is v2 with:
* in each Paper 2 layout cell (L1-L3 x HO/CRP/CRS/OT/BO), a share `V3_FIELD_REPLACE`
  (L1 1/3; L2, L3 2/3) of the episodes replaced by **new three-panel arrangements**
  (`FS-<code>` cells):
  `field_training.sample` layouts (Paper 2-style motifs, never near L1-L3, an A*
  route, the field rules, space-time solvable), matching encounter and FIX/VAR,
  legs `V3_FS_STRAIGHT` straight / the rest slanted, own hull start-clear;
* `V3_FROZEN_REPLACE` of the slanted frozen-basin episodes that keep all three
  panels at reset replaced by new straight-leg episodes of the same class and
  target behaviour that also do (panels placed at reset as in the frozen suite --
  the CPA guard drops panels near the encounter, so of 130 basin episodes asking
  for three only 31 keep them), start-clear and space-time solvable.
* for balance: `V3_NT_ADD` no-target and `V3_OT_DENSE` dense overtaking field-style
  episodes in place of overtaking episodes with no static obstacle (175 of v2's
  238 overtaking episodes had none).
Everything else is v2's; the development sets are untouched.  Default
`--version 3`; results in `results/test_set/v3/<tag>/`.

**Version 4 (2026-10-04, decision: near-impossible episodes leave the test set).**
`src/oracle_feasibility.py` grades every v3 episode with a manoeuvre library rolled
out under perfect foresight, the true vessel model and the policy's action space
(`results/feasibility/oracle_test.csv`).  An episode stays only if some manoeuvre
that starts at or after the onboard perception first tracks the target reaches
the goal while keeping `V4_MARGIN_M` clearance throughout (`margin_after_track`).
The 67 that fail -- tight (solvable only under the margin, 35), decided before the
target can be seen (25) and unsolvable (7) -- are each replaced by a fresh episode
of the same cell, target behaviour or speed profile and leg type, drawn from new
seed blocks (`V4_*`) and accepted only if it passes the same oracle test.  The
1,000 total and the cell balance stay as designed; everything else is v3's.
Run with `--version 4`; results in `results/test_set/v4/<tag>/`.

    python tools/tiers/test_set.py --build                      # write the definition
    python tools/tiers/test_set.py --model runs/.../best_model.zip --tag sacs0_bl3 --safety off

Writes `results/test_set/definition.csv` (+ `definition.json`: digest and
trimming report) and, per evaluation, `results/test_set/<tag>/episodes.csv`
and `summary.txt`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import paper2_set as p2  # noqa: E402
import suite  # noqa: E402
import train_formulation as tf  # noqa: E402
from common import run_pool  # noqa: E402

OUT = ROOT / "results" / "test_set"
TARGET_EPISODES = 1000
MAX_CELL_CUT = 0.25
TIER_B_SEED = 400_000                 # frozen_suite.py's headline seed base
SET_VERSION = "1.0"                   # the version being built or evaluated (set by --version)
START_GAP_M = 0.30                    # v2: a regenerated episode's own hull starts this clear of every panel
L1_BO_START_S = 8.0                   # v2: L1 being-overtaken starts 8 m along the leg (y = 10),
                                      # its stern 0.6 m past the (5, 8) panel
CELL_FIXES = {("L1", "BO"): {"own_start_s": L1_BO_START_S}}
V3_FIELD_REPLACE = {"L1": 1 / 3, "L2": 2 / 3, "L3": 2 / 3}   # v3: share of each Paper 2 layout cell given a
                                      # new arrangement (L2/L3 legs are slanted, L1's straight)
V3_FS_STRAIGHT = 0.75                 # v3: of the new field-style episodes, straight legs
V3_FROZEN_REPLACE = 17                # v3: of the 25 slanted frozen-basin episodes that keep all three
                                      # panels at reset, this many replaced by straight ones that also do
V3_NT_ADD = 27                        # v3 balance: no-target field-style episodes added (3 -> 30) ...
V3_OT_DENSE = 70                      # ... and dense field-style overtaking, both in place of overtaking
                                      # episodes with no static obstacle at all (175 of 238 in v2)
V3_FS_SEED_BASE = 450_000             # v3 generator seeds (seed_fn adds 200,000): no other set uses them
V3_FS_EPISODE_SEED = 496_000
V3_FROZEN_SEED_INDEX = 8_500          # frozen_eval namespace index (Tier B uses 0-7,999)
V3_FROZEN_EPISODE_SEED = 497_000
V4_ORACLE = ROOT / "results" / "feasibility" / "oracle_test.csv"   # v4: the oracle verdicts on v3
V4_MARGIN_M = 0.20                    # v4: the clearance a post-tracking solution must keep (oracle MARGIN_M)
V4_FS_SEED_BASE = 600_000             # v4 replacement seed blocks, unused by every other set
V4_FS_EPISODE_SEED = 498_000
V4_FROZEN_SEED_INDEX = 10_500         # frozen_eval index (Tier B 0-7,999; v3 8,500+; Tier A 9,600-9,999)
V4_FROZEN_EPISODE_SEED = 499_000
V4_P2_SEED_BASE = 342_000             # + cell index x 1,000 (the Paper 2 set uses 310,000-339,999)
V4_P2_EPISODE_SEED = 499_500
V4_CANDIDATES = 2                     # replacement candidates drawn per open slot and round
V4_ROUNDS = 8


def out_dir(version=None):
    version = version or SET_VERSION
    return OUT if version.startswith("1") else OUT / ("v" + version.split(".")[0])


def _features(b) -> np.ndarray:
    """One fixed-length vector for any scenario (absent parts are zeros)."""
    w, h = float(cfg.MAP_WIDTH), float(cfg.MAP_HEIGHT)
    oh = math.radians(float(b.own_heading))
    f = [b.own_spawn[0] / w, b.own_spawn[1] / h, math.sin(oh), math.cos(oh),
         b.path_midpoint[0] / w, b.path_midpoint[1] / h,
         float(b.slant_realised_deg) / 45.0, float(b.bend_deg) / 90.0,
         float(b.nominal_width) / 10.0]
    has_target = b.encounter_class not in ("null", "no_target") and float(b.target_speed) > 0.0
    if has_target:
        th = math.radians(float(b.target_heading))
        f += [b.target_spawn[0] / w, b.target_spawn[1] / h, math.sin(th), math.cos(th),
              float(b.target_speed), float(b.dcpa_m) / 3.0, float(b.tcpa_s) / 30.0,
              float(b.ct_deg) / 180.0]
    else:
        f += [0.0] * 8
    centres = sorted((float(np.mean([p[0] for p in poly])), float(np.mean([p[1] for p in poly])))
                     for poly in (b.obstacles or ()))
    centres = sorted(centres, key=lambda c: c[1])[:3]
    for k in range(3):
        f += ([centres[k][0] / w, centres[k][1] / h, 1.0] if k < len(centres) else [0.0, 0.0, 0.0])
    return np.asarray(f, dtype=float)


def _distinct_scenarios(env):
    """Step 1: one entry per underlying scenario, the variant rotating in its cell."""
    items = []
    tier_b, short_b = suite.build_tier_b()
    cells = suite.tier_b_cells()
    variants = {}
    for v, twin, behaviour in suite.robustness_variants(tier_b):
        variants.setdefault(twin, {})[behaviour] = v
    for i, b in enumerate(tier_b):
        c_index, n = int(b.case_id.split("-")[1]), int(b.case_id.split("-")[2])
        cell = cells[c_index]
        options = [("cv", b)] + sorted(variants.get(i, {}).items())
        behaviour, scen = options[n % len(options)]
        items.append({"built": scen, "episode_seed": TIER_B_SEED + i, "source": "frozen",
                      "cell": f"{cell['stratum']}-{cell['class']}", "class": cell["class"],
                      "stratum": cell["stratum"], "variant": behaviour.upper(),
                      "test_id": suite.test_id(scen.case_id), "origin_id": suite.test_id(b.case_id)})
    records, short_p = p2.build(env)
    pair = Counter()
    for r in records:
        if r["encounter"] == "NT":
            if not r["test_id"].endswith("-01"):
                continue                                  # 10 noise realisations of one geometry
            variant = ""
        else:
            k = int(r["test_id"].rsplit("-", 1)[1]) - 1    # pair index within the cell
            want = "FIX" if k % 2 == 0 else "VAR"
            if r["speed"] != want:
                continue
            variant = want
        items.append({"built": r["built"], "episode_seed": r["episode_seed"], "source": "paper2",
                      "cell": f"{r['layout']}-{r['encounter']}", "class": r["built"].encounter_class,
                      "stratum": "field", "variant": variant, "test_id": r["test_id"],
                      "origin_id": r["twin"] or r["test_id"]})
    return items, {"frozen_shortfall": short_b, "paper2_shortfall": short_p,
                   "source_episodes": {"frozen": 800 + 700 + 200, "paper2": len(records)}}


def _trim(items, target=TARGET_EPISODES, max_cut=MAX_CELL_CUT):
    """Step 2: drop the most redundant scenario (closest same-cell neighbour) until `target`."""
    X = np.stack([_features(it["built"]) for it in items])
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
    cells = np.array([it["cell"] for it in items])
    size = Counter(cells)
    cut = Counter()
    alive = np.ones(len(items), dtype=bool)
    D = np.full((len(items), len(items)), np.inf)
    for c in size:
        idx = np.flatnonzero(cells == c)
        sub = np.linalg.norm(X[idx, None, :] - X[None, idx, :], axis=-1)
        np.fill_diagonal(sub, np.inf)
        D[np.ix_(idx, idx)] = sub
    nn = D.min(axis=1)
    removed = []
    while alive.sum() > target:
        cand = np.where(alive & np.array([cut[c] < int(max_cut * size[c]) for c in cells]), nn, np.inf)
        j = int(np.argmin(cand))
        if not np.isfinite(cand[j]):
            break
        alive[j] = False
        cut[cells[j]] += 1
        removed.append((items[j]["test_id"], items[j]["cell"], float(nn[j])))
        D[j, :] = np.inf
        D[:, j] = np.inf
        nn = np.where(alive, D.min(axis=1), np.inf)
    for i, it in enumerate(items):
        it["nn_distance"] = float(D[i][alive].min()) if alive[i] and np.isfinite(D[i][alive]).any() else float("nan")
    return [it for i, it in enumerate(items) if alive[i]], removed, dict(cut)


def build(env=None):
    """The set, from the cache `results/test_set[/vN]/set_v<version>.pkl` when its
    digest matches `definition.json`, else built and cached (v1 about 25 min)."""
    if SET_VERSION.startswith("4"):
        return build_v4(env)
    if SET_VERSION.startswith("3"):
        return build_v3(env)
    if SET_VERSION.startswith("2"):
        return build_v2(env)
    return build_v1(env)


def build_v1(env=None):
    import pickle
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    cache, meta = OUT / "set_v1.0.pkl", OUT / "definition.json"
    if cache.exists() and meta.exists():
        with open(cache, "rb") as fh:
            kept, report = pickle.load(fh)
        if report.get("manifest_digest") == json.loads(meta.read_text()).get("manifest_digest"):
            return kept, report
    if env is None:
        from env import ASVLidarEnv
        env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    items, report = _distinct_scenarios(env)
    kept, removed, cut = _trim(items)
    digests = {it["test_id"]: it["built"].digest() for it in kept}
    blob = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    report.update({"version": "1.0", "distinct_scenarios": len(items), "episodes": len(kept),
                   "removed_near_duplicates": len(removed), "removed_by_cell": cut,
                   "removed": removed, "manifest_digest": hashlib.sha256(blob.encode()).hexdigest()})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump((kept, report), fh)
    return kept, report




def _start_clear(env, built, seed, gap=START_GAP_M):
    """The own hull at reset: not in contact with anything, and `gap` clear of every panel."""
    env.reset(seed=seed, options={"generated": built})
    hull = np.asarray(env.hull_polygon(), dtype=float)
    if env.collision_kind(env.hull_polygon()) is not None:
        return False
    return all(p2.polygon_distance(hull, np.asarray(poly, dtype=float)) >= gap for poly in env.obstacles)


def _paper2_cell(env, layout, code, fix):
    """One Paper 2 cell regenerated with `fix` in its flags: p2.build's loop (same
    seed block, episode seeds and field rules) plus the start-clearance check."""
    import scenario as scn
    import targets as tgt
    cells = p2.cells()
    c_index = next(i for i, c in enumerate(cells) if c["layout"] == layout and c["encounter"] == code)
    cell = cells[c_index]
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="frozen_eval")
    records, got, tries = [], 0, 0
    while got < p2.PER_CELL and tries < p2.SEEDS_PER_CELL:
        seed = p2.SEED_BASE + c_index * p2.SEEDS_PER_CELL + tries
        tries += 1
        test_id = f"P2v2-{layout}-{code}-FIX-{got + 1:02d}"
        extra = {"side": cell["side"]} if cell["side"] else {}
        built = generator.sample(seed, case_id=test_id, encounter_class=cell["class"], behaviour=tgt.T_CV,
                                 geometry_mode="basin",
                                 flags=p2._flags(layout, target_stop_box=p2.stop_box(), **extra, **fix))
        if built is None:
            continue
        lo, hi = p2.ct.FIELD_TARGET_SPEED_RANGE
        if not lo <= float(built.target_speed) <= hi:
            continue
        twin = p2.varying_twin(built, seed)
        if twin is None:
            continue
        episode_seed = p2.EPISODE_SEED_BASE + c_index * p2.PER_CELL + got
        if not all(_start_clear(env, b, episode_seed) for b in (built, twin)):
            continue
        if not all(p2.field_feasible(p2.nominal_check(env, b, episode_seed)) for b in (built, twin)):
            continue
        for speed, b in (("FIX", built), ("VAR", twin)):
            records.append({"built": b, "test_id": b.case_id, "speed": speed, "episode_seed": episode_seed,
                            "twin": built.case_id if speed == "VAR" else ""})
        got += 1
    items = []
    for r in records:
        k = int(r["test_id"].rsplit("-", 1)[1]) - 1
        if r["speed"] != ("FIX" if k % 2 == 0 else "VAR"):
            continue
        items.append({"built": r["built"], "episode_seed": r["episode_seed"], "source": "paper2",
                      "cell": f"{layout}-{code}", "class": r["built"].encounter_class, "stratum": "field",
                      "variant": r["speed"], "test_id": r["test_id"], "origin_id": r["twin"] or r["test_id"]})
    return items, {"cell": f"{layout}-{code}", "pairs": got, "seeds_tried": tries}


def build_v2(env=None):
    """v1 with every episode that does not start clear replaced (see the docstring)."""
    import pickle
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    out = out_dir("2.0")
    cache, meta = out / "set_v2.0.pkl", out / "definition.json"
    if cache.exists() and meta.exists():
        with open(cache, "rb") as fh:
            kept, report = pickle.load(fh)
        if report.get("manifest_digest") == json.loads(meta.read_text()).get("manifest_digest"):
            return kept, report
    if env is None:
        from env import ASVLidarEnv
        env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    v1, v1_report = build_v1(env)
    # Replaced: episodes that start in contact (a tight but clear start, as L3-BO's
    # ~0.25 m, is a valid if hard case).  Regenerated episodes must start START_GAP_M clear.
    bad = [it for it in v1 if not _start_clear(env, it["built"], it["episode_seed"], gap=1e-6)]
    by_cell = Counter(it["cell"] for it in bad)
    bad_ids = {it["test_id"] for it in bad}
    kept = [it for it in v1 if it["test_id"] not in bad_ids]
    regenerated = []
    for cell, n in by_cell.items():
        layout, code = cell.split("-", 1)
        fix = CELL_FIXES.get((layout, code))
        if fix is None:
            raise SystemExit(f"{n} episode(s) of {cell} start in contact and no fix is defined")
        items, info = _paper2_cell(env, layout, code, fix)
        chosen, _, _ = _trim(items, target=n, max_cut=1.0)
        kept += chosen
        regenerated.append({**info, "replaced": n, "kept": len(chosen), "fix": fix})
    digests = {it["test_id"]: it["built"].digest() for it in kept}
    blob = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    report = {"version": "2.0", "base": "1.0", "base_digest": v1_report["manifest_digest"],
              "episodes": len(kept), "distinct_scenarios": v1_report["distinct_scenarios"],
              "removed_near_duplicates": v1_report["removed_near_duplicates"],
              "start_in_contact": sorted(bad_ids), "replaced_by_cell": dict(by_cell),
              "regenerated": regenerated, "start_gap_m": START_GAP_M,
              "manifest_digest": hashlib.sha256(blob.encode()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump((kept, report), fh)
    return kept, report


def _hash_order(items):
    return sorted(items, key=lambda it: hashlib.sha256(it["test_id"].encode()).hexdigest())


def _panels(it):
    """Panels actually placed at reset (frozen-suite panels near the encounter are
    dropped by the CPA guard, so the requested count can overstate it)."""
    if "panels" in it:
        return int(it["panels"])
    fixed = (it["built"].flags or {}).get("fixed_obstacles")
    return len(fixed) if fixed else int(it["built"].n_obstacles)


def _fs_episode(env, code, variant, straight, k, *, seed_base=V3_FS_SEED_BASE,
                episode_base=V3_FS_EPISODE_SEED, namespace="test_v3"):
    """One new field-style three-panel episode (see the v3 docstring)."""
    import field_training as ft
    import scenario as scn
    while True:
        base = seed_base + k * 20
        built = ft.sample(np.random.default_rng(base), namespace=namespace, encounter=code,
                          varying=(variant == "VAR"),
                          generator=scn.ScenarioGenerator(stage=5, seed_namespace="frozen_eval"),
                          seed_fn=lambda j, base=base: base + 200_000 + j, solvable_only=True,
                          near=False, straight=straight)
        episode_seed = episode_base + k
        k += 1
        if _start_clear(env, built, episode_seed):
            return built, episode_seed, k


def _frozen_straight(env, cls, variant, k, *, index_base=V3_FROZEN_SEED_INDEX,
                     episode_base=V3_FROZEN_EPISODE_SEED):
    """One new straight-leg three-panel frozen-basin episode (see the v3 docstring)."""
    import feasibility_st
    import scenario as scn
    gen = scn.ScenarioGenerator(stage=5, seed_namespace="frozen_eval")
    while True:
        seed = scn.seed_for("frozen_eval", index_base + k)
        episode_seed = episode_base + k
        k += 1
        x = float(np.random.default_rng(seed + 1).uniform(*cfg.BASIN_X_RANGE))
        built = gen.sample(seed, case_id=f"v3-{seed}", encounter_class=cls,
                           behaviour=suite.target_model(variant.lower(), cls), geometry_mode="basin",
                           flags={"stratum": "basin",
                                  "basin_leg": [[x, float(cfg.BASIN_START_Y)], [x, float(cfg.BASIN_GOAL_Y)]]})
        if built is None or int(built.n_obstacles) != 3:
            continue
        if not _start_clear(env, built, episode_seed):       # resets: the panels are placed
            continue
        if len(env.obstacles) != 3:                          # all three must survive the CPA guard
            continue
        if not feasibility_st.scenario_solvable(built, [list(map(tuple, p)) for p in env.obstacles])["solvable"]:
            continue
        return built, episode_seed, k


def build_v3(env=None):
    """v2 with new obstacle arrangements and more straight legs (see the docstring)."""
    import pickle
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    out = out_dir("3.0")
    cache, meta = out / "set_v3.0.pkl", out / "definition.json"
    if cache.exists() and meta.exists():
        with open(cache, "rb") as fh:
            kept, report = pickle.load(fh)
        if report.get("manifest_digest") == json.loads(meta.read_text()).get("manifest_digest"):
            return kept, report
    if env is None:
        from env import ASVLidarEnv
        env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    v2, v2_report = build_v2(env)
    for it in v2:                                         # realised panel counts (placed at reset)
        env.reset(seed=it["episode_seed"], options={"generated": it["built"]})
        it["panels"] = len(env.obstacles)
    drop, new = set(), []
    # 1. Paper 2 layout cells: half get a new arrangement (same encounter and FIX/VAR).
    field = [it for it in v2 if it["source"] == "paper2" and not it["cell"].endswith("-NT")]
    k_fs, j_fs, counts = 0, 0, {}
    for cell in sorted({it["cell"] for it in field}):
        members = _hash_order([it for it in field if it["cell"] == cell])
        for it in members[:int(round(V3_FIELD_REPLACE[cell[:2]] * len(members)))]:
            drop.add(it["test_id"])
            code = cell.split("-", 1)[1]
            straight = int((j_fs + 1) * V3_FS_STRAIGHT) > int(j_fs * V3_FS_STRAIGHT)   # exact share, spread out
            built, seed, k_fs = _fs_episode(env, code, it["variant"], straight, k_fs)
            n = counts[code] = counts.get(code, 0) + 1
            tid = f"FS-{code}-{it['variant']}-{n:03d}"
            built.case_id = tid
            new.append({"built": built, "episode_seed": seed, "source": "paper2", "cell": f"FS-{code}",
                        "class": built.encounter_class, "stratum": "field", "variant": it["variant"],
                        "test_id": tid, "origin_id": tid, "leg": "straight" if straight else "slanted",
                        "panels": len(built.flags["fixed_obstacles"])})
            j_fs += 1
            print(f"[v3] {tid} ({'straight' if straight else 'slanted'}) replaces {it['test_id']}", flush=True)
    # 2. Frozen basin: slanted three-panel episodes made straight, cell by cell in proportion.
    slanted = [it for it in v2 if it["source"] == "frozen" and it["built"].geometry_mode == "basin"
               and _panels(it) >= 3 and abs(float(it["built"].slant_realised_deg)) > 2.0]
    quota = {}
    for cell in sorted({it["cell"] for it in slanted}):
        quota[cell] = int(round(V3_FROZEN_REPLACE * sum(it["cell"] == cell for it in slanted) / len(slanted)))
    while sum(quota.values()) != V3_FROZEN_REPLACE:                 # rounding: settle on the largest cell
        big = max(quota, key=lambda c: sum(it["cell"] == c for it in slanted))
        quota[big] += 1 if sum(quota.values()) < V3_FROZEN_REPLACE else -1
    k_fr, made = 0, {}
    for cell, q in quota.items():
        for it in _hash_order([it for it in slanted if it["cell"] == cell])[:q]:
            drop.add(it["test_id"])
            cls = it["class"]
            built, seed, k_fr = _frozen_straight(env, cls, it["variant"], k_fr)
            n = made[(cls, it["variant"])] = made.get((cls, it["variant"]), 0) + 1
            tid = f"BAS-{suite.CLASS_CODES[cls]}-{it['variant']}-S{n:03d}"
            built.case_id = tid
            new.append({"built": built, "episode_seed": seed, "source": "frozen", "cell": cell, "class": cls,
                        "stratum": "basin", "variant": it["variant"], "test_id": tid, "origin_id": tid,
                        "leg": "straight", "panels": 3})
            print(f"[v3] {tid} (straight) replaces {it['test_id']}", flush=True)
    # 3. Balance (paper): overtaking was mostly empty water and no-target field
    #    layouts had three episodes.  Replace obstacle-free overtaking episodes with
    #    no-target and dense overtaking field-style episodes.
    empty_ot = _hash_order([it for it in v2 if it["source"] == "frozen" and it["class"] == "overtaking"
                            and _panels(it) == 0 and it["test_id"] not in drop])
    plan = [("NT", "")] * V3_NT_ADD + [("OT", "FIX" if i % 2 == 0 else "VAR") for i in range(V3_OT_DENSE)]
    for it, (code, variant) in zip(empty_ot, plan):
        drop.add(it["test_id"])
        straight = int((j_fs + 1) * V3_FS_STRAIGHT) > int(j_fs * V3_FS_STRAIGHT)
        built, seed, k_fs = _fs_episode(env, code, variant, straight, k_fs)
        n = counts[code] = counts.get(code, 0) + 1
        tid = f"FS-{code}-{variant or 'NT'}-{n:03d}" if code != "NT" else f"FS-NT-{n:03d}"
        built.case_id = tid
        new.append({"built": built, "episode_seed": seed, "source": "paper2", "cell": f"FS-{code}",
                    "class": built.encounter_class, "stratum": "field", "variant": variant,
                    "test_id": tid, "origin_id": tid, "leg": "straight" if straight else "slanted",
                    "panels": len(built.flags["fixed_obstacles"])})
        j_fs += 1
        print(f"[v3] {tid} ({'straight' if straight else 'slanted'}) replaces {it['test_id']} (balance)", flush=True)
    kept = [it for it in v2 if it["test_id"] not in drop] + new
    three = [it for it in kept if _panels(it) >= 3]
    straight_share = float(np.mean([abs(float(it["built"].slant_realised_deg)) <= 2.0 for it in three]))
    digests = {it["test_id"]: it["built"].digest() for it in kept}
    blob = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    report = {"version": "3.0", "base": "2.0", "base_digest": v2_report["manifest_digest"],
              "episodes": len(kept), "distinct_scenarios": v2_report.get("distinct_scenarios"),
              "removed_near_duplicates": v2_report.get("removed_near_duplicates"), "replaced": sorted(drop), "new": [it["test_id"] for it in new],
              "new_field_style": sum(it["cell"].startswith("FS-") for it in new),
              "new_frozen_straight": sum(not it["cell"].startswith("FS-") for it in new),
              "three_panel_episodes": len(three), "three_panel_straight_share": round(straight_share, 3),
              "settings": {"field_replace": V3_FIELD_REPLACE, "fs_straight": V3_FS_STRAIGHT,
                           "frozen_replace": V3_FROZEN_REPLACE, "nt_add": V3_NT_ADD, "ot_dense": V3_OT_DENSE},
              "manifest_digest": hashlib.sha256(blob.encode()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump((kept, report), fh)
    return kept, report


def _frozen_episode(env, item, k):
    """v4: one new episode of a frozen-suite (Tier B) cell -- the cell's class,
    geometry and channel-width range, the item's target behaviour -- from the v4
    frozen_eval index block, start-clear."""
    import scenario as scn
    cls, geometry = item["class"], item["built"].geometry_mode
    cell = next(c for c in suite.tier_b_cells() if c["class"] == cls and c["geometry_mode"] == geometry)
    gen = scn.ScenarioGenerator(stage=5, seed_namespace="frozen_eval")
    while True:
        seed = scn.seed_for("frozen_eval", V4_FROZEN_SEED_INDEX + k)
        episode_seed = V4_FROZEN_EPISODE_SEED + k
        k += 1
        rng = np.random.default_rng(seed + 7)
        width = None if cell["width_range"] is None else float(rng.uniform(*cell["width_range"]))
        built = gen.sample(seed, case_id=f"v4-{seed}", encounter_class=cls, width=width,
                           behaviour=suite.target_model(item["variant"].lower(), cls),
                           geometry_mode=geometry, flags={"stratum": cell["stratum"]})
        if built is not None and _start_clear(env, built, episode_seed, gap=1e-6):
            return built, episode_seed, k


def _layout_episode(env, layout, code, variant, k):
    """v4: one new episode on a Paper 2 deployment layout (L1-L3): p2.build's
    sampling and field rules from the v4 seed block, the VAR twin when the item
    was varying speed, plus the v2 start check and cell fixes."""
    import scenario as scn
    import targets as tgt
    cells = p2.cells()
    c_index = next(i for i, c in enumerate(cells) if c["layout"] == layout and c["encounter"] == code)
    cell = cells[c_index]
    fix = CELL_FIXES.get((layout, code), {})
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="frozen_eval")
    while True:
        seed = V4_P2_SEED_BASE + c_index * 1_000 + k
        episode_seed = V4_P2_EPISODE_SEED + c_index * 100 + k
        k += 1
        extra = {"side": cell["side"]} if cell["side"] else {}
        built = generator.sample(seed, case_id=f"v4-{seed}", encounter_class=cell["class"], behaviour=tgt.T_CV,
                                 geometry_mode="basin",
                                 flags=p2._flags(layout, target_stop_box=p2.stop_box(), **extra, **fix))
        if built is None:
            continue
        lo, hi = p2.ct.FIELD_TARGET_SPEED_RANGE
        if not lo <= float(built.target_speed) <= hi:
            continue
        if variant == "VAR":
            built = p2.varying_twin(built, seed)
            if built is None:
                continue
        if not _start_clear(env, built, episode_seed):
            continue
        if not p2.field_feasible(p2.nominal_check(env, built, episode_seed)):
            continue
        return built, episode_seed, k


_ORACLE = {}


def _oracle_init():
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _ORACLE["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)


def _oracle_job(args):
    import oracle_feasibility as of
    slot, built, seed = args
    return slot, of.assess(_ORACLE["env"], built, seed)


def _passes(rep) -> bool:
    return bool(rep["solved_after_track"]) and bool(rep["margin_after_track"])


def build_v4(env=None, processes: int = 6):
    """v3 with every episode that fails the oracle test replaced (see the docstring)."""
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    import oracle_feasibility as of
    assert abs(of.MARGIN_M - V4_MARGIN_M) < 1e-9, "the oracle's margin is the v4 margin"
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    out = out_dir("4.0")
    cache, meta = out / "set_v4.0.pkl", out / "definition.json"
    if cache.exists() and meta.exists():
        with open(cache, "rb") as fh:
            kept, report = pickle.load(fh)
        if report.get("manifest_digest") == json.loads(meta.read_text()).get("manifest_digest"):
            return kept, report
    if env is None:
        from env import ASVLidarEnv
        env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    v3, v3_report = build_v3(env)
    verdict = pd.read_csv(V4_ORACLE, keep_default_na=False, na_values=[""]).set_index("key")
    missing = [it["test_id"] for it in v3 if f"test:{it['test_id']}" not in verdict.index]
    if missing:
        raise SystemExit(f"{len(missing)} v3 episodes have no oracle verdict, e.g. {missing[:3]}")
    def tier(it):
        r = verdict.loc[f"test:{it['test_id']}"]
        if not (r.n_success > 0 or r.nominal_outcome == "goal"):
            return "unsolvable"
        if not bool(r.solved_after_track):
            return "decided before tracking"
        return "tight" if not bool(r.margin_after_track) else "ok"
    tiers = {it["test_id"]: tier(it) for it in v3}
    out_items = [it for it in v3 if tiers[it["test_id"]] != "ok"]
    print(f"[v4] {len(out_items)} of {len(v3)} v3 episodes fail the oracle test: "
          f"{dict(Counter(tiers[it['test_id']] for it in out_items))}", flush=True)
    # One open slot per excluded episode; candidates are drawn in the main process
    # (cheap) and graded by the oracle in a pool (the cost), round after round.
    slots = {i: it for i, it in enumerate(out_items)}
    filled, ks, log = {}, {"fs": 0, "frozen": 0, "straight": 0}, []
    ks_layout = Counter()

    def draw(it):
        if it["source"] == "frozen":
            if it["test_id"].rsplit("-", 1)[-1].startswith("S"):
                b, sd, ks["straight"] = _frozen_straight(env, it["class"], it["variant"], ks["straight"],
                                                         index_base=V4_FROZEN_SEED_INDEX + 5_000,
                                                         episode_base=V4_FROZEN_EPISODE_SEED + 300)
            else:
                b, sd, ks["frozen"] = _frozen_episode(env, it, ks["frozen"])
            return b, sd
        cell = it["cell"]
        code = cell.split("-", 1)[1]
        if cell.startswith("FS-"):
            straight = abs(float(it["built"].slant_realised_deg)) <= 2.0
            b, sd, ks["fs"] = _fs_episode(env, code, it["variant"], straight, ks["fs"],
                                          seed_base=V4_FS_SEED_BASE, episode_base=V4_FS_EPISODE_SEED,
                                          namespace="test_v4")
            return b, sd
        layout = cell.split("-", 1)[0]
        b, sd, ks_layout[cell] = _layout_episode(env, layout, code, it["variant"], ks_layout[cell])
        return b, sd

    with ProcessPoolExecutor(max_workers=processes, initializer=_oracle_init) as pool:
        for rnd in range(V4_ROUNDS):
            open_slots = [i for i in slots if i not in filled]
            if not open_slots:
                break
            jobs = []
            for i in open_slots:
                for _ in range(V4_CANDIDATES):
                    b, sd = draw(slots[i])
                    jobs.append((i, b, sd))
            print(f"[v4] round {rnd + 1}: {len(open_slots)} open slots, {len(jobs)} candidates", flush=True)
            for (i, b, sd), (_, rep) in zip(jobs, pool.map(_oracle_job, jobs, chunksize=1)):
                ok = _passes(rep)
                log.append({"slot": i, "replaces": slots[i]["test_id"], "round": rnd + 1, "seed": sd,
                            "passes": ok, **{k: rep[k] for k in ("solved_after_track", "margin_after_track",
                                                                 "n_success", "t_track_s", "best_clearance_m")}})
                if ok and i not in filled:
                    filled[i] = (b, sd, rep)
    if len(filled) < len(slots):
        raise SystemExit(f"[v4] {len(slots) - len(filled)} slots still open after {V4_ROUNDS} rounds")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out / "replacement_candidates.csv", index=False)
    new, made = [], Counter()
    for i, it in slots.items():
        b, sd, rep = filled[i]
        if it["source"] == "frozen":
            stem = it["test_id"].rsplit("-", 1)[0]
        elif it["cell"].startswith("FS-"):
            stem = f"FS-{it['cell'].split('-', 1)[1]}-{it['variant']}"
        else:
            stem = f"P2v4-{it['cell']}-{it['variant']}"
        made[stem] += 1
        tid = f"{stem}-R{made[stem]:03d}"
        b.case_id = tid
        new.append({**{k: it[k] for k in ("source", "cell", "class", "stratum", "variant")},
                    "built": b, "episode_seed": sd, "test_id": tid, "origin_id": tid,
                    "replaces": it["test_id"], "replaced_tier": tiers[it["test_id"]],
                    "leg": "straight" if abs(float(b.slant_realised_deg)) <= 2.0 else "slanted"})
        env.reset(seed=sd, options={"generated": b})
        new[-1]["panels"] = len(env.obstacles)
        print(f"[v4] {tid} replaces {it['test_id']} ({tiers[it['test_id']]})", flush=True)
    drop = {it["test_id"] for it in out_items}
    kept = [it for it in v3 if it["test_id"] not in drop] + new
    digests = {it["test_id"]: it["built"].digest() for it in kept}
    blob = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    report = {"version": "4.0", "base": "3.0", "base_digest": v3_report["manifest_digest"],
              "episodes": len(kept), "distinct_scenarios": v3_report.get("distinct_scenarios"),
              "removed_near_duplicates": v3_report.get("removed_near_duplicates"),
              "oracle": str(V4_ORACLE.relative_to(ROOT)), "margin_m": V4_MARGIN_M,
              "excluded_by_tier": dict(Counter(tiers[t] for t in drop)), "replaced": sorted(drop),
              "new": [it["test_id"] for it in new], "candidates_graded": len(log),
              "manifest_digest": hashlib.sha256(blob.encode()).hexdigest()}
    with open(cache, "wb") as fh:
        pickle.dump((kept, report), fh)
    return kept, report


def _write_definition(kept, report):
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    rows = [{k: it[k] for k in ("test_id", "origin_id", "source", "cell", "class", "stratum", "variant",
                                "episode_seed")} | {"nn_distance": it.get("nn_distance"), "panels": _panels(it),
                                                     "leg": "straight" if abs(float(it["built"].slant_realised_deg)) <= 2.0 else "slanted",
                                                     "digest": it["built"].digest()} for it in kept]
    pd.DataFrame(rows).to_csv(out / "definition.csv", index=False)
    (out / "definition.json").write_text(json.dumps(report, indent=1, default=str))


def _summary(d: pd.DataFrame, group) -> pd.DataFrame:
    g = d.groupby(group)
    return pd.DataFrame({
        "n": g.size(),
        "success": g.apply(lambda x: (x.outcome == "goal").mean(), include_groups=False),
        "coll_target": g.collided_target.mean(),
        "coll_other": g.apply(lambda x: (x.collided & ~x.collided_target).mean(), include_groups=False),
        "timeout": g.apply(lambda x: (x.outcome == "timeout").mean(), include_groups=False),
    }).round(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="write the set definition only")
    ap.add_argument("--version", choices=("1", "2", "3", "4"), default="3")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--policy", help="classical comparator(s) instead of a model, comma-separated: "
                                     "los_dwa, colregs_vo, encounter_vo, reference")
    ap.add_argument("--tag")
    ap.add_argument("--safety", choices=("off", "on", "both"), default="off")
    ap.add_argument("--safety-version", type=int, default=1)
    ap.add_argument("--processes", type=int, default=3,
                    help="evaluation processes (default 3: a training run shares the CPU)")
    args = ap.parse_args()
    global SET_VERSION
    SET_VERSION = args.version + ".0"
    t0 = time.time()
    kept, report = build()
    _write_definition(kept, report)
    comp = pd.DataFrame([{k: it[k] for k in ("source", "cell", "variant")} for it in kept])
    if report.get("regenerated"):
        print(f"v2: {len(report['start_in_contact'])} episodes started in contact and were replaced: "
              f"{report['regenerated']}", flush=True)
    print(f"test set {SET_VERSION}: {len(kept)} episodes from {report['distinct_scenarios']} distinct "
          f"scenarios ({report['removed_near_duplicates']} near-duplicates removed), digest "
          f"{report['manifest_digest'][:16]}, built in {time.time() - t0:.0f} s", flush=True)
    print(comp.groupby(["source", "cell"]).size().to_string(), flush=True)
    if args.build or not (args.model or args.policy):
        return 0
    if args.policy:
        for policy in args.policy.split(","):
            evaluate(kept, report, policy=policy.strip(), tag=policy.strip(), args=args, t0=time.time())
        return 0
    model = args.model if args.model.is_absolute() else ROOT / args.model
    evaluate(kept, report, model=model, tag=args.tag or model.parent.name, args=args, t0=t0)
    return 0


def _reusable(tag, mode, kept):
    """v2: the same controller's v1 rows for the episodes v2 shares with v1 (same
    test id, scenario digest and seed).  Evaluation is deterministic: SAC's v1 run
    reproduced all 1,000 of its earlier frozen-suite and Paper 2 rows."""
    if SET_VERSION.startswith("1"):
        return None
    prev = {"2": "1.0", "3": "2.0", "4": "3.0"}[SET_VERSION[0]]    # the version this one is built on
    v1_rows, v1_def = out_dir(prev) / tag / "episodes.csv", out_dir(prev) / "definition.csv"
    if not (v1_rows.exists() and v1_def.exists()):
        return None
    rows = pd.read_csv(v1_rows)
    if "safety" in rows:
        rows = rows[rows.safety == mode]
    same = pd.read_csv(v1_def).set_index("test_id").digest.to_dict()
    ids = {it["test_id"] for it in kept if same.get(it["test_id"]) == it["built"].digest()}
    return rows[rows.test_id.isin(ids)]


def evaluate(kept, report, *, tag, args, t0, model=None, policy="model"):
    out = out_dir() / tag
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for mode in (("off", "on") if args.safety == "both" else (args.safety,)):
        reuse = _reusable(tag, mode, kept)
        done = set(reuse.test_id) if reuse is not None else set()
        jobs = [(it["built"], it["episode_seed"], policy, None,
                 {**{k: it[k] for k in ("test_id", "origin_id", "source", "cell", "stratum", "variant")},
                  "_keep_traj": True})
                for it in kept if it["test_id"] not in done]
        if done:
            print(f"[{tag}] reusing {len(done)} rows of the previous version; evaluating {len(jobs)} episodes", flush=True)
        rows = run_pool(jobs, model_path=model, processes=args.processes,
                        overrides={"EMERGENCY_STOP_ENABLED": mode == "on",
                                   "SAFETY_VERSION": int(args.safety_version)}) if jobs else []
        # The true trajectories go to their own file, for the offline COLREGs checks
        # (tools/diagnostics/colregs_compliance.py), not into the CSV.
        trajs = {r["test_id"]: r.pop("_traj") for r in rows if "_traj" in r}
        if trajs:
            np.savez_compressed(out / f"trajectories_{mode}.npz",
                                **{f"{tid}|{k}": v for tid, tr in trajs.items() for k, v in tr.items()})
        f = pd.DataFrame(rows)
        f["safety"] = mode
        if reuse is not None and len(reuse):
            f = pd.concat([reuse.assign(safety=mode), f], ignore_index=True)
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(out / "episodes.csv", index=False)
    pd.set_option("display.width", 200)
    text = [f"Test set {SET_VERSION} ({tag}) -- {model.relative_to(ROOT) if model else 'classical: ' + policy}",
            f"{len(kept)} episodes, digest {report['manifest_digest'][:16]}, {time.time() - t0:.0f} s", "",
            "-- overall", _summary(d, "safety").to_string(), ""]
    for mode in d.safety.unique():
        x = d[d.safety == mode]
        text += [f"-- by source (safety {mode})", _summary(x, "source").to_string(), "",
                 f"-- by cell (safety {mode})", _summary(x, ["source", "cell"]).to_string(), "",
                 f"-- by variant (safety {mode})", _summary(x, ["source", "variant"]).to_string(), ""]
    body = "\n".join(text) + "\n"
    (out / "summary.txt").write_text(body, encoding="utf-8")
    print(body, flush=True)


if __name__ == "__main__":
    sys.exit(main())
