"""The test set: the frozen suite and the Paper 2 deployment-layout set merged,
twins and near-duplicates trimmed, 1,000 episodes (your call, 2026-10-02).

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
SET_VERSION = "1.0"


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
    """The set, from the cache `results/test_set/set_v<version>.pkl` when its digest
    matches `definition.json`, else built (about 25 min) and cached."""
    import pickle
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    cache, meta = OUT / f"set_v{SET_VERSION}.pkl", OUT / "definition.json"
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
    report.update({"version": SET_VERSION, "distinct_scenarios": len(items), "episodes": len(kept),
                   "removed_near_duplicates": len(removed), "removed_by_cell": cut,
                   "removed": removed, "manifest_digest": hashlib.sha256(blob.encode()).hexdigest()})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump((kept, report), fh)
    return kept, report


def _write_definition(kept, report):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{k: it[k] for k in ("test_id", "origin_id", "source", "cell", "class", "stratum", "variant",
                                "episode_seed", "nn_distance")} | {"digest": it["built"].digest()} for it in kept]
    pd.DataFrame(rows).to_csv(OUT / "definition.csv", index=False)
    (OUT / "definition.json").write_text(json.dumps(report, indent=1, default=str))


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
    ap.add_argument("--model", type=Path)
    ap.add_argument("--policy", help="classical comparator(s) instead of a model, comma-separated: "
                                     "los_dwa, colregs_vo, encounter_vo, reference")
    ap.add_argument("--tag")
    ap.add_argument("--safety", choices=("off", "on", "both"), default="off")
    ap.add_argument("--safety-version", type=int, default=1)
    ap.add_argument("--processes", type=int, default=3,
                    help="evaluation processes (default 3: a training run shares the CPU)")
    args = ap.parse_args()
    t0 = time.time()
    kept, report = build()
    _write_definition(kept, report)
    comp = pd.DataFrame([{k: it[k] for k in ("source", "cell", "variant")} for it in kept])
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


def evaluate(kept, report, *, tag, args, t0, model=None, policy="model"):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(it["built"], it["episode_seed"], policy, None,
             {k: it[k] for k in ("test_id", "origin_id", "source", "cell", "stratum", "variant")})
            for it in kept]
    frames = []
    for mode in (("off", "on") if args.safety == "both" else (args.safety,)):
        rows = run_pool(jobs, model_path=model, processes=args.processes,
                        overrides={"EMERGENCY_STOP_ENABLED": mode == "on",
                                   "SAFETY_VERSION": int(args.safety_version)})
        f = pd.DataFrame(rows)
        f["safety"] = mode
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
