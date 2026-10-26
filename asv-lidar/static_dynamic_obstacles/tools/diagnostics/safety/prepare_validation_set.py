"""Freeze the baseline-v3 validation set (270 episodes) as the safety-layer development set (V21 phase).

The deployed policy (SAC seed 0, best checkpoint at 3.0 M) was selected on these
episodes: 120 decoupled (`train_formulation.development_set(20)`, built with the
`formulation_v3` overlay installed, which sets the stage-5 panel draws) and 150
coupled (`formulation_v3.field_development_set`, the DV3 cases of the earlier safety
work).  Episode seeds follow the selection callback: 900_000 + index, decoupled first.

* DVD (decoupled): built here and cached under `results/safety_dev/scenario_cache/dvd_v3/`.
  The runner must install the same overlay, because the environment draws the panels
  of these scenes at reset (`tools/diagnostics/safety/v21_iteration.py` does).
* DV3 (coupled): the existing canonical cache; this tool rebuilds the set and checks
  every cached scene's digest and seed against it.

    python -B tools/diagnostics/safety/prepare_validation_set.py

Writes `results/safety_dev/v21_development/selection_validation.json`.  No episode runs.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers")]
CACHE = ROOT / "results/safety_dev/scenario_cache/dvd_v3"
OUT = ROOT / "results/safety_dev/v21_development/selection_validation.json"
INVENTORY = ROOT / "results/safety_dev/suites/inventory_v5/manifest.json"
CODES = {"head_on": "HO", "crossing": "CR", "overtaking": "OT", "being_overtaken": "BO",
         "null": "NU", "no_target": "NT"}
SEED_BASE = 900_000


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    import constants as cfg
    import formulation_v3
    formulation_v3.apply(cfg)
    import train_formulation as tf

    decoupled = tf.development_set(20)
    coupled = formulation_v3.field_development_set()
    if len(decoupled) != 120 or len(coupled) != 150:
        raise SystemExit(f"unexpected set sizes {len(decoupled)} / {len(coupled)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    cases, hashes, counter = [], {}, Counter()
    for i, built in enumerate(decoupled):
        cls = str(built.encounter_class)
        counter[cls] += 1
        name = f"DVD-{CODES[cls]}-{counter[cls]:02d}"
        raw = pickle.dumps(built)
        path = CACHE / f"{name}.pkl"
        if path.exists() and pickle.loads(path.read_bytes()).digest() != built.digest():
            raise SystemExit(f"cached {name} differs from the rebuilt scene")
        if not path.exists():
            path.write_bytes(raw)
        hashes[path.relative_to(ROOT).as_posix()] = sha(path.read_bytes())
        cases.append({"case": f"DVD:{name}", "scenario_id": name, "dataset": "dvd", "seed": SEED_BASE + i,
                      "scenario_sha256": built.digest(), "selection_stratum": f"decoupled|{cls}",
                      "source": "decoupled", "encounter": cls, "panels": int(built.n_obstacles or 0)})

    inventory = {c["case"]: c for c in json.loads(INVENTORY.read_text())["cases"] if c["suite"] == "dev_field"}
    import v9_paired_pilot as pilot
    cached, _ = pilot.load_scenes([{"case": f"DV3:{c}", "dataset": "dv3", "seed": e["seed"],
                                    "scenario_sha256": e["scenario_sha256"]} for c, e in inventory.items()])
    count_only = 0
    for j, built in enumerate(coupled):
        entry = inventory.get(getattr(built, "case_id", ""))
        seed = SEED_BASE + 120 + j
        if entry is None or int(entry["seed"]) != seed:
            raise SystemExit(f"coupled validation episode {j} has no DV3 cache entry with seed {seed}")
        # The cached scenes predate a generator change that rewrote `n_obstacles` on
        # fixed layouts; the environment ignores it when `fixed_obstacles` is set
        # (env._load_generated), so every other field must agree exactly.
        a, b = vars(cached[f"DV3:{entry['case']}"]), vars(built)
        differ = [k for k in set(a) | set(b) if repr(a.get(k)) != repr(b.get(k))]
        if differ and (differ != ["n_obstacles"] or not (built.flags or {}).get("fixed_obstacles")):
            raise SystemExit(f"DV3 cache entry {entry['case']} differs from the rebuilt scene: {differ}")
        count_only += bool(differ)
        code = entry["case"].split("-")[1]
        cases.append({"case": f"DV3:{entry['case']}", "scenario_id": entry["case"], "dataset": "dv3",
                      "seed": seed, "scenario_sha256": entry["scenario_sha256"],
                      "selection_stratum": f"coupled|{code}", "source": "coupled", "encounter": code,
                      "case_id": getattr(built, "case_id", "")})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    selection = {
        "name": "baseline-v3 validation set (safety development, V21 phase)",
        "evaluation_role": "development",
        "scope": "Development only: the 270 validation episodes the deployed policy was selected on; "
                 "no claimed population estimate or formal safety guarantee",
        "overlay": "formulation_v3 (required for the DVD scenes; the runner installs it)",
        "seed_rule": "900000 + index, decoupled (120) then coupled (150), as the selection callback",
        "dv3_cache_note": f"{count_only} of 150 cached DV3 scenes differ from the rebuilt set only in "
                          "n_obstacles on a fixed layout, which the environment ignores",
        "input_sha256": hashes,
        "cases": cases,
    }
    OUT.write_text(json.dumps(selection, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases ({Counter(c['dataset'] for c in cases)}) -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
