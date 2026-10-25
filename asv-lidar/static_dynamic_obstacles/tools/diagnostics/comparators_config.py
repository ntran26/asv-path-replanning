"""Freeze and check the classical comparators' settings (`configs/comparators_v<N>.json`).

The comparators read every parameter from the `CLASSICAL_*` names in
`src/constant_temp.py`; a frozen file records all of them and their digest
(sha256 of the sorted JSON of the values, first 16 hex digits -- the recipe of
`configs/comparators_v1.json`, digest 0dd8eab26bbf954f).  The formulation's digest
(`configs/baseline_v3.json`) does not cover them, so a comparator run checks
against this file instead.

    python tools/diagnostics/comparators_config.py --check configs/comparators_v2.json
    python tools/diagnostics/comparators_config.py --write-v2

`--write-v2` (comparators v2, decision S18 / A35): the encounter-specific VO with
its passing side judged on the encounter, tuned on the decoupled part of the
validation set; provenance read from `results/classical_tuning/encounter_vo/`.
LOS-PID + DWA keeps its comparators v1 values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TUNING = ROOT / "results" / "classical_tuning" / "encounter_vo"
V1 = ROOT / "configs" / "comparators_v1.json"
V2 = ROOT / "configs" / "comparators_v2.json"


def values() -> dict:
    import constant_temp as ct
    raw = {k: getattr(ct, k) for k in dir(ct) if k.startswith("CLASSICAL_")}
    return json.loads(json.dumps(raw, default=str, sort_keys=True))


def digest(block: dict) -> str:
    return hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()[:16]


def check(path: Path) -> list:
    saved = json.loads(path.read_text())["values"]
    now = values()
    return [f"{k}: file {saved.get(k)!r}, code {now.get(k)!r}"
            for k in sorted(set(saved) | set(now)) if saved.get(k) != now.get(k)]


def write_v2() -> dict:
    best = json.loads((TUNING / "best.json").read_text())
    validation = (TUNING / "validation.txt").read_text().strip().replace(
        "development episodes", "decoupled validation episodes")
    tuned = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best["values"].items()}
    now = values()
    differ = {k: (now.get(k), v) for k, v in tuned.items() if now.get(k) != v}
    if "keep tuned" in validation and differ:
        raise SystemExit(f"validation kept the tuned values but the code holds others: {differ}")
    v1 = json.loads(V1.read_text())["values"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    # Source files changed since that commit (the frozen values are in `values` either way).
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "tools/diagnostics/tune_classical.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    dirty = sorted(line[3:].strip() for line in dirty if line.strip())
    block = {
        "id": "comparators-v2",
        "frozen_at": time.strftime("%Y-%m-%d"),
        "supersedes": "comparators-v1",
        "git": {"head": head, "uncommitted": dirty},
        "digest": digest(now),
        "provenance": {
            "decision": "S18 / A35: the encounter-specific VO is the velocity-obstacle comparator; its "
                        "passing side is judged on the encounter (CLASSICAL_VO_SIDE_ON_ENCOUNTER) -- it binds "
                        "every closing candidate while the encounter context reports the encounter engaged",
            "tuned_on": f"validation set, decoupled part ({best['episodes']} of 120 episodes in the search, "
                        f"{best['per_class']} per class; baseline-v3 build, formulation_v3 overlay); "
                        "the test set was not used",
            "score": "goal rate - 2 x collision rate, the rule RL checkpoints are selected by",
            "search": "coordinate descent over the encounter_vo grid in tools/diagnostics/tune_classical.py "
                      "(results/classical_tuning/encounter_vo/search.csv)",
            "validation": validation,
            "search_best": {"score": best["score"], "values": tuned},
            "los_dwa": "unchanged from comparators v1 (tuned 2026-09-23, defaults kept)",
            "colregs_vo": "not a comparator (S18); its CLASSICAL_KVO_* values stay because the test set's "
                          "reactive target (T-RE) runs its rule",
            "changed_from_v1": {k: {"v1": v1.get(k), "v2": now.get(k)}
                                for k in sorted(set(v1) | set(now)) if v1.get(k) != now.get(k)},
        },
        "values": now,
    }
    V2.write_text(json.dumps(block, indent=1) + "\n", encoding="utf-8")
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", type=Path, help="compare the code's CLASSICAL_* values with this file")
    group.add_argument("--write-v2", action="store_true")
    args = ap.parse_args()
    if args.write_v2:
        block = write_v2()
        print(f"wrote {V2.relative_to(ROOT)}: digest {block['digest']}; changed from v1: "
              f"{sorted(block['provenance']['changed_from_v1'])}")
        return 0
    diffs = check(args.check)
    saved = json.loads(args.check.read_text())
    if diffs:
        print(f"code differs from {saved.get('id')} ({len(diffs)}):\n  " + "\n  ".join(diffs))
        return 1
    print(f"code matches {saved.get('id')} (digest {digest(values())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
