"""Environment set-ups by name: one format for the test set and the field test suite.

A set-up is everything `ASVLidarEnv.reset` needs to replay an episode, with its name:

    {"id": "medium_2", "built": <scenario.Scenario>, "seed": 886037, "class": "crossing",
     "about": "field suite, medium, L1, crossing, target to starboard (give way)",
     "set": "field", "plant": "identified"}

    env = ASVLidarEnv(render_mode=None, emergency_stop=False, **setups.env_kwargs(s))
    obs, _ = env.reset(seed=s["seed"], options={"generated": s["built"]})

`plant` is the vessel model the set is run on: "randomised" for test set v4 (each episode seed
draws the plant from the training randomisation, as `tools/tiers/test_set.py` runs it) and
"identified" for the field suite (the identified July plant, as it was verified;
`tools/tiers/field_suite.py`).  `env_kwargs` gives the matching constructor arguments.

Names:

    easy_1 ... easy_5, medium_1 ... medium_5, hard_1 ... hard_5
        the field test suite (`field_test_cases.py`; the numbering is explained there)
    BAS-HO-CV-001, CH-CR-RE-002, P2-L1-HO-FIX-03, FS-CRP-FIX-002, ...
        test set v4 (`results/test_set/v4/definition.csv`); a "TS4:" prefix is accepted

    import setups
    s = setups.load("easy_1")            # or setups.load("BAS-HO-CV-001")
    every = setups.suite("field")        # or setups.suite("test")

The validation set is built with a run's formulation overlay and stays with `test_asv.py`.
`test_asv.py` (simulation) and `field_deployment/udp_live_p3.py --scenario` (the water) both load
through this module.
"""
from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
TEST_SET_DIR = ROOT / "results" / "test_set" / "v4"
SETS = ("test", "field")
FIELD_NAME = re.compile(r"^(easy|medium|hard)_\d+$")

_TEST: Optional[Dict[str, Dict]] = None
_DIGESTS: Optional[Dict[str, str]] = None


def which(ref: str) -> Optional[str]:
    """The set a name belongs to ("field" or "test"), or None if it names neither."""
    ref = ref.strip()
    if FIELD_NAME.match(ref):
        return "field"
    return "test" if ref.split(":", 1)[-1] in _test_set() else None


def _test_set() -> Dict[str, Dict]:
    """Test set v4 as built (`tools/tiers/test_set.py build_v4`), checked against its manifest."""
    global _TEST
    if _TEST is None:
        with open(TEST_SET_DIR / "set_v4.0.pkl", "rb") as fh:
            kept, report = pickle.load(fh)
        meta = json.loads((TEST_SET_DIR / "definition.json").read_text())
        if report.get("manifest_digest") != meta.get("manifest_digest"):
            raise ValueError("results/test_set/v4/set_v4.0.pkl does not match definition.json")
        _TEST = {it["test_id"]: it for it in kept}
    return _TEST


def _test_digests() -> Dict[str, str]:
    global _DIGESTS
    if _DIGESTS is None:
        with open(TEST_SET_DIR / "definition.csv", newline="", encoding="utf-8") as fh:
            _DIGESTS = {row["test_id"]: row["digest"] for row in csv.DictReader(fh)}
    return _DIGESTS


def _test_setup(item: Dict) -> Dict:
    return {"id": item["test_id"], "built": item["built"], "seed": int(item["episode_seed"]),
            "class": item["class"], "about": f"test set v4, {item['source']} {item['cell']} {item['variant']}",
            "set": "test", "plant": "randomised"}


def _field_setup(name: str) -> Dict:
    import field_test_cases as ftc
    built, seed = ftc.TestCase().scenario(name)
    c = ftc.CASES[name]
    return {"id": name, "built": built, "seed": int(seed), "class": built.encounter_class,
            "about": f"field suite, {c['mode']}, {c['layout']}, {ftc.ENCOUNTER_NAMES[c['encounter']]}",
            "set": "field", "plant": "identified"}


def load(ref: str) -> Dict:
    """One set-up by name, checked against its record (digest)."""
    ref = ref.strip()
    if FIELD_NAME.match(ref):
        return _field_setup(ref)
    test_id = ref.split(":", 1)[-1]
    item = _test_set().get(test_id)
    if item is None:
        raise KeyError(f"{ref}: not a field case (easy_1 ... hard_5) or a test set v4 ID")
    if item["built"].digest() != _test_digests().get(test_id):
        raise ValueError(f"{test_id}: the cached scenario does not match the set definition")
    return _test_setup(item)


def suite(name: str) -> List[Dict]:
    """Every set-up of a set, in its own order."""
    if name == "field":
        import field_test_cases as ftc
        return [_field_setup(n) for n in ftc.TestCase().names()]
    if name == "test":
        return [_test_setup(it) for it in _test_set().values()]
    raise KeyError(f"{name}: one of {', '.join(SETS)}")


def env_kwargs(setup: Dict, plant: Optional[str] = None) -> Dict:
    """`ASVLidarEnv` arguments for the set-up's plant ("identified" or "randomised"; `plant`
    overrides it)."""
    import constants as cfg
    plant = plant or setup["plant"]
    if plant == "identified":
        return {"vessel_randomisation": None}
    if plant == "randomised":
        return {"vessel_randomisation": cfg.VESSEL_RANDOMISATION_SCALE}
    raise ValueError(f"plant {plant!r}: identified or randomised")
