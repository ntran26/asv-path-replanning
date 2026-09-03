"""
evaluate_reward_v2.py
=====================

Runs the fixed 500-case holdout suite against the repaired-reward environment
``rl_env_reward_v2.py``, reusing ``evaluate_sac_suite.py`` unmodified.

Unlike the original (which uses a USER SETTINGS block at the top of the file),
this runner takes argparse flags and defaults to a SEPARATE output directory so
existing results in ``eval_results/eval_suite/`` are never overwritten.

Baseline control run -- evaluate the protected 1M checkpoint under the v2 env.
The policy is deterministic and does not consume reward at inference, so this
should reproduce the 0.940 baseline exactly; if it does not, the v2 env changed
something beyond the reward:

    python evaluate_reward_v2.py --model-path models/sac_model_1M.zip \
        --out-dir eval_results/eval_suite_v2env_baseline

Evaluate a fine-tuned v2 policy:

    python evaluate_reward_v2.py --model-path models/sac_reward_v2.zip \
        --out-dir eval_results/eval_suite_reward_v2

Quick smoke test on the first N scenarios:

    python evaluate_reward_v2.py --limit 10
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ENV_MODULE = "rl_env_reward_v2"

# The alias must be installed before evaluate_sac_suite executes its
# `from rl_env import ...` at import time, which is before argparse runs.
# So --env-module is pulled off sys.argv here rather than in parse_args().
ENV_MODULE = DEFAULT_ENV_MODULE
if "--env-module" in sys.argv:
    _i = sys.argv.index("--env-module")
    if _i + 1 < len(sys.argv):
        ENV_MODULE = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
else:
    for _a in list(sys.argv):
        if _a.startswith("--env-module="):
            ENV_MODULE = _a.split("=", 1)[1]
            sys.argv.remove(_a)
            break

_env_v2 = importlib.import_module(ENV_MODULE)  # noqa: E402

sys.modules["rl_env"] = _env_v2

import evaluate_sac_suite as ev  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="models/sac_reward_v2.zip")
    ap.add_argument("--suite-json", default="eval_suite/asv_eval_suite.json")
    ap.add_argument("--out-dir", default="eval_results/eval_suite_reward_v2")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N scenarios (smoke test).")
    ap.add_argument("--force", action="store_true",
                    help="Allow writing into a non-empty --out-dir.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model_path):
        sys.exit(f"[v2] model not found: {args.model_path}")
    if not os.path.isfile(args.suite_json):
        sys.exit(f"[v2] suite not found: {args.suite_json}")
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.force:
        sys.exit(f"[v2] --out-dir {args.out_dir} is not empty. Use --force to overwrite.")

    ev.MODEL_PATH = args.model_path
    ev.SUITE_JSON = args.suite_json
    ev.OUT_DIR = args.out_dir
    ev.DETAIL_CSV = os.path.join(args.out_dir, "eval_suite_details.csv")
    ev.DETAIL_JSON = os.path.join(args.out_dir, "eval_suite_details.json")
    ev.SUMMARY_JSON = os.path.join(args.out_dir, "eval_suite_summary.json")
    ev.SUMMARY_CSV = os.path.join(args.out_dir, "eval_suite_summary.csv")
    ev.LIMIT_SCENARIOS = args.limit

    print(f"[v2] env module : {_env_v2.__name__} (OA_GAIN={_env_v2.OA_GAIN}, "
          f"floor={_env_v2.OA_DIST_FLOOR}, K_CTE_RECOVERY={_env_v2.K_CTE_RECOVERY}, "
          f"K_WRONG_SIDE_ACTION={_env_v2.K_WRONG_SIDE_ACTION})")
    if getattr(_env_v2, "BLOCK_USE_WIDE_ARC", False):
        print(f"[v2] wide-arc gate ON (BLOCK_ARC_DEG={_env_v2.BLOCK_ARC_DEG})")
    print(f"[v2] model      : {args.model_path}")
    print(f"[v2] out dir    : {args.out_dir}")

    ev.main()

    # Echo the headline numbers the README asks to be tracked separately.
    try:
        with open(ev.SUMMARY_JSON, "r") as f:
            summary = json.load(f)
        print("\n[v2] summary (obstacle vs border collisions reported separately):")
        print(json.dumps(summary, indent=2)[:2000])
    except Exception as exc:  # pragma: no cover
        print(f"[v2] could not re-read summary: {exc}")


if __name__ == "__main__":
    main()
