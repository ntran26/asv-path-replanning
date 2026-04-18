"""Run the narrow faithful 4DOF motion + turn refinement sweeps in sequence.

Run with:
    python run_bluefin_4dof_refine.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(script_name: str) -> None:
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(ROOT / script_name)], check=True)

def main() -> None:
    run("sweep_bluefin_4dof_motion_refine.py")
    run("sweep_bluefin_4dof_turn_refine.py")
    print("\nDone. Check:")
    print("  - bluefin_4dof_motion_sweep_refine/")
    print("  - bluefin_4dof_turn_sweep_refine/")

if __name__ == "__main__":
    main()
