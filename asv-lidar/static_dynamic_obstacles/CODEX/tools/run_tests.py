"""Run the complete regression suite and retain its output inside CODEX."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "results" / "tests"
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(destination / "pytest.xml")]
    started = time.time()
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    with (destination / "pytest.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, creationflags=flags)
        code = process.wait()
    result = {"command": command, "exit_code": code, "seconds": time.time() - started}
    (destination / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print((destination / "pytest.log").read_text(encoding="utf-8"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
