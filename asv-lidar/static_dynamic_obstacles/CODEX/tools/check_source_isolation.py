"""Verify that the copied parent sources still match the creation snapshot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    changed = []
    for record in manifest["files"]:
        path = ROOT.parent / record["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            changed.append(record["path"])
    result = {"checked_source_files": len(manifest["files"]),
              "unchanged": not changed, "changed_since_snapshot": changed}
    print(json.dumps(result, indent=2))
    return int(bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
