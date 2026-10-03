# Safety artifact maintenance

Transparently compressed 495 resident JSONL traces in20 completed runs, preserving their paths and all SHA-256 hashes. No files were deleted.

- File allocation reclaimed: 945,003,157 bytes (0.880 GiB).
- Whole-drive free-space change: 909,721,600 bytes; concurrent disk activity can differ from the per-file total.
- Verification: 2939 historical report hash references match, covering 495 distinct traces.
- Tests:15 synthetic maintenance tests passed; no simulator episodes.

[Inventory](inventory.json), [per-file journal](progress.jsonl), [summary](summary.json), [verification](verification.json), [storage audit](storage_audit.json).

The audit retained all evaluated source archives, results, manifests and scenario caches. The150 byte-identical old/new scenario-cache pairs use only4.4 MB; their provenance paths remain available. Six regenerable safety bytecode files totaling28,852 bytes were listed separately for the main agent; this maintenance process did not delete them.
