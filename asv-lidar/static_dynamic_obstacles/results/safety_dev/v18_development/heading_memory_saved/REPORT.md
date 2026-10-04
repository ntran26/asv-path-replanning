# Saved FIX05 hull-heading-memory applicability

Created 2026-10-03T11:31:22.414375+00:00. No new episodes, environment reset/step or policy inference.

The new adapter was replayed causally over all 39 captured V13 decision snapshots. It changes only source21 at decisions34–36, carrying the114° fit from decision33/scan476 for1–3 decisions. Memory expires at decision37. The synthetic persistent view remains unchanged.

At decision34 the original velocity-course fallback is147.561654°, versus held114° and scoring-only true hull115.783768°. The exact previously reconstructed policy plan reproduces the logged clearance **+0.134074279m**. Changing only this axis yields **−0.290731058m**, first violation0.125s.

This verifies gated applicability and geometry sensitivity, not rescue. The target centre remains biased; the earlier full-truth-target sensitivity passes the same plan at+0.007051m. A more accurate axis alone therefore does not establish a more accurate complete collision forecast. Future raw observations remain those of the saved V13 episode.

The model separates hull extent/orientation from translational kinematics, inspired by [Granstrom, Baum & Reuter's extended-object tracking overview](https://arxiv.org/abs/1604.00970). The bounded hold here is an engineering adaptation, not that paper's probabilistic estimator or a safety guarantee.

`audit.json` records source, trace, manifest and original-plan-audit hashes. The evaluated script bytes are preserved beside it. Reproduce from the project directory with a NEW output path:

```powershell
python -B tools/diagnostics/safety/audit_heading_memory_saved.py --output-dir results/safety_dev/v18_development/heading_memory_saved_repeat
```

Existing output directories are refused. The working script lives under tools; the saved byte copy is an immutable provenance artifact.
