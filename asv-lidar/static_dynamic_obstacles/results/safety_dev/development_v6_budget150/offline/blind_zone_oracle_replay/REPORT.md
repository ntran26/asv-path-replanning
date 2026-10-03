# Offline blind-zone plant feasibility audit

2026-10-02T15:33:25.755985+00:00

No new policy episodes, environment objects, reset/step calls, or evaluation tokens. This audit uses **oracle** recorded state, true geometry and seed-derived simulator parameters. It does not use or refit the measurement-only braking calibration.

| Case | Initial inflated-hull SAT gap | Sampled first-command result | Immediate straight brake |
|---|---:|---|---|
| DV3-BO-CV-10 | 0.217902 m | All 10,426 contact by 0.5 s | Contact at 0.5 s |
| DV3-BO-VS-05 | 0.460558 m | 401 sampled commands remain clear through 1.2 s | Clear for 3 s; minimum gap 0.135734 m |

BO-VS-05: taking the recorded first decision and then braking at 0.5 s contacts at 0.9 s for all three checked rudders. Its demonstrated immediate-stop clearance is below the filter's existing 0.15 m trigger even before the static gap subtraction. Initial astern is also excluded by the current no-traffic candidate gate. Neither limitation is changed here.

The recorded partial decisions are reproduced to numerical precision; durations are inferred by matching each saved after-state at the original 0.1 s collision cadence. The full 10,426-command grid uses 401 rudders and 26 RPM choices. These samples are **not a proof over continuous controls**. Static-panel survival is not full-episode success.

An empty rudder delay line initially fills with the first command (ship.py:219–234); the initial test therefore already permits immediate arrival at the servo. Later commands retain the randomized actuator delay. All source, cache, trace and scenario hashes plus the exact seeds and oracle parameters are in results.json.

The remaining lateral-drift issue is a terminal viability gap. A safe terminal region with a known controller is required by [Wabersich–Zeilinger, §4.1 Eq. 5f and Assumption 4.2](https://arxiv.org/html/1812.05506v4). Low surge speed or an additional 3 s predicted tail does not establish it. Terminal changes are deferred pending the broader trajectory-search screen; extending a tail alone largely repeats a previously unsuccessful horizon extension.

Reproduce from the project directory with a fresh output tag:

```powershell
python -B tools/diagnostics/safety/blind_zone_feasibility.py --tag blind_zone_reproduction
```
