# V20 development results

Plan, pre-registered gates and revision 1: [planning/SAFETY_V20_PLAN.md](../../../planning/SAFETY_V20_PLAN.md) (section 13 records the gate results and the revised configuration). Candidate: `src/safety_v20.py`, V16 plus a committed stop-terminal contingency. In the phase-1 default it replaces only V16's unchecked `last certificate` and `no escape` decisions; revision 1 (`REVISION1_OPTIONS`, runner mode `v20_r1`) also enforces the certificate at checked decisions (gatekeeper). Certificate: `src/safety_v20_contingency.py`. Allowance tables: `src/safety_v20_tubes.py`.

This directory holds development evidence only. Test-set-v4 runs are under [`../testset_v4_main/`](../testset_v4_main/). Nothing here was tuned on a test-set outcome.

## Runtime environment

Python 3.11.17 virtual environment with numpy 1.26.4, torch 2.2.1, gymnasium 0.29.1, stable-baselines3 2.3.2, sb3-contrib 2.3.0 (PyPI; the PyTorch CPU index is blocked by the egress proxy). Each run manifest freezes the interpreter, platform and installed package versions. Historical runs used Python 3.10 on Windows; fresh V16 reproduces historical probe outcomes but not bit-identical commands (rudder differences up to about 4e-5 from the first decision), so only fresh runs in this environment are paired. One Torch thread and one native thread per process; at most two evaluation processes.

## Contents

| Path | Stage | Summary |
| --- | --- | --- |
| [`phase1_saved_cascade_audit/`](phase1_saved_cascade_audit/REPORT.md) | Phase 1 | 21 of 22 V16 contacts in the 72 development cases follow unchecked decisions; braking, rest and the first intervention do not separate losses from rescues |
| [`allowance_calibration/g1_v1/`](allowance_calibration/g1_v1/calibration.json) | Gate G1 | Split-conformal 95 % forecast-error tables (own ship, constant-velocity targets) |
| `saved_state_replay/g2g3_v1/` | Gate G2/G3 | Aborted before any result (too slow; see `ABORTED.md`) |
| `saved_state_replay/g2g3_v2/` | Gate G2/G3 | Aborted before any result (memory; see `ABORTED.md`) |
| [`saved_state_replay/g2g3_v3/`](saved_state_replay/g2g3_v3/replay.json) | Gate G2/G3 | Certificate replay at 3,569 saved V16 decisions in 68 development episodes, for both allowance settings |
| [`offline_gates/gates_v1/`](offline_gates/gates_v1/gates.json) | Gates G0-G3 | Default (calibrated) fails, with one-step survival 0.805 when a target is present and 3/21 precursors certified; the nominal option meets the rule at 0.920/0.940 and 11/21 |
| [`recheck_attribution/attribution_v1/`](recheck_attribution/attribution_v1/attribution.json) | Diagnosis | 194 nominal recheck failures: own state 104, world update 58 (55 target), both 12, interaction 20 |
| [`offline_variants/variants_v1/`](offline_variants/variants_v1/margin.json) | Diagnosis | One-step survival versus commit margin (0.965/0.969 at 0.127 m); `options_episodes/` holds the near-V16 certified-option analysis of revision 1 |
| `offline_variants/variants_v2/` | Check | Identical rerun of the margin analysis after the merge (see `NOTE.md`) |
| [`runs/e0_probe12_off_v16/`](runs/e0_probe12_off_v16/manifest.json) | E0 | Fresh OFF 10/12, fresh V16 5/12 on the 12-case probe |
