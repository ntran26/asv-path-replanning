# V20 development results

Plan and pre-registered gates: [planning/SAFETY_V20_PLAN.md](../../../planning/SAFETY_V20_PLAN.md). Candidate: `src/safety_v20.py` (V16 plus a committed stop-terminal contingency; replaces only V16's unchecked `last certificate` and `no escape` decisions), certificate `src/safety_v20_contingency.py`, allowance tables `src/safety_v20_tubes.py`.

This directory holds development evidence only. Test-set-v4 runs are under [`../testset_v4_main/`](../testset_v4_main/). Nothing here was tuned on a test-set outcome.

## Runtime environment

Python 3.11.17 virtual environment with numpy 1.26.4, torch 2.2.1, gymnasium 0.29.1, stable-baselines3 2.3.2, sb3-contrib 2.3.0 (PyPI; the PyTorch CPU index is blocked by the egress proxy). Each run manifest freezes the interpreter, platform and installed package versions. Historical runs used Python 3.10 on Windows; fresh V16 reproduces historical probe outcomes but not bit-identical commands (rudder differences up to about 4e-5 from the first decision), so only fresh runs in this environment are paired. One Torch thread and one native thread per process; at most two evaluation processes.

## Contents

| Path | Stage | Summary |
| --- | --- | --- |
| [`phase1_saved_cascade_audit/`](phase1_saved_cascade_audit/REPORT.md) | Phase 1 | 21 of 22 V16 contacts in the 72 development cases follow unchecked decisions; braking, rest and the first intervention do not separate losses from rescues |
| [`allowance_calibration/g1_v1/`](allowance_calibration/g1_v1/calibration.json) | Gate G1 | Split-conformal 95 % forecast-error tables (own ship, constant-velocity targets) |
| `saved_state_replay/g2g3_v1/` | Gate G2/G3 | Aborted before any result (see `ABORTED.md`) |
| `saved_state_replay/g2g3_v2/` | Gate G2/G3 | Certificate replay at saved V16 states |
| [`runs/e0_probe12_off_v16/`](runs/e0_probe12_off_v16/manifest.json) | E0 | Fresh OFF 10/12, fresh V16 5/12 on the 12-case probe |
