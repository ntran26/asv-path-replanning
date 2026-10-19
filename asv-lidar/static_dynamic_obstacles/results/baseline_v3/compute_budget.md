# Compute budget of the baseline-v3 campaign (ideal conditions)

Generated 2026-10-17 08:29 by `tools/diagnostics/compute_budget.py`; the method is in its docstring.
Ideal conditions: on mains power, nothing else training or evaluating, no OneDrive sync or other heavy process.

## Ideal training speed by curriculum stage (steps/s)

| Learner | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO | 222.6 | 187.9 | 131.3 | 124.1 | 117.7 | 66.5 | 54.8 |
| RecurrentPPO | 89.4 | 95.7 | 99.4 | 113.1 | 108.9 | 61.7 | 52.5 |
| SAC | 22.6 | 21.3 | 20.4 | 20.3 | 20.0 | 17.8 | 17.1 |
| TQC | 16.9 | 15.6 | 16.3 | 16.1 | 15.9 | 14.9 | 14.2 |

Post-training evaluations, ideal (fastest measured): Tier 1 3 min, frozen suite 16 min, reference layouts 18 min, test set v4 9 min, final-model test set 9 min: **0.90 h per run**.

## By run (hours)

| Learner | Seed | Status | Ideal training | Ideal evaluations | **Ideal total** | Actual training | Actual evaluations | Remaining, ideal |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| PPO | 0 | done (final-model check to run) | 11.2 | 0.9 | **12.1** | 23.0 | 2.8 | 0.1 |
| PPO | 1 | done (final-model check to run) | 11.2 | 0.9 | **12.1** | 12.4 | 1.1 | 0.1 |
| PPO | 2 | trained, evaluations held | 11.2 | 0.9 | **12.1** | 12.4 | — | 0.9 |
| RecurrentPPO | 0 | done (final-model check to run) | 12.5 | 0.9 | **13.4** | 14.9 | 0.8 | 0.1 |
| RecurrentPPO | 1 | trained, evaluations held | 12.5 | 0.9 | **13.4** | 14.2 | — | 0.9 |
| RecurrentPPO | 2 | to do | 12.5 | 0.9 | **13.4** | — | — | 13.4 |
| SAC | 0 | done (final-model check to run) | 45.3 | 0.9 | **46.2** | 53.8 | 1.1 | 0.1 |
| SAC | 1 | done (final-model check to run) | 45.3 | 0.9 | **46.2** | 46.5 | 1.0 | 0.1 |
| SAC | 2 | training | 45.3 | 0.9 | **46.2** | 15.6 | — | 32.5 |
| TQC | 0 | done (final-model check to run) | 55.7 | 0.9 | **56.6** | 57.6 | 0.8 | 0.1 |
| TQC | 1 | to do | 55.7 | 0.9 | **56.6** | — | — | 56.6 |
| TQC | 2 | to do | 55.7 | 0.9 | **56.6** | — | — | 56.6 |

## By learner, three seeds (hours)

| Learner | **Ideal total** | Actual so far | Remaining, ideal |
|---|---:|---:|---:|
| PPO | **36.3** | 51.7 | 1.2 |
| RecurrentPPO | **40.3** | 29.8 | 14.5 |
| SAC | **138.5** | 117.9 | 32.8 |
| TQC | **169.9** | 58.3 | 113.4 |
| **All four** | **385.1** (16.0 days) | 257.8 (10.7 days) | **161.9** (6.7 days) |

Actual training excludes aborted attempts; actual evaluations are missing where the milestone log does not hold the full sequence (seed 0's older evaluation runs) or the evaluations are still held. Remaining work from 2026-10-17 08:29 at ideal speeds would end around Sat 24 Oct 02:22 if run back to back.
