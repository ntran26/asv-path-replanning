# Test set v4: fresh OFF, V16 and V20 revision 2

Frozen selection: [`selection.json`](selection.json), all 1,000 scenarios of test set v4 in definition order. [`inventory.json`](inventory.json) describes the suite. Runs read the trusted saved cache, verified against its definition and manifest digest, and never regenerate a scene. Test set v4 overlaps earlier v2/v3 development exposure, so it is not wholly unseen.

Protocol and decisions: [planning/SAFETY_V20_PLAN.md](../../../planning/SAFETY_V20_PLAN.md) section 17. Runs used the Python 3.11.17 environment of the V20 development runs, with one thread per process, at most two processes, and compact gzip traces.

| Path | Content |
| --- | --- |
| `runs/ts4_off_v16_c01`-`c10` | Fresh OFF and V16, 100 scenarios per chunk |
| `runs/ts4_r2_c01`-`c19`, `c20b`, `c21b`, `c22`-`c40` | V20 revision 2 (`REVISION2_OPTIONS`), 25 scenarios per chunk |
| `runs/ts4_r2_c20`, `runs/ts4_r2_c21` | Interrupted by a container restart; kept as records (`INTERRUPTED.md`), excluded from reports |
| [`reports/off_v16_1000/`](reports/off_v16_1000/report.md) | Paired OFF/V16 report |
| [`reports/off_v16_r2_1000/`](reports/off_v16_r2_1000/report.md) | Paired OFF/V16/revision-2 report |

| Controller | Goals | Obstacle | Boundary | Target | Timeout | Rescued SAC failures | Lost SAC successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 865 | 70 | 12 | 53 | 0 | - | - |
| V16 | 904 | 46 | 25 | 25 | 0 | 50 | 11 |
| V20 revision 2 | 881 | 39 | 33 | 24 | 23 | 53 | 37 |

Revision 2 has the same contact count as V16 (96) with 23 fewer goals. V16 remains the reference.

## Emergency-stop protocol (2026-10-11)

`runs/ts4_estop_c01`-`c20` run V16 and V17 with the permanent emergency stop ([protocol](../../../planning/SAFETY_ESTOP_PROTOCOL.md)). Report: [`reports/off_v16_estop_1000/`](reports/off_v16_estop_1000/report.md).

| Controller | Goals | Own-ship contacts | Target strikes on the stopped ship | Stopped | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| V16 | 904 | 96 | - | - | 0 |
| V16 + stop | 865 | 78 | 15 | 41 | 1 |
| V17 + stop | 866 | 77 | 11 | 46 | 0 |

