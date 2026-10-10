# E1b: V20 revision 2 on the 12-case probe

12 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 12 | 10 | 1 | 0 | 1 | 0 |
| v16 | 12 | 5 | 0 | 4 | 3 | 0 |
| v20_r2 | 12 | 6 | 0 | 2 | 3 | 1 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_vs_off | 12 | 10 | 5 | 1 | 6 |
| v20_r2_vs_off | 12 | 10 | 6 | 1 | 5 |
| v20_r2_vs_v16 | 12 | 5 | 6 | 1 | 0 |

Outcome transitions (reference -> candidate):

- v16_vs_off: collision:obstacle -> goal x1; goal -> collision:boundary x4; goal -> collision:target x2
- v20_r2_vs_off: collision:obstacle -> goal x1; goal -> collision:boundary x2; goal -> collision:target x2; goal -> timeout x1
- v20_r2_vs_v16: collision:boundary -> goal x1; collision:boundary -> timeout x1
