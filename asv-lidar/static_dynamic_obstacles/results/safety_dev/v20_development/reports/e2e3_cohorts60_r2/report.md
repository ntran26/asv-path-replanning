# E2/E3: V20 revision 2 on the 60 held-out development cases

60 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 60 | 27 | 15 | 3 | 15 | 0 |
| v16 | 60 | 45 | 6 | 3 | 6 | 0 |
| v20_r2 | 60 | 40 | 6 | 4 | 6 | 4 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_vs_off | 60 | 27 | 45 | 18 | 0 |
| v20_r2_vs_off | 60 | 27 | 40 | 16 | 3 |
| v20_r2_vs_v16 | 60 | 45 | 40 | 2 | 7 |

Outcome transitions (reference -> candidate):

- v16_vs_off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:target x2; collision:obstacle -> goal x7; collision:target -> collision:obstacle x1; collision:target -> goal x10
- v20_r2_vs_off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> timeout x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:target x2; collision:obstacle -> goal x7; collision:obstacle -> timeout x1; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x8; collision:target -> timeout x1; goal -> collision:boundary x1; goal -> collision:obstacle x1; goal -> timeout x1
- v20_r2_vs_v16: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> timeout x1; collision:obstacle -> goal x1; collision:obstacle -> timeout x1; goal -> collision:boundary x4; goal -> collision:obstacle x1; goal -> timeout x2
