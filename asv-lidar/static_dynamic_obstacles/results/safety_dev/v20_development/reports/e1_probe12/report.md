# E1: V20 nominal and revision 1 on the 12-case probe

12 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 12 | 10 | 1 | 0 | 1 | 0 |
| v16 | 12 | 5 | 0 | 4 | 3 | 0 |
| v20_nominal | 12 | 3 | 0 | 3 | 4 | 2 |
| v20_r1 | 12 | 2 | 0 | 2 | 5 | 3 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_vs_off | 12 | 10 | 5 | 1 | 6 |
| v20_nominal_vs_off | 12 | 10 | 3 | 0 | 7 |
| v20_nominal_vs_v16 | 12 | 5 | 3 | 0 | 2 |
| v20_r1_vs_off | 12 | 10 | 2 | 0 | 8 |
| v20_r1_vs_v16 | 12 | 5 | 2 | 0 | 3 |

Outcome transitions (reference -> candidate):

- v16_vs_off: collision:obstacle -> goal x1; goal -> collision:boundary x4; goal -> collision:target x2
- v20_nominal_vs_off: collision:obstacle -> collision:target x1; collision:target -> timeout x1; goal -> collision:boundary x3; goal -> collision:target x3; goal -> timeout x1
- v20_nominal_vs_v16: collision:boundary -> timeout x1; collision:target -> timeout x1; goal -> collision:target x2
- v20_r1_vs_off: collision:obstacle -> collision:target x1; collision:target -> timeout x1; goal -> collision:boundary x2; goal -> collision:target x4; goal -> timeout x2
- v20_r1_vs_v16: collision:boundary -> timeout x2; collision:target -> timeout x1; goal -> collision:target x3
