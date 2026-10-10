# Revision 3 on all 72 development cases

72 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 72 | 37 | 16 | 3 | 16 | 0 |
| v16 | 72 | 50 | 6 | 7 | 9 | 0 |
| v20_r3 | 72 | 46 | 8 | 7 | 9 | 2 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_vs_off | 72 | 37 | 50 | 19 | 6 |
| v20_r3_vs_off | 72 | 37 | 46 | 16 | 7 |
| v20_r3_vs_v16 | 72 | 50 | 46 | 2 | 6 |

Outcome transitions (reference -> candidate):

- v16_vs_off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:target x2; collision:obstacle -> goal x8; collision:target -> collision:obstacle x1; collision:target -> goal x10; goal -> collision:boundary x4; goal -> collision:target x2
- v20_r3_vs_off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> timeout x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:target x2; collision:obstacle -> goal x7; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x8; collision:target -> timeout x1; goal -> collision:boundary x4; goal -> collision:obstacle x1; goal -> collision:target x2
- v20_r3_vs_v16: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x2; collision:boundary -> timeout x1; collision:obstacle -> collision:boundary x1; goal -> collision:boundary x3; goal -> collision:obstacle x2; goal -> timeout x1
