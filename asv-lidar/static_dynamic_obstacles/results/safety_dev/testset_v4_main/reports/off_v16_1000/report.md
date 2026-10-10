# Test set v4: fresh OFF and V16 on 1,000 scenarios

1000 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1000 | 865 | 70 | 12 | 53 | 0 |
| v16 | 1000 | 904 | 46 | 25 | 25 | 0 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_vs_off | 1000 | 865 | 904 | 50 | 11 |

Outcome transitions (reference -> candidate):

- v16_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x10; collision:obstacle -> goal x23; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x7; collision:target -> goal x26; goal -> collision:boundary x6; goal -> collision:obstacle x2; goal -> collision:target x3
