# Test set v4: fresh OFF, V16 and V20 revision 2 on 1,000 scenarios

1000 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1000 | 865 | 70 | 12 | 53 | 0 |
| v16 | 1000 | 904 | 46 | 25 | 25 | 0 |
| v20_r2 | 1000 | 881 | 39 | 33 | 24 | 23 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_vs_off | 1000 | 865 | 904 | 50 | 11 |
| v20_r2_vs_off | 1000 | 865 | 881 | 53 | 37 |
| v20_r2_vs_v16 | 1000 | 904 | 881 | 9 | 32 |

Outcome transitions (reference -> candidate):

- v16_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x10; collision:obstacle -> goal x23; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x7; collision:target -> goal x26; goal -> collision:boundary x6; goal -> collision:obstacle x2; goal -> collision:target x3
- v20_r2_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:boundary -> timeout x3; collision:obstacle -> collision:boundary x7; collision:obstacle -> goal x28; collision:obstacle -> timeout x6; collision:target -> collision:boundary x3; collision:target -> collision:obstacle x5; collision:target -> goal x24; collision:target -> timeout x3; goal -> collision:boundary x18; goal -> collision:obstacle x5; goal -> collision:target x3; goal -> timeout x11
- v20_r2_vs_v16: collision:boundary -> collision:obstacle x2; collision:boundary -> goal x4; collision:boundary -> timeout x4; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x4; collision:obstacle -> timeout x7; collision:target -> goal x1; goal -> collision:boundary x15; goal -> collision:obstacle x5; goal -> timeout x12
