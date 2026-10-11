# Test set v4: OFF, V16 and V16/V17 with the emergency stop on 1,000 scenarios

1000 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout | Stopped | Target strikes stopped ship |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1000 | 865 | 70 | 12 | 53 | 0 | 0 | 0 |
| v16 | 1000 | 904 | 46 | 25 | 25 | 0 | 0 | 0 |
| v16_estop | 1000 | 865 | 46 | 21 | 11 | 1 | 41 | 15 |
| v17_estop | 1000 | 866 | 42 | 23 | 12 | 0 | 46 | 11 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_estop_vs_off | 1000 | 865 | 865 | 38 | 38 |
| v16_estop_vs_v16 | 1000 | 904 | 865 | 0 | 39 |
| v16_vs_off | 1000 | 865 | 904 | 50 | 11 |
| v17_estop_vs_off | 1000 | 865 | 866 | 42 | 41 |
| v17_estop_vs_v16 | 1000 | 904 | 866 | 11 | 49 |

Outcome transitions (reference -> candidate):

- v16_estop_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:boundary -> stopped x4; collision:obstacle -> collision:boundary x9; collision:obstacle -> collision:by_target x1; collision:obstacle -> goal x20; collision:obstacle -> stopped x4; collision:target -> collision:boundary x2; collision:target -> collision:by_target x9; collision:target -> collision:obstacle x7; collision:target -> goal x17; collision:target -> stopped x11; goal -> collision:boundary x6; goal -> collision:by_target x5; goal -> collision:obstacle x3; goal -> collision:target x1; goal -> stopped x22; goal -> timeout x1
- v16_estop_vs_v16: collision:boundary -> collision:by_target x1; collision:boundary -> stopped x6; collision:boundary -> timeout x1; collision:obstacle -> stopped x4; collision:target -> collision:boundary x1; collision:target -> collision:by_target x8; collision:target -> collision:obstacle x1; collision:target -> stopped x4; goal -> collision:boundary x3; goal -> collision:by_target x6; goal -> collision:obstacle x3; goal -> stopped x27
- v16_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x10; collision:obstacle -> goal x23; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x7; collision:target -> goal x26; goal -> collision:boundary x6; goal -> collision:obstacle x2; goal -> collision:target x3
- v17_estop_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x2; collision:boundary -> stopped x3; collision:obstacle -> collision:boundary x10; collision:obstacle -> collision:by_target x1; collision:obstacle -> goal x23; collision:obstacle -> stopped x3; collision:target -> collision:boundary x3; collision:target -> collision:by_target x6; collision:target -> collision:obstacle x5; collision:target -> goal x17; collision:target -> stopped x15; goal -> collision:boundary x6; goal -> collision:by_target x4; goal -> collision:obstacle x4; goal -> collision:target x2; goal -> stopped x25
- v17_estop_vs_v16: collision:boundary -> collision:by_target x2; collision:boundary -> goal x3; collision:boundary -> stopped x5; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x6; collision:obstacle -> stopped x4; collision:target -> collision:boundary x1; collision:target -> collision:by_target x5; collision:target -> goal x2; collision:target -> stopped x5; goal -> collision:boundary x5; goal -> collision:by_target x4; goal -> collision:obstacle x8; goal -> stopped x32
