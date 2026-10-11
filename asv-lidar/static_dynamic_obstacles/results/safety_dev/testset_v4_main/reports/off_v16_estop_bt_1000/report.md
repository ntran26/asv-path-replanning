# Test set v4: OFF, V16 and V16/V17 with the boundary-and-targets emergency stop on 1,000 scenarios

1000 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout | Stopped | Target strikes stopped ship |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1000 | 865 | 70 | 12 | 53 | 0 | 0 | 0 |
| v16 | 1000 | 904 | 46 | 25 | 25 | 0 | 0 | 0 |
| v16_estop_bt | 1000 | 893 | 46 | 20 | 11 | 1 | 21 | 8 |
| v17_estop_bt | 1000 | 894 | 42 | 23 | 12 | 0 | 23 | 6 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_estop_bt_vs_off | 1000 | 865 | 893 | 47 | 19 |
| v16_estop_bt_vs_v16 | 1000 | 904 | 893 | 0 | 11 |
| v16_vs_off | 1000 | 865 | 904 | 50 | 11 |
| v17_estop_bt_vs_off | 1000 | 865 | 894 | 52 | 23 |
| v17_estop_bt_vs_v16 | 1000 | 904 | 894 | 13 | 23 |

Outcome transitions (reference -> candidate):

- v16_estop_bt_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:boundary -> stopped x4; collision:obstacle -> collision:boundary x9; collision:obstacle -> collision:by_target x1; collision:obstacle -> goal x22; collision:obstacle -> stopped x1; collision:target -> collision:boundary x2; collision:target -> collision:by_target x5; collision:target -> collision:obstacle x8; collision:target -> goal x24; collision:target -> stopped x7; goal -> collision:boundary x5; goal -> collision:by_target x2; goal -> collision:obstacle x1; goal -> collision:target x1; goal -> stopped x9; goal -> timeout x1
- v16_estop_bt_vs_v16: collision:boundary -> collision:by_target x2; collision:boundary -> stopped x6; collision:boundary -> timeout x1; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:by_target x5; collision:target -> collision:obstacle x1; collision:target -> stopped x7; goal -> collision:boundary x3; goal -> collision:by_target x1; goal -> stopped x7
- v16_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x10; collision:obstacle -> goal x23; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x7; collision:target -> goal x26; goal -> collision:boundary x6; goal -> collision:obstacle x2; goal -> collision:target x3
- v17_estop_bt_vs_off: collision:boundary -> collision:target x3; collision:boundary -> goal x2; collision:boundary -> stopped x3; collision:obstacle -> collision:boundary x10; collision:obstacle -> goal x23; collision:obstacle -> stopped x2; collision:target -> collision:boundary x3; collision:target -> collision:by_target x4; collision:target -> collision:obstacle x4; collision:target -> goal x27; collision:target -> stopped x8; goal -> collision:boundary x6; goal -> collision:by_target x2; goal -> collision:obstacle x3; goal -> collision:target x2; goal -> stopped x10
- v17_estop_bt_vs_v16: collision:boundary -> collision:by_target x1; collision:boundary -> goal x3; collision:boundary -> stopped x6; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x7; collision:obstacle -> stopped x2; collision:target -> collision:boundary x1; collision:target -> collision:by_target x4; collision:target -> goal x3; collision:target -> stopped x5; goal -> collision:boundary x5; goal -> collision:by_target x1; goal -> collision:obstacle x7; goal -> stopped x10
