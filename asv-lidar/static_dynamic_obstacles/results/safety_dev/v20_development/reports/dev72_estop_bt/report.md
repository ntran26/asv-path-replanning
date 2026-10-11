# Emergency stop, boundary-and-targets trigger: V16 and V17 on the 72 development cases

72 scenes; fresh runs paired by case ID, episode seed and scenario digest. Rescue/loss are relative to the fresh OFF record of the same scene.

| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout | Stopped | Target strikes stopped ship |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 72 | 37 | 16 | 3 | 16 | 0 | 0 | 0 |
| v16 | 72 | 50 | 6 | 7 | 9 | 0 | 0 | 0 |
| v16_estop_bt | 72 | 50 | 7 | 4 | 2 | 1 | 4 | 4 |
| v17_estop_bt | 72 | 49 | 5 | 6 | 2 | 0 | 7 | 3 |

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16_estop_bt_vs_off | 72 | 37 | 50 | 19 | 6 |
| v16_estop_bt_vs_v16 | 72 | 50 | 50 | 0 | 0 |
| v16_vs_off | 72 | 37 | 50 | 19 | 6 |
| v17_estop_bt_vs_off | 72 | 37 | 49 | 21 | 9 |
| v17_estop_bt_vs_v16 | 72 | 50 | 49 | 5 | 6 |

Outcome transitions (reference -> candidate):

- v16_estop_bt_vs_off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:by_target x1; collision:obstacle -> goal x8; collision:target -> collision:by_target x1; collision:target -> collision:obstacle x1; collision:target -> goal x10; collision:target -> stopped x2; goal -> collision:boundary x2; goal -> collision:by_target x2; goal -> stopped x1; goal -> timeout x1
- v16_estop_bt_vs_v16: collision:boundary -> stopped x2; collision:boundary -> timeout x1; collision:target -> collision:by_target x4; collision:target -> collision:obstacle x1; collision:target -> stopped x2
- v16_vs_off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:target x2; collision:obstacle -> goal x8; collision:target -> collision:obstacle x1; collision:target -> goal x10; goal -> collision:boundary x4; goal -> collision:target x2
- v17_estop_bt_vs_off: collision:boundary -> goal x2; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:by_target x1; collision:obstacle -> goal x9; collision:target -> collision:by_target x1; collision:target -> goal x10; collision:target -> stopped x3; goal -> collision:boundary x4; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x3
- v17_estop_bt_vs_v16: collision:boundary -> goal x1; collision:boundary -> stopped x2; collision:obstacle -> goal x2; collision:target -> collision:by_target x3; collision:target -> goal x2; collision:target -> stopped x2; goal -> collision:boundary x2; goal -> collision:obstacle x1; goal -> stopped x3
