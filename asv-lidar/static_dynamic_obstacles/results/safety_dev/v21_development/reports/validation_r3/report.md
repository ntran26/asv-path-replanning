# V21 paired report: validation_r3

Tags: val_off_check, val_v16_bt_c1, val_v21_r3_c1, val_v16_bt_c2, val_v21_r3_c2, val_v16_bt_c3, val_v21_r3_c3, val_v16_bt_c4, val_v21_r3_c4, val_v16_bt_c5, val_v21_r3_c5, val_v16_bt_c6, val_v21_r3_c6. 270 cases with every mode (off, v16, v16_estop_bt, v21_off, v21_r3); paired by case, episode seed and scenario digest.

## Outcomes

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 270 | 217 | 32 | 4 | 17 | 0 | 0 | 0 | 53 |
| v16 | 270 | 240 | 20 | 7 | 3 | 0 | 0 | 0 | 30 |
| v16_estop_bt | 270 | 236 | 20 | 6 | 1 | 1 | 6 | 0 | 28 |
| v21_off | 270 | 240 | 20 | 7 | 3 | 0 | 0 | 0 | 30 |
| v21_r3 | 270 | 240 | 20 | 7 | 3 | 0 | 0 | 0 | 30 |

### coupled

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 150 | 111 | 26 | 2 | 11 | 0 | 0 | 0 | 39 |
| v16 | 150 | 128 | 16 | 3 | 3 | 0 | 0 | 0 | 22 |
| v16_estop_bt | 150 | 127 | 16 | 2 | 1 | 0 | 4 | 0 | 19 |
| v21_off | 150 | 128 | 16 | 3 | 3 | 0 | 0 | 0 | 22 |
| v21_r3 | 150 | 128 | 16 | 3 | 3 | 0 | 0 | 0 | 22 |

### decoupled

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 120 | 106 | 6 | 2 | 6 | 0 | 0 | 0 | 14 |
| v16 | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| v16_estop_bt | 120 | 109 | 4 | 4 | 0 | 1 | 2 | 0 | 9 |
| v21_off | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| v21_r3 | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |

## Paired changes

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
|---|---:|---:|---:|---:|---:|
| v16 vs off | 270 | 217 | 240 | 28 | 5 |
| v16_estop_bt vs off | 270 | 217 | 236 | 28 | 9 |
| v21_off vs off | 270 | 217 | 240 | 28 | 5 |
| v21_r3 vs off | 270 | 217 | 240 | 28 | 5 |
| off vs v16 | 270 | 240 | 217 | 5 | 28 |
| v16_estop_bt vs v16 | 270 | 240 | 236 | 0 | 4 |
| v21_off vs v16 | 270 | 240 | 240 | 0 | 0 |
| v21_r3 vs v16 | 270 | 240 | 240 | 0 | 0 |
| off vs v16_estop_bt | 270 | 236 | 217 | 9 | 28 |
| v16 vs v16_estop_bt | 270 | 236 | 240 | 4 | 0 |
| v21_off vs v16_estop_bt | 270 | 236 | 240 | 4 | 0 |
| v21_r3 vs v16_estop_bt | 270 | 236 | 240 | 4 | 0 |

Outcome transitions (reference -> candidate):

- v16 vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x1; goal -> collision:obstacle x3; goal -> collision:target x1
- v16_estop_bt vs off: collision:boundary -> collision:by_target x1; collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x14; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; collision:target -> stopped x2; goal -> collision:boundary x3; goal -> collision:obstacle x3; goal -> collision:target x1; goal -> stopped x2
- v21_off vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x1; goal -> collision:obstacle x3; goal -> collision:target x1
- v21_r3 vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x1; goal -> collision:obstacle x3; goal -> collision:target x1
- off vs v16: collision:boundary -> collision:obstacle x3; collision:boundary -> collision:target x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x14; goal -> collision:target x13
- v16_estop_bt vs v16: collision:boundary -> collision:by_target x1; collision:boundary -> stopped x2; collision:target -> stopped x2; goal -> collision:boundary x2; goal -> stopped x2
- v21_off vs v16: 
- v21_r3 vs v16: 
- off vs v16_estop_bt: collision:boundary -> collision:obstacle x2; collision:boundary -> collision:target x1; collision:boundary -> goal x3; collision:by_target -> collision:boundary x1; collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x14; goal -> collision:target x13; stopped -> collision:boundary x1; stopped -> collision:obstacle x1; stopped -> collision:target x2; stopped -> goal x2
- v16 vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21_off vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21_r3 vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2

