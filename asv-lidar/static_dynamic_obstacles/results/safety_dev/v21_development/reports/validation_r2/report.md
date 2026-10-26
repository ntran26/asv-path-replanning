# V21 paired report: validation_r2

Tags: val_off_check, val_v16_bt_c1, val_v21_r1_c1, val_v21_r2_c1, val_v16_bt_c2, val_v21_r1_c2, val_v21_r2_c2, val_v16_bt_c3, val_v21_r1_c3, val_v21_r2_c3, val_v16_bt_c4, val_v21_r1_c4, val_v21_r2_c4, val_v16_bt_c5, val_v21_r1_c5, val_v21_r2_c5, val_v16_bt_c6, val_v21_r1_c6, val_v21_r2_c6. 270 cases with every mode (off, v16, v16_estop_bt, v21, v21_lc, v21_estop_bt, v21_r2, v21_r12, v21_r2_estop_bt, v21_r12_estop_bt); paired by case, episode seed and scenario digest.

## Outcomes

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 270 | 217 | 32 | 4 | 17 | 0 | 0 | 0 | 53 |
| v16 | 270 | 240 | 20 | 7 | 3 | 0 | 0 | 0 | 30 |
| v16_estop_bt | 270 | 236 | 20 | 6 | 1 | 1 | 6 | 0 | 28 |
| v21 | 270 | 240 | 16 | 10 | 4 | 0 | 0 | 0 | 30 |
| v21_lc | 270 | 239 | 16 | 11 | 4 | 0 | 0 | 0 | 31 |
| v21_estop_bt | 270 | 236 | 17 | 11 | 1 | 0 | 5 | 0 | 29 |
| v21_r2 | 270 | 238 | 18 | 10 | 4 | 0 | 0 | 0 | 32 |
| v21_r12 | 270 | 240 | 16 | 10 | 4 | 0 | 0 | 0 | 30 |
| v21_r2_estop_bt | 270 | 233 | 20 | 9 | 1 | 1 | 6 | 0 | 31 |
| v21_r12_estop_bt | 270 | 235 | 18 | 11 | 1 | 0 | 5 | 0 | 30 |

### coupled

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 150 | 111 | 26 | 2 | 11 | 0 | 0 | 0 | 39 |
| v16 | 150 | 128 | 16 | 3 | 3 | 0 | 0 | 0 | 22 |
| v16_estop_bt | 150 | 127 | 16 | 2 | 1 | 0 | 4 | 0 | 19 |
| v21_estop_bt | 150 | 127 | 13 | 6 | 1 | 0 | 3 | 0 | 20 |
| v21 | 150 | 128 | 12 | 6 | 4 | 0 | 0 | 0 | 22 |
| v21_lc | 150 | 127 | 12 | 7 | 4 | 0 | 0 | 0 | 23 |
| v21_r2 | 150 | 126 | 14 | 6 | 4 | 0 | 0 | 0 | 24 |
| v21_r12 | 150 | 128 | 12 | 6 | 4 | 0 | 0 | 0 | 22 |
| v21_r2_estop_bt | 150 | 124 | 16 | 5 | 1 | 0 | 4 | 0 | 22 |
| v21_r12_estop_bt | 150 | 126 | 14 | 6 | 1 | 0 | 3 | 0 | 21 |

### decoupled

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 120 | 106 | 6 | 2 | 6 | 0 | 0 | 0 | 14 |
| v16 | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| v16_estop_bt | 120 | 109 | 4 | 4 | 0 | 1 | 2 | 0 | 9 |
| v21 | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| v21_lc | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| v21_estop_bt | 120 | 109 | 4 | 5 | 0 | 0 | 2 | 0 | 9 |
| v21_r2 | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| v21_r12 | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| v21_r2_estop_bt | 120 | 109 | 4 | 4 | 0 | 1 | 2 | 0 | 9 |
| v21_r12_estop_bt | 120 | 109 | 4 | 5 | 0 | 0 | 2 | 0 | 9 |

## Paired changes

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
|---|---:|---:|---:|---:|---:|
| v16 vs off | 270 | 217 | 240 | 28 | 5 |
| v16_estop_bt vs off | 270 | 217 | 236 | 28 | 9 |
| v21 vs off | 270 | 217 | 240 | 28 | 5 |
| v21_lc vs off | 270 | 217 | 239 | 27 | 5 |
| v21_estop_bt vs off | 270 | 217 | 236 | 28 | 9 |
| v21_r2 vs off | 270 | 217 | 238 | 26 | 5 |
| v21_r12 vs off | 270 | 217 | 240 | 28 | 5 |
| v21_r2_estop_bt vs off | 270 | 217 | 233 | 26 | 10 |
| v21_r12_estop_bt vs off | 270 | 217 | 235 | 28 | 10 |
| off vs v16 | 270 | 240 | 217 | 5 | 28 |
| v16_estop_bt vs v16 | 270 | 240 | 236 | 0 | 4 |
| v21 vs v16 | 270 | 240 | 240 | 0 | 0 |
| v21_lc vs v16 | 270 | 240 | 239 | 0 | 1 |
| v21_estop_bt vs v16 | 270 | 240 | 236 | 0 | 4 |
| v21_r2 vs v16 | 270 | 240 | 238 | 1 | 3 |
| v21_r12 vs v16 | 270 | 240 | 240 | 1 | 1 |
| v21_r2_estop_bt vs v16 | 270 | 240 | 233 | 1 | 8 |
| v21_r12_estop_bt vs v16 | 270 | 240 | 235 | 1 | 6 |
| off vs v16_estop_bt | 270 | 236 | 217 | 9 | 28 |
| v16 vs v16_estop_bt | 270 | 236 | 240 | 4 | 0 |
| v21 vs v16_estop_bt | 270 | 236 | 240 | 4 | 0 |
| v21_lc vs v16_estop_bt | 270 | 236 | 239 | 4 | 1 |
| v21_estop_bt vs v16_estop_bt | 270 | 236 | 236 | 0 | 0 |
| v21_r2 vs v16_estop_bt | 270 | 236 | 238 | 5 | 3 |
| v21_r12 vs v16_estop_bt | 270 | 236 | 240 | 5 | 1 |
| v21_r2_estop_bt vs v16_estop_bt | 270 | 236 | 233 | 1 | 4 |
| v21_r12_estop_bt vs v16_estop_bt | 270 | 236 | 235 | 1 | 2 |

Outcome transitions (reference -> candidate):

- v16 vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x1; goal -> collision:obstacle x3; goal -> collision:target x1
- v16_estop_bt vs off: collision:boundary -> collision:by_target x1; collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x14; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; collision:target -> stopped x2; goal -> collision:boundary x3; goal -> collision:obstacle x3; goal -> collision:target x1; goal -> stopped x2
- v21 vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x5; collision:obstacle -> collision:target x1; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x2; goal -> collision:obstacle x2; goal -> collision:target x1
- v21_lc vs off: collision:boundary -> collision:obstacle x1; collision:obstacle -> collision:boundary x5; collision:obstacle -> collision:target x1; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x2; goal -> collision:obstacle x2; goal -> collision:target x1
- v21_estop_bt vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x5; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; collision:target -> stopped x2; goal -> collision:boundary x4; goal -> collision:obstacle x2; goal -> collision:target x1; goal -> stopped x2
- v21_r2 vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x4; collision:obstacle -> collision:target x1; collision:obstacle -> goal x12; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x3; goal -> collision:obstacle x1; goal -> collision:target x1
- v21_r12 vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x4; collision:obstacle -> collision:target x1; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x3; goal -> collision:obstacle x1; goal -> collision:target x1
- v21_r2_estop_bt vs off: collision:boundary -> collision:by_target x1; collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x12; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; collision:target -> stopped x2; goal -> collision:boundary x5; goal -> collision:obstacle x2; goal -> collision:target x1; goal -> stopped x2
- v21_r12_estop_bt vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x4; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; collision:target -> stopped x2; goal -> collision:boundary x5; goal -> collision:obstacle x2; goal -> collision:target x1; goal -> stopped x2
- off vs v16: collision:boundary -> collision:obstacle x3; collision:boundary -> collision:target x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x14; goal -> collision:target x13
- v16_estop_bt vs v16: collision:boundary -> collision:by_target x1; collision:boundary -> stopped x2; collision:target -> stopped x2; goal -> collision:boundary x2; goal -> stopped x2
- v21 vs v16: collision:obstacle -> collision:boundary x3; collision:obstacle -> collision:target x1
- v21_lc vs v16: collision:obstacle -> collision:boundary x3; collision:obstacle -> collision:target x1; goal -> collision:boundary x1
- v21_estop_bt vs v16: collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x3; collision:target -> stopped x2; goal -> collision:boundary x2; goal -> stopped x2
- v21_r2 vs v16: collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x1; goal -> collision:boundary x2; goal -> collision:obstacle x1
- v21_r12 vs v16: collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:target x1; collision:obstacle -> goal x1; goal -> collision:boundary x1
- v21_r2_estop_bt vs v16: collision:boundary -> collision:by_target x1; collision:boundary -> stopped x2; collision:obstacle -> collision:boundary x1; collision:obstacle -> goal x1; collision:target -> stopped x2; goal -> collision:boundary x4; goal -> collision:obstacle x2; goal -> stopped x2
- v21_r12_estop_bt vs v16: collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x1; collision:target -> stopped x2; goal -> collision:boundary x3; goal -> collision:obstacle x1; goal -> stopped x2
- off vs v16_estop_bt: collision:boundary -> collision:obstacle x2; collision:boundary -> collision:target x1; collision:boundary -> goal x3; collision:by_target -> collision:boundary x1; collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x14; goal -> collision:target x13; stopped -> collision:boundary x1; stopped -> collision:obstacle x1; stopped -> collision:target x2; stopped -> goal x2
- v16 vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21 vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> collision:target x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21_lc vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> collision:target x1; goal -> collision:boundary x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21_estop_bt vs v16_estop_bt: collision:by_target -> stopped x1; collision:obstacle -> collision:boundary x3; stopped -> collision:boundary x2
- v21_r2 vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x1; goal -> collision:boundary x2; goal -> collision:obstacle x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21_r12 vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> collision:target x1; collision:obstacle -> goal x1; goal -> collision:boundary x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21_r2_estop_bt vs v16_estop_bt: collision:obstacle -> collision:boundary x1; collision:obstacle -> goal x1; goal -> collision:boundary x2; goal -> collision:obstacle x2
- v21_r12_estop_bt vs v16_estop_bt: collision:by_target -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x1; stopped -> collision:boundary x2

