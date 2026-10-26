# V21 paired report: validation_e1s

Tags: val_off_check, val_v16_bt_c1, val_v21_e1_c1, val_v21_e1s_c1, val_v16_bt_c2, val_v21_e1_c2, val_v21_e1s_c2, val_v16_bt_c3, val_v21_e1_c3, val_v21_e1s_c3, val_v16_bt_c4, val_v21_e1_c4, val_v21_e1s_c4, val_v16_bt_c5, val_v21_e1_c5, val_v21_e1s_c5, val_v16_bt_c6, val_v21_e1_c6, val_v21_e1s_c6. 270 cases with every mode (off, v16, v16_estop_bt, v21_estop_off, v21_e1_estop_bt, v21_r3_e1_estop_bt, v21_e1s15_estop, v21_e1s25_estop, v21_e1s35_estop); paired by case, episode seed and scenario digest.

## Outcomes

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts | own_vessel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 270 | 217 | 32 | 4 | 17 | 0 | 0 | 0 | 53 | 53 |
| v16 | 270 | 240 | 20 | 7 | 3 | 0 | 0 | 0 | 30 | 30 |
| v16_estop_bt | 270 | 236 | 20 | 6 | 1 | 1 | 6 | 0 | 28 | 27 |
| v21_estop_off | 270 | 236 | 20 | 6 | 1 | 1 | 6 | 0 | 28 | 27 |
| v21_e1_estop_bt | 270 | 222 | 18 | 6 | 0 | 1 | 23 | 0 | 25 | 24 |
| v21_r3_e1_estop_bt | 270 | 222 | 18 | 7 | 0 | 1 | 22 | 0 | 26 | 25 |
| v21_e1s15_estop | 270 | 213 | 17 | 6 | 0 | 4 | 30 | 0 | 27 | 23 |
| v21_e1s25_estop | 270 | 213 | 14 | 6 | 0 | 1 | 36 | 0 | 21 | 20 |
| v21_e1s35_estop | 270 | 212 | 13 | 6 | 0 | 1 | 38 | 0 | 20 | 19 |

### coupled

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts | own_vessel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 150 | 111 | 26 | 2 | 11 | 0 | 0 | 0 | 39 | 39 |
| v16 | 150 | 128 | 16 | 3 | 3 | 0 | 0 | 0 | 22 | 22 |
| v16_estop_bt | 150 | 127 | 16 | 2 | 1 | 0 | 4 | 0 | 19 | 19 |
| v21_r3_e1_estop_bt | 150 | 118 | 15 | 3 | 0 | 1 | 13 | 0 | 19 | 18 |
| v21_estop_off | 150 | 127 | 16 | 2 | 1 | 0 | 4 | 0 | 19 | 19 |
| v21_e1_estop_bt | 150 | 118 | 15 | 3 | 0 | 1 | 13 | 0 | 19 | 18 |
| v21_e1s35_estop | 150 | 114 | 11 | 3 | 0 | 1 | 21 | 0 | 15 | 14 |
| v21_e1s15_estop | 150 | 114 | 15 | 3 | 0 | 3 | 15 | 0 | 21 | 18 |
| v21_e1s25_estop | 150 | 114 | 12 | 3 | 0 | 1 | 20 | 0 | 16 | 15 |

### decoupled

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts | own_vessel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 120 | 106 | 6 | 2 | 6 | 0 | 0 | 0 | 14 | 14 |
| v16 | 120 | 112 | 4 | 4 | 0 | 0 | 0 | 0 | 8 | 8 |
| v16_estop_bt | 120 | 109 | 4 | 4 | 0 | 1 | 2 | 0 | 9 | 8 |
| v21_estop_off | 120 | 109 | 4 | 4 | 0 | 1 | 2 | 0 | 9 | 8 |
| v21_e1_estop_bt | 120 | 104 | 3 | 3 | 0 | 0 | 10 | 0 | 6 | 6 |
| v21_r3_e1_estop_bt | 120 | 104 | 3 | 4 | 0 | 0 | 9 | 0 | 7 | 7 |
| v21_e1s15_estop | 120 | 99 | 2 | 3 | 0 | 1 | 15 | 0 | 6 | 5 |
| v21_e1s25_estop | 120 | 99 | 2 | 3 | 0 | 0 | 16 | 0 | 5 | 5 |
| v21_e1s35_estop | 120 | 98 | 2 | 3 | 0 | 0 | 17 | 0 | 5 | 5 |

## Paired changes

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
|---|---:|---:|---:|---:|---:|
| v16 vs off | 270 | 217 | 240 | 28 | 5 |
| v16_estop_bt vs off | 270 | 217 | 236 | 28 | 9 |
| v21_estop_off vs off | 270 | 217 | 236 | 28 | 9 |
| v21_e1_estop_bt vs off | 270 | 217 | 222 | 22 | 17 |
| v21_r3_e1_estop_bt vs off | 270 | 217 | 222 | 22 | 17 |
| v21_e1s15_estop vs off | 270 | 217 | 213 | 17 | 21 |
| v21_e1s25_estop vs off | 270 | 217 | 213 | 17 | 21 |
| v21_e1s35_estop vs off | 270 | 217 | 212 | 17 | 22 |
| off vs v16 | 270 | 240 | 217 | 5 | 28 |
| v16_estop_bt vs v16 | 270 | 240 | 236 | 0 | 4 |
| v21_estop_off vs v16 | 270 | 240 | 236 | 0 | 4 |
| v21_e1_estop_bt vs v16 | 270 | 240 | 222 | 0 | 18 |
| v21_r3_e1_estop_bt vs v16 | 270 | 240 | 222 | 0 | 18 |
| v21_e1s15_estop vs v16 | 270 | 240 | 213 | 0 | 27 |
| v21_e1s25_estop vs v16 | 270 | 240 | 213 | 0 | 27 |
| v21_e1s35_estop vs v16 | 270 | 240 | 212 | 0 | 28 |
| off vs v16_estop_bt | 270 | 236 | 217 | 9 | 28 |
| v16 vs v16_estop_bt | 270 | 236 | 240 | 4 | 0 |
| v21_estop_off vs v16_estop_bt | 270 | 236 | 236 | 0 | 0 |
| v21_e1_estop_bt vs v16_estop_bt | 270 | 236 | 222 | 0 | 14 |
| v21_r3_e1_estop_bt vs v16_estop_bt | 270 | 236 | 222 | 0 | 14 |
| v21_e1s15_estop vs v16_estop_bt | 270 | 236 | 213 | 0 | 23 |
| v21_e1s25_estop vs v16_estop_bt | 270 | 236 | 213 | 0 | 23 |
| v21_e1s35_estop vs v16_estop_bt | 270 | 236 | 212 | 0 | 24 |

Outcome transitions (reference -> candidate):

- v16 vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x14; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; goal -> collision:boundary x1; goal -> collision:obstacle x3; goal -> collision:target x1
- v16_estop_bt vs off: collision:boundary -> collision:by_target x1; collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x14; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; collision:target -> stopped x2; goal -> collision:boundary x3; goal -> collision:obstacle x3; goal -> collision:target x1; goal -> stopped x2
- v21_estop_off vs off: collision:boundary -> collision:by_target x1; collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x2; collision:obstacle -> goal x14; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x1; collision:target -> goal x13; collision:target -> stopped x2; goal -> collision:boundary x3; goal -> collision:obstacle x3; goal -> collision:target x1; goal -> stopped x2
- v21_e1_estop_bt vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x2; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x13; collision:obstacle -> stopped x2; collision:target -> goal x8; collision:target -> stopped x9; goal -> collision:boundary x3; goal -> collision:by_target x1; goal -> collision:obstacle x3; goal -> stopped x10
- v21_r3_e1_estop_bt vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x13; collision:obstacle -> stopped x2; collision:target -> goal x8; collision:target -> stopped x9; goal -> collision:boundary x3; goal -> collision:by_target x1; goal -> collision:obstacle x3; goal -> stopped x10
- v21_e1s15_estop vs off: collision:boundary -> collision:by_target x1; collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x1; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x11; collision:obstacle -> stopped x5; collision:target -> collision:obstacle x1; collision:target -> goal x5; collision:target -> stopped x11; goal -> collision:boundary x3; goal -> collision:by_target x3; goal -> collision:obstacle x2; goal -> stopped x13
- v21_e1s25_estop vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x2; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x11; collision:obstacle -> stopped x6; collision:target -> goal x5; collision:target -> stopped x12; goal -> collision:boundary x3; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x16
- v21_e1s35_estop vs off: collision:boundary -> collision:obstacle x1; collision:boundary -> goal x1; collision:boundary -> stopped x2; collision:obstacle -> collision:boundary x3; collision:obstacle -> goal x11; collision:obstacle -> stopped x7; collision:target -> goal x5; collision:target -> stopped x12; goal -> collision:boundary x3; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x17
- off vs v16: collision:boundary -> collision:obstacle x3; collision:boundary -> collision:target x1; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x14; goal -> collision:target x13
- v16_estop_bt vs v16: collision:boundary -> collision:by_target x1; collision:boundary -> stopped x2; collision:target -> stopped x2; goal -> collision:boundary x2; goal -> stopped x2
- v21_estop_off vs v16: collision:boundary -> collision:by_target x1; collision:boundary -> stopped x2; collision:target -> stopped x2; goal -> collision:boundary x2; goal -> stopped x2
- v21_e1_estop_bt vs v16: collision:boundary -> stopped x3; collision:obstacle -> stopped x3; collision:target -> stopped x3; goal -> collision:boundary x2; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x14
- v21_r3_e1_estop_bt vs v16: collision:boundary -> stopped x2; collision:obstacle -> stopped x3; collision:target -> stopped x3; goal -> collision:boundary x2; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x14
- v21_e1s15_estop vs v16: collision:boundary -> collision:by_target x1; collision:boundary -> stopped x2; collision:obstacle -> collision:by_target x1; collision:obstacle -> stopped x3; collision:target -> stopped x3; goal -> collision:boundary x2; goal -> collision:by_target x2; goal -> collision:obstacle x1; goal -> stopped x22
- v21_e1s25_estop vs v16: collision:boundary -> stopped x3; collision:obstacle -> stopped x7; collision:target -> stopped x3; goal -> collision:boundary x2; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x23
- v21_e1s35_estop vs v16: collision:boundary -> stopped x3; collision:obstacle -> stopped x8; collision:target -> stopped x3; goal -> collision:boundary x2; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x24
- off vs v16_estop_bt: collision:boundary -> collision:obstacle x2; collision:boundary -> collision:target x1; collision:boundary -> goal x3; collision:by_target -> collision:boundary x1; collision:obstacle -> collision:boundary x1; collision:obstacle -> collision:target x1; collision:obstacle -> goal x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x14; goal -> collision:target x13; stopped -> collision:boundary x1; stopped -> collision:obstacle x1; stopped -> collision:target x2; stopped -> goal x2
- v16 vs v16_estop_bt: collision:boundary -> goal x2; collision:by_target -> collision:boundary x1; stopped -> collision:boundary x2; stopped -> collision:target x2; stopped -> goal x2
- v21_estop_off vs v16_estop_bt: 
- v21_e1_estop_bt vs v16_estop_bt: collision:boundary -> stopped x1; collision:by_target -> stopped x1; collision:obstacle -> stopped x3; collision:target -> stopped x1; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x12; stopped -> collision:boundary x1
- v21_r3_e1_estop_bt vs v16_estop_bt: collision:boundary -> stopped x1; collision:by_target -> stopped x1; collision:obstacle -> stopped x3; collision:target -> stopped x1; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x12; stopped -> collision:boundary x2
- v21_e1s15_estop vs v16_estop_bt: collision:boundary -> stopped x1; collision:obstacle -> collision:by_target x1; collision:obstacle -> stopped x3; collision:target -> stopped x1; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x21; stopped -> collision:boundary x1; stopped -> collision:by_target x1
- v21_e1s25_estop vs v16_estop_bt: collision:boundary -> stopped x1; collision:by_target -> stopped x1; collision:obstacle -> stopped x7; collision:target -> stopped x1; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x21; stopped -> collision:boundary x1
- v21_e1s35_estop vs v16_estop_bt: collision:boundary -> stopped x1; collision:by_target -> stopped x1; collision:obstacle -> stopped x8; collision:target -> stopped x1; goal -> collision:by_target x1; goal -> collision:obstacle x1; goal -> stopped x22; stopped -> collision:boundary x1

