# V21 paired report: ts4_final

Tags: ts4_final_c01, ts4_final_c02, ts4_final_c03, ts4_final_c04, ts4_final_c05, ts4_final_c06, ts4_final_c07, ts4_final_c08, ts4_final_c09, ts4_final_c10, ts4_final_e1_c01, ts4_final_e1_c02, ts4_final_e1_c03, ts4_final_e1_c04, ts4_final_e1_c05, ts4_final_e1_c06, ts4_final_e1_c07, ts4_final_e1_c08, ts4_final_e1_c09, ts4_final_e1_c10, ts4_final_e1s35_c01, ts4_final_e1s35_c02, ts4_final_e1s35_c03, ts4_final_e1s35_c04, ts4_final_e1s35_c05, ts4_final_e1s35_c06, ts4_final_e1s35_c07, ts4_final_e1s35_c08, ts4_final_e1s35_c09, ts4_final_e1s35_c10. 1000 cases with every mode (off, v16, v16_estop_bt, v21_e1_estop_bt, v21_e1s35_estop); paired by case, episode seed and scenario digest.

## Outcomes

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts | own_vessel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 1000 | 865 | 70 | 12 | 53 | 0 | 0 | 0 | 135 | 135 |
| v16 | 1000 | 903 | 47 | 24 | 26 | 0 | 0 | 0 | 97 | 97 |
| v16_estop_bt | 1000 | 892 | 47 | 19 | 12 | 8 | 21 | 1 | 86 | 78 |
| v21_e1_estop_bt | 1000 | 868 | 45 | 17 | 8 | 9 | 52 | 1 | 79 | 70 |
| v21_e1s35_estop | 1000 | 820 | 43 | 17 | 7 | 12 | 100 | 1 | 79 | 67 |

## Paired changes

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
|---|---:|---:|---:|---:|---:|
| v16 vs off | 1000 | 865 | 903 | 49 | 11 |
| v16_estop_bt vs off | 1000 | 865 | 892 | 46 | 19 |
| v21_e1_estop_bt vs off | 1000 | 865 | 868 | 39 | 36 |
| v21_e1s35_estop vs off | 1000 | 865 | 820 | 28 | 73 |
| off vs v16 | 1000 | 903 | 865 | 11 | 49 |
| v16_estop_bt vs v16 | 1000 | 903 | 892 | 0 | 11 |
| v21_e1_estop_bt vs v16 | 1000 | 903 | 868 | 0 | 35 |
| v21_e1s35_estop vs v16 | 1000 | 903 | 820 | 0 | 83 |
| off vs v16_estop_bt | 1000 | 892 | 865 | 19 | 46 |
| v16 vs v16_estop_bt | 1000 | 892 | 903 | 11 | 0 |
| v21_e1_estop_bt vs v16_estop_bt | 1000 | 892 | 868 | 0 | 24 |
| v21_e1s35_estop vs v16_estop_bt | 1000 | 892 | 820 | 0 | 72 |

Outcome transitions (reference -> candidate):

- v16 vs off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x9; collision:obstacle -> goal x23; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x7; collision:target -> goal x25; goal -> collision:boundary x6; goal -> collision:obstacle x2; goal -> collision:target x3
- v16_estop_bt vs off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:boundary -> stopped x4; collision:obstacle -> collision:boundary x8; collision:obstacle -> collision:by_target x1; collision:obstacle -> goal x22; collision:obstacle -> stopped x1; collision:target -> collision:boundary x2; collision:target -> collision:by_target x5; collision:target -> collision:obstacle x8; collision:target -> goal x23; collision:target -> stopped x7; goal -> collision:boundary x5; goal -> collision:by_target x2; goal -> collision:obstacle x1; goal -> collision:target x1; goal -> stopped x9; goal -> timeout x1
- v21_e1_estop_bt vs off: collision:boundary -> collision:target x1; collision:boundary -> goal x1; collision:boundary -> stopped x8; collision:obstacle -> collision:boundary x6; collision:obstacle -> goal x22; collision:obstacle -> stopped x4; collision:target -> collision:boundary x2; collision:target -> collision:by_target x3; collision:target -> collision:obstacle x6; collision:target -> goal x16; collision:target -> stopped x20; goal -> collision:boundary x7; goal -> collision:by_target x6; goal -> collision:obstacle x1; goal -> collision:target x1; goal -> stopped x20; goal -> timeout x1
- v21_e1s35_estop vs off: collision:boundary -> collision:by_target x1; collision:boundary -> goal x1; collision:boundary -> stopped x8; collision:obstacle -> collision:boundary x6; collision:obstacle -> goal x18; collision:obstacle -> stopped x13; collision:target -> collision:boundary x2; collision:target -> collision:by_target x5; collision:target -> collision:obstacle x6; collision:target -> goal x9; collision:target -> stopped x25; goal -> collision:boundary x7; goal -> collision:by_target x6; goal -> collision:obstacle x4; goal -> collision:target x1; goal -> stopped x54; goal -> timeout x1
- off vs v16: collision:boundary -> collision:obstacle x9; collision:boundary -> collision:target x1; collision:boundary -> goal x6; collision:obstacle -> collision:target x7; collision:obstacle -> goal x2; collision:target -> collision:boundary x3; collision:target -> goal x3; goal -> collision:boundary x1; goal -> collision:obstacle x23; goal -> collision:target x25
- v16_estop_bt vs v16: collision:boundary -> collision:by_target x2; collision:boundary -> stopped x6; collision:boundary -> timeout x1; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:by_target x5; collision:target -> collision:obstacle x1; collision:target -> stopped x7; goal -> collision:boundary x3; goal -> collision:by_target x1; goal -> stopped x7
- v21_e1_estop_bt vs v16: collision:boundary -> stopped x13; collision:boundary -> timeout x1; collision:obstacle -> stopped x3; collision:target -> collision:boundary x1; collision:target -> collision:by_target x2; collision:target -> collision:obstacle x1; collision:target -> stopped x14; goal -> collision:boundary x6; goal -> collision:by_target x7; goal -> stopped x22
- v21_e1s35_estop vs v16: collision:boundary -> collision:obstacle x1; collision:boundary -> stopped x12; collision:boundary -> timeout x1; collision:obstacle -> stopped x10; collision:target -> collision:boundary x1; collision:target -> collision:by_target x3; collision:target -> collision:obstacle x1; collision:target -> stopped x14; goal -> collision:boundary x6; goal -> collision:by_target x9; goal -> collision:obstacle x4; goal -> stopped x64
- off vs v16_estop_bt: collision:boundary -> collision:obstacle x8; collision:boundary -> collision:target x2; collision:boundary -> goal x5; collision:by_target -> collision:obstacle x1; collision:by_target -> collision:target x5; collision:by_target -> goal x2; collision:obstacle -> collision:target x8; collision:obstacle -> goal x1; collision:target -> collision:boundary x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x22; goal -> collision:target x23; stopped -> collision:boundary x4; stopped -> collision:obstacle x1; stopped -> collision:target x7; stopped -> goal x9; timeout -> goal x1
- v16 vs v16_estop_bt: collision:boundary -> collision:target x1; collision:boundary -> goal x3; collision:by_target -> collision:boundary x2; collision:by_target -> collision:target x5; collision:by_target -> goal x1; collision:obstacle -> collision:target x1; stopped -> collision:boundary x6; stopped -> collision:obstacle x1; stopped -> collision:target x7; stopped -> goal x7; timeout -> collision:boundary x1
- v21_e1_estop_bt vs v16_estop_bt: collision:boundary -> stopped x5; collision:by_target -> stopped x6; collision:obstacle -> stopped x2; collision:target -> stopped x4; goal -> collision:boundary x3; goal -> collision:by_target x7; goal -> stopped x14
- v21_e1s35_estop vs v16_estop_bt: collision:boundary -> stopped x5; collision:by_target -> stopped x6; collision:obstacle -> stopped x9; collision:target -> collision:by_target x1; collision:target -> stopped x4; goal -> collision:boundary x3; goal -> collision:by_target x9; goal -> collision:obstacle x4; goal -> stopped x56; stopped -> collision:obstacle x1

