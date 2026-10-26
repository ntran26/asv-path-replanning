# V21 paired report: ts4_final_off_v16_bt

Tags: ts4_final_c01, ts4_final_c02, ts4_final_c03, ts4_final_c04, ts4_final_c05, ts4_final_c06, ts4_final_c07, ts4_final_c08, ts4_final_c09, ts4_final_c10. 1000 cases with every mode (off, v16, v16_estop_bt); paired by case, episode seed and scenario digest.

## Outcomes

| Mode | episodes | goal | obstacle | boundary | target | by_target | stopped | timeout | contacts | own_vessel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 1000 | 865 | 70 | 12 | 53 | 0 | 0 | 0 | 135 | 135 |
| v16 | 1000 | 903 | 47 | 24 | 26 | 0 | 0 | 0 | 97 | 97 |
| v16_estop_bt | 1000 | 892 | 47 | 19 | 12 | 8 | 21 | 1 | 86 | 78 |

## Paired changes

| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |
|---|---:|---:|---:|---:|---:|
| v16 vs off | 1000 | 865 | 903 | 49 | 11 |
| v16_estop_bt vs off | 1000 | 865 | 892 | 46 | 19 |
| off vs v16 | 1000 | 903 | 865 | 11 | 49 |
| v16_estop_bt vs v16 | 1000 | 903 | 892 | 0 | 11 |
| off vs v16_estop_bt | 1000 | 892 | 865 | 19 | 46 |
| v16 vs v16_estop_bt | 1000 | 892 | 903 | 11 | 0 |

Outcome transitions (reference -> candidate):

- v16 vs off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:obstacle -> collision:boundary x9; collision:obstacle -> goal x23; collision:target -> collision:boundary x1; collision:target -> collision:obstacle x7; collision:target -> goal x25; goal -> collision:boundary x6; goal -> collision:obstacle x2; goal -> collision:target x3
- v16_estop_bt vs off: collision:boundary -> collision:target x3; collision:boundary -> goal x1; collision:boundary -> stopped x4; collision:obstacle -> collision:boundary x8; collision:obstacle -> collision:by_target x1; collision:obstacle -> goal x22; collision:obstacle -> stopped x1; collision:target -> collision:boundary x2; collision:target -> collision:by_target x5; collision:target -> collision:obstacle x8; collision:target -> goal x23; collision:target -> stopped x7; goal -> collision:boundary x5; goal -> collision:by_target x2; goal -> collision:obstacle x1; goal -> collision:target x1; goal -> stopped x9; goal -> timeout x1
- off vs v16: collision:boundary -> collision:obstacle x9; collision:boundary -> collision:target x1; collision:boundary -> goal x6; collision:obstacle -> collision:target x7; collision:obstacle -> goal x2; collision:target -> collision:boundary x3; collision:target -> goal x3; goal -> collision:boundary x1; goal -> collision:obstacle x23; goal -> collision:target x25
- v16_estop_bt vs v16: collision:boundary -> collision:by_target x2; collision:boundary -> stopped x6; collision:boundary -> timeout x1; collision:obstacle -> stopped x1; collision:target -> collision:boundary x1; collision:target -> collision:by_target x5; collision:target -> collision:obstacle x1; collision:target -> stopped x7; goal -> collision:boundary x3; goal -> collision:by_target x1; goal -> stopped x7
- off vs v16_estop_bt: collision:boundary -> collision:obstacle x8; collision:boundary -> collision:target x2; collision:boundary -> goal x5; collision:by_target -> collision:obstacle x1; collision:by_target -> collision:target x5; collision:by_target -> goal x2; collision:obstacle -> collision:target x8; collision:obstacle -> goal x1; collision:target -> collision:boundary x3; collision:target -> goal x1; goal -> collision:boundary x1; goal -> collision:obstacle x22; goal -> collision:target x23; stopped -> collision:boundary x4; stopped -> collision:obstacle x1; stopped -> collision:target x7; stopped -> goal x9; timeout -> goal x1
- v16 vs v16_estop_bt: collision:boundary -> collision:target x1; collision:boundary -> goal x3; collision:by_target -> collision:boundary x2; collision:by_target -> collision:target x5; collision:by_target -> goal x1; collision:obstacle -> collision:target x1; stopped -> collision:boundary x6; stopped -> collision:obstacle x1; stopped -> collision:target x7; stopped -> goal x7; timeout -> collision:boundary x1

