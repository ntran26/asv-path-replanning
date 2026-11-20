# Section 5 tables (generated)

### f1_runs_by_scenario

| scenario | family | encounter | layout | valid / runs | goals | contact panel | contact boundary | contact target | operator stops | would-be layer stops | min target range m (median, range) | simulated twin cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-L1 | A | none | L1 | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 |  |  |
| B-being_overtaken | B | being_overtaken | open | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 | 2.17 (2.17-2.17) |  |
| B-crossing_port | B | crossing_port | open | 2 / 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.93 (1.47-2.39) |  |
| B-head_on | B | head_on | open | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 | 3.29 (3.29-3.29) |  |
| B-overtaking | B | overtaking | open | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 | 2.14 (2.14-2.14) |  |
| C-L1-CRP | C | crossing_port | L1 | 1 / 1 | 0 | 0 | 0 | 1 | 0 | 1 | 1.53 (1.53-1.53) | L1-CRP: 2 / 11 |
| C-L1-CRS | C | crossing_starboard | L1 | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 | 2.81 (2.81-2.81) | L1-CRS: 7 / 12 |
| C-L1-HO | C | head_on | L1 | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1.46 (1.46-1.46) | L1-HO: 10 / 10 |
| C-L1-OT | C | overtaking | L1 | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 | 4.34 (4.34-4.34) | L1-OT: 6 / 10 |
| C-L2-HO | C | head_on | L2 | 1 / 1 | 0 | 1 | 0 | 0 | 0 | 0 | 3.07 (3.07-3.07) | L2-HO: 0 / 5 |
| C-L3-HO | C | head_on | L3 | 1 / 1 | 1 | 0 | 0 | 0 | 0 | 0 | 4.07 (4.07-4.07) | L3-HO: 4 / 5 |

### f2_family_b

| run_id | encounter | valid | outcome | reached_cpa_without_stop | first_manoeuvre | max_lateral_m | max_course_alteration_deg | first_alteration_after_engagement_s | target_side_at_cpa | crossed_ahead | cpa_m | speed_reduction_mps | rule | rule_compliant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-BAS-HO-CV-001 | head_on | True | goal | True | port | 2.91 | 30.5 | 4.5 | starboard | 0 | 3.29 | 0.0216 | Rule 14 | 0 |
| syn-BAS-CR-CV-003 | crossing_port | True | goal | True | port | 1.74 | 49.8 | 5 | starboard | 0 | 1.47 | 0.242 | crossing from port | 1 |
| syn-BAS-CR-CV-005 | crossing_port | True | goal | True | starboard | 1.19 | 20.7 | 3 | port | 1 | 2.39 | 0 | crossing from port | 0 |
| syn-BAS-OT-CV-003 | overtaking | True | goal | True | starboard | 3.06 | 36.7 | 2.5 | port | 0 | 2.14 | 0.00751 | Rule 13 | 0 |
| syn-BAS-BO-CV-001 | being_overtaken | True | goal | True | none | 0.262 | 15.1 |  | port | 0 | 2.17 | 0.241 | Rule 17 | 0 |

### f3_outcome_agreement

| model | field_success | twin_success | runs |
|---|---|---|---|
| calibrated | False | False | 2 |
| calibrated | True | False | 1 |
| calibrated | True | True | 9 |
| training | False | False | 2 |
| training | True | False | 1 |
| training | True | True | 9 |

### f5_perception

| run_id | family | encounter | layout | valid | target_entered_scan_s | track_confirmation_delay_s | target_tracked_share | track_position_error_median_m | track_position_error_p90_m | track_heading_error_median_deg | classification_agreement | classification_decisions | engaged_truth_none_decisions | phantom_eligible_decisions | phantom_share | decisions_with_moving_track |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-P2-L1-HO-FIX-03 | C | head_on | L1 | True | 3 | 3 | 0.889 | 0.738 | 0.877 | 4.01 | 1 | 15 | 9 | 6 | 0 | 0.803 |
| syn-P2-L1-CRS-FIX-05 | C | crossing_starboard | L1 | True | 0 | 1.5 | 0.776 | 0.366 | 0.496 | 2.07 | 1 | 13 | 0 | 0 |  | 0.655 |
| syn-P2-L1-CRP-FIX-05 | C | crossing_port | L1 | True | 0 |  | 0 |  |  |  |  | 0 | 0 | 0 |  | 0 |
| syn-P2-L1-OT-FIX-01 | C | overtaking | L1 | True | 0 | 2 | 0.739 | 0.294 | 0.802 | 4.77 | 1 | 7 | 0 | 0 |  | 0.739 |
| syn-P2-L2-HO-VAR-04 | C | head_on | L2 | True | 0 | 2.5 | 0.783 | 0.707 | 0.923 | 4.23 | 1 | 13 | 5 | 0 |  | 0.783 |
| syn-P2-L3-HO-FIX-05 | C | head_on | L3 | True | 0 | 1.5 | 0.75 | 0.54 | 0.749 | 2.23 | 1 | 7 | 3 | 0 |  | 0.66 |
| syn-P2-L1-NT-01 | A | none | L1 | True |  |  |  |  |  |  |  | 0 | 0 | 63 | 0 | 0 |
| syn-BAS-HO-CV-001 | B | head_on | open | True | 0 | 1.5 | 0.943 | 0.656 | 0.887 | 3.42 | 1 | 14 | 13 | 0 |  | 0.943 |
| syn-BAS-CR-CV-003 | B | crossing_port | open | True | 0 | 1.5 | 0.767 | 0.469 | 0.833 | 3.99 | 1 | 16 | 0 | 0 |  | 0.603 |
| syn-BAS-CR-CV-005 | B | crossing_port | open | True | 0 | 1.5 | 0.775 | 0.38 | 0.586 | 4.69 | 1 | 10 | 0 | 0 |  | 0.66 |
| syn-BAS-OT-CV-003 | B | overtaking | open | True | 0 | 2 | 0.917 | 0.644 | 0.795 | 6.09 | 1 | 20 | 4 | 0 |  | 0.917 |
| syn-BAS-BO-CV-001 | B | being_overtaken | open | True | 0 | 1 | 0.949 | 0.318 | 0.39 | 0.801 | 1 | 15 | 6 | 0 |  | 0.949 |

### Verdicts

```json
{
  "C-16": {
    "per_type": {
      "head_on": {
        "runs": 1,
        "eligible": 1,
        "pass": 0,
        "met": false
      },
      "crossing_starboard": {
        "runs": 0,
        "eligible": 0,
        "pass": 0,
        "met": false
      },
      "crossing_port": {
        "runs": 2,
        "eligible": 2,
        "pass": 1,
        "met": false
      },
      "overtaking": {
        "runs": 1,
        "eligible": 1,
        "pass": 0,
        "met": false
      }
    },
    "met": false,
    "rule": "at least 2 eligible runs pass in each of the four core types"
  },
  "C-17": {
    "field_goals": 4,
    "field_runs": 6,
    "pooled_sim_rate": 0.5471698113207547,
    "lower_bound": 1,
    "met": true,
    "variants": [
      "FIX",
      "VAR"
    ],
    "results": "C:\\Users\\hntran\\OneDrive - University of Tasmania\\Documents\\PhD\\asv-path-replanning\\asv-lidar\\static_dynamic_obstacles\\results\\test_set\\v4\\sacs0_bl3_metrics\\episodes.csv",
    "run_weighted_sim_rate": 0.5275252525252525,
    "per_cell": [
      {
        "cell": "L1-CRP",
        "field_goals": 0,
        "field_runs": 1,
        "sim_goals": 2,
        "sim_episodes": 11
      },
      {
        "cell": "L1-CRS",
        "field_goals": 1,
        "field_runs": 1,
        "sim_goals": 7,
        "sim_episodes": 12
      },
      {
        "cell": "L1-HO",
        "field_goals": 1,
        "field_runs": 1,
        "sim_goals": 10,
        "sim_episodes": 10
      },
      {
        "cell": "L1-OT",
        "field_goals": 1,
        "field_runs": 1,
        "sim_goals": 6,
        "sim_episodes": 10
      },
      {
        "cell": "L2-HO",
        "field_goals": 0,
        "field_runs": 1,
        "sim_goals": 0,
        "sim_episodes": 5
      },
      {
        "cell": "L3-HO",
        "field_goals": 1,
        "field_runs": 1,
        "sim_goals": 4,
        "sim_episodes": 5
      }
    ],
    "rule": "field goals >= binom.ppf(0.05, runs, pooled simulated rate)"
  },
  "C-20": {
    "runs": 12,
    "would_change_rate_median": 0.0,
    "runs_with_stop_commitment": 1,
    "descriptive": true
  },
  "C-15 water": {
    "decisions": 574,
    "observations_outside_space": 0,
    "first_outside": [],
    "command_path_p99_s": 0.07400214701017829,
    "met": true
  },
  "C-18 closed loop": {
    "outcome_agreement": [
      {
        "model": "calibrated",
        "field_success": false,
        "twin_success": false,
        "runs": 2
      },
      {
        "model": "calibrated",
        "field_success": true,
        "twin_success": false,
        "runs": 1
      },
      {
        "model": "calibrated",
        "field_success": true,
        "twin_success": true,
        "runs": 9
      },
      {
        "model": "training",
        "field_success": false,
        "twin_success": false,
        "runs": 2
      },
      {
        "model": "training",
        "field_success": true,
        "twin_success": false,
        "runs": 1
      },
      {
        "model": "training",
        "field_success": true,
        "twin_success": true,
        "runs": 9
      }
    ],
    "descriptive": true
  }
}
```
