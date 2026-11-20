# Trial analysis (Paper 3, Section 5)

Scripts that turn the trial sessions' logs into the Section 5 tables F1-F5 and the verdicts of
the field claims C-15 to C-20 (`static_dynamic_obstacles/planning/PAPER3_CLAIMS_AND_TABLES.md`),
as `FIELD_TRIALS_PLAN.md` sections 6 and 7 define them. They read files only and open no
socket: the bridge (`udp_live_p3.py`) writes the logs, and the analysis runs afterwards.
Committed with the trial pre-registration, before session T1; nothing under
`static_dynamic_obstacles/src/` or `configs/` changes.

| File | Role |
| --- | --- |
| `trial_io.py` | loaders: bridge run logs, the run sheet, ground-truth tracks; formats below |
| `trial_metrics.py` | per-run metrics (section 6 axes), the C-16 rule checks, perception (F5), the shadow layer (C-20) |
| `twin.py` | closed-loop twin of each run in the training model and the calibrated model (section 7, level 2) |
| `open_loop.py` | logged commands through both models, error against horizon (section 7, level 1; C-18) |
| `coverage.py` | identified dynamics against the training randomisation (F4, C-19) |
| `make_tables.py` | F1, F2, F3, F5 and the verdicts of C-15 (water part), C-16, C-17, C-18 (closed loop), C-20 |
| `synthetic.py` | a synthetic session from simulated test set v4 episodes through the bridge core, for testing |
| `selftest.py` | the analysis against the simulator's own reference on the synthetic session |

## Inputs

```
<session>/run_sheet.csv
<session>/runs/<run_id>/decisions.jsonl, shadow.jsonl     (bridge output, unchanged)
<session>/ground_truth/<run_id>_target.csv                (decision D-2)
<session>/ground_truth/<run_id>_own.csv                   (optional)
```

**Run sheet**, one row per run, invalid runs kept: `run_id, session, scenario_id, family`
(A, B, C)`, encounter` (none, head_on, crossing_starboard, crossing_port, overtaking,
being_overtaken)`, layout` (L1, L2, L3, open)`, repeat, valid` (1/0)`, outcome` (goal,
contact_panel, contact_boundary, contact_target, operator_stop, timeout)`, notes`. The outcome
is the operator's record; the geometric checks (goal test, hull contact) are reported beside it.

**Ground-truth track**: `t_run` (s from the run's first decision) or `t_unix` (UTC s, matched
through the log's clock record), `x`, `y` (m, registered map frame), optional `heading_deg`
(compass; else the direction of motion) and `speed` (m/s; else from position differences).
The target ground-truth method is fixed after the calibration sessions (D-2); any method that
writes this file fits. Without an own-ship file the own track is the registered pose the bridge
logged (`--origin map`).

## Order of use

```bash
python twin.py <session> --calibrated <session-1 fit>.json
python make_tables.py <session> [<session2>] --calibrated <session-1 fit>.json
python open_loop.py --bluefin-data <calibration logs> --sessions <session-2 folder> --calibrated <fit>.json
python coverage.py --identified <fit>.json --intervals <bootstrap or day-to-day sets>.json
```

## Definitions fixed here (to confirm before pre-registration)

Each choice below decides a registered verdict or a reported number; the scripts implement it
as stated, and a change is a change of the pre-registration.

| Item | Implemented |
| --- | --- |
| C-16 counting | per core encounter type, family B, valid runs: met when at least 2 of the runs that reach the closest approach without an operator stop pass the Section 3.4.3 conditions (`colregs_compliance.episode` on a 0.5 s grid); claim met when all four types are |
| Closest approach reached | the minimum of the true range over the whole log lies before the end of the log and before the first command not from the policy |
| C-17 bound | one-sided 95 %: field goals >= `binom.ppf(0.05, n, p)`, n the valid family C runs, p the pooled simulated rate (goals over episodes) of the cells run; the run-weighted rate is reported beside it |
| C-17 and F1 reference | `results/test_set/v4/sacs0_bl3_metrics/episodes.csv` (the deployed best checkpoint, 0.865), layer off, both target variants FIX and VAR as in the plan's L-cell table; `--variants FIX` restricts to fixed-speed targets, closer to the field's |
| C-18 open loop | calibrated below training in heading and position RMS at every horizon (1, 2, 3, 5, 10 s and free run) on the session-2 holdout |
| C-18 closed loop | descriptive: field against twin outcome, a twin "succeeds" when at least half its seeds reach the goal; each metric's field value as a percentile of the twins' |
| Twin plant | training twin: the identified July parameters (`ship.IDENTIFIED`); calibrated twin: the session-1 fit; plant randomisation off in both; 10 noise seeds from 910,000, shared by both |
| C-19 envelope | inside when the identified value lies in the central 95 % of 2,000 draws of the training randomisation; groups turn rate and turning radius (steady turns at rpm 12, helm 25-100 %, and at the cruise setting rpm 6, helm 50 and 100 %, both signs, as the S1-D circles), thrust curve, actuator lag (delay + lag) |
| C-15 water part | every logged observation finite and inside the environment's observation space; LiDAR line to sent command p99 below 0.5 s |
| Target track association | the tracker's dynamic track nearest the target's true position within the simulator's pairing radius (`TARGET_MATCH_RADIUS`, one hull length, 1.73 m); any other dynamic track is a phantom |
| Classification agreement (F5) | the simulator's own measure: the perceived class of the target track's engaged context against `cls_true` (true geometry, own course taken as the path's, A19), over engaged decisions whose true geometry has a class; engaged decisions with no true class (a context held after the passing) reported apart |
| Phantom share (F5) | decisions with a phantom, over family A runs and over the decisions before the target enters the scan (inside the boundary and within the 16 m LiDAR range) |
| Field speed | ground speed of the own track by centred differences over 1 s |
| Scored states | as the simulator: own ship as the environment's metric hull (length and beam plus the hull margin), states from the first after the start to the goal capture, or to the last decision under the policy |
| Rudder reversal | sign change of the issued rudder between consecutive decisions, both at least 5 % |

## Self-test (2026-11-11)

`synthetic.py` ran 12 test set v4 episodes of the deployed policy through the bridge core (L1-L3
cells, the four open-basin encounter types with both crossing sides, a no-target layout) and
wrote them as a session with the simulator's true tracks as ground truth; `selftest.py`
compares the analysis with the simulator. Outputs under `p3_bench/trial_analysis_selftest/`
(the logs are not committed; they are reproducible).

| Check | Result |
| --- | --- |
| Rule conditions (`encounter_behaviour` against `colregs_compliance.episode` on the simulator's trajectory) | 11 of 11 agree on compliance, first manoeuvre, passing side and crossing ahead; the CPA differs only in the two collision runs, whose last state comes after the last logged decision |
| Twin reproduction (episode plant and seed, start and target from the ground truth) | 12 of 12 outcomes equal; own track within 1e-6 m in 11 runs, 0.18 m at the final (collision) state of the twelfth, where the scripted target interpolates a step cut short by the collision |
| Clearances to panels, boundary and target against `run_episode` | equal in every goal run; the remaining differences in path efficiency (about -0.02), cross-track RMS (about +0.01 m) and speed come from the terminal state, which a synthetic log cannot carry and a field log does (the bridge logs past the goal) |
| Classification agreement against the simulator's own perceived-against-true measure | equal on the four episodes checked (15/15, 14/14, 7/7, 16/16) |
| Open loop: logged commands through the episode's own plant against the July model | 3 s heading RMS 0.2-0.7 deg with the episode's plant (the floor between the batch integrator and the simulator's substepped plant), against up to 3.5 deg with the July model; alike in the runs with little turning. Sign and units of the logged commands confirmed |
| End to end (`twin.py`, 3 seeds, a stand-in calibrated set; `make_tables.py`) | every table and verdict produced; C-15 water part: 574 observations inside the space, command path p99 0.07 s |

One simulator finding from the self-test: in `P2-L1-CRP-FIX-05` the target, moving at 0.42 m/s,
is never classified as moving by the tracker, and the policy collides with it.
