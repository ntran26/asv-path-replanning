# Calibration fit: reading the new logs, and the P-6 pre-registration

Date: 2026-11-24. Status: specification. The reader and fit script it describes are not yet
written; `feb2026/` holds the pieces that exist.

P-6 of `planning/FIELD_CALIBRATION_PLAN.md` requires the fit script to read the new log format and
the analysis to be pre-registered before session S1b: empty tables, metric definitions, thresholds
and a committed script that S1-F runs unchanged. This note states what the identification pipeline
needs for that, from the log formats of `field_deployment/udp_raw_logger.py` and
`field_deployment/udp_manoeuvre.py` and from what the February analysis showed.

## 1. What the pipeline reads today

`build_dataset.py` reads only the July bridge logs:
- `#CONFIG` and `#ACTION` lines, the command from the literal `$CMD,<rudder>,<thrust>` inside
  `#ACTION` (`limited_rudder` as fallback), rpm from the `rpm=` field;
- pose lines (x = second field, y and heading negated, as `log_parser.py`);
- segments cut at clock gaps over 5 s, at least 20 samples, sessions named in the code;
- 2 Hz throughout: `simulate.make_batch` takes yaw rate and speed from a 5-sample quadratic and
  speed from the path length, and `simulate_pop` starts every run from rest at the origin.

None of that fits the new logs: they have no `#ACTION` line, may run at 10 Hz, start manoeuvres
under way, and carry IMU lines of a format first seen on 2026-11-26.

## 2. The new log formats

**`udp_raw_logger.py`** (D0, and S1a under radio control). The log holds every datagram unchanged,
one per line, each telemetry line starting with the vessel's stamp `[HH:MM:SS.ffffff]`:
pose, HDG, LiDAR, receiver (`S1:... S2:... RC`, or `WD` without a transmitter) and, from D0, IMU
lines. Beside it: `<out>.arrivals.csv` (`n,t_unix_s,t_since_start_s,bytes`, one row per datagram)
and `<out>.meta.json` (options, times, counts of line kinds). There is no command line: under radio
control the only record of the commands is the receiver line, at about 1 Hz in February.

**`udp_manoeuvre.py`** (S1a when its first check passes, S1b, S2-B). The same datagram lines, and
between them, on the program's monotonic clock:

    #CMD,t_mono=<s>,t_unix=<s>,phase=<name>,reason=<text>,rudder_pct=<sim %>,rpm=<units>,s2=<%>,cmd=$CMD,<vessel rudder>,<S2>,sent=<0|1>
    #MAN,{"kind": ..., "t_manoeuvre_s": ..., "t_mono_s": ..., ...}

`rudder_pct` is in the simulator convention (positive turns to starboard); the transmitted rudder
is `--rudder-sign` (-1) times it. `s2` = rpm / 24 x 100, negative only with `--allow-astern`.
`<out>.commands.csv` repeats the `#CMD` fields; `<out>.arrivals.csv` uses the same monotonic origin
as `t_mono`, so commands and telemetry arrivals share one host clock; `<out>.meta.json` holds the
manoeuvre, its events (switch times, overshoots, the stop) and how it ended.

## 3. What the reader needs

| Id | Requirement | Why |
| --- | --- | --- |
| R1 | Classify every line; parse the known kinds; count and keep unknown kinds with their stamp; never stop on an unknown line | IMU lines are new; lab tools must not fail on unknown data |
| R2 | Pose as `log_parser.py`: x = field 2, y = -field 1, heading = -field 3, unwrapped; any rate | 10 Hz in February, 2 Hz in July |
| R3 | One time base. Telemetry carries vessel stamps; commands carry host monotonic time. Match the n-th datagram line to row n of `arrivals.csv`, fit vessel stamp = a + b t_since_start on the lower envelope of (arrival - stamp), report the spread, handle a midnight wrap | commands and pose on one clock (P-5); the link latency to the vessel stays inside the fitted transport delay |
| R4 | Commands from `#CMD` lines with `sent=1`, held between lines (keepalives included); the transmitted `cmd=` string is authoritative, as in July; rudder to the model as rud_sim = -(transmitted rudder); rpm = S2 x 24 / 100, negative for astern | the July sign chain (`REPORT.md` 1.2) |
| R5 | Receiver lines: during a `udp_manoeuvre.py` run, any S1 or S2 away from neutral marks a manual override and the window is excluded; in a radio-control log, S2 and S1 map to commands only through the mapping established by the bench item of `feb2026/REPORT_FEB2026.md` section 9 (vessel program) or the A/B pair, else they are stated as an assumption. The receiver's S2 is the rudder; the `$CMD` S2 is propulsion | February: S1 throttle (neutral 1511), S2 rudder (1000 / 1467 / 1967) |
| R6 | Segments by run identifier (one log per run, run sheet) and by the `phase` of `#CMD` and the events of `#MAN`, not by clock gaps | manoeuvres start under way and are labelled |
| R7 | At 10 Hz: derivatives for window initialisation from a smoothing fit over about 1 s (Savitzky-Golay, 11 samples, quadratic, as `feb_logs.kinematics`); speed from velocity, not path length; or decimation to 2 Hz for the July objective | at 0.1 m resolution a 0.5 s quadratic is noisy and a 10 Hz path length is biased upwards |
| R8 | Every run and window starts from its measured state (position, heading, surge, yaw rate), as `feb_analysis.replay_pop` | `simulate_pop` assumes rest at the origin |
| R9 | The model compared at the pose reference point surveyed at S1-A (O-9): x_p = x + l sin(psi), y_p = y + l cos(psi), with l = +0.8625 m if the pose is the LiDAR's; drift computed at the same point | the February drift in turns (0-0.5 deg) matches v3 at the LiDAR (0.9 deg), not at midship (10.2 deg) |
| R10 | A constant heading offset between pose and hull axis, fitted or surveyed | +3.7 deg in February, +2.2 deg in July, in straight running |
| R11 | Astern in the fitter: the braking split of `ship.py` with reverse efficiency and onset delay as parameters; the model clips surge at zero, so astern motion after a stop is excluded unless the structure is extended | S1-C2, T-9 to T-13 |
| R12 | IMU, once D0 shows the format: gyro z as yaw rate (window initialisation, a possible fitting target), accelerometer x for surge; the clock offset from the hand push (P-5); the sign of the gyro against the pose heading change | O-2; the zero-speed yaw damping from the push decay |
| R13 | Exclusion rules fixed in advance: telemetry gaps over 1.5 s; operator override; wall contact (distance to the surveyed walls or minimum LiDAR range below a set value); hauled motion (refit v4's rule: three steps of at least 0.15 m more than 90 deg off the heading) | the July retrievals moved the yaw parameters by more than their intervals |
| R14 | Join with the run sheet: validity, settings, battery voltage; invalid runs kept and reported | `FIELD_SESSIONS_PLAN.md` section 2.3 |

## 4. What the P-6 pre-registration needs

1. **The committed reader and fit script**, with the commit recorded in the session record, run
   unchanged at S1-F. Tested before S1b on: the July logs (reproducing `REPORT.md` section 6 and
   the refit v4 and v4b figures of `REVIEW.md` section 8, as `feb2026/refit_report.txt` does), the
   February logs (reproducing `feb2026/feb_analysis.txt`), the D0 logs (format, IMU, clock) and a
   `udp_manoeuvre.py --selftest` log (the command path).
2. **The split**: fit on S1 only, validate on S2 only, never refit on S2 data (calibration plan
   section 6).
3. **The structures, in a fixed order, and the rule that picks one.**
   - M0: the v3 structure (`dynamics.py`).
   - M1: M0 with N_rr = Y_vv L^3 / 32 (cross-flow yaw damping; `feb2026/REPORT_FEB2026.md` section 6).
   - M2: M1 with the rudder sway force acting aft (the MMG sign; `MODEL_EXPLAINED.md` section 4).
   - M3: M2 compared at the pose reference point with the heading offset (R9, R10).
   - M4: M3 with separate rudder gains per side, only if S1-D shows the asymmetry under `$CMD`.
   - The rule, in the form of refit v4's: a structure replaces the simpler one only if it lowers the
     S2 heading error at 3 s and 10 s by at least 5 % and raises no other S2 metric by more than 5 %.
4. **Parameters**: free, fixed and tied, with bounds (N_uv on [-250, 250] as in refit v4; N_rr tied
   or free; actuator terms free only with S1-B data meeting T-1 and T-2).
5. **Objective**: blocks (S1-B steps, S1-C surge, S1-D turns, S1-E zig-zags, S1-C2 stops), their
   weights, the horizons, the distribution terms kept or dropped, and no steady-turn priors (S1-D
   replaces them, T-5).
6. **Thrust model**: T = T12 (n / 12)^k with k fitted (T-3), or one value per RPM point; any
   propulsion delay and lag; the reverse efficiency and onset (T-9, T-10).
7. **Metric definitions** for every threshold T-1 to T-13: e.g. steady turn rate as the mean over the
   part of the turn after the first 90 deg, radius as U / r and as a circle fit, head reach from the
   `#CMD` stop time to surge below 0.05 m/s along the initial heading, heading change and lateral
   drift to rest (yaw rate below 0.5 deg/s, sway below 0.02 m/s).
8. **Uncertainty**: a run-level bootstrap within S1 (number of resamples fixed), and the day-to-day
   comparison of S1 with S2-A as the honest width.
9. **The coverage report**: each identified quantity against the training randomisation
   (`explain_numbers.txt`: full-helm radius at RPM 6 over 200 draws 2.29-5.40 m, steady speed at
   RPM 6 0.532-0.582 m/s, and the parameter ranges).
10. **Acceptance** of the analysis model: `acceptance.py` A1-A4, A6 and A7 (A5 only if it is used for
    randomisation).
11. **The empty tables** of the outputs in calibration plan section 6.
12. **The role of the February data**: joined to the calibration only through the A/B pair (R5);
    otherwise kept as the exploratory record of `feb2026/`.

## 5. What exists to start from

| Piece | Where | Gives |
| --- | --- | --- |
| Receiver logs at 10 Hz | `feb2026/feb_logs.py` | pose, receiver and HDG parsing; Savitzky-Golay kinematics; receiver-to-command maps; `Run` objects for the July pipeline |
| Replay from a measured state | `feb2026/feb_analysis.py` `replay_pop`, `replay_errors` | windowed and free-run errors for runs that start under way |
| Manoeuvre measures | `feb2026/feb_analysis.py` | settled turns, stops (reverse force, head reach, heading change), yaw decay, coast-downs |
| M1 structure in a fit | `feb2026/refit_feb.py` | the N_rr tie, equal weighting of two data sets, structural priors without the turn bands |
| Trim and adoption rules | `refit_v4/refit_v4.py` | the hauled-motion rule and the 5 % adoption tolerance |
| July `#ACTION` reader | `build_dataset.py` | the July logs, for the regression check of item 1 |
