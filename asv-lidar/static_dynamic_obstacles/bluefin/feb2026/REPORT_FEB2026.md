# The February 2026 radio-control logs against the v3 vessel model

Date: 2026-11-24. Status: analysis; the training plant and baseline-v3 are unchanged.

The four February 2026 logs (`field_deployment/data/test_1.log` to `test_4.log`) were recorded
under radio control with the telemetry at 10 Hz. Unlike the July logs they hold sustained
full-rudder turns, crash stops under reverse throttle and coast-downs. This report checks the
first measurements made from them, tests the v3 plant against them, fits a candidate analysis
model, and lists what the calibration sessions must measure to settle each point.

| File | Content |
| --- | --- |
| `feb_logs.py` | reader for the February logs; receiver-to-command maps |
| `feb_analysis.py`, `feb_analysis.txt` | sections 1-8 below: inventory, channels, throttle, turns, stops, yaw decay, coast-downs, windowed replay |
| `july_full_command.py`, `july_full_command.txt` | July yaw rates during full-command holds |
| `refit_feb.py`, `refit_report.txt`, `out/candidate.json`, `out/fit.log` | the candidate analysis model v3f and its checks |

Conventions: "port" and "starboard" follow the sign of the yaw rate in the pose frame of
`log_parser.py` (r < 0 and r > 0); the handedness of that frame against the basin has not been
checked. Receiver values are pulse widths in microseconds.

## 1. Summary

1. **The first measurements hold.** Settled full-rudder turns: radius 2.42 m at 0.85 m/s and
   20.2 deg/s (port; circle fit 2.42 m), and 3.85 m at 1.32 m/s and 19.6 deg/s (starboard; circle
   fit 3.93 m). `test_1` spans 10.9 m sideways. S1 is the throttle and S2 the rudder.
2. **v3 turns too slowly at full helm, by a factor of 2.2 (port) and 1.4 (starboard)** in yaw
   rate at the same speed (radius 5.33-5.37 m against 2.42 and 3.85 m), if a full receiver
   deflection equals a `$CMD` of +-100 %. The turns are asymmetric: port 2.3-2.6 m, starboard
   3.5-4.1 m, each roughly independent of speed. On the February turn segments v3's windowed
   heading error is 10.9 deg at 3 s and 38.5 deg at 10 s (July holdout: 7.9 and 13.8 deg).
3. **The February data can replace the full-helm turn priors, but only as an analysis model and
   only under the assumption above.** That assumption cannot be checked from these logs: the
   vessel program's mapping from `$CMD` to pulse width decides it. A candidate (v3f) fitted on the
   July training session and the February turns predicts the July holdout better than v3 from 2 s
   to 10 s (heading 6.48 against 7.90 deg at 3 s, 11.35 against 13.82 deg at 10 s) but worse at
   1 s and in free run (28.96 against 23.53 deg), and worse than the July-only refit v4b at every
   horizon from 2 s. It turns at a symmetric 3.1 m. It is a sensitivity bracket, not the
   calibrated model.
4. **The low-speed yaw artefact is real.** From full-helm turns the real vessel stopped under full
   reverse with a heading change of 21-26 deg in 5 s and 0.6-2.9 deg/s left when surge fell below
   0.15 m/s. v3, through the same manoeuvre with the measured reverse force, keeps 9.8-10.5 deg/s at
   zero surge and turns 186-196 deg in 60 s. A quadratic yaw damping tied by strip theory to the
   identified cross-flow drag (N_rr = Y_vv L^3 / 32) removes it without harming the July holdout.
   The two straight full-reverse stops also turned, by 18-19 deg in 5 s, where v3 turns none; all
   the stops ended near an end of the basin, so wall contact is not excluded.
5. **Reverse thrust exists and is strong.** The ESC reverses under radio control: astern speeds of
   0.72-0.80 m/s, and full reverse braked at 15.3-19.1 N (mean 16.8 N) from above 1 m/s, stopping
   from 1.2-1.35 m/s in 3.5-5.3 s and 2.6-3.1 m. That is 0.26 (0.24-0.30) of v3's T(24), the weak
   end of the safety layer's bracket (0.25-0.5 at full astern, -24 rpm-units); the weak model
   through the same manoeuvres stops in 4.1-4.4 s and 2.2-2.7 m. The simulator's full astern
   (32.1 N, the strong end) is about twice the measured force. Full throttle reached at least 1.56 m/s (v3 at RPM 24: 2.23 m/s), so the
   square-law extrapolation above about RPM 14 is not supported.
6. **The pose behaves as a point near the bow.** In the settled turns the drift angle at the pose
   point is 0-0.5 deg after removing a constant +3.7 deg offset between pose heading and direction
   of travel; v3 predicts 10.2 deg at midship and 0.9 deg at the LiDAR position.

## 2. The logs

| Log | Duration | Content |
| --- | --- | --- |
| `test_1` | 198.6 s | origin bottom right; turns both ways, coast-downs, astern runs, a crash stop from a turn |
| `test_2` | 75.8 s | port turns, straight runs at S1 about 1550 and 1590, a coast-down |
| `test_3` | 73.4 s | straight runs, two crash stops from 1.3 m/s, astern run, full throttle (S1 1926-1942) |
| `test_4` | 213.2 s | a 726 deg port turn and an 882 deg starboard turn, a crash stop from a turn, coast-down |

Pose at 10.0 Hz (0.1 m and 0.1 deg resolution), receiver lines at 0.99 Hz, a second heading
channel (HDG) at 10 Hz and LiDAR scans at 10 Hz; `WD` lines (S1 and S2 at 1500) while no
transmitter is bound. The pose extends 10.9 m sideways in `test_1` and 22.9 m along the basin in
`test_4`; each log starts its own origin.

**Limits of the data.** The receiver is sampled once a second, so a stick movement between two
samples is unknown and event times carry up to 1 s of uncertainty. There is no rudder angle, and
the receiver's end points are not known to equal the `$CMD` end points. S1 is a pulse width, not an
rpm-unit. At neutral (1511) the vessel crept at about 0.1 m/s ahead (`test_1` 157-198 s). The
logs are from a different day, and possibly a different set-up, than July.

## 3. Checks of the first measurements

| Claim | Check (`feb_analysis.txt`) | Result |
| --- | --- | --- |
| Port turn, radius 2.40 m at 0.85 m/s, 19.4 deg/s | `test_4` 20.4-56.3 s: 0.85 m/s, 20.2 deg/s, U/r = 2.42 m, circle fit 2.42 m (rms 0.17 m), 726 deg | confirmed |
| Starboard turn, 3.93 m at 1.34 m/s, 19.3 deg/s | `test_4` 140.8-185.8 s: 1.32 m/s, 19.6 deg/s, U/r = 3.85 m, circle fit 3.93 m, 882 deg | confirmed; the throttle rose from about 1630 to 1780 during the turn (sd 53) and the speed from 1.15 to 1.41 m/s, so it is not strictly steady |
| Prior: 5.3 m and 10.0 deg/s at RPM 12, 0.93 m/s in the turn | `results/field_sessions/manoeuvre_predictions.txt`; v3 at the measured speeds: 5.33 m, 9.2 deg/s at 0.85 m/s; 5.37 m, 14.0 deg/s at 1.32 m/s | confirmed; "twice as fast" holds for the port turn (2.2) only; starboard is 1.4 |
| `test_1` crosses 10.9 m sideways | x from -10.9 to 0.0 m | confirmed (pose of the reference point; the hull extends further) |
| S1 throttle, S2 rudder | S2 at 999-1010 gives r = -17.2 deg/s mean while moving (`test_4`), at 1950-1967 +16.7 deg/s, at 1455-1480 +0.5 deg/s; S1 tracks speed (section 6) and S1 below neutral decelerates and drives astern | confirmed; the `log_parser.py` docstring ("S1 - rudder, S2 - thruster") is wrong for these lines |

The name S2 means two things in this repository: the throttle-like `$CMD` field of the bridges
(S2 = rpm / 24 x 100) and the rudder channel of the receiver lines. A reader of mixed logs has to
keep them apart.

## 4. v3 against the turns

Settled full-rudder segments (S2 held at an end point, the first 2 s dropped), measured and v3 at
the same speed in the turn (`feb_analysis.txt` section 4):

| Segment | Side | S1 | U (m/s) | r (deg/s) | R (m) | v3 at same U: r, R | Ratio of yaw rates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `test_4` 20-56 s | port | 1629 | 0.85 | 20.2 | 2.42 | 9.2, 5.33 | 2.21 |
| `test_4` 141-186 s | stbd | 1709 | 1.32 | 19.6 | 3.85 | 14.0, 5.37 | 1.39 |
| `test_4` 61-67 s | stbd | 1644 | 1.18 | 16.5 | 4.11 | 12.7, 5.36 | 1.30 |
| `test_2` 26-34 s | port | 1590 | 0.73 | 16.0 | 2.59 | 7.8, 5.31 | 2.05 |
| `test_3` 37-46 s | port | 1709 | 0.69 | 17.3 | 2.29 | 7.5, 5.31 | 2.31 |
| `test_1` 59-65 s | stbd | 1811 | 0.89 | 14.5 | 3.50 | 9.5, 5.33 | 1.53 |

Speed in the turn over the straight speed at the same throttle: 0.68 and 0.78 (port), 0.90
(starboard), against v3's 0.83. The port turns are tighter and slower, the starboard ones wider
and faster. With S2 at 1000 and 1967 lying 467 and 500 microseconds from the neutral 1467, the
larger nominal deflection is on the side that turns less, so the asymmetry is not explained by the
receiver end points; servo travel, rudder stops or propeller torque are possible, and S1's rudder
angle measurement (P-3) separates them.

**Drift.** After the constant +3.7 deg offset of straight running is removed, the drift at the pose
point in the settled turns is +0.4 deg (port) and +0.1 and +0.5 deg (starboard), outward. v3's
full-helm turn has 10.2 deg at midship and 0.9 deg at the LiDAR position: the pose point behaves
like a point near the bow, close to the pivot point, consistent with a pose reported at the LiDAR
(O-9). The July fit compared the model's midship with the pose, so the identified sway is that of
a point near the bow.

**Windowed replay** of the forward-motion segments (pose at 2 Hz as in July, receiver values mapped
to +-100 % `$CMD` and to rpm through the measured throttle map; each window starts from the
measured state):

| Horizon | Hold heading | Constant yaw rate | v3 heading | v3 position | v3, July holdout |
| --- | --- | --- | --- | --- | --- |
| 1 s | 6.39 | 0.54 | 2.03 deg | 0.11 m | 1.31 deg |
| 3 s | 26.53 | 4.83 | 10.91 deg | 0.49 m | 7.90 deg |
| 5 s | 44.73 | 16.78 | 19.09 deg | 1.00 m | 10.17 deg |
| 10 s | 89.27 | 55.79 | 38.52 deg | 2.71 m | 13.82 deg |

In sustained turns a constant-yaw-rate extrapolation beats v3 up to 5 s.

**Coverage of the training randomisation** (`bluefin/explain_numbers.txt`): over 200 draws of
`sample_params(scale = 1.0)` the full-helm radius at RPM 6 spans 2.29-5.40 m (5-95 %:
3.14-5.29 m). The starboard turn (3.9 m) lies inside, the port turn (2.4 m) at the lowest draw.

## 5. Can the February data re-identify the turn regime?

**What they constrain.** The full-helm steady turn on both sides at 0.65-1.32 m/s (rate, radius,
speed loss), the yaw decay after the rudder is centred, and the yaw at the end of stops from a
turn.

**What they cannot.** Intermediate helm (the transmitter was used at its end points); actuator
timing (1 s receiver sampling); the rudder angle; rpm-units (S1 is a pulse width); the cause of
the side asymmetry; and the equality of a full receiver deflection with a `$CMD` of +-100 %.

**Full receiver deflection against `$CMD` +-100 %.** Not decidable from the logs. The evidence:
- the receiver's end points are 1000 and 1967 around a neutral of 1467, not symmetric;
- in July, 12 holds of |`$CMD`| >= 95 % for 1.5-2.5 s at 1.03-1.24 m/s reached 9.6-22.2 deg/s
  (median 15.6) before the policy reversed the rudder (`july_full_command.txt`); in February the
  full-receiver turns settled at 19.6-20.2 deg/s and the step into the port turn at 1.34 m/s
  peaked at 28.4 deg/s within 3 s (`feb_analysis.txt` section 4). The July holds are too short to
  reach a steady turn, so the two are of the same size without being comparable;
- the joint fit below does not contradict the July holdout from 2 s to 10 s, but costs free-run
  accuracy.

The vessel program's mapping from the `$CMD` rudder to the servo pulse decides it, and one
`$CMD` turn at +-100 % beside one transmitter full-helm turn at the same throttle in the same
session measures it directly.

**Candidate analysis model v3f** (`refit_feb.py`, set-up fixed in its docstring before the fit):
the v3 structure unchanged; nine turn-regime parameters free; thrust, resistance and actuator fixed
at v3; N_rr tied to Y_vv by strip theory; the July training session (retrievals trimmed by refit
v4's rule) and the February forward-motion segments weighted equally at the v3 solution; v3's
steady-turn bands dropped and its two structural priors kept. 80 generations of differential
evolution from v3 and from a random population, 497 s; the start from v3 reached J = 1.777, the
random start 1.871 (J(v3) = 2.347), so the optimum is not unique.

| Parameter | v3 | v3f |
| --- | --- | --- |
| X_vv | 50.64 | 15.02 |
| X_rr | 0.058 | 3.00 |
| X_delta | 0.116 | 0.407 |
| Y_v | 0.924 | 0.000 (at bound) |
| Y_vv | 220.98 | 244.58 |
| k_R | 2.18 | 6.05 |
| k_race | 0.0135 | 0.0166 (at bound) |
| N_r | 1.04 | 26.47 |
| N_rr (tied) | 0.017 | 39.23 |
| N_uv | -83.97 | -29.39 |

The yaw damping moves from the drift moment (v3: net +24.89 u v) to yaw-rate damping (N_r u r and
N_rr r|r|), with the net drift moment now destabilising (-29.69 u v), as in refit v4b
(N_r = 25.93, N_uv = -47.01, `REVIEW.md` section 8.3).

| Check (`refit_report.txt`) | v3 | v4 | v4b | v3f |
| --- | --- | --- | --- | --- |
| July holdout heading, 1 / 2 / 3 / 5 / 10 s (deg) | 1.31 / 5.19 / 7.90 / 10.17 / 13.82 | 1.45 / 4.21 / 6.75 / 7.49 / 11.82 | 1.42 / 3.80 / 6.25 / 6.75 / 9.33 | 1.61 / 4.05 / 6.48 / 7.64 / 11.35 |
| July holdout heading, free run (deg) | 23.53 | 24.12 | 21.14 | 28.96 |
| July holdout position at 10 s / free (m) | 1.25 / 3.53 | 0.86 / 3.46 | 0.92 / 3.26 | 0.83 / 3.97 |
| July holdout path length, mean absolute error (m) | 1.76 | 1.25 | 2.41 | 1.31 |
| February segments heading, 3 / 10 s (deg) | 10.91 / 38.52 | 9.62 / 35.45 | 9.41 / 34.51 | 6.94 / 18.37 |
| Port turn at 0.85 m/s (measured 20.2 deg/s, 2.42 m) | 9.2, 5.33 | 11.3, 4.33 | 11.8, 4.15 | 15.7, 3.10 |
| Starboard turn at 1.32 m/s (measured 19.6 deg/s, 3.85 m) | 14.0, 5.37 | 17.1, 4.40 | 18.0, 4.19 | 24.2, 3.12 |
| Stop from the `test_4` turn, 17 N: heading change in 5 s / yaw rate at zero surge (measured 21 deg / below 3 deg/s) | 21 deg / 9.8 deg/s | 18 / 1.9 | 7 / 0.7 | 19 / 0.0 |
| From 10 deg/s at rest: time to 0.5 deg/s, heading change | 28.3 s, 131 deg | 7.5 s, 17 deg | 11.9 s, 56 deg | 5.6 s, 17 deg |

The v4 and v4b holdout figures reproduce `REVIEW.md` section 8 exactly, which checks the
evaluation. The July-only refits already turn tighter than v3 (4.2-4.4 m); the February turns pull
further (3.1 m) at a cost on the July data against v4b. The candidate's steady turns: 1.1 / 3.8 /
6.0 / 7.6 deg/s at helm 25-100 % and RPM 6 (radius 3.06 m, speed ratio 0.72), 2.2 / 7.5 / 11.8 /
15.0 deg/s at RPM 12; bounded at RPM 24 and full helm.

**Decision.** v3f is an analysis candidate. It must not replace the training plant (frozen) nor be
presented as the calibrated model, which the S1 fit produces. Until then it can serve one purpose:
a second plant for the sim-to-field comparison that brackets v3 in turning authority, to show how
much a closed-loop twin depends on it. Its parameters load with `ship.ShipModel(params=...)`; it
has no bootstrap and is not for `sample_params`.

## 6. The low-speed yaw artefact

**Mechanism.** In v3 every yaw moment except N_rr r|r| (N_rr = 0.017) scales with surge. At zero
surge the only remaining moment is the rudder's at its 0.05 m/s inflow floor, so a yaw rate of
10 deg/s decays over 28.3 s through 131 deg. This is what produces the 95-162 deg of heading
change over 60-80 s after a stop from a settled turn at RPM 12 in
`results/field_sessions/manoeuvre_predictions.txt` (full astern at -24 rpm-units, strong and weak
reverse models).

**Measured.** Two full-reverse stops from settled starboard turns (`test_1` 65 s from 1.35 m/s
and 19.0 deg/s; `test_4` 68 s from 1.20 m/s and 16.7 deg/s): heading change 26 and 21 deg in the
5 s from half a second before the first reverse sample, about 9 deg of which can precede the stick
movement; yaw rate 2.9 and 0.6 deg/s when surge fell below 0.15 m/s.

**Straight stops turn too.** The two straight full-reverse stops of `test_3` (from 1.30 m/s, rudder
neutral) turned 19 and 18 deg in 5 s, both to port, with 6.1 and 10.6 deg/s at the moment surge
fell below 0.15 m/s; the partial reverse at `test_4` 107 s turned 14 deg. v3 has no yaw moment
from reverse thrust and turns none. A propeller-induced side force when going astern would do
this; so would contact with the end wall, since every stop ended within 0.1-1.0 m (pose) of the
largest or smallest y its log reached. S1-C2's straight stops, mid-basin and with the IMU,
separate the two; if the turning is the propeller's, T-13 applies to straight stops as well.

**v3, the same manoeuvre** (settled at the measured speed in its own full-helm turn, rudder
centred, 17 N reverse): 21-24 deg in 5 s, close to the measurement, but 9.8-10.5 deg/s left at zero
surge and 186-196 deg in 60 s; with the weak bracket model (16.1 N after 0.75 s) 20 deg in 5 s,
9.5-9.8 deg/s at zero surge and 169-180 deg in 60 s. The first seconds are right; what follows zero
surge is not.

**Consequence.** Any simulated stop that begins with a yaw rate overstates the heading change
after the vessel comes to rest; the predictions for T-13 (drift after a stop from a turn) are
affected, the straight stops are not.

**Repair for analysis models.** A cross-flow yaw damping consistent with the identified sway
drag: for a hull of constant draft the cross-flow drag that gives Y_vv v|v| gives
N_rr = Y_vv L^3 / 32 in yaw (35.4 for v3's Y_vv). In v3f this leaves 0-0.8 deg/s at zero surge and
the July holdout is not worse from 2 s on. The complete form is Fossen's cross-flow drag integral,
which couples v and r and acts at any surge; it is a structural option for the S1 fit.

## 7. Thrust curve from the throttle

Settled straight speeds (`feb_analysis.txt` section 3) and the thrust they imply through v3's
resistance X_uu U^2:

| S1 (microseconds above neutral) | Straight speed (m/s) | Implied thrust (N) | v3 rpm-units for that speed |
| --- | --- | --- | --- |
| 1544-1551 (+33 to +40) | 0.65-0.68 (one sample 0.50) | 5.4-6.0 | 7.0-7.3 |
| 1587-1598 (+76 to +87) | 0.88-0.97 | 10.0-12.1 | 9.4-10.4 |
| 1627-1652 (+116 to +141) | 1.22-1.34 | 19.2-23.1 | 13.1-14.4 |
| 1926-1942 (+415 to +431) | at least 1.56 (still accelerating at the basin's end) | at least 31.3 | at least 16.8 |

The curve rises steeply over the first 130 microseconds of stick travel and then flattens: full
throttle gave at least 1.56 m/s, where v3's square law puts RPM 24 at 2.23 m/s (64.3 N). If a
`$CMD` of 100 % equals full throttle, v3 overstates the top of the thrust range by up to a factor
of two. The July point, `$CMD` 50 % (RPM 12) at 1.14 m/s, lies at S1 about 1612 on this curve; a
map placing 50 % at the middle of the forward stick travel (about 1725) would give at least
1.4 m/s. Either the vessel program maps `$CMD` onto part of the range, or the vessel differed
between the two days; the program's mapping answers it.

**Reverse.** Full reverse (S1 1007-1012) braked at 15.3-19.1 N, mean 16.8 N, over the window where
reverse is certain. Against the full forward thrust (at least 31.3 N) the ratio is at most 0.54,
in line with the assumed `REVERSE_THRUST_EFFICIENCY` of 0.5 as a ratio; but the simulator's full
astern is 0.5 x T(24) = 32.1 N in absolute terms, about twice the measured force, which is
0.26 x T(24) (0.24-0.30), the weak end of the safety layer's bracket. Run through the same
manoeuvres, the weak model (0.25 x T(24) after a 0.75 s onset) stops in 4.1-4.4 s and 2.2-2.7 m
from 1.2-1.35 m/s, the strong one in 2.0-2.3 s and 1.1-1.4 m, the measurement in 3.5-5.3 s and
2.6-3.1 m (`feb_analysis.txt` section 5). These hold if a full-reverse stick equals `$CMD`
S2 = -100; the onset delay cannot be measured with 1 Hz receiver samples. Under reverse the
vessel went astern at 0.72-0.80 m/s (`test_1` 51-57 s and 133-139 s, `test_3` 31-35 s). One partial
reverse (S1 1105-1115, `test_4` 192 s) braked only briefly (about 10 N) and the vessel held
0.4-0.5 m/s: a dead band, a brake-before-reverse behaviour of the ESC, or a stick movement between
receiver samples. `test_1` went from S1 1869 to 1011 between two consecutive samples, a direct
full-ahead to full-astern reversal (O-7).

**Coast-downs** at neutral throttle decay more slowly than v3 with zero thrust (from 0.90 m/s:
0.58 against 0.49 m/s after 5 s, 0.48 against 0.33 m/s after 10 s; similar from 0.45-0.57 m/s).
Either the neutral 1511 is not zero thrust (it held 0.1 m/s ahead) or the resistance at low speed
is below the quadratic law. This bears on the policy's speed range (RPM 4.5-7.5, 0.42-0.70 m/s).

## 8. What the calibration sessions must measure

| Point | Measurement that settles it | In the plan | Change proposed |
| --- | --- | --- | --- |
| Full receiver deflection against `$CMD` +-100 % | read the vessel program's `$CMD` to pulse-width mapping (rudder and propulsion); one `udp_manoeuvre.py turn` at +-100 % beside one transmitter full-helm turn at the same throttle, same session | not covered | a bench item before S1b; the A/B pair in S1a or at the start of S1-D |
| Turn regime | S1-D circles at helm 25-100 %, both sides, RPM 6 and 12; rudder angle at each setting (P-3) | S1-D, T-4, T-5 | expected results to show the February measurements and v3f beside v3; full circles at 75-100 % helm |
| Side asymmetry | rudder angle at both end points and at 25-75 % (protractor or feedback), both sides of S1-D | S1-B, S1-D (both signs) | record the servo end points explicitly |
| Yaw damping at low speed | the IMU's yaw rate after the hand push at rest (D0-2, S1-A) gives the zero-speed decay directly; stops from turns (S1-C2) give it with surge falling | D0-2, S1-A, S1-C2, T-13 | log the push decay at the IMU rate and fit N_rr from it |
| Reverse force and onset | S1-C2 under `udp_manoeuvre.py stop` with `#CMD` timing (the receiver's 1 Hz could not time the onset) | S1-C2, T-9, T-10, T-12 | expected results to show the February 15-19 N (0.24-0.30 of T(24), the weak end) and 2.6-3.1 m from 1.2-1.35 m/s |
| Yaw in straight stops | S1-C2 straight stops in mid-basin, IMU logged, heading change and yaw rate to rest | S1-C2, T-13 | extend T-13 to the straight stops |
| ESC reverse through `$CMD` | the February logs show reverse under radio control; whether the vessel program passes a negative `$CMD` S2 to the ESC remains | P-8, O-6 | narrow P-8 to the `$CMD` path and the dead band |
| Thrust curve | S1-C RPM sweep by `#CMD`; the vessel program's mapping | S1-C, T-3 | the top settings (RPM 18-24) are where v3 and the February curve disagree most |
| Coast-down drag | coast-downs with the propeller at true zero thrust (ESC neutral confirmed) | S1-C | note the 0.1 m/s creep at the February neutral |
| Pose reference point | survey (S1-A); the fit to compare the model at the pose point, not at midship | S1-A, O-9 | the drift evidence above supports the LiDAR position |
| Heading offset | +3.7 deg (February), +2.2 deg (July) between pose heading and course in straight running | not covered | fit a constant heading offset, or measure the LiDAR's yaw alignment at the survey |
| Basin width | at least 10.9 m of pose excursion | P-7 survey | the map's 10 m width to be replaced by the surveyed one |

## 9. Proposed changes to the session plans (for the owner of `planning/`)

1. `FIELD_SESSIONS_PLAN.md` S1a and S1-D expected results: add the February measurements
   (2.42 m and 20.2 deg/s port at 0.85 m/s; 3.85 m and 19.6 deg/s starboard at 1.32 m/s) and the
   v3f predictions (radius 3.06-3.12 m, 7.6 deg/s at RPM 6, 15.0 deg/s at RPM 12) beside v3's,
   with the side asymmetry flagged.
2. S1-C2 expected results: add the February full-reverse stops (15-19 N, the weak end of the
   bracket; from 1.2-1.35 m/s, 3.5-5.3 s and 2.6-3.1 m from half a second before the reverse
   sample; 21-26 deg in 5 s from a turn, and 18-19 deg from the straight stops) and state that the
   95-162 deg v3 turns after surge reaches zero is a model artefact (section 6).
3. A bench item before S1b: read the vessel program's `$CMD` mapping for rudder and propulsion
   (pulse width per percent, end points, dead band, whether a negative S2 reaches the ESC).
4. An A/B pair in S1a or S1-D: a `$CMD` +-100 % turn and a transmitter full-helm turn at the same
   throttle, so the February data join the calibration.
5. D0-2 and S1-A: keep the IMU's record of the yaw push to rest; it measures the zero-speed yaw
   damping that section 6 is about.
6. D0-4 (manual driving): repeat the February manoeuvres with the IMU (full-helm circles both
   ways at two throttles, a crash stop from straight and from a turn, a coast-down), so that the
   February results gain a rudder-independent yaw-rate record.
7. P-8 and O-6: the ESC reverses under radio control (astern to 0.80 m/s); the open question is the
   `$CMD` path. O-7: one direct full-ahead to full-astern reversal occurred in `test_1` at 65 s.
