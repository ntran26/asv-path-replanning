# D0 operations, Thursday 2026-11-26 (data collection, then S1a if D0 passes)

Step-by-step procedure for the basin session of 2026-11-26 on the Ubuntu lab laptop (host
SIMDELL), the shore computer. Sources:
- `static_dynamic_obstacles/planning/FIELD_SESSIONS_PLAN.md` sections 2 (roles, equipment,
  records, checks, abort rules), 4 (D0) and 5.1 (S1a);
- `static_dynamic_obstacles/planning/FIELD_CALIBRATION_PLAN.md` (items and thresholds);
- `static_dynamic_obstacles/results/field_sessions/manoeuvre_predictions.txt` (model predictions);
- the route figures in `static_dynamic_obstacles/planning/field_sessions_figures/`.

The scripts are described in full in `FIELD_SCRIPTS_GUIDE.md`. Every command here that needs no
vessel was run on the laptop on 2026-11-25 (Python 3.10.11, repository at b87cc9bed); the quoted
output is from those runs. Steps that need the vessel are marked **on the day**.

Contents:
1. Before leaving
2. Roles
3. At the basin: laptop set-up
4. Session start checks
5. D0-1 to D0-5
6. The S1a gate
7. S1a (about 90 min)
8. Abort rules
9. End of session
10. After the session
11. Run sheet

## 1. Before leaving

- [ ] Bluefin with the IMU fitted and secured.
- [ ] At least two charged battery sets for the Bluefin; the radio-control transmitter charged,
      with its batteries.
- [ ] The vessel-control laptop charged, its vessel program working.
- [ ] The Ubuntu laptop charged, with its charger. Python and the tools were checked on
      2026-11-25: both self-tests passed and the B-8 bench passed.
- [ ] Tape measure, tape or chalk for the marks (three rest positions; four points 1 m from the
      walls), a protractor for the rudder post if available, a side camera or phone.
- [ ] The run sheet (section 11), printed or on a second device.
- [ ] The risk assessment with the abort rules of section 8.

## 2. Roles (FIELD_SESSIONS_PLAN.md section 2.1)

| Role | Task |
| --- | --- |
| Operator | the Ubuntu laptop: logger and manoeuvre sender, run identifiers, Ctrl+C |
| Safety | the radio-control transmitter; takes over on any stop call; recovers the vessel. Never also the operator during a scripted run |
| Recorder | run sheet, times, battery voltage, video, deviations |

One person may hold two roles in D0. During any run in which `udp_manoeuvre.py` sends commands,
the operator and safety roles are held by different people.

## 3. At the basin: laptop set-up

1. **Network.** Join the network used in July (P-11), not home Wi-Fi. Check the laptop's
   address:

   ```bash
   ip -brief addr
   ```

   The Wi-Fi line (`wlp2s0`) must show an address on the vessel-control laptop's network. At home
   on 2026-11-25 it showed `192.168.0.213/24`, the home network, which is wrong for the basin.
2. **Vessel-control laptop's IP.** Read it on that laptop (10.201.205.110 in July) and write it on
   the run sheet. Below it is `<IP>`. Check that it answers (**on the day**):

   ```bash
   ping -c 3 <IP>
   ```

   A ping that fails does not settle it (the other laptop may drop pings); the logger's handshake
   in D0-1 does.
3. **Firewall stays off; port 5000 free.**

   ```bash
   grep ENABLED /etc/ufw/ufw.conf
   ss -ulpn | grep -E ':5000\b' || echo "UDP 5000 free"
   ```

   Expected (as on 2026-11-25):

   ```
   ENABLED=no
   UDP 5000 free
   ```

4. **Python and the working folder.** Python 3.10.11 is the command `python`. On this laptop
   `python3` is Ubuntu's own 3.12; the tools also run on it, but the project uses `python`.

   ```bash
   cd ~/asv-path-replanning/asv-lidar/field_deployment
   python --version
   git log -1 --oneline
   mkdir -p 2026-11-26
   ```

   Expected: `Python 3.10.11`. Write the commit (b87cc9bed or later) on the run sheet. Do not run
   `git pull` at the basin.
5. **Self-test, once more:**

   ```bash
   python udp_raw_logger.py --selftest
   ```

   Expected: `[SELFTEST] PASSED: handshake, 14 datagrams logged byte for byte ...`.

All logs go in `2026-11-26/`, one file per run, named by the run identifier (D0-1 ... D0-5;
S1a-CHK-01, S1a-A-01, ...). An existing name is never overwritten (`_2` is added). Only one
program receives at a time: stop the logger (Ctrl+C) before the manoeuvre sender starts, and the
other way round.

## 4. Session start checks (FIELD_SESSIONS_PLAN.md section 2.4)

D0-1 and D0-2 are the link check and the clock and IMU check of section 2.4; they are not
repeated before them. The third check:

- **Radio-control stop (on the day):** with the vessel moving slowly under radio control, the
  safety person stops it once by the override or kill (P-12). Note it on the run sheet. The E key
  belongs to the bridge, which is not used today.

## 5. D0-1 to D0-5 (on the day)

Before each experiment, the vessel is at its mark, the operator states the run identifier, and the
recorder notes the start time. After each, run `--check` on its log (section 5.6).

### 5.1 D0-1 Link (5 min)

```bash
python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/D0-1.log --duration-s 120
```

- **Procedure:** vessel at rest at the mid-basin mark; start the logger; record 2 min (it stops by
  itself).
- **Expected:**
  - `[RAW] first datagram after` a few seconds; `hey` lines in the table, and `vessel answered
    hey` in the meta file;
  - LiDAR lines of 720 integers and pose lines of three values, at 2 Hz (the July mode) or 10 Hz
    (the February mode);
  - IMU lines as a new kind, printed in full when they first arrive. About 100 per second if the
    IMU streams raw at 100 Hz.
- **If nothing arrives within 10 s:**
  - the logger prints that the handshake was sent again;
  - check the IP, the network (section 3), and that the vessel program runs and listens on 5050;
  - Ctrl+C, correct, start again (the next file is `D0-1_2.log`; keep both).
- **If the IMU lines show `decoder error` or `taken as a scan`:** expected for a new bracketed
  format, and recorded as such. The logger keeps everything; only the bridges would need the
  decoder change (B-3).

### 5.2 D0-2 At rest, with the yaw push (6 min)

```bash
python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/D0-2.log --duration-s 360
```

- **Procedure:** vessel at rest at a marked spot, heading noted.
  - 0:00-5:00: untouched.
  - About 5:00: one sharp yaw push by hand at the hull. The recorder notes the time to the second.
  - 5:00-6:00: untouched.
- **Expected:**
  - gyro bias and noise in the IMU lines;
  - the pose steady within its 0.1 m and 0.1 deg resolution, with no jumps;
  - the push visible in both the IMU yaw rate and the pose heading. Their time difference is the
    clock offset (P-5), read after the session.

### 5.3 D0-3 Telemetry modes (5 min; only if the vessel program has a rate setting)

```bash
python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/D0-3a.log --duration-s 60
```

- One log per mode (`D0-3a`, `D0-3b`, ...), vessel at rest. Note each mode's setting on the run
  sheet.
- **Expected:** the highest rate the stream sustains (the status line's rate, and `--check`), the
  pose resolution in each mode, and whether the IMU's rate depends on the mode.
- If there is no rate setting, skip D0-3 and note "no rate setting".

### 5.4 D0-4 Manual driving (15 min)

```bash
python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/D0-4.log
```

- **Procedure, under radio control:**
  - straight runs along the basin at two steady throttle settings;
  - gentle and hard turns to both sides;
  - stops by cutting the throttle.

  The recorder notes each segment's start time and setting. Stop the logger with Ctrl+C at the
  end.
- **Expected:**
  - the IMU yaw rate follows the pose-derived yaw rate (sign and scale);
  - the surge acceleration matches the change of the pose-derived speed;
  - `RC` lines (about 1 Hz) carry the stick positions.

  February reference (radio control, `data/test_4.log`): full-rudder circles of radius 2.4 m at
  0.85 m/s and 3.9 m at 1.34 m/s, both at about 19 deg/s; straight runs up to 1.62 m/s.
- If time is left at the end of the booking and S1a does not run, repeat D0-4 with more variety
  (it serves the IMU decoder, B-3).

### 5.5 D0-5 Basin frame (5 min)

```bash
python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/D0-5.log
```

- **Procedure:** hold the vessel for about 30 s at each of four marks: 1 m from the left wall,
  1 m from the right wall, 1 m from the near end, and 1 m from the far end. The recorder notes the
  time and the mark for each. Ctrl+C at the end.
- **Expected:**
  - the basin's edges in the pose frame;
  - the pose origin and the axis directions. The February logs put the origin at the bottom right
    in some runs and the bottom middle in others, so the frame is read again today.
- **Fence for S1a:** the extremes of the pose over the log, each pushed out by 1 m to the wall
  (pose frame: for a raw line `[ts]A,B,C`, x = B and y = -A):

  ```bash
  python -c "import sys, udp_manoeuvre as m; p=[(float(g['x']), -float(g['y'])) for g in (m.RE_POSE.match(l.strip()) for l in open(sys.argv[1], errors='ignore')) if g]; xs, ys = [a for a, b in p], [b for a, b in p]; print('pose lines', len(p), '| x', min(xs), 'to', max(xs), '| y', min(ys), 'to', max(ys)); print('--fence=%.1f,%.1f,%.1f,%.1f' % (min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1))" 2026-11-26/D0-5.log
  ```

  Output of the same command on a rehearsal log (the July log `2026-07-03/1.log`, not a basin
  frame):

  ```
  pose lines 31 | x 0.1 to 1.3 | y 0.0 to 14.6
  --fence=-0.9,2.3,-1.0,15.6
  ```

  - Check the result against the marks: the box should be about as wide and as long as the
    measured basin. Any pose taken while moving between marks lies inside the four, so it does
    not widen the box. A jump in the pose (a localisation glitch) would: if the box is clearly
    too large, read the pose at each noted time instead.
  - Write the `--fence=...` string on the run sheet. Below it is `<FENCE>`.
  - Open point for the day: every manoeuvre is planned in a 10 m x 25 m basin, while the real
    basin is wider (the February logs crossed 10.9 m). The fence from D0-5 is the real basin; the
    routes in section 7 keep to the planned 10 m.

### 5.6 After each experiment

```bash
python udp_raw_logger.py --check 2026-11-26/D0-1.log
```

Output format (from the rehearsal log):

```
.../D0-mock.log: 61 lines
  line kind                             count   rate /s  decoder
  pose (3 values)                          31       2.0  read
  LiDAR scan (720)                         30       2.0  read
```

Note on the run sheet each line kind with its rate. The rates come from the vessel's time stamps,
which answers the IMU rate question directly.

## 6. The S1a gate (FIELD_SESSIONS_PLAN.md section 4.5)

S1a starts only if all four hold:

| Condition | How to check | Result |
| --- | --- | --- |
| The link ran for 5 min with no gap longer than 1 s | the largest gap in D0-2 (6 min), command below | |
| IMU lines are present and carry the vessel's time stamps | `--check` on D0-1: an IMU kind with a rate (a rate needs the vessel's stamps) | |
| The basin frame is read (D0-5) | a `<FENCE>` string written down and checked against the marks | |
| At least 90 min of basin time and a fresh battery set remain | the clock; the battery sets | |

Largest gap between datagrams in a log:

```bash
python -c "import csv, sys; t = [float(r['t_since_start_s']) for r in csv.DictReader(open(sys.argv[1] + '.arrivals.csv'))]; print('datagrams %d, span %.1f s, largest gap %.2f s' % (len(t), t[-1] - t[0], max(b - a for a, b in zip(t, t[1:]))))" 2026-11-26/D0-2.log
```

Output on the rehearsal log: `datagrams 61, span 15.1 s, largest gap 0.50 s`. A gap of 0.50 s is
the 2 Hz stream itself; the gate needs the largest gap at or below 1 s.

If any condition fails, S1a does not run. Repeat D0-4 with more variety for the rest of the
booking, and note which condition failed.

## 7. S1a (about 90 min; on the day)

All S1a commands use the same start. Below it is written `...`:

```bash
python udp_manoeuvre.py --server-ip <IP> --fence=<FENCE> --out 2026-11-26/<run id>.log
```

How every run goes:
1. The vessel is at its start mark; the safety person is ready on the transmitter.
2. Start the command. Wait for `[MAN] telemetry flowing`.
3. If the fence line says `OUTSIDE`, do not press Enter: move the vessel or correct the fence.
4. Press Enter to start.
5. Ctrl+C stops at once (three zero commands).

Stops:
- A fence trip cuts propulsion and prints `TAKE OVER BY RADIO CONTROL`; the safety person takes
  over. Crash stops (astern) are not run today: the ESC reversal (P-8) is untested, so no command
  uses `--allow-astern`.
- If the vessel-control laptop does not take `$CMD` at all, the blocks run under radio control
  instead (FIELD_SESSIONS_PLAN.md section 5.3, RC-1 to RC-3), logged with `udp_raw_logger.py`.

### 7.1 First check on the vessel (rudder sign)

```bash
... --out 2026-11-26/S1a-CHK-01.log hold --rpm 6 --duration-s 10
... --out 2026-11-26/S1a-CHK-02.log turn --rpm 6 --rudder 50 --duration-s 5
```

- **Hold:** S2 25 sent (`$CMD,0.00,25.00`); the vessel moves ahead slowly. Model: steady speed
  0.56 m/s at RPM 6, 90 % of it after 14 s and 4.4 m, so after 10 s about 0.5 m/s and 3-4 m run.
  - If the vessel does not move: the vessel-control laptop is not taking `$CMD`. Check its program,
    then go to the radio-control back-up.
  - If the vessel goes astern: stop (Ctrl+C) and report. Do not continue.
- **Turn:**
  - a 14 s straight run-up at RPM 6 (the default `--runup-s`), then rudder +50 % for 5 s, then
    straight for 3 s: about 22 s and 10 m in all, so start at one end of the basin;
  - the vessel receives `$CMD,-50.00,25.00` (rudder sign -1);
  - model: about 2.5 deg/s at steady state (helm 50 %, RPM 6). The February turns were about
    twice as fast;
  - it reports `turning side=starboard (heading increasing), as_expected=True`. If it reports
    `port ... as_expected=False`, the sign is reversed: repeat as `S1a-CHK-03` with
    `--rudder-sign 1` added before the subcommand, and use that setting for every later S1a
    command.

### 7.2 S1-A Sensing baseline (25 min)

At rest, so the logger is used (no command needed):

```bash
python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/S1a-A-01.log --duration-s 330
```

- Three runs, `S1a-A-01` to `S1a-A-03`, at the three rest marks: mid-basin, near the light wall,
  near the black wall (figure `cal_01_rest_positions.png`).
- At each: heading noted, 5 min untouched, one sharp yaw push at about 5:00 (time noted), then
  30 s.
- **Expected:**
  - gyro bias;
  - localisation drift at rest (T-8);
  - LiDAR range noise;
  - the return rate against the black wall (T-7);
  - the pose reference point (O-9), from the marks' measured positions.

### 7.3 Coast-downs (20 min)

```bash
... --out 2026-11-26/S1a-C-01.log coast --rpm 12 --runup-s 10
... --out 2026-11-26/S1a-C-04.log coast --rpm 9 --runup-s 12
```

- Three runs each (`S1a-C-01` to `-03` at RPM 12, `-04` to `-06` at RPM 9), started at one end of
  the basin, along its length (figure `cal_02_surge_coast.png`).
- The coast lasts 20 s (`--coast-s` default).
- **Expected (model):**
  - from RPM 12 (1.12 m/s): half speed after 4.7 s; 0.1 m/s after 48 s and 12.8 m;
  - from RPM 9 (0.84 m/s): half speed after 6.3 s; 0.1 m/s after 47 s and 11.3 m.
- The first 5-10 s carry most of the drag information. The far wall cuts the tail: a fence trip
  during the coast only adds the take-over call.

### 7.4 Full-rudder turning circles (30 min)

```bash
... --out 2026-11-26/S1a-D-01.log turn --rpm 6 --rudder 100 --heading-change 720
... --out 2026-11-26/S1a-D-03.log turn --rpm 6 --rudder -100 --heading-change 720
... --out 2026-11-26/S1a-D-05.log turn --rpm 12 --rudder 100 --heading-change 720
... --out 2026-11-26/S1a-D-07.log turn --rpm 12 --rudder -100 --heading-change 720
```

- Two runs each (`S1a-D-01` to `-08`).
- Run-up along the side opposite the turn, so the circle opens across the basin (figure
  `cal_03_turning_circles.png`): starboard turns (`--rudder 100`) along x = 2.5 m, port turns
  (`--rudder -100`) along x = 7.5 m of the 10 m width.
- At RPM 6, add `--max-s 240` before the subcommand: two full turns take about 140 s in the model,
  plus the 14 s run-up, close to the 180 s default.
- **Expected:**

  | Setting | Model (current simulator plant) | February 2026, radio control |
  | --- | --- | --- |
  | RPM 6, helm 100 % | 5.1 deg/s, radius 5.3 m, 0.46 m/s in the turn, 360 deg in 69 s, tactical diameter 10.5 m | - |
  | RPM 12, helm 100 % | 10.0 deg/s, radius 5.3 m, 0.93 m/s in the turn (0.83 of straight), 360 deg in 36 s, tactical diameter 10.7 m | radius 2.4 m at 0.85 m/s and 3.9 m at 1.34 m/s, both at about 19 deg/s |

- The measured circles are expected near February's, not the model's. A 2.4 m circle (4.8 m
  across) fits the 10 m width; the 3.9 m one (8.6 m across in February) and the model's would
  reach the fence and end as arcs (a fence trip: take over by radio control).
- RPM 18 runs only if the RPM 12 circle measured no more than 7 m across (the width inside the
  fence).

### 7.5 First zig-zag (15 min)

```bash
... --out 2026-11-26/S1a-E-01.log zigzag --rpm 12 --delta 20 --psi 20 --switches 2
... --out 2026-11-26/S1a-E-02.log zigzag --rpm 12 --delta -20 --psi 20 --switches 2
```

- From one end of the basin, both initial sides: `--delta 20` starts to starboard, `--delta -20`
  to port (figure `cal_05_zigzags.png`). A dry run on 2026-11-25 confirmed that `--delta -20` gives
  `first helm -50` (20 deg of the 40 deg maximum) and sends `$CMD,50.00,50.00`.
- Two switches give the first and second overshoots; four do not fit in 25 m.
- **Expected (model, 20/20 at RPM 12):**
  - overshoots 4.8 and 5.2 deg;
  - period 15.9 s;
  - lateral excursion 1.5 m;
  - 13.8 m run up to the second overshoot (with the 7 s run-up, about 20 of the 25 m).

  With turns as tight as February's, larger overshoots are likely. The overshoots and switch times
  are in the run's `.meta.json` events.

## 8. Abort rules (FIELD_SESSIONS_PLAN.md section 2.5)

Stop the run (Ctrl+C on the laptop, and the radio-control override) when:
- telemetry stops for more than 1.5 s (the manoeuvre sender aborts by itself: `abort: telemetry
  lost for 1.5 s`);
- the vessel will strike a wall and no avoiding action has begun;
- the vessel leaves the navigable area by more than 0.5 m;
- the safety person calls a stop, or anything else is unsafe.

After any stop: the vessel recovered and at rest, the run marked on the run sheet with the reason,
its log kept.

## 9. End of session

1. **Clock and IMU check, repeated** (section 2.4 item 4), so drift over the session is measured:

   ```bash
   python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/END-1.log --duration-s 90
   ```

   30 s at rest, one sharp yaw push (time noted), 60 s at rest.
2. Vessel out, batteries off, the end time noted.

## 10. After the session

1. `--check` on every log:

   ```bash
   for f in 2026-11-26/*.log; do python udp_raw_logger.py --check "$f"; done
   ```

   Manoeuvre logs also hold `#CMD` and `#MAN` lines, which count as their own kinds.
2. Keep every log exactly as recorded, with its `.arrivals.csv`, `.meta.json` and, for the
   manoeuvre sender, `.commands.csv`. Nothing is edited or deleted. A run that went wrong is kept
   and marked invalid on the run sheet, with the reason.
3. Copy the folder `2026-11-26/` to a second place (USB stick or the personal laptop) before
   leaving the basin.
4. Report back to the development laptop the same day:
   - the list of logs with their line kinds and rates;
   - the IMU line format (one example line);
   - the telemetry mode;
   - the largest gap;
   - the `<FENCE>` string and the pose frame's origin and axes;
   - the rudder-sign result;
   - which S1a blocks ran;
   - any deviation.

   Committing the logs (to `field_deployment/2026-11-26/`) waits for a decision.
5. Next on the bench: the IMU decoder (B-3), the fit script's reader (P-6), and the decisions on
   P-1 and O-1.

## 11. Run sheet

Session D0 and S1a, 2026-11-26. Vessel-control IP: ________ Network: ________ Commit: ________
Vessel software version: ________ Telemetry mode (rate, resolution): ________
Basin state (water level, wind or draught from the doors): ________
Fence (`--fence=`): ________ Rudder sign: ________

| Run identifier | Start | End | Settings (helm, RPM or throttle, mark) | Battery V | Deviation | Valid? (reason if not) |
| --- | --- | --- | --- | --- | --- | --- |
| D0-1 | | | rest, mid-basin | | | |
| D0-2 | | | rest; push at ____ | | | |
| D0-3a | | | mode: | | | |
| D0-3b | | | mode: | | | |
| D0-4 | | | RC: throttle settings ____ / ____ | | | |
| D0-5 | | | marks at ____ / ____ / ____ / ____ | | | |
| S1a-CHK-01 | | | hold, RPM 6, 10 s | | | |
| S1a-CHK-02 | | | turn, RPM 6, helm 50 %, 5 s; side: | | | |
| S1a-A-01 | | | rest, mid-basin; push at ____ | | | |
| S1a-A-02 | | | rest, light wall; push at ____ | | | |
| S1a-A-03 | | | rest, black wall; push at ____ | | | |
| S1a-C-01 | | | coast, RPM 12 | | | |
| S1a-C-02 | | | coast, RPM 12 | | | |
| S1a-C-03 | | | coast, RPM 12 | | | |
| S1a-C-04 | | | coast, RPM 9 | | | |
| S1a-C-05 | | | coast, RPM 9 | | | |
| S1a-C-06 | | | coast, RPM 9 | | | |
| S1a-D-01 | | | turn, RPM 6, helm +100 % | | | |
| S1a-D-02 | | | turn, RPM 6, helm +100 % | | | |
| S1a-D-03 | | | turn, RPM 6, helm -100 % | | | |
| S1a-D-04 | | | turn, RPM 6, helm -100 % | | | |
| S1a-D-05 | | | turn, RPM 12, helm +100 % | | | |
| S1a-D-06 | | | turn, RPM 12, helm +100 % | | | |
| S1a-D-07 | | | turn, RPM 12, helm -100 % | | | |
| S1a-D-08 | | | turn, RPM 12, helm -100 % | | | |
| S1a-E-01 | | | zig-zag 20/20, RPM 12, starboard first | | | |
| S1a-E-02 | | | zig-zag 20/20, RPM 12, port first | | | |
| END-1 | | | rest; push at ____ | | | |

Block times (start and end): D0 ______ S1-A ______ coast ______ circles ______ zig-zag ______
