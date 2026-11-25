# Field scripts: how to use them (calibration and field trials)

Written 2026-11-25 on the Ubuntu lab laptop (host SIMDELL; Python 3.10.11; repository at
b87cc9bed), the shore computer for every basin session. Every command below that needs no vessel
was run on that laptop before it was written down; the quoted output is from those runs (paths
shortened). Steps that need the vessel are marked **on the day**.

Each script's own `--help` and docstring remain the reference; this guide shows how the scripts
fit together. All commands run from `asv-lidar/field_deployment/` with `python` (3.10.11, the
version on all three computers). On the Ubuntu laptop `python3` is Ubuntu's own 3.12 and is not
used for the project.

Contents:
1. Overview: which script for which job
2. The link to the vessel (ports, handshake, command line)
3. `udp_raw_logger.py`: raw telemetry logger (D0, at-rest blocks)
4. `udp_manoeuvre.py`: scripted calibration manoeuvres
5. `udp_live_p3.py`: the Paper 3 bridge (live, shadow mode, no-send mode)
6. Mock vessel: testing the UDP connection without the vessel
7. Replaying and viewing saved logs
8. Bench tools
9. Older and helper scripts
10. Same as July, and what changed
11. Testing the UDP connection, in order
12. Quick reference

## 1. Overview: which script for which job

| Job | Script | Sends commands? | Needs |
| --- | --- | --- | --- |
| Record the raw telemetry (D0, at-rest blocks, any new line format) | `udp_raw_logger.py` | never | standard library only |
| Calibration manoeuvres with exact, timed commands (S1, S2-B) | `udp_manoeuvre.py` | yes (`--dry-run`: no) | standard library only |
| The deployed policy on the vessel (S2-C, trials) | `udp_live_p3.py` | yes (`--no-send`: no) | the pinned Python environment, the policy |
| A stand-in vessel replaying a log over UDP (bench) | `fake_vessel_replay.py` | - | standard library only |
| The bridge on saved logs, in process and over UDP (bench) | `p3_shadow_replay.py` | no | the pinned environment |
| Bridge against simulator, decision by decision (bench) | `p3_parity_replay.py` | no | the pinned environment |
| Run time per decision (bench) | `p3_timing.py` | no | the pinned environment |
| Offline viewer of a raw log | `log_viewer.py`, `rl_log_viewer.py` | no | pygame |

Only one program can receive at a time: the logger, the manoeuvre sender and the bridge all listen
on UDP 5000. Stop one (Ctrl+C) before starting the next.

## 2. The link to the vessel

- The shore computer binds `0.0.0.0:5000` and sends the handshake `START` to the vessel-control
  laptop at `<server-ip>:5050` (10.201.205.110 in July; noted again before every session).
- The vessel answers with `hey` lines, then streams telemetry to the address the `START` came
  from: LiDAR lines (`[HH:MM:SS.ffffff][r0, r1, ..., r719]`, 720 integers), pose lines
  (`[HH:MM:SS.ffffff]A,B,C`) and, depending on the vessel program, `HDG` and `RC` lines and, from
  2026-11-26, IMU lines in a format still to be seen (D0).
- Commands go back as `$CMD,<vessel rudder>,<S2>`, two decimals each, for example
  `$CMD,0.00,25.00`.
  - Vessel rudder = `--rudder-sign` x rudder, with `--rudder-sign` -1 by default: the vessel's
    rudder sign is opposite to the simulator's.
  - S2 = RPM / 24 x 100. RPM 6 (cruise) is S2 25, RPM 12 is S2 50, and S2 = -100 is full astern.
- Pose frame used by the decoder (`log_parser.BluefinStreamDecoder`) and by `udp_manoeuvre.py`:
  for a pose line `[ts]A,B,C`, x = B, y = -A and heading = -C. The `--fence` box is written in
  this frame.
- Network: the July set-up (P-11). The Ubuntu laptop's firewall (ufw) is disabled and stays so;
  UDP 5000, 5001 and 5050 are free on it (checked 2026-11-25).

## 3. `udp_raw_logger.py`: raw telemetry logger

Sends `START` (repeated every `--hello-every-s` until data arrives) and writes every datagram,
unchanged, to a file. Nothing is decoded on the way to the file and no command is ever sent, so a
new line type (an IMU line, a longer pose line, binary data) cannot stop the recording. Standard
library only; the file can be copied on its own to any computer.

### 3.1 Self-test (loopback, no vessel)

```bash
python udp_raw_logger.py --selftest
```

Output on 2026-11-25:

```
[SELFTEST] formats equal to log_parser's
[SELFTEST] files in /tmp/udp_raw_logger_selftest_kowztunz
[SELFTEST] PASSED: handshake, 14 datagrams logged byte for byte (LiDAR, pose, IMU variants, binary, two lines in one datagram), line kinds as expected
```

### 3.2 Live recording (on the day)

```bash
python udp_raw_logger.py --server-ip 10.201.205.110 --out 2026-11-26/D0-1.log
```

- Stop with Ctrl+C; every datagram is on disk as soon as it arrives.
- `--duration-s N` stops by itself after N seconds.
- `--no-summary` leaves only the recording and the status line (no line-kind table), for a very
  slow computer or a very fast stream.
- An existing file name is never overwritten: `_2`, `_3`, ... is added.

Files written:

| File | Holds |
| --- | --- |
| `<out>` | every datagram as received, one per line (the July log format, without the bridge's `#CONFIG` line) |
| `<out>.arrivals.csv` | arrival number, wall-clock time, time since start, size of each datagram |
| `<out>.meta.json` | the options, start and end times, whether the vessel answered `hey`, the count and rate of each line kind |

Console, from a rehearsal against the mock vessel (section 6) streaming the July log
`2026-07-03/1.log`:

```
[RAW] recording to .../D0-mock.log
[RAW] listening on 0.0.0.0:5000; handshake 'START' to 127.0.0.1:5050; no command is ever sent; Ctrl+C stops
[RAW] first datagram after 0.0 s
[RAW] new line kind: pose (3 values)  (decoder: read)
      [09:25:31.714600]-000.0,+000.1,-007.6
[RAW] new line kind: LiDAR scan (720)  (decoder: read)
      [09:25:31.712034][  0,  0,  0,  0, ...
[RAW]   2.0 s  datagrams 9 (4.0/s)  14.6 kB  last 0.0 s ago
...
  line kind                             count   rate /s  decoder
  pose (3 values)                          20       2.0  read
  LiDAR scan (720)                         21       2.0  read
...
[RAW] stopped (--duration-s): 61 datagrams, 88.1 kB in .../D0-mock.log
```

- The status line every 2 s gives datagrams, rate, data volume and the time since the last
  datagram. A growing "last ... ago" means the stream has stopped.
- The first line of every new kind is printed in full, so a new IMU line shows its format at once.
- The table gives each kind's count and rate (from the vessel's own time stamps) and what the
  bridges' decoder would do with it:
  - `read`: a LiDAR scan (720 integers), pose, HDG or RC line, as in July;
  - `ignored`: the decoder skips it (a `hey` line, a labelled IMU line, ...);
  - `taken as a scan`: a bracketed list of integers that is not 720 long; the decoder would read it
    as a LiDAR scan;
  - `decoder error`: bracketed values that are not integers. The July bridge
    (`udp_live_rl.py`) stops on these; `udp_live_p3.py` logs and skips them.
- The July log holds no `hey` lines, so the rehearsal shows none (`vessel_answered_hey: false` in
  the meta file). From the vessel, `hey` lines are expected within seconds of `START`.

### 3.3 Check a saved log (no network)

```bash
python udp_raw_logger.py --check 2026-11-26/D0-1.log
```

Output for the rehearsal log:

```
.../D0-mock.log: 61 lines
  line kind                             count   rate /s  decoder
  pose (3 values)                          31       2.0  read
  LiDAR scan (720)                         30       2.0  read
  pose (3 values): [09:25:31.714600]-000.0,+000.1,-007.6
  LiDAR scan (720): [09:25:31.712034][  0,  0,  0,  0, ...
```

The rates come from the vessel's time stamps, so `--check` answers the rate question for every
line kind, the IMU's included, directly from the file.

## 4. `udp_manoeuvre.py`: scripted calibration manoeuvres

Exact rudder and propulsion commands sent over UDP at exact times, with the raw telemetry logged on
the same clock. Same link as the bridges (section 2). Each manoeuvre starts when Enter is pressed,
after the telemetry is flowing. Standard library only.

### 4.1 Self-test (loopback, no vessel)

```bash
python udp_manoeuvre.py --selftest
```

Output on 2026-11-25:

```
[SELFTEST] files in /tmp/udp_manoeuvre_selftest_co5y389r
[SELFTEST] PASSED: steps (sign and format), zig-zag switching on the heading, stop with astern and its refusal without --allow-astern, telemetry watchdog, basin fence, sequence file
```

### 4.2 Common options (before the subcommand)

| Option | Default | Meaning |
| --- | --- | --- |
| `--server-ip` | 127.0.0.1 | the vessel-control laptop |
| `--server-port`, `--local-port`, `--bind-ip` | 5050, 5000, 0.0.0.0 | the July ports |
| `--out` | `manoeuvre_logs/<manoeuvre>_<date>_<time>.log` | the log file; name it by the run identifier |
| `--rudder-sign` | -1 | vessel rudder = sign x rudder, as the bridges |
| `--rpm-max` | 24 | S2 = rpm / rpm-max x 100 |
| `--cmd-rate` | 2 Hz | the current command is repeated this often (changes go out at once) |
| `--fence=XMIN,XMAX,YMIN,YMAX` | none | the basin box in the pose frame (section 2). Written with `=`, because a negative first value would otherwise read as an option |
| `--fence-margin-m` | 1.5 m | the manoeuvre stops when the pose comes this close to an edge (the pose is the hull reference point; the bow is 0.86 m ahead) |
| `--watchdog-s` | 1.5 s | abort when telemetry (or, for turn, zigzag and stop, pose lines) stops this long; 0 turns it off |
| `--max-s` | 180 s | abort a manoeuvre running longer than this |
| `--speed-window-s` | 1.0 s | speed from pose lines this far apart |
| `--tail-s` | 3 s | telemetry kept after the final zero command |
| `--allow-astern` | off | permit negative S2 (full astern). Only after the ESC reversal bench test (P-8) |
| `--dry-run` | off | log every command but send none |

Rudder is in percent of full rudder in the simulator's convention: positive turns to starboard
(heading increasing). RPM is the simulator's unit (0-24, cruise 6).

### 4.3 Subcommands, one example each

Replace `<IP>` with the vessel-control laptop's address and `<FENCE>` with the box from D0-5.

| Subcommand | What it does | Example |
| --- | --- | --- |
| `hold` | a constant command for a time | `python udp_manoeuvre.py --server-ip <IP> --out 2026-11-26/S1a-CHK-01.log --fence=<FENCE> hold --rpm 6 --duration-s 10` |
| `coast` | run-up at an RPM (`--runup-s`), then zero propulsion for `--coast-s` (default 20 s) | `... coast --rpm 12 --runup-s 10` |
| `steps` | rudder steps 0 -> +a -> 0 -> -a -> 0 per amplitude, `--repeats` (5), `--hold-s` (4 s); `--reversal` adds +a -> -a -> +a -> 0 | `... steps --rpm 0 --rudder 25 50 100 --repeats 5 --reversal` |
| `ramp` | rudder ramp at `--rate` %/s to `--to`, to minus `--to`, back to 0 | `... ramp --rpm 0 --rate 10 --to 100` |
| `turn` | run-up, then a fixed rudder until `--heading-change` (deg) or `--duration-s`; then `--after-s` straight. Reports the side the vessel turned to | `... turn --rpm 12 --rudder 100 --heading-change 720` |
| `zigzag` | run-up, then a delta/psi zig-zag switching on the measured heading. `--delta` in degrees of the 40 deg maximum; negative starts to port | `... zigzag --rpm 12 --delta 20 --psi 20 --switches 2` |
| `stop` | run-up (optionally into a turn with `--rudder`, `--turn-s`), then full astern, rudder centred, until the measured speed reads below `--stop-speed` (0.08 m/s), 1-8 s; needs `--allow-astern` | `... --allow-astern stop --rpm 12` |
| `sequence` | a fixed command sequence from a CSV file, zero-order hold | `... sequence --file seq.csv` |

Sequence-file format: a header line, then one row per change, `t_s,rudder_pct,rpm`. Example (the
self-test's file):

```
t_s,rudder_pct,rpm
0,0,6
0.5,30,6
1.0,-30,6
1.5,0,0
```

Negative RPM in a sequence needs `--allow-astern`.

The `turn`, `zigzag` and `stop` manoeuvres need pose lines (the heading or the speed). If no pose
line is recognised in 2 s after the stream starts, they refuse to start and suggest
`udp_raw_logger.py --check`, or `hold`, `steps` or `sequence`.

### 4.4 Files written

| File | Holds |
| --- | --- |
| `<out>` | every datagram as received, with `#CMD` lines (every command sent) and `#MAN` lines (phases and events) between them, on one clock |
| `<out>.arrivals.csv` | arrival number, wall-clock time, time since start, size of each datagram |
| `<out>.commands.csv` | every command: times, phase, reason, rudder, rpm, S2, the text sent, and whether it was sent (`sent=0` in a dry run) |
| `<out>.meta.json` | options, manoeuvre, events (switch times, overshoots, stop), how it ended |

### 4.5 What a run looks like (dry run against the mock vessel)

```bash
python udp_manoeuvre.py --server-ip 127.0.0.1 --out .../S1a-mock-hold.log --dry-run hold --rpm 6 --duration-s 10
```

```
[MAN] hold rudder +0 %, rpm 6.0 for 10 s; vessel 127.0.0.1:5050; DRY RUN: no command is sent
[MAN] telemetry flowing
[MAN] pose lines 4, last heading 9.6. Press Enter to start, Ctrl+C to quit.
[MAN] manoeuvre start manoeuvre=hold
[CMD]   0.00 s  hold       $CMD,0.00,25.00  (start)
[MAN]    2.0 s  hold       heading change    -3.7 deg  speed 1.03 m/s  telemetry age 0.50 s
...
[CMD]  10.01 s  end        $CMD,0.00,0.00  (manoeuvre complete)
[CMD]  10.11 s  end        $CMD,0.00,0.00  (manoeuvre complete)
[CMD]  10.21 s  end        $CMD,0.00,0.00  (manoeuvre complete)
[MAN] ended (complete): 70 datagrams, 23 commands; .../S1a-mock-hold.log
```

The heading and speed in a dry run against a replayed log are the log's, not a response to the
commands.

### 4.6 Stops: Ctrl+C, watchdog, time limit, fence

- **Ctrl+C** at any time: zero rudder and zero propulsion, sent three times, then the program
  ends. The log records `abort: Ctrl+C`.
- **Watchdog** (`--watchdog-s`, 1.5 s): the same three zero commands. From a dry run in which the
  mock vessel was stopped mid-manoeuvre:

  ```
  [MAN]    4.0 s  hold       heading change   -18.7 deg  speed 0.90 m/s  telemetry age 1.50 s
  [CMD]   4.02 s  end        $CMD,0.00,0.00  (abort: telemetry lost for 1.5 s)
  [CMD]   4.12 s  end        $CMD,0.00,0.00  (abort: telemetry lost for 1.5 s)
  [CMD]   4.22 s  end        $CMD,0.00,0.00  (abort: telemetry lost for 1.5 s)
  [MAN] ended (abort: telemetry lost for 1.5 s): 28 datagrams, 11 commands; ...
  ```

  If telemetry is lost, the program can no longer see the vessel. Zero commands are sent, but the
  radio-control override stays the physical stop.
- **Time limit** (`--max-s`, 180 s): the same zero commands.
- **Fence**: when the pose comes within `--fence-margin-m` of the box.
  - With `--allow-astern`: full astern (S2 = -100), rudder centred, until the speed reads stopped
    (at most 8 s), then zero.
  - Without it (the case until P-8 passes): propulsion is cut at once and the vessel coasts (about
    13 m from RPM 12 in the model). The program prints the line below, and the safety person takes
    over by radio control.

    ```
    [MAN] *** FENCE: propulsion cut; the vessel coasts. TAKE OVER BY RADIO CONTROL ***
    ```

  - Before the start, the program prints whether the current pose is inside the fence less its
    margin. From a dry run with a deliberately offset fence (`--fence=2,12,0,25`):

    ```
    [MAN] pose (0.2, 0.1) is OUTSIDE the fence less its 1.5 m margin
    ...
    [MAN] *** FENCE: propulsion cut; the vessel coasts. TAKE OVER BY RADIO CONTROL ***
    [CMD]   0.00 s  end        $CMD,0.00,0.00  (abort: fence: pose (0.5, 1.1) within 1.5 m of the basin edge)
    ```

    An `OUTSIDE` line before Enter means the vessel or the box is wrong: do not press Enter.

### 4.7 The rudder-sign check (first use on the vessel)

`turn` reports the side the vessel turned to as soon as the heading has moved:

```
[CMD]   0.01 s  turn       $CMD,-50.00,25.00  (helm +50)
[MAN] turning side=port (heading decreasing), as_expected=False, note=positive rudder should turn to starboard
```

That line is from a dry run against a replayed log, where the heading does not follow the command,
so its verdict means nothing there. On the vessel, `turn --rudder 50` must report
`side=starboard (heading increasing), as_expected=True`. Otherwise, rerun with `--rudder-sign 1`
and check again.

## 5. `udp_live_p3.py`: the Paper 3 bridge

The deployed SAC policy (seed 0, baseline-v3) on the Bluefin. Each LiDAR frame (2 Hz) goes
through the following steps:
1. The frame is paired with its own pose line and mapped into the simulator's frame.
2. The simulator's perception builds the 70-value observation.
3. The policy acts, and the command is resolved as `ASVLidarEnv.step` resolves it.
4. `$CMD` goes out.
5. The decision goes to the safety layer (E1) in a shadow worker process.

It needs the pinned Python environment and the policy file, whose hash it checks at start-up. A
wrong model or a changed constant stops it at start-up.

### 5.1 Live run (on the day: S2-C, trials)

```bash
python udp_live_p3.py --server-ip 10.201.205.110 --scenario easy_1 --run-id T1-easy_1-r1
```

- Geometry, one of:
  - `--scenario <name>`: a field-suite case (`easy_1` ... `hard_5`) or a test set v4 ID;
  - `--case <test set v4 ID>`;
  - `--layout L1|L2|L3`;
  - `--start x,y --goal x,y` (a basin leg);
  - `--test-case N` (the July `test_run.py` cases, for replaying July logs).

  A field case prints its set-up at start, for example:

  ```
  [P3] field case easy_1: head-on, OPEN; own ship passes (5.0, 2.0) at cruise, heading 0.0 deg, at t = 0; target from (4.69, 16.01), heading 181.1 deg, 0.57 m/s
  ```

- `--run-id`: the run identifier from the run sheet (default: a time stamp).
- Map registration:
  - `--origin start` (default, the July rule): the first pose is placed at the path start;
  - `--origin map`: the telemetry is in the map frame up to `--map-rotation` (deg, clockwise) and
    `--map-offset dx,dy` (m, after the rotation);
  - `--pose-point origin|sensor`: whether the telemetry pose is the vessel reference point (the
    July assumption) or the LiDAR, 0.86 m ahead. Settled at S1-A (O-9).
- Other options:
  - `--ego-window-s` (default 0.5 s, the decision interval): surge, sway and yaw rate from pose
    lines that far apart;
  - `--min-decision-interval` (0.4 s);
  - `--lidar-raw-beams` (720), `--lidar-index0-deg` (0);
  - `--rudder-sign` (-1);
  - `--model-path` (default: the deployed policy).

### 5.2 No-send mode: `--no-send` (alias `--shadow`, the July name)

```bash
python udp_live_p3.py --server-ip 10.201.205.110 --layout L1 --run-id link-2026-11-26 --no-send
```

The policy decides and every decision is logged, but no `$CMD` is sent. This is the first check
of the bridge on the vessel (UDP_BENCH_HANDOVER.md step 7) and the mode for every bench run.
Abbreviations are refused (`allow_abbrev=False`): in July, `--shadow` prefix-matched another option
and would have sent commands.

### 5.3 Safety layer: `--safety shadow|active|off`

- `shadow` (default, the field setting): E1 runs in a worker process beside the policy and records
  what it would have done (would-change, would-stop) without ever touching a command.
- `active` is **not used**. The safety layer is not ready for the command path (decision of
  2026-11-25): E1 in the loop misses the 0.5 s decision interval on all three computers (Ubuntu
  p99 0.70 s, 3.9 % of decisions over 0.5 s, B-8), and the ESC reversal (P-8) is untested. It is
  refined after the trials.
- `off`: no layer at all.
- `--shadow-inline` runs the shadow layer in the main process after each command (bench);
  `--full-filter-trace` keeps its bulky diagnostics.

### 5.4 Stops

- **E key**: the operator stop. It latches full astern (S2 = -100) until the measured surge reads
  stopped (`--estop-stop-speed`, 0.05 m/s; at most `--estop-max-brake-s`, 8 s), then zero
  propulsion.
- **R key**: hands control back after the minimum hold (`--estop-min-hold-s`, 2 s).
- **Esc**: quits.
- The keys act only while the bridge's pygame window has the focus.
- The radio-control override is the physical stop and works whatever the shore computer does.
- `NO TELEMETRY FOR > 1.5 s` appears on the display when the stream stops: an abort condition
  (FIELD_SESSIONS_PLAN.md section 2.5).

### 5.5 Bench-only options

- `--headless`: no window, and therefore no E key. Never on the water.
- `--max-frames N`: stop after N decisions.
- `--idle-exit-s S`: stop when no line has arrived for S seconds after the first one.

### 5.6 Outputs: `p3_runs/<run-id>/`

| File | Holds |
| --- | --- |
| `telemetry.log` | every received line, as received (unknown lines included) |
| `decisions.jsonl` | per decision: sensor input, observation, action, command, tracks and encounter contexts, timings |
| `shadow.jsonl` | every shadow-layer decision |
| `summary.json` | run identifier, geometry, safety mode, lines, decisions, skipped frames, errors, lines not decoded |

`--out-dir` changes the parent folder (default `field_deployment/p3_runs/`).

### 5.7 Unknown lines (IMU)

A bracketed line that is not 720 integers (an IMU record, say) is logged in `telemetry.log` and
skipped, not decoded; a labelled line (`IMU:...`) counts as `other`. The run carries on. Checked
on 2026-11-25 with the July log with IMU-style lines added, streamed by the mock vessel:

```
[P3] line not decoded (bracketed values that are not integers), kept in telemetry.log; first one: [09:25:32.214989][0.012,-0.034,9.810,0.10,-0.20,0.30]
[P3] 20 decisions logged to .../p3_runs/guide-imu-easy1; ...
```

The run's `summary.json` gave `decisions 20, errors 0, lines_not_decoded 8`, with line kinds
`{'pose': 20, 'lidar': 20, 'not decoded': 8, 'other': 8}`. The IMU is decoded only after D0 has
shown its format (bench item B-3).

## 6. Mock vessel: testing the UDP connection without the vessel

`fake_vessel_replay.py` plays the vessel-control laptop on this computer: it listens on a port,
waits for `START`, then streams a recorded log to the sender in real time (by the log's time
stamps). It reads no commands, so the receiving program runs with `--no-send` or `--dry-run`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--log` | required | the log to stream |
| `--port` | 5050 | the port it listens on (the vessel's) |
| `--speed` | 1.0 | replay speed (2.0 = twice real time) |
| `--loop` | off | start the log again at its end |
| `--ignore-rc` | off | leave out the `RC` lines |

### 6.1 Mock vessel to the bridge (two terminals)

Terminal 1:

```bash
python fake_vessel_replay.py --log 2026-07-03/1.log --port 5050
```

```
[FAKE VESSEL] Listening on {}:{} 0.0.0.0 5050
[FAKE VESSEL] Waiting for START...
[FAKE VESSEL] Got START from {}, streaming will start ('127.0.0.1', 5000)
```

(The braces are printed as shown; the script's messages are not formatted.)

Terminal 2:

```bash
python udp_live_p3.py --test-case 1 --run-id guide-mock-0703-1 --no-send --headless --idle-exit-s 5
```

```
[P3] run guide-mock-0703-1: test_run case 1, safety=shadow, send=no
[UDP] START sent to 127.0.0.1:5050; listening on 0.0.0.0:5000
Max velocity: 0.283 m/s
...
[P3] 49 decisions logged to .../p3_runs/guide-mock-0703-1; shadow: {'type': 'done', 'decisions': 49, ...
```

`summary.json`: 98 lines, 49 decisions, 0 errors, 0 lines not decoded. Leave out `--headless` to
see the display. The `Max velocity` lines come from the July decoder over 0.1 s, not from the
bridge's ego estimate.

### 6.2 Mock vessel to the logger or the manoeuvre sender

The same terminal 1, then in terminal 2:

```bash
python udp_raw_logger.py --server-ip 127.0.0.1 --out mock_test.log --duration-s 15
```

```bash
python udp_manoeuvre.py --server-ip 127.0.0.1 --out mock_hold.log --dry-run hold --rpm 6 --duration-s 10
```

Section 3.2 and section 4.5 show their output. Restart terminal 1 for each run: the mock vessel
streams the log once and exits (unless `--loop`).

## 7. Replaying and viewing saved logs

### 7.1 `p3_shadow_replay.py`: the bridge on saved logs

- **Part A, in process, every log:** the log's lines go through the bridge's own per-frame path
  and the shadow layer's code path. Checked: every frame decided, every observation finite and
  inside the observation space, one shadow decision per decision, no errors.
- **Part B, over UDP:** the mock vessel streams selected logs in real time to `udp_live_p3.py`
  (headless, `--no-send`, the shadow layer in its worker). Checked against part A: the same
  decisions; and the decision log alone reproduces every shadow decision offline. The run-time
  figures (P-10) come from part B.

```bash
python p3_shadow_replay.py --tag udp_ubuntu_i5 --port 5050 --local-port 5000
python p3_shadow_replay.py --skip-inproc --tag udp_ubuntu_i5_table --port 5050 --local-port 5000
python p3_shadow_replay.py --logs "data/test_*.log" --udp-logs data/test_3.log --test-case 0
```

- `--logs` (glob) and `--udp-logs` choose the logs for parts A and B. On Windows PowerShell write
  `--udp-logs=` for an empty value.
- `--test-case N` gives the geometry for raw logs without a `#CONFIG` line (the February logs in
  `data/`).
- `--port` and `--local-port` set the mock vessel's port and the bridge's port.
- Output: `p3_bench/shadow_replay/<tag>/` (`summary.json`, `timing.csv`, each run's own logs).

Results on this laptop (2026-11-25):
- part A, 19/19 July logs, 1,085 decisions, every per-log figure equal to the personal laptop's;
- part B, logs 2026-07-03/1-3, 49/43/41 decisions, reproduced offline;
- LiDAR line to command sent p99 50.6 ms, with the stopping table.

### 7.2 Replaying a saved vessel log through the bridge by hand

Any raw log (the logger's, the manoeuvre sender's, or a bridge's `telemetry.log`) can be streamed
by the mock vessel (section 6.1) into `udp_live_p3.py --no-send`. That shows what the policy would
have done on that stream, with its own `p3_runs/<run-id>/` record.

### 7.3 Viewers: `log_viewer.py`, `rl_log_viewer.py`

Offline pygame viewers of a raw log: decoded pose, derived velocity, LiDAR statistics, and the
trajectory on a map panel.

```bash
python log_viewer.py 2026-07-03/1.log --rate 2
python rl_log_viewer.py 2026-07-03/1.log --test-case 1
```

- `rl_log_viewer.py` also draws the reference path and obstacles of a `test_run.py` case
  (`--test-case`).
- Both take `--rate` (playback speed), `--zoom`, `--full`, `--no-map`, `--record` (MP4), `--plot`
  (trajectory PNG) and `--metrics` (JSON).
- Keys:
  - Space pauses; R restarts;
  - F shows the full LiDAR list; M toggles the map;
  - O re-zeros the origin; C clears the path; G toggles follow mode; W/A/S/D pan;
  - P takes a snapshot; Esc quits.
- Not run for this guide (they open a window).

## 8. Bench tools

| Tool | What it checks | Result on this laptop (2026-11-25, with the stopping table) |
| --- | --- | --- |
| `python p3_parity_replay.py --tag <tag>` | 16 test set v4 episodes through the simulator and through the bridge, shadow and active, decision by decision; also against the recorded E1 traces (float diagnostics within 1e-9) | 32/32 runs, 1,497 decisions, traces 16/16, identical to the UTAS laptop's run |
| `python p3_timing.py --tag <tag>` | run time per decision on 61 encounter-rich episodes, alone on the computer | command path p99 18.4 ms (required below 0.5 s: met); E1 in the loop p99 696 ms, 3.9 % over 0.5 s (not met; active mode not used) |

Outputs: `p3_bench/parity/<tag>/` and `p3_bench/timing/<tag>/`. Read the figures from
`summary.json` and `timing.csv`; the console rounds to one decimal.

The bridge loads the stopping-profile table `p3_stopping_profiles.pkl` (602 entries) if its
fingerprint matches the sources. Otherwise it prints:

```
[P3] WARNING: stopping-profile table not used (the table predates a change to its sources or constants); each new surge value costs ~0.5 s once.
```

and runs slower but with the same results. `python p3_bridge.py --build-stopping-table` rebuilds
the table after a deliberate change to the vessel model; it is not run in the field.

## 9. Older and helper scripts

- `udp_live_rl.py`: the July 2026 bridge, unchanged. It runs the Paper 2 policy
  (`best_model.zip`, the Paper 2 observation). Its decoder raises on bracketed lines that are not
  integers, so it stops on IMU-style lines. Not used for Paper 3.
- `udp_listener.py`: the early pygame telemetry viewer and recorder (`--server-ip`,
  `--record-log`, `--print-raw`), without a policy. `udp_raw_logger.py` replaces it for recording.
- `udp_send_receive_template.py`: a minimal example of the handshake and the receive and send loop
  (hard-coded addresses). Reference only.
- `test_run.py`: the July bridge's test cases (`TestCase`, `--test-case N`): the path and obstacle
  geometry used when replaying the July logs.
- `log_parser.py`: the July decoder (`BluefinStreamDecoder`), shared by every bridge.
- `policy_render.py`, `lidar_pooling.py`, `asv_lidar.py`, `ship_model.py`, `rl_env.py`: support
  modules of the July bridge and viewers.

## 10. Same as July, and what changed

Checked in the code on the UTAS laptop (2026-11-25).

**Same as July:**
- bind `0.0.0.0:5000`, `START` sent to `<server-ip>:5050`, a `recvfrom` loop;
- the same decoder (`log_parser.BluefinStreamDecoder`) and the same telemetry lines;
- the same command line, `$CMD,<rudder>,<S2>` with two decimals;
- vessel rudder = -1 x simulator rudder (`--rudder-sign`);
- S2 = RPM / 24 x 100 (July's `rpm_to_s2_cmd`, now `emergency_stop.rpm_to_s2`);
- pose synchronisation and the emergency-stop options;
- the July option names (`--server-ip`, `--local-port`, `--server-port`, `--shadow`);
- a pygame display with the E key.

**Changed in the Paper 3 bridge (`udp_live_p3.py`):**
- the observation: 70 values from the onboard perception pipeline and target tracker, in place of
  the Paper 2 observation;
- the policy: SAC seed 0 of Paper 3, in place of the Paper 2 `best_model.zip`;
- propulsion: 0-12 RPM with cruise at RPM 6, against July's cruise at RPM 12;
- the command is resolved as `ASVLidarEnv.step` resolves it;
- the safety layer runs as a shadow worker;
- the scenario is chosen by name (`--scenario`);
- map registration options (`--origin`, `--map-rotation`, `--map-offset`, `--pose-point`);
- one output folder per run (`p3_runs/<run-id>/`) in place of one `--record-log` file;
- unknown lines are logged and skipped;
- the rendering.

**New calibration tools:** `udp_raw_logger.py` and `udp_manoeuvre.py` are standalone programs
with no policy, standard library only. They use the same handshake, ports and `$CMD` format.

## 11. Testing the UDP connection, in order

1. **Self-tests on loopback** (any time, no vessel): `python udp_raw_logger.py --selftest` and
   `python udp_manoeuvre.py --selftest`. Both passed on this laptop on 2026-11-25.
2. **Mock vessel to the bridge over loopback** (no vessel): section 6.1. Passed on 2026-11-25 (49
   decisions, 0 errors); the bench version is `p3_shadow_replay.py` part B (section 7.1).
3. **On the day, the raw logger's handshake with the vessel-control laptop:**
   `python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/D0-1.log`.
   - Pass: `[RAW] first datagram after` a few seconds, `hey` lines, LiDAR and pose lines at a
     steady rate.
   - Nothing arriving means a wrong IP, a different network, or the vessel program not running.
     The logger repeats `START` every 2 s until data arrives.
4. **Later (calibration session 2, trials), the bridge against the vessel with `--no-send`:**
   UDP_BENCH_HANDOVER.md step 7.
   `python udp_live_p3.py --server-ip <IP> --layout L1 --run-id link-<date> --no-send`.
   Pass: frames and pose lines arrive and pair at the vessel's rate; no `$CMD` sent.

## 12. Quick reference

```bash
python udp_raw_logger.py --selftest
python udp_raw_logger.py --server-ip <IP> --out 2026-11-26/D0-1.log
python udp_raw_logger.py --check 2026-11-26/D0-1.log
python udp_manoeuvre.py --selftest
python udp_manoeuvre.py --server-ip <IP> --out 2026-11-26/S1a-CHK-01.log --fence=<FENCE> hold --rpm 6 --duration-s 10
python udp_manoeuvre.py --server-ip <IP> --out 2026-11-26/S1a-CHK-02.log --fence=<FENCE> turn --rpm 6 --rudder 50 --duration-s 5
python fake_vessel_replay.py --log 2026-07-03/1.log --port 5050
python udp_live_p3.py --test-case 1 --run-id bench-0703-1 --no-send --headless --idle-exit-s 5
python udp_live_p3.py --server-ip <IP> --layout L1 --run-id link-<date> --no-send
```
