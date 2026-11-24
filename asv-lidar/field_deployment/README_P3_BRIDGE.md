# Paper 3 field bridge

The live UDP bridge that runs Paper 3's deployed policy on the Bluefin vessel, with the E1
safety layer in shadow mode. Design: `static_dynamic_obstacles/planning/FIELD_TRIALS_PLAN.md`
sections 3 and 5; bench items P-9 and P-10 of `planning/FIELD_CALIBRATION_PLAN.md`.
`udp_live_rl.py` (the July 2026 bridge, Paper 2 observation) is unchanged and still reproduces
the July runs.

| File | Role |
| --- | --- |
| `p3_bridge.py` | Core, no sockets or display: `LiveEnv` (observation adapter), `Policy`, `resolve_command`, `SafetyLayer` (E1, shadow or active), `ShadowRunner` and `shadow_worker`, logging helpers, the stopping-profile table |
| `udp_live_p3.py` | The live bridge: UDP, pose sync, operator stop, display, logs |
| `p3_parity_replay.py` | Bench parity test on simulated test set v4 episodes (P-9) |
| `p3_shadow_replay.py` | Bench shadow session on the July logs, in process and over UDP with `fake_vessel_replay.py` |
| `p3_timing.py` | Run time per decision on this computer (P-10, T-14) |
| `p3_stopping_profiles.pkl` | Precomputed stopping profiles (built by `python p3_bridge.py --build-stopping-table`) |
| `udp_raw_logger.py` | Handshake and raw log only, no decoder, no command; standard library only (data-collection sessions, a new line type such as the IMU's) |
| `p3_bench/` | Bench outputs |

## What the bridge runs

| Item | Setting |
| --- | --- |
| Policy | `static_dynamic_obstacles/runs/sac_formulation_seed0_bl3/best_model.zip`, refused unless its SHA-256 is `993db156...c2bdc8` and its `config.json` agrees with the code (schema `a25-v3-context`, 70 values, propulsion stage 4, recorded constants) |
| Decision rate | one decision per LiDAR frame, 2 Hz |
| Rudder | `100 * clip(a0)`, no limiter (`RUDDER_COMMAND_LIMIT = False`; the bridge refuses to run with it on); vessel sign reversed (`--rudder-sign -1`) |
| Propulsion | `CRUISE_RPM + RPM_DELTA * a1` clipped to 0-12 rpm-units, `S2 = rpm / 24 * 100` (0-50); negative S2 only from a stop |
| Safety layer | `safety_v21.SafetyFilterV21EC(estop_window_s=2.5, use_static_points=False, braking_fallback=False)`, shadow mode by default |
| Physical stop | the operator stop (E key), always armed |

### Observation adapter

`p3_bridge.LiveEnv` is `ASVLidarEnv` itself (a subclass, `env.py` unchanged) with its sensor
models replaced by the supplied frame: the raycast by the measured scan (passed through the
simulator's 1 m dead zone and aft mask, dropout off), the pose estimate and the ego measurement
by the measured ones, the staleness draw by the pose-sync flag. The environment's own
`_perceive` and `_get_obs` then run unchanged: the gate against the map polygon, the 27 pooled
sectors, the 7 boundary rays at the estimated pose, clustering, the tracker, encounter
classification and latch, and the 70-value observation. No physics and no targets are
simulated; the attributes the environment would hold as truth hold the estimate.

Conventions at the field boundary (`frame_to_sensor`):

* the decoder's 720 beams run clockwise from `--lidar-index0-deg` (0 in the July logs), the
  simulator's from dead ahead; the scan is rolled accordingly;
* the decoder's sway is positive to port, the simulator's to starboard: its sign is reversed;
  surge and yaw rate (compass sense, deg/s) pass through;
* `--origin start` (default) anchors the first pose at the path start, as the July bridge did,
  with the telemetry axes taken as the basin's; `--origin map` takes telemetry in the map frame
  up to `--map-rotation` (deg, clockwise) and `--map-offset dx,dy`, for a pose registered on
  the landmark survey (P-7);
* `--pose-point origin` (default, the July assumption) takes the telemetry pose as the
  simulator's vessel reference point; `--pose-point sensor` takes it as the LiDAR's, 0.8625 m
  ahead on the centreline, and moves it back (and removes the lever arm from the sway);
* a frame released without its own pose line is stale: the environment holds the previous pose
  and ego estimate and does not feed the tracker, as its staleness model does; a lost frame
  makes the tracker predict over the whole gap;
* every released frame is processed in order (a backlog is caught up, never dropped), and a
  frame closer than `--min-decision-interval` (0.4 s) to the last decision is logged but not
  acted on, so a faster telemetry stream (P-1) is decimated to the 2 Hz the policy and the
  tracker assume;
* the ego motion (surge, sway, yaw rate) is the difference between the newest pose line and
  the one whose age is nearest `--ego-window-s` (0.5 s, the decision interval), with the
  decoder's conventions and arithmetic (`EgoWindow` in `udp_live_p3.py`, 2026-11-20), not the
  decoder's difference between consecutive lines. At 2 Hz that is the previous line: on the
  19 July logs every frame's ego motion equals the decoder's, and the shadow replay (in process,
  all 19 logs; over UDP loopback, three) gives decisions and shadow decisions equal to the
  2026-11-05 run. At the vessel's 10 Hz stream (the February 2026 logs, `data/test_1-4.log`)
  consecutive lines are 0.1 s apart and the 0.1 m pose is unchanged on 26-62 % of them: the
  decoder's speed read zero on 18-46 % of the frames moving faster than 0.2 m/s, which the
  operator stop takes for "stopped", and peaked at 2.2-2.3 m/s against at most 1.8 m/s over
  1 s. Over 0.5 s the speed's error against the speed over 1 s is 0.08-0.10 m/s RMS (decoder
  0.39-0.44), no moving frame reads zero, and the in-process replay of the four logs passes
  (1,179 decisions). The `HDG` lines (0.01 deg, 10 Hz) are logged but not used: their rate
  follows the pose heading's (slope 0.93-1.01), while their offset from it drifts within a run
  (standard deviation 4-102 deg), so they come from another heading source, not yet identified.

### Safety layer in shadow mode (D-4)

The layer reads the bridge's environment as its environment interface (estimated pose,
measured ego motion, tracks and encounter contexts, `boundary_polygon`, the gated scan) and is
told the issued command through `observe_issued_command`, in the order of `env.step`. In
shadow mode nothing it decides reaches the vessel: its brake request is cleared, and its model
of its own rudder (`actuators`) is advanced with the command actually issued (the policy's)
whenever its proposal differs. Its plan memory and its stop commitment remain its own, so the
first would-be stop is permanent, as an active stop would be; later decisions then report
`emergency stop`.

By default the layer runs in a worker process (`p3_bridge.shadow_worker`) with its own
`LiveEnv` fed the same frames, so its run time never delays a command; each worker decision
checks that its observation equals the bridge's. `--shadow-inline` runs it in the bridge
process after the command has gone.

`--safety active` puts E1 between the policy and the command, with its brake as full astern
(S2 = -100) above the stop speed and zero below it. It is for the optional active block only,
after the ESC is shown to reverse (P-8, O-6) and the layer meets T-14 (see Timing).

### Stopping-profile table

The encounter context's slowdown test integrates a stop on the identified hull
(`stopping._profile`), cached per cm/s of measured surge, at about 0.55 s per new value. A
simulation warms that cache over many steps; a bridge starts cold, and surge measured from
0.1 m poses jumps between values. On the July logs this held 32 of 49 decisions over 0.5 s.
`p3_stopping_profiles.pkl` holds the function's own results for 0-300 cm/s in both modes; the
bridge and its worker serve them in the function's place at start-up, after checking a
fingerprint of the sources and constants it depends on (`stopping.py`, `ship.py`,
`emergency_stop.py`, `constants.py`, `bluefin/*.py|*.json` and the stop-test constants) and
recomputing some entries. A key outside the table still goes to the function. After any change
to those sources the bridge warns and runs without it; rebuild with

```bash
python p3_bridge.py --build-stopping-table
```

## Running it

Live, on the shore laptop (one bridge process per run; the run identifier is the run sheet's):

```bash
python udp_live_p3.py --server-ip 10.201.205.110 --layout L1 --run-id T1-A-L1-r1
```

The link is the July one: the shore laptop runs the bridge, which listens on 5000 and sends to
5050 on the laptop that controls the vessel (`--server-ip`). The command line keeps the July
bridge's (`udp_live_rl.py`) names and defaults for the options both share: `--bind-ip`
(0.0.0.0), `--local-port` (5000), `--server-ip`, `--server-port` (5050), `--test-case`,
`--lidar-index0-deg`, `--rudder-sign`, `--no-pose-sync`, `--pose-wait-s` and the `--estop-*`
options; `--model-path` keeps its name, with the deployed model as its default; `--shadow` is
accepted for `--no-send`. Abbreviated options are refused, so a July option this bridge does
not take stops it at start-up instead of matching a longer option (the July `--shadow` would
otherwise have matched `--shadow-inline`, and commands would have been sent). Not carried over: `--record-log` (every received line is logged to
`telemetry.log` in the run folder), `--record-video`, `--fixed-rpm` and the RPM, S2 and
rudder-scale options (the command is resolved as the simulator resolves it), the Paper 2
observation options and the display options.

Geometry, one of: `--scenario <name>` (a set-up by name through `static_dynamic_obstacles/src/setups.py`,
the format `src/test_asv.py` also loads: a field test case, `easy_1` ... `hard_5`, with its target
set-up printed at start-up, or a test set v4 ID),
`--layout L1|L2|L3` (Paper 2's basin layouts, `paper2_set.LAYOUTS`),
`--case <test set v4 ID>` (the simulated twin's map and path, e.g. `P2-L1-HO-FIX-03`),
`--start x,y --goal x,y` (a basin leg), `--test-case N` (the July bridge's `test_run.py`).

Shadow session on the bench, against a July log:

```bash
python fake_vessel_replay.py --log 2026-07-03/1.log
```

```bash
python udp_live_p3.py --test-case 1 --run-id bench-0703-1 --no-send
```

Keys: E operator stop (full astern until the measured surge reads stopped, then zero
propulsion), R release after the minimum hold, ESC quit. The key acts only while the bridge
runs and its window has focus; a stop that does not depend on the shore laptop is outside the
bridge. A frame the bridge cannot process is logged and answered with `$CMD,0.00,0.00`.

The start-up takes about half a minute for the imports and as long again for the shadow
worker; START goes to the vessel only once both are ready.

### Unexpected line types (added 2026-11-23)

The decoder (`log_parser.BluefinStreamDecoder`) reads four line types: pose, LiDAR scan, `HDG`
and RC. Any other line is skipped, except one in the scan's shape: every `[time][...]` line is
read as a scan. A bracketed line of another kind, such as an IMU record, would make the decoder
raise (values that are not integers, an empty list), which stopped the bridge, or would pass as a
scan when its values are integers (six values stretched to 720 beams). The bridge therefore keeps
from the decoder any bracketed line that is not 720 integers (`--lidar-raw-beams`, 0 for any
length), and catches any decoder error. Such a line is still written to `telemetry.log`, listed
under `other_lines` in the next decision record with the reason, counted in `summary.json`
(`lines_not_decoded`, `decode_errors`, `line_kinds`) and printed once per reason. All 53,058
LiDAR lines of the April, May, July and February logs and the bench runs hold 720 integers, so
the change leaves their decisions unchanged. If a new firmware appends fields to the pose line,
the decoder no longer finds a pose, the bridge makes no decision, and the display shows the
count of pose, LiDAR and other lines received while it waits. The July bridge
(`udp_live_rl.py`, unchanged) still stops on such a bracketed line.

`udp_raw_logger.py` records a session with no decoder at all: the handshake, then every datagram
as received, written to disk before anything reads it, with its arrival time in a CSV beside it.
It prints each new line kind with an example and with what the decoder would do with it (read,
ignored, taken as a scan, decoder error). It needs only Python 3.6 or later and no other file, so
it can be copied on its own to any machine. `--selftest` runs a loopback test with a stand-in
vessel, and `--check <log>` gives the line kinds of a saved log.

```bash
python3 udp_raw_logger.py --server-ip 10.201.205.110 --out imu_test_1.log
```

## Logs (one clock)

Everything is keyed to the frame's telemetry time (`t_frame`, seconds from the first line; the
vessel's `ts` string kept) and to the bridge's monotonic clock; the header records both clocks
at start. Per run, under `p3_runs/<run-id>/`:

| File | Content |
| --- | --- |
| `decisions.jsonl` | a header (run identifier, checkpoint and config hashes, layer class and options, constants, geometry: boundary polygon, path, panels; arguments; source hashes; git head), then one record per decision: the 720 ranges and the pose, ego and staleness fed to perception, the previous executed action, the 70-value observation by branch, the policy action, the issued command (rudder, rpm, S2, vessel rudder, source, `$CMD` text, sent), the tracker's tracks and the encounter contexts, the raw telemetry pose and its pose-sync age and wait, any unrecognised line received since the last decision (an IMU line would appear here), the operator-stop state, the timings |
| `shadow.jsonl` | one record per decision: branch (`why`), mode, would-change, brake, stop commitment, the command it would issue, the detector's T*, the checked clearances, its plan, its diagnostics (bulky keys dropped unless `--full-filter-trace`), its run time and its lag behind the frame |
| `telemetry.log` | every received line, as received, with `#P3` lines for the commands; readable by `log_parser.py` and replayable by `fake_vessel_replay.py` |
| `summary.json` | counts, pose-sync statistics, the shadow layer's totals (decisions, would-change, first change, stop), operator stops |

The decision log alone reproduces every shadow decision offline (`ShadowRunner` fed
`decisions.jsonl`; checked in part B of the shadow replay). Target ground truth (D-2), video and
the run sheet are kept outside the bridge.

## Bench results

Measured 2026-11-05 on the development laptop (Intel Family 6 Model 154, 12 logical CPUs,
Windows 11, Python 3.10.11, one native thread). Outputs under `p3_bench/`.

### Parity with the simulator (`p3_parity_replay.py`, P-9)

| Run | Result |
| --- | --- |
| 16 test set v4 episodes (stops, target contacts, interventions, every class, L1-L3, a channel, a null encounter, a no-target leg) x {shadow, active} | **32/32 pass, 1,497 decisions**: observation (all 70 values), policy action, issued command, executed action and the layer's decision (branch, command, brake, stop, plan, every diagnostic) exactly equal; in shadow mode the worker's code path equal too (`parity/ts4_selection16_table/`, with the stopping table; `parity/ts4_selection16/` without it, same result) |
| Simulator reference against the recorded E1 traces of the same episodes (`ts4_final_e1_*`) | 16/16 equal, decision by decision |
| Same 16 episodes x {shadow, active} on the personal laptop (2026-11-14) | 32/32 pass, 1,497 decisions, recorded E1 traces 16/16 equal (`parity/personal_i7_shadow/`, `parity/personal_i7_active/`, with the stopping table) |
| 3 episodes x {shadow, active} with the environment's staleness model on (p = 0.2, 49 stale frames) | 6/6 pass (`parity/stale02/`) |
| Telemetry encoding, full precision | ranges within the 0.05 m quantisation, pose within 5e-7 m; decoded u, v, r against the simulator's motion: slopes 0.91-1.18 (signs agree); a LiDAR-point pose decoded with `--pose-point sensor` recovers the reference point within 5e-7 m |

Not covered by the comparison: lost frames (the simulator never drops one; the gap logic is
the bridge's), the operator stop, and the conversion from real telemetry, which only the
vessel can show (S1-A: stationary at a known pose and heading).

### Shadow session on the July logs (`p3_shadow_replay.py`)

| Part | Result |
| --- | --- |
| A, in process, all 19 July logs (1,085 decisions) | pass: every frame decided, every observation finite and inside the space, one shadow decision per decision, the worker path's perception equal to the bridge's, no errors (`shadow_replay/july_inproc/`) |
| B, UDP over loopback, `fake_vessel_replay.py` -> `udp_live_p3.py` (headless, worker), 2026-07-03 logs 1-3 (133 decisions) | pass: decisions and shadow decisions equal to part A's; the decision log alone reproduces every shadow decision offline (`shadow_replay/july_udp/`). Repeated 2026-11-14 on the personal laptop, bridge on its live port 5000 (`shadow_replay/udp_personal_i7/`): same result, part A and part B equal to the development laptop's |

The July runs are open loop for Paper 3: the vessel followed the Paper 2 policy at RPM 12,
about 1.1-1.3 m/s against the deployed policy's 0.56 m/s, with no target. Their shadow
decisions test the pipeline, not the trials.

### Run time (`p3_timing.py`, `p3_shadow_replay.py` part B; P-10, T-14)

Test set v4, 2 episodes per cell (61 episodes, 2,590 decisions, the first of each apart):

| Component | p50 | p95 | p99 | max | > 0.5 s |
| --- | --- | --- | --- | --- | --- |
| observation | 7.4 ms | 23.5 ms | 35.7 ms | 56.6 ms | 0 % |
| policy | 0.8 ms | 2.1 ms | 3.2 ms | 4.3 ms | 0 % |
| command path (observation + policy + command) | 8.4 ms | 25.9 ms | 37.9 ms | 58.7 ms | 0 % |
| E1 layer, one decision | 33 ms | 293 ms | 880 ms | 1.87 s | 1.8 % |
| E1 in the loop (command path + layer) | 42 ms | 302 ms | 890 ms | 1.90 s | 1.9 % |
| shadow worker, frame to decision at 2 Hz | 41 ms | 318 ms | 1.29 s | 7.6 s | 2.5 % |

Personal laptop, 2026-11-14 (Intel Core i7-6600U, 2 cores / 4 logical CPUs, 7.9 GB, Windows 10,
Python 3.10.11; `timing/personal_i7/`), same 61 episodes and 2,590 decisions:

| Component | p50 | p95 | p99 | max | > 0.5 s |
| --- | --- | --- | --- | --- | --- |
| observation | 14.3 ms | 20.7 ms | 25.6 ms | 32.3 ms | 0 % |
| policy | 1.2 ms | 2.0 ms | 2.7 ms | 4.8 ms | 0 % |
| command path (observation + policy + command) | 15.6 ms | 22.1 ms | 27.2 ms | 34.2 ms | 0 % |
| E1 layer, one decision | 60 ms | 480 ms | 872 ms | 1.65 s | 5.0 % |
| E1 in the loop (command path + layer) | 76 ms | 496 ms | 887 ms | 1.67 s | 5.0 % |
| shadow worker, frame to decision at 2 Hz | 75 ms | 783 ms | 2.08 s | 3.67 s | 6.9 % |

July logs over loopback UDP (130 decisions): observation p99 75 ms (real scans hold about 550
returns); frame release to `$CMD` sent p50 34 ms, p99 79 ms; LiDAR line to sent (pose-sync
wait included) p99 86 ms; E1 layer p50 149 ms, p95 1.55 s, p99 2.06 s; the worker fell up to
13 s behind the frames and caught up after each run. Personal laptop (`shadow_replay/
udp_personal_i7/`, bridge on port 5000): observation p99 80 ms; frame release to `$CMD` sent
p50 34 ms, p99 85 ms; LiDAR line to sent p50 39 ms, p99 92 ms, max 103 ms; E1 layer p50 83 ms,
p95 759 ms, p99 962 ms.

**T-14.** The command path meets it with a wide margin: p99 about 0.09 s from the LiDAR line
to the command, against 0.5 s. **E1 in the loop does not**: its p99 is 0.89 s on test-set
streams and 2.06 s on the July logs. Shadow mode is unaffected, because the worker runs it
off the command path and its decisions do not depend on when they are computed. The
vessel-link latency is not in these figures: the bench has no vessel.

## Open items

| Item | Finding | Consequence |
| --- | --- | --- |
| E1 run time (T-14) | p99 0.89-2.06 s per decision, above the 0.5 s interval | active mode (`--safety active`) does not run at 2 Hz on this laptop; the optional active block (D-4) needs a faster layer or computer, measured again on the shore laptop |
| Phantom tracks | on the July logs (no target), 42 % of decisions carry a dynamic track; 61 % of these lie within 1 m of the map boundary or outside it, 35 % elsewhere, 4 % near a panel; median speed 0.23 m/s, p90 0.65 m/s. They reach the target branch, and the shadow layer would commit a stop in 15 of 19 replays (as early as decision 2, on a phantom 3.3 m away given 1.08 m/s) | most look like basin-wall returns surviving the gate under the start-anchored map frame and 0.1 m / 0.1 deg poses; registration on the landmark survey (P-7, `--origin map`) and the S2-C dress rehearsal must show them gone before the trials |
| Telemetry resolution (P-1, O-2) | at the July resolution (pose 0.1 m and 0.1 deg at 2 Hz, ego by pose differences), the observation moves from the simulator's on average by 0.11-0.16 (ego), 0.03-0.09 (path), up to 0.40 (target) and 0.53 (context), in normalised units | higher pose resolution, or an ego estimate from the IMU, before the trials. The rate itself is set on the vessel: the July logs carry frames exactly 0.5 s apart by the vessel's own time stamps, with `hey` lines and no `HDG` or RC lines; the February logs (radio-controlled) 0.1 s apart, with `HDG` and RC lines and no `hey`; a receiver cannot thin a stream that regularly or remove line types, so the difference is the onboard program's mode or setting. Since 2026-11-20 the ego estimate spans the decision interval at either rate (`--ego-window-s`) |
| Pose point | the July bridge took the telemetry pose as the vessel reference point; whether the localisation reports the LiDAR's (0.86 m ahead) is not established | settle at S1-A, then `--pose-point` |
| Map frame | `--origin start` assumes the telemetry axes are the basin's and the vessel sits on the path start at the first frame | `--origin map` with the P-7 registration |
| UDP ports on this laptop | UDP 5000, 5001 and 5050 are held by system services (`nidmsrv.exe`, `lktsrv.exe`, `svchost.exe`); the July bridge listened on 5000 | free ports, or another laptop, before a session; UDP itself is blocked on this laptop's network (UTAS). Personal laptop (2026-11-14), the shore laptop for the link to the laptop that controls the vessel: 5000 free, 5050 held by `svchost.exe` (CDPSvc), which affects only a fake vessel on the same machine; UDP not restricted on its networks; step 6 of `UDP_BENCH_HANDOVER.md` not run (the link is not between two bench laptops; loopback taken as the bench test); the link is first tested at step 7, at the next basin booking, with the July ports |
| Operator stop | the E key acts only while the bridge runs and has focus | a stop independent of the shore laptop belongs to the session's safety plan |
| Active stop | full astern (S2 = -100) | void until P-8 (ESC reverse) and T-12 |
| Target footprint (D-1) | the tracker's admission gates and the layer's hull completion assume a target of the own ship's size | measure the real target's detected footprint on the bench |
| Start-up | about 30 s of imports, and as long again for the worker; the stopping table removes the cold-cache delays | start the bridge before the vessel is released |
