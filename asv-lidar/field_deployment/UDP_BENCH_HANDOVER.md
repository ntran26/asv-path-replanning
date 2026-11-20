# UDP bench of the Paper 3 bridge on a second laptop (handover, 2026-11-14)

The university laptop cannot carry this part of the bench: the university network blocks UDP,
and UDP 5000, 5001 and 5050 on that laptop are held by system services (`README_P3_BRIDGE.md`,
open items). The UDP part of the bridge bench is therefore repeated on a personal laptop, which
also makes that laptop the shore-laptop candidate (`static_dynamic_obstacles/planning/
FIELD_CALIBRATION_PLAN.md`, O-10). The items it closes:

| Item | Content | State on the university laptop |
| --- | --- | --- |
| P-9, part B | `fake_vessel_replay.py` -> `udp_live_p3.py` over UDP, decisions equal to the in-process replay | passed over loopback, marked provisional |
| P-10, T-14 | run time per decision with communication: command path p99 below 0.5 s (required); E1 in the loop p99 below 0.5 s (only for the optional active block) | command path met (p99 0.09 s, LiDAR line to sent); E1 not met (p99 0.89-2.06 s) |
| P-11 | free UDP ports for the bridge and the vessel link; UDP allowed on the session network | ports taken; network blocks UDP |

Nothing here sends a command to the vessel: every bridge run below uses `--no-send` or the fake
vessel. P-12 (an operator stop independent of the shore laptop) is a hardware item, outside
this note.

## 1. Set-up

- Clone `https://github.com/ntran26/asv-path-replanning.git` (master); every path in this note
  is relative to its `asv-lidar/` folder. The deployed model
  `static_dynamic_obstacles/runs/sac_formulation_seed0_bl3/best_model.zip`, the stopping table
  `field_deployment/p3_stopping_profiles.pkl`, the July logs `field_deployment/2026-07-0*/` and
  test set v4 (`static_dynamic_obstacles/results/test_set/v4/definition.*`) are tracked.
- Python 3.10 with `static_dynamic_obstacles/requirements.txt` (numpy 1.26.4, torch 2.2.1,
  gymnasium 0.29.1, stable-baselines3 2.3.2, sb3-contrib 2.3.0, pandas 2.2.1), plus the versions
  on the university laptop of the packages the bridge also imports: pygame 2.5.2,
  opencv-python 4.11.0.86, scipy 1.15.3, matplotlib 3.8.3.
- The bridge refuses a checkpoint whose hash or run configuration differs from the code's, so a
  wrong model or a changed constant stops it at start-up rather than running silently.
- Record the laptop's CPU model, logical core count, RAM and operating system with the results;
  the run-time figures are only meaningful with them.

## 2. Checks, in order (from `field_deployment/`)

| Step | Command | Pass | Item |
| --- | --- | --- | --- |
| 1. Environment parity (no network) | `python p3_parity_replay.py --tag <laptop>` | `summary.json` passed: 32 of 32 runs (16 episodes, shadow and active), every decision equal to the simulator's (1,497 on the university laptop) | the clone and packages reproduce the simulator |
| 2. Ports | Windows: `Get-NetUDPEndpoint -LocalPort 5000,5001,5050` in PowerShell (or `netstat -ano -p udp`); macOS or Linux: `lsof -nP -iUDP` | nothing holds 5000 (bridge) or 5050 (the vessel's port, for the fake vessel); a holder is recorded with its process | P-11 |
| 3. Firewall | first listening run of Python | Windows Defender Firewall asks once; Python allowed on private networks only; the vessel's network marked private | P-11 |
| 4. UDP over loopback, live ports | `python p3_shadow_replay.py --tag udp_<laptop> --port 5050 --local-port 5000` (part A, all 19 July logs in process, then part B over UDP on logs 2026-07-03 1-3) | `PASSED`; for each part-B log `same_as_A=True` and `offline=True`; `timing.csv` LiDAR line to sent p99 below 0.5 s | P-9 part B, P-10 |
| 5. Run time on encounter-rich streams | `python p3_timing.py --tag <laptop>` (test set v4, 2 episodes per cell) | command path p99 below 0.5 s (required); E1 in the loop p99 reported against 0.5 s | P-10, T-14 |
| 6. UDP across two devices | on a network other than the university's (the vessel's router or a phone hotspot): a second computer runs `python fake_vessel_replay.py --log 2026-07-03/1.log --port 5050`; the laptop runs `python udp_live_p3.py --server-ip <that computer's IP> --test-case 1 --run-id lan-<laptop> --no-send --headless` | frames received and decided for the whole log; `decisions.jsonl` complete | P-11 |
| 7. Vessel link, at the basin | vessel on its stand or moored, telemetry and LiDAR running: `python udp_live_p3.py --server-ip 10.201.205.110 --layout L1 --run-id link-<date> --no-send` | START reaches the vessel; frames and pose lines arrive and pair at 2 Hz; no `$CMD` sent | P-11 (the full test of the link is the dress rehearsal, S2-C) |

If step 4 fails only because a port is held, it is rerun on the free pair of step 2 (as the
university laptop ran it, 15050 and 15000), and the live session uses that pair through
`--local-port` and the vessel's configured port.

University-laptop reference figures (Intel Family 6 Model 154, 12 logical CPUs; 2026-11-05):
command path p99 37.9 ms on test-set streams; LiDAR line to sent p99 86 ms on the July logs over
loopback UDP; E1 in the loop p99 0.89 s (test set) and 2.06 s (July logs).

## 3. Where the results go

- Outputs stay where the scripts put them (`p3_bench/parity/<tag>/`, `p3_bench/shadow_replay/
  <tag>/`, `p3_bench/timing/<tag>/`, `runs/` of the bridge for steps 6 and 7).
- `README_P3_BRIDGE.md`: the part-B row loses "provisional" if step 4 passes; the run-time
  section gains this laptop's figures beside the university laptop's; the UDP-ports open item
  records the outcome of steps 2, 3 and 6.
- Section 4 of this note: a dated line per step with its result.
- The planning files (`static_dynamic_obstacles/planning/`, including `00_CURRENT_STATE.md`,
  `FIELD_CALIBRATION_PLAN.md` and `PAPER3_*`) hold uncommitted edits on the university laptop and
  are updated there; editing them on this laptop would conflict on the next pull.

## 4. Results

Personal laptop: Intel Core i7-6600U @ 2.60 GHz (Intel64 Family 6 Model 78), 2 cores / 4
logical CPUs, 7.9 GB RAM, Windows 10 Pro 10.0.19045; Python 3.10.11 (python.org) in a `.venv`
from `static_dynamic_obstacles/requirements.txt` (torch 2.2.1+cpu). The clone was first checked
out with `core.autocrlf=true`, which changed the bytes of tracked text files and disabled the
stopping table (fingerprint mismatch); it was set to `false` and re-checked out before the runs
below, after which the table installs (602 entries). After step 5 the same pinned packages
(identical `pip freeze`) were installed into the base Python 3.10.11, which is first on the
path, and the `.venv` was removed; shadow parity on the base install (`parity/personal_i7_base/`,
36fb118cd) passed 16/16 runs, 600 decisions.

- 2026-11-14, step 1, shadow (`parity/personal_i7_shadow/`, e35252f4c): 16/16 runs pass,
  600 decisions; per-case decision counts equal the university laptop's; stopping table on.
- 2026-11-14, step 1, active (`parity/personal_i7_active/`, fefb19172): 16/16 runs pass,
  897 decisions; recorded E1 traces 16/16 checked and equal; no mismatch. With the shadow runs,
  32/32 and 1,497 decisions, as on the university laptop. `parity/personal_i7/` is an aborted
  run from before the line-ending fix (it stopped at the first active case, whose trace was not
  yet committed) and has no `summary.json`.
- 2026-11-14, step 2: UDP 5000, 5001, 15000 and 15050 free; 5050 held by `svchost.exe`
  (service CDPSvc, Connected Devices Platform). 5050 is the port of the laptop that controls the
  vessel, so the live link (bridge on 5000, as in July) is not affected; only the fake vessel
  moves to 15050 on this laptop.
- 2026-11-14, step 3: no firewall prompt. Python listened on UDP and then TCP 0.0.0.0:5000
  (90 s and 120 s); no prompt appeared and no rule for `python.exe` exists afterwards
  (notification on listen enabled in all profiles; shell not elevated). The Wi-Fi profile is
  Public. Loopback (steps 4-5) is unaffected; UDP is not restricted on this laptop's networks.
- 2026-11-14, step 4 (`shadow_replay/udp_personal_i7/`, e35252f4c, `--port 15050
  --local-port 5000`): `PASSED`. Part A 19/19 logs, 1,085 decisions; 15 logs with a would-be
  stop, would-change 424, pose stale 0. Part B, 2026-07-03 logs 1-3: 49, 43 and 41 decisions,
  `same_as_A=True` and `offline=True` on each (would-change 0 / 7 / 0, first change - / 36 / -,
  stop - / 38 / -), as on the university laptop. LiDAR line to sent p50 38.9 ms, p95 73.0 ms,
  p99 91.9 ms, max 102.7 ms (n = 130); shadow layer p99 962 ms. Bridge, shadow worker and fake
  vessel shared the 4 logical CPUs.
- 2026-11-14, step 5 (`timing/personal_i7/`, fefb19172): 61 episodes, 2,590 decisions, run
  alone after the active parity run. Command path p50 15.6 ms, p95 22.1 ms, p99 27.2 ms, max
  34.2 ms, 0 % over 0.5 s (required limit met). E1 layer p99 872 ms (5.0 % over 0.5 s); E1 in
  the loop p50 76 ms, p95 496 ms, p99 887 ms, max 1.67 s, 5.0 % over 0.5 s (not met); worker lag
  p99 2.08 s, max 3.67 s, 6.9 % over 0.5 s. Background at the start: about 58 % CPU with no
  Python running (a desktop chat application about 44 %, `dwm` about 13 %); Microsoft Store,
  qBittorrent and Proton VPN closed, VPN off.
- 2026-11-14, step 6: not run. The link is between this laptop and the laptop that controls
  the vessel, not between two bench laptops; the single-machine loopback test (step 4) stands
  as the bench test of the UDP path, and the link across devices is first exercised at step 7.
- Step 7: scheduled for the start of the next basin booking, before any session. The July
  set-up is kept: this laptop runs the bridge on 5000, the laptop that controls the vessel
  listens on 5050 at `--server-ip` (10.201.205.110 in July), and the command line keeps the July
  bridge's option names (`--shadow` is accepted for `--no-send`):
  `python udp_live_p3.py --server-ip 10.201.205.110 --layout L1 --run-id link-<date> --no-send`.
  On the day, other applications are closed. First check on the day: frames arrive within
  seconds of START. Unlike the July Ubuntu machine, this laptop's host firewall is on (no rule
  for Python, Wi-Fi profile Public) and passes inbound UDP only as replies to START, so the
  vessel-control laptop sends from 5050 after START. If frames do not arrive, an inbound allow
  rule for the Python 3.10.11 interpreter (`%LOCALAPPDATA%\Programs\Python\Python310\python.exe`;
  a venv's `python.exe` only starts it), added with administrator rights.
- 2026-11-20, February 2026 logs (`data/test_1.log` to `test_4.log`: raw 10 Hz vessel streams
  with no `#CONFIG` line and no IMU), shadow replay with `--test-case 0` (option added in
  04d6fd731), fake vessel on 15050 and bridge on 5000: `PASSED` before and after the ego window.
  Before it (`shadow_replay/feb_logs_personal_i7/`, 35736af0a): part A 4/4 logs, 418, 162, 156
  and 443 decisions (1,179), equal to the university laptop's replay of the same logs; pose
  stale 0; part B 4/4 with `same_as_A=True` and `offline=True`. With it
  (`shadow_replay/feb_logs_personal_i7_ego/`, 76a984450, `--ego-window-s` 0.5 s): the same
  decision counts and part-B result; `telemetry.ego_dt_s` median 0.500 s, p99 0.507 s, max
  0.602 s (n = 1,176); the largest speed in the decisions fell from 2.03, 1.45, 2.33 and
  2.24 m/s to 1.41, 1.08, 1.71 and 1.62 m/s. The console's "Max velocity" (2.367 m/s) comes from
  `log_parser`'s decoder over 0.1 s and is not the bridge's ego estimate. The shadow layer
  changed only on test_2 (would-change 124 -> 98, first would-stop at decision 38 -> 68) and
  test_4 (would-change 401 -> 402). LiDAR line to sent p50 37.0 ms, p99 82.0 ms, max 111.2 ms
  (n = 1,175) with other applications running; p99 below 0.5 s on both runs. Phantom dynamic
  tracks on 36, 19, 19 and 46 % of decisions (no map registration, P-7), so the would-stops
  under test-case-0 geometry carry no meaning. The outputs stay on that laptop, uncommitted.

## 5. Conventions of this repository

- Nothing under `static_dynamic_obstacles/src/` or `configs/` changes (baseline-v3, config
  digest 03a9c0c22f53ee0b); new behaviour goes in new modules.
- Commits and pushes only on request; no permanent deletion of data; history rewrites (gc,
  filter-repo, force-push) are left to the repository owner.
- Repository documents use an impersonal voice, British spelling in planning files, the terms
  "coupled" and "decoupled", "validation set", "COLREGs-aware" (never "compliant").
- The safety layer runs in shadow mode on the water; active mode (`--safety active`) is not for
  the water before the ESC reversal test (P-8) and T-14 for E1 are met.
