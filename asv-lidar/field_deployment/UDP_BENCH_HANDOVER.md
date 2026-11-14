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

(empty until the steps run)

## 5. Conventions of this repository

- Nothing under `static_dynamic_obstacles/src/` or `configs/` changes (baseline-v3, config
  digest 03a9c0c22f53ee0b); new behaviour goes in new modules.
- Commits and pushes only on request; no permanent deletion of data; history rewrites (gc,
  filter-repo, force-push) are left to the repository owner.
- Repository documents use an impersonal voice, British spelling in planning files, the terms
  "coupled" and "decoupled", "validation set", "COLREGs-aware" (never "compliant").
- The safety layer runs in shadow mode on the water; active mode (`--safety active`) is not for
  the water before the ESC reversal test (P-8) and T-14 for E1 are met.
