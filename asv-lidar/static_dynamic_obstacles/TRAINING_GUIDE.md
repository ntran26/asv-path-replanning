# Training guide — running the baseline campaign

How to start, check, hold, pause, resume and move the five-learner campaign
(PPO, RecurrentPPO, SAC, TQC × 3 seeds; TD3 dropped and 5 seeds → 3 on 2026-09-24) on the frozen formulation
**baseline-v2** (`configs/baseline_v2.json`). Runs on the previous formulation
-- run 11 and the `_bl1` folders -- are not part of this campaign (F96).

Commands are run from this folder (`asv-lidar/static_dynamic_obstacles`). Bash
commands work in Git Bash; PowerShell commands in a PowerShell terminal.

Background: `PROJECT_STATE.md` F93 (the first freeze), F95 (the campaign and its
safeguards), F96 (baseline-v2); `OPEN_PROBLEMS.md` A26 (the budget decisions).

---

## 1. What is where

| Thing | Location |
|---|---|
| The frozen formulation and campaign settings | `configs/baseline_v2.json` (`configs/baseline_v1.json` is kept for the record) |
| The campaign launcher | `results/baseline_campaign.sh` |
| One folder per run | `runs/<learner>_formulation_seed<N>_bl2/` |
| Training output of a run | `runs/<learner>_formulation_seed<N>_bl2.log` |
| Campaign event log | `results/baseline_campaign.log` |
| Tier 1 of a finished run (development set, a diagnostic) | `results/tiers/tier1_<learner>s<N>_bl2_supervisor_{off,on}/summary.txt` |
| **Frozen suite of a finished run (what the paper reports)** | `results/frozen_suite/<learner>s<N>_bl2/summary.txt` — **Tier B** (39 cells x 20 = 780 episodes, suite 3.1), supervisor off and on. Tier A is the **extended** set and runs only on request (`--tiers a,b`) |
| TensorBoard curves | `runs/tensorboard/` |
| Off-policy replay buffers (SAC/TQC only) | `PhD/asv_replay_buffers/<run>/` — **outside the repository**, latest only, deleted when the run finishes |

Inside a run folder:

| File | What it is |
|---|---|
| `best_model.zip` | **the model the paper uses** — best development-set score (goal − 2 × collision, supervisor off) |
| `final_model.zip` | the model at 2 M steps; its presence means the run is **finished** |
| `<learner>_<steps>_steps.zip` | checkpoints every 250 k steps (what a resume starts from) |
| `*vecnormalize*.pkl` | reward normalisation statistics (a few KB) |
| `config.json` | everything the run used: settings, baseline id and digest, software versions, git commit, any resumes |
| `eval_summary.json`, `eval_episodes.csv` | development-set evaluations every 200 k steps |
| `monitor.csv`, `curriculum.json` | per-episode training log, curriculum stage changes |

**Campaign order:** seed 0 of all four learners, then seeds 1–2 seed by seed.
**The frozen suite runs separately** (your call, 2026-09-23): the campaign does
Tier 1 after each run, and `bash results/frozen_seed0.sh` evaluates Tier B on
every seed-0 model once the four trainings are done.
Approximate hours per run on this machine: PPO 6, RecurrentPPO 7–9, SAC 33,
TQC 31 (SAC measured at 16.8 steps/s on an otherwise idle machine).

---

## 2. Before starting

1. **Stop the computer sleeping.** Settings → System → Power → *When plugged
   in, put my device to sleep after* → **Never**. A sleeping machine pauses
   training.
2. **Windows Update.** Set active hours or pause updates for the campaign. A
   restart stops training; see §7 to continue.
3. **Disk space.** C: needs a few GB free. Each off-policy replay buffer is
   ~0.58 GB (one at a time); a buffer is not saved below 3 GB free.
4. **Check the code matches baseline-v2** (the launcher also does this and
   refuses to start otherwise):

```bash
python src/baseline_config.py --check
```

Expected: `code matches baseline-v2`.

---

## 3. Start the campaign

**Detached (recommended)** — keeps running if the terminal or the Claude app
closes. Run in PowerShell from this folder:

```powershell
Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe" -ArgumentList '-lc', '"bash results/baseline_campaign.sh"' -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput results\baseline_campaign.console.log -RedirectStandardError results\baseline_campaign.console.err
```

There is no window: check it with §4.

**In the foreground** — simpler, but stops if the terminal closes:

```bash
bash results/baseline_campaign.sh
```

**Part of the campaign only.** `JOBS` overrides the run list; the campaign then
stops when that list is done. This is how the 23 Sep launch ran seed 0 of four
learners, TD3 excluded:

```bash
JOBS="ppo:0 recurrent_ppo:0 sac:0 tqc:0" bash results/baseline_campaign.sh
```

Each job is `learner:seed`. To run any other subset later:

```bash
JOBS="sac:1 tqc:1" bash results/baseline_campaign.sh
```

With no `JOBS`, the full 12-run plan runs in the order in the script: seed 0 of
all four learners, then seeds 1-2. Finished runs are skipped either way, so a
full launch after a partial one simply continues.

**Only one campaign at a time.** Starting it twice runs two trainings on the
same run folder. Check first that nothing is running (§4, last item).

The launcher is safe to start again at any time: finished runs are skipped, a
run with a checkpoint continues from it, and a run that died before its first
checkpoint is moved to `<run>_incomplete_<date>` and restarted.

---

## 4. Check progress

**Status of every run** (quickest):

```bash
python tools/campaign_status.py
```

Shows done / training x % / pending for all 12 runs, each run's best
development-set goal rate so far, whether a hold is set, and the last campaign
log lines.

**Live training output** of the current run (Ctrl+C closes the view, not the
training) — replace the run name:

```powershell
Get-Content runs\ppo_formulation_seed0_bl2.log -Wait -Tail 20
```

Useful lines: `total_timesteps` (progress), `[EVAL] t=... goal ... collision
...` (development-set result every 200 k steps; an evaluation takes a few
minutes, during which the log is quiet), `[CURRICULUM]` (stage changes).

**Campaign events** (run started / finished / evaluated / stopped):

```powershell
Get-Content results\baseline_campaign.log -Tail 10
```

**Learning curves:**

```bash
tensorboard --logdir runs/tensorboard
```

then open <http://localhost:6006>.

**Is anything running?** In Task Manager, a running training is 11 Python
processes (the learner and 10 environment workers). Or:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag bl2 -Action status
```

`no running learner for tag bl2` means nothing is training (the campaign may
still be running a Tier 1 evaluation between runs — check the campaign log).

---

## 5. Hold before the next run (stop file)

To let the **current** run finish (including its Tier 1 evaluation) and then
stop before the next one starts:

```bash
touch runs/CAMPAIGN_STOP
```

The campaign log then shows `stop file found, stopping before <learner> seed
<N>`. To release the hold and continue: delete the file, then start the
campaign again (§3).

```bash
rm runs/CAMPAIGN_STOP
```

`campaign_status.py` reports when the stop file is present.

---

## 6. Pause and resume

There are two ways, depending on how long.

### 6a. Short pause, in memory (minutes to hours)

Freezes the training processes where they are; resuming continues exactly
where it stopped, losing nothing:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag bl2 -Action pause
```

```powershell
powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag bl2 -Action resume
```

The pause does **not** survive a reboot or sign-out, and a pause across sleep
was once found to leave the processes idle after resuming (F86). For anything
overnight, use 6b.

### 6b. Stop now, resume later (any length, survives reboots)

Stop the campaign and all its processes:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'baseline_campaign\.sh' } | ForEach-Object { taskkill /PID $_.ProcessId /T /F }
```

Resume by starting the campaign again (§3). The interrupted run continues from
its last 250 k-step checkpoint, so up to 250 k steps are redone (about 1 h for
RecurrentPPO, up to ~6 h for SAC). Off-policy runs also reload their replay
buffer from `PhD/asv_replay_buffers/`; the run's `config.json` records each
resume and whether the buffer was restored.

To lose less, stop just after a checkpoint appears in the run folder
(`<learner>_<steps>_steps.zip`).

---

## 7. After a reboot, crash or power cut

Start the campaign again (§3). It skips finished runs and resumes the
interrupted one from its last checkpoint. If the interrupted run had not
reached its first checkpoint, it is set aside as `<run>_incomplete_<date>`
and restarted; that folder can be deleted.

---

## 8. Run or resume one learner by hand

One run outside the campaign order (same settings, same checks):

```bash
python src/train_formulation.py --config configs/baseline_v2.json --algo sac --seed 0 --tag bl2
```

`--algo` is one of `ppo`, `recurrent_ppo`, `sac`, `tqc` (`td3` still works but is
not part of the baseline). With
`--config`, only `--algo`, `--seed` and `--tag` come from the command line;
everything else comes from the file.

Resume a stopped run by hand:

```bash
python src/train_formulation.py --config configs/baseline_v2.json --resume runs/td3_formulation_seed0_bl2
```

The frozen suite of a finished run — **Tier B**, both supervisor modes, which is
the evaluation the paper reports (the campaign does this automatically after
each run). Add `--tiers a,b` only when the extended named cases are wanted:

```bash
python tools/tiers/frozen_suite.py --model runs/td3_formulation_seed0_bl2/best_model.zip --tag td3s0_bl2 --supervisor both
```

Tier 1, the development-set diagnostic (also automatic), supervisor `off` and
`on`:

```bash
python tools/tiers/tier1_replay.py --model runs/td3_formulation_seed0_bl2/best_model.zip --tag td3s0_bl2_supervisor_off --supervisor off
```

A quick launch check that trains 4,096 steps into a `_smoke` folder (delete it
afterwards):

```bash
python src/train_formulation.py --config configs/baseline_v2.json --algo sac --seed 0 --tag check --smoke
```

---

## 9. Do not change the formulation mid-campaign

Every run checks the code against `configs/baseline_v2.json` and refuses to
start if any constant, switch, curriculum or learner setting differs
(`BASELINE CHECK FAILED` in the campaign log, or `code does not match
baseline-v2` in a run log). This keeps all learners on the same formulation.

- Editing documents, diagnostics or the launcher is fine.
- Editing `src/constants.py`, the reward, the observation, the curriculum or a
  learner's hyperparameters makes a **new** formulation: it needs a new config
  version (`python src/baseline_config.py --write` after deliberate changes)
  and every learner retrained on it. Do not do this while the campaign runs.
- To check a finished run against the baseline:

```bash
python src/baseline_config.py --verify-run runs/td3_formulation_seed0_bl2
```

---

## 10. Saving to git and GitHub

The campaign does not commit anything; commit and push when convenient.
Pushing while training runs is safe (the training only writes its own run
folder).

- Replay buffers can never be committed: they are outside the repository, and
  `.gitignore` blocks `*_replay_buffer_*.pkl`.
- Model checkpoints are small (PPO ~3 MB, RecurrentPPO ~13 MB) but add up:
  ~130 MB per RecurrentPPO run. If pushes get slow, commit only
  `best_model.zip` and `final_model.zip` per run and keep the step checkpoints
  local.
- Committing a run while it is still training is fine; its later files are
  simply committed later.

---

## 11. Moving the remaining runs to a cluster

1. **Hold** the campaign here (§5) and let the current run finish — or stop it
   (§6b) if an off-policy run would take too long.
2. **Commit and push** the finished run folders, so the cluster can see which
   runs are done.
3. On the cluster: clone the repository, install `requirements.txt` (the
   software versions each run used are in its `config.json` → `platform`), and
   run the launcher or single runs (§8). Finished runs present in `runs/` are
   skipped.
4. Replay buffers go next to the clone by default (`asv_replay_buffers/`
   beside the repository); set `ASV_REPLAY_BUFFER_DIR` to put them elsewhere,
   e.g. on scratch storage. An off-policy run stopped here mid-way can only
   resume on the cluster with its buffer copied across from
   `PhD/asv_replay_buffers/<run>/`; without it, the resume refills 10 k steps
   before learning again (recorded in `config.json`).
5. Cluster job scripts (Slurm/PBS) are not written yet — ask for them once
   access is set up.

---

## 12. Troubleshooting

| Message or symptom | Meaning | What to do |
|---|---|---|
| `BASELINE CHECK FAILED` (campaign log) | the code differs from baseline-v2 | run `python src/baseline_config.py --check` to see what changed; undo it (§9) |
| `<learner> SEED <N> FAILED (see ...log)` | a run crashed | read the end of that run's log; start the campaign again — it resumes from the last checkpoint |
| `stop file found, stopping before ...` | a hold is set (§5) | delete `runs/CAMPAIGN_STOP` and start again, when wanted |
| `[BUFFER] ... GB free -- replay buffer NOT saved` | C: is nearly full | free disk space; training continues, but a resume from that checkpoint refills the buffer |
| `[BUFFER] a replay-buffer file stayed locked` | OneDrive held a file | harmless; an older buffer may be left in `PhD/asv_replay_buffers/` — delete it once that run is finished |
| `<run>_incomplete_<date>` folder | a run died before its first checkpoint and was restarted | delete it |
| log quiet for a few minutes | a development-set evaluation (every 200 k steps) | wait; `[EVAL]` lines follow |
| nothing in the log for much longer, no CPU use | the processes are stuck or paused | `pause_run.ps1 ... -Action status`; if paused, resume (§6a); otherwise stop and restart (§6b) |
