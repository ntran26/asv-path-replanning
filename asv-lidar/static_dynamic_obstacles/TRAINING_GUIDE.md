# Training guide — training the baseline runs on demand

> **Naming (2026-10-01).** "Supervisor" is now called the **safety layer** ("safety" for short) everywhere in documents, logs and command-line options (`--safety`, `--train-safety`, `--eval-safety`; the old flags still work). Identifiers frozen in the baseline configs and run records keep their old names, because renaming them would break the baseline-v2/v3 digests and the ability to read past results: `train_supervisor`, `eval_supervisor`, `ESTOP_TRIGGER = "supervisor"`, and the `supervisor` column in episode CSVs and `eval_summary.json`.


How to train, check, pause, resume and evaluate the baseline runs (PPO,
RecurrentPPO, SAC, TQC × 3 seeds) on the frozen formulation **baseline-v2**
(`configs/baseline_v2.json`). **Each run is trained on demand, one learner and
seed at a time** (your call, 2026-09-27): the campaign script that chained all
twelve is gone. Runs on the previous formulation -- run 11 and the `_bl1`
folders -- are not part of the baseline (F96).

Commands are run from this folder (`asv-lidar/static_dynamic_obstacles`). Bash
commands work in Git Bash; PowerShell commands in a PowerShell terminal.

Background: `PROJECT_STATE.md` F93 (the first freeze), F95 (the run safeguards),
F96 (baseline-v2), F102 (the frozen suite, suite 3.4); `OPEN_PROBLEMS.md` A26
(the budget decisions).

---

## 1. What is where

| Thing | Location |
|---|---|
| The frozen formulation, and the planned learners × seeds | `configs/baseline_v2.json` (`campaign` block: 4 learners × seeds 0–2; `configs/baseline_v1.json` is kept for the record) |
| Train and evaluate one run | `results/train_seed.sh` |
| Status of every planned run | `python tools/run_status.py` |
| One folder per run | `runs/<learner>_formulation_seed<N>_bl2/` |
| Training output of a run | `runs/<learner>_formulation_seed<N>_bl2.log` |
| Event log (started / resumed / finished / evaluated) | `results/train_seed.log` (`results/baseline_campaign.log` holds the seed-0 campaign's events) |
| Tier 1 of a finished run (development set, a diagnostic) | `results/tiers/tier1_<learner>s<N>_bl2_supervisor_{off,on}/summary.txt` |
| **Frozen suite of a finished run (what the paper reports)** | `results/frozen_suite/<learner>s<N>_bl2/summary.txt` — **Tier B**, suite 3.4: the 800-episode headline and the 900-episode robustness set, safety layer off and on. Tier A is the **extended** set and runs only on request (`--tiers a,b`) |
| TensorBoard curves | `runs/tensorboard/` |
| Off-policy replay buffers (SAC/TQC only) | `PhD/asv_replay_buffers/<run>/` — **outside the repository**; the latest checkpoint's while training, and the final one kept when the run ends so it can be continued |

Inside a run folder:

| File | What it is |
|---|---|
| `best_model.zip` | **the model the paper uses** — best development-set score (goal − 2 × collision, safety layer off) |
| `final_model.zip` | the model at 2 M steps; its presence means the run is **finished** |
| `<learner>_<steps>_steps.zip` | checkpoints every 250 k steps (what a resume starts from) |
| `*vecnormalize*.pkl` | reward normalisation statistics (a few KB) |
| `config.json` | everything the run used: settings, baseline id and digest, software versions, git commit, any resumes |
| `eval_summary.json`, `eval_episodes.csv` | development-set evaluations every 200 k steps |
| `monitor.csv`, `curriculum.json` | per-episode training log, curriculum stage changes |

Approximate hours per run on this machine: PPO 6, RecurrentPPO 7–9, SAC 33–35,
TQC 31–44 (TQC seed 0 took 44 h including a slow spell on the efficiency cores),
plus about 40 min for the evaluations afterwards.

---

## 2. Before starting a run

1. **Stop the computer sleeping.** Settings → System → Power → *When plugged
   in, put my device to sleep after* → **Never**. A sleeping machine pauses
   training. Keep it **plugged in**: on battery it slows to about two thirds.
2. **Windows Update.** Set active hours or pause updates while a run trains. A
   restart stops training; see §6 to continue.
3. **Disk space.** C: needs a few GB free. Each off-policy replay buffer is
   ~0.58 GB; a buffer is not saved below 3 GB free.
4. **Only one run at a time.** Check nothing is training (§4, last item).
5. **Check the code matches baseline-v2** (the script also does this and
   refuses to start otherwise):

```bash
python src/baseline_config.py --check
```

Expected: `code matches baseline-v2`.

---

## 3. Train one run

```bash
bash results/train_seed.sh sac 1
```

The learner is one of `ppo`, `recurrent_ppo`, `sac`, `tqc`; the seed is 0, 1 or 2
for the planned runs. The script:

1. checks the code against baseline-v2;
2. trains the run -- or **resumes** it if the folder has a checkpoint, or skips
   training if it is already finished; a run that died before its first
   checkpoint is moved to `<run>_incomplete_<date>` and restarted;
3. runs **Tier 1** (development set, safety layer off and on) and the **frozen
   suite** (headline + robustness set, safety layer off and on) on the run's
   `best_model.zip`, skipping either if it is already done.

It is safe to start again at any time with the same learner and seed: it picks up
where it stopped. (`bash results/train_seed.sh tqc 0` would, for instance, run
the one thing left for TQC seed 0 -- its Tier 1.)

**Detached (recommended)** — keeps running if the terminal or the Claude app
closes. In PowerShell from this folder, replacing the learner and seed:

```powershell
Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe" -ArgumentList '-lc', '"bash results/train_seed.sh sac 1"' -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput results\train_seed.console.log -RedirectStandardError results\train_seed.console.err
```

There is no window: check it with §4.

**Several runs in a row**, if you ever want them, are just one call after
another:

```bash
bash results/train_seed.sh ppo 1 && bash results/train_seed.sh recurrent_ppo 1
```

---

## 4. Check progress

**Status of every planned run** (quickest):

```bash
python tools/run_status.py
```

Shows, for all 12 planned runs, done / started (with the last step count) / not
started, the best development-set goal rate so far, whether the frozen suite has
run, and the last event-log lines.

**Live training output** of a run (Ctrl+C closes the view, not the training) —
replace the run name:

```powershell
Get-Content runs\sac_formulation_seed1_bl2.log -Wait -Tail 20
```

Useful lines: `total_timesteps` (progress), `[EVAL] t=... goal ... collision
...` (development-set result every 200 k steps; an evaluation takes a few
minutes, during which the log is quiet), `[CURRICULUM]` (stage changes).

**Learning curves:**

```bash
tensorboard --logdir runs/tensorboard
```

then open <http://localhost:6006>.

**Is anything training?** A running training is 11 Python processes (the
learner and 10 environment workers). Or:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag bl2 -Action status
```

`no running learner for tag bl2` means nothing is training (the script may still
be running an evaluation -- check `results/train_seed.log`).

---

## 5. Pause and resume

There are two ways, depending on how long.

### 5a. Short pause, in memory (minutes to hours)

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
overnight, use 5b. (`tools\resume_training.ps1` resumes every suspended training
process at once.)

### 5b. Stop now, resume later (any length, survives reboots)

Stop the script and all its processes:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_seed\.sh' } | ForEach-Object { taskkill /PID $_.ProcessId /T /F }
```

Resume by starting the same command again (§3). The run continues from its
last 250 k-step checkpoint, so up to 250 k steps are redone (about 1 h for
RecurrentPPO, up to ~6 h for SAC). Off-policy runs also reload their replay
buffer from `PhD/asv_replay_buffers/`; the run's `config.json` records each
resume and whether the buffer was restored.

To lose less, stop just after a checkpoint appears in the run folder
(`<learner>_<steps>_steps.zip`).

---

## 6. After a reboot, crash or power cut

Start the same command again (§3). It resumes the run from its last checkpoint,
or restarts it if it had not reached one (the old folder is kept as
`<run>_incomplete_<date>` and can be deleted).

---

## 7. Pieces by hand

The script is these steps; each can be run alone with the same settings and
checks.

Train, or resume, a run:

```bash
python src/train_formulation.py --config configs/baseline_v2.json --algo sac --seed 1 --tag bl2
python src/train_formulation.py --config configs/baseline_v2.json --algo sac --seed 1 --tag bl2 --resume runs/sac_formulation_seed1_bl2
```

With `--config`, only `--algo`, `--seed`, `--tag` (and `--resume`) come from the
command line; everything else comes from the file. `td3` still works but is not
part of the baseline.

The frozen suite of a finished run -- what the paper reports -- (add
`--tiers a,b,r` only when the extended named cases are wanted too):

```bash
python tools/tiers/frozen_suite.py --model runs/sac_formulation_seed1_bl2/best_model.zip --tag sacs1_bl2 --supervisor both
```

Tier 1, the development-set diagnostic, safety layer `off` and `on`:

```bash
python tools/tiers/tier1_replay.py --model runs/sac_formulation_seed1_bl2/best_model.zip --tag sacs1_bl2_supervisor_off --supervisor off
```

The Paper 2 deployment-layout set -- a separate set, not part of the frozen
suite (F104): the three published field layouts with and without a target ship,
630 episodes per safety layer mode, into `results/paper2_set/<tag>/`:

```bash
python tools/tiers/paper2_suite.py --model runs/sac_formulation_seed1_bl2/best_model.zip --tag sacs1_bl2
```

One frozen-suite or Paper 2 set test, with a trajectory figure (test IDs are in
`results/frozen_gallery/index.csv` and `results/paper2_gallery/index.csv`):

```bash
python tools/tiers/run_test.py --model runs/sac_formulation_seed1_bl2/best_model.zip CH-CR-CV-007 P2-L2-CRP-VAR-07
```

The field fine-tune (F106): continue a finished run from 2 M to 3 M with Paper
2-style field layouts mixed in, then test it on the Paper 2 set and the frozen
suite. It writes a new `runs/<run>_ftfield1/` folder; the source run is untouched:

```bash
bash results/finetune_field.sh sac 0
```

baseline-v3 (F107; prepared, not the paper's formulation): baseline-v2 plus
field-layout stages 6-7, 2.5 M steps, run folders tagged `bl3`. Check, then train:

```bash
python src/baseline_config.py --check --config configs/baseline_v3.json
python src/train_formulation.py --config configs/baseline_v3.json --algo sac --seed 0 --tag bl3
```

A quick launch check that trains 4,096 steps into a `_smoke` folder (delete it
afterwards):

```bash
python src/train_formulation.py --config configs/baseline_v2.json --algo sac --seed 0 --tag check --smoke
```

---

## 8. Do not change the formulation between runs

Every run checks the code against `configs/baseline_v2.json` and refuses to
start if any constant, switch, curriculum or learner setting differs
(`BASELINE CHECK FAILED` in `results/train_seed.log`, or `code does not match
baseline-v2` in a run log). This keeps every learner and seed on the same
formulation, however far apart in time they are trained.

- Editing documents, diagnostics, the evaluation tools or the frozen suite is
  fine (the suite is evaluation, not formulation).
- Editing `src/constants.py`, the reward, the observation, the curriculum or a
  learner's hyperparameters makes a **new** formulation: it needs a new config
  version (`python src/baseline_config.py --write` after deliberate changes)
  and every learner retrained on it.
- To check a finished run against the baseline:

```bash
python src/baseline_config.py --verify-run runs/sac_formulation_seed1_bl2
```

---

## 9. Saving to git and GitHub

Nothing commits automatically; commit and push when convenient. Pushing while
a run trains is safe (training only writes its own run folder).

- Replay buffers can never be committed: they are outside the repository, and
  `.gitignore` blocks `*_replay_buffer_*.pkl`.
- Model checkpoints are small (PPO ~3 MB, RecurrentPPO ~13 MB, SAC/TQC ~6 MB)
  but add up: ~130 MB per RecurrentPPO run. If pushes get slow, commit only
  `best_model.zip` and `final_model.zip` per run and keep the step checkpoints
  local.

---

## 10. Training a run on a cluster

1. **Commit and push** the finished run folders, so the cluster can see which
   runs are done.
2. On the cluster: clone the repository, install `requirements.txt` (the
   software versions each run used are in its `config.json` → `platform`), and
   run `bash results/train_seed.sh <learner> <seed>` or the pieces in §7.
3. Replay buffers go next to the clone by default (`asv_replay_buffers/`
   beside the repository); set `ASV_REPLAY_BUFFER_DIR` to put them elsewhere,
   e.g. on scratch storage. An off-policy run stopped here mid-way can only
   resume on the cluster with its buffer copied across from
   `PhD/asv_replay_buffers/<run>/`; without it, the resume refills 10 k steps
   before learning again (recorded in `config.json`).
4. Cluster job scripts (Slurm/PBS) are not written yet — ask for them once
   access is set up.

---

## 11. Troubleshooting

| Message or symptom | Meaning | What to do |
|---|---|---|
| `BASELINE CHECK FAILED` (`results/train_seed.log`) | the code differs from baseline-v2 | run `python src/baseline_config.py --check` to see what changed; undo it (§8) |
| `<learner> SEED <N> FAILED (see ...log)` | a run crashed | read the end of that run's log; start the same command again — it resumes from the last checkpoint |
| `note: ... is outside the planned learners x seeds` | a learner or seed not in the config's plan | fine for an extra run; it is simply not one of the twelve |
| `[BUFFER] ... GB free -- replay buffer NOT saved` | C: is nearly full | free disk space; training continues, but a resume from that checkpoint refills the buffer |
| `[BUFFER] a replay-buffer file stayed locked` | OneDrive held a file | harmless; an older buffer may be left in `PhD/asv_replay_buffers/` — delete it once that run is finished |
| `<run>_incomplete_<date>` folder | a run died before its first checkpoint and was restarted | delete it |
| log quiet for a few minutes | a development-set evaluation (every 200 k steps) | wait; `[EVAL]` lines follow |
| nothing in the log for much longer, no CPU use | the processes are stuck or paused | `pause_run.ps1 ... -Action status`; if paused, resume (§5a); otherwise stop and restart (§5b) |
| steps/s falls to a third (e.g. TQC 14 → 4) with the learner on CPUs 4–11 and CPUs 0–3 idle | Windows moved the hidden process to the efficiency cores (EcoQoS), not heat; `CurrentClockSpeed` 1600 MHz is only the base clock and always reads that | runs opt out at start (`_no_efficiency_mode`, 2026-09-26); anything else heavy running beside a training (evaluations, figure generation, OneDrive uploading many files) also slows it |
| steps/s falls to about two thirds | the laptop is on battery | plug it in |
| `Get-Process python` finds nothing | the Store Python's processes are named `python3.10` | use `Get-Process python3.10`, or `pause_run.ps1 ... -Action status` |
