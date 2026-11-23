# Safety-layer results: what is tracked and how runs are reproduced (2026-11-23)

## Tracked

- Every report (`*/reports/`), selection, inventory, scenario cache and suite definition, and the calibration and audit files.
- The deployed layer's test-set runs in full: `testset_v4_main/runs/ts4_final_e1_c01`-`c10` (per-episode results and decision traces).
- The run logs of the final test-set evaluation: `testset_v4_main/logs/`.
- Five V20 run manifests and one V5 report cited by `planning_records/SAFETY_V20_PLAN.md` and `planning_records/SAFETY_LAYER_COMPLETE_SUMMARY.md`.

The paper and thesis numbers come from the reports. The C-7 result is `testset_v4_main/reports/ts4_final/`: `report.md`, and `paired.csv` with every case in every mode.

## Kept on disk only

Since commit 22515c2fd, the following are not tracked:
- the per-episode folders (`runs/`, `traces/`, `attempts/`) of the development runs and of the earlier test-set runs;
- 24 large intermediate files listed in the repository `.gitignore`.

This includes the other two run sets behind `reports/ts4_final`:
- `ts4_final_c01`-`c10`: off, V16, V16 + stop;
- `ts4_final_e1s35_c01`-`c10`.

Where a removed file is now:
- **Never 1 MiB or larger:** unchanged in commit f04f2e0a9, the parent of 22515c2fd.
- **Any version of 1 MiB or more:** removed from the history on 2026-11-23 (see `history_rewrite_2026-11-23/` at the repository root). The current version is on disk, unchanged, as an OneDrive online-only file. Every version not on disk was saved first in `PhD/asv_git_purged_files/`, listed in `MANIFEST.tsv` there.

List the removed files still in the history:

    git diff --name-only --diff-filter=D f04f2e0a9 22515c2fd -- asv-lidar/static_dynamic_obstacles/results/safety_dev

Restore one of them, or a folder, exactly:

    git restore --source=f04f2e0a9 -- <path>

The removed files that are no longer in the history are listed in `PhD/asv_git_purged_files/purge_paths.txt`.

## Re-running

Each run folder's `manifest.json` records the full setup:
- the runner;
- the selection file and its SHA-256;
- the modes and their constructor options;
- the checkpoint and its SHA-256;
- the SHA-256 of every source file;
- the runtime environment.

The runners load the deployed checkpoint from `runs/sac_formulation_seed0_bl3/kept_best_3M/`. That folder is tracked, and its files are byte-identical to `best_model.zip` and `best_vecnormalize.pkl` beside it.

The final test-set runs use 10 chunks of 100 cases. All three run sets share each chunk's case list, which is the `cases` list in the tracked `testset_v4_main/runs/ts4_final_e1_cNN/manifest.json`.

To re-run chunk NN for one run set, from `asv-lidar/static_dynamic_obstacles`:

    python -B tools/diagnostics/safety/v21_iteration.py --tag <new tag> \
        --selection results/safety_dev/testset_v4_main/selection.json \
        --cases <the chunk's case IDs, comma-separated> \
        --modes off,v16,v16_estop_bt --compact-traces --gzip-traces

The `--modes` value is one of:
- `off,v16,v16_estop_bt`;
- `v21_e1_estop_bt` (the deployed layer);
- `v21_e1s35_estop`.

Paired report:

    python -B tools/diagnostics/safety/v21_report.py --runs results/safety_dev/testset_v4_main/runs \
        --reports results/safety_dev/testset_v4_main/reports --tags <tags, comma-separated> --name <name>

## Reproducibility check (2026-11-23)

Ten recorded episodes were re-run from a clean checkout of 22515c2fd, with `kept_best_3M/` added:
- test set v4, cases BAS-HO-NC-041 and BAS-CR-CV-013, in modes off, `v16_estop_bt` and `v21_e1_estop_bt`;
- validation set, cases DVD-HO-01 and DVD-HO-07, in modes `v21_estop_off` and `v21_e1_estop_bt`.

Every field of every result file matched the record, apart from the attempt number and the elapsed time. The decision traces (544 records) differed only in the timing fields `step_seconds` and `filter.estop_seconds`.

Without `kept_best_3M/`, the runner stops at start-up because the checkpoint folder is missing.
