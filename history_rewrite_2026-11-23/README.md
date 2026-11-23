# History rewrite of 2026-11-23

On 2026-11-23 the repository history was rewritten with `git filter-repo`, which shrank `.git` from 8.1 GiB to 2.2 GiB.

## What changed

- **Removed from history:** 1,019 paths. Each was absent from the current files and had at least one version of 1 MiB or more, such as old checkpoints, videos and large intermediate outputs.
- **Unchanged:** the current files. The tree of the last commit is `7d6da78c` both before and after.
- **Changed:** every commit hash. Master moved from `ba8288d37` to `615f5ea5b`. The tag `baseline-v1` now points at commit `13cf65f26`.

## Files here

- `commit-map.txt`: the new hash for each of the 838 old commits. One old commit, `ec283bb7d`, became empty and maps to zeros.
- `ref-map.txt`: the old and new tips of `master` and `baseline-v1`.

## Old hashes

- **Documents updated on 2026-11-23:** the commit hashes cited in `planning/`, `planning_records/`, `PORTING_MANIFEST.md` and `results/safety_dev/README.md` now give the new hashes. All of these are under `asv-lidar/static_dynamic_obstacles/`.
- **Records unchanged:** the following keep the hash they were written with, and `commit-map.txt` translates it:
  - configs, including the frozen `configs/`, whose digest must not change;
  - run `config.json` files;
  - bench summaries, run manifests, logs and decision logs.
- **Commit messages:** a commit message that cites another commit still cites its old hash.

## Removed files

Every removed version was saved before the rewrite in `PhD/asv_git_purged_files/`. That folder is in OneDrive, outside the repository. It holds:
- `MANIFEST.tsv`: every saved version with its blob hash;
- `purge_paths.txt`: the removed paths;
- `README.md`: the layout.

A version identical to the file still on disk was not copied.

On GitHub, the old commits remain readable through the read-only pull-request refs 1-3 until GitHub Support removes them.
