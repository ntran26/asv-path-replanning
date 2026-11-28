# History rewrite of 2026-11-28: one author

On 2026-11-28 the repository history was rewritten with `git filter-repo` so that every commit has
the repository owner as its only author. This removes the account `Claude <noreply@anthropic.com>`
from GitHub's contributor list.

## What changes

- **Authors and committers:** 344 commits by `Claude <noreply@anthropic.com>` get the author and
  committer `ntran26 <hoangnamtran2610@gmail.com>` (`mailmap.txt`).
- **Messages:** 423 lines `Co-Authored-By: Claude ... <noreply@anthropic.com>` are removed, along
  with any `Generated with [Claude Code]` line. `git filter-repo` also updates the commit hashes
  that 12 messages quote, to the new hashes. Nothing else in a message changes.
- **Unchanged:** every commit's files and dates, the number of commits (856) and the files at the
  tip.
- **Changed:** every commit hash. `commit-map.txt` gives the new hash for each old one, and
  `ref-map.txt` gives the old and new tips of `master` and `baseline-v1`.

The 2026-11-28 rehearsal on a copy, at `fba15e521`, gave these results: tree `2fd0963bd` at the
tip before and after, 856 commits, no Claude author, committer or co-author line, and `master`
`fba15e521` -> `01b1a8866`. The rewrite is deterministic: the same input gives the same new hashes.

## Steps

1. `bash 1_rewrite.sh` saves the whole history to
   `~/asv_git_backups/asv-path-replanning_before_2026-11-28.bundle` and rewrites a bare copy in
   `~/asv_rewrite_2026-11-28/repo.git`. It then checks the result and copies the maps here.
   Nothing is pushed. This takes about 3 minutes, and the bundle is about 2.3 GB.
2. `bash 1b_push_in_parts.sh` pushes the copy to GitHub. GitHub refuses a push over 2 GiB,
   and a rewritten history shares no commit with GitHub's, so a single push sends all of it,
   about 2.4 GB; on 2026-11-28 that push was refused ("pack exceeds maximum allowed size") and
   left GitHub unchanged. The script pushes to a temporary branch `rewrite-tmp` in five parts of
   0.3-0.6 GB (first-parent commits 300, 350, 400, 750 and the tip), then moves `master` and
   `baseline-v1` onto it and deletes the branch.
3. `bash 2_after_push.sh` moves the working repository onto the new history with
   `git reset --soft`. No file in the working folder is touched.
4. Commit this folder.

To restore the old history, run `git clone ~/asv_git_backups/asv-path-replanning_before_2026-11-28.bundle`.

## Afterwards

- **Other clones:** the UTAS laptop and the personal laptop have to be cloned again. Save any
  uncommitted work there first. A `git pull` on the old history fails or merges the two histories.
- **Old hashes in documents:** the guides, bench summaries (`git_head`), run manifests and
  planning records keep the hashes they were written with, and `commit-map.txt` translates them.
  The maps of the 2026-11-23 rewrite (`../history_rewrite_2026-11-23/`) translate hashes from
  before that date into the history as it stood before this rewrite.
- **GitHub:** the contributor list is recalculated within about a day. The read-only pull-request
  refs 1-3 still hold old commits, including Claude-authored ones; only GitHub Support can remove
  them.
- **New commits:** no `Co-Authored-By: Claude` line.
