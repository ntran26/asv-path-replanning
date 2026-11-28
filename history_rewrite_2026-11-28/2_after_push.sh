#!/bin/bash
# Step 2 of 2, after the force push: move the working repository onto GitHub's new history.
#
# The files at the tip are identical before and after the rewrite, so only the branch pointer moves
# (git reset --soft): no file in the working folder is touched, untracked and ignored files
# included.  The old commits stay in the backup bundle (step 1).
#
# Run:  bash ~/asv-path-replanning/history_rewrite_2026-11-28/2_after_push.sh
set -euo pipefail

REPO=/home/nam/asv-path-replanning
fail() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

cd "$REPO"
[ "$(git rev-parse --abbrev-ref HEAD)" = master ] || fail "the repository is not on master"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "uncommitted changes to tracked files; commit or put them aside first"
OLD_HEAD=$(git rev-parse master)
git fetch -q --force origin '+refs/heads/master:refs/remotes/origin/master' '+refs/tags/*:refs/tags/*'
NEW_HEAD=$(git rev-parse origin/master)
[ "$OLD_HEAD" != "$NEW_HEAD" ] || fail "GitHub still has the old history; push first (1b_push_in_parts.sh)"
[ "$(git rev-parse 'origin/master^{tree}')" = "$(git rev-parse 'master^{tree}')" ] \
    || fail "GitHub's tip files differ from this repository's; stop and check"
[ "$(git log origin/master --format='%ae%n%ce' | grep -ci anthropic || true)" = 0 ] || fail "GitHub's history still has Claude commits"
git reset -q --soft origin/master
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the working tree is not clean after the move; check git status"
echo "master: $OLD_HEAD -> $NEW_HEAD (files unchanged)"
echo "authors on master:"; git log --format='  %an <%ae>' | sort | uniq -c | sort -rn
git log --oneline -3
cat <<EOF

The working repository is on the new history.  Optional, to free the old commits' space here:
  git -C $REPO reflog expire --expire=now --all && git -C $REPO gc --prune=now
(the backup bundle keeps them).
EOF
