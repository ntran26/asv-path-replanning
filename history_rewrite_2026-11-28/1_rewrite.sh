#!/bin/bash
# Step 1 of 2: rewrite a copy of the repository's history without Claude as author or co-author.
#
# Changes nothing in the working repository and nothing on GitHub. It:
#   1. checks that the working repository is committed and level with GitHub;
#   2. saves the whole current history to one backup file (a git bundle);
#   3. installs git-filter-repo 2.47.0 for Python 3.12 (user site, no sudo) if it is missing;
#   4. makes a bare copy of the repository in $WORK and rewrites it there:
#      - commits by "Claude <noreply@anthropic.com>" get the author and committer
#        "ntran26 <hoangnamtran2610@gmail.com>" (mailmap.txt);
#      - "Co-Authored-By: ... Claude/anthropic" and "Generated with [Claude Code]" lines are
#        removed from the messages; nothing else in a message changes;
#   5. checks the result: the same number of commits, the same files at the tip, no Claude
#      author, committer or co-author line left;
#   6. copies the old-to-new commit map next to this script and prints the push commands.
#
# Run from anywhere:  bash ~/asv-path-replanning/history_rewrite_2026-11-28/1_rewrite.sh
set -euo pipefail

REPO=/home/nam/asv-path-replanning
HERE=$REPO/history_rewrite_2026-11-28
WORK=${WORK:-$HOME/asv_rewrite_2026-11-28/repo.git}
BACKUP_DIR=${BACKUP_DIR:-$HOME/asv_git_backups}
BUNDLE=$BACKUP_DIR/asv-path-replanning_before_2026-11-28.bundle
GITHUB=git@github.com:ntran26/asv-path-replanning.git
FILTER_REPO_VERSION=2.47.0

say() { printf '\n== %s\n' "$*"; }
fail() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

say "1. The working repository"
cd "$REPO"
[ "$(git rev-parse --abbrev-ref HEAD)" = master ] || fail "the repository is not on master"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "uncommitted changes to tracked files; commit or put them aside first"
git fetch -q origin
[ "$(git rev-parse master)" = "$(git rev-parse origin/master)" ] || fail "master differs from GitHub's; pull or push first"
OLD_HEAD=$(git rev-parse master)
OLD_TREE=$(git rev-parse 'master^{tree}')
OLD_COUNT=$(git rev-list --count --all)
echo "master $OLD_HEAD, $OLD_COUNT commits, level with GitHub"
echo "Claude as author: $(git log --all --format=%ae | grep -c '^noreply@anthropic.com$' || true) commits;" \
     "with a Claude co-author line: $(git log --all --format=%B%x00 | tr '\0\n' '\n ' | grep -ci 'co-authored-by:[^<]*<noreply@anthropic.com>' || true)"

say "2. Backup of the whole history: $BUNDLE"
mkdir -p "$BACKUP_DIR"
if [ -e "$BUNDLE" ]; then
    echo "already there; checking it"
else
    git bundle create "$BUNDLE" --all
fi
git bundle verify -q "$BUNDLE" && echo "bundle verified ($(du -h "$BUNDLE" | cut -f1))"

say "3. git-filter-repo"
if ! git filter-repo --version >/dev/null 2>&1; then
    python3 -m pip install --user --break-system-packages --quiet "git-filter-repo==$FILTER_REPO_VERSION"
    hash -r
fi
git filter-repo --version >/dev/null 2>&1 || fail "git-filter-repo is not on PATH (~/.local/bin)"
echo "git-filter-repo $(python3 -m pip show git-filter-repo 2>/dev/null | sed -n 's/^Version: //p')"

say "4. Rewrite a copy: $WORK"
[ -e "$WORK" ] && fail "$WORK exists already; move it aside (or set WORK=...) and run again"
mkdir -p "$(dirname "$WORK")"
git clone -q --bare --no-local "$REPO" "$WORK"
cd "$WORK"
git filter-repo --mailmap "$HERE/mailmap.txt" --message-callback '
import re
keep = [line for line in message.split(b"\n")
        if not re.match(rb"(?i)\s*co-authored-by:.*(claude|anthropic)", line)
        and not re.search(rb"(?i)generated with \[?claude code", line)]
return b"\n".join(keep).rstrip(b"\n") + b"\n"
'

say "5. Checks"
NEW_HEAD=$(git rev-parse master)
NEW_TREE=$(git rev-parse 'master^{tree}')
NEW_COUNT=$(git rev-list --count --all)
[ "$NEW_TREE" = "$OLD_TREE" ] || fail "the files at the tip differ ($OLD_TREE -> $NEW_TREE)"
[ "$NEW_COUNT" = "$OLD_COUNT" ] || fail "commit count changed ($OLD_COUNT -> $NEW_COUNT)"
LEFT_AUTHORS=$(git log --all --format='%an <%ae>%n%cn <%ce>' | grep -ci 'anthropic\|^claude' || true)
LEFT_LINES=$(git log --all --format=%B | grep -ci 'co-authored-by:.*\(claude\|anthropic\)\|generated with \[\?claude code' || true)
[ "$LEFT_AUTHORS" = 0 ] || fail "$LEFT_AUTHORS Claude author/committer entries left"
[ "$LEFT_LINES" = 0 ] || fail "$LEFT_LINES Claude co-author lines left"
git cat-file -p baseline-v1 | head -1
echo "tip files identical (tree $NEW_TREE), $NEW_COUNT commits, no Claude author, committer or co-author line"
echo "authors now:"; git log --all --format='  %an <%ae>' | sort | uniq -c | sort -rn

say "6. Commit map"
cp filter-repo/commit-map "$HERE/commit-map.txt"
cp filter-repo/ref-map "$HERE/ref-map.txt"
echo "master $OLD_HEAD -> $NEW_HEAD"
echo "maps copied to $HERE (commit-map.txt, ref-map.txt)"

cat <<EOF

Done. Nothing has been pushed and the working repository is unchanged.
To replace the history on GitHub (this cannot be undone on GitHub; the backup bundle can restore it),
in parts, as GitHub refuses a single push over 2 GiB:

bash $HERE/1b_push_in_parts.sh

Then run step 2 to bring the working repository onto the new history:

bash $HERE/2_after_push.sh
EOF
