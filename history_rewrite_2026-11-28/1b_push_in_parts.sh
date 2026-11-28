#!/bin/bash
# Between step 1 and step 2: push the rewritten history to GitHub in parts.
#
# GitHub refuses a push over 2 GiB, and a rewritten history shares no commit with GitHub's, so a
# single push sends everything (about 2.4 GB).  This pushes it to a temporary branch,
# `rewrite-tmp`, oldest commits first, in five parts of 0.3-0.6 GB (first-parent commits 300,
# 350, 400, 750 and the tip); each part sends only what the earlier parts have not.  Then `master`
# and `baseline-v1` are moved onto it (nothing left to send) and the temporary branch is deleted.
# GitHub's `master` stays on the old history until that last move.
#
# Run:  bash ~/asv-path-replanning/history_rewrite_2026-11-28/1b_push_in_parts.sh
# If a part fails (network), run the script again: parts already on GitHub are not sent again.
set -euo pipefail

WORK=${WORK:-$HOME/asv_rewrite_2026-11-28/repo.git}
GITHUB=git@github.com:ntran26/asv-path-replanning.git
CUTS=(300 350 400 750)
fail() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

cd "$WORK" || fail "no rewritten copy at $WORK; run 1_rewrite.sh first"
[ "$(git log master baseline-v1 --format='%ae%n%ce' | grep -ci anthropic || true)" = 0 ] || fail "the copy still has Claude commits"
mapfile -t FP < <(git rev-list --reverse --first-parent master)
TIP=$(git rev-parse master)
echo "rewritten master $TIP, ${#FP[@]} first-parent commits"
echo "GitHub master now: $(git ls-remote "$GITHUB" refs/heads/master | cut -c1-12)"
read -r -p "Push the new history to GitHub in parts, then move master onto it? [y/N] " ok
[ "$ok" = y ] || [ "$ok" = Y ] || fail "nothing pushed"

part=0
for n in "${CUTS[@]}" tip; do
    part=$((part + 1))
    if [ "$n" = tip ]; then c=$TIP; else c=${FP[$((n - 1))]}; fi
    printf '\n== part %d of %d: up to %s\n' "$part" $(( ${#CUTS[@]} + 1 )) "$(git log -1 --format='%h %s' "$c")"
    git push --force "$GITHUB" "$c:refs/heads/rewrite-tmp"
done

printf '\n== master and baseline-v1 onto the new history\n'
git push --force "$GITHUB" master
git push --force "$GITHUB" baseline-v1
git push "$GITHUB" --delete rewrite-tmp

NOW=$(git ls-remote "$GITHUB" refs/heads/master | cut -f1)
[ "$NOW" = "$TIP" ] || fail "GitHub's master is $NOW, not $TIP"
echo
echo "GitHub's master is $TIP. Next: bash ~/asv-path-replanning/history_rewrite_2026-11-28/2_after_push.sh"
