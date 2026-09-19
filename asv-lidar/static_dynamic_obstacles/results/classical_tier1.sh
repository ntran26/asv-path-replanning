#!/usr/bin/env bash
# B8: the classical comparators on the Tier 1 sets, supervisor off and on --
# the same sets, seeds and switches as the learned policies' Tier 1 replays.
cd "$(dirname "$0")/.."
for m in off on; do
  for p in los_dwa encounter_vo; do
    python tools/tiers/tier1_replay.py --policy $p --tag ${p}_supervisor_$m --supervisor $m --processes 3 \
      > results/tiers/tier1_${p}_supervisor_$m.log 2>&1
    echo "== tier1 $p supervisor $m exit $?"
  done
done
echo "== CLASSICAL DONE"
