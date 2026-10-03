#!/usr/bin/env bash
# Queued by the user (2026-10-03): once the G4 pilots are done, revise the
# baseline-v4 draft to 4.1 (straight survey lanes for dense draws, more
# varying-speed field targets, encounter weights -- see
# tools/diagnostics/v4_gates/revise_v4_after_g4.py) and re-run gate G3 on it.
# The pilots read the 4.0 draft, so nothing changes until they have finished.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
START_LINE=$(wc -l < "$LOG")
note "== v4 revision watcher armed: waiting for the G4 pilots to finish"
# Match only v4_pilot_run.sh's own lines (a note of this script once matched itself).
done_line() { tail -n +"$((START_LINE + 1))" "$LOG" | grep -v "watcher" | grep -qE "$1"; }
until done_line "[0-9] == G4 done$" || done_line "G4 PILOT [AB] FAILED"; do sleep 300; done
if done_line "G4 PILOT [AB] FAILED"; then
  note "   G4 failed: v4 revision NOT applied"; exit 1
fi
python tools/diagnostics/v4_gates/revise_v4_after_g4.py >> results/v4_revision.log 2>&1
note "   v4 draft revision exit $? ($(tail -1 results/v4_revision.log))"
python tools/diagnostics/v4_gates/gates.py g3 2 > results/v4_gates_g3_v41.log 2>&1
note "   G3 on the 4.1 draft exit $? (results/v4_gates/g3_summary.txt)"
note "== v4 revision done"
