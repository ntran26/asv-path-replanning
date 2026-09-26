#!/usr/bin/env bash
# One-off hand-over (2026-09-26).  A34 option (a) is built (suite 3.2: the
# reactive and non-compliant targets now react), so once TQC seed 0 finishes:
#   1. redraw the frozen gallery with the fixed targets (idle machine);
#   2. lift the frozen-suite hold and run the frozen suite on the four seed-0
#      models (results/frozen_seed0.sh);
#   3. continue the campaign -- seeds 1-2, each followed by its frozen suite.
# Nothing here runs beside training.  The old launcher was detached first: it
# had been edited since it started, and bash reads a script as it runs.
cd "$(dirname "$0")/.."
LOG=results/baseline_campaign.log
TQC_PID=29780
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
alive() { tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -q " $1 "; }

note "== hand-over armed (suite 3.2): waiting for TQC seed 0 (PID $TQC_PID)"
while alive $TQC_PID; do sleep 60; done
if [ -f runs/tqc_formulation_seed0_bl2/final_model.zip ]; then
  note "== tqc seed 0 finished (final_model.zip)"
else
  note "== tqc seed 0 learner exited WITHOUT final_model.zip -- the campaign will resume it"
fi
note "   frozen gallery (suite 3.2)"
python tools/diagnostics/frozen_gallery.py > results/frozen_gallery.log 2>&1
note "   frozen gallery exit $? ($(ls results/frozen_gallery/scenarios 2>/dev/null | wc -l) figures)"
rm -f runs/FROZEN_HOLD runs/CAMPAIGN_STOP
note "== frozen suite on the seed-0 models (hold lifted)"
bash results/frozen_seed0.sh
note "== continuing the campaign: seeds 1-2"
exec bash results/baseline_campaign.sh
