#!/usr/bin/env bash
# Decision (2026-10-04): near-impossible development episodes are replaced in place
# (tools/diagnostics/feasibility/dev_replacements.py), after the development-set
# oracle with the evaluation seeds has finished (results/feasibility_deveval.sh).
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
START_LINE=$(wc -l < "$LOG")
note "== development-set replacements queued: waiting for the development-set oracle"
until tail -n +"$((START_LINE + 1))" "$LOG" | grep -v "queued" | grep -qE "development-set oracle done$"; do sleep 120; done
python tools/diagnostics/feasibility/dev_replacements.py --processes 6 > results/feasibility/dev_replacements.log 2>&1
note "   development-set replacements exit $? ($(tail -1 results/feasibility/dev_replacements.log | cut -c1-120))"
note "== development-set replacements done"
