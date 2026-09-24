#!/usr/bin/env bash
# SAC seed 0 started before KEEP_FINAL_BUFFER existed, so its own cleanup will
# delete the replay buffer when the run ends.  This copies each late checkpoint's
# buffer aside so the run can still be continued past 2 M.
SRC="$HOME/OneDrive - University of Tasmania/Documents/PhD/asv_replay_buffers/sac_formulation_seed0_bl2"
DST="$HOME/OneDrive - University of Tasmania/Documents/PhD/asv_replay_buffers/sac_formulation_seed0_bl2_kept"
mkdir -p "$DST"
for i in $(seq 1 2000); do
  for f in "$SRC"/sac_replay_buffer_1750000_steps.pkl "$SRC"/sac_replay_buffer_2000000_steps.pkl; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    if [ ! -f "$DST/$name" ]; then
      cp "$f" "$DST/$name.partial" && mv "$DST/$name.partial" "$DST/$name"
      echo "$(date '+%H:%M:%S') kept $name"
    fi
  done
  [ -f "$DST/sac_replay_buffer_2000000_steps.pkl" ] && { echo "final buffer kept"; break; }
  sleep 60
done
