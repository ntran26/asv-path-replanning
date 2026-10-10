# Aborted replay attempt g2g3_v2

Launched 2026-10-10 with the code in `replay_code_identity.json`. Part 1 was killed by the memory-cgroup out-of-memory killer (about 7.3 GB resident) after 20 of 34 episodes, because the tool parsed every full-snapshot line of an episode into memory before replaying it. Part 2 was then stopped by the operator after 17 of 34 episodes to avoid the same failure. No part file was written, so no result from this attempt exists or is reported. Rerun as `g2g3_v3` with line-by-line streaming and per-episode result files; the certificate code is unchanged.
