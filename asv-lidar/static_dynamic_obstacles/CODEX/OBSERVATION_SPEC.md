# CODEX observation schema 3

Configuration identifier: `codex-v3-context`. All branches are float32 and
clipped to their declared Gymnasium spaces. Heading is nautical: zero is north,
positive is clockwise. Previous policies with five branches are incompatible.

| Branch | Size, one target | Meaning |
| --- | ---: | --- |
| lidar | 27 | Gated and pooled obstacle closeness |
| boundary | 7 | Map raycast closeness from the shared estimated pose |
| ego | 3 | Measured surge/sway divided by SPEED_SCALE; yaw rate in degrees per second divided by 180 |
| path | 3 | Signed cross-track error divided by local channel half-width; current and lookahead course errors divided by 180 degrees |
| target | 16 | Domain distance; bearing sine/cosine; relative course sine/cosine; target and relative speeds; DCPA; TCPA; CRI; five class indicators; presence |
| context | 14 | Twelve encounter values below, followed by two previous executed actions |

Target feature names and normalizers are defined in `src/observation.py`.
Slots remain bound to track identity until loss; unoccupied slots and their
context values are zeroed. The feature extractor gates both target and context
embeddings with the corresponding target presence bit.

| Context offset per slot | Meaning |
| ---: | --- |
| 0 | ENGAGED indicator |
| 1 | CLEARING indicator |
| 2 | Latched compliant turn sense: +1 starboard, -1 port, 0 neither |
| 3 | Wrapped perceived heading change since engagement / 180 |
| 4 | Perceived surge change since engagement / SPEED_SCALE |
| 5 | Turn admissible |
| 6 | Slowing is predicted to clear |
| 7 | Admissibility has been evaluated |
| 8 | Action-required gate |
| 9 | Proximity gate |
| 10 | In-extremis indicator |
| 11 | Engagement age in seconds / T_ENGAGE, capped at 1 |

After all context slots come previous executed rudder and throttle, each in
[-1,1]. Supervisor reverse thrust saturates the throttle representation at -1;
the standard training supervisor is off. Multi-target sizing is parameterized
but the training and behavioral claim remains one dynamic target.

The total dimension for N slots is `42 + 28*N` (70 for N=1). The episode starts
with zero previous action. One perception update invalidates the observation
cache; repeated reads return copies without advancing the encounter latch.
Truth-only context fields are attached after observation encoding for
diagnostics, using a separate wholly true own-ship reference frame.
