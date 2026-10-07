Paper snapshots (2026-10-06)
============================

Figure: paper_snapshots.pdf (vector) and paper_snapshots.png (300 dpi), six panels,
7.1 x 3.5 in, sized for a full-width figure.  Made by tools/paper_snapshots.py.

Policy: kept SAC baseline-v3, seed 0, best checkpoint of the 3 M run
(runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip), safety layer off.
Episodes: test set v4, replayed with their own seeds; every replay reproduces the
outcome recorded in results/test_set/v4/sacs0_bl3/episodes.csv.

Selection
---------
These are chosen successful episodes, one per encounter type, not a random sample.
The policy succeeds in 86.5 % of test set v4 (865 of 1,000); the figure shows what
a success looks like, not how often one occurs.

1. Successful episodes of each encounter type, ranked by COLREGs-violation frames
   and cross-track error; candidates with a target encounter needed a closest
   approach between 1.2 and 4.5 m (a real encounter, not a distant pass).
2. 18 candidates rendered (candidates.png) and checked from their traces for the
   manoeuvre the formulation asks for.  Targets never give way, so the own ship
   gives way to every crossing target by passing astern (decision A17,
   src/colregs/context.py): starboard turn for a starboard crossing, port turn for
   a port crossing.  Head-on: starboard alteration, port-to-port passing.
   Overtaking: pass to port of the target.  Being overtaken: hold course and speed.
   Rejected on that check: BAS-HO-NC-035 (head-on passed starboard to starboard),
   P2-L1-OT-FIX-01 (overtook to starboard), BAS-CR-CV-007 (port turn for a
   starboard crossing), BAS-BO-RE-094 and P2v2-L1-BO-FIX-05 (episode ended before
   the target had overtaken).
3. From the rest, one per type, spreading scenario family and geometry (coupled:
   a generated coupled layout and the reference layouts L2 and L3; decoupled:
   channel and basin), panel count (0, 1, 3) and leg shape (2 straight,
   4 slanted), with one varying-speed target.

Panels (paper_snapshots_summary.csv has the numbers)
------
(a) No target, FS-NT-010: coupled layout, slanted leg, 3 panels crowding the
    path.  Passes west of all three, rejoins the path near the goal, 26.0 s.
(b) Head-on, P2-L3-HO-FIX-15: Paper 2 layout L3, slanted leg.  Alters about 48 deg
    to starboard, passes port to port, CPA 4.3 m at 13.0 s, then returns to the path.
(c) Port crossing, CH-CR-CV-035: channel, straight leg, 1 panel on the path.  Turns
    about 40 deg to port and slows to 0.29 m/s, passes astern of the target
    (CPA 2.5 m at 8.5 s), then clears the panel on the path.
(d) Starboard crossing, P2-L2-CRS-VAR-10: layout L2, slanted leg, varying-speed
    target.  Turns about 50 deg to starboard and passes astern (CPA 3.0 m at 7.0 s).
(e) Overtaking, BAS-OT-CV-073: basin, straight leg, 1 panel on the path.  Moves out
    to port of the slower target, passes it (CPA 3.0 m at 18.0 s), regains the path.
(f) Being overtaken, BAS-BO-RE-016: basin, slanted leg, no panels.  Holds its course
    within about 20 deg while the target passes on its port side (CPA 2.2 m at
    10.0 s); speed eases from 0.5 to 0.4 m/s.

Draft caption
-------------
Trajectories of the SAC policy in six test episodes, one per encounter type:
(a) no target, (b) head-on, (c) port crossing, (d) starboard crossing,
(e) overtaking and (f) being overtaken.  Panels (b) and (d) are coupled scenarios,
where the encounter can occur beside the obstacles (reference layouts L3 and L2),
and (a) uses a generated coupled layout without a target; (c), (e) and (f) are
decoupled, with obstacles kept clear of the encounter (channel and basin).  Layouts
also vary in number of static panels and leg shape.  Own-ship (blue) and target (red) hulls are drawn every 5 s; equal numbers
mark the same instant, and later hulls are darker.  The dotted segment marks the
closest point of approach.  Targets keep their course and do not give way; the own
ship passes astern of crossing targets, alters to starboard head-on and overtakes
to port.

Files
-----
paper_snapshots.pdf / .png     the figure
paper_snapshots_summary.csv    per panel: case, layout, leg, panels, duration, CPA, minimum speed
<case>.png                     single panels (all 18 candidates)
<case>_trace.csv               per-step own-ship state and target positions
candidates.pdf / .png          the 18 candidates, for reselection

To regenerate with another policy (for example the SAC v4.3 run once it is evaluated
on test set v4):
    python tools/paper_snapshots.py --model runs/sac_formulation_seed0_bl4/best_model.zip --tag sacs0_bl4
The replay check then needs those cases to be successes for that policy too.
