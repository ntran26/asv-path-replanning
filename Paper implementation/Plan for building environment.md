GLOBAL MAP
- Construct map
    + ASV/Agent: heading angle and observation radius
    + Path
    + Start + Goal points
    + Static obstacles
    + Map boundary


LOCAL MAP
- Generate collision grid inside the observation radius ()
- Map the GLOBAL MAP to the collision grid:
    + Keep ASV/Agent
    + Keep free space unoccupied as white (can move to) - STATE 0
    + Path marked as green (follow)                     - STATE 1
    + Goal marked as yellow (reset when reached)        - STATE 2
    + Obstacles and boundary marked as red (avoid)      - STATE 3
- Calculate reward based on the state the ASV is on:
    + STATE 0 (free space): -10
    + STATE 1 (path): +1
    + STATE 2 (goal): +50
    + STATE 3 (obstacles): -50
