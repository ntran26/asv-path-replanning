GLOBAL MAP
- Construct map
    + ASV/Agent: heading angle and observation radius
    + Path
    + Start + Goal points
    + Static obstacles
    + Map boundary
- Show the ASV's movements by applying the "taken steps" array from the LOCAL MAP

LOCAL MAP
- Generate collision grid inside the observation radius ()
- Map the GLOBAL MAP to the collision grid:
    + Keep ASV/Agent same as GLOBAL MAP
    + Keep free space unoccupied as white (can move to) - STATE 0
    + Path marked as green (follow)                     - STATE 1
    + Goal marked as yellow (reset when reached)        - STATE 2
    + Obstacles and boundary marked as red (avoid)      - STATE 3
- Calculate reward based on the state the ASV is on:
    + STATE 0 (free space): -10
    + STATE 1 (path): +1
    + STATE 2 (goal): +50
    + STATE 3 (obstacles): -100
- Local map will be agent's observation space and action taken based on observation
- After each episode, save the agent's taken steps in an array

TO BE DISPLAYED:
- GLOBAL MAP plotted on the right
- LOCAL MAP plotted on the left
- Preferably, both maps should be running at the same time for real-time observation
