This repository is for the porject ASV Path Replanning and Collision Avoidance using Deep Reinforcement Learning

The agent needs to follow a predefined path while avoiding static obstacles.

asv-lidar: The current work in progress
- The environment is created with Gymnasium and PyGame
- The ASV emits 21 lidar beams distributed evenly in 90 degrees ahead of the ASV.
- The lidar beams have a maximum range of 150 m, if there is an obstacle, it will return a value of distance to obstacle.
- Observation space (MultiDiscrete):
  + lidar: a list of lidar range [150, 150,..., 150]
  + pos: coordinate of ASV [x,y]
  + hdg: current heading angle [a value between 0-360]
  + dhdg: change of heading [a value between 0-36]
  + tgt: target waypoint towards the path (distance from pos to path on x-axis)
- Action space (Discrete):
  + 0: turn to port (rudder = -25)
  + 1: keep ahead (rudder = 0)
  + 2: turn to starboard (rudder = 25)
- Reward function (to be updated):
  + Outside of path (> 50m): 0 or -1
  + Near path: 1 - (tgt / path_range), where path_range = 50m from path
  + Move in reverse: -10
  + Collision/off border: -20

Paper-implementation: First attempt - inspired from the paper "Dynamic trajectory planning for ships in dense environment using collision grid with deep reinforcement learning"
- The environment is created with Gymnasium and Matplotlib
- A field-of-view around the ASV is generated to act as the local view.
- Observation space (Discrete):
  + The field-of-view is divided into grids, each grid is appended to the observation list, updating every timesteps.
  + Each grid has 4 possible states: free space, on path, obstacle, goal point.
  + A virtual goal is generated to guide the agent towards the goal when the goal is not visible in the field-of-view.
- Action space (Discete):
  + 0: turn left (+5 heading angle)
  + 1: turn right (-5 heading angle)
  + 2: go straight (keep heading)
- Reward function:
  + On free space: -1
  + On path: 0
  + Reach goal: +20
  + Collide with obstacle/border: -100
