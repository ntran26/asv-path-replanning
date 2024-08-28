This directory focuses on recreating the methods for constructing environment and training DRL model from the paper
"Dynamic trajectory planning for ships in dense environment using collision grid with deep reinforcement learning"

NOTE: File name with "_env" is an OpenAI Gym environment, otherwise, a regular environment to visualise

1. asv_trajectory.py: 
    - This file simulate the ASV's trajectory in 20 steps (STEP), with 3 possible actions
        + Turn left: +5 heading angle at every step
        + Turn right: -5 heading angle at every step
        + Go straight: keep the same heading angle
    - The speed is set to 1 for all actions
    - The 3 paths are plotted and the ASV's movements are simulated, navigating along the paths

2. collision_grid.py:
    - This file generate the LiDAR simulation (observation radius) of the ASV on the left pannel
    - The collision grid showing obstacles locations on the right pannel
    - The collision grid is generated to simplify the ASV observation space for DRL training
    - The file also simulates the ASV moving straight
    - The static obstacles are updated at every step in the collision grid on the right pannel

3. obstacle_detection.py:
    - This file combines the previous 2 programs: simulate the ASV going left combine with collision grid
    - Dynamic obstacles are generated in this program, and their trajectories are predefined
    - On the right pannel, the obstacles will be invisible if they're not in the observation radius range

4. static_obstacle.py:
    - This program focus on constructing a path for the ASV to follow and static obstacles only
    - The purpose is to finalize the structure to ready for building gym environment for DRL training
    - There will be a path (green) for the ASV to follow
    - The path will be mapped to the collision grid (right) pannel to show where the path is
    - The static obstacles will work like before: invisible when outside of observation radius

5. static_obstacle_env.py (OpenAI Gym environment):
    - The following features are included in the environment
        + Static obstacles are placed in multiple places with different shapes
        + A straight path/line for the ASV to follow, which will also be mapped to the collision grid
        + A goal point at the end of the path
        + Boundaries around the map are set to keep the ASV from straying off the map
        + The ASV's observation radius is 100m
        + The distance from start to goal point is 300m
    - There will be a global map and a local map:
        + Global map: Show every obstacles, path, goal point, ASV and observation radius
        + Local map: Only show the ASV, observation radius and collision grid (this will be the
          observation space)
    - Observation space: The observation radius as boundary, along with coordinates of the obstacles in collision grid
    - State space: Location of the ASV, speed and heading angle
    - Action space: 3 actions with the same speed of 1
        + Go left: heading angle +5
        + Go right: heading angle -5
        + Go straight: keep heading angle
    - Collision grid/ Observation space:
        + Green: path to follow
        + Red: static obstacles
        + Black: map boundary
        + White: free space
    - Reward design: 
        + Reach goal: +5
        + Stay on the path: -1
        + Stray off the path (on the free space): -10
        + Hit the obstacle/boundary: -100
    - Reset conditions:
        + Hit the obstacle/boundary
        + Reach the goal
    - When reset, perform the following:
        + Set ASV position to start point
        + Set heading angle (and speed) to initial value
        + Reset total reward
    - Each episode will be different where the location of the obstacles will be changed


    Training time report:
    - For the env with 313 grids (radius = 100, grid size = 10): 100,000 episodes ~ 6 minutes
    - For the env with 77 grids (radius = 20, grid size = 4): 100,000 episodes ~ 3 minutes
