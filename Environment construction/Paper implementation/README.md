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

5. static_obstacle_env.py:
    - 