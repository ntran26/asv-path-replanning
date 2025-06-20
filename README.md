This repository is for the porject ASV Path Replanning and Collision Avoidance using Deep Reinforcement Learning

# DESCRIPTION

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

# USAGE

The main folder is located in 'asv-lidar'
- 'asv_lidar.py': simulated lidar sensor.
- 'ship_model.py': model of the simulated ASV.
- 'asv_lidar_gym_continuous.py': environment uses in training PPO and SAC agents.
- 'test_run.py': a set of test scenarios for testing the trained agent.
  + TEST_CASE == 0: random environment setup
  + TEST_CASE == 1 to 6: test scenarios
  + Else: use a selected environment setup from data/env-setup
- 'train_test_asv': main script to train/test model
  + To train a new model (can choose between ppo and sac)
  ```
      python train_test_asv.py --mode train --algo sac
  ```
  + To test a model (can choose between ppo and sac, case from 0 to 6)
  ```
      python train_test_asv.py --mode test --algo sac --case 0
  ```
  + If no arguments passed, use test mode, SAC agent, test case = 0 (random)

- Data folder:
  + After testing the agent, a test data will be saved as .json file listed in 'data' folder. 
  + Plotting the data requires both PPO and SAC agents data.
  + If the random environment setup is selected (TEST_CASE == 0), data of the environment is saved as .json file,   listed in 'data/env-setup'.

- Tensorboard:
  + Data from training sessions
  + Access 'ppo_log' or 'sac_log' and run 'tensorboard --logdir==$filename$\

- Results:
  + Recorded videos stored in 'videos'
  + ASV trajectories stored in 'results'

- Models:
  + ppo_asv_model_180: discrete action with 180 degrees lidar swath
  + ppo_asv_model_270: discrete action with 270 degrees lidar swath
  The rest are continuous action space
  + ppo_asv_model_v1: PPO model v1 - fixed path, random obstacles, fixed number of obstacles (10)
  + ppo_asv_model_v2: PPO model v2 - random path and obstacles, random number of obstacles (0-10)
  + sac_asv_model_v1: SAC model v1 - random path and obstacles, fixed number of obstacles (10)
  + sac_asv_model_v2: SAC model v2 - random path and obstacles, random number of obstacles (0-10)
