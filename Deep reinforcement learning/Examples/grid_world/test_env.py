from stable_baselines3.common.env_checker import check_env
from grid_world import GridWorldEnv

env = GridWorldEnv()
# It will check your custom environment and output additional warnings if needed
env.render()
check_env(env)