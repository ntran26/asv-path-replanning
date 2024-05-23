from stable_baselines3.common.env_checker import check_env
from path_follow_env import PathFollowEnv

env = PathFollowEnv()
# It will check your custom environment and output additional warnings if needed
env.render()
check_env(env)