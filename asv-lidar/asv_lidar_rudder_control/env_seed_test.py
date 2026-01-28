import numpy as np
from asv_lidar_gym_continuous import ASVLidarEnv

def get_env_signature(env, seed):
    obs, _ = env.reset(seed=seed)
    # Collect a "fingerprint" of the environment
    signature = {
        "start_pos": (env.start_x, env.start_y),
        "goal_pos": (env.goal_x, env.goal_y),
        "num_obs": env.num_obs,
        "obstacles": env.obstacles.copy(),  # assuming it's an array or list
        "obs_sample": np.array(env.obstacles).flatten()[:5].tolist()  # first few obstacle coords
    }
    return signature

if __name__ == "__main__":
    env = ASVLidarEnv()

    seed = 12345
    sig1 = get_env_signature(env, seed)
    sig2 = get_env_signature(env, seed)

    print("First run signature:", sig1)
    print("Second run signature:", sig2)
    print("\nAre signatures identical?", sig1 == sig2)

    # Check with different seed
    sig3 = get_env_signature(env, seed + 1)
    print("\nDifferent seed identical to first?", sig1 == sig3)
