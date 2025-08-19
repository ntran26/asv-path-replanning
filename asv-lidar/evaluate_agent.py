from asv_lidar_gym_continuous import ASVLidarEnv
from stable_baselines3 import PPO, SAC
import numpy as np
import json
import os
from tqdm import trange

NUM_EPS = 100

class PPOAgent:
    def __init__(self, path):
        self.model = PPO.load(path)
    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

class SACAgent:
    def __init__(self, path):
        self.model = SAC.load(path)
    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

def evaluate_agent(agent, agent_name, seeds, render=False):
    env = ASVLidarEnv(render_mode="human" if render else None)

    success_count = 0
    all_cross_track_errors = []
    total_rewards = []
    time_efficiencies = []

    for seed in trange(len(seeds), desc=f"Evaluating {agent_name}"):
        obs, _ = env.reset(seed=seeds[seed])
        done = False
        total_reward = 0
        cte_list = []
        start_time = env.elapsed_time

        while not done:
            action = agent.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            cte_list.append(abs(obs['tgt'][0]))

        reached_goal = env.distance_to_goal <= env.collision + 30
        collided = np.any(env.lidar.ranges.astype(np.int64) <= env.collision)
        success = reached_goal and not collided

        if success:
            success_count += 1

        total_rewards.append(total_reward)
        all_cross_track_errors.append(np.mean(cte_list))
        time_efficiencies.append(env.elapsed_time - start_time)

    # Aggregate Results
    results = {
        "agent": agent_name,
        "episodes": len(seeds),
        "success_rate": success_count / len(seeds),
        "avg_cross_track_error": float(np.mean(all_cross_track_errors)),
        "std_cross_track_error": float(np.std(all_cross_track_errors)),
        "avg_total_reward": float(np.mean(total_rewards)),
        "avg_time": float(np.mean(time_efficiencies))
    }

    return results

if __name__ == "__main__":
    agents = {
        "PPO_0_5": PPOAgent("models/ppo_asv_model_0_5.zip"),
        "PPO_0_6": PPOAgent("models/ppo_asv_model_0_6.zip"),
        "PPO_0_7": PPOAgent("models/ppo_asv_model_0_7.zip"),
        "PPO_0_8": PPOAgent("models/ppo_asv_model_0_8.zip"),
        "PPO_0_9": PPOAgent("models/ppo_asv_model_0_9.zip"),
        "SAC_0_5": SACAgent("models/sac_asv_model_0_5.zip"),
        "SAC_0_6": SACAgent("models/sac_asv_model_0_6.zip"),
        "SAC_0_7": SACAgent("models/sac_asv_model_0_7.zip"),
        "SAC_0_8": SACAgent("models/sac_asv_model_v2.zip"),
        "SAC_0_9": SACAgent("models/sac_asv_model_0_9.zip"),
    }

    # Pre-generate fixed seeds for all agents
    eval_seeds = [i for i in range(NUM_EPS)]

    results_list = []
    for name, agent in agents.items():
        result = evaluate_agent(agent, name, eval_seeds, render=False)
        results_list.append(result)
        print(f"\n{name} Results:")
        print(json.dumps(result, indent=2))
    
    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/all_agents_eval.json", "w") as f:
        json.dump(results_list, f, indent=4)
