import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Toggle between train and test
TRAIN = True

# Import your environment
from asv_lidar_gym import ASVLidarEnv

# Create the environment
env = ASVLidarEnv(render_mode=None)
env = Monitor(env)  # For logging episode rewards

# Hyperparamters
learn_rate = 0.0001
ent_coef = 0.1


# Model save path
MODEL_PATH = "ppo_asv_model"

if TRAIN:
    # Initialize PPO model
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_asv_tensorboard/",
                learning_rate=learn_rate,
                ent_coef=ent_coef)

    # Training parameters
    timesteps = 5000000

    # Lists to store rewards for plotting
    reward_log = []
    episode_rewards = []

    # Callback function to log rewards
    def reward_callback(locals_, globals_):
        global episode_rewards
        if "reward" in locals_:
            episode_rewards.append(locals_["reward"])
        if "dones" in locals_ and locals_["dones"]:
            reward_log.append(sum(episode_rewards))
            episode_rewards = []
        return True

    # Train the model
    model.learn(total_timesteps=timesteps, callback=reward_callback)

    # Save the model
    model.save("ppo_asv_model")
    print("Model saved!")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Plot the average reward over episodes
    mean_rewards = np.zeros(len(reward_log))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(reward_log[max(0,x-99):(x+1)])
    plt.plot(mean_rewards) 
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Training Progress: Average Reward per Episode")
    plt.savefig("reward_plot.png")
    plt.show()

    env.close()

else:
    # Load the trained model and test it
    model = PPO.load(MODEL_PATH)
    test_env = ASVLidarEnv(render_mode="human")

    obs, _ = test_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Use learned policy
        obs, reward, done, _, _ = test_env.step(action)
        total_reward += reward

    print(f"Test episode completed. Total reward: {total_reward}")