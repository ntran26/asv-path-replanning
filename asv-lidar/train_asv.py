import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from asv_lidar_gym import ASVLidarEnv

# Toggle between train and test
TRAIN = 1

# Create the environment
env = ASVLidarEnv(render_mode=None)
env = Monitor(env)  # For logging episode rewards

# Hyperparamters
learn_rate = 0.0001
n_steps = 2048
batch_size = 256
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
clip_range_vf = None
ent_coef = 0.01
vf_coef = 0.5

class CustomCallback(BaseCallback):
    def __init__(self, save_freq=100000, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_save_counter = 0
        self.policy_loss = []
        self.value_loss = []
        self.rewards = []

    def _on_step(self) -> bool:
        # Save model at regular intervals
        if self.num_timesteps % self.save_freq == 0:
            model_path = f"models/model_{self.num_timesteps}.zip"
            print(f"Saving model at {self.num_timesteps} timesteps")
            self.model.save(model_path)
            self.model_save_counter += 1

        if self.locals.get("loss", None) is not None:
            loss = self.locals["loss"]
            self.policy_loss.append(loss.get("policy_loss", 0))
            self.value_loss.append(loss.get("value_loss", 0))

        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
            if "loss" in self.model.ep_info_buffer[0]:
                self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
                self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])
        return True

# Model save path
MODEL_PATH = "ppo_asv_model"

if TRAIN == 1:
    # Initialize PPO model
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_asv_tensorboard/",
                learning_rate=learn_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, clip_range_vf=clip_range_vf,
                ent_coef=ent_coef, vf_coef=vf_coef)

    # Training parameters
    timesteps = 5000000
    callback = CustomCallback()

    # # Lists to store rewards for plotting
    # reward_log = []
    # episode_rewards = []

    # Train the model
    model.learn(total_timesteps=timesteps, tb_log_name="asv_ppo", callback=callback)

    # Save the model
    model.save("ppo_asv_model")
    print("Model saved!")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Plot rewards/episodes
    fig = plt.figure(1)
    plt.plot(callback.rewards, label="Rewards")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')
    fig.savefig("reward_plot.png")

    fig = plt.figure(2)
    plt.subplot(221)
    plt.plot(callback.policy_loss)
    plt.title("Policy Loss")
    plt.xlabel("Training Steps")

    plt.subplot(222)
    plt.plot(callback.value_loss)
    plt.title("Value Loss")
    plt.xlabel("Training Steps")

    plt.show()

    env.close()

else:
    # Load the trained model and test it
    model = PPO.load(MODEL_PATH)
    # model = PPO.load("ppo_path_follow.zip")

    test_env = ASVLidarEnv(render_mode="human")

    obs, _ = test_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Use learned policy
        obs, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        print(total_reward)

    print(f"Test episode completed. Total reward: {total_reward}")