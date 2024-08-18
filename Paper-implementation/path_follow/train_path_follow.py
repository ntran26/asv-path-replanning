import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from path_follow_env import ASVEnv
import optuna

#                               -------- CONFIGURATION --------
# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.policy_loss = []
        self.value_loss = []
        self.rewards = []
        self.best_mean_reward = -np.inf  
        self.best_model_path = "best_model.zip"

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
            if "loss" in self.model.ep_info_buffer[0]:
                self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
                self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])

            # # Calculate the mean reward for the last 1000 steps
            # if len(self.rewards) >= 1000:
            #     mean_reward = np.mean(self.rewards[-1000:])
            #     if mean_reward > self.best_mean_reward:
            #         self.best_mean_reward = mean_reward
            #         print(f"New best mean reward: {mean_reward}. Saving model...")
            #         self.model.save(self.best_model_path)
        return True

# Create environment
env = ASVEnv()

# Adjust hyperparameters
learning_rate = 0.001
batch_size = 128
n_epochs = 10
gamma = 0.99
clip_range = 0.1
vf_coef = 0.5
ent_coef = 0.01

# model = PPO('MlpPolicy', env, verbose=1,
#             learning_rate=learning_rate,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             gamma=gamma,
#             clip_range=clip_range,
#             vf_coef=vf_coef,
#             ent_coef=ent_coef)
model = PPO('MlpPolicy', env, verbose=1)
callback = CustomCallback()
num_timesteps = int(1e5)

# Train the model
model.learn(total_timesteps=num_timesteps, callback=callback)

# Calculate mean reward
mean_reward = np.mean(callback.rewards[-1000:])
print(f"Mean reward: {mean_reward}")

# Save the model
model.save("ppo_path_follow")

# Plot rewards
smoothed_rewards = np.convolve(callback.rewards, np.ones(100)/100, mode='valid')
plt.plot(smoothed_rewards)
plt.xlabel('Steps')
plt.ylabel('Smoothed Reward')
plt.title('Smoothed Reward over Steps')
plt.show()

plt.plot(callback.policy_loss, label="Policy Loss")
plt.plot(callback.value_loss, label="Value Loss")
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Policy and Value Loss over Steps')
plt.legend()
plt.show()
# plt.plot(callback.rewards, label="Rewards")
# plt.xlabel('Steps')
# plt.ylabel('Reward')
# plt.title('Reward over Steps with Tuned Hyperparameters')
# plt.show()