import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from path_follow_env import ASVEnv
import optuna
from timeit import default_timer as timer

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
learning_rate = 0.002
batch_size = 256
n_epochs = 15
gamma = 0.95
clip_range = 0.37
vf_coef = 0.9
ent_coef = 9.412368520429483e-05

# model = PPO('MlpPolicy', env, verbose=1,
#             learning_rate=learning_rate,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             gamma=gamma,
#             clip_range=clip_range,
#             vf_coef=vf_coef,
#             ent_coef=ent_coef)
# model = PPO('MlpPolicy', env, verbose=1)

# DQN Hyperparameters
learning_rate = 0.0001
batch_size = 128
gamma = 0.99
target_update_interval = 10000  # Update target network every 10k steps
exploration_fraction = 0.1  # Fraction of total timesteps spent exploring
exploration_final_eps = 0.01  # Final exploration rate

# model = DQN('MlpPolicy', env, verbose=1,
#             learning_rate=learning_rate,
#             batch_size=batch_size,
#             gamma=gamma,
#             target_update_interval=target_update_interval,
#             exploration_fraction=exploration_fraction,
#             exploration_final_eps=exploration_final_eps)
model = DQN('MlpPolicy', env, verbose=1)

callback = CustomCallback()
num_timesteps = int(1e6)

start_time = timer()

# Train the model
model.learn(total_timesteps=num_timesteps, callback=callback)

end_time = timer()
time = end_time - start_time

hour = int(time//3600)
time = int(time - 3600*hour)
minute = int(time//60)
second = int(time - 60*minute)

print(f"Time elapsed = {hour} : {minute} : {second}")

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

# Trial 18 finished with value: -15.668539422000002 and parameters: 
# {'learning_rate': 0.005113216497561994, 'batch_size': 4096, 
# 'n_epochs': 10, 'gamma': 0.9218685744361972, 
# 'clip_range': 0.39952643431246604, 'gae_lambda': 0.9190472693604935, 
# 'vf_coef': 0.9041047435339762, 'ent_coef': 1.1432355666615247e-05}.class CustomCallback(BaseCallback):
#'learning_rate': 0.002227113238141113, 'batch_size': 256, 'n_epochs': 15, 'gamma': 0.950172079879918, 'clip_range': 0.3756287558375899, 'gae_lambda': 0.987440460569382, 'vf_coef': 0.8960319702073843, 'ent_coef': 9.412368520429483e-05}