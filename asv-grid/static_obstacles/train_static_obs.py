import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from static_obs_env import ASVEnv
import optuna
from timeit import default_timer as timer
import pandas as pd

#                               -------- CONFIGURATION --------
# Number of timesteps/episodes
NUM_EPISODES = int(1e6)
SAVE_FREQENCY = int(1e6)

# Adjust hyperparameters
learning_rate = 0.0001
batch_size = 30000
n_epochs = 10
gamma = 0.99
clip_range = 0.1
gae_lambda = 0.86
vf_coef = 0.5
ent_coef = 0.01

class CustomCallback(BaseCallback):
    def __init__(self, save_freq=100000, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_save_counter = 0
        self.policy_loss = []
        self.value_loss = []
        self.rewards = []
        self.timesteps = []

    def _on_step(self):
        # Save model at regular intervals
        if self.num_timesteps % self.save_freq == 0:
            model_path = f"Paper-implementation/static_obstacles/static_obs_{self.num_timesteps}.zip"
            print(f"Saving model at {self.num_timesteps} timesteps")
            self.model.save(model_path)
            self.model_save_counter += 1

        # Append reward data
        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
            self.timesteps.append(self.num_timesteps)

        return True

    def _on_rollout_end(self):
        # Collect the policy and value loss at the end of each rollout
        logs = self.model.logger.name_to_value
        # self.rewards.append(logs['rollout/ep_rew_mean'])
        self.policy_loss.append(logs['rollout/ep_rew_mean'])
        self.value_loss.append(logs['train/value_loss'])

        # if 'rollout/ep_rew_mean' in logs:
        #     self.rewards.append(logs['rollout/ep_rew_mean'])
        #     self.policy_loss.append(logs['rollout/ep_rew_mean'])
        # else:
        #     self.policy_loss.append(None)

        # if 'train/value_loss' in logs:
        #     self.value_loss.append(logs['train/value_loss'])
        # else:
        #     self.value_loss.append(None)


# Save data to CSV for external use (e.g., in Excel)
def save_to_csv(callback, filename="training_data.csv"):
    data = {
        # 'timesteps': callback.timesteps,
        'rewards': callback.rewards
        # 'policy_loss': callback.policy_loss,
        # 'value_loss': callback.value_loss
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Smoothing function for plots
def smooth(data, window=50):
    return pd.Series(data).rolling(window=window).mean()

# Plot rewards, policy loss, and value loss
def plot_metrics(callback):
    plt.figure(figsize=(15, 5))

    # Plot smoothed rewards
    plt.subplot(1, 3, 1)
    plt.plot(smooth(callback.rewards), label="Mean Rewards", color="b")
    plt.fill_between(range(len(callback.rewards)), 
                     smooth(callback.rewards) - pd.Series(callback.rewards).rolling(window=50).std(),
                     smooth(callback.rewards) + pd.Series(callback.rewards).rolling(window=50).std(),
                     color="blue", alpha=0.2)
    plt.title('Smoothed Mean Rewards')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid(True)

    # # Plot policy loss
    # plt.subplot(1, 3, 2)
    # if callback.policy_loss:
    #     plt.plot(smooth(callback.policy_loss), label="Policy Loss", color="r")
    #     plt.title('Smoothed Policy Loss')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Loss')
    #     plt.grid(True)

    # # Plot value loss
    # plt.subplot(1, 3, 3)
    # if callback.value_loss:
    #     plt.plot(smooth(callback.value_loss), label="Value Loss", color="g")
    #     plt.title('Smoothed Value Loss')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Loss')
    #     plt.grid(True)

    plt.tight_layout()
    plt.show()

# Create environment
env = ASVEnv()

# Create the DQN model
model = DQN('MlpPolicy', env, verbose=1,
            learning_rate=learning_rate,
            gamma=gamma)

# # Create the PPO model
# # model = PPO('MlpPolicy', env, verbose=1)
# model = PPO('MlpPolicy', env, verbose=1,
#             learning_rate=learning_rate,
#             batch_size=batch_size,
#             n_epochs=n_epochs,
#             gamma=gamma,
#             clip_range=clip_range,
#             vf_coef=vf_coef,
#             ent_coef=ent_coef)

callback = CustomCallback(save_freq=SAVE_FREQENCY)
num_timesteps = NUM_EPISODES

start_time = timer()

# Train the model
model.learn(total_timesteps=num_timesteps, callback=callback)

end_time = timer()

# Calculate mean reward
mean_reward = np.mean(callback.rewards[-1000:])
print(f"Mean reward = {mean_reward}")

time = end_time - start_time

hour = int(time//3600)
time = int(time - 3600*hour)
minute = int(time//60)
second = int(time - 60*minute)

print(f"Total time = {hour} : {minute} : {second}")

# Save the model
# model.save("Paper-implementation/static_obstacles/ppo_static_obstacles")
model.save("Paper-implementation/static_obstacles/dqn_static_obstacles")

# Save training data to CSV
save_to_csv(callback, "dqn_training_data.csv")

# Plot metrics
plot_metrics(callback)

# Plot rewards
plt.plot(callback.rewards, label="Rewards")
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Reward over Steps with Tuned Hyperparameters')
plt.show()
