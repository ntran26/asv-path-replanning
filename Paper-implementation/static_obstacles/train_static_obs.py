import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
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

#                               -------- CONFIGURATION --------
# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Number of timesteps/episodes
NUM_EPISODES = int(1e6)

# if __name__ == '__main__':
#     # Create the environment
#     env = ASVEnv()

#     # Check the environment
#     check_env(env)

#     # Define the model
#     model = PPO('MlpPolicy', env, verbose=1)

#     # Train the model with callback
#     callback = CustomCallback()
#     model.learn(total_timesteps=100000, callback=callback)

#     # Save the model
#     model.save("ppo_asv_model")

#     # Plot the rewards
#     plt.figure(figsize=(10, 5))
#     plt.plot(callback.rewards, label='Rewards')
#     plt.xlabel('Step')
#     plt.ylabel('Reward')
#     plt.title('Reward over Steps')
#     plt.legend()
#     plt.show()

# class CustomActorCritic(nn.Module):
#     def __init__(self, input_dim, action_dim):
#         super(CustomActorCritic, self).__init__()
#         # Shared layers for both actor and critic
#         # self.fc1 = nn.Linear(input_dim, 512)
#         # self.fc2 = nn.Linear(512, 512)
#         # self.fc3 = nn.Linear(512, 512)
#         self.fc1 = nn.Linear(313, 512)
#         self.fc2 = nn.Linear(512, 1252)
#         self.fc3 = nn.Linear(1252, 512)
        
#         # Actor head
#         self.actor = nn.Linear(512, action_dim)

#         # Critic head
#         self.critic = nn.Linear(512, 1)
    
#     # def forward(self, x):
#     #     x = th.relu(self.fc1(x))
#     #     x = th.relu(self.fc2(x))
#     #     x = th.relu(self.fc3(x))
#     #     return x
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     def actor_forward(self, x):
#         x = self.forward(x)
#         return self.actor(x)
    
#     def critic_forward(self, x):
#         x = self.forward(x)
#         return self.critic(x)

# class CustomPolicy(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim=1252, action_space=None):
#         super(CustomPolicy, self).__init__(observation_space, features_dim)
#         input_dim = features_dim
#         action_dim = env.action_space.n

#         self.actor_critic = CustomActorCritic(input_dim, action_dim)

#     def forward(self, x):
#         return self.actor_critic.forward(x)

#     def forward_actor(self, x):
#         return self.actor_critic.actor_forward(x)

#     def forward_critic(self, x):
#         return self.actor_critic.critic_forward(x)

# def objective(trial):
#     # Define hyperparameters
#     learning_rate = 8.515786595813231e-05
#     batch_size = 128
#     n_epochs = 8
#     gamma = 0.9339239258707902
#     clip_range = 0.20259480665235446
#     gae_lambda = 0.9222467745570867
#     vf_coef = 0.517316849734512
#     ent_coef = 3.7569404673013434e-05

#     # Create environment
#     env = ASVEnv()
#     check_env(env)

#     # Define and train the model
#     model = PPO('MlpPolicy', env, verbose=0, 
#                 learning_rate=learning_rate,
#                 batch_size=batch_size,
#                 n_epochs=n_epochs,
#                 gamma=gamma,
#                 clip_range=clip_range,
#                 gae_lambda=gae_lambda,
#                 vf_coef=vf_coef,
#                 ent_coef=ent_coef)

#     callback = CustomCallback()
#     model.learn(total_timesteps=500000, callback=callback)

#     # Calculate mean reward
#     mean_reward = np.mean(callback.rewards[-1000:])
#     return mean_reward

# # Create study and optimize
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# # Best hyperparameters
# print("Best hyperparameters:", study.best_params)

# # Train final model with best hyperparameters
# best_params = study.best_params

class CustomCallback(BaseCallback):
    def __init__(self, save_freq=100000, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_save_counter = 0
        self.policy_loss = []
        self.value_loss = []
        self.rewards = []
        self.best_mean_reward = -np.inf  # Initialize to a very low value
        self.best_model_path = "best_model.zip"

    def _on_step(self):
        # Save model at regular intervals
        if self.num_timesteps % self.save_freq == 0:
            model_path = f"static_obs_{self.num_timesteps}.zip"
            print(f"Saving model at {self.num_timesteps} timesteps")
            self.model.save(model_path)
            self.model_save_counter += 1

        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
            if "loss" in self.model.ep_info_buffer[0]:
                self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
                self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])

            # Calculate the mean reward for the last 1000 steps
            if len(self.rewards) >= 1000:
                mean_reward = np.mean(self.rewards[-1000:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"New best mean reward: {mean_reward}. Saving model...")
                    self.model.save(self.best_model_path)
        return True

# Create environment
env = ASVEnv()

# Adjust hyperparameters
learning_rate = 0.0001
batch_size = 30000
n_epochs = 10
gamma = 0.99
clip_range = 0.1
gae_lambda = 0.86
vf_coef = 0.5
ent_coef = 0.01

# Create the PPO model with the custom policy
model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            clip_range=clip_range,
            vf_coef=vf_coef,
            ent_coef=ent_coef)
# model = PPO('MlpPolicy', env, verbose=1)
callback = CustomCallback(save_freq=int(1e5))
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
model.save("ppo_static_obstacles")

# Plot rewards
plt.plot(callback.rewards, label="Rewards")
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Reward over Steps with Tuned Hyperparameters')
plt.show()

# {'learning_rate': 0.00907329821451761, 'batch_size': 64, 'n_epochs': 2, 'gamma': 0.9988044963952771, 'clip_range': 0.21608474944379513, 'gae_lambda': 0.8696976357115177, 'vf_coef': 0.3492943287414059, 'ent_coef': 4.4723499358631903e-07}