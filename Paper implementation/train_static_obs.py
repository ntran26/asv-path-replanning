import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from static_obs_env import ASVEnv
import optuna
from tqdm import tqdm

#                               -------- CONFIGURATION --------
# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Define map dimensions and start/goal points, number of static obstacles
WIDTH = 200
HEIGHT = 300
START = (100, 30)
GOAL = (100, 250)
NUM_STATIC_OBS = 5

# Define observation radius and grid size
RADIUS = 100
SQUARE_SIZE = 10
SPEED = 2
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define initial heading angle, turn rate and number of steps
INITIAL_HEADING = 90
TURN_RATE = 5

# Define states
FREE_STATE = 0          # free space
PATH_STATE = 1          # path
COLLISION_STATE = 2     # obstacle or border
GOAL_STATE = 3          # goal point

# class CustomCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         self.policy_loss = []
#         self.value_loss = []
#         self.rewards = []

#     def _on_step(self):
#         if len(self.model.ep_info_buffer) > 0:
#             self.rewards.append(self.model.ep_info_buffer[0]["r"])
#             if "loss" in self.model.ep_info_buffer[0]:
#                 self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
#                 self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])
#         return True

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

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.policy_loss = []
        self.value_loss = []
        self.rewards = []

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
            if "loss" in self.model.ep_info_buffer[0]:
                self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
                self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])
        return True

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
env = ASVEnv()
learning_rate = 8.515786595813231e-05
batch_size = 128
n_epochs = 8
gamma = 0.9339239258707902
clip_range = 0.20259480665235446
gae_lambda = 0.9222467745570867
vf_coef = 0.517316849734512
ent_coef = 3.7569404673013434e-05
# model = PPO('MlpPolicy', env, verbose=1, **best_params)
model = PPO('MlpPolicy', env, verbose=1, 
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                clip_range=clip_range,
                gae_lambda=gae_lambda,
                vf_coef=vf_coef,
                ent_coef=ent_coef)
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001, gamma=0.99)
callback = CustomCallback()
num_timesteps = int(5e5)
# Initialize tqdm progress bar
with tqdm(total=num_timesteps, desc="Training Progress") as pbar:
    timestep = 0

    while timestep < num_timesteps:
        # Perform one learning step
        model.learn(total_timesteps=1000, callback=callback, reset_num_timesteps=False)
        # Calculate mean reward
        mean_reward = np.mean(callback.rewards[-1000:])
        
        # Update the timestep and progress bar
        timestep += 1000
        pbar.update(1000)
# model.learn(total_timesteps=num_timesteps, callback=callback)

model.save("ppo_asv_model")
plt.plot(callback.rewards, label="Rewards")
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Reward over Steps with Tuned Hyperparameters')
plt.show()


# [I 2024-08-08 22:04:56,912] Trial 99 finished with value: -1577.840389703 and parameters: 
# {'learning_rate': 1.2247035372437565e-05, 'batch_size': 64, 'n_epochs': 3, 'gamma': 0.9336413752404424, 
# 'clip_range': 0.34830383187752906, 'gae_lambda': 0.9744300867560989, 'vf_coef': 0.1532606515006814, 'ent_coef': 1.2917604627782274e-05}. 
# Best is trial 65 with value: -1096.794644681.
# Best hyperparameters: {'learning_rate': 8.515786595813231e-05, 'batch_size': 128, 
# 'n_epochs': 8, 'gamma': 0.9339239258707902, 'clip_range': 0.20259480665235446, 
# 'gae_lambda': 0.9222467745570867, 'vf_coef': 0.517316849734512, 'ent_coef': 3.7569404673013434e-05}