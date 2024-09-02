import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from collision_avoidance_env import ASVEnv
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

class CustomCallback(BaseCallback):
    def __init__(self, save_freq=100000, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_save_counter = 0
        self.policy_loss = []
        self.value_loss = []
        self.rewards = []

    def _on_step(self):
        # Save model at regular intervals
        if self.num_timesteps % self.save_freq == 0:
            model_path = f"Paper-implementation/collision_avoidance/model_{self.num_timesteps}.zip"
            print(f"Saving model at {self.num_timesteps} timesteps")
            self.model.save(model_path)
            self.model_save_counter += 1

        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[0]["r"])
            if "loss" in self.model.ep_info_buffer[0]:
                self.policy_loss.append(self.model.ep_info_buffer[0]["loss"]["policy_loss"])
                self.value_loss.append(self.model.ep_info_buffer[0]["loss"]["value_loss"])
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
model.save("Paper-implementation/collision_avoidance/ppo_collision_avoidance")

# Plot rewards
plt.plot(callback.rewards, label="Rewards")
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Reward over Steps with Tuned Hyperparameters')
plt.show()