import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from draft import ASVEnv  # Import the environment script

# Create the environment
env = gymnasium.make('ASVEnv-v0', render_mode='human')

# Wrap the environment
env = DummyVecEnv([lambda: env])

# Create a callback for saving models
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/', name_prefix='ppo_asv_model')

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1)  # Verbose is set to 1 for training logs

# Train the model
model.learn(total_timesteps=1000000, callback=checkpoint_callback)

# Save the final model
model.save("models/ppo_asv_model_1000000_steps")
