import gymnasium as gym
from asv_lidar_gym import ASVLidarEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration
TOTAL_TIMESTEPS = 100000

# Create the Gym environment
env = ASVLidarEnv(render_mode=None)  # no rendering during training

# Create a callback to save models periodically.
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoints/', name_prefix='asv_model')

# Create the PPO model
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./asv_tensorboard/")

# Train the model
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

# Save the final model
model.save("asv-lidar/asv_follow_path_model")
print("Model saved")

# Test the model
obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
env.close()
