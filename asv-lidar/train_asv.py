import gymnasium as gym
from asv_lidar_gym import ASVLidarEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Create the Gym environment
env = ASVLidarEnv(render_mode=None)  # no rendering during training

# Optionally wrap the environment (e.g. using Monitor for logging, or VecEnv for parallelism)
# Here we keep it simple.
# Note: Gymnasium now uses seed in reset(), so you can pass seed values to env.reset(seed=123).

# Create a callback to save models periodically.
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoints/', name_prefix='asv_model')

# Create the PPO model.
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./asv_tensorboard/")

# Train the model. (Adjust total_timesteps as desired.)
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final model.
model.save("asv-lidar/asv_follow_path_model")
print("Model saved")

# Test the trained model.
obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Test run: Total reward: {total_reward}")
env.close()
