import gymnasium
from stable_baselines3 import PPO

# Import your custom environment
from draft import ASVEnv  # Adjust the import path as needed

# Register the environment
gymnasium.envs.registration.register(id='ASVEnv-v0', entry_point=ASVEnv, kwargs={'render_mode': 'human'})

# Create the environment
env = gymnasium.make('ASVEnv-v0', render_mode='human')

# Load the trained model from the specified path
model_path = "models/ppo_asv_model_1000000_steps.zip"
model = PPO.load(model_path)

# Test the trained model
obs = env.reset()
for step in range(300):  # Adjust the range as needed
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if step % 10 == 0:  # Render every 10 steps to speed up the process
        env.render()
    if done:
        obs = env.reset()

env.close()
