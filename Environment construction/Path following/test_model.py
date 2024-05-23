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
model = PPO.load("ppo_asv_model_final")

# Test the trained model
obs, _ = env.reset()
for step in range(30000):  # Adjust the range as needed
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render()

env.close()
