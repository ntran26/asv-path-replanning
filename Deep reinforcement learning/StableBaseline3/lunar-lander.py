import gym

env = gym.make("LunarLander-v2")
env.reset()

print("Sample action:", env.action_space.sample())
print("Observation space shape:", env.observation_space.shape)
print("Sample observation:", env.observation_space.sample())
