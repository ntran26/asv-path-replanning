import gym
import gym_image_maze

if __name__=='__main__':
    env = gym.make('ImageMaze-v1')
    env.reset()
    for i in range(500):
        env.render()
        action = env.action_space.sample()
        observation, reward, finish, _ = env.step(action)
        
    env.close()